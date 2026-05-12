#include "porch/tensor.hpp"
#include "porch/cuda_jit.hpp"

#include <float.h>

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace porch {

    tensor make_materialized_tensor(
        std::vector<index_t> shape, std::vector<float32_t> values,
        cuda_jit::device_buffer device_values, bool host_current,
        device placement
    );

    enum class expr_kind {
        input,
        scalar,
        add,
        subtract,
        multiply,
        matmul,
        slice,
        reshape,
        transpose,
        broadcast,
        concat,
        sum,
        max,
        exp,
    };

    struct tensor_expr::node {
        expr_kind kind;
        const tensor* input = nullptr;
        float32_t scalar = 0.0F;
        std::shared_ptr<const node> lhs;
        std::shared_ptr<const node> rhs;
        std::vector<index_t> slice_shape;
        std::vector<index_t> slice_strides;
        std::vector<index_t> shape;
        index_t slice_offset = 0;
        size_t axis = 0;
    };

    struct realized_storage {
        realized_storage(
            std::vector<float32_t> initial_values,
            cuda_jit::device_buffer initial_device_values,
            bool initial_host_current
        )
            : values(std::move(initial_values)),
              device_values(std::move(initial_device_values)),
              host_current(initial_host_current) {}

        mutable std::vector<float32_t> values;
        cuda_jit::device_buffer device_values;
        mutable bool host_current = true;
        mutable std::mutex host_mutex;
    };

    struct lazy_materialization {
        explicit lazy_materialization(
            std::shared_ptr<const tensor_expr::node> initial_expression
        )
            : expression(std::move(initial_expression)) {}

        std::shared_ptr<realized_storage> storage();

        std::shared_ptr<const tensor_expr::node> expression;
        std::once_flag once;
        std::shared_ptr<realized_storage> realized;
    };

    struct tensor::state {
        state(
            tensor_layout initial_layout, std::vector<float32_t> initial_values,
            cuda_jit::device_buffer initial_device_values,
            bool initial_host_current, device initial_placement,
            std::shared_ptr<const tensor_expr::node> initial_expression
        )
            : layout(std::move(initial_layout)), placement(initial_placement) {
            if (initial_expression == nullptr) {
                realized = std::make_shared<realized_storage>(
                    std::move(initial_values), std::move(initial_device_values),
                    initial_host_current
                );
            }
            else {
                lazy = std::make_shared<lazy_materialization>(
                    std::move(initial_expression)
                );
                realized = nullptr;
            }
        }

        [[nodiscard]] std::shared_ptr<realized_storage> storage() const {
            if (realized != nullptr) return realized;
            return lazy->storage();
        }

        tensor_layout layout;
        device placement;
        std::shared_ptr<realized_storage> realized;
        std::shared_ptr<lazy_materialization> lazy;
    };

    namespace {

        std::shared_ptr<const tensor_expr::node> make_node(
            expr_kind kind, const tensor* input = nullptr,
            float32_t scalar = 0.0F,
            std::shared_ptr<const tensor_expr::node> lhs = nullptr,
            std::shared_ptr<const tensor_expr::node> rhs = nullptr
        ) {
            auto node = std::make_shared<tensor_expr::node>();
            node->kind = kind;
            node->input = input;
            node->scalar = scalar;
            node->lhs = std::move(lhs);
            node->rhs = std::move(rhs);
            return node;
        }

        std::shared_ptr<const tensor_expr::node> make_slice_node(
            std::shared_ptr<const tensor_expr::node> source,
            std::vector<index_t> shape, std::vector<index_t> strides,
            index_t offset
        ) {
            auto node = std::make_shared<tensor_expr::node>();
            node->kind = expr_kind::slice;
            node->lhs = std::move(source);
            node->slice_shape = std::move(shape);
            node->slice_strides = std::move(strides);
            node->slice_offset = offset;
            return node;
        }

        std::shared_ptr<const tensor_expr::node> make_shape_node(
            expr_kind kind, std::shared_ptr<const tensor_expr::node> source,
            std::vector<index_t> shape
        ) {
            auto node = std::make_shared<tensor_expr::node>();
            node->kind = kind;
            node->lhs = std::move(source);
            node->shape = std::move(shape);
            return node;
        }

        std::shared_ptr<const tensor_expr::node> make_axis_node(
            expr_kind kind, std::shared_ptr<const tensor_expr::node> source,
            size_t axis
        ) {
            auto node = std::make_shared<tensor_expr::node>();
            node->kind = kind;
            node->lhs = std::move(source);
            node->axis = axis;
            return node;
        }

        std::shared_ptr<const tensor_expr::node> make_concat_node(
            std::shared_ptr<const tensor_expr::node> lhs,
            std::shared_ptr<const tensor_expr::node> rhs, size_t axis
        ) {
            auto node = std::make_shared<tensor_expr::node>();
            node->kind = expr_kind::concat;
            node->lhs = std::move(lhs);
            node->rhs = std::move(rhs);
            node->axis = axis;
            return node;
        }

        size_t checked_numel(std::span<const index_t> shape) {
            size_t count = 1;
            for (const index_t extent : shape) {
                if (extent < 0) {
                    throw std::invalid_argument(
                        "tensor shape extents must be non-negative"
                    );
                }
                count *= static_cast<size_t>(extent);
            }
            return count;
        }

        std::vector<index_t> contiguous_strides(
            std::span<const index_t> shape
        ) {
            std::vector<index_t> strides(shape.size(), 1);
            index_t stride = 1;
            for (size_t index = shape.size(); index > 0; --index) {
                strides[index - 1] = stride;
                stride *= shape[index - 1];
            }
            return strides;
        }

        bool is_contiguous_layout(
            std::span<const index_t> shape, std::span<const index_t> strides
        ) {
            if (shape.size() != strides.size()) return false;

            index_t expected_stride = 1;
            for (size_t index = shape.size(); index > 0; --index) {
                if (strides[index - 1] != expected_stride) return false;
                expected_stride *= shape[index - 1];
            }
            return true;
        }

        index_t normalize_index(index_t index, index_t extent) {
            if (index < 0) index += extent;
            if (index < 0 || index >= extent) {
                throw std::out_of_range("tensor index is out of bounds");
            }
            return index;
        }

        index_t normalize_slice_endpoint(index_t index, index_t extent) {
            if (index < 0) index += extent;
            if (index < 0) return 0;
            if (index > extent) return extent;
            return index;
        }

        size_t slice_length(index_t start, index_t stop, index_t step) {
            if (step <= 0) {
                throw std::invalid_argument(
                    "tensor slice step must be positive"
                );
            }
            if (stop <= start) return 0;
            return static_cast<size_t>((stop - start + step - 1) / step);
        }

        struct expression_context {
            std::vector<const tensor*> inputs;
            std::map<const tensor*, size_t> input_indices;
            device placement;
            bool initialized = false;
            size_t temporary_count = 0;
        };

        void record_input(expression_context& context, const tensor& value) {
            if (!context.initialized) {
                context.placement = value.placement();
                context.initialized = true;
            }
            else {
                if (context.placement != value.placement())
                    throw std::invalid_argument(
                        "tensor expression devices must match"
                    );
            }

            if (context.input_indices.contains(&value)) return;
            const size_t index = context.inputs.size();
            context.inputs.push_back(&value);
            context.input_indices.emplace(&value, index);
        }

        struct expression_info {
            std::vector<index_t> shape;
            bool scalar = true;
        };

        void require_same_shape(
            const expression_info& lhs, const expression_info& rhs
        ) {
            if (lhs.scalar || rhs.scalar) return;
            if (std::ranges::equal(lhs.shape, rhs.shape)) return;
            throw std::invalid_argument("tensor expression shapes must match");
        }

        void require_axis(
            size_t axis, size_t rank, std::string_view operation
        ) {
            if (axis >= rank) {
                throw std::invalid_argument(
                    std::string{operation} + " axis is out of bounds"
                );
            }
        }

        std::vector<index_t> reduced_shape(
            std::span<const index_t> shape, size_t axis
        ) {
            std::vector<index_t> result;
            result.reserve(shape.size() - 1);
            for (size_t dim = 0; dim < shape.size(); ++dim) {
                if (dim != axis) result.push_back(shape[dim]);
            }
            if (result.empty()) result.push_back(1);
            return result;
        }

        void require_broadcastable(
            std::span<const index_t> source, std::span<const index_t> target
        ) {
            if (source.size() > target.size()) {
                throw std::invalid_argument(
                    "broadcast target rank must be at least source rank"
                );
            }

            const size_t offset = target.size() - source.size();
            for (size_t dim = 0; dim < source.size(); ++dim) {
                const index_t source_extent = source[dim];
                const index_t target_extent = target[offset + dim];
                if (source_extent != 1 && source_extent != target_extent) {
                    throw std::invalid_argument(
                        "broadcast shapes are not compatible"
                    );
                }
            }
        }

        expression_info infer_expression(
            const tensor_expr::node& node, expression_context& context
        ) {
            switch (node.kind) {
                case expr_kind::input:
                    record_input(context, *node.input);
                    return {
                        {node.input->shape().begin(),
                         node.input->shape().end()},
                        false
                    };
                case expr_kind::scalar:
                    return {};
                case expr_kind::add:
                case expr_kind::subtract:
                case expr_kind::multiply: {
                    expression_info lhs = infer_expression(*node.lhs, context);
                    expression_info rhs = infer_expression(*node.rhs, context);
                    require_same_shape(lhs, rhs);
                    if (lhs.scalar) return rhs;
                    return lhs;
                }
                case expr_kind::matmul: {
                    expression_info lhs = infer_expression(*node.lhs, context);
                    expression_info rhs = infer_expression(*node.rhs, context);
                    if (lhs.scalar || rhs.scalar) {
                        throw std::invalid_argument(
                            "matmul operands must be tensors"
                        );
                    }
                    if (lhs.shape.size() != 2 || rhs.shape.size() != 2) {
                        throw std::invalid_argument(
                            "matmul requires rank-2 tensors"
                        );
                    }
                    if (lhs.shape[1] != rhs.shape[0]) {
                        throw std::invalid_argument(
                            "matmul inner dimensions must match"
                        );
                    }
                    return {{lhs.shape[0], rhs.shape[1]}, false};
                }
                case expr_kind::slice:
                    (void)infer_expression(*node.lhs, context);
                    return {node.slice_shape, false};
                case expr_kind::reshape: {
                    expression_info source =
                        infer_expression(*node.lhs, context);
                    if (source.scalar) {
                        throw std::invalid_argument(
                            "reshape operand must be a tensor"
                        );
                    }
                    if (checked_numel(source.shape) !=
                        checked_numel(node.shape)) {
                        throw std::invalid_argument(
                            "reshape must preserve tensor element count"
                        );
                    }
                    return {node.shape, false};
                }
                case expr_kind::transpose: {
                    expression_info source =
                        infer_expression(*node.lhs, context);
                    if (source.scalar) {
                        throw std::invalid_argument(
                            "transpose operand must be a tensor"
                        );
                    }
                    if (source.shape.size() != 2) {
                        throw std::invalid_argument(
                            "transpose currently requires a rank-2 tensor"
                        );
                    }
                    return {{source.shape[1], source.shape[0]}, false};
                }
                case expr_kind::broadcast: {
                    expression_info source =
                        infer_expression(*node.lhs, context);
                    if (source.scalar) return {node.shape, false};
                    require_broadcastable(source.shape, node.shape);
                    return {node.shape, false};
                }
                case expr_kind::concat: {
                    expression_info lhs = infer_expression(*node.lhs, context);
                    expression_info rhs = infer_expression(*node.rhs, context);
                    if (lhs.scalar || rhs.scalar) {
                        throw std::invalid_argument(
                            "concat operands must be tensors"
                        );
                    }
                    if (lhs.shape.size() != rhs.shape.size()) {
                        throw std::invalid_argument(
                            "concat operands must have the same rank"
                        );
                    }
                    require_axis(node.axis, lhs.shape.size(), "concat");
                    std::vector<index_t> shape = lhs.shape;
                    for (size_t dim = 0; dim < shape.size(); ++dim) {
                        if (dim == node.axis) continue;
                        if (lhs.shape[dim] != rhs.shape[dim]) {
                            throw std::invalid_argument(
                                "concat non-axis dimensions must match"
                            );
                        }
                    }
                    shape[node.axis] += rhs.shape[node.axis];
                    return {std::move(shape), false};
                }
                case expr_kind::sum:
                case expr_kind::max: {
                    expression_info source =
                        infer_expression(*node.lhs, context);
                    if (source.scalar) {
                        throw std::invalid_argument(
                            "reduction operand must be a tensor"
                        );
                    }
                    require_axis(node.axis, source.shape.size(), "reduction");
                    return {reduced_shape(source.shape, node.axis), false};
                }
                case expr_kind::exp: {
                    expression_info source =
                        infer_expression(*node.lhs, context);
                    if (source.scalar) return source;
                    return {source.shape, false};
                }
            }
            throw std::invalid_argument("unknown tensor expression node");
        }

        std::string scalar_literal(float32_t value) {
            std::ostringstream output;
            output << std::setprecision(FLT_DECIMAL_DIG) << value;
            std::string literal = output.str();
            if (literal.find('.') == std::string::npos &&
                literal.find('e') == std::string::npos &&
                literal.find('E') == std::string::npos) {
                literal += ".0";
            }
            literal += "f";
            return literal;
        }

        std::string emit_strided_index(
            std::span<const index_t> shape, std::span<const index_t> strides,
            index_t storage_offset, std::string_view index_expression
        ) {
            std::ostringstream expression;
            expression << storage_offset;

            index_t dimension_size = 1;
            for (size_t dim = shape.size(); dim > 0; --dim) {
                const size_t index = dim - 1;
                if (strides[index] != 0) {
                    expression << " + (((" << index_expression << ") / "
                               << dimension_size << ") % " << shape[index]
                               << ") * " << strides[index];
                }
                dimension_size *= shape[index];
            }

            return "(" + expression.str() + ")";
        }

        std::string emit_broadcast_index(
            std::span<const index_t> source_shape,
            std::span<const index_t> target_shape,
            std::string_view index_expression
        ) {
            std::ostringstream expression;
            expression << "0";

            const std::vector<index_t> source_strides =
                contiguous_strides(source_shape);
            index_t target_dimension_size = 1;
            std::vector<index_t> target_dimension_sizes(target_shape.size(), 1);
            for (size_t dim = target_shape.size(); dim > 0; --dim) {
                target_dimension_sizes[dim - 1] = target_dimension_size;
                target_dimension_size *= target_shape[dim - 1];
            }

            const size_t offset = target_shape.size() - source_shape.size();
            for (size_t dim = 0; dim < source_shape.size(); ++dim) {
                if (source_shape[dim] == 1) continue;
                const size_t target_dim = offset + dim;
                expression << " + (((" << index_expression << ") / "
                           << target_dimension_sizes[target_dim] << ") % "
                           << target_shape[target_dim] << ") * "
                           << source_strides[dim];
            }

            return "(" + expression.str() + ")";
        }

        std::string emit_transpose_index(
            std::span<const index_t> source_shape,
            std::string_view index_expression
        ) {
            return "(((" + std::string{index_expression} + ") % " +
                   std::to_string(source_shape[0]) + ") * " +
                   std::to_string(source_shape[1]) + " + ((" +
                   std::string{index_expression} + ") / " +
                   std::to_string(source_shape[0]) + "))";
        }

        std::string emit_reduction_source_index(
            std::span<const index_t> source_shape, size_t axis,
            std::string_view index_expression, std::string_view axis_index
        ) {
            const std::vector<index_t> source_strides =
                contiguous_strides(source_shape);
            const std::vector<index_t> output_shape =
                reduced_shape(source_shape, axis);
            std::vector<index_t> output_dimension_sizes(output_shape.size(), 1);
            index_t output_dimension_size = 1;
            for (size_t dim = output_shape.size(); dim > 0; --dim) {
                output_dimension_sizes[dim - 1] = output_dimension_size;
                output_dimension_size *= output_shape[dim - 1];
            }

            std::ostringstream expression;
            expression << "0";
            size_t output_dim = 0;
            for (size_t dim = 0; dim < source_shape.size(); ++dim) {
                if (dim == axis) {
                    expression << " + (" << axis_index << ") * "
                               << source_strides[dim];
                    continue;
                }

                expression << " + (((" << index_expression << ") / "
                           << output_dimension_sizes[output_dim] << ") % "
                           << output_shape[output_dim] << ") * "
                           << source_strides[dim];
                ++output_dim;
            }

            return "(" + expression.str() + ")";
        }

        std::string emit_expression(
            const tensor_expr::node& node, expression_context& context,
            std::string_view index_expression, std::ostringstream& statements
        ) {
            switch (node.kind) {
                case expr_kind::input:
                    return "in" +
                           std::to_string(
                               context.input_indices.at(node.input)
                           ) +
                           "[" + std::string{index_expression} + "]";
                case expr_kind::scalar:
                    return scalar_literal(node.scalar);
                case expr_kind::add:
                    return "(" +
                           emit_expression(
                               *node.lhs, context, index_expression, statements
                           ) +
                           " + " +
                           emit_expression(
                               *node.rhs, context, index_expression, statements
                           ) +
                           ")";
                case expr_kind::subtract:
                    return "(" +
                           emit_expression(
                               *node.lhs, context, index_expression, statements
                           ) +
                           " - " +
                           emit_expression(
                               *node.rhs, context, index_expression, statements
                           ) +
                           ")";
                case expr_kind::multiply:
                    return "(" +
                           emit_expression(
                               *node.lhs, context, index_expression, statements
                           ) +
                           " * " +
                           emit_expression(
                               *node.rhs, context, index_expression, statements
                           ) +
                           ")";
                case expr_kind::matmul: {
                    expression_info lhs = infer_expression(*node.lhs, context);
                    expression_info rhs = infer_expression(*node.rhs, context);
                    const std::string prefix =
                        "matmul" + std::to_string(context.temporary_count++);
                    const std::string row = prefix + "_row";
                    const std::string column = prefix + "_column";
                    const std::string inner_index = prefix + "_inner";
                    const std::string sum = prefix + "_sum";
                    const std::string lhs_index = "(" + row + " * " +
                                                  std::to_string(lhs.shape[1]) +
                                                  " + " + inner_index + ")";
                    const std::string rhs_index = "(" + inner_index + " * " +
                                                  std::to_string(rhs.shape[1]) +
                                                  " + " + column + ")";

                    statements
                        << "    const uint64_t " << row << " = ("
                        << index_expression << ") / " << rhs.shape[1] << ";\n"
                        << "    const uint64_t " << column << " = ("
                        << index_expression << ") % " << rhs.shape[1] << ";\n"
                        << "    float " << sum << " = 0.0f;\n"
                        << "    for (uint64_t " << inner_index << " = 0; "
                        << inner_index << " < " << lhs.shape[1] << "; ++"
                        << inner_index << ") {\n";
                    std::ostringstream loop_statements;
                    const std::string lhs_value = emit_expression(
                        *node.lhs, context, lhs_index, loop_statements
                    );
                    const std::string rhs_value = emit_expression(
                        *node.rhs, context, rhs_index, loop_statements
                    );
                    statements << loop_statements.str() << "        " << sum
                               << " += " << lhs_value << " * " << rhs_value
                               << ";\n"
                               << "    }\n";
                    return sum;
                }
                case expr_kind::slice: {
                    const std::string source_index = emit_strided_index(
                        node.slice_shape, node.slice_strides, node.slice_offset,
                        index_expression
                    );
                    return emit_expression(
                        *node.lhs, context, source_index, statements
                    );
                }
                case expr_kind::reshape:
                    return emit_expression(
                        *node.lhs, context, index_expression, statements
                    );
                case expr_kind::transpose: {
                    const expression_info source =
                        infer_expression(*node.lhs, context);
                    const std::string source_index =
                        emit_transpose_index(source.shape, index_expression);
                    return emit_expression(
                        *node.lhs, context, source_index, statements
                    );
                }
                case expr_kind::broadcast: {
                    const expression_info source =
                        infer_expression(*node.lhs, context);
                    if (source.scalar) {
                        return emit_expression(
                            *node.lhs, context, index_expression, statements
                        );
                    }
                    const std::string source_index = emit_broadcast_index(
                        source.shape, node.shape, index_expression
                    );
                    return emit_expression(
                        *node.lhs, context, source_index, statements
                    );
                }
                case expr_kind::concat: {
                    const expression_info lhs =
                        infer_expression(*node.lhs, context);
                    const expression_info rhs =
                        infer_expression(*node.rhs, context);
                    std::vector<index_t> output_shape = lhs.shape;
                    output_shape[node.axis] += rhs.shape[node.axis];
                    const std::vector<index_t> output_strides =
                        contiguous_strides(output_shape);
                    const std::string prefix =
                        "concat" + std::to_string(context.temporary_count++);
                    const std::string axis_coord = prefix + "_axis";
                    const std::string lhs_index = emit_strided_index(
                        output_shape, contiguous_strides(lhs.shape), 0,
                        index_expression
                    );

                    std::vector<index_t> rhs_strides =
                        contiguous_strides(rhs.shape);
                    std::ostringstream rhs_expression;
                    rhs_expression << "0";
                    for (size_t dim = 0; dim < output_shape.size(); ++dim) {
                        const std::string coord =
                            "(((" + std::string{index_expression} + ") / " +
                            std::to_string(output_strides[dim]) + ") % " +
                            std::to_string(output_shape[dim]) + ")";
                        if (dim == node.axis) {
                            rhs_expression << " + ((" << coord << ") - "
                                           << lhs.shape[dim] << ") * "
                                           << rhs_strides[dim];
                        }
                        else {
                            rhs_expression << " + (" << coord << ") * "
                                           << rhs_strides[dim];
                        }
                    }

                    statements << "    const uint64_t " << axis_coord << " = (("
                               << index_expression << ") / "
                               << output_strides[node.axis] << ") % "
                               << output_shape[node.axis] << ";\n";
                    return "((" + axis_coord + " < " +
                           std::to_string(lhs.shape[node.axis]) + ") ? " +
                           emit_expression(
                               *node.lhs, context, lhs_index, statements
                           ) +
                           " : " +
                           emit_expression(
                               *node.rhs, context,
                               "(" + rhs_expression.str() + ")", statements
                           ) +
                           ")";
                }
                case expr_kind::sum:
                case expr_kind::max: {
                    const expression_info source =
                        infer_expression(*node.lhs, context);
                    const std::string prefix =
                        "reduce" + std::to_string(context.temporary_count++);
                    const std::string axis_index = prefix + "_axis";
                    const std::string total = prefix + "_total";
                    const std::string initial_value =
                        node.kind == expr_kind::sum ? "0.0f" : "-FLT_MAX";
                    statements << "    float " << total << " = "
                               << initial_value << ";\n"
                               << "    for (uint64_t " << axis_index << " = 0; "
                               << axis_index << " < " << source.shape[node.axis]
                               << "; ++" << axis_index << ") {\n";
                    std::ostringstream loop_statements;
                    const std::string source_index =
                        emit_reduction_source_index(
                            source.shape, node.axis, index_expression,
                            axis_index
                        );
                    const std::string value = emit_expression(
                        *node.lhs, context, source_index, loop_statements
                    );
                    statements << loop_statements.str();
                    if (node.kind == expr_kind::sum) {
                        statements << "        " << total << " += " << value
                                   << ";\n";
                    }
                    else {
                        statements << "        " << total << " = " << total
                                   << " > " << value << " ? " << total << " : "
                                   << value << ";\n";
                    }
                    statements << "    }\n";
                    return total;
                }
                case expr_kind::exp: {
                    return "__expf(" +
                           emit_expression(
                               *node.lhs, context, index_expression, statements
                           ) +
                           ")";
                }
            }
            throw std::invalid_argument("unknown tensor expression node");
        }

        std::string fused_kernel_source(
            const expression_context& context, std::string_view statements,
            std::string_view expression
        ) {
            std::ostringstream source;
            source << "typedef unsigned long long uint64_t;\n"
                   << "#define FLT_MAX 3.4028234663852886e+38f\n"
                   << "extern \"C\" __global__ void porch_fused_elementwise(";
            for (size_t index = 0; index < context.inputs.size(); ++index) {
                if (index != 0) source << ", ";
                source << "const float* in" << index;
            }
            if (!context.inputs.empty()) source << ", ";
            source << "float* out, uint64_t count) {\n"
                   << "    const uint64_t index = blockIdx.x * blockDim.x + "
                      "threadIdx.x;\n"
                   << "    if (index >= count) return;\n"
                   << statements << "    out[index] = " << expression << ";\n"
                   << "}\n";
            return source.str();
        }

        tensor materialize_expression(const tensor_expr::node& node) {
            expression_context context;
            expression_info info = infer_expression(node, context);
            if (!context.initialized)
                throw std::invalid_argument(
                    "tensor expression must reference at least one tensor"
                );
            if (info.scalar)
                throw std::invalid_argument(
                    "scalar expression cannot materialize to a tensor"
                );

            const size_t count = checked_numel(info.shape);
            std::vector<float32_t> result_values(count);
            cuda_jit::device_buffer result_device{
                result_values.size() * sizeof(float32_t)
            };
            if (result_values.empty()) {
                return make_materialized_tensor(
                    std::move(info.shape), std::move(result_values),
                    std::move(result_device), true, context.placement
                );
            }

            std::ostringstream statements;
            const std::string expression_source =
                emit_expression(node, context, "index", statements);
            const std::string kernel_source = fused_kernel_source(
                context, statements.str(), expression_source
            );
            const std::string ptx = cuda_jit::compile_to_ptx(kernel_source);

            std::vector<const cuda_jit::device_buffer*> inputs;
            inputs.reserve(context.inputs.size());
            for (const tensor* input : context.inputs) {
                inputs.push_back(&input->device_data());
            }

            cuda_jit::launch_fused_elementwise(
                ptx, "porch_fused_elementwise", inputs, result_device, count
            );
            return make_materialized_tensor(
                std::move(info.shape), std::move(result_values),
                std::move(result_device), false, context.placement
            );
        }

    } // namespace

    std::shared_ptr<realized_storage> lazy_materialization::storage() {
        std::call_once(once, [this] {
            tensor materialized = materialize_expression(*expression);
            realized = std::make_shared<realized_storage>(
                std::vector<float32_t>(materialized.numel()),
                materialized.device_data(), false
            );
        });
        return realized;
    }

    tensor_index::tensor_index(index_t index)
        : kind_(kind::index), index_(index) {}

    tensor_index::tensor_index(slice range)
        : kind_(kind::slice), slice_(range) {}

    tensor_index::tensor_index(all_t) : kind_(kind::all) {}

    tensor_layout::tensor_layout(std::vector<index_t> shape)
        : shape_(std::move(shape)), strides_(contiguous_strides(shape_)) {
        (void)checked_numel(shape_);
    }

    tensor_layout::tensor_layout(
        std::vector<index_t> shape, std::vector<index_t> strides,
        index_t storage_offset
    )
        : shape_(std::move(shape)), strides_(std::move(strides)),
          storage_offset_(storage_offset) {
        if (shape_.size() != strides_.size()) {
            throw std::invalid_argument("tensor layout rank mismatch");
        }
        if (storage_offset_ < 0) {
            throw std::invalid_argument(
                "tensor layout offset must be non-negative"
            );
        }
        for (const index_t stride : strides_) {
            if (stride < 0) {
                throw std::invalid_argument(
                    "tensor layout strides must be non-negative"
                );
            }
        }
        (void)checked_numel(shape_);
    }

    std::span<const index_t> tensor_layout::shape() const noexcept {
        return shape_;
    }

    std::span<const index_t> tensor_layout::strides() const noexcept {
        return strides_;
    }

    index_t tensor_layout::storage_offset() const noexcept {
        return storage_offset_;
    }

    size_t tensor_layout::rank() const noexcept { return shape_.size(); }

    size_t tensor_layout::numel() const noexcept {
        return checked_numel(shape_);
    }

    bool tensor_layout::is_contiguous() const noexcept {
        return storage_offset_ == 0 && is_contiguous_layout(shape_, strides_);
    }

    tensor::tensor(
        std::vector<index_t> shape, std::vector<float32_t> values,
        device placement
    )
        : state_(new state{
              tensor_layout{std::move(shape)}, std::move(values),
              cuda_jit::device_buffer{}, true, placement, nullptr
          }) {
        const size_t expected = state_->layout.numel();
        std::shared_ptr<realized_storage> storage = state_->storage();
        if (storage->values.size() != expected) {
            throw std::invalid_argument(
                "tensor data size does not match shape"
            );
        }
        storage->device_values = cuda_jit::make_device_buffer(storage->values);
    }

    tensor::tensor(
        std::vector<index_t> shape, std::vector<float32_t> values,
        cuda_jit::device_buffer device_values, device placement
    )
        : tensor(
              std::move(shape), std::move(values), std::move(device_values),
              true, placement
          ) {}

    tensor::tensor(
        std::vector<index_t> shape, std::vector<float32_t> values,
        cuda_jit::device_buffer device_values, bool host_current,
        device placement
    )
        : tensor(
              tensor_layout{std::move(shape)}, std::move(values),
              std::move(device_values), host_current, placement
          ) {}

    tensor::tensor(
        tensor_layout layout, std::vector<float32_t> values,
        cuda_jit::device_buffer device_values, bool host_current,
        device placement
    )
        : state_(new state{
              std::move(layout), std::move(values), std::move(device_values),
              host_current, placement, nullptr
          }) {
        const size_t expected = state_->layout.numel();
        std::shared_ptr<realized_storage> storage = state_->storage();
        if (storage->values.size() != expected) {
            throw std::invalid_argument(
                "tensor data size does not match shape"
            );
        }
        if (!state_->layout.is_contiguous()) {
            throw std::invalid_argument(
                "non-contiguous tensor storage is not implemented yet"
            );
        }
        if (storage->device_values.size_bytes() <
            storage->values.size() * sizeof(float32_t)) {
            throw std::invalid_argument("tensor device storage is too small");
        }
    }

    std::span<const index_t> tensor::shape() const noexcept {
        return state_->layout.shape();
    }

    std::span<const index_t> tensor::strides() const noexcept {
        return state_->layout.strides();
    }

    const tensor_layout& tensor::layout() const noexcept {
        return state_->layout;
    }

    index_t tensor::storage_offset() const noexcept {
        return state_->layout.storage_offset();
    }

    bool tensor::is_contiguous() const noexcept {
        return state_->layout.is_contiguous();
    }

    size_t tensor::rank() const noexcept { return state_->layout.rank(); }

    size_t tensor::numel() const noexcept { return state_->layout.numel(); }

    device tensor::placement() const noexcept { return state_->placement; }

    std::span<const float32_t> tensor::data() const {
        std::shared_ptr<realized_storage> storage = state_->storage();
        std::scoped_lock lock{storage->host_mutex};
        if (!storage->host_current) {
            cuda_jit::copy_to_host(storage->device_values, storage->values);
            storage->host_current = true;
        }
        return storage->values;
    }

    std::vector<float32_t> tensor::cpu() const {
        const std::span<const float32_t> host = data();
        return {host.begin(), host.end()};
    }

    const cuda_jit::device_buffer& tensor::device_data() const {
        return state_->storage()->device_values;
    }

    const tensor& tensor::realize() const {
        ensure_materialized();
        return *this;
    }

    void tensor::synchronize() const {
        ensure_materialized();
        cuda_jit::synchronize();
    }

    void tensor::ensure_materialized() const { (void)state_->storage(); }

    tensor_expr tensor::operator[](
        std::initializer_list<tensor_index> indices
    ) const {
        if (indices.size() > rank()) {
            throw std::invalid_argument("too many tensor indices");
        }

        std::vector<index_t> output_shape;
        output_shape.reserve(rank());
        std::vector<index_t> output_source_strides;
        output_source_strides.reserve(rank());
        index_t source_offset = storage_offset();

        size_t dim = 0;
        for (const tensor_index& index : indices) {
            const index_t extent = shape()[dim];
            const index_t source_stride = strides()[dim];
            switch (index.kind_) {
                case tensor_index::kind::index:
                    source_offset +=
                        normalize_index(index.index_, extent) * source_stride;
                    break;
                case tensor_index::kind::slice: {
                    const index_t start =
                        normalize_slice_endpoint(index.slice_.start, extent);
                    const index_t stop =
                        normalize_slice_endpoint(index.slice_.stop, extent);
                    const size_t length =
                        slice_length(start, stop, index.slice_.step);
                    source_offset += start * source_stride;
                    output_shape.push_back(static_cast<index_t>(length));
                    output_source_strides.push_back(
                        source_stride * index.slice_.step
                    );
                    break;
                }
                case tensor_index::kind::all:
                    output_shape.push_back(extent);
                    output_source_strides.push_back(source_stride);
                    break;
            }
            ++dim;
        }

        for (; dim < rank(); ++dim) {
            output_shape.push_back(shape()[dim]);
            output_source_strides.push_back(strides()[dim]);
        }

        (void)checked_numel(output_shape);
        return tensor_expr{make_slice_node(
            tensor_expr{*this}.root_, std::move(output_shape),
            std::move(output_source_strides), source_offset
        )};
    }

    tensor_expr tensor::operator[](tensor_index index) const {
        return (*this)[{index}];
    }

    tensor_expr::tensor_expr(const tensor& value)
        : root_(
              value.state_->lazy == nullptr
                  ? make_node(expr_kind::input, &value)
                  : value.state_->lazy->expression
          ) {}

    tensor_expr::tensor_expr(float32_t value)
        : root_(make_node(expr_kind::scalar, nullptr, value)) {}

    tensor_expr::tensor_expr(std::shared_ptr<const node> root)
        : root_(std::move(root)) {}

    tensor tensor_expr::eval() const { return materialize(*this); }

    tensor_expr::operator tensor() const {
        expression_context context;
        expression_info info = infer_expression(*root_, context);
        if (!context.initialized)
            throw std::invalid_argument(
                "tensor expression must reference at least one tensor"
            );
        if (info.scalar)
            throw std::invalid_argument(
                "scalar expression cannot convert to a tensor"
            );

        const size_t count = checked_numel(info.shape);
        tensor result{{0}, {}};
        result.state_ = std::shared_ptr<tensor::state>(new tensor::state{
            tensor_layout{std::move(info.shape)}, std::vector<float32_t>(count),
            cuda_jit::device_buffer{}, false, context.placement, root_
        });
        return result;
    }

    tensor full(std::vector<index_t> shape, float32_t value, device placement) {
        const size_t count = checked_numel(shape);
        return tensor{
            std::move(shape), std::vector<float32_t>(count, value), placement
        };
    }

    tensor zeros(std::vector<index_t> shape, device placement) {
        return full(std::move(shape), 0.0F, placement);
    }

    tensor add(const tensor& lhs, const tensor& rhs) {
        return materialize(tensor_expr{lhs} + tensor_expr{rhs});
    }

    tensor subtract(const tensor& lhs, const tensor& rhs) {
        return materialize(tensor_expr{lhs} - tensor_expr{rhs});
    }

    tensor multiply(const tensor& lhs, const tensor& rhs) {
        return materialize(tensor_expr{lhs} * tensor_expr{rhs});
    }

    tensor make_materialized_tensor(
        std::vector<index_t> shape, std::vector<float32_t> values,
        cuda_jit::device_buffer device_values, bool host_current,
        device placement
    ) {
        return tensor{
            std::move(shape), std::move(values), std::move(device_values),
            host_current, placement
        };
    }

    tensor_expr matmul(tensor_expr lhs, tensor_expr rhs) {
        return tensor_expr{make_node(
            expr_kind::matmul, nullptr, 0.0F, std::move(lhs.root_),
            std::move(rhs.root_)
        )};
    }

    tensor_expr reshape(tensor_expr value, std::vector<index_t> shape) {
        return tensor_expr{make_shape_node(
            expr_kind::reshape, std::move(value.root_), std::move(shape)
        )};
    }

    tensor_expr transpose(tensor_expr value) {
        return tensor_expr{make_node(
            expr_kind::transpose, nullptr, 0.0F, std::move(value.root_)
        )};
    }

    tensor_expr broadcast_to(tensor_expr value, std::vector<index_t> shape) {
        return tensor_expr{make_shape_node(
            expr_kind::broadcast, std::move(value.root_), std::move(shape)
        )};
    }

    tensor_expr concat(tensor_expr lhs, tensor_expr rhs, size_t axis) {
        return tensor_expr{
            make_concat_node(std::move(lhs.root_), std::move(rhs.root_), axis)
        };
    }

    tensor_expr sum(tensor_expr value, size_t axis) {
        return tensor_expr{
            make_axis_node(expr_kind::sum, std::move(value.root_), axis)
        };
    }

    tensor_expr max(tensor_expr value, size_t axis) {
        return tensor_expr{
            make_axis_node(expr_kind::max, std::move(value.root_), axis)
        };
    }

    tensor_expr exp(tensor_expr value) {
        return tensor_expr{
            make_node(expr_kind::exp, nullptr, 0.0F, std::move(value.root_))
        };
    }

    tensor materialize(const tensor_expr& expression) {
        return materialize_expression(*expression.root_);
    }

    tensor_expr operator+(tensor_expr lhs, tensor_expr rhs) {
        return tensor_expr{make_node(
            expr_kind::add, nullptr, 0.0F, std::move(lhs.root_),
            std::move(rhs.root_)
        )};
    }

    tensor_expr operator-(tensor_expr lhs, tensor_expr rhs) {
        return tensor_expr{make_node(
            expr_kind::subtract, nullptr, 0.0F, std::move(lhs.root_),
            std::move(rhs.root_)
        )};
    }

    tensor_expr operator*(tensor_expr lhs, tensor_expr rhs) {
        return tensor_expr{make_node(
            expr_kind::multiply, nullptr, 0.0F, std::move(lhs.root_),
            std::move(rhs.root_)
        )};
    }

} // namespace porch
