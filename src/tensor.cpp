#include "porch/tensor.hpp"
#include "porch/cuda_jit.hpp"

#include <float.h>

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace porch {

    enum class expr_kind {
        input,
        scalar,
        add,
        subtract,
        multiply,
    };

    struct tensor_expr::node {
        expr_kind kind;
        const tensor* input = nullptr;
        float32_t scalar = 0.0F;
        std::shared_ptr<const node> lhs;
        std::shared_ptr<const node> rhs;
    };

    struct tensor::state {
        tensor_layout layout;
        mutable std::vector<float32_t> values;
        cuda_jit::device_buffer device_values;
        mutable bool host_current = true;
        device placement;
    };

    namespace {

        size_t checked_numel(std::span<const index_t> shape) {
            size_t count = 1;
            for (const index_t extent : shape) {
                if (extent < 0) {
                    throw std::invalid_argument(
                        "tensor shape extents must be non-negative");
                }
                count *= static_cast<size_t>(extent);
            }
            return count;
        }

        std::vector<index_t>
        contiguous_strides(std::span<const index_t> shape) {
            std::vector<index_t> strides(shape.size(), 1);
            index_t stride = 1;
            for (size_t index = shape.size(); index > 0; --index) {
                strides[index - 1] = stride;
                stride *= shape[index - 1];
            }
            return strides;
        }

        bool is_contiguous_layout(std::span<const index_t> shape,
                                  std::span<const index_t> strides) {
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
                    "tensor slice step must be positive");
            }
            if (stop <= start) return 0;
            return static_cast<size_t>((stop - start + step - 1) / step);
        }

        size_t flat_index(std::span<const index_t> indices,
                          std::span<const index_t> strides,
                          index_t storage_offset) {
            index_t offset = storage_offset;
            for (size_t dim = 0; dim < indices.size(); ++dim) {
                offset += indices[dim] * strides[dim];
            }
            return static_cast<size_t>(offset);
        }

        struct expression_context {
            std::vector<const tensor*> inputs;
            std::map<const tensor*, size_t> input_indices;
            std::vector<index_t> shape;
            device placement;
            bool initialized = false;
        };

        void record_input(expression_context& context, const tensor& value) {
            if (!context.initialized) {
                context.shape.assign(value.shape().begin(),
                                     value.shape().end());
                context.placement = value.placement();
                context.initialized = true;
            }
            else {
                if (!std::ranges::equal(context.shape, value.shape()))
                    throw std::invalid_argument(
                        "tensor expression shapes must match");
                if (context.placement != value.placement())
                    throw std::invalid_argument(
                        "tensor expression devices must match");
            }

            if (context.input_indices.contains(&value)) return;
            const size_t index = context.inputs.size();
            context.inputs.push_back(&value);
            context.input_indices.emplace(&value, index);
        }

        void analyze_expression(const tensor_expr::node& node,
                                expression_context& context) {
            switch (node.kind) {
            case expr_kind::input:
                record_input(context, *node.input);
                return;
            case expr_kind::scalar:
                return;
            case expr_kind::add:
            case expr_kind::subtract:
            case expr_kind::multiply:
                analyze_expression(*node.lhs, context);
                analyze_expression(*node.rhs, context);
                return;
            }
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

        std::string emit_expression(const tensor_expr::node& node,
                                    const expression_context& context) {
            switch (node.kind) {
            case expr_kind::input:
                return "in" +
                       std::to_string(context.input_indices.at(node.input)) +
                       "[index]";
            case expr_kind::scalar:
                return scalar_literal(node.scalar);
            case expr_kind::add:
                return "(" + emit_expression(*node.lhs, context) + " + " +
                       emit_expression(*node.rhs, context) + ")";
            case expr_kind::subtract:
                return "(" + emit_expression(*node.lhs, context) + " - " +
                       emit_expression(*node.rhs, context) + ")";
            case expr_kind::multiply:
                return "(" + emit_expression(*node.lhs, context) + " * " +
                       emit_expression(*node.rhs, context) + ")";
            }
            throw std::invalid_argument("unknown tensor expression node");
        }

        std::string fused_kernel_source(const expression_context& context,
                                        std::string_view expression) {
            std::ostringstream source;
            source << "typedef unsigned long long uint64_t;\n"
                   << "extern \"C\" __global__ void porch_fused_elementwise(";
            for (size_t index = 0; index < context.inputs.size(); ++index) {
                if (index != 0) source << ", ";
                source << "const float* in" << index;
            }
            if (!context.inputs.empty()) source << ", ";
            source << "float* out, uint64_t count) {\n"
                   << "    const uint64_t index = blockIdx.x * blockDim.x + "
                      "threadIdx.x;\n"
                   << "    if (index < count) out[index] = " << expression
                   << ";\n"
                   << "}\n";
            return source.str();
        }

    } // namespace

    tensor_index::tensor_index(index_t index)
        : kind_(kind::index), index_(index) {}

    tensor_index::tensor_index(slice range)
        : kind_(kind::slice), slice_(range) {}

    tensor_index::tensor_index(all_t) : kind_(kind::all) {}

    tensor_layout::tensor_layout(std::vector<index_t> shape)
        : shape_(std::move(shape)), strides_(contiguous_strides(shape_)) {
        (void)checked_numel(shape_);
    }

    tensor_layout::tensor_layout(std::vector<index_t> shape,
                                 std::vector<index_t> strides,
                                 index_t storage_offset)
        : shape_(std::move(shape)), strides_(std::move(strides)),
          storage_offset_(storage_offset) {
        if (shape_.size() != strides_.size()) {
            throw std::invalid_argument("tensor layout rank mismatch");
        }
        if (storage_offset_ < 0) {
            throw std::invalid_argument(
                "tensor layout offset must be non-negative");
        }
        for (const index_t stride : strides_) {
            if (stride < 0) {
                throw std::invalid_argument(
                    "tensor layout strides must be non-negative");
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

    tensor::tensor(std::vector<index_t> shape, std::vector<float32_t> values,
                   device placement)
        : state_(std::make_shared<state>(state{tensor_layout{std::move(shape)},
                                               std::move(values),
                                               {},
                                               true,
                                               placement})) {
        const size_t expected = state_->layout.numel();
        if (state_->values.size() != expected) {
            throw std::invalid_argument(
                "tensor data size does not match shape");
        }
        state_->device_values = cuda_jit::make_device_buffer(state_->values);
    }

    tensor::tensor(std::vector<index_t> shape, std::vector<float32_t> values,
                   cuda_jit::device_buffer device_values, device placement)
        : tensor(std::move(shape), std::move(values), std::move(device_values),
                 true, placement) {}

    tensor::tensor(std::vector<index_t> shape, std::vector<float32_t> values,
                   cuda_jit::device_buffer device_values, bool host_current,
                   device placement)
        : tensor(tensor_layout{std::move(shape)}, std::move(values),
                 std::move(device_values), host_current, placement) {}

    tensor::tensor(tensor_layout layout, std::vector<float32_t> values,
                   cuda_jit::device_buffer device_values, bool host_current,
                   device placement)
        : state_(std::make_shared<state>(
              state{std::move(layout), std::move(values),
                    std::move(device_values), host_current, placement})) {
        const size_t expected = state_->layout.numel();
        if (state_->values.size() != expected) {
            throw std::invalid_argument(
                "tensor data size does not match shape");
        }
        if (!state_->layout.is_contiguous()) {
            throw std::invalid_argument(
                "non-contiguous tensor storage is not implemented yet");
        }
        if (state_->device_values.size_bytes() <
            state_->values.size() * sizeof(float32_t)) {
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

    size_t tensor::numel() const noexcept { return state_->values.size(); }

    device tensor::placement() const noexcept { return state_->placement; }

    std::span<const float32_t> tensor::data() const {
        if (!state_->host_current) {
            cuda_jit::copy_to_host(state_->device_values, state_->values);
            state_->host_current = true;
        }
        return state_->values;
    }

    const cuda_jit::device_buffer& tensor::device_data() const noexcept {
        return state_->device_values;
    }

    tensor
    tensor::operator[](std::initializer_list<tensor_index> indices) const {
        if (indices.size() > rank()) {
            throw std::invalid_argument("too many tensor indices");
        }

        struct dimension_selection {
            index_t start = 0;
            index_t step = 1;
            index_t fixed = 0;
            bool keeps_dimension = true;
        };

        std::vector<dimension_selection> selections;
        selections.reserve(rank());
        std::vector<index_t> output_shape;
        output_shape.reserve(rank());

        size_t dim = 0;
        for (const tensor_index& index : indices) {
            const index_t extent = shape()[dim];
            dimension_selection selection;
            switch (index.kind_) {
            case tensor_index::kind::index:
                selection.fixed = normalize_index(index.index_, extent);
                selection.keeps_dimension = false;
                break;
            case tensor_index::kind::slice: {
                const index_t start =
                    normalize_slice_endpoint(index.slice_.start, extent);
                const index_t stop =
                    normalize_slice_endpoint(index.slice_.stop, extent);
                const size_t length =
                    slice_length(start, stop, index.slice_.step);
                selection.start = start;
                selection.step = index.slice_.step;
                output_shape.push_back(static_cast<index_t>(length));
                break;
            }
            case tensor_index::kind::all:
                selection.start = 0;
                selection.step = 1;
                output_shape.push_back(extent);
                break;
            }
            selections.push_back(selection);
            ++dim;
        }

        for (; dim < rank(); ++dim) {
            selections.push_back(dimension_selection{0, 1, 0, true});
            output_shape.push_back(shape()[dim]);
        }

        const size_t output_count = checked_numel(output_shape);
        std::vector<float32_t> output_values(output_count);
        const std::span<const float32_t> source_values = data();

        for (size_t linear = 0; linear < output_count; ++linear) {
            size_t remaining = linear;
            size_t output_dim = output_shape.size();
            std::vector<index_t> source_indices(rank(), 0);

            for (size_t source_dim = rank(); source_dim > 0; --source_dim) {
                const dimension_selection& selection =
                    selections[source_dim - 1];
                if (!selection.keeps_dimension) {
                    source_indices[source_dim - 1] = selection.fixed;
                    continue;
                }

                --output_dim;
                const index_t extent = output_shape[output_dim];
                const index_t coordinate =
                    extent == 0 ? 0
                                : static_cast<index_t>(
                                      remaining % static_cast<size_t>(extent));
                if (extent != 0) {
                    remaining /= static_cast<size_t>(extent);
                }
                source_indices[source_dim - 1] =
                    selection.start + coordinate * selection.step;
            }

            output_values[linear] = source_values[flat_index(
                source_indices, strides(), storage_offset())];
        }

        return tensor{std::move(output_shape), std::move(output_values),
                      state_->placement};
    }

    tensor tensor::operator[](tensor_index index) const {
        return (*this)[{index}];
    }

    tensor_expr::tensor_expr(const tensor& value)
        : root_(std::make_shared<node>(
              node{expr_kind::input, &value, 0.0F, nullptr, nullptr})) {}

    tensor_expr::tensor_expr(float32_t value)
        : root_(std::make_shared<node>(
              node{expr_kind::scalar, nullptr, value, nullptr, nullptr})) {}

    tensor_expr::tensor_expr(std::shared_ptr<const node> root)
        : root_(std::move(root)) {}

    tensor tensor_expr::eval() const { return materialize(*this); }

    tensor_expr::operator tensor() const { return eval(); }

    tensor full(std::vector<index_t> shape, float32_t value, device placement) {
        const size_t count = checked_numel(shape);
        return tensor{std::move(shape), std::vector<float32_t>(count, value),
                      placement};
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

    tensor materialize(const tensor_expr& expression) {
        expression_context context;
        analyze_expression(*expression.root_, context);
        if (!context.initialized)
            throw std::invalid_argument(
                "tensor expression must reference at least one tensor");

        const size_t count = checked_numel(context.shape);
        std::vector<float32_t> result_values(count);
        cuda_jit::device_buffer result_device{result_values.size() *
                                              sizeof(float32_t)};
        if (result_values.empty()) {
            return tensor{std::move(context.shape), std::move(result_values),
                          std::move(result_device), context.placement};
        }

        const std::string expression_source =
            emit_expression(*expression.root_, context);
        const std::string kernel_source =
            fused_kernel_source(context, expression_source);
        const std::string ptx = cuda_jit::compile_to_ptx(kernel_source);

        std::vector<const cuda_jit::device_buffer*> inputs;
        inputs.reserve(context.inputs.size());
        for (const tensor* input : context.inputs) {
            inputs.push_back(&input->device_data());
        }

        cuda_jit::launch_fused_elementwise(ptx, "porch_fused_elementwise",
                                           inputs, result_device, count);
        return tensor{std::move(context.shape), std::move(result_values),
                      std::move(result_device), false, context.placement};
    }

    tensor_expr operator+(tensor_expr lhs, tensor_expr rhs) {
        return tensor_expr{std::make_shared<tensor_expr::node>(
            tensor_expr::node{expr_kind::add, nullptr, 0.0F,
                              std::move(lhs.root_), std::move(rhs.root_)})};
    }

    tensor_expr operator-(tensor_expr lhs, tensor_expr rhs) {
        return tensor_expr{std::make_shared<tensor_expr::node>(
            tensor_expr::node{expr_kind::subtract, nullptr, 0.0F,
                              std::move(lhs.root_), std::move(rhs.root_)})};
    }

    tensor_expr operator*(tensor_expr lhs, tensor_expr rhs) {
        return tensor_expr{std::make_shared<tensor_expr::node>(
            tensor_expr::node{expr_kind::multiply, nullptr, 0.0F,
                              std::move(lhs.root_), std::move(rhs.root_)})};
    }

} // namespace porch
