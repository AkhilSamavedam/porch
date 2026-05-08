#include "porch/tensor.hpp"
#include "porch/cuda_jit.hpp"

#include <float.h>

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
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

struct expression_context {
    std::vector<const tensor*> inputs;
    std::map<const tensor*, size_t> input_indices;
    std::vector<index_t> shape;
    device placement;
    bool initialized = false;
};

void record_input(expression_context& context, const tensor& value) {
    if (!context.initialized) {
        context.shape.assign(value.shape().begin(), value.shape().end());
        context.placement = value.placement();
        context.initialized = true;
    }
    else {
        if (!std::ranges::equal(context.shape, value.shape()))
            throw std::invalid_argument("tensor expression shapes must match");
        if (context.placement != value.placement())
            throw std::invalid_argument("tensor expression devices must match");
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
        return "in" + std::to_string(context.input_indices.at(node.input)) +
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
           << "    if (index < count) out[index] = " << expression << ";\n"
           << "}\n";
    return source.str();
}

} // namespace

tensor::tensor(std::vector<index_t> shape, std::vector<float32_t> values,
               device placement)
    : shape_(std::move(shape)), values_(std::move(values)),
      placement_(placement) {
    const size_t expected = checked_numel(shape_);
    if (values_.size() != expected) {
        throw std::invalid_argument("tensor data size does not match shape");
    }
    device_values_ = cuda_jit::make_device_buffer(values_);
}

tensor::tensor(std::vector<index_t> shape, std::vector<float32_t> values,
               cuda_jit::device_buffer device_values, device placement)
    : tensor(std::move(shape), std::move(values), std::move(device_values),
             true, placement) {}

tensor::tensor(std::vector<index_t> shape, std::vector<float32_t> values,
               cuda_jit::device_buffer device_values, bool host_current,
               device placement)
    : shape_(std::move(shape)), values_(std::move(values)),
      device_values_(std::move(device_values)), host_current_(host_current),
      placement_(placement) {
    const size_t expected = checked_numel(shape_);
    if (values_.size() != expected) {
        throw std::invalid_argument("tensor data size does not match shape");
    }
    if (device_values_.size_bytes() < values_.size() * sizeof(float32_t)) {
        throw std::invalid_argument("tensor device storage is too small");
    }
}

std::span<const index_t> tensor::shape() const noexcept { return shape_; }

size_t tensor::rank() const noexcept { return shape_.size(); }

size_t tensor::numel() const noexcept { return values_.size(); }

device tensor::placement() const noexcept { return placement_; }

std::span<const float32_t> tensor::data() const {
    if (!host_current_) {
        cuda_jit::copy_to_host(device_values_, values_);
        host_current_ = true;
    }
    return values_;
}

const cuda_jit::device_buffer& tensor::device_data() const noexcept {
    return device_values_;
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

    cuda_jit::launch_fused_elementwise(ptx, "porch_fused_elementwise", inputs,
                                       result_device, count);
    return tensor{std::move(context.shape), std::move(result_values),
                  std::move(result_device), false, context.placement};
}

tensor_expr operator+(tensor_expr lhs, tensor_expr rhs) {
    return tensor_expr{std::make_shared<tensor_expr::node>(
        tensor_expr::node{expr_kind::add, nullptr, 0.0F, std::move(lhs.root_),
                          std::move(rhs.root_)})};
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
