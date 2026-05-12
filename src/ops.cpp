#include "porch/ops.hpp"

namespace porch {

    tensor_expr relu(tensor_expr value) { return maximum(value, 0.0F); }

    tensor_expr softmax(tensor_expr value, size_t axis) {
        const std::vector<index_t> output_shape = value.shape();
        tensor_expr shifted =
            value - broadcast_to(max(value, axis, true), output_shape);
        tensor_expr numerator = exp(shifted);
        return numerator /
               broadcast_to(sum(numerator, axis, true), output_shape);
    }

    tensor_expr logistic(tensor_expr value) {
        return reciprocal(1.0F + exp(0.0F - value));
    }

} // namespace porch
