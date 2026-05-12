#pragma once

#include "porch/tensor.hpp"

namespace porch {

    [[nodiscard]] tensor_expr relu(tensor_expr value);
    [[nodiscard]] tensor_expr softmax(tensor_expr value, size_t axis);
    [[nodiscard]] tensor_expr logistic(tensor_expr value);

} // namespace porch
