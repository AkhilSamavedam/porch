#include "porch/tensor.hpp"

#include <algorithm>
#include <stdexcept>

namespace porch {
namespace {

std::size_t checked_numel(std::span<const index_t> shape) {
    std::size_t count = 1;
    for (const index_t extent : shape) {
        if (extent < 0) {
            throw std::invalid_argument(
                "tensor shape extents must be non-negative");
        }
        count *= static_cast<std::size_t>(extent);
    }
    return count;
}

void require_same_shape(const tensor& lhs, const tensor& rhs) {
    if (!std::ranges::equal(lhs.shape(), rhs.shape())) {
        throw std::invalid_argument("tensor shapes must match");
    }
}

void require_same_device(const tensor& lhs, const tensor& rhs) {
    if (lhs.placement() != rhs.placement()) {
        throw std::invalid_argument("tensor devices must match");
    }
}

template <typename Operation>
tensor elementwise_binary(const tensor& lhs, const tensor& rhs,
                          Operation operation) {
    require_same_shape(lhs, rhs);
    require_same_device(lhs, rhs);

    std::vector<float> result(lhs.numel());
    std::ranges::transform(lhs.data(), rhs.data(), result.begin(), operation);
    return tensor{std::vector<index_t>(lhs.shape().begin(), lhs.shape().end()),
                  std::move(result), lhs.placement()};
}

} // namespace

tensor::tensor(std::vector<index_t> shape, std::vector<float> values,
               device placement)
    : shape_(std::move(shape)), values_(std::move(values)),
      placement_(placement) {
    const std::size_t expected = checked_numel(shape_);
    if (values_.size() != expected) {
        throw std::invalid_argument("tensor data size does not match shape");
    }
}

std::span<const index_t> tensor::shape() const noexcept { return shape_; }

std::size_t tensor::rank() const noexcept { return shape_.size(); }

std::size_t tensor::numel() const noexcept { return values_.size(); }

device tensor::placement() const noexcept { return placement_; }

std::span<const float> tensor::data() const noexcept { return values_; }

tensor full(std::vector<index_t> shape, float value, device placement) {
    const std::size_t count = checked_numel(shape);
    return tensor{std::move(shape), std::vector<float>(count, value),
                  placement};
}

tensor zeros(std::vector<index_t> shape, device placement) {
    return full(std::move(shape), 0.0F, placement);
}

tensor add(const tensor& lhs, const tensor& rhs) {
    return elementwise_binary(lhs, rhs, std::plus<>{});
}

tensor subtract(const tensor& lhs, const tensor& rhs) {
    return elementwise_binary(lhs, rhs, std::minus<>{});
}

tensor multiply(const tensor& lhs, const tensor& rhs) {
    return elementwise_binary(lhs, rhs, std::multiplies<>{});
}

tensor operator+(const tensor& lhs, const tensor& rhs) { return add(lhs, rhs); }

tensor operator-(const tensor& lhs, const tensor& rhs) {
    return subtract(lhs, rhs);
}

tensor operator*(const tensor& lhs, const tensor& rhs) {
    return multiply(lhs, rhs);
}

} // namespace porch
