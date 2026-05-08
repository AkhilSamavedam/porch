#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace porch {

using index_t = std::int64_t;

enum class device_kind {
    cpu,
    gpu,
};

class device {
  public:
    constexpr explicit device(device_kind kind = device_kind::cpu,
                              int ordinal = 0) noexcept
        : kind_(kind), ordinal_(ordinal) {}

    [[nodiscard]] constexpr device_kind kind() const noexcept { return kind_; }
    [[nodiscard]] constexpr int ordinal() const noexcept { return ordinal_; }
    [[nodiscard]] constexpr bool is_cpu() const noexcept {
        return kind_ == device_kind::cpu;
    }
    [[nodiscard]] constexpr bool is_gpu() const noexcept {
        return kind_ == device_kind::gpu;
    }

    friend constexpr bool operator==(const device&,
                                     const device&) noexcept = default;

  private:
    device_kind kind_;
    int ordinal_;
};

class tensor {
  public:
    tensor(std::vector<index_t> shape, std::vector<float> values,
           device placement = device{});

    [[nodiscard]] std::span<const index_t> shape() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t numel() const noexcept;
    [[nodiscard]] device placement() const noexcept;
    [[nodiscard]] std::span<const float> data() const noexcept;

  private:
    std::vector<index_t> shape_;
    std::vector<float> values_;
    device placement_;
};

[[nodiscard]] tensor full(std::vector<index_t> shape, float value,
                          device placement = device{});
[[nodiscard]] tensor zeros(std::vector<index_t> shape,
                           device placement = device{});
[[nodiscard]] tensor add(const tensor& lhs, const tensor& rhs);
[[nodiscard]] tensor subtract(const tensor& lhs, const tensor& rhs);
[[nodiscard]] tensor multiply(const tensor& lhs, const tensor& rhs);

[[nodiscard]] tensor operator+(const tensor& lhs, const tensor& rhs);
[[nodiscard]] tensor operator-(const tensor& lhs, const tensor& rhs);
[[nodiscard]] tensor operator*(const tensor& lhs, const tensor& rhs);

} // namespace porch
