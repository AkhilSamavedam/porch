#pragma once

#include "porch/tensor.hpp"

#include <string_view>

namespace porch {

enum class backend_kind {
    cuda_jit,
};

class backend {
  public:
    constexpr explicit backend(
        backend_kind kind = backend_kind::cuda_jit) noexcept
        : kind_(kind) {}

    [[nodiscard]] constexpr backend_kind kind() const noexcept { return kind_; }
    [[nodiscard]] constexpr bool is_cuda_jit() const noexcept {
        return kind_ == backend_kind::cuda_jit;
    }

    friend constexpr bool operator==(const backend&,
                                     const backend&) noexcept = default;

  private:
    backend_kind kind_;
};

[[nodiscard]] backend backend_for(device placement) noexcept;
[[nodiscard]] bool is_backend_available(backend target) noexcept;
[[nodiscard]] std::string_view backend_name(backend target) noexcept;

} // namespace porch
