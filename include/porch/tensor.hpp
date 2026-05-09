#pragma once

#include "porch/cuda_jit.hpp"
#include "porch/types.hpp"

#include <memory>
#include <span>
#include <vector>

namespace porch {

    class tensor_expr;

    enum class device_kind {
        gpu,
    };

    class device {
      public:
        constexpr explicit device(device_kind kind = device_kind::gpu,
                                  int ordinal = 0) noexcept
            : kind_(kind), ordinal_(ordinal) {}

        [[nodiscard]] constexpr device_kind kind() const noexcept {
            return kind_;
        }
        [[nodiscard]] constexpr int ordinal() const noexcept {
            return ordinal_;
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

    class tensor_layout {
      public:
        explicit tensor_layout(std::vector<index_t> shape);
        tensor_layout(std::vector<index_t> shape, std::vector<index_t> strides,
                      index_t storage_offset = 0);

        [[nodiscard]] std::span<const index_t> shape() const noexcept;
        [[nodiscard]] std::span<const index_t> strides() const noexcept;
        [[nodiscard]] index_t storage_offset() const noexcept;
        [[nodiscard]] size_t rank() const noexcept;
        [[nodiscard]] size_t numel() const noexcept;
        [[nodiscard]] bool is_contiguous() const noexcept;

      private:
        std::vector<index_t> shape_;
        std::vector<index_t> strides_;
        index_t storage_offset_ = 0;
    };

    class tensor {
      public:
        tensor(std::vector<index_t> shape, std::vector<float32_t> values,
               device placement = device{});
        tensor(std::vector<index_t> shape, std::vector<float32_t> values,
               cuda_jit::device_buffer device_values,
               device placement = device{});

        [[nodiscard]] std::span<const index_t> shape() const noexcept;
        [[nodiscard]] std::span<const index_t> strides() const noexcept;
        [[nodiscard]] const tensor_layout& layout() const noexcept;
        [[nodiscard]] index_t storage_offset() const noexcept;
        [[nodiscard]] bool is_contiguous() const noexcept;
        [[nodiscard]] size_t rank() const noexcept;
        [[nodiscard]] size_t numel() const noexcept;
        [[nodiscard]] device placement() const noexcept;
        [[nodiscard]] std::span<const float32_t> data() const;
        [[nodiscard]] const cuda_jit::device_buffer&
        device_data() const noexcept;

      private:
        friend tensor materialize(const tensor_expr& expression);

        tensor(std::vector<index_t> shape, std::vector<float32_t> values,
               cuda_jit::device_buffer device_values, bool host_current,
               device placement);
        tensor(tensor_layout layout, std::vector<float32_t> values,
               cuda_jit::device_buffer device_values, bool host_current,
               device placement);

        tensor_layout layout_;
        mutable std::vector<float32_t> values_;
        cuda_jit::device_buffer device_values_;
        mutable bool host_current_ = true;
        device placement_;
    };

    class tensor_expr {
      public:
        struct node;

        tensor_expr(const tensor& value);
        tensor_expr(float32_t value);

        [[nodiscard]] tensor eval() const;

        operator tensor() const;

      private:
        explicit tensor_expr(std::shared_ptr<const node> root);

        friend tensor_expr operator+(tensor_expr lhs, tensor_expr rhs);
        friend tensor_expr operator-(tensor_expr lhs, tensor_expr rhs);
        friend tensor_expr operator*(tensor_expr lhs, tensor_expr rhs);
        friend tensor materialize(const tensor_expr& expression);

        std::shared_ptr<const node> root_;
    };

    [[nodiscard]] tensor full(std::vector<index_t> shape, float32_t value,
                              device placement = device{});
    [[nodiscard]] tensor zeros(std::vector<index_t> shape,
                               device placement = device{});
    [[nodiscard]] tensor add(const tensor& lhs, const tensor& rhs);
    [[nodiscard]] tensor subtract(const tensor& lhs, const tensor& rhs);
    [[nodiscard]] tensor multiply(const tensor& lhs, const tensor& rhs);

    [[nodiscard]] tensor materialize(const tensor_expr& expression);

    [[nodiscard]] tensor_expr operator+(tensor_expr lhs, tensor_expr rhs);
    [[nodiscard]] tensor_expr operator-(tensor_expr lhs, tensor_expr rhs);
    [[nodiscard]] tensor_expr operator*(tensor_expr lhs, tensor_expr rhs);

} // namespace porch
