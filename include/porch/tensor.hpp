#pragma once

#include "porch/cuda_jit.hpp"
#include "porch/types.hpp"

#include <initializer_list>
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

    struct all_t {};

    inline constexpr all_t all{};

    struct slice {
        index_t start;
        index_t stop;
        index_t step = 1;
    };

    class tensor_index {
      public:
        tensor_index(index_t index);
        tensor_index(slice range);
        tensor_index(all_t marker);

      private:
        friend class tensor;

        enum class kind {
            index,
            slice,
            all,
        };

        kind kind_;
        index_t index_ = 0;
        slice slice_{0, 0, 1};
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
        [[nodiscard]] const cuda_jit::device_buffer& device_data() const;
        [[nodiscard]] tensor_expr operator[](
            std::initializer_list<tensor_index> indices) const;
        [[nodiscard]] tensor_expr operator[](tensor_index index) const;

      private:
        friend tensor materialize(const tensor_expr& expression);
        friend class tensor_expr;
        friend tensor make_materialized_tensor(
            std::vector<index_t> shape, std::vector<float32_t> values,
            cuda_jit::device_buffer device_values, bool host_current,
            device placement);

        struct state;

        tensor(std::vector<index_t> shape, std::vector<float32_t> values,
               cuda_jit::device_buffer device_values, bool host_current,
               device placement);
        tensor(tensor_layout layout, std::vector<float32_t> values,
               cuda_jit::device_buffer device_values, bool host_current,
               device placement);

        void ensure_materialized() const;

        std::shared_ptr<state> state_;
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
        friend tensor_expr matmul(tensor_expr lhs, tensor_expr rhs);
        friend class tensor;
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
    [[nodiscard]] tensor_expr matmul(tensor_expr lhs, tensor_expr rhs);

    [[nodiscard]] tensor materialize(const tensor_expr& expression);

    [[nodiscard]] tensor_expr operator+(tensor_expr lhs, tensor_expr rhs);
    [[nodiscard]] tensor_expr operator-(tensor_expr lhs, tensor_expr rhs);
    [[nodiscard]] tensor_expr operator*(tensor_expr lhs, tensor_expr rhs);

} // namespace porch
