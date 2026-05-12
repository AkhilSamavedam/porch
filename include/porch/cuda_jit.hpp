#pragma once

#include "porch/types.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace porch::cuda_jit {

    struct device_buffer_state;

    class device_buffer {
      public:
        device_buffer() = default;
        explicit device_buffer(size_t bytes);

        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_t size_bytes() const noexcept;

      private:
        friend void copy_to_device(
            device_buffer& buffer, std::span<const float32_t> source
        );
        friend void copy_to_host(
            const device_buffer& buffer, std::span<float32_t> target
        );
        friend void launch_elementwise_binary(
            std::string_view ptx, std::string_view function,
            const device_buffer& lhs, const device_buffer& rhs,
            device_buffer& out, size_t count
        );
        friend void launch_fused_elementwise(
            std::string_view ptx, std::string_view function,
            const std::vector<const device_buffer*>& inputs, device_buffer& out,
            size_t count
        );
        friend void launch_fused_reduction(
            std::string_view ptx, std::string_view function,
            const std::vector<const device_buffer*>& inputs, device_buffer& out,
            size_t count
        );

        std::shared_ptr<device_buffer_state> state_;
    };

    class compilation_error : public std::runtime_error {
      public:
        explicit compilation_error(std::string log);

        [[nodiscard]] const std::string& log() const noexcept;

      private:
        std::string log_;
    };

    [[nodiscard]] bool is_available() noexcept;

    [[nodiscard]] std::string compile_to_ptx(
        std::string_view source, const std::vector<std::string>& options = {}
    );

    [[nodiscard]] device_buffer make_device_buffer(
        std::span<const float32_t> source
    );
    void copy_to_device(
        device_buffer& buffer, std::span<const float32_t> source
    );
    void copy_to_host(const device_buffer& buffer, std::span<float32_t> target);
    void synchronize();

    void launch_elementwise_binary(
        std::string_view ptx, std::string_view function,
        std::span<const float32_t> lhs, std::span<const float32_t> rhs,
        std::span<float32_t> out
    );

    void launch_elementwise_binary(
        std::string_view ptx, std::string_view function,
        const device_buffer& lhs, const device_buffer& rhs, device_buffer& out,
        size_t count
    );

    void launch_fused_elementwise(
        std::string_view ptx, std::string_view function,
        const std::vector<const device_buffer*>& inputs, device_buffer& out,
        size_t count
    );

    void launch_fused_reduction(
        std::string_view ptx, std::string_view function,
        const std::vector<const device_buffer*>& inputs, device_buffer& out,
        size_t count
    );

} // namespace porch::cuda_jit
