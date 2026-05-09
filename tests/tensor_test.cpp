#include "porch/backend.hpp"
#include "porch/cuda_jit.hpp"
#include "porch/tensor.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

    void construction_tracks_shape_and_device() {
        const porch::tensor values{{2, 3},
                                   {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                   porch::device{porch::device_kind::gpu, 1}};

        assert(values.rank() == 2);
        assert(values.numel() == 6);
        assert((std::vector<porch::index_t>{values.shape().begin(),
                                            values.shape().end()} ==
                std::vector<porch::index_t>{2, 3}));
        assert(values.placement().is_gpu());
        assert(values.placement().ordinal() == 1);
    }

    void tensor_is_small_handle() {
        assert(sizeof(porch::tensor) <= 2 * sizeof(void*));
    }

    void construction_builds_contiguous_layout() {
        const porch::tensor values{{2, 3, 4},
                                   std::vector<porch::float32_t>(24, 1.0F)};

        assert(values.rank() == 3);
        assert(values.numel() == 24);
        assert(values.is_contiguous());
        assert(values.storage_offset() == 0);
        assert((std::vector<porch::index_t>{values.strides().begin(),
                                            values.strides().end()} ==
                std::vector<porch::index_t>{12, 4, 1}));
        assert(&values.layout() != nullptr);
    }

    void bracket_slice_selects_contiguous_copy() {
        const porch::tensor values{{6}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}};

        const porch::tensor sliced = values[{porch::slice{1, 6, 2}}];

        assert((std::vector<porch::index_t>{sliced.shape().begin(),
                                            sliced.shape().end()} ==
                std::vector<porch::index_t>{3}));
        assert(sliced.is_contiguous());
        assert((std::vector<porch::float32_t>{sliced.data().begin(),
                                              sliced.data().end()} ==
                std::vector<porch::float32_t>{1.0F, 3.0F, 5.0F}));
    }

    void bracket_slice_returns_lazy_expression() {
        const porch::tensor values{{6}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}};

        auto expression = values[{porch::slice{1, 6, 2}}];
        static_assert(std::is_same_v<decltype(expression), porch::tensor_expr>);

        const porch::tensor result = expression * 2.0F;

        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{2.0F, 6.0F, 10.0F}));
    }

    void bracket_slice_handles_multiple_dimensions() {
        const porch::tensor values{{3, 4},
                                   {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                                    7.0F, 8.0F, 9.0F, 10.0F, 11.0F}};

        const porch::tensor sliced = values[{1, porch::slice{1, 4, 2}}];

        assert((std::vector<porch::index_t>{sliced.shape().begin(),
                                            sliced.shape().end()} ==
                std::vector<porch::index_t>{2}));
        assert((std::vector<porch::float32_t>{sliced.data().begin(),
                                              sliced.data().end()} ==
                std::vector<porch::float32_t>{5.0F, 7.0F}));
    }

    void zeros_fills_gpu_tensor() {
        const porch::tensor values = porch::zeros({2, 2});

        assert(values.numel() == 4);
        assert(values.placement().is_gpu());
        assert(!values.device_data().empty());
        for (const porch::float32_t value : values.data()) {
            assert(value == 0.0F);
        }
    }

    void add_uses_cuda_backend() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        const porch::tensor result = lhs + rhs;
        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{11.0F, 22.0F, 33.0F, 44.0F}));
        assert(!result.device_data().empty());
    }

    void subtract_uses_cuda_backend() {
        const porch::tensor lhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};
        const porch::tensor rhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};

        const porch::tensor result = lhs - rhs;
        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{9.0F, 18.0F, 27.0F, 36.0F}));
        assert(!result.device_data().empty());
    }

    void multiply_uses_cuda_backend() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        const porch::tensor result = lhs * rhs;
        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{10.0F, 40.0F, 90.0F, 160.0F}));
        assert(!result.device_data().empty());
    }

    void fused_expression_materializes_once() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        const porch::tensor result = lhs + rhs * 2.0F - 1.0F;

        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{20.0F, 41.0F, 62.0F, 83.0F}));
        assert(!result.device_data().empty());
    }

    void multiline_expression_stays_lazy_with_auto() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        auto expression = lhs + rhs;
        expression = expression * 2.0F - 1.0F;
        const porch::tensor result = expression;

        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{21.0F, 43.0F, 65.0F, 87.0F}));
        assert(!result.device_data().empty());
    }

    void scalar_elementwise_ops_use_cuda_backend() {
        const porch::tensor values{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};

        const porch::tensor result = 10.0F - values * 2.0F;

        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{8.0F, 6.0F, 4.0F, 2.0F}));
        assert(!result.device_data().empty());
    }

    void matmul_uses_cuda_backend() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{{3, 2},
                                {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}};

        const porch::tensor result = porch::matmul(lhs, rhs);

        assert((std::vector<porch::index_t>{result.shape().begin(),
                                            result.shape().end()} ==
                std::vector<porch::index_t>{2, 2}));
        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{58.0F, 64.0F, 139.0F, 154.0F}));
        assert(!result.device_data().empty());
    }

    void matmul_returns_lazy_expression() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{{3, 2},
                                {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}};

        auto expression = porch::matmul(lhs, rhs);
        static_assert(std::is_same_v<decltype(expression), porch::tensor_expr>);

        const porch::tensor result = expression;

        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{58.0F, 64.0F, 139.0F, 154.0F}));
    }

    void matmul_composes_with_elementwise_ir() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{{3, 2},
                                {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}};

        auto expression = porch::matmul(lhs, rhs) * 2.0F - 1.0F;
        const porch::tensor result = expression;

        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{115.0F, 127.0F, 277.0F, 307.0F}));
    }

    void matmul_accepts_expression_operands() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{{3, 2},
                                {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}};

        auto expression = porch::matmul(lhs + lhs, rhs);
        const porch::tensor result = expression;

        assert((std::vector<porch::float32_t>{result.data().begin(),
                                              result.data().end()} ==
                std::vector<porch::float32_t>{116.0F, 128.0F, 278.0F, 308.0F}));
    }

    void matmul_rejects_invalid_shapes() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};

        bool threw = false;
        try {
            const porch::tensor result = porch::matmul(lhs, rhs);
            (void)result;
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    void rejects_invalid_shapes() {
        bool threw = false;
        try {
            (void)porch::tensor{{2, 2}, {1.0F, 2.0F, 3.0F}};
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    void gpu_devices_select_cuda_jit_backend() {
        const porch::backend selected =
            porch::backend_for(porch::device{porch::device_kind::gpu});

        assert(selected.is_cuda_jit());
        assert(porch::backend_name(selected) == "cuda-jit");
    }

    void cuda_jit_availability_matches_backend_availability() {
        assert(porch::cuda_jit::is_available() ==
               porch::is_backend_available(
                   porch::backend{porch::backend_kind::cuda_jit}));
    }

    void cuda_jit_compiles_or_reports_missing_nvrtc() {
        constexpr std::string_view source = R"cuda(
            extern "C" __global__ void add_kernel(const float* lhs,
                                                  const float* rhs,
                                                  float* out) {
                const int index = threadIdx.x;
                out[index] = lhs[index] + rhs[index];
            }
        )cuda";

        bool threw = false;
        try {
            const std::string ptx = porch::cuda_jit::compile_to_ptx(source);
            assert(ptx.find(".version") != std::string::npos);
            assert(ptx.find("add_kernel") != std::string::npos);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        (void)threw;
    }

    void device_buffer_round_trips_values() {
        const std::vector<porch::float32_t> source{1.0F, 2.0F, 3.0F, 4.0F};
        porch::cuda_jit::device_buffer buffer =
            porch::cuda_jit::make_device_buffer(source);

        std::vector<porch::float32_t> target(source.size());
        porch::cuda_jit::copy_to_host(buffer, target);

        assert(target == source);
    }

    struct test_case {
        std::string_view name;
        void (*run)();
        bool requires_cuda;
    };

    constexpr test_case tests[] = {
        {"construction_tracks_shape_and_device",
         construction_tracks_shape_and_device, true},
        {"tensor_is_small_handle", tensor_is_small_handle, false},
        {"construction_builds_contiguous_layout",
         construction_builds_contiguous_layout, true},
        {"bracket_slice_selects_contiguous_copy",
         bracket_slice_selects_contiguous_copy, true},
        {"bracket_slice_returns_lazy_expression",
         bracket_slice_returns_lazy_expression, true},
        {"bracket_slice_handles_multiple_dimensions",
         bracket_slice_handles_multiple_dimensions, true},
        {"zeros_fills_gpu_tensor", zeros_fills_gpu_tensor, true},
        {"add_uses_cuda_backend", add_uses_cuda_backend, true},
        {"subtract_uses_cuda_backend", subtract_uses_cuda_backend, true},
        {"multiply_uses_cuda_backend", multiply_uses_cuda_backend, true},
        {"fused_expression_materializes_once",
         fused_expression_materializes_once, true},
        {"multiline_expression_stays_lazy_with_auto",
         multiline_expression_stays_lazy_with_auto, true},
        {"scalar_elementwise_ops_use_cuda_backend",
         scalar_elementwise_ops_use_cuda_backend, true},
        {"matmul_uses_cuda_backend", matmul_uses_cuda_backend, true},
        {"matmul_returns_lazy_expression", matmul_returns_lazy_expression,
         true},
        {"matmul_composes_with_elementwise_ir",
         matmul_composes_with_elementwise_ir, true},
        {"matmul_accepts_expression_operands",
         matmul_accepts_expression_operands, true},
        {"matmul_rejects_invalid_shapes", matmul_rejects_invalid_shapes, true},
        {"rejects_invalid_shapes", rejects_invalid_shapes, false},
        {"gpu_devices_select_cuda_jit_backend",
         gpu_devices_select_cuda_jit_backend, false},
        {"cuda_jit_availability_matches_backend_availability",
         cuda_jit_availability_matches_backend_availability, false},
        {"cuda_jit_compiles_or_reports_missing_nvrtc",
         cuda_jit_compiles_or_reports_missing_nvrtc, false},
        {"device_buffer_round_trips_values", device_buffer_round_trips_values,
         true},
    };

    bool run_test(const test_case& test) {
        if (test.requires_cuda && !porch::cuda_jit::is_available()) {
            std::cout << "skipped: CUDA JIT execution is unavailable\n";
            return true;
        }

        test.run();
        return true;
    }

} // namespace

int main(int argc, char** argv) {
    if (argc == 2) {
        const std::string_view requested = argv[1];
        for (const test_case& test : tests) {
            if (test.name == requested) {
                return run_test(test) ? 0 : 1;
            }
        }

        std::cerr << "unknown test: " << requested << '\n';
        return 1;
    }

    for (const test_case& test : tests) {
        (void)run_test(test);
    }
}
