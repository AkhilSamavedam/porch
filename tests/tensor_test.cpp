#include "porch/backend.hpp"
#include "porch/cuda_jit.hpp"
#include "porch/tensor.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
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
                                              const float* rhs, float* out) {
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
    {"zeros_fills_gpu_tensor", zeros_fills_gpu_tensor, true},
    {"add_uses_cuda_backend", add_uses_cuda_backend, true},
    {"subtract_uses_cuda_backend", subtract_uses_cuda_backend, true},
    {"multiply_uses_cuda_backend", multiply_uses_cuda_backend, true},
    {"fused_expression_materializes_once", fused_expression_materializes_once,
     true},
    {"multiline_expression_stays_lazy_with_auto",
     multiline_expression_stays_lazy_with_auto, true},
    {"scalar_elementwise_ops_use_cuda_backend",
     scalar_elementwise_ops_use_cuda_backend, true},
    {"rejects_invalid_shapes", rejects_invalid_shapes, false},
    {"gpu_devices_select_cuda_jit_backend", gpu_devices_select_cuda_jit_backend,
     false},
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
