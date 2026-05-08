#include "porch/backend.hpp"
#include "porch/tensor.hpp"

#include <cassert>
#include <stdexcept>
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

void zeros_fills_cpu_tensor() {
    const porch::tensor values = porch::zeros({2, 2});

    assert(values.numel() == 4);
    for (const float value : values.data()) {
        assert(value == 0.0F);
    }
}

void add_sums_same_shape_tensors() {
    const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
    const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

    const porch::tensor result = lhs + rhs;

    assert((std::vector<float>{result.data().begin(), result.data().end()} ==
            std::vector<float>{11.0F, 22.0F, 33.0F, 44.0F}));
}

void subtract_differences_same_shape_tensors() {
    const porch::tensor lhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};
    const porch::tensor rhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};

    const porch::tensor result = lhs - rhs;

    assert((std::vector<float>{result.data().begin(), result.data().end()} ==
            std::vector<float>{9.0F, 18.0F, 27.0F, 36.0F}));
}

void multiply_products_same_shape_tensors() {
    const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
    const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

    const porch::tensor result = lhs * rhs;

    assert((std::vector<float>{result.data().begin(), result.data().end()} ==
            std::vector<float>{10.0F, 40.0F, 90.0F, 160.0F}));
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

void cpu_backend_is_always_available() {
    assert(
        porch::is_backend_available(porch::backend{porch::backend_kind::cpu}));
}

} // namespace

int main() {
    construction_tracks_shape_and_device();
    zeros_fills_cpu_tensor();
    add_sums_same_shape_tensors();
    subtract_differences_same_shape_tensors();
    multiply_products_same_shape_tensors();
    rejects_invalid_shapes();
    gpu_devices_select_cuda_jit_backend();
    cpu_backend_is_always_available();
}
