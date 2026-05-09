#include "porch/backend.hpp"
#include "porch/cuda_jit.hpp"

namespace porch {

    backend backend_for(device placement) noexcept {
        (void)placement;
        return backend{backend_kind::cuda_jit};
    }

    bool is_backend_available(backend target) noexcept {
        switch (target.kind()) {
        case backend_kind::cuda_jit:
            return cuda_jit::is_available();
        }
        return false;
    }

    std::string_view backend_name(backend target) noexcept {
        switch (target.kind()) {
        case backend_kind::cuda_jit:
            return "cuda-jit";
        }
        return "unknown";
    }

} // namespace porch
