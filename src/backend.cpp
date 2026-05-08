#include "porch/backend.hpp"

#if defined(PORCH_ENABLE_CUDA_JIT)
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace porch {
namespace {

bool cuda_driver_available() noexcept {
#if defined(PORCH_ENABLE_CUDA_JIT)
#if defined(_WIN32)
    HMODULE handle = LoadLibraryA("nvcuda.dll");
    if (handle == nullptr) {
        return false;
    }
    FreeLibrary(handle);
    return true;
#else
    void* handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
        handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
    }
    if (handle == nullptr) {
        return false;
    }
    dlclose(handle);
    return true;
#endif
#else
    return false;
#endif
}

} // namespace

backend backend_for(device placement) noexcept {
    return placement.is_gpu() ? backend{backend_kind::cuda_jit}
                              : backend{backend_kind::cpu};
}

bool is_backend_available(backend target) noexcept {
    switch (target.kind()) {
    case backend_kind::cpu:
        return true;
    case backend_kind::cuda_jit:
        return cuda_driver_available();
    }
    return false;
}

std::string_view backend_name(backend target) noexcept {
    switch (target.kind()) {
    case backend_kind::cpu:
        return "cpu";
    case backend_kind::cuda_jit:
        return "cuda-jit";
    }
    return "unknown";
}

} // namespace porch
