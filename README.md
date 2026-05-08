# porch

`porch` is an early C++ tensor library scaffold intended to grow toward a
Torch-like programming model with ordinary C++ compilers and a CUDA JIT backend.

CUDA support is modeled as a runtime JIT backend. The public library is built as
ordinary C++ and does not enable CMake's CUDA language, so users of the library
do not need `nvcc` in their build. CUDA driver discovery happens dynamically at
runtime through the `cuda-jit` backend. There is no CPU tensor backend.

## Build and test

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The current implementation provides a small `porch::tensor` API, runtime NVRTC
PTX compilation, and CUDA-only operator dispatch stubs. Tensor arithmetic no
longer falls back to host-side computation; the next layer is CUDA driver
context, memory, module loading, and kernel launch support.
