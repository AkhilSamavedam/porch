# porch

`porch` is an early C++ tensor library scaffold intended to grow toward a
Torch-like programming model with ordinary C++ compilers and pluggable CPU/GPU
backends.

CUDA support is modeled as a runtime JIT backend. The public library is built as
ordinary C++ and does not enable CMake's CUDA language, so users of the library
do not need `nvcc` in their build. CUDA driver discovery happens dynamically at
runtime through the `cuda-jit` backend.

## Build and test

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The current implementation provides a small `porch::tensor` API and CPU-backed
operations so the project has a testable foundation before adding accelerator
backends and compilation infrastructure.

Disable the CUDA JIT backend boundary with:

```sh
cmake -S . -B build -DPORCH_ENABLE_CUDA_JIT=OFF
```
