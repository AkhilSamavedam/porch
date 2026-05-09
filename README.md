# porch

`porch` is an early C++ tensor library scaffold intended to grow toward a
Torch-like programming model with ordinary C++ compilers and a CUDA JIT backend.

CUDA support is modeled as a runtime JIT backend. The public library is built as
ordinary C++ and does not enable CMake's CUDA language, so users of the library
do not need `nvcc` in their build. CUDA driver discovery happens dynamically at
runtime through the `cuda-jit` backend. There is no CPU tensor backend.

## Execution model

The C++ library itself is compiled ahead of time with a normal C++20 compiler.
Tensor programs are recorded at runtime as a small IR graph. Operations such as
elementwise arithmetic, slicing, and `porch::matmul` append graph nodes instead
of immediately launching CUDA work.

```cpp
porch::tensor x = a + b;
porch::tensor y = x * 2.0F - 1.0F;
```

In this example, assigning `a + b` to `x` does not force a kernel launch. `x`
stores the pending graph, and the later expression can inline that graph into
`y`.

At materialization time, porch lowers the connected graph to generated CUDA C++
source, compiles it with NVRTC, caches the resulting PTX, and launches it through
the CUDA driver API. The current lowering emits a single fused kernel for
supported connected graphs, including elementwise ops, strided slicing, and
simple rank-2 matmul expression nodes.

GPU work is launched on a CUDA stream and kernel launch does not automatically
block the CPU. The explicit boundaries are:

- `realize()`: compile and launch pending graph work, keeping the result on GPU.
- `synchronize()`: wait for the CUDA stream.
- `cpu()`: copy values to host memory; this implies materialization and stream
  synchronization.
- `data()`: returns a host span for convenience; this also implies
  materialization and stream synchronization.
- `device_data()`: materializes GPU storage and returns the device buffer handle
  without copying to host.

The IR is not built at C++ compile time. It is recorded dynamically while the
user program runs, which lets porch preserve laziness across ordinary statement
boundaries without requiring a special compiler.

## Threading model

`porch::tensor` is a cheap shared handle and can be copied across threads. Lazy
tensor materialization and host-cache updates are synchronized internally, so two
threads touching the same pending tensor will not corrupt its state.

The CUDA backend owns one process-wide CUDA context and uses a thread-local
current stream, similar in spirit to LibTorch's per-thread current stream model.
Each thread can enqueue work independently. Device buffers track the stream that
last produced them; host copies and cross-stream dependencies synchronize that
producer stream before reading.
Internal streams have shared lifetime ownership. A device buffer keeps its
producer stream alive until any dependent async free or cross-stream dependency
has been made safe, so buffers can outlive the thread that produced them without
holding a dangling stream handle.

Device memory uses CUDA's stream-ordered allocation APIs
`cuMemAllocAsync`/`cuMemFreeAsync`. Host transfers use the async copy APIs and
then synchronize only where host visibility requires it. Direct legacy
`cuMemAlloc`/`cuMemFree` allocation is not used.

Library invariants and tensor lifetimes are protected for concurrent reads and
lazy materialization. Concurrent logical mutation of the same tensor data is not
part of the API yet and should still be treated as user-synchronized.

## Build and test

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The current implementation provides a small `porch::tensor` API, CUDA-only GPU
storage, runtime NVRTC PTX compilation, CUDA driver context/memory/module/kernel
launch support, lazy tensor graph recording, fused elementwise codegen, GPU
slicing, and simple matmul lowering.
