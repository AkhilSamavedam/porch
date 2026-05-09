add_executable(porch_tensor_test tensor_test.cpp)
target_link_libraries(porch_tensor_test PRIVATE porch::porch)

set(
    PORCH_TENSOR_TESTS
    construction_tracks_shape_and_device
    tensor_is_small_handle
    construction_builds_contiguous_layout
    bracket_slice_selects_contiguous_copy
    bracket_slice_returns_lazy_expression
    bracket_slice_handles_multiple_dimensions
    zeros_fills_gpu_tensor
    add_uses_cuda_backend
    subtract_uses_cuda_backend
    multiply_uses_cuda_backend
    fused_expression_materializes_once
    multiline_expression_stays_lazy_with_auto
    scalar_elementwise_ops_use_cuda_backend
    matmul_uses_cuda_backend
    matmul_returns_lazy_expression
    matmul_composes_with_elementwise_ir
    matmul_accepts_expression_operands
    matmul_rejects_invalid_shapes
    rejects_invalid_shapes
    gpu_devices_select_cuda_jit_backend
    cuda_jit_availability_matches_backend_availability
    cuda_jit_compiles_or_reports_missing_nvrtc
    device_buffer_round_trips_values
)

foreach(test_name IN LISTS PORCH_TENSOR_TESTS)
    add_test(NAME porch.tensor.${test_name} COMMAND porch_tensor_test ${test_name})
endforeach()
