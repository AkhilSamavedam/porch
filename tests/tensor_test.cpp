#include "porch/backend.hpp"
#include "porch/cuda_jit.hpp"
#include "porch/ops.hpp"
#include "porch/tensor.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

    void construction_tracks_shape_and_device() {
        const porch::tensor values{
            {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, porch::device{1}
        };

        assert(values.rank() == 2);
        assert(values.numel() == 6);
        assert(
            (std::vector<porch::index_t>{
                 values.shape().begin(), values.shape().end()
             } == std::vector<porch::index_t>{2, 3})
        );
        assert(values.placement().is_cuda());
        assert(values.placement().ordinal() == 1);
    }

    void tensor_is_small_handle() {
        assert(sizeof(porch::tensor) <= 2 * sizeof(void*));
    }

    void construction_builds_contiguous_layout() {
        const porch::tensor values{
            {2, 3, 4}, std::vector<porch::float32_t>(24, 1.0F)
        };

        assert(values.rank() == 3);
        assert(values.numel() == 24);
        assert(values.is_contiguous());
        assert(values.storage_offset() == 0);
        assert(
            (std::vector<porch::index_t>{
                 values.strides().begin(), values.strides().end()
             } == std::vector<porch::index_t>{12, 4, 1})
        );
        assert(&values.layout() != nullptr);
    }

    void bracket_slice_selects_contiguous_copy() {
        const porch::tensor values{{6}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}};

        const porch::tensor sliced = values[{porch::slice{1, 6, 2}}];

        assert(
            (std::vector<porch::index_t>{
                 sliced.shape().begin(), sliced.shape().end()
             } == std::vector<porch::index_t>{3})
        );
        assert(sliced.is_contiguous());
        assert(
            (std::vector<porch::float32_t>{
                 sliced.data().begin(), sliced.data().end()
             } == std::vector<porch::float32_t>{1.0F, 3.0F, 5.0F})
        );
    }

    void bracket_slice_returns_lazy_expression() {
        const porch::tensor values{{6}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}};

        auto expression = values[{porch::slice{1, 6, 2}}];
        static_assert(std::is_same_v<decltype(expression), porch::tensor_expr>);

        const porch::tensor result = expression * 2.0F;

        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{2.0F, 6.0F, 10.0F})
        );
    }

    void bracket_slice_handles_multiple_dimensions() {
        const porch::tensor values{
            {3, 4},
            {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
             11.0F}
        };

        const porch::tensor sliced = values[{1, porch::slice{1, 4, 2}}];

        assert(
            (std::vector<porch::index_t>{
                 sliced.shape().begin(), sliced.shape().end()
             } == std::vector<porch::index_t>{2})
        );
        assert(
            (std::vector<porch::float32_t>{
                 sliced.data().begin(), sliced.data().end()
             } == std::vector<porch::float32_t>{5.0F, 7.0F})
        );
    }

    void zeros_fills_gpu_tensor() {
        const porch::tensor values = porch::zeros({2, 2});

        assert(values.numel() == 4);
        assert(values.placement().is_cuda());
        assert(!values.device_data().empty());
        for (const porch::float32_t value : values.data()) {
            assert(value == 0.0F);
        }
    }

    void add_uses_cuda_backend() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        const porch::tensor result = lhs + rhs;
        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{11.0F, 22.0F, 33.0F, 44.0F})
        );
        assert(!result.device_data().empty());
    }

    void subtract_uses_cuda_backend() {
        const porch::tensor lhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};
        const porch::tensor rhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};

        const porch::tensor result = lhs - rhs;
        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{9.0F, 18.0F, 27.0F, 36.0F})
        );
        assert(!result.device_data().empty());
    }

    void multiply_uses_cuda_backend() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        const porch::tensor result = lhs * rhs;
        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{10.0F, 40.0F, 90.0F, 160.0F})
        );
        assert(!result.device_data().empty());
    }

    void elementwise_arithmetic_matches_expected_values() {
        const porch::tensor lhs{{2, 3}, {-3.0F, -1.0F, 0.0F, 2.0F, 4.0F, 8.0F}};
        const porch::tensor rhs{{2, 3}, {6.0F, -5.0F, 3.0F, 2.0F, -4.0F, 0.5F}};

        const porch::tensor sum = lhs + rhs;
        const porch::tensor difference = lhs - rhs;
        const porch::tensor product = lhs * rhs;

        assert(
            (sum.cpu() ==
             std::vector<porch::float32_t>{3.0F, -6.0F, 3.0F, 4.0F, 0.0F, 8.5F})
        );
        assert((
            difference.cpu() ==
            std::vector<porch::float32_t>{-9.0F, 4.0F, -3.0F, 0.0F, 8.0F, 7.5F}
        ));
        assert(
            (product.cpu() == std::vector<porch::float32_t>{
                                  -18.0F, 5.0F, 0.0F, 4.0F, -16.0F, 4.0F
                              })
        );
    }

    void chained_arithmetic_matches_expected_values() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{{2, 3}, {6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F}};

        const porch::tensor result = (lhs + rhs) * (lhs - 1.0F) - rhs;

        assert(
            (result.cpu() == std::vector<porch::float32_t>{
                                 -6.0F, 2.0F, 10.0F, 18.0F, 26.0F, 34.0F
                             })
        );
    }

    void fused_expression_materializes_once() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        const porch::tensor result = lhs + rhs * 2.0F - 1.0F;

        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{20.0F, 41.0F, 62.0F, 83.0F})
        );
        assert(!result.device_data().empty());
    }

    void multiline_expression_stays_lazy_with_auto() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        auto expression = lhs + rhs;
        expression = expression * 2.0F - 1.0F;
        const porch::tensor result = expression;

        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{21.0F, 43.0F, 65.0F, 87.0F})
        );
        assert(!result.device_data().empty());
    }

    void tensor_assignment_keeps_graph_lazy() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        const porch::tensor delayed = lhs + rhs;
        assert(
            (std::vector<porch::index_t>{
                 delayed.shape().begin(), delayed.shape().end()
             } == std::vector<porch::index_t>{2, 2})
        );

        const porch::tensor result = delayed * 2.0F - 1.0F;

        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{21.0F, 43.0F, 65.0F, 87.0F})
        );
        assert(!result.device_data().empty());
    }

    void explicit_gpu_sync_and_cpu_copy() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};

        const porch::tensor result = lhs + rhs;
        result.realize();
        assert(!result.device_data().empty());
        result.synchronize();

        assert(
            (result.cpu() ==
             std::vector<porch::float32_t>{11.0F, 22.0F, 33.0F, 44.0F})
        );
    }

    void concurrent_same_lazy_tensor_materializes_once_safely() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};
        const porch::tensor result = (lhs + rhs) * 2.0F - 1.0F;

        std::exception_ptr left_error;
        std::exception_ptr right_error;
        std::vector<porch::float32_t> first;
        std::vector<porch::float32_t> second;

        std::thread left{[&] {
            try {
                first = result.cpu();
            }
            catch (...) {
                left_error = std::current_exception();
            }
        }};
        std::thread right{[&] {
            try {
                second = result.cpu();
            }
            catch (...) {
                right_error = std::current_exception();
            }
        }};

        left.join();
        right.join();
        if (left_error) std::rethrow_exception(left_error);
        if (right_error) std::rethrow_exception(right_error);

        const std::vector<porch::float32_t> expected{
            21.0F, 43.0F, 65.0F, 87.0F
        };
        assert(first == expected);
        assert(second == expected);
    }

    void concurrent_independent_lazy_tensors_use_thread_streams() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}};
        const porch::tensor left_result = lhs + rhs;
        const porch::tensor right_result = rhs - lhs;

        std::exception_ptr left_error;
        std::exception_ptr right_error;
        std::vector<porch::float32_t> left_values;
        std::vector<porch::float32_t> right_values;

        std::thread left{[&] {
            try {
                left_values = left_result.cpu();
            }
            catch (...) {
                left_error = std::current_exception();
            }
        }};
        std::thread right{[&] {
            try {
                right_values = right_result.cpu();
            }
            catch (...) {
                right_error = std::current_exception();
            }
        }};

        left.join();
        right.join();
        if (left_error) std::rethrow_exception(left_error);
        if (right_error) std::rethrow_exception(right_error);

        assert(
            (left_values ==
             std::vector<porch::float32_t>{11.0F, 22.0F, 33.0F, 44.0F})
        );
        assert(
            (right_values ==
             std::vector<porch::float32_t>{9.0F, 18.0F, 27.0F, 36.0F})
        );
    }

    void scalar_elementwise_ops_use_cuda_backend() {
        const porch::tensor values{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};

        const porch::tensor result = 10.0F - values * 2.0F;

        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{8.0F, 6.0F, 4.0F, 2.0F})
        );
        assert(!result.device_data().empty());
    }

    void scalar_arithmetic_order_matches_expected_values() {
        const porch::tensor values{
            {2, 3}, {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F}
        };

        const porch::tensor left_scalar = 10.0F - values * 3.0F;
        const porch::tensor right_scalar = (values + 5.0F) * 2.0F;

        assert((
            left_scalar.cpu() ==
            std::vector<porch::float32_t>{16.0F, 13.0F, 10.0F, 7.0F, 4.0F, 1.0F}
        ));
        assert(
            (right_scalar.cpu() == std::vector<porch::float32_t>{
                                       6.0F, 8.0F, 10.0F, 12.0F, 14.0F, 16.0F
                                   })
        );
    }

    void matmul_uses_cuda_backend() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{
            {3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}
        };

        const porch::tensor result = porch::matmul(lhs, rhs);

        assert(
            (std::vector<porch::index_t>{
                 result.shape().begin(), result.shape().end()
             } == std::vector<porch::index_t>{2, 2})
        );
        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{58.0F, 64.0F, 139.0F, 154.0F})
        );
        assert(!result.device_data().empty());
    }

    void matmul_rectangular_values_are_correct() {
        const porch::tensor lhs{{3, 2}, {1.0F, 2.0F, -1.0F, 3.0F, 4.0F, -2.0F}};
        const porch::tensor rhs{
            {2, 4}, {2.0F, 0.0F, -1.0F, 5.0F, 3.0F, -2.0F, 4.0F, 1.0F}
        };

        const porch::tensor result = porch::matmul(lhs, rhs);

        assert(
            (std::vector<porch::index_t>{
                 result.shape().begin(), result.shape().end()
             } == std::vector<porch::index_t>{3, 4})
        );
        assert(
            (result.cpu() == std::vector<porch::float32_t>{
                                 8.0F, -4.0F, 7.0F, 7.0F, 7.0F, -6.0F, 13.0F,
                                 -2.0F, 2.0F, 4.0F, -12.0F, 18.0F
                             })
        );
    }

    void matmul_returns_lazy_expression() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{
            {3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}
        };

        auto expression = porch::matmul(lhs, rhs);
        static_assert(std::is_same_v<decltype(expression), porch::tensor_expr>);

        const porch::tensor result = expression;

        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{58.0F, 64.0F, 139.0F, 154.0F})
        );
    }

    void matmul_composes_with_elementwise_ir() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{
            {3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}
        };

        auto expression = porch::matmul(lhs, rhs) * 2.0F - 1.0F;
        const porch::tensor result = expression;

        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{115.0F, 127.0F, 277.0F, 307.0F})
        );
    }

    void matmul_accepts_expression_operands() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{
            {3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}
        };

        auto expression = porch::matmul(lhs + lhs, rhs);
        const porch::tensor result = expression;

        assert(
            (std::vector<porch::float32_t>{
                 result.data().begin(), result.data().end()
             } == std::vector<porch::float32_t>{116.0F, 128.0F, 278.0F, 308.0F})
        );
    }

    void matmul_rejects_invalid_shapes() {
        const porch::tensor lhs{{2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
        const porch::tensor rhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};

        bool threw = false;
        try {
            const porch::tensor result = porch::matmul(lhs, rhs);
            (void)result;
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    void reshape_uses_lazy_device_ir() {
        const porch::tensor values{
            {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
        };

        const porch::tensor result = porch::reshape(values + 1.0F, {3, 2});

        assert(
            (std::vector<porch::index_t>{
                 result.shape().begin(), result.shape().end()
             } == std::vector<porch::index_t>{3, 2})
        );
        assert(
            (result.cpu() ==
             std::vector<porch::float32_t>{2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F})
        );
    }

    void transpose_uses_lazy_device_ir() {
        const porch::tensor values{
            {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
        };

        const porch::tensor result = porch::transpose(values);

        assert(
            (std::vector<porch::index_t>{
                 result.shape().begin(), result.shape().end()
             } == std::vector<porch::index_t>{3, 2})
        );
        assert(
            (result.cpu() ==
             std::vector<porch::float32_t>{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F})
        );
    }

    void broadcast_to_uses_lazy_device_ir() {
        const porch::tensor values{{3}, {1.0F, 2.0F, 3.0F}};

        const porch::tensor result = porch::broadcast_to(values, {2, 3}) * 2.0F;

        assert(
            (result.cpu() ==
             std::vector<porch::float32_t>{2.0F, 4.0F, 6.0F, 2.0F, 4.0F, 6.0F})
        );
    }

    void concat_uses_lazy_device_ir() {
        const porch::tensor lhs{{2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}};
        const porch::tensor rhs{{2, 1}, {10.0F, 20.0F}};

        const porch::tensor result = porch::concat(lhs, rhs, 1);

        assert(
            (std::vector<porch::index_t>{
                 result.shape().begin(), result.shape().end()
             } == std::vector<porch::index_t>{2, 3})
        );
        assert((
            result.cpu() ==
            std::vector<porch::float32_t>{1.0F, 2.0F, 10.0F, 3.0F, 4.0F, 20.0F}
        ));
    }

    void reductions_use_lazy_device_ir() {
        const porch::tensor values{
            {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
        };

        const porch::tensor column_sums = porch::sum(values, 0);
        const porch::tensor row_maxes = porch::max(values, 1);

        assert((
            column_sums.cpu() == std::vector<porch::float32_t>{5.0F, 7.0F, 9.0F}
        ));
        assert((row_maxes.cpu() == std::vector<porch::float32_t>{3.0F, 6.0F}));
    }

    void keepdim_reductions_use_lazy_device_ir() {
        const porch::tensor values{
            {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
        };

        const porch::tensor sums = porch::sum(values, 1, true);
        const porch::tensor maxes = porch::max(values, 0, true);

        assert(
            (std::vector<porch::index_t>{
                 sums.shape().begin(), sums.shape().end()
             } == std::vector<porch::index_t>{2, 1})
        );
        assert((sums.cpu() == std::vector<porch::float32_t>{6.0F, 15.0F}));
        assert(
            (std::vector<porch::index_t>{
                 maxes.shape().begin(), maxes.shape().end()
             } == std::vector<porch::index_t>{1, 3})
        );
        assert(
            (maxes.cpu() == std::vector<porch::float32_t>{4.0F, 5.0F, 6.0F})
        );
    }

    void block_reductions_handle_axis_larger_than_block() {
        std::vector<porch::float32_t> values(513);
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<porch::float32_t>(index + 1);
        }
        values[400] = 1000.0F;

        const porch::tensor input{{1, 513}, values};
        const porch::tensor total = porch::sum(input, 1);
        const porch::tensor high = porch::max(input, 1);

        assert((total.cpu() == std::vector<porch::float32_t>{132440.0F}));
        assert((high.cpu() == std::vector<porch::float32_t>{1000.0F}));
    }

    void nested_reductions_materialize_with_block_reduction() {
        std::vector<porch::float32_t> values(513);
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<porch::float32_t>(index + 1);
        }

        const porch::tensor input{{1, 513}, values};
        const porch::tensor result = porch::sum(input, 1) * 2.0F + 1.0F;

        assert((result.cpu() == std::vector<porch::float32_t>{263683.0F}));
    }

    void softmax_style_expression_uses_nested_block_reduction() {
        const porch::tensor values{{2, 2}, {0.0F, 0.0F, 1.0F, 1.0F}};

        auto exponentials = porch::exp(values);
        auto denominators =
            porch::broadcast_to(porch::sum(exponentials, 1, true), {2, 2});
        const porch::tensor result = exponentials / denominators;
        const std::vector<porch::float32_t> host = result.cpu();

        assert(host.size() == 4);
        for (const porch::float32_t value : host) {
            assert(value > 0.49F && value < 0.51F);
        }
    }

    void relu_compound_op_uses_primitives() {
        const porch::tensor values{{4}, {-2.0F, -0.5F, 0.0F, 3.0F}};

        const porch::tensor result = porch::relu(values);

        assert(
            (result.cpu() ==
             std::vector<porch::float32_t>{0.0F, 0.0F, 0.0F, 3.0F})
        );
    }

    void softmax_compound_op_uses_primitives() {
        const porch::tensor values{{2, 2}, {0.0F, 0.0F, 1.0F, 1.0F}};

        const porch::tensor result = porch::softmax(values, 1);
        const std::vector<porch::float32_t> host = result.cpu();

        assert(host.size() == 4);
        for (const porch::float32_t value : host) {
            assert(value > 0.49F && value < 0.51F);
        }
    }

    void logistic_compound_op_uses_primitives() {
        const porch::tensor values{{3}, {-1.0F, 0.0F, 1.0F}};

        const porch::tensor result = porch::logistic(values);
        const std::vector<porch::float32_t> host = result.cpu();

        assert(host[0] > 0.26F && host[0] < 0.27F);
        assert(host[1] == 0.5F);
        assert(host[2] > 0.73F && host[2] < 0.74F);
    }

    void exp_uses_lazy_device_ir() {
        const porch::tensor values{{3}, {-1.0F, 0.0F, 2.0F}};

        const porch::tensor result = porch::exp(values);
        const std::vector<porch::float32_t> host = result.cpu();

        assert(host[0] > 0.36F && host[0] < 0.37F);
        assert(host[1] == 1.0F);
        assert(host[2] > 7.38F && host[2] < 7.39F);
    }

    void divide_and_reciprocal_use_lazy_device_ir() {
        const porch::tensor lhs{{3}, {2.0F, 4.0F, 8.0F}};
        const porch::tensor rhs{{3}, {1.0F, 2.0F, 4.0F}};

        const porch::tensor quotient = lhs / rhs;
        const porch::tensor inverse = porch::reciprocal(lhs);

        assert(
            (quotient.cpu() == std::vector<porch::float32_t>{2.0F, 2.0F, 2.0F})
        );
        assert((
            inverse.cpu() == std::vector<porch::float32_t>{0.5F, 0.25F, 0.125F}
        ));
    }

    void elementwise_minimum_maximum_use_lazy_device_ir() {
        const porch::tensor lhs{{4}, {-1.0F, 5.0F, 3.0F, 9.0F}};
        const porch::tensor rhs{{4}, {2.0F, 4.0F, 7.0F, 1.0F}};

        const porch::tensor high = porch::maximum(lhs, rhs);
        const porch::tensor low = porch::minimum(lhs, rhs);

        assert((
            high.cpu() == std::vector<porch::float32_t>{2.0F, 5.0F, 7.0F, 9.0F}
        ));
        assert((
            low.cpu() == std::vector<porch::float32_t>{-1.0F, 4.0F, 3.0F, 1.0F}
        ));
    }

    void unsqueeze_and_expression_shape_are_lazy_metadata() {
        const porch::tensor values{
            {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
        };

        auto expression = porch::unsqueeze(values + 1.0F, 1);
        const porch::tensor leading = porch::unsqueeze(values, 0);
        const porch::tensor trailing = porch::unsqueeze(values, 2);

        assert((expression.shape() == std::vector<porch::index_t>{2, 1, 3}));
        assert(expression.rank() == 3);
        assert(expression.numel() == 6);
        assert(
            (std::vector<porch::index_t>{
                 leading.shape().begin(), leading.shape().end()
             } == std::vector<porch::index_t>{1, 2, 3})
        );
        assert(
            (std::vector<porch::index_t>{
                 trailing.shape().begin(), trailing.shape().end()
             } == std::vector<porch::index_t>{2, 3, 1})
        );
        assert(
            (leading.cpu() ==
             std::vector<porch::float32_t>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})
        );
        assert(
            (trailing.cpu() ==
             std::vector<porch::float32_t>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})
        );

        const porch::tensor result = expression;
        assert(
            (std::vector<porch::index_t>{
                 result.shape().begin(), result.shape().end()
             } == std::vector<porch::index_t>{2, 1, 3})
        );
        assert(
            (result.cpu() ==
             std::vector<porch::float32_t>{2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F})
        );
    }

    void rejects_invalid_shapes() {
        bool threw = false;
        try {
            (void)porch::tensor{{2, 2}, {1.0F, 2.0F, 3.0F}};
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    void cuda_devices_select_cuda_jit_backend() {
        const porch::backend selected = porch::backend_for(porch::device{});

        assert(selected.is_cuda_jit());
        assert(porch::backend_name(selected) == "cuda-jit");
    }

    void cuda_jit_availability_matches_backend_availability() {
        assert(
            porch::cuda_jit::is_available() ==
            porch::is_backend_available(
                porch::backend{porch::backend_kind::cuda_jit}
            )
        );
    }

    void cuda_jit_compiles_or_reports_missing_nvrtc() {
        constexpr std::string_view source = R"cuda(
            extern "C" __global__ void add_kernel(
                const float* lhs, const float* rhs, float* out
            ) {
                const int index = threadIdx.x;
                out[index] = lhs[index] + rhs[index];
            }
        )cuda";

        bool threw = false;
        try {
            const std::string ptx = porch::cuda_jit::compile_to_ptx(source);
            assert(ptx.find(".version") != std::string::npos);
            assert(ptx.find("add_kernel") != std::string::npos);
        }
        catch (const std::runtime_error&) {
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
        {"elementwise_arithmetic_matches_expected_values",
         elementwise_arithmetic_matches_expected_values, true},
        {"chained_arithmetic_matches_expected_values",
         chained_arithmetic_matches_expected_values, true},
        {"fused_expression_materializes_once",
         fused_expression_materializes_once, true},
        {"multiline_expression_stays_lazy_with_auto",
         multiline_expression_stays_lazy_with_auto, true},
        {"tensor_assignment_keeps_graph_lazy",
         tensor_assignment_keeps_graph_lazy, true},
        {"explicit_gpu_sync_and_cpu_copy", explicit_gpu_sync_and_cpu_copy,
         true},
        {"concurrent_same_lazy_tensor_materializes_once_safely",
         concurrent_same_lazy_tensor_materializes_once_safely, true},
        {"concurrent_independent_lazy_tensors_use_thread_streams",
         concurrent_independent_lazy_tensors_use_thread_streams, true},
        {"scalar_elementwise_ops_use_cuda_backend",
         scalar_elementwise_ops_use_cuda_backend, true},
        {"scalar_arithmetic_order_matches_expected_values",
         scalar_arithmetic_order_matches_expected_values, true},
        {"matmul_uses_cuda_backend", matmul_uses_cuda_backend, true},
        {"matmul_rectangular_values_are_correct",
         matmul_rectangular_values_are_correct, true},
        {"matmul_returns_lazy_expression", matmul_returns_lazy_expression,
         true},
        {"matmul_composes_with_elementwise_ir",
         matmul_composes_with_elementwise_ir, true},
        {"matmul_accepts_expression_operands",
         matmul_accepts_expression_operands, true},
        {"matmul_rejects_invalid_shapes", matmul_rejects_invalid_shapes, true},
        {"reshape_uses_lazy_device_ir", reshape_uses_lazy_device_ir, true},
        {"transpose_uses_lazy_device_ir", transpose_uses_lazy_device_ir, true},
        {"broadcast_to_uses_lazy_device_ir", broadcast_to_uses_lazy_device_ir,
         true},
        {"concat_uses_lazy_device_ir", concat_uses_lazy_device_ir, true},
        {"reductions_use_lazy_device_ir", reductions_use_lazy_device_ir, true},
        {"keepdim_reductions_use_lazy_device_ir",
         keepdim_reductions_use_lazy_device_ir, true},
        {"block_reductions_handle_axis_larger_than_block",
         block_reductions_handle_axis_larger_than_block, true},
        {"nested_reductions_materialize_with_block_reduction",
         nested_reductions_materialize_with_block_reduction, true},
        {"softmax_style_expression_uses_nested_block_reduction",
         softmax_style_expression_uses_nested_block_reduction, true},
        {"relu_compound_op_uses_primitives", relu_compound_op_uses_primitives,
         true},
        {"softmax_compound_op_uses_primitives",
         softmax_compound_op_uses_primitives, true},
        {"logistic_compound_op_uses_primitives",
         logistic_compound_op_uses_primitives, true},
        {"exp_uses_lazy_device_ir", exp_uses_lazy_device_ir, true},
        {"divide_and_reciprocal_use_lazy_device_ir",
         divide_and_reciprocal_use_lazy_device_ir, true},
        {"elementwise_minimum_maximum_use_lazy_device_ir",
         elementwise_minimum_maximum_use_lazy_device_ir, true},
        {"unsqueeze_and_expression_shape_are_lazy_metadata",
         unsqueeze_and_expression_shape_are_lazy_metadata, true},
        {"rejects_invalid_shapes", rejects_invalid_shapes, false},
        {"cuda_devices_select_cuda_jit_backend",
         cuda_devices_select_cuda_jit_backend, false},
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
