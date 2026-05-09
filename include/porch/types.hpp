#pragma once

#include <stddef.h>
#include <stdint.h>

namespace porch {

    using index_t = int64_t;

#if defined(__FLT16_MANT_DIG__)
    using float16_t = _Float16;
#endif

    using float32_t = float;
    using float64_t = double;

#if defined(__SIZEOF_FLOAT128__)
    using float128_t = __float128;
#endif

} // namespace porch
