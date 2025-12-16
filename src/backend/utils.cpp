#include <torch/extension.h>

#include <cstdint>

#if defined(_MSC_VER)

#include <intrin.h>
static inline int32_t clz32(uint32_t x)
{
    if (x == 0) return 32;
    unsigned long index;
    _BitScanReverse(&index, x);
    return 31 - static_cast<int32_t>(index);
}

#elif defined(__GNUC__) || defined(__clang__)

static inline int32_t clz32(uint32_t x)
{
    if (x == 0) return 32;
    return __builtin_clz(x);
}

#else
#error "Unsupported compiler."
#endif

uint32_t get_icosph_level(uint32_t V)
{
    uint32_t fourPowL = (V - 2) / 10;
    TORCH_CHECK(fourPowL * 10 + 2 == V, "Invalid input array. (No matching level)");
    uint32_t twoL = clz32(fourPowL);
    TORCH_CHECK(2 << twoL == fourPowL, "Invalid input array. (No matching level)");
    return twoL / 2;
}