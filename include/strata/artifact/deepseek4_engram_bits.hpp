// include/strata/artifact/deepseek4_engram_bits.hpp - the engram's fp8/bf16 conversions, host and device.
//
// P4 requires the engram row dequant to be BIT-EXACT: every a row value is one E4M3 code times one UE8M0
// power of two, both exact in f32, rounded once to the bf16 grid.  There is no accumulation and no judgment in
// it, so the device kernel and the host oracle must run the same code and a second copy is how that stops
// being true.  Keeping the conversions in their own header lets `strata/kernels/deepseek4_engram.hpp` include
// them from a .cu without dragging in the host-only sidecar reader (`deepseek4_engram.hpp`: filesystem, POSIX
// descriptors, std::vector tables).
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_DSV4_HD __host__ __device__
#else
#define STRATA_DSV4_HD
#endif

namespace strata::deepseek4 {

STRATA_DSV4_HD inline float engram_bits_to_f32(uint32_t u) {
    // memcpy rather than __uint_as_float: this is a __host__ __device__ function and the intrinsics are
    // __device__-only.  A 4-byte memcpy is a register move in device code and -fno-strict-aliasing-safe on host.
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

STRATA_DSV4_HD inline uint32_t engram_f32_to_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

/// 2^e for e in [-127, 127]; a power of two is exact in f32 and this avoids a libm call in device code.
STRATA_DSV4_HD inline float engram_pow2(int e) {
    return engram_bits_to_f32(static_cast<uint32_t>(e + 127) << 23);
}

STRATA_DSV4_HD inline float engram_e4m3_to_f32(uint8_t x) {
    const uint8_t abs = x & 0x7f;
    const bool sign = (x & 0x80) != 0;
    if (abs == 0) return sign ? -0.0f : 0.0f;
    if (abs == 0x7f) return engram_bits_to_f32(0x7fc00000u);   // E4M3FN: both signed 0x7f are NaN
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    const float v = exp == 0 ? static_cast<float>(man) * engram_pow2(-9)
                             : (1.0f + static_cast<float>(man) * 0.125f) * engram_pow2(exp - 7);
    return sign ? -v : v;
}

STRATA_DSV4_HD inline float engram_e8m0_to_f32(uint8_t e) {
    return engram_bits_to_f32(e == 0 ? 0x00400000u : (static_cast<uint32_t>(e) << 23));   // e=0 is 2^-127
}

/// RNE round f32 -> bf16 -> f32, the engine's `v41_bf16r` bit for bit (Inf and NaN pass through).
STRATA_DSV4_HD inline float engram_bf16r(float x) {
    uint32_t u = engram_f32_to_bits(x);
    if ((u & 0x7f800000u) != 0x7f800000u) {
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
    }
    return engram_bits_to_f32(u);
}

/// Element `d` of a raw row (`head_dim` weights, then `head_dim/32` scales).
STRATA_DSV4_HD inline float engram_row_value(const uint8_t* row, uint32_t head_dim, uint32_t d) {
    return engram_bf16r(engram_e4m3_to_f32(row[d]) * engram_e8m0_to_f32(row[head_dim + d / 32]));
}

}  // namespace strata::deepseek4
