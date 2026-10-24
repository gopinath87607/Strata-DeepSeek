// src/ds4/ds4_iq4nl.cuh - one value of a ggml IQ4_NL row (18-byte blocks of 32 values: an F16 scale and 16 four-bit
// codebook indices - the block's first 16 values in the low nibbles, the next 16 in the high ones), as ggml's
// dequantize_row_iq4_nl computes it: d * kvalues_iq4nl[code] - the Engram tables a GSQ-RCO GGUF holds in IQ4_NL.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace strata::ds4 {

constexpr int kIq4nlBytes = 18;

__device__ __forceinline__ float iq4nl_value(const uint8_t* __restrict__ row, int e) {
    const uint8_t* b = row + (size_t) (e / 32) * kIq4nlBytes;
    const int j = e & 31;
    const uint8_t q = b[2 + (j & 15)];
    constexpr int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const float d = __half2float(__ushort_as_half((uint16_t) (b[0] | b[1] << 8)));
    return d * (float) kv[j < 16 ? (q & 0xF) : (q >> 4)];
}

}  // namespace strata::ds4
