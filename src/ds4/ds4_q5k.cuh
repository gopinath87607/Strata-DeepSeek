// src/ds4/ds4_q5k.cuh - a row of ggml Q5_K (256 values a 176-byte block: d and dmin F16, 12 scale / min bytes six bits
// each, 32 fifth-bit bytes, 128 four-bit bytes) as BF16, the values exactly as ggml's dequantize_row_q5_K computes them
// (d * sc * v - min * m, the products in that order) and rounded to the engine's BF16 (nearest even) - the dense
// matrices a GSQ-RCO GGUF holds in Q5_K (mat_bf16, the embedding table).
#pragma once

#include <cstdint>
#include <cstring>

namespace strata::ds4 {

inline float q5k_f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            u = s | (uint32_t) (127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) u = s | 0x7f800000u | m << 13;
    else u = s | (e - 15 + 127) << 23 | m << 13;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t q5k_f32_to_bf16(float f) {   // round to nearest even (ggml's ggml_fp32_to_bf16_rne)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) ((u >> 16) | ((u & 0xffffu) ? 1u : 0u));
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// ggml's get_scale_min_k4: a Q5_K block's 12 scale bytes' j-th six-bit scale and min
inline void q5k_scale_min(const uint8_t* q, int j, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (uint8_t) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (uint8_t) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// k values (a whole number of 256-value blocks) as BF16, the order of ggml's dequantize_row_q5_K
inline void q5k_decode(const uint8_t* b, int k, uint16_t* y) {
    for (int blk = 0; blk < k / 256; ++blk, b += 176, y += 256) {
        const uint8_t* sc = b + 4;    // the 8 scales and 8 mins, six bits each
        const uint8_t* qh = b + 16;   // the values' fifth bits
        const uint8_t* ql = b + 48;   // four bits each
        const float d = q5k_f16_to_f32((uint16_t) (b[0] | b[1] << 8));
        const float mn = q5k_f16_to_f32((uint16_t) (b[2] | b[3] << 8));
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 4; ++j, ql += 32, is += 2, u1 <<= 2, u2 <<= 2) {
            uint8_t s0, m0, s1, m1;
            q5k_scale_min(sc, is, s0, m0);
            q5k_scale_min(sc, is + 1, s1, m1);
            const float d0 = d * (float) s0, mn0 = mn * (float) m0;
            const float d1 = d * (float) s1, mn1 = mn * (float) m1;
            uint16_t* yy = y + 64 * j;
            for (int l = 0; l < 32; ++l) {
                const int v0 = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
                const int v1 = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                yy[l] = q5k_f32_to_bf16(d0 * (float) v0 - mn0);
                yy[l + 32] = q5k_f32_to_bf16(d1 * (float) v1 - mn1);
            }
        }
    }
}

}  // namespace strata::ds4
