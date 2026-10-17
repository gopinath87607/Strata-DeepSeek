// src/kernels/cuda/deepseek4_vq.cu - P2: DeepSeek-V4.1-Flash routed-expert VQ decode and row dot on the GPU.
//
// The on-disk format and the layout constants are explained once in
// include/strata/kernels/deepseek4_vq.hpp; `tools/dsv4/vq_decode.hpp` is the CPU decoder this file must agree
// with, and `tests/core/deepseek4_vq_cuda_test.cpp` is the gate.  Two kernels:
//
//   deepseek4_vq_dequant  one thread per 8-column group, writes the whole matrix as f32.  The P2 gate is
//                         BYTE-FOR-BYTE equality with the CPU decoder on the engine's v2 fixtures and on real
//                         expert 0 of the layer-0 blob, so every arithmetic step here is the CPU's: the same
//                         3-byte index window, the same integer f16/E4M3 conversion (not `__half`/hardware
//                         FP8, whose edge cases are the CPU's own code), and one final multiply by the gain.
//                         No FMA: the expression is a single product per value.
//
//   deepseek4_vq_gemv_bf16  one warp per row, reading the packed index stream DIRECTLY - the matrix is never
//                         materialised.  This is the reference engine's single-token mapping (lane j takes
//                         index groups j, j+32, ...; 8-value groups; xor-tree warp reduction; gain last).
//                         `x` is bf16 packed two per word because the engine's activation path rounds to bf16
//                         first; the reference measured that as throughput at equal output, not accuracy.
//
// WHY THE INDEX WIDTH AND THE VERSION ARE TEMPLATES IN THE DOT.  The reference measured one runtime
// `order ? :` at 21% on its gate/up kernel; more to the point, a compile-time width makes the three-byte
// window and the row stride constants, and a wrong shift is then a compile-time impossibility rather than a
// value.  The decode kernel does not need this - it is not on a model path - and stays runtime so one launch
// serves every width.
//
// THE E4M3 CONVERSION IS ARITHMETIC, NOT `cvt.rn.f16x2.e4m3x2`.  The instruction is exact (E4M3 is a subset
// of f16) but it needs sm_89+, and this fork builds for sm_86 and HIP too.  The engine measured the scalar
// table as slower - 26% per decode step, 20% prefill - which is a reason to add the instruction behind an
// arch guard once P7 measures it, not a reason for a second, different conversion now.
#include "strata/kernels/deepseek4_vq.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {

// f16 -> f32 with subnormals exact (the CPU decoder's dsv4_f16_to_f32, bit for bit; NaN payloads included).
__device__ __forceinline__ float dsv4_vq_f16(uint16_t h) {
    const uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            e = 112;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3FFu;
            f = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112) << 23) | (m << 13);
    }
    return __uint_as_float(f);
}

// E4M3FN -> f32 (the CPU decoder's ds4_e4m3fn_to_f32): 0x7f/0xff are NaN; exp 0 is mant * 2^-9.  Every
// product here is exact (a 3-bit mantissa and powers of two), so this is the same value as the conversion
// instruction, without being an instruction.
__device__ __forceinline__ float dsv4_vq_e4m3(uint8_t x) {
    const uint32_t abs = x & 0x7Fu;
    const bool sign = (x & 0x80u) != 0;
    if (abs == 0) return sign ? __int_as_float(0x80000000u) : 0.0f;
    if (abs == 0x7Fu) return __int_as_float(0x7fc00000u);   // the CPU returns NAN, positive
    const uint32_t exp = (x >> 3) & 0x0Fu;
    const uint32_t man = x & 0x07u;
    const float value = exp == 0 ? static_cast<float>(man) * 0.001953125f
                                 : (1.0f + static_cast<float>(man) * 0.125f) *
                                       static_cast<float>(1u << (exp - 1)) * 0.015625f;
    return sign ? -value : value;
}

// The three-byte window.  The parser is the only bounds gate: it requires the last window's three bytes to
// exist (the format's 8 B pad), exactly as the CPU decoder does, so a descriptor that reached a kernel is
// safe to read this way.
__device__ __forceinline__ uint32_t dsv4_vq_window(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16);
}

// One 8-value codeword, f32.  v3 reads nc*8 E4M3 bytes; v2 reads nc*8 f16 (16 B per codeword).
__device__ __forceinline__ void dsv4_vq_codeword(const Deepseek4VqMatrix& m, uint32_t idx, float c[8]) {
    if (m.v3) {
        const uint8_t* p = m.codebook + static_cast<uint64_t>(idx) * 8u;
        #pragma unroll
        for (uint32_t d = 0; d < 8; d++) c[d] = dsv4_vq_e4m3(p[d]);
    } else {
        const uint8_t* p = m.codebook + static_cast<uint64_t>(idx) * 16u;
        #pragma unroll
        for (uint32_t d = 0; d < 8; d++) c[d] = dsv4_vq_f16(deepseek4_vq_rd_u16(p + d * 2));
    }
}

// Runtime index extraction, identical to dsv4::VqMatrix::index: v2 8-bit bytes or an NBIT window per group;
// v3 a fixed 12-bit window per row plus the 13th bit from the plane.  The window may start at any bit.
__device__ __forceinline__ uint32_t dsv4_vq_index(const Deepseek4VqMatrix& m, uint32_t r, uint32_t i) {
    if (!m.v3) {
        const uint32_t gi = r * m.nidx_row + i;
        if (m.nbit == 8) return m.stream[gi];
        const uint64_t bit = static_cast<uint64_t>(gi) * m.nbit;
        const uint32_t w = dsv4_vq_window(m.stream + (bit >> 3));
        return (w >> (bit & 7u)) & ((1u << m.nbit) - 1u);
    }
    const uint32_t mrow = (m.nidx_row * 12u + 7u) / 8u;
    const uint8_t* row_stream = m.stream + static_cast<uint64_t>(r) * mrow;
    const uint32_t bit = i * 12u;
    uint32_t v = dsv4_vq_window(row_stream + (bit >> 3)) >> (bit & 7u);
    v &= 0xFFFu;
    if (m.plane != nullptr) {
        const uint32_t prow = (m.nidx_row + 7u) / 8u;
        const uint8_t byte = m.plane[static_cast<uint64_t>(r) * prow + (i >> 3)];
        v |= static_cast<uint32_t>((byte >> (i & 7u)) & 1u) << 12;
    }
    return v;
}

// ---- decode -------------------------------------------------------------------------------------------

__global__ void deepseek4_vq_dequant_kernel(Deepseek4VqMatrix m, float* __restrict__ out) {
    const uint64_t g = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(m.rows) * m.nidx_row;
    if (g >= total) return;
    const uint32_t r = static_cast<uint32_t>(g / m.nidx_row);
    const uint32_t i = static_cast<uint32_t>(g - static_cast<uint64_t>(r) * m.nidx_row);
    const uint32_t idx = dsv4_vq_index(m, r, i);
    float c[8];
    dsv4_vq_codeword(m, idx, c);
    const float gain = m.gain_override != nullptr ? m.gain_override[r]
                                                  : dsv4_vq_f16(deepseek4_vq_rd_u16(m.gains + static_cast<uint64_t>(r) * 2u));
    float* o = out + static_cast<uint64_t>(r) * m.cols + static_cast<uint64_t>(i) * 8u;
    #pragma unroll
    for (uint32_t d = 0; d < 8; d++) o[d] = c[d] * gain;
}

// ---- row dot ------------------------------------------------------------------------------------------

// One round's index for lane j in row r, with the width and version known at compile time.  v3's main stream
// is always 12 bits (the 13th rides the plane); the EXT branch is its own template so the 13-bit layers do
// not pay for a runtime `plane != nullptr` per group.
template <int NBIT, int V3, int EXT>
__device__ __forceinline__ uint32_t dsv4_vq_index_t(const Deepseek4VqMatrix& m, uint32_t r, uint32_t i) {
    if (V3) {
        const uint32_t mrow = (m.nidx_row * 12u + 7u) / 8u;
        const uint8_t* row_stream = m.stream + static_cast<uint64_t>(r) * mrow;
        const uint32_t bit = i * 12u;
        uint32_t v = (dsv4_vq_window(row_stream + (bit >> 3)) >> (bit & 7u)) & 0xFFFu;
        if (EXT) {
            const uint32_t prow = (m.nidx_row + 7u) / 8u;
            const uint8_t byte = m.plane[static_cast<uint64_t>(r) * prow + (i >> 3)];
            v |= static_cast<uint32_t>((byte >> (i & 7u)) & 1u) << 12;
        }
        return v;
    }
    const uint32_t gi = r * m.nidx_row + i;
    const uint64_t bit = static_cast<uint64_t>(gi) * NBIT;
    const uint32_t w = dsv4_vq_window(m.stream + (bit >> 3));
    return (w >> (bit & 7u)) & ((1u << NBIT) - 1u);
}

// The reference's one multiply-add expression (ds41_vq_row.cuh v41_vq_dot8_cw), kept as ONE expression: under
// nvcc's default contraction the sum order and the FMA merge are what a different spelling would change.
// A uint32 holds two bf16, low half = even column; the shifts restore f32.
__device__ __forceinline__ float dsv4_vq_dot8_cw(const float c[8], uint32_t x0, uint32_t x1, uint32_t x2,
                                                 uint32_t x3) {
    return c[0] * __uint_as_float(x0 << 16) + c[1] * __uint_as_float(x0 & 0xffff0000u)
         + c[2] * __uint_as_float(x1 << 16) + c[3] * __uint_as_float(x1 & 0xffff0000u)
         + c[4] * __uint_as_float(x2 << 16) + c[5] * __uint_as_float(x2 & 0xffff0000u)
         + c[6] * __uint_as_float(x3 << 16) + c[7] * __uint_as_float(x3 & 0xffff0000u);
}

// One warp, one row.  Lane j takes groups i = j, j + 32, ... - the reference's lane<->index map - so each
// lane's per-group accumulation and the xor tree below are its single-token path.  `x` is cols/2 uint32.
template <int NBIT, int V3, int EXT>
__device__ __forceinline__ float dsv4_vq_row_dot(const Deepseek4VqMatrix& m, uint32_t r, const uint32_t* x) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t nidx = m.nidx_row;
    float acc = 0.0f;
    for (uint32_t i = lane; i < nidx; i += 32u) {
        const uint32_t v = dsv4_vq_index_t<NBIT, V3, EXT>(m, r, i);
        float c[8];
        dsv4_vq_codeword(m, v, c);
        const uint32_t* xw = x + static_cast<uint64_t>(i) * 4u;   // 8 bf16 = 16 B
        acc += dsv4_vq_dot8_cw(c, xw[0], xw[1], xw[2], xw[3]);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    const float gain = dsv4_vq_f16(deepseek4_vq_rd_u16(m.gains + static_cast<uint64_t>(r) * 2u));
    return acc * gain * (m.gain_override != nullptr ? m.gain_override[r] : 1.0f);
}

template <int NBIT, int V3, int EXT>
__global__ void deepseek4_vq_gemv_kernel(Deepseek4VqMatrix m, const uint32_t* __restrict__ x,
                                         float* __restrict__ y) {
    // warp w of the block takes row (blockIdx.x * warps + w); the whole warp tests the same r, so the
    // full-mask shuffles below never run with a partially exited warp.
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t r = blockIdx.x * warps + (threadIdx.x >> 5);
    if (r >= m.rows) return;
    y[r] = dsv4_vq_row_dot<NBIT, V3, EXT>(m, r, x);
}

// ---- launchers ----------------------------------------------------------------------------------------

const char* dsv4_vq_error_name(uint32_t err) {
    switch (err) {
        case kDsv4VqBadContainer: return "not a DQVL v2/v3 blob";
        case kDsv4VqExpertOutOfRange: return "expert index outside the slot table";
        case kDsv4VqSlotAbsent: return "expert matrix absent";
        case kDsv4VqPayloadOutside: return "payload offset outside the blob";
        case kDsv4VqPayloadMagic: return "payload magic is neither DQVQ nor DQV3";
        case kDsv4VqShape: return "payload shape does not match the caller";
        case kDsv4VqDim: return "payload dim is not 8";
        case kDsv4VqV3Flags: return "v3 flags/main width not understood";
        case kDsv4VqV3PlaneFlag: return "v3 plane flag disagrees with the index width";
        case kDsv4VqV3Codebook: return "v3 codebook outside the blob";
        case kDsv4VqV3Gains: return "v3 gains outside the blob";
        case kDsv4VqV3Stream: return "v3 index stream truncated";
        case kDsv4VqV3PlaneTrunc: return "v3 plane truncated";
        case kDsv4VqV2CodebookGains: return "v2 codebook/gains outside the blob";
        case kDsv4VqV2Stream: return "v2 index stream truncated";
        default: return "no error";
    }
}

}  // namespace

bool deepseek4_vq_open(const uint8_t* blob, uint64_t size, uint32_t expert, uint32_t which, uint32_t rows,
                       uint32_t cols, Deepseek4VqMatrix* out, const char** err) {
    const Deepseek4VqMatrix m = deepseek4_vq_parse(blob, size, expert, which, rows, cols);
    if (err != nullptr) *err = m.ok ? nullptr : dsv4_vq_error_name(m.err);
    if (out != nullptr) *out = m;
    return m.ok != 0;
}

void deepseek4_vq_dequant(const Deepseek4VqMatrix& m, float* out, void* stream) {
    if (!m.ok || out == nullptr || m.rows == 0) return;
    const uint64_t total = static_cast<uint64_t>(m.rows) * m.nidx_row;
    const uint32_t threads = 256;
    const uint64_t blocks = (total + threads - 1) / threads;
    deepseek4_vq_dequant_kernel<<<static_cast<unsigned>(blocks), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        m, out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "deepseek4_vq_dequant launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

bool deepseek4_vq_gemv_bf16(const Deepseek4VqMatrix& m, const uint32_t* x_bf16_pairs, float* y, void* stream) {
    if (!m.ok || x_bf16_pairs == nullptr || y == nullptr || m.rows == 0) return false;
    const uint32_t threads = 256;             // 8 rows per block
    const uint32_t warps = threads / 32u;
    const unsigned blocks = (m.rows + warps - 1u) / warps;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    bool launched = true;
    if (m.v3) {
        if (m.nbit == 13 && m.plane != nullptr) {
            deepseek4_vq_gemv_kernel<13, 1, 1><<<blocks, threads, 0, s>>>(m, x_bf16_pairs, y);
        } else if (m.nbit == 12 && m.plane == nullptr) {
            deepseek4_vq_gemv_kernel<12, 1, 0><<<blocks, threads, 0, s>>>(m, x_bf16_pairs, y);
        } else {
            launched = false;
        }
    } else {
        switch (m.nbit) {
            case 8: deepseek4_vq_gemv_kernel<8, 0, 0><<<blocks, threads, 0, s>>>(m, x_bf16_pairs, y); break;
            case 9: deepseek4_vq_gemv_kernel<9, 0, 0><<<blocks, threads, 0, s>>>(m, x_bf16_pairs, y); break;
            case 10: deepseek4_vq_gemv_kernel<10, 0, 0><<<blocks, threads, 0, s>>>(m, x_bf16_pairs, y); break;
            case 11: deepseek4_vq_gemv_kernel<11, 0, 0><<<blocks, threads, 0, s>>>(m, x_bf16_pairs, y); break;
            case 12: deepseek4_vq_gemv_kernel<12, 0, 0><<<blocks, threads, 0, s>>>(m, x_bf16_pairs, y); break;
            default: launched = false; break;
        }
    }
    if (!launched) return false;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "deepseek4_vq_gemv launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

}  // namespace strata::kernels
