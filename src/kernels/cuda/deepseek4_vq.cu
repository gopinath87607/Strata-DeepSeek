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
// THE ARITHMETIC IS NOT IN THIS FILE.  The lane map, the codeword decode, the multiply-add expression, the xor
// tree and the gain live in include/strata/kernels/deepseek4_vq.hpp (as __host__ __device__ functions) and in
// deepseek4_vq_device.hpp (the one-warp form).  The P3 MoE kernels and the MoE host oracle use the same
// functions, so there is exactly one copy of the expression the reference's bit-identity depends on.
//
// THE E4M3 CONVERSION IS ARITHMETIC, NOT `cvt.rn.f16x2.e4m3x2`.  The instruction is exact (E4M3 is a subset
// of f16) but it needs sm_89+, and this fork builds for sm_86 and HIP too.  The engine measured the scalar
// table as slower - 26% per decode step, 20% prefill - which is a reason to add the instruction behind an
// arch guard once P7 measures it, not a reason for a second, different conversion now.
#include "strata/kernels/deepseek4_vq.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "deepseek4_vq_device.hpp"

namespace strata::kernels {

namespace {

// ---- decode -------------------------------------------------------------------------------------------

__global__ void deepseek4_vq_dequant_kernel(Deepseek4VqMatrix m, float* __restrict__ out) {
    const uint64_t g = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(m.rows) * m.nidx_row;
    if (g >= total) return;
    const uint32_t r = static_cast<uint32_t>(g / m.nidx_row);
    const uint32_t i = static_cast<uint32_t>(g - static_cast<uint64_t>(r) * m.nidx_row);
    const uint32_t idx = deepseek4_vq_index_at(m, r, i);
    float c[8];
    deepseek4_vq_codeword_f32(m, idx, c);
    const float gain = m.gain_override != nullptr ? m.gain_override[r]
                                                  : deepseek4_vq_f16_to_f32(deepseek4_vq_rd_u16(
                                                        m.gains + static_cast<uint64_t>(r) * 2u));
    float* o = out + static_cast<uint64_t>(r) * m.cols + static_cast<uint64_t>(i) * 8u;
    #pragma unroll
    for (uint32_t d = 0; d < 8; d++) o[d] = c[d] * gain;
}

// ---- row dot ------------------------------------------------------------------------------------------

template <int NBIT, int V3, int EXT>
__global__ void deepseek4_vq_gemv_kernel(Deepseek4VqMatrix m, const uint32_t* __restrict__ x,
                                         float* __restrict__ y) {
    // warp w of the block takes row (blockIdx.x * warps + w); the whole warp tests the same r, so the
    // full-mask shuffles below never run with a partially exited warp.
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t r = blockIdx.x * warps + (threadIdx.x >> 5);
    if (r >= m.rows) return;
    y[r] = deepseek4_vq_row_dot<NBIT, V3, EXT>(m, r, x);
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
