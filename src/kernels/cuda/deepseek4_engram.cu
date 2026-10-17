// src/kernels/cuda/deepseek4_engram.cu - P4: the engram's device half (rows, wkv, gate).
//
// The format and the P4 contract are explained once in include/strata/kernels/deepseek4_engram.hpp; the host
// oracle is include/strata/artifact/deepseek4_engram.hpp, and tests/core/deepseek4_engram_cuda_test.cpp is the
// gate.  The reference implementations this file is ported from are named per kernel below.
//
// The conversions are NOT written here: both sides call strata::deepseek4::engram_e4m3_to_f32 /
// engram_e8m0_to_f32 / engram_bf16r (include/strata/artifact/deepseek4_engram_bits.hpp, __host__ __device__),
// which is what makes the row dequant's bit-equality structural rather than a promise.
#include "strata/kernels/deepseek4_engram.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

namespace {

using strata::deepseek4::engram_bf16r;
using strata::deepseek4::engram_e4m3_to_f32;
using strata::deepseek4::engram_e8m0_to_f32;
using strata::deepseek4::engram_f32_to_bits;
using strata::deepseek4::engram_row_value;

void dsv4_engram_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---- rows: fp8 (E4M3 x UE8M0/32) -> the bf16 grid ------------------------------------------------
//
// One thread per element (the reference's cuda_v41_4.inc.cu:302-308): a row's scale plane is its own tail,
// at `head_dim + d/32`.  The value is the shared conversion (engram_row_value), so this is bit-identical to
// strata::deepseek4::engram_rows_f32 by construction.
__global__ void dsv4_engram_rows_kernel(float* __restrict__ out, const uint8_t* __restrict__ raw,
                                        uint32_t n_rows, uint32_t head_dim) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<uint64_t>(n_rows) * head_dim) return;
    const uint32_t r = static_cast<uint32_t>(i / head_dim);
    const uint32_t d = static_cast<uint32_t>(i % head_dim);
    const uint32_t stride = head_dim + head_dim / 32u;
    const uint8_t* row = raw + static_cast<uint64_t>(r) * stride;
    out[i] = engram_row_value(row, head_dim, d);
}

// ---- the gate -------------------------------------------------------------------------------------
//
// The reference's kernel (cuda_v41_3.inc.cu:109-141), one block of 256 threads per (token, hc row): three
// f32 reductions per thread stride, then a shared-memory tree.  The compare against the f64 emulation is a
// bound (see the header), not equality - the kernels, not the arithmetic, are why.
__global__ void dsv4_engram_gate_kernel(float* __restrict__ hc, const float* __restrict__ kv,
                                        const float* __restrict__ qw, const float* __restrict__ kw,
                                        uint32_t e, uint32_t n_hc, float eps) {
    const uint32_t t = blockIdx.y, c = blockIdx.x;
    float* h = hc + (static_cast<uint64_t>(t) * n_hc + c) * e;
    const float* key = kv + static_cast<uint64_t>(t) * (n_hc + 1u) * e + static_cast<uint64_t>(c) * e;
    const float* val = kv + static_cast<uint64_t>(t) * (n_hc + 1u) * e + static_cast<uint64_t>(n_hc) * e;
    float sh = 0.f, sk = 0.f, sd = 0.f;
    for (uint32_t d = threadIdx.x; d < e; d += blockDim.x) {
        const float hv = h[d], kvd = key[d], w = qw[static_cast<uint64_t>(c) * e + d] * kw[static_cast<uint64_t>(c) * e + d];
        sh += hv * hv;
        sk += kvd * kvd;
        sd += hv * w * kvd;
    }
    __shared__ float r[3][256];
    r[0][threadIdx.x] = sh;
    r[1][threadIdx.x] = sk;
    r[2][threadIdx.x] = sd;
    __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            r[0][threadIdx.x] += r[0][threadIdx.x + k];
            r[1][threadIdx.x] += r[1][threadIdx.x + k];
            r[2][threadIdx.x] += r[2][threadIdx.x + k];
        }
        __syncthreads();
    }
    const float rstd = rsqrtf(r[0][0] / static_cast<float>(e) + eps) * rsqrtf(r[1][0] / static_cast<float>(e) + eps);
    const float dot = r[2][0] * rstd * rsqrtf(static_cast<float>(e));
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float z = copysignf(mag, dot);
    const float gate = 1.0f / (1.0f + expf(-z));
    for (uint32_t d = threadIdx.x; d < e; d += blockDim.x) h[d] = engram_bf16r(h[d] + gate * val[d]);
}

// ---- the wkv activation copy ----------------------------------------------------------------------
//
// The reference's v41_x_to_bf16_kernel (cuda_v41_1.inc.cu:154-157), spelled with the shared RNE round so the
// grid is the same one engram_bf16r and the host oracle put values on (no hardware bf16 type needed).
__global__ void dsv4_engram_x_to_bf16_kernel(uint16_t* __restrict__ out, const float* __restrict__ x, uint64_t n) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<uint16_t>(engram_f32_to_bits(engram_bf16r(x[i])) >> 16);
}

// ---- wkv: fp8_32x32 weight GEMV ---------------------------------------------------------------
//
// The reference's v41_fp8blk_gemv_kernel (ds41_fp8blk.cuh:179-256), single group, without PDL: one warp per
// output row, each lane eats 16 consecutive E4M3 values per step (one 128-bit load), so its 16 elements always
// sit in one 32-wide scale tile and the scale is looked up once.  XB=0 reads f32 activations (a single token);
// XB=1 reads the caller's bf16 copy as raw uint4 words and restores f32 by shifting, the reference's arm
// verbatim.  ksplit splits a row across warps when the output is small, and its two arms are kept separate
// because a shared float xv[16] intermediate changes register allocation (the reference measured it).
template <uint32_t NT, uint32_t XB>
__global__ void dsv4_engram_wkv_kernel(float* __restrict__ out, const uint8_t* __restrict__ w,
                                       const uint8_t* __restrict__ sc, const float* __restrict__ x,
                                       const uint16_t* __restrict__ x16, uint32_t in_dim, uint32_t out_dim,
                                       uint32_t sbc, uint32_t ksplit) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) {
        const uint8_t* wr = w + static_cast<uint64_t>(r) * in_dim;
        const uint8_t* scr = sc + static_cast<uint64_t>(r >> 5) * sbc;
        for (uint32_t c = kpart * 512u + lane * 16u; c < in_dim; c += ksplit * 512u) {
            const uint4 raw = *reinterpret_cast<const uint4*>(wr + c);
            uint8_t b[16];
            memcpy(b, &raw, 16);
            const float s = engram_e8m0_to_f32(scr[c >> 5]);
            float wv[16];
            #pragma unroll
            for (uint32_t j = 0; j < 16u; j++) wv[j] = engram_e4m3_to_f32(b[j]) * s;
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                if (XB) {
                    const uint4* xp = reinterpret_cast<const uint4*>(x16 + static_cast<uint64_t>(t) * in_dim + c);
                    const uint4 x0 = xp[0], x1 = xp[1];
                    const uint32_t xu[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; j++) {
                        acc[t] += wv[2u * j] * __uint_as_float(xu[j] << 16);
                        acc[t] += wv[2u * j + 1u] * __uint_as_float(xu[j] & 0xffff0000u);
                    }
                } else {
                    const float* xt = x + static_cast<uint64_t>(t) * in_dim + c;
                    #pragma unroll
                    for (uint32_t j = 0; j < 16u; j++) acc[t] += wv[j] * xt[j];
                }
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t] = v;
        }
    }
    if (ksplit > 1u) {
        if (lane == 0) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) red[warp][t] = acc[t];
        }
        __syncthreads();
        if (kpart == 0 && lane == 0 && r < out_dim) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float v = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) v += red[rloc * ksplit + k][t];
                out[static_cast<uint64_t>(t) * out_dim + r] = v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) out[static_cast<uint64_t>(t) * out_dim + r] = acc[t];
    }
}

}  // namespace

bool deepseek4_engram_rows(const uint8_t* raw, float* out, uint32_t n_rows, uint32_t head_dim, void* stream) {
    if (raw == nullptr || out == nullptr || n_rows == 0 || head_dim == 0 || (head_dim % 32u) != 0u) return false;
    const uint64_t total = static_cast<uint64_t>(n_rows) * head_dim;
    const uint32_t threads = 256;
    const uint64_t blocks = (total + threads - 1) / threads;
    dsv4_engram_rows_kernel<<<static_cast<unsigned>(blocks), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        out, raw, n_rows, head_dim);
    dsv4_engram_check("deepseek4_engram_rows");
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

bool deepseek4_engram_gate(float* hc, const float* kv, const float* qw, const float* kw, uint32_t e,
                           uint32_t n_hc, uint32_t n_tok, float eps, void* stream) {
    if (hc == nullptr || kv == nullptr || qw == nullptr || kw == nullptr || e == 0 || n_hc == 0 || n_tok == 0) {
        return false;
    }
    dsv4_engram_gate_kernel<<<dim3(n_hc, n_tok), 256, 0, static_cast<cudaStream_t>(stream)>>>(hc, kv, qw, kw, e, n_hc,
                                                                                              eps);
    dsv4_engram_check("deepseek4_engram_gate");
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

bool deepseek4_engram_x_to_bf16(const float* x, uint16_t* x16, uint64_t n, void* stream) {
    if (x == nullptr || x16 == nullptr || n == 0) return false;
    const uint32_t threads = 256;
    const uint64_t blocks = (n + threads - 1) / threads;
    dsv4_engram_x_to_bf16_kernel<<<static_cast<unsigned>(blocks), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        x16, x, n);
    dsv4_engram_check("deepseek4_engram_x_to_bf16");
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

bool deepseek4_engram_wkv(const uint8_t* wkv, const float* x, const uint16_t* x_bf16, float* y, uint32_t in_dim,
                          uint32_t out_dim, uint32_t n_tok, void* stream) {
    if (wkv == nullptr || x == nullptr || y == nullptr || n_tok == 0 || out_dim == 0 || in_dim == 0 ||
        (in_dim % 32u) != 0u) {
        return false;
    }
    if (n_tok > 1 && x_bf16 == nullptr) return false;

    // The reference's launcher (ds41_fp8blk.cuh:291-297): split a row across warps only while the row count is
    // small enough that the grid would otherwise starve, and never more than the 512-wide steps give.
    uint32_t ksplit = 1;
    while (ksplit < 8u && static_cast<uint64_t>(out_dim) * ksplit < 32768u) ksplit <<= 1;
    const uint32_t nseg = (in_dim + 511u) / 512u;
    while (ksplit > 1u && ksplit > nseg) ksplit >>= 1;
    const uint32_t rows_per_block = 8u / ksplit;
    const uint32_t sbc = (in_dim + 31u) / 32u;
    const unsigned blocks = (out_dim + rows_per_block - 1u) / rows_per_block;
    const uint8_t* sc = wkv + static_cast<uint64_t>(in_dim) * out_dim;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);

    // Batches over 8 run in 8-row segments (the reference's grouped entry does the same); the token count is
    // the template's NT so the accumulator array stays in registers.
    for (uint32_t off = 0; off < n_tok; off += 8u) {
        const uint32_t nn = n_tok - off < 8u ? n_tok - off : 8u;
        const float* xo = x + static_cast<uint64_t>(off) * in_dim;
        const uint16_t* xo16 = x_bf16 != nullptr ? x_bf16 + static_cast<uint64_t>(off) * in_dim : nullptr;
        float* yo = y + static_cast<uint64_t>(off) * out_dim;
        switch (nn) {
            case 1: dsv4_engram_wkv_kernel<1, 0><<<blocks, 256, 0, s>>>(yo, wkv, sc, xo, xo16, in_dim, out_dim, sbc, ksplit); break;
            case 2: dsv4_engram_wkv_kernel<2, 1><<<blocks, 256, 0, s>>>(yo, wkv, sc, xo, xo16, in_dim, out_dim, sbc, ksplit); break;
            case 3: dsv4_engram_wkv_kernel<3, 1><<<blocks, 256, 0, s>>>(yo, wkv, sc, xo, xo16, in_dim, out_dim, sbc, ksplit); break;
            case 4: dsv4_engram_wkv_kernel<4, 1><<<blocks, 256, 0, s>>>(yo, wkv, sc, xo, xo16, in_dim, out_dim, sbc, ksplit); break;
            case 5: dsv4_engram_wkv_kernel<5, 1><<<blocks, 256, 0, s>>>(yo, wkv, sc, xo, xo16, in_dim, out_dim, sbc, ksplit); break;
            case 6: dsv4_engram_wkv_kernel<6, 1><<<blocks, 256, 0, s>>>(yo, wkv, sc, xo, xo16, in_dim, out_dim, sbc, ksplit); break;
            case 7: dsv4_engram_wkv_kernel<7, 1><<<blocks, 256, 0, s>>>(yo, wkv, sc, xo, xo16, in_dim, out_dim, sbc, ksplit); break;
            default: dsv4_engram_wkv_kernel<8, 1><<<blocks, 256, 0, s>>>(yo, wkv, sc, xo, xo16, in_dim, out_dim, sbc, ksplit); break;
        }
    }
    dsv4_engram_check("deepseek4_engram_wkv");
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

}  // namespace strata::kernels
