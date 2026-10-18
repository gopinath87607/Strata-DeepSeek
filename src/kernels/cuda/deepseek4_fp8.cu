// src/kernels/cuda/deepseek4_fp8.cu - P5: the DSpark towers' fp8 skeleton matmuls and the draft's small helpers.
//
// The format and the contract are in include/strata/kernels/deepseek4_fp8.hpp; the reference kernels this file
// is ported from are named per kernel below.  The single-matrix GEMV is deliberately NOT re-implemented: it is
// exactly the engram's wkv (deepseek4_engram.cu, the reference's v41_fp8blk_gemv_kernel with n_groups = 1), so
// the entry calls that kernel and adds only the reference's optional bf16 output rounding.
#include "strata/kernels/deepseek4_fp8.hpp"

#include "strata/artifact/deepseek4_engram_bits.hpp"
#include "strata/kernels/deepseek4_dense.hpp"
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

void dsv4_fp8_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---- grouped fp8 GEMV: the towers' attn_output_a ------------------------------------------------
//
// The reference's v41_fp8blk_gemv_kernel with grid.y = the group (ds41_fp8blk.cuh:179-256, the grouped
// launcher :275-319): group g owns the row segment [g*out_dim, (g+1)*out_dim) of both planes, so it is a plain
// offset.  One warp per output row, 16 consecutive E4M3 values per lane per step, one scale lookup per step;
// XB=1 reads the caller's bf16 activation copy (the reference's knife, same as the engram wkv).
template <uint32_t NT, uint32_t XB>
__global__ void dsv4_fp8_grouped_kernel(float* __restrict__ out, const uint8_t* __restrict__ w,
                                        const uint8_t* __restrict__ sc, const float* __restrict__ x,
                                        const uint16_t* __restrict__ x16, uint32_t in_dim, uint32_t out_dim,
                                        uint32_t sbc, uint32_t ksplit, uint32_t sc_gstride, uint32_t x_stride,
                                        uint32_t out_stride, uint32_t x_gstride, uint32_t out_gstride) {
    const uint32_t g = blockIdx.y;
    const uint8_t* wg = w + static_cast<uint64_t>(g) * in_dim * out_dim;
    const uint8_t* scg = sc + static_cast<uint64_t>(g) * sc_gstride;
    const float* xg = x + static_cast<uint64_t>(g) * x_gstride;
    const uint16_t* xg16 = XB ? x16 + static_cast<uint64_t>(g) * x_gstride : nullptr;
    float* og = out + static_cast<uint64_t>(g) * out_gstride;

    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) {
        const uint8_t* wr = wg + static_cast<uint64_t>(r) * in_dim;
        const uint8_t* scr = scg + static_cast<uint64_t>(r >> 5) * sbc;
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
                    const uint4* xp = reinterpret_cast<const uint4*>(xg16 + static_cast<uint64_t>(t) * x_stride + c);
                    const uint4 x0 = xp[0], x1 = xp[1];
                    const uint32_t xu[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; j++) {
                        acc[t] += wv[2u * j] * __uint_as_float(xu[j] << 16);
                        acc[t] += wv[2u * j + 1u] * __uint_as_float(xu[j] & 0xffff0000u);
                    }
                } else {
                    const float* xt = xg + static_cast<uint64_t>(t) * x_stride + c;
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
                og[static_cast<uint64_t>(t) * out_stride + r] = v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) og[static_cast<uint64_t>(t) * out_stride + r] = acc[t];
    }
}

// ---- the main-hidden capture ---------------------------------------------------------------------
//
// The reference's v41_hc_mean_kernel (cuda/ds41_draft.cuh:167-178), the eager arm (posd = null): one thread per
// (row, d); s / n_hc in f32 exactly as written, no bf16 round.
__global__ void dsv4_hc_mean_kernel(float* __restrict__ out, const float* __restrict__ hc, uint32_t e,
                                    uint32_t n_hc, uint32_t n_rows, uint32_t src_row0, uint32_t slot,
                                    uint32_t n_slot, uint32_t dst_pos0, uint32_t cap) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<uint64_t>(n_rows) * e) return;
    const uint32_t t = static_cast<uint32_t>(i / e), d = static_cast<uint32_t>(i % e);
    const float* h = hc + static_cast<uint64_t>(src_row0 + t) * n_hc * e + d;
    float s = 0.f;
    for (uint32_t c = 0; c < n_hc; c++) s += h[static_cast<uint64_t>(c) * e];
    out[static_cast<uint64_t>((dst_pos0 + t) % cap) * n_slot * e + static_cast<uint64_t>(slot) * e + d] =
        s / static_cast<float>(n_hc);
}

// ---- ring -> contiguous --------------------------------------------------------------------------
//
// The reference's v41_ring_rows_kernel (cuda/ds41_draft.cuh:196-204), the eager arm (firstd = null).
__global__ void dsv4_ring_rows_kernel(float* __restrict__ dst, const float* __restrict__ ring, uint32_t row_floats,
                                      uint32_t cap, uint32_t first_row, uint32_t count) {
    const uint32_t t = blockIdx.x;
    if (t >= count) return;
    const float* src = ring + static_cast<uint64_t>((first_row + t) % cap) * row_floats;
    float* d = dst + static_cast<uint64_t>(t) * row_floats;
    for (uint32_t j = threadIdx.x; j < row_floats; j += blockDim.x) d[j] = src[j];
}

// ---- markov row gather (bf16 table -> f32 output) -------------------------------------------------
//
// The reference's v41_row_gather_kernel (cuda/ds41_draft.cuh:218-229), the bf16 arm.  A negative id writes
// zeros; __bfloat162float is the exact bit extension the reference uses.
__global__ void dsv4_row_gather_bf16_kernel(float* __restrict__ out, const uint16_t* __restrict__ tab,
                                            const int32_t* __restrict__ ids, uint32_t which, uint32_t dim,
                                            uint32_t out_row) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const int32_t id = ids[which];
    float v = 0.f;
    if (id >= 0) {
        const uint16_t h = tab[static_cast<uint64_t>(id) * dim + d];
        v = __uint_as_float(static_cast<uint32_t>(h) << 16);
    }
    out[static_cast<uint64_t>(out_row) * dim + d] = v;
}

// ---- argmax -------------------------------------------------------------------------------------
//
// One block, one row: the driver's host rule (dsv4_run.cu argmax_logits) transcribed - start at 0, strict >,
// so ties take the lower index.  The reduction keeps the (value, index) pair and prefers the lower index on
// equality, which makes the result independent of thread scheduling.  NaN: out of contract (as in the
// reference; a healthy run's logits are finite).
__global__ void dsv4_argmax_kernel(int32_t* __restrict__ out, const float* __restrict__ x, uint32_t n) {
    __shared__ float bv[1024];
    __shared__ uint32_t bi[1024];
    const uint32_t tid = threadIdx.x;
    float best = x[0];
    uint32_t idx = 0;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > best || (i > 0 && v == best && i < idx)) { best = v; idx = i; }
    }
    bv[tid] = best;
    bi[tid] = idx;
    __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) {
            const float ov = bv[tid + k];
            const uint32_t oi = bi[tid + k];
            if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) { bv[tid] = ov; bi[tid] = oi; }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = static_cast<int32_t>(bi[0]);
}

// ---- row add: the markov head's bias onto one logits row ----------------------------------------
//
// The reference's v41_row_add_kernel (cuda/ds41_draft.cuh), one f32 add per element, no round.
__global__ void dsv4_row_add_kernel(float* __restrict__ out, const float* __restrict__ add, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += add[i];
}

}  // namespace

bool deepseek4_matmul_fp8(float* d_out, const uint8_t* d_w, const float* d_x, const uint16_t* d_x_bf16,
                          uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, bool round_out, void* stream) {
    if (d_out == nullptr || d_w == nullptr || d_x == nullptr || n_tok == 0 || out_dim == 0 || in_dim == 0 ||
        (in_dim % 32u) != 0u) {
        return false;
    }
    if (!deepseek4_engram_wkv(d_w, d_x, d_x_bf16, d_out, in_dim, out_dim, n_tok, stream)) return false;
    if (!round_out) return true;
    return deepseek4_round_bf16(d_out, static_cast<uint64_t>(n_tok) * out_dim, stream);
}

bool deepseek4_matmul_fp8_grouped(float* d_out, const uint8_t* d_w, const float* d_x, const uint16_t* d_x_bf16,
                                  uint32_t n_groups, uint32_t group_dim, uint32_t rank, uint32_t n_tok,
                                  bool round_out, void* stream) {
    if (d_out == nullptr || d_w == nullptr || d_x == nullptr || n_groups == 0 || group_dim == 0 || rank == 0 ||
        n_tok == 0 || (group_dim % 32u) != 0u || (rank % 32u) != 0u) {
        return false;
    }
    if (n_tok > 1 && d_x_bf16 == nullptr) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const uint64_t out_all = static_cast<uint64_t>(n_groups) * rank;
    const uint8_t* sc = d_w + static_cast<uint64_t>(n_groups) * group_dim * rank;
    const uint32_t sbc = group_dim / 32u;   // scale row-block width inside one group
    const uint32_t sc_gstride = (rank / 32u) * sbc;

    // The reference's launcher (ds41_fp8blk.cuh:291-297): split a row across warps while the total output is
    // small, never more than the 512-wide steps give.
    uint32_t ksplit = 1;
    while (ksplit < 8u && static_cast<uint64_t>(rank) * ksplit * n_groups < 32768u) ksplit <<= 1;
    const uint32_t nseg = (group_dim + 511u) / 512u;
    while (ksplit > 1u && ksplit > nseg) ksplit >>= 1;
    const uint32_t rows_per_block = 8u / ksplit;
    const dim3 grid((rank + rows_per_block - 1u) / rows_per_block, n_groups);

    // Batches over 8 run in 8-row segments, as the reference's grouped entry does.
    for (uint32_t off = 0; off < n_tok; off += 8u) {
        const uint32_t nn = n_tok - off < 8u ? n_tok - off : 8u;
        const float* xo = d_x + static_cast<uint64_t>(off) * n_groups * group_dim;
        const uint16_t* xo16 = d_x_bf16 != nullptr
                                   ? d_x_bf16 + static_cast<uint64_t>(off) * n_groups * group_dim
                                   : nullptr;
        float* yo = d_out + static_cast<uint64_t>(off) * out_all;
        const uint32_t x_stride = n_groups * group_dim, out_stride = static_cast<uint32_t>(out_all);
        switch (nn) {
            case 1: dsv4_fp8_grouped_kernel<1, 0><<<grid, 256, 0, s>>>(yo, d_w, sc, xo, xo16, group_dim, rank, sbc, ksplit, sc_gstride, x_stride, out_stride, group_dim, rank); break;
            case 2: dsv4_fp8_grouped_kernel<2, 1><<<grid, 256, 0, s>>>(yo, d_w, sc, xo, xo16, group_dim, rank, sbc, ksplit, sc_gstride, x_stride, out_stride, group_dim, rank); break;
            case 3: dsv4_fp8_grouped_kernel<3, 1><<<grid, 256, 0, s>>>(yo, d_w, sc, xo, xo16, group_dim, rank, sbc, ksplit, sc_gstride, x_stride, out_stride, group_dim, rank); break;
            case 4: dsv4_fp8_grouped_kernel<4, 1><<<grid, 256, 0, s>>>(yo, d_w, sc, xo, xo16, group_dim, rank, sbc, ksplit, sc_gstride, x_stride, out_stride, group_dim, rank); break;
            case 5: dsv4_fp8_grouped_kernel<5, 1><<<grid, 256, 0, s>>>(yo, d_w, sc, xo, xo16, group_dim, rank, sbc, ksplit, sc_gstride, x_stride, out_stride, group_dim, rank); break;
            case 6: dsv4_fp8_grouped_kernel<6, 1><<<grid, 256, 0, s>>>(yo, d_w, sc, xo, xo16, group_dim, rank, sbc, ksplit, sc_gstride, x_stride, out_stride, group_dim, rank); break;
            case 7: dsv4_fp8_grouped_kernel<7, 1><<<grid, 256, 0, s>>>(yo, d_w, sc, xo, xo16, group_dim, rank, sbc, ksplit, sc_gstride, x_stride, out_stride, group_dim, rank); break;
            default: dsv4_fp8_grouped_kernel<8, 1><<<grid, 256, 0, s>>>(yo, d_w, sc, xo, xo16, group_dim, rank, sbc, ksplit, sc_gstride, x_stride, out_stride, group_dim, rank); break;
        }
    }
    dsv4_fp8_check("deepseek4_matmul_fp8_grouped");
    if (stream == nullptr) cudaDeviceSynchronize();
    return round_out ? deepseek4_round_bf16(d_out, static_cast<uint64_t>(n_tok) * out_all, stream) : true;
}

bool deepseek4_hc_mean(float* d_out, const float* d_hc, uint32_t n_embd, uint32_t n_hc, uint32_t n_rows,
                       uint32_t src_row0, uint32_t slot, uint32_t n_slot, uint32_t dst_pos0, uint32_t cap,
                       void* stream) {
    if (d_out == nullptr || d_hc == nullptr || n_embd == 0 || n_hc == 0 || n_rows == 0 || n_rows > cap ||
        cap == 0 || n_slot == 0) {
        return false;
    }
    const uint64_t n = static_cast<uint64_t>(n_rows) * n_embd;
    const unsigned blocks = static_cast<unsigned>((n + 255u) / 256u);
    dsv4_hc_mean_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_out, d_hc, n_embd, n_hc, n_rows,
                                                                               src_row0, slot, n_slot, dst_pos0, cap);
    dsv4_fp8_check("deepseek4_hc_mean");
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

bool deepseek4_ring_rows(float* d_out, const float* d_ring, uint32_t row_elems, uint32_t cap, uint32_t first,
                         uint32_t rows, void* stream) {
    if (d_out == nullptr || d_ring == nullptr || rows == 0 || row_elems == 0 || cap == 0 || rows > cap) return false;
    dsv4_ring_rows_kernel<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_out, d_ring, row_elems, cap, first,
                                                                              rows);
    dsv4_fp8_check("deepseek4_ring_rows");
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

bool deepseek4_row_gather_bf16(float* d_out, const uint16_t* d_table, uint32_t n_rows, uint32_t row_elems,
                               const int32_t* d_ids, uint32_t src_index, uint32_t dst_index, void* stream) {
    if (d_out == nullptr || d_table == nullptr || d_ids == nullptr || n_rows == 0 || row_elems == 0) return false;
    const unsigned blocks = (row_elems + 255u) / 256u;
    dsv4_row_gather_bf16_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        d_out, d_table, d_ids, src_index, row_elems, dst_index);
    dsv4_fp8_check("deepseek4_row_gather_bf16");
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

bool deepseek4_argmax(int32_t* d_out, const float* d_x, uint32_t n, void* stream) {
    if (d_out == nullptr || d_x == nullptr || n == 0) return false;
    const uint32_t threads = n < 1024u ? 1024u : 1024u;   // one block; the scan strides
    dsv4_argmax_kernel<<<1, threads, 0, static_cast<cudaStream_t>(stream)>>>(d_out, d_x, n);
    dsv4_fp8_check("deepseek4_argmax");
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

bool deepseek4_row_add(float* d_out_row, const float* d_add, uint32_t n, void* stream) {
    if (d_out_row == nullptr || d_add == nullptr || n == 0) return false;
    const unsigned blocks = (n + 255u) / 256u;
    dsv4_row_add_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_out_row, d_add, n);
    dsv4_fp8_check("deepseek4_row_add");
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

}  // namespace strata::kernels
