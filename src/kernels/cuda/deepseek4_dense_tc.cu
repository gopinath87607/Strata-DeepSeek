// src/kernels/cuda/deepseek4_dense_tc.cu - the dense matmuls' bf16 tensor-core tiles (--tc-gemm, opt-in).
//
// The q4_K / grouped wo_a / bf16 batched matmuls of deepseek4_dense.cu run on SIMT fmaf and reach a few
// percent of the cards' fp32 peak.  These tiles feed the bf16 tensor cores instead: the MoE prefill arm's
// validated shape (128 weight rows x 128 tokens, 16 warps, 64 K columns a round, ldmatrix + m16n8k16), with
// the A tile built by decoding q4_K nibbles to bf16 in registers (AT 0) or by copying bf16 weights (AT 1),
// and the B tile by converting the f32 activations on the way into shared.  The four k16 groups of a round
// meet in an f32 accumulator; the epilogue applies the same bf16 round the SIMT path does, so a run's ids
// stay on the same bf16 grid (small f32 rounding differences apart).
//
// The mma.sync.bf16 needs sm_80 or newer, above the floor the rest of the tree is compiled for, so this
// file is its own static library with the prefill arm's arch set (SASS for the 86 and 120 cards the port is
// measured on, PTX for the arches between).  Off by default: deepseek4_dense.cu keeps its SIMT tiles.
#include "strata/kernels/deepseek4_dense.hpp"
#include "strata/kernels/deepseek4_dense_tc.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cstdio>
#include <cstdint>

namespace strata {
namespace kernels {
namespace {

// ---- tensor-core tile (--tc-gemm, opt-in; its own library, like the prefill arm) --------------------
//
// The SIMT tiles above run a few percent of the cards' fp32 peak: every product is an fmaf on a dequantized
// value.  The bf16 tensor cores are where the missing throughput is, and this tile feeds them: the MoE
// prefill arm's validated shape (128 weight rows x 128 tokens, 16 warps, 64 K columns a round, ldmatrix +
// m16n8k16), with the A tile built by decoding q4_K nibbles to bf16 in registers (AT 0) or by copying bf16
// weights (AT 1), and the B tile by converting the f32 activations on the way into shared.  The four k16
// groups of a round meet in the accumulator; nothing else changes, so a run's ids stay on the same bf16
// grid the SIMT path rounds onto.  Off by default; the dispatch keeps the kernels above.
constexpr uint32_t kTcRows = 128u;   // weight rows a block owns
constexpr uint32_t kTcTok = 128u;    // tokens per block tile
constexpr uint32_t kTcK = 64u;       // K columns per round (a quarter of a q4_K block)
constexpr uint32_t kTcThreads = 512u;
constexpr uint32_t kTcSlices = kTcTok / 8u;          // 16 n8 slices
constexpr uint32_t kTcSlicesPerWarp = kTcSlices / 2u;   // 8, one token half per warp group
constexpr uint32_t kTcABytes = kTcRows * kTcK * 2u;    // 16 KiB
constexpr uint32_t kTcBBytes = kTcTok * kTcK * 2u;     // 16 KiB

__device__ __forceinline__ void tc_ldsm4(uint32_t* r, const uint8_t* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void tc_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A tile row is 64 bf16 = 8 chunks of 16 B; chunk ^ (row & 7) puts the 8 rows one ldmatrix reads into 8
// different bank groups.
__device__ __forceinline__ uint32_t tc_swz(uint32_t row, uint32_t chunk) {
    return row * 128u + ((chunk ^ (row & 7u)) << 4);
}

__device__ __forceinline__ uint16_t tc_bf16_hi(float v) {
    return static_cast<uint16_t>(__float_as_uint(deepseek4_dense_bf16r(v)) >> 16);
}

// GROUPED selects the wo_a layout exactly as the SIMT tiles do.  AT 0 decodes q4_K weights; AT 1 copies bf16
// weights.  Every tile is 128x128 with bounds checks in the B fill, the MMA loop and the epilogue, so a
// ragged chunk edge costs only the skipped slices.
template <bool GROUPED, int AT>
__global__ __launch_bounds__(kTcThreads, 1) void dense_matmul_tc_kernel(
        float* __restrict__ out, const uint8_t* __restrict__ w, const float* __restrict__ x, uint32_t blocks,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, bool round_out) {
    __shared__ __align__(16) uint8_t tcsh[kTcABytes + kTcBBytes];
    uint8_t* As = tcsh;
    uint8_t* Bs = tcsh + kTcABytes;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t wr = warp & 7u, th = warp >> 3u;   // 8 row warps x 2 token halves
    const uint32_t ar = tid / 4u, sub = tid & 3u;     // 4 threads a row, 16 cols each
    const uint32_t g = GROUPED ? blockIdx.z : 0u;
    const float* xg = GROUPED ? x + (size_t) g * in_dim : x;
    float* yg = GROUPED ? out + (size_t) g * out_dim : out;
    const size_t xr = GROUPED ? (size_t) in_dim * gridDim.z : (size_t) in_dim;
    const size_t yr = GROUPED ? (size_t) out_dim * gridDim.z : (size_t) out_dim;
    const size_t wrow = (size_t) (AT ? in_dim * 2u : blocks * kDeepseek4Q4kBytes);   // bf16 rows are 2 B/value
    w += (size_t) g * out_dim * wrow;
    const uint32_t r0 = blockIdx.x * kTcRows;
    const uint32_t t0 = blockIdx.y * kTcTok;
    const uint32_t nt = t0 < n_tok ? n_tok - t0 : 0u;   // this tile's real token count
    const uint32_t nt_max = (nt + 7u) & ~7u;            // slices past this token are never read
    const uint32_t grow = r0 + ar;
    const bool rv = grow < out_dim;
    const uint8_t* rowp = w + (size_t) (rv ? grow : 0u) * wrow;
    const uint32_t nit = in_dim / kTcK;           // rounds
    float acc[kTcSlicesPerWarp][4];
#pragma unroll
    for (uint32_t f = 0; f < kTcSlicesPerWarp; f++) acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f;
    for (uint32_t i = 0; i < nit; i++) {
        // ---- the A tile: 64 K columns of 128 weight rows ----
        if (AT == 0) {
            const uint8_t* blk = rowp + (size_t) (i >> 2) * kDeepseek4Q4kBytes;
            const uint32_t c = i & 3u;
            const float d = f32_from_f16((uint16_t) (blk[0] | (blk[1] << 8)));
            const float dmin = f32_from_f16((uint16_t) (blk[2] | (blk[3] << 8)));
            const uint32_t gi = 2u * c + (sub >> 1u);   // this thread's scale group
            const uint32_t hb = sub >> 1u;              // its nibble: low for an even group, high for an odd one
            uint32_t sc = 0, m = 0;
            deepseek4_q4k_scale_min(gi, blk + 4, sc, m);
            const float a = d * (float) sc, b = dmin * (float) m;
            // ggml packs one value a byte, not two: this thread's 16 values are 16 bytes, and the group's
            // nibble picks each.  Reading both nibbles of 8 bytes would write the neighbouring group's
            // values into this one (deepseek4_q4k_element is the reference layout).
            const uint8_t* qp = blk + 16u + 32u * c + (sub & 1u) * 16u;
            if (rv) {
                const uint4 qb = __ldg(reinterpret_cast<const uint4*>(qp));
                const uint32_t qw[4] = {qb.x, qb.y, qb.z, qb.w};
#pragma unroll
                for (uint32_t wq = 0; wq < 2; wq++) {
                    uint32_t cw[4];
#pragma unroll
                    for (uint32_t j = 0; j < 2u; j++) {
                        const uint32_t wv = qw[2u * wq + j];
                        uint32_t hw[2] = {0u, 0u};
#pragma unroll
                        for (uint32_t k = 0; k < 4u; k++) {
                            const uint32_t byt = (wv >> (8u * k)) & 255u;
                            const uint32_t nib = hb ? (byt >> 4) : (byt & 15u);
                            hw[k >> 1] |= (uint32_t) tc_bf16_hi(a * (float) nib - b) << (16u * (k & 1u));
                        }
                        cw[2u * j] = hw[0];
                        cw[2u * j + 1u] = hw[1];
                    }
                    *reinterpret_cast<uint4*>(As + tc_swz(ar, 2u * sub + wq)) =
                        make_uint4(cw[0], cw[1], cw[2], cw[3]);
                }
            } else {
                *reinterpret_cast<uint4*>(As + tc_swz(ar, 2u * sub)) = make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(As + tc_swz(ar, 2u * sub + 1u)) = make_uint4(0u, 0u, 0u, 0u);
            }
        } else {
#pragma unroll
            for (uint32_t wq = 0; wq < 2; wq++) {
                const uint32_t ch = 2u * sub + wq;
                if (rv)
                    *reinterpret_cast<uint4*>(As + tc_swz(ar, ch)) =
                        __ldg(reinterpret_cast<const uint4*>(rowp + (size_t) i * 2u * kTcK + ch * 16u));
                else
                    *reinterpret_cast<uint4*>(As + tc_swz(ar, ch)) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        // ---- the B tile: 64 K columns of 128 activation rows, f32 -> bf16 on the way in ----
        // Each uint4 is an 8-value chunk and a token row is kTcK/8 chunks, so 512 threads take two passes
        // (the prefill arm's (kPfTok * kPfCw) / kPfThreads, kPfCw = 8).
#pragma unroll
        for (uint32_t c = 0; c < (kTcTok * (kTcK / 8u)) / kTcThreads; c++) {
            const uint32_t ch = tid + c * kTcThreads, bt = ch >> 3, bc = ch & 7u;
            if (bt < nt) {
                const float* xs = xg + (size_t) (t0 + bt) * xr + (size_t) i * kTcK + bc * 8u;
                uint32_t r[4];
#pragma unroll
                for (uint32_t j = 0; j < 4; j++) {
                    const uint32_t h0 = tc_bf16_hi(xs[2u * j]);
                    const uint32_t h1 = tc_bf16_hi(xs[2u * j + 1u]);
                    r[j] = (h1 << 16) | h0;
                }
                *reinterpret_cast<uint4*>(Bs + tc_swz(bt, bc)) =
                    make_uint4(r[0], r[1], r[2], r[3]);
            }
        }
        __syncthreads();
#pragma unroll
        for (uint32_t kk = 0; kk < kTcK / 16u; kk++) {
            uint32_t a[4];
            {
                const uint32_t mt = lane >> 3u, row = wr * 16u + (lane & 7u) + (mt & 1u) * 8u;
                tc_ldsm4(a, As + tc_swz(row, kk * 2u + (mt >> 1)));
            }
#pragma unroll
            for (uint32_t np = 0; np < kTcSlicesPerWarp / 2u; np++) {
                const uint32_t f0 = th * kTcSlicesPerWarp + 2u * np;   // this warp's 2np-th slice
                if (f0 * 8u >= nt_max) break;
                uint32_t b[4];
                const uint32_t mt = lane >> 3u, tok = (f0 + (mt >> 1)) * 8u + (lane & 7u);
                tc_ldsm4(b, Bs + tc_swz(tok, kk * 2u + (mt & 1u)));
                float t0[4] = {0.f, 0.f, 0.f, 0.f}, t1[4] = {0.f, 0.f, 0.f, 0.f};
                tc_mma(t0, a, b[0], b[1]);
#pragma unroll
                for (int c = 0; c < 4; c++) acc[2u * np][c] += t0[c];
                if ((f0 + 1u) * 8u < nt_max) {
                    tc_mma(t1, a, b[2], b[3]);
#pragma unroll
                    for (int c = 0; c < 4; c++) acc[2u * np + 1u][c] += t1[c];
                }
            }
        }
        __syncthreads();
    }
    // exit: c0,c1 = row lane/4 at tokens (lane&3)*2+{0,1}; c2,c3 = row +8.
    const uint32_t rr[2] = {r0 + wr * 16u + (lane >> 2u), r0 + wr * 16u + (lane >> 2u) + 8u};
#pragma unroll
    for (uint32_t h = 0; h < 2u; h++) {
        const uint32_t r = rr[h];
        if (r >= out_dim) continue;
#pragma unroll
        for (uint32_t f = 0; f < kTcSlicesPerWarp; f++) {
            const uint32_t gf = th * kTcSlicesPerWarp + f;
            if (gf * 8u >= nt_max) break;
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t t = gf * 8u + (lane & 3u) * 2u + e;
                if (t >= nt) continue;
                const float v = round_out ? deepseek4_dense_bf16r(acc[f][h * 2u + e]) : acc[f][h * 2u + e];
                yg[(size_t) (t0 + t) * yr + r] = v;
            }
        }
    }
}

bool finish(cudaStream_t stream, const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "deepseek4_dense_tc: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    if (stream == nullptr) {
        const cudaError_t s = cudaDeviceSynchronize();
        if (s != cudaSuccess) {
            std::fprintf(stderr, "deepseek4_dense_tc: %s (sync): %s\n", what, cudaGetErrorString(s));
            return false;
        }
    }
    return true;
}

}  // namespace

bool deepseek4_matmul_tc_bf16(float* d_out, const uint16_t* d_w, const float* d_x, uint32_t in_dim,
                              uint32_t out_dim, uint32_t n_tok, bool round_out, void* stream) {
    dense_matmul_tc_kernel<false, 1>
        <<<dim3((out_dim + kTcRows - 1) / kTcRows, (n_tok + kTcTok - 1) / kTcTok), kTcThreads, 0,
           static_cast<cudaStream_t>(stream)>>>(d_out, reinterpret_cast<const uint8_t*>(d_w), d_x,
                                                in_dim / kDeepseek4Q4kBlock, in_dim, out_dim, n_tok, round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_bf16 tc tile");
}

bool deepseek4_matmul_tc_q4k(float* d_out, const void* d_w, const float* d_x, uint32_t in_dim, uint32_t out_dim,
                             uint32_t n_tok, bool round_out, void* stream) {
    dense_matmul_tc_kernel<false, 0>
        <<<dim3((out_dim + kTcRows - 1) / kTcRows, (n_tok + kTcTok - 1) / kTcTok), kTcThreads, 0,
           static_cast<cudaStream_t>(stream)>>>(d_out, static_cast<const uint8_t*>(d_w), d_x,
                                                in_dim / kDeepseek4Q4kBlock, in_dim, out_dim, n_tok, round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_q4k tc tile");
}

bool deepseek4_matmul_tc_q4k_grouped(float* d_out, const void* d_w, const float* d_x, uint32_t n_groups,
                                     uint32_t group_dim, uint32_t rank, uint32_t n_tok, bool round_out,
                                     void* stream) {
    dense_matmul_tc_kernel<true, 0>
        <<<dim3((rank + kTcRows - 1) / kTcRows, (n_tok + kTcTok - 1) / kTcTok, n_groups), kTcThreads, 0,
           static_cast<cudaStream_t>(stream)>>>(d_out, static_cast<const uint8_t*>(d_w), d_x,
                                                group_dim / kDeepseek4Q4kBlock, group_dim, rank, n_tok, round_out);
    return finish(static_cast<cudaStream_t>(stream), "matmul_q4k_grouped tc tile");
}

}  // namespace kernels
}  // namespace strata
