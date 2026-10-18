// include/strata/kernels/deepseek4_dense.hpp - P3: the trunk's dense primitives.
//
// Everything the attention and MoE groups stand on that is not a matrix product of VQ experts or block-fp8:
// the RMS norm with the bf16 round, the plain add / bf16 round / scale-and-round, the embedding row fetch, the
// clamped SwiGLU, and the three skeleton matmuls the artifact's trunk uses (f32, bf16, q4_K) with the grouped
// q4_K form for the attention output projection.
//
// The semantics are the reference engine's (ds4-dfm-rs cuda/ds41_dense.cuh, cuda/ds41_q4k.cuh and
// cuda/ds41_router.cuh) and the host functions below are both the specification and the gate's oracle:
//
//   rms_norm   out[t][d] = bf16r(w[d] * (x[t][d] * rsqrt(mean_d(x^2) + eps)))           (f32, weight on disk)
//   swiglu     up = clamp(up, -limit, +limit); gate = min(gate, +limit);
//              h = bf16r((gate / (1 + exp(-gate))) * up)                                 (official Expert.forward)
//   add        a[i] += b[i]
//   round      x[i] = bf16r(x[i])
//   scale      x[i] = bf16r(x[i] * s)                                                   (indexer's weight scale)
//   expand_hc  hc[t][c][d] = x[t][d] for every route c                                   (embedding -> lanes)
//   embed      out[t][d] = the q4_K row of token t, decoded to f32, unrounded
//   matmul     out[t][o] = [round_out ? bf16r : identity](sum_i w[o][i] * x[t][i])
//
// The q4_K decoder is the ggml block layout (144 B / 256 values) that src/kernels/cuda/dequant_bf16.cu carries
// and the engine's own `ds4_deq_q4_K` golden defines; it lives here as one __host__ __device__ function so the
// host oracle and the kernels cannot disagree about a header.  The gate also checks it against `dequant_bf16`,
// the tree's independently validated decoder.
//
// Every device entry takes device pointers, f32 activations and an optional stream (null = default stream, the
// call synchronizes); a bad shape or a launch error returns false.  The matmuls are row-per-block correctness
// kernels: the reference's warp/ksplit GEMVs and its dequant-to-bf16 GEMM are P7's performance work.
#pragma once

#include <cstdint>

#include "strata/artifact/deepseek4_engram_bits.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

namespace strata::kernels {

inline constexpr uint32_t kDeepseek4Q4kBlock = 256;   // values per q4_K block
inline constexpr uint32_t kDeepseek4Q4kBytes = 144;   // bytes per q4_K block

// ---- q4_K scalar decode (one block, one element) ----
// ggml's 6-bit (scale, min) table, expression for expression the decoder in dequant_bf16.cu:
// the first four entries are the low six bits of bytes 4..7 and 8..11, the high four split a nibble and a
// two-bit suffix across the two halves.
STRATA_DSV4_HD inline void deepseek4_q4k_scale_min(uint32_t j, const uint8_t* q, uint32_t& sc, uint32_t& m) {
    if (j < 4) {
        sc = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        sc = (uint32_t) (q[j + 4] & 0xFu) | ((uint32_t) (q[j - 4] >> 6) << 4);
        m = (uint32_t) (q[j + 4] >> 4) | ((uint32_t) (q[j] >> 6) << 4);
    }
}

/// Element `e` of one 256-value q4_K block: d * sc * nibble - dmin * m, the reference's order of operations
/// (`(d * s) * nib - dmin * m`), with d and dmin read as fp16.
STRATA_DSV4_HD inline float deepseek4_q4k_element(const uint8_t* blk, uint32_t e) {
    const uint32_t gi = e >> 5, l = e & 31u, j64 = gi >> 1, hi = gi & 1u;
    const float d = f32_from_f16((uint16_t) (blk[0] | (blk[1] << 8)));
    const float dmin = f32_from_f16((uint16_t) (blk[2] | (blk[3] << 8)));
    uint32_t sc = 0, m = 0;
    deepseek4_q4k_scale_min(gi, blk + 4, sc, m);
    const uint8_t byte = blk[16 + 32 * j64 + l];
    const uint32_t nib = hi ? (uint32_t) (byte >> 4) : (uint32_t) (byte & 0xFu);
    return (d * (float) sc) * (float) nib - dmin * (float) m;
}

/// bf16 round-to-nearest-even, the engine's v41_bf16r (shared with the HC and engram modules).
STRATA_DSV4_HD inline float deepseek4_dense_bf16r(float x) { return deepseek4::engram_bf16r(x); }

// ---- host oracles (the gate's specification) ----

/// out[t][d] = bf16r(w[d] * (x[t][d] * rsqrt(mean_d(x[t][d]^2) + eps))); x and out are [n_tok][dim].
void deepseek4_rms_norm_host(float* out, const float* x, const float* w, uint32_t dim, uint32_t n_tok, float eps);

/// h[i] = bf16r(clamp_up(gate_i) * silu(min(gate_i, limit))) over n values; limit <= 0 disables the clamp.
void deepseek4_swiglu_host(float* h, const float* gate, const float* up, uint64_t n, float limit);

/// Decodes one q4_K row (`dim` values, a multiple of 256) to f32.
void deepseek4_q4k_row_host(const uint8_t* row_blocks, uint32_t dim, float* out);

/// out[t][d] = the decoded q4_K row of tok[t], f32 and unrounded.  An out-of-range id writes a zero row: the
/// reference returns without writing (and a valid model never produces one), so this is our determinism choice,
/// made the same way by the oracle and the kernel so the gate can compare them.
void deepseek4_embed_q4k_host(float* out, const int32_t* tok, const uint8_t* w, uint32_t n_vocab, uint32_t n_tok,
                              uint32_t dim);

/// out[t][o] = sum_i w[o][i] * x[t][i], rounded to the bf16 grid when round_out.  w is [out_dim][in_dim].
void deepseek4_matmul_f32_host(float* out, const float* w, const float* x, uint32_t in_dim, uint32_t out_dim,
                               uint32_t n_tok, bool round_out);
void deepseek4_matmul_bf16_host(float* out, const uint16_t* w, const float* x, uint32_t in_dim, uint32_t out_dim,
                                uint32_t n_tok, bool round_out);
void deepseek4_matmul_q4k_host(float* out, const uint8_t* w, const float* x, uint32_t in_dim, uint32_t out_dim,
                               uint32_t n_tok, bool round_out);

/// Grouped: out[t][g*rank + o] = sum_c w_g[o][c] * x[t][g*group_dim + c]; the weight is n_groups row-major
/// [rank][group_dim] matrices stored back to back (the reference's wo_a layout: [HD*NG, rank*NG] on disk).
void deepseek4_matmul_q4k_grouped_host(float* out, const uint8_t* w, const float* x, uint32_t n_groups,
                                       uint32_t group_dim, uint32_t rank, uint32_t n_tok, bool round_out);

// ---- device entry points ----

bool deepseek4_rms_norm(float* d_out, const float* d_x, const float* d_w, uint32_t dim, uint32_t n_tok, float eps,
                        void* stream);
bool deepseek4_add(float* d_a, const float* d_b, uint64_t n, void* stream);
bool deepseek4_round_bf16(float* d_x, uint64_t n, void* stream);
bool deepseek4_scale_round(float* d_x, uint64_t n, float s, void* stream);
bool deepseek4_expand_hc(float* d_hc, const float* d_x, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok,
                         void* stream);
bool deepseek4_swiglu(float* d_h, const float* d_gate, const float* d_up, uint64_t n, float limit, void* stream);

bool deepseek4_embed_q4k(float* d_out, const int32_t* d_tokens, const void* d_w, uint32_t n_vocab, uint32_t n_tok,
                         uint32_t dim, void* stream);

bool deepseek4_matmul_f32(float* d_out, const float* d_w, const float* d_x, uint32_t in_dim, uint32_t out_dim,
                          uint32_t n_tok, bool round_out, void* stream);
bool deepseek4_matmul_bf16(float* d_out, const uint16_t* d_w, const float* d_x, uint32_t in_dim, uint32_t out_dim,
                           uint32_t n_tok, bool round_out, void* stream);
bool deepseek4_matmul_q4k(float* d_out, const void* d_w, const float* d_x, uint32_t in_dim, uint32_t out_dim,
                          uint32_t n_tok, bool round_out, void* stream);
bool deepseek4_matmul_q4k_grouped(float* d_out, const void* d_w, const float* d_x, uint32_t n_groups,
                                  uint32_t group_dim, uint32_t rank, uint32_t n_tok, bool round_out, void* stream);

}  // namespace strata::kernels
