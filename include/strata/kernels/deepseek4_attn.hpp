// include/strata/kernels/deepseek4_attn.hpp - P3: attention primitives.
//
// The first group is RoPE, which every attention path runs: the q path and the window kv at the layer's base,
// the compressor's pooled key and the indexer's key at the compressor base, and the attention output through
// the inverse rotation.  This is the reference engine's cuda/ds41_attn.cuh:52-94, which in turn is the official
// precompute_freqs_cis + apply_rotary_emb:
//
//   * adjacent elements are one complex number, and only the LAST `n_rot` values of each head rotate
//     (head_dim 512 with n_rot 64 leaves 448 untouched; the indexer key is head_dim 128 with the same 64);
//   * f(i) = theta^(-2i/n_rot), with the YaRN ramp applied when original_seq_len > 0 (compressor layers):
//     low/high corrected dimensions from ln(original_seq_len/(beta*2pi)), a linear ramp between them, and
//     f = f/factor outside the ramp, f inside it; the base is then 160000 rather than 10000;
//   * angle = position * f, and the rotated pair is written with one bf16 round:
//     (a*c - b*s, a*s + b*c), the inverse rotation negating s.
//
// The frequency function is shared between the host oracle and the kernel, so both compute the same f32 angle
// from the same position and only their libm `cosf`/`sinf` can differ.
#pragma once

#include <cmath>
#include <cstdint>

#include "strata/artifact/deepseek4_engram_bits.hpp"

namespace strata::kernels {

/// The reference's v41_rope_freq (cuda/ds41_attn.cuh:52-70).  `rot_dim` is n_rot, not head_dim: the exponent
/// and the YaRN ramp are both over the rotated width.  `osl == 0` disables YaRN.
STRATA_DSV4_HD inline float deepseek4_rope_freq(uint32_t i, uint32_t rot_dim, float theta, uint32_t original_seq_len,
                                                float factor, float beta_fast, float beta_slow) {
    constexpr float kPi = 3.14159265358979323846f;
    float f = 1.0f / powf(theta, (float) (2u * i) / (float) rot_dim);
    if (original_seq_len > 0u) {
        const float lt = 2.0f * logf(theta);
        float low = floorf((float) rot_dim * logf((float) original_seq_len / (beta_fast * 2.0f * kPi)) / lt);
        float high = ceilf((float) rot_dim * logf((float) original_seq_len / (beta_slow * 2.0f * kPi)) / lt);
        low = fmaxf(low, 0.0f);
        high = fminf(high, (float) (rot_dim - 1u));
        float ramp = ((float) i - low) / fmaxf(high - low, 1e-3f);
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        const float smooth = 1.0f - ramp;
        f = f / factor * (1.0f - smooth) + f * smooth;
    }
    return f;
}

/// Rotates the last `n_rot` values of each head in place.  x is [n_tok][n_head][head_dim] f32 (values on the
/// bf16 grid, as every producer of this tensor rounds them), pos is [n_tok].  The rotated pair is bf16-rounded
/// once, matching the reference's write-back into its bf16 tensor.
void deepseek4_rope_host(float* x, const int32_t* pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                         uint32_t n_rot, float theta, uint32_t original_seq_len, float factor, float beta_fast,
                         float beta_slow, bool inverse);

/// The GPU form.  Refuses odd n_rot, n_rot > 128 or n_rot > head_dim (the reference's entry refuses the first
/// two; the third would write outside the head), and any null pointer or empty shape.  n_rot == 0 is a no-op.
bool deepseek4_rope(float* d_x, const int32_t* d_pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                    uint32_t n_rot, float theta, uint32_t original_seq_len, float factor, float beta_fast,
                    float beta_slow, bool inverse, void* stream);

// ---- the quantized-KV primitives ----
//
// The window key, the indexer query and key and both packed caches share one block quant, written once in the
// reference (cuda/ds41_attn.cuh:171-267) because "the packed value equals the f32 it replaces" depends on the
// encode steps being identical, not merely close.  One warp owns one block: a magnitude max over the block,
// one scale, each value rounded and written back as bf16r(q * s).  The mode picks the value range and scale:
//
//   0  fp8 E4M3 (|v| <= 448), scale 2^ceil(log2(amax / 448))   the window kv, block 32
//   1  fp4 E2M1 (|v| <= 6),   scale 2^ceil(log2(amax / 6))     the index query and key, block 32
//   2  fp4 E2M1,              scale E4M3(amax / 6)             the compressed KV, block 16
//
// A packed cache keeps the quantized pair at its native width:
//
//   compressed KV (ckv)  512 dims = 256 B nibbles + 32 B E4M3 scales = 288 B per group
//   index key     (idxk) 128 dims =  64 B nibbles +  4 B E8M0 scales = 68 B, padded to 72
//
// Element e is the low half of byte e/2 when e is even and the high half when odd, and shares scale byte
// e / block.  Reading one back computes bf16r(nibble * scale), the expression the in-place quant evaluates,
// so the packed element is exactly the f32 value it replaced.  The index key's four pad bytes are never read
// (the pad only keeps a warp's 64 B read on one 32 B sector); this port zeroes them so a cache's bytes are
// deterministic.  The E4M3/E8M0 decoders are the engram header's (deepseek4_engram_bits.hpp), which the
// reference's ds4_e4m3fn_to_f32 / ds4_e8m0_to_f32 match value for value.

inline constexpr uint32_t kDeepseek4CkvBlock = 16;         // values per compressed-KV quant block
inline constexpr uint32_t kDeepseek4CkvDim = 512;          // values per compressed-KV group
inline constexpr uint32_t kDeepseek4CkvNibbleBytes = 256;  // 512 nibbles
inline constexpr uint32_t kDeepseek4CkvBytes = 288;        // + 32 E4M3 scale bytes

inline constexpr uint32_t kDeepseek4IdxkBlock = 32;        // values per index-key quant block
inline constexpr uint32_t kDeepseek4IdxkDim = 128;         // values per index-key group
inline constexpr uint32_t kDeepseek4IdxkNibbleBytes = 64;  // 128 nibbles
inline constexpr uint32_t kDeepseek4IdxkBytes = 72;        // 64 + 4 E8M0 scale bytes + 4 pad

inline constexpr int kDeepseek4QuantFp8E8m0 = 0;  // window kv
inline constexpr int kDeepseek4QuantFp4E8m0 = 1;  // index query and index key
inline constexpr int kDeepseek4QuantFp4E4m3 = 2;  // compressed KV

/// 2^ceil(log2 v), the reference's fast_round_scale (cuda/ds41_attn.cuh:34-37; cuda_v41_1.inc.cu:130-133).
STRATA_DSV4_HD inline float deepseek4_pow2_ceil_log2(float v) {
    int e;
    const float m = frexpf(v, &e);  // v = m * 2^e with m in [0.5, 1)
    return ldexpf(1.0f, (m == 0.5f) ? e - 1 : e);
}

/// E4M3FN code `i`'s value (cuda/ds41_fp8blk.cuh:85-94): exp 0 is subnormal man * 2^-9, else (1 + man/8) 2^(exp-7).
STRATA_DSV4_HD inline float deepseek4_e4m3_value(int i) {
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0 ? (float) mant * 0.001953125f
                    : (1.0f + (float) mant * 0.125f) * ldexpf(1.0f, exp - 7);
}

/// Nearest E4M3FN value to x, ties to an even mantissa code (cuda/ds41_fp8blk.cuh:95-109).
STRATA_DSV4_HD inline float deepseek4_e4m3_round(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (deepseek4_e4m3_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - deepseek4_e4m3_value(best));
        const float next_diff = fabsf(ax - deepseek4_e4m3_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0))
            best++;
    }
    return sign * deepseek4_e4m3_value(best);
}

/// E2M1 (fp4 magnitude) table entry `i` in 0..7 (cuda/ds41_fp8blk.cuh:110-113).
STRATA_DSV4_HD inline float deepseek4_e2m1_value(int i) {
    switch (i & 7) {
        case 0: return 0.0f;
        case 1: return 0.5f;
        case 2: return 1.0f;
        case 3: return 1.5f;
        case 4: return 2.0f;
        case 5: return 3.0f;
        case 6: return 4.0f;
        default: return 6.0f;
    }
}

/// Nearest E2M1 value to x (|x| clamped to 6), ties to an even table index (cuda/ds41_fp8blk.cuh:114-127).
STRATA_DSV4_HD inline float deepseek4_e2m1_round(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - deepseek4_e2m1_value(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - deepseek4_e2m1_value(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * deepseek4_e2m1_value(best);
}

/// FP4 nibble (sign in bit 3) -> f32 (cuda/ds41_fp8blk.cuh:128-134; nibble 8 is -0.0).
STRATA_DSV4_HD inline float deepseek4_fp4_from_nibble(uint8_t n) {
    const float v = deepseek4_e2m1_value(n & 7);
    return (n & 8u) ? -v : v;
}

/// f32 -> FP4 nibble (cuda/ds41_fp8blk.cuh:135-143).
STRATA_DSV4_HD inline uint8_t deepseek4_fp4_to_nibble(float x) {
    const float v = deepseek4_e2m1_round(x);
    const float av = fabsf(v);
    uint8_t n = 0;
    for (int i = 1; i < 8; i++) {
        if (av == deepseek4_e2m1_value(i)) { n = (uint8_t) i; break; }   // exact table values
    }
    return n == 0 ? (uint8_t) 0 : (uint8_t)(v < 0.0f ? (n | 8u) : n);
}

/// E4M3 scale byte (cuda/ds41_fp8blk.cuh:146-155).
STRATA_DSV4_HD inline uint8_t deepseek4_e4m3_byte(float x) {
    const float v = deepseek4_e4m3_round(x);
    const float av = fabsf(v);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (deepseek4_e4m3_value(mid) <= av) lo = mid;
        else hi = mid - 1;
    }
    return (uint8_t)((v < 0.0f && av != 0.0f) ? (lo | 0x80) : lo);
}

/// UE8M0 scale byte (cuda/ds41_fp8blk.cuh:144-148); s is a power of two (or 2^-127, stored as 0).
STRATA_DSV4_HD inline uint8_t deepseek4_e8m0_byte(float s) {
    const uint32_t bits = strata::deepseek4::engram_f32_to_bits(s);
    if (bits == 0x00400000u) return 0;
    return (uint8_t)((bits >> 23) & 0xffu);
}

/// The scale one quant block shares; the floors are the smallest value each mode's scale can carry
/// (cuda/ds41_attn.cuh:180-187 and :225-227, one copy).
STRATA_DSV4_HD inline float deepseek4_quant_scale(float absmax, int mode) {
    if (mode == kDeepseek4QuantFp8E8m0)
        return deepseek4_pow2_ceil_log2(fmaxf(absmax, 1e-4f) / 448.0f);
    if (mode == kDeepseek4QuantFp4E8m0)
        return deepseek4_pow2_ceil_log2(fmaxf(absmax, 6.0f * ldexpf(1.0f, -126)) / 6.0f);
    return deepseek4_e4m3_round(fmaxf(absmax, 6.0f * ldexpf(1.0f, -9)) / 6.0f);
}

/// One value through the block quant: bf16r(round(clamp(v / s)) * s), the reference's write-back.
STRATA_DSV4_HD inline float deepseek4_quant_value(float v, float s, int mode) {
    if (mode == kDeepseek4QuantFp8E8m0) {
        const float q = deepseek4_e4m3_round(fminf(fmaxf(v / s, -448.0f), 448.0f));
        return strata::deepseek4::engram_bf16r(q * s);
    }
    const float q = deepseek4_e2m1_round(fminf(fmaxf(v / s, -6.0f), 6.0f));
    return strata::deepseek4::engram_bf16r(q * s);
}

/// Element `d` of one packed compressed-KV row (cuda/ds41_attn.cuh:239-244).
STRATA_DSV4_HD inline float deepseek4_ckv_get(const uint8_t* row, uint32_t d) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return strata::deepseek4::engram_bf16r(
        deepseek4_fp4_from_nibble(nib) *
        strata::deepseek4::engram_e4m3_to_f32(row[kDeepseek4CkvNibbleBytes + (d >> 4)]));
}

/// Element `d` of one packed index-key row (cuda/ds41_attn.cuh:251-256).
STRATA_DSV4_HD inline float deepseek4_idxk_get(const uint8_t* row, uint32_t d) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return strata::deepseek4::engram_bf16r(
        deepseek4_fp4_from_nibble(nib) *
        strata::deepseek4::engram_e8m0_to_f32(row[kDeepseek4IdxkNibbleBytes + (d >> 5)]));
}

/// Compressor pooling (cuda/ds41_attn.cuh:86-104): group g spans ratio consecutive rows; each dimension is a
/// softmax over the group's scores weighting the group's kv values, bf16-rounded once.  Only complete groups
/// are written - floor(n_tok / ratio) rows; the tail is the caller's pending rows.
void deepseek4_compress_pool_host(float* out, const float* kv, const float* score, uint32_t n_tok, uint32_t ratio,
                                  uint32_t dim);

/// One block of 256 threads per complete group; n_tok / ratio == 0 is a successful no-op.
bool deepseek4_compress_pool(float* d_out, const float* d_kv, const float* d_score, uint32_t n_tok, uint32_t ratio,
                             uint32_t dim, void* stream);

/// In-place block quant (cuda/ds41_attn.cuh:171-205): one warp per block, block <= 32 and dim a multiple of it.
void deepseek4_act_quant_host(float* x, uint32_t n_rows, uint32_t dim, uint32_t block, int mode);
bool deepseek4_act_quant(float* d_x, uint32_t n_rows, uint32_t dim, uint32_t block, int mode, void* stream);

/// Pack `n_rows` f32 rows into groups [g0, g0 + n_rows) of the cache (cuda/ds41_attn.cuh:207-284).  Only those
/// rows are touched; the caller allocates at least g0 + n_rows rows.
void deepseek4_ckv_pack_host(uint8_t* cache, uint32_t g0, const float* rows, uint32_t n_rows);
bool deepseek4_ckv_pack(uint8_t* d_cache, uint32_t g0, const float* d_rows, uint32_t n_rows, void* stream);
void deepseek4_idxk_pack_host(uint8_t* cache, uint32_t g0, const float* rows, uint32_t n_rows);
bool deepseek4_idxk_pack(uint8_t* d_cache, uint32_t g0, const float* d_rows, uint32_t n_rows, void* stream);

/// The unpack direction over rows [0, n_rows), for the gate and any f32 consumer of a packed cache.
void deepseek4_ckv_unpack_host(const uint8_t* cache, uint32_t n_rows, float* out);
bool deepseek4_ckv_unpack(const uint8_t* d_cache, uint32_t n_rows, float* d_out, void* stream);
void deepseek4_idxk_unpack_host(const uint8_t* cache, uint32_t n_rows, float* out);
bool deepseek4_idxk_unpack(const uint8_t* d_cache, uint32_t n_rows, float* d_out, void* stream);

// ---- sparse attention ----
//
// The layer's attention over the causal SWA window plus the topk compressed keys (cuda/ds41_attn.cuh:304-370).
// One block per (query, 8-head group), 128 threads = 4 warps x 2 heads, lane l owning dims l*per ..
// l*per+per-1 of a head.  The key order is the window rows in ascending position, then the topk compressed
// rows, 8 keys at a time through shared.  The online softmax keeps a running (max, sum, acc): each tile takes
// its max, one rescale of the running values, and every probability is bf16-rounded before it multiplies a key
// (the official acc_s_cast).  An invalid topk entry - negative, or >= ng - scores -1e30 and its shared row is
// zeroed, because 0 times stale shared bits is not 0 when those bits are NaN.  The sink enters the denominator
// only: out = bf16r(acc / (sum + exp(sink[h] - mx))).
//
// The scalar form is the reference engine's own fallback, kept here as P3's correctness path; its tensor-core
// and split-K routes (cuda/ds41_attn_mma.cuh) are P7's performance work and must reproduce this arithmetic.

inline constexpr uint32_t kDeepseek4AttnHeadsPerBlock = 8;  // 4 warps x 2 heads
inline constexpr uint32_t kDeepseek4AttnKeyTile = 8;        // keys staged through shared per step

/// Which row of the SWA window buffer absolute position `a` occupies (cuda/ds41_attn.cuh:39-50): [0, window)
/// is the history segment, [window, window + n) this batch's rows.  ring=1 keeps a % window; ring=0 (the
/// DSpark tower) is the linear segment its positions live in.
STRATA_DSV4_HD inline uint64_t deepseek4_win_row(int64_t a, uint32_t pos0, uint32_t window, uint32_t ring) {
    if (a >= (int64_t) pos0) return (uint64_t)((int64_t) window + a - (int64_t) pos0);
    return ring ? (uint64_t)(a % (int64_t) window)
                : (uint64_t)(a - ((int64_t) pos0 - (int64_t) window));
}

/// The host oracle: the reference kernel's gather, tile order, reduction tree and online softmax, written for
/// one thread.  `kvw` has window + n_tok rows of head_dim f32; the compressed entries are the header's packed
/// rows.  `full_block` is 0 for the causal main path, else the draft block's length (all its positions visible
/// to every query).  When `kvc` is null `topk` is treated as 0.
void deepseek4_sparse_attn_host(float* o, const float* q, const float* kvw, const uint8_t* kvc, const int32_t* idx,
                                const float* sink, uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng,
                                uint32_t topk, uint32_t n_head, uint32_t head_dim, float scale, uint32_t full_block,
                                bool ring, uint32_t win_lo);

/// The GPU form.  head_dim must be a multiple of 32 and at most 512 (the shared tile's width), n_head a
/// multiple of 8; kvc and idx must be both null or both present.  `win_lo` clamps the window's lower bound
/// (ring cells before it were never written); 0 disables it.
bool deepseek4_sparse_attn(float* d_o, const float* d_q, const float* d_kv_win, const uint8_t* d_kv_comp,
                           const int32_t* d_idx, const float* d_sink, uint32_t n_tok, uint32_t pos0, uint32_t window,
                           uint32_t ng, uint32_t topk, uint32_t n_head, uint32_t head_dim, float scale,
                           uint32_t full_block, bool ring, uint32_t win_lo, void* stream);

// ---- SWA window ring commit ----
//
// The batch's rows sit in the window buffer at [window, window + n) until committed; committing moves row i to
// ring cell (pos0 + i) % window (cuda/ds41_attn.cuh:382-405).  Only the last window rows move when n > window:
// ring cells alias at +window, so the ascending write's net result is exactly "the last window rows in place",
// and the surviving destinations are distinct, which makes the block order irrelevant.  `pos`, when non-null,
// overrides pos0 with pos[0] (the graph route's dynamic position); that route takes n <= 8 and is refused
// otherwise.  The caller allocates at least (window + n) * head_dim floats in win.

void deepseek4_win_commit_host(float* win, uint32_t pos0, uint32_t n, uint32_t window, uint32_t head_dim,
                               const int32_t* pos);
bool deepseek4_win_commit(float* d_win, uint32_t pos0, uint32_t n, uint32_t window, uint32_t head_dim,
                          const int32_t* d_pos, void* stream);

// The spec-decode snapshot (cuda/ds41_attn.cuh:407-425): save (back=false) copies ring cell
// (pos0 + i) % window into snap[i]; restore (back=true) writes it back.  Only rows [i0, n) are touched, snap is
// indexed by batch row and holds n rows of head_dim.  n <= i0 is a successful no-op.
void deepseek4_win_ring_snap_host(float* win, float* snap, uint32_t pos0, uint32_t i0, uint32_t n, uint32_t window,
                                  uint32_t head_dim, bool back, const int32_t* pos);
bool deepseek4_win_ring_snap(float* d_win, float* d_snap, uint32_t pos0, uint32_t i0, uint32_t n, uint32_t window,
                             uint32_t head_dim, bool back, const int32_t* d_pos, void* stream);

// ---- sparse indexer ----
//
// Three pieces from cuda/ds41_indexer.cuh: the score, the candidate-block list and the topk.  The score is
// score[i][g] = bf16(sum_h bf16(relu(bf16(q_h . k_g)) * w[i][h])) with invisible groups at -inf, a warp
// advancing four groups at once (one group per warp would be bit-identical: the MAC order, the xor tree, the
// bf16 rounding points and the head order are the same).  The candidate selector keeps the block holding the
// query's newest position (score +inf) and the top-scoring blocks; the topk selects min(index_topk, visible)
// keys in position order with ties taking the smaller index and unreachable slots -1.
//
// The optional candidate list is the C2 contract (ds4_gpu_v41.h): row i is [count][ascending block numbers],
// stride 1 + cand_cap; with a list the score kernel writes a compact row of
// deepseek4_cand_ns(ng, bs, cap) = min(cap * bs, ng) entries whose item c maps back to block cl[1 + c/bs]
// times bs plus c % bs, and items past the list are dead (score -inf).  The list is ascending, so compact
// order is position order and the tie rule is unchanged.
//
// The float -> key mapping is the reference's order-preserving u32 (`-inf` and NaN map to 0, so they are
// never selected; positives set the top bit, negatives are complemented).

inline constexpr uint32_t kDeepseek4IdxGroupsPerWarp = 4;  // cuda/ds41_indexer.cuh:36
inline constexpr uint32_t kDeepseek4TopkThreads = 1024;    // the reference's topk block size
inline constexpr uint32_t kDeepseek4TopkBatch = 8;         // scores read per thread per pass

STRATA_DSV4_HD inline uint32_t deepseek4_cand_ns(uint32_t ng, uint32_t cand_bs, uint32_t cand_cap) {
    const uint64_t full = (uint64_t) cand_cap * cand_bs;
    return full < ng ? (uint32_t) full : ng;
}

STRATA_DSV4_HD inline uint32_t deepseek4_topk_key(float f) {
    if (!(f > -INFINITY)) return 0u;
    const uint32_t u = strata::deepseek4::engram_f32_to_bits(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// score[i][g] over the whole key cache (cand null) or the compact row (cand present); `dk` a multiple of 32
/// and at most 128, the packed index keys are the header's 72 B rows.  `pos`, when non-null, replaces pos0 with
/// pos[0] and recomputes ng = (pos0 + n_tok) / ratio (the captured-graph convention, n_tok <= 8); ng == 0
/// succeeds without touching the output.
void deepseek4_indexer_score_host(float* score, const float* q, const uint8_t* idxk, const float* w,
                                  const int32_t* cand, uint32_t pos0, uint32_t ng, uint32_t n_tok, uint32_t n_head,
                                  uint32_t dk, uint32_t ratio, uint32_t cand_bs, uint32_t cand_cap,
                                  const int32_t* pos);
bool deepseek4_indexer_score(float* d_score, const float* d_q, const uint8_t* d_idxk, const float* d_w,
                             const int32_t* d_cand, uint32_t pos0, uint32_t ng, uint32_t n_tok, uint32_t n_head,
                             uint32_t dk, uint32_t ratio, uint32_t cand_bs, uint32_t cand_cap, const int32_t* d_pos,
                             void* stream);

/// The candidate list, rows of 1 + topk_blocks int32.  `scratch` holds n_tok * nb_cap floats followed by
/// n_tok * nb_cap bytes (the block scores and the flags), nb_cap >= ceil(ng / block_size).
void deepseek4_candidate_blocks_host(int32_t* list, const float* score, uint32_t n_tok, uint32_t pos0, uint32_t ng,
                                     uint32_t ratio, uint32_t topk_blocks, uint32_t block_size,
                                     const int32_t* pos);
bool deepseek4_candidate_blocks(int32_t* d_list, const float* d_score, float* d_scratch, uint32_t n_tok,
                                uint32_t pos0, uint32_t ng, uint32_t ratio, uint32_t topk_blocks,
                                uint32_t block_size, uint32_t nb_cap, const int32_t* d_pos, void* stream);

/// The selected groups, rows of `topk` int32 (the row stride is the topk the call actually uses, which the
/// device position may shrink in the graph convention).  With a list the scan covers the compact row.
void deepseek4_indexer_topk_host(int32_t* idx, const float* score, uint32_t n_tok, uint32_t ng, uint32_t topk,
                                 uint32_t ratio, const int32_t* cand, uint32_t cand_bs, uint32_t cand_cap,
                                 const int32_t* pos);
bool deepseek4_indexer_topk(int32_t* d_idx, const float* d_score, uint32_t n_tok, uint32_t ng, uint32_t topk,
                            uint32_t ratio, const int32_t* d_cand, uint32_t cand_bs, uint32_t cand_cap,
                            const int32_t* d_pos, void* stream);

}  // namespace strata::kernels
