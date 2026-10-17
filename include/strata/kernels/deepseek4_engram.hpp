// include/strata/kernels/deepseek4_engram.hpp - P4: the engram's device half (rows, gate, wkv).
//
// The format and the host oracle are in `strata/artifact/deepseek4_engram.hpp`; this is the GPU side that the
// forward wires.  Three pieces, in the order the forward calls them:
//
//   rows       a token's 24 random table rows (fp8 weights + one UE8M0 scale per 32) -> the bf16 grid.  The
//              device kernel must equal `deepseek4::engram_rows_f32` BYTE FOR BYTE: every product in it is
//              exact, so a mismatch is a conversion or layout bug, never accumulation drift.  Both sides call
//              the same conversions (`strata/artifact/deepseek4_engram_bits.hpp`), not two copies.
//
//   wkv        the layer's fp8_32x32 projection [out_dim][in_dim] (E4M3 plane then a UE8M0 scale per 32x32
//              tile, tile (r/32, c/32)) against the dequantized rows: y = W x.  One warp per output row reads
//              16 consecutive weights per step (one 128-bit load, one scale lookup); n > 1 rounds the
//              activations to bf16 first, the engine's activation knife, so both the single-token and the
//              batched paths are the reference engine's (ds41_fp8blk.cuh).
//
//   gate       per (token, hyper-connection row): three reductions over the row (h^2, key^2, h*(qw*kw)*key),
//              then h[d] = bf16r(h[d] + sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)) * val[d]).  One block
//              of 256 threads per (token, row), three tree reductions in shared memory - the reference's
//              kernel (cuda_v41_3.inc.cu:109-141).  The device runs f32 lane/tree sums where the oracle runs
//              f64, so the P4 compare is a bound, not equality: one bf16 grid step at the output's magnitude
//              plus 1e-5 of the terms being summed (the absorb case cancels below that and must be exact).
//
// WHY THE WKV ENTRY TAKES THE BF16 COPY AS AN ARGUMENT.  The reference engine keeps a persistent scratch and
// converts x on the same stream; a library allocating (or holding a process-wide buffer) would either break
// graph capture or race the server's concurrent requests.  Here the caller owns the buffer:
// `deepseek4_engram_x_to_bf16` fills it, and `deepseek4_engram_wkv` requires it when n_tok > 1 and never
// allocates.  n_tok == 1 reads the f32 x directly (the reference's byte-bound single-token arm).
//
// A null `stream` launches on the default stream and synchronizes, the same convention as the VQ entries.
#pragma once

#include <cstdint>

#include "strata/artifact/deepseek4_engram_bits.hpp"

namespace strata::kernels {

/// `n_rows` raw rows (stride head_dim + head_dim/32, weights then scales) -> `n_rows * head_dim` f32 on the
/// bf16 grid.  Bit-identical to strata::deepseek4::engram_rows_f32.  Returns false, launching nothing, when a
/// pointer is null, n_rows is 0, or head_dim is not a positive multiple of 32.
bool deepseek4_engram_rows(const uint8_t* raw, float* out, uint32_t n_rows, uint32_t head_dim, void* stream);

/// The gate in place: `hc` is [n_tok][n_hc][e], `kv` is [n_tok][n_hc+1][e] (rows 0..n_hc-1 are the keys, row
/// n_hc the values), `qw`/`kw` are [n_hc][e] f32.  Returns false without launching on a null pointer, e == 0,
/// n_hc == 0 or n_tok == 0.
bool deepseek4_engram_gate(float* hc, const float* kv, const float* qw, const float* kw, uint32_t e,
                           uint32_t n_hc, uint32_t n_tok, float eps, void* stream);

/// RNE f32 -> bf16 for the wkv activation copy: `x16[i]` is the bf16 bit pattern of `x[i]`, as the engine's
/// `v41_x_to_bf16_kernel` writes it.  Returns false without launching when n == 0 or a pointer is null.
bool deepseek4_engram_x_to_bf16(const float* x, uint16_t* x16, uint64_t n, void* stream);

/// y = W x for the engram's fp8_32x32 wkv.  `wkv` holds in_dim*out_dim E4M3 bytes, then
/// ceil(out_dim/32)*ceil(in_dim/32) E8M0 scale bytes (tile (r/32, c/32)).  `x` is [n_tok][in_dim] f32, `y` is
/// [n_tok][out_dim] f32, both row-major; `x_bf16` is the [n_tok][in_dim] copy `deepseek4_engram_x_to_bf16`
/// wrote and is required when n_tok > 1 (the reference's XB contract: batches read activations on the bf16
/// grid, a single token reads f32).  `wkv` and `x_bf16` must be 16-byte aligned.  Batches larger than 8 run in
/// 8-row segments, as the reference's grouped entry does.  Returns false, launching nothing, when a pointer
/// is null (x_bf16 only when n_tok > 1), n_tok is 0, out_dim is 0, or in_dim is not a positive multiple of 32.
bool deepseek4_engram_wkv(const uint8_t* wkv, const float* x, const uint16_t* x_bf16, float* y, uint32_t in_dim,
                          uint32_t out_dim, uint32_t n_tok, void* stream);

}  // namespace strata::kernels
