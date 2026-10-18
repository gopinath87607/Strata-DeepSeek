// src/kernels/cuda/deepseek4_moe.cu - P3: the DeepSeek-V4.1-Flash router and routed VQ MoE on the GPU.
//
// The reference's kernels, expression for expression:
//
//   deepseek4_moe_router_kernel   cuda/ds41_router.cuh v41_router_kernel: one warp per token, 12 experts per
//                                 lane, softplus with torch's threshold 20, selection score prob + bias,
//                                 top-k by warp argmax (ties to the smaller expert id), weights from the
//                                 UNbiased prob divided once by (wsum + 1e-20) * route_scale.  The optional
//                                 image bias (bias_vl at vl_rows[t] >= 0) is the official torch.where.
//
//   deepseek4_moe_gateup_kernel   cuda/ds41_vq_decode.cuh v41_vq_gateup_kernel's arithmetic: gate and up rows
//                                 are dotted off the packed stream with the shared deepseek4_vq_row_dot, each
//                                 rounded to bf16, SwiGLU with the layer clamp, and the mid parked as bf16 in
//                                 h - which is exactly what the down kernel reads back.  This port keeps the
//                                 reference's global-gather arm (one warp per row, codebook in global memory);
//                                 its shared-codebook arm is a P7 performance variant with identical values.
//
//   deepseek4_moe_down_kernel     v41_vq_down_kernel: partial[pair][o] = bf16(down_row * mid), with the
//                                 optional per-expert gain override; a refused payload writes zeros.
//
//   deepseek4_moe_reduce_kernel   v41_vq_reduce_kernel: out[t][o] = sum_k w[t][k] * partial[t*topk+k][o] in
//                                 ascending k.
//   deepseek4_moe_tail_kernel     v41_vq_tail_kernel: y[t][o] = bf16(the same sum + shared_out[t][o]).
//
// The xpack kernel is v41_vq_xpack_kernel: the activation is rounded to bf16 once per call, because the VQ
// dot's packed input is bf16 (the reference measured that as throughput at equal output, not accuracy).
//
// The reference's n=1 persist arm and its multi-token grouped and prefill arms are performance dispatches
// with verified byte-identical output (its own P4-2 evidence); they are not ported here.  This file's plain
// per-pair kernels run at every n_tok, so the gate can compare the arithmetic directly.
//
// The VQ lane map, codeword decode, multiply-add expression, xor tree and gain are the VQ header's one copy
// (deepseek4_vq_device.hpp), not repeated here.
#include "strata/kernels/deepseek4_moe.hpp"

#include <cuda_runtime.h>

#include <cstdio>

#include "strata/kernels/deepseek4_dense.hpp"
#include "strata/kernels/deepseek4_vq.hpp"
#include "deepseek4_vq_device.hpp"

namespace strata::kernels {

namespace {

constexpr uint32_t kMoeWarps = 8u;   // the reference's global-gather block: one warp per row

// ---- router -------------------------------------------------------------------------------------------

__global__ void deepseek4_moe_router_kernel(int32_t* __restrict__ sel, float* __restrict__ wts,
                                            const float* __restrict__ logits, const float* __restrict__ bias_txt,
                                            const float* __restrict__ bias_vl, const int32_t* __restrict__ vl_rows,
                                            uint32_t n_tok, uint32_t n_expert, uint32_t topk,
                                            float route_scale) {
    const uint32_t t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (t >= n_tok) return;
    const float* lg = logits + static_cast<uint64_t>(t) * n_expert;
    // Image positions use bias_vl (official torch.where(image_mask, bias_vl, bias)); all 32 lanes read the
    // same predicate, so the warp does not diverge.
    const float* bias = (bias_vl != nullptr && vl_rows != nullptr && vl_rows[t] >= 0) ? bias_vl : bias_txt;
    float pr[kDeepseek4RouterPerLane], sc[kDeepseek4RouterPerLane];
    #pragma unroll
    for (uint32_t k = 0; k < kDeepseek4RouterPerLane; k++) {
        const uint32_t e = lane + 32u * k;
        if (e < n_expert) {
            const float z = lg[e];
            const float sp = z > 20.0f ? z : log1pf(expf(z));   // softplus (beta=1, threshold 20 as in torch)
            pr[k] = sqrtf(sp);
            sc[k] = pr[k] + bias[e];
        } else {
            pr[k] = 0.f;
            sc[k] = -INFINITY;
        }
    }
    uint32_t used = 0;
    float wsum = 0.f;
    for (uint32_t r = 0; r < topk; r++) {
        float bv = -INFINITY;
        int bk = -1;
        #pragma unroll
        for (uint32_t k = 0; k < kDeepseek4RouterPerLane; k++)
            if (!((used >> k) & 1u) && sc[k] > bv) { bv = sc[k]; bk = static_cast<int>(k); }
        int be = bk >= 0 ? static_cast<int>(lane + 32u * static_cast<uint32_t>(bk)) : -1;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oe = __shfl_xor_sync(0xffffffffu, be, off);
            if (oe >= 0 && (be < 0 || ov > bv || (ov == bv && oe < be))) { bv = ov; be = oe; }
        }
        float myp = 0.f;
        if (be >= 0 && static_cast<uint32_t>(be & 31) == lane) {
            const uint32_t k = static_cast<uint32_t>(be) >> 5;
            used |= 1u << k;
            myp = pr[k];
        }
        for (int off = 16; off > 0; off >>= 1) myp += __shfl_xor_sync(0xffffffffu, myp, off);   // one lane nonzero
        // The unnormalized myp is stored and divided by wsum at the end: myp/(wsum+1e-20)*route_scale.
        if (lane == 0) {
            sel[static_cast<uint64_t>(t) * topk + r] = be;
            wts[static_cast<uint64_t>(t) * topk + r] = myp;
        }
        wsum += myp;
    }
    if (lane == 0)
        for (uint32_t r = 0; r < topk; r++)
            wts[static_cast<uint64_t>(t) * topk + r] =
                wts[static_cast<uint64_t>(t) * topk + r] / (wsum + 1e-20f) * route_scale;
}

// ---- the routed workers -------------------------------------------------------------------------------

// f32 (already on the bf16 grid) -> packed bf16, one value per uint16; two per word to the row dot.
__global__ void deepseek4_moe_xpack_kernel(uint16_t* __restrict__ dst, const float* __restrict__ src, uint64_t n) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<uint16_t>(__float_as_uint(deepseek4_dense_bf16r(src[i])) >> 16);
}

// gate and up for this block's rows, SwiGLU folded in; h[pair][r] is the bf16 mid.
template <int NBIT, int V3, int EXT>
__global__ void deepseek4_moe_gateup_kernel(uint16_t* __restrict__ h, const uint8_t* __restrict__ blob,
                                            uint64_t blob_size, const int32_t* __restrict__ sel,
                                            const uint32_t* __restrict__ xb, uint32_t in_dim, uint32_t mid_dim,
                                            uint32_t topk, float clamp) {
    const uint32_t r = blockIdx.x * kMoeWarps + (threadIdx.x >> 5);
    const uint32_t pair = blockIdx.y;
    const int32_t e = sel[pair];
    uint16_t* hp = h + static_cast<uint64_t>(pair) * mid_dim;
    if (r >= mid_dim) return;
    if (e < 0) { hp[r] = 0u; return; }   // our determinism choice: a negative selection leaves a zero mid
    const Deepseek4VqMatrix mg = deepseek4_vq_parse(blob, blob_size, static_cast<uint32_t>(e), 0, mid_dim, in_dim);
    const Deepseek4VqMatrix mu = deepseek4_vq_parse(blob, blob_size, static_cast<uint32_t>(e), 1, mid_dim, in_dim);
    if (!mg.ok || !mu.ok) { hp[r] = 0u; return; }
    const uint32_t t = pair / topk;
    const uint32_t* xs = xb + static_cast<uint64_t>(t) * (in_dim / 2u);
    const float gv = deepseek4_dense_bf16r(deepseek4_vq_row_dot<NBIT, V3, EXT>(mg, r, xs));
    const float ui = deepseek4_dense_bf16r(deepseek4_vq_row_dot<NBIT, V3, EXT>(mu, r, xs));
    // v41_vq_swiglu: clamp -> silu(g)*u -> bf16, stored as the high half of the f32.  Every lane holds the
    // same reduced dot (the xor tree is commutative at each step), the lead lane stores, as the reference does.
    float g = gv, u = ui;
    if (clamp > 0.0f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    if ((threadIdx.x & 31u) == 0u)
        hp[r] = static_cast<uint16_t>(__float_as_uint(deepseek4_dense_bf16r((g / (1.0f + expf(-g))) * u)) >> 16);
}

// partial[pair][o] = bf16(down row o * mid); a refused payload writes zeros for this block's rows.
template <int NBIT, int V3, int EXT>
__global__ void deepseek4_moe_down_kernel(float* __restrict__ partial, const uint8_t* __restrict__ blob,
                                          uint64_t blob_size, const int32_t* __restrict__ sel,
                                          const uint16_t* __restrict__ h, uint32_t mid_dim, uint32_t out_dim,
                                          const float* __restrict__ gain_override) {
    const uint32_t o = blockIdx.x * kMoeWarps + (threadIdx.x >> 5);
    const uint32_t pair = blockIdx.y;
    const int32_t e = sel[pair];
    float* pp = partial + static_cast<uint64_t>(pair) * out_dim;
    if (o >= out_dim) return;
    if (e < 0) { pp[o] = 0.f; return; }
    Deepseek4VqMatrix md = deepseek4_vq_parse(blob, blob_size, static_cast<uint32_t>(e), 2, out_dim, mid_dim);
    if (!md.ok) { pp[o] = 0.f; return; }
    md.gain_override = gain_override != nullptr ? gain_override + static_cast<size_t>(e) * out_dim : nullptr;
    const uint32_t* hs = reinterpret_cast<const uint32_t*>(h + static_cast<uint64_t>(pair) * mid_dim);
    const float v = deepseek4_dense_bf16r(deepseek4_vq_row_dot<NBIT, V3, EXT>(md, o, hs));
    if ((threadIdx.x & 31u) == 0u) pp[o] = v;   // the lead lane's, as the reference
}

// out[t][o] = sum_k w[t][k] * partial[t*topk+k][o] (f32; the official y += weights * expert_out).
__global__ void deepseek4_moe_reduce_kernel(float* __restrict__ out, const float* __restrict__ partial,
                                            const float* __restrict__ w, uint32_t topk, uint32_t out_dim) {
    const uint32_t t = blockIdx.y, o = blockIdx.x * 256u + threadIdx.x;
    if (o >= out_dim) return;
    float a = 0.f;
    for (uint32_t k = 0; k < topk; k++) {
        const uint64_t pair = static_cast<uint64_t>(t) * topk + k;
        a += w[pair] * partial[pair * out_dim + o];
    }
    out[static_cast<uint64_t>(t) * out_dim + o] = a;
}

__global__ void deepseek4_moe_tail_kernel(float* __restrict__ y, const float* __restrict__ partial,
                                          const float* __restrict__ w, const float* __restrict__ so,
                                          uint32_t topk, uint32_t out_dim) {
    const uint32_t t = blockIdx.y, o = blockIdx.x * 256u + threadIdx.x;
    if (o >= out_dim) return;
    float a = 0.f;
    for (uint32_t k = 0; k < topk; k++) {
        const uint64_t pair = static_cast<uint64_t>(t) * topk + k;
        a += w[pair] * partial[pair * out_dim + o];
    }
    y[static_cast<uint64_t>(t) * out_dim + o] =
        deepseek4_dense_bf16r(a + so[static_cast<uint64_t>(t) * out_dim + o]);
}

// ---- launchers ----------------------------------------------------------------------------------------

bool dsv4_moe_ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "deepseek4 moe %s: %s\n", what, cudaGetErrorString(e));
    (void) cudaGetLastError();
    return false;
}

// The template instance, chosen from the layer's metadata exactly as the reference's dispatcher does.
bool dsv4_moe_launch_gateup(uint32_t nc, uint32_t blob_v3, dim3 grid, cudaStream_t s, uint16_t* h,
                            const uint8_t* blob, uint64_t blob_size, const int32_t* sel, const uint32_t* xb,
                            uint32_t in_dim, uint32_t mid_dim, uint32_t topk, float clamp) {
    const uint32_t nbit = deepseek4_vq_index_bits(nc);
    if (blob_v3) {
        if (nbit == 13u) {
            deepseek4_moe_gateup_kernel<13, 1, 1><<<grid, kMoeWarps * 32u, 0, s>>>(
                h, blob, blob_size, sel, xb, in_dim, mid_dim, topk, clamp);
            return dsv4_moe_ok(cudaGetLastError(), "gateup");
        }
        if (nbit == 12u) {
            deepseek4_moe_gateup_kernel<12, 1, 0><<<grid, kMoeWarps * 32u, 0, s>>>(
                h, blob, blob_size, sel, xb, in_dim, mid_dim, topk, clamp);
            return dsv4_moe_ok(cudaGetLastError(), "gateup");
        }
        return false;
    }
    switch (nbit) {
        case 8: deepseek4_moe_gateup_kernel<8, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(h, blob, blob_size, sel, xb, in_dim, mid_dim, topk, clamp); break;
        case 9: deepseek4_moe_gateup_kernel<9, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(h, blob, blob_size, sel, xb, in_dim, mid_dim, topk, clamp); break;
        case 10: deepseek4_moe_gateup_kernel<10, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(h, blob, blob_size, sel, xb, in_dim, mid_dim, topk, clamp); break;
        case 11: deepseek4_moe_gateup_kernel<11, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(h, blob, blob_size, sel, xb, in_dim, mid_dim, topk, clamp); break;
        case 12: deepseek4_moe_gateup_kernel<12, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(h, blob, blob_size, sel, xb, in_dim, mid_dim, topk, clamp); break;
        default: return false;
    }
    return dsv4_moe_ok(cudaGetLastError(), "gateup");
}

bool dsv4_moe_launch_down(uint32_t nc, uint32_t blob_v3, dim3 grid, cudaStream_t s, float* partial,
                          const uint8_t* blob, uint64_t blob_size, const int32_t* sel, const uint16_t* h,
                          uint32_t mid_dim, uint32_t out_dim, const float* gain_override) {
    const uint32_t nbit = deepseek4_vq_index_bits(nc);
    if (blob_v3) {
        if (nbit == 13u) {
            deepseek4_moe_down_kernel<13, 1, 1><<<grid, kMoeWarps * 32u, 0, s>>>(
                partial, blob, blob_size, sel, h, mid_dim, out_dim, gain_override);
            return dsv4_moe_ok(cudaGetLastError(), "down");
        }
        if (nbit == 12u) {
            deepseek4_moe_down_kernel<12, 1, 0><<<grid, kMoeWarps * 32u, 0, s>>>(
                partial, blob, blob_size, sel, h, mid_dim, out_dim, gain_override);
            return dsv4_moe_ok(cudaGetLastError(), "down");
        }
        return false;
    }
    switch (nbit) {
        case 8: deepseek4_moe_down_kernel<8, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(partial, blob, blob_size, sel, h, mid_dim, out_dim, gain_override); break;
        case 9: deepseek4_moe_down_kernel<9, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(partial, blob, blob_size, sel, h, mid_dim, out_dim, gain_override); break;
        case 10: deepseek4_moe_down_kernel<10, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(partial, blob, blob_size, sel, h, mid_dim, out_dim, gain_override); break;
        case 11: deepseek4_moe_down_kernel<11, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(partial, blob, blob_size, sel, h, mid_dim, out_dim, gain_override); break;
        case 12: deepseek4_moe_down_kernel<12, 0, 0><<<grid, kMoeWarps * 32u, 0, s>>>(partial, blob, blob_size, sel, h, mid_dim, out_dim, gain_override); break;
        default: return false;
    }
    return dsv4_moe_ok(cudaGetLastError(), "down");
}

}  // namespace

bool deepseek4_router(int32_t* d_sel, float* d_wts, const float* d_logits, const float* d_bias,
                      const float* d_bias_vl, const int32_t* d_vl_rows, uint32_t n_tok, uint32_t n_expert,
                      uint32_t topk, float route_scale, void* stream) {
    if (d_sel == nullptr || d_wts == nullptr || d_logits == nullptr || d_bias == nullptr || topk == 0) return false;
    if (topk > kDeepseek4MoeMaxTopk || n_expert > kDeepseek4MoeMaxExperts || n_expert == 0) return false;
    if (n_tok == 0) return true;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    deepseek4_moe_router_kernel<<<(n_tok + 7u) / 8u, 256, 0, s>>>(
        d_sel, d_wts, d_logits, d_bias, d_bias_vl, d_vl_rows, n_tok, n_expert, topk, route_scale);
    if (!dsv4_moe_ok(cudaGetLastError(), "router")) return false;
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

bool deepseek4_routed_moe(float* d_out, float* d_partial, const void* d_blob, uint64_t blob_size,
                          uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim, const int32_t* d_sel,
                          const float* d_w, uint32_t topk, float clamp, const float* d_x, uint32_t n_tok,
                          uint32_t nc, uint32_t blob_v3, const float* d_gain_override, void* stream) {
    if (d_blob == nullptr || d_sel == nullptr || d_w == nullptr || d_x == nullptr) return false;
    if (n_tok == 0 || topk == 0 || in_dim == 0 || mid_dim == 0 || out_dim == 0) return false;
    if ((in_dim % 8u) != 0 || (mid_dim % 8u) != 0) return false;
    if (d_out == nullptr && d_partial == nullptr) return false;

    const uint64_t np = static_cast<uint64_t>(n_tok) * topk;
    if (np > 65535u) return false;   // gridDim.y's bound; the reference's prefill arm (P7) lifts it
    // With d_out the reduce only needs the partials; a caller that passes neither buffer still gets them,
    // because this correctness entry owns the whole chain.
    float* d_part = d_partial;
    if (d_out != nullptr && d_part == nullptr) {
        if (cudaMalloc(&d_part, static_cast<size_t>(np) * out_dim * sizeof(float)) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
    }
    uint16_t* d_h = nullptr;
    uint16_t* d_xb = nullptr;
    if (cudaMalloc(&d_h, static_cast<size_t>(np) * mid_dim * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_xb, static_cast<size_t>(n_tok) * in_dim * sizeof(uint16_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        if (d_h != nullptr) cudaFree(d_h);
        if (d_xb != nullptr) cudaFree(d_xb);
        if (d_part != d_partial) cudaFree(d_part);
        return false;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const uint64_t nx = static_cast<uint64_t>(n_tok) * in_dim;
    deepseek4_moe_xpack_kernel<<<static_cast<unsigned>((nx + 255u) / 256u), 256, 0, s>>>(d_xb, d_x, nx);
    bool ok = dsv4_moe_ok(cudaGetLastError(), "xpack");
    if (ok) {
        ok = dsv4_moe_launch_gateup(nc, blob_v3, dim3((mid_dim + kMoeWarps - 1u) / kMoeWarps, static_cast<unsigned>(np)),
                                    s, d_h, static_cast<const uint8_t*>(d_blob), blob_size, d_sel,
                                    reinterpret_cast<const uint32_t*>(d_xb), in_dim, mid_dim, topk, clamp);
    }
    if (ok) {
        ok = dsv4_moe_launch_down(nc, blob_v3, dim3((out_dim + kMoeWarps - 1u) / kMoeWarps, static_cast<unsigned>(np)),
                                  s, d_part, static_cast<const uint8_t*>(d_blob), blob_size, d_sel, d_h,
                                  mid_dim, out_dim, d_gain_override);
    }
    if (ok && d_out != nullptr) {
        deepseek4_moe_reduce_kernel<<<dim3((out_dim + 255u) / 256u, n_tok), 256, 0, s>>>(d_out, d_part, d_w,
                                                                                       topk, out_dim);
        ok = dsv4_moe_ok(cudaGetLastError(), "reduce");
    }
    // The entry owns its scratch, so it must not free it before the kernels have read it.
    if (stream == nullptr) (void) cudaDeviceSynchronize();
    else (void) cudaStreamSynchronize(s);
    cudaFree(d_h);
    cudaFree(d_xb);
    if (d_part != d_partial) cudaFree(d_part);
    return ok;
}

bool deepseek4_moe_tail(float* d_y, const float* d_partial, const float* d_w, const float* d_so,
                        uint32_t n_tok, uint32_t topk, uint32_t out_dim, void* stream) {
    if (d_y == nullptr || d_partial == nullptr || d_w == nullptr || d_so == nullptr) return false;
    if (n_tok == 0 || topk == 0 || out_dim == 0) return true;
    if (n_tok > 65535u) return false;   // gridDim.y's bound
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    deepseek4_moe_tail_kernel<<<dim3((out_dim + 255u) / 256u, n_tok), 256, 0, s>>>(d_y, d_partial, d_w, d_so,
                                                                                  topk, out_dim);
    if (!dsv4_moe_ok(cudaGetLastError(), "tail")) return false;
    if (stream == nullptr) cudaDeviceSynchronize();
    return true;
}

}  // namespace strata::kernels
