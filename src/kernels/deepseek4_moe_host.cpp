// src/kernels/deepseek4_moe_host.cpp - P3: the MoE group's host oracles.
//
// This is the specification the CUDA kernels in src/kernels/cuda/deepseek4_moe.cu are compared against.  Three
// pieces, each written as the reference engine's arithmetic and not as a convenient approximation:
//
//   * the router emulates v41_router_kernel lane for lane: 12 experts per warp lane, the per-lane strict
//     maximum among the unused (so a lane's smaller expert id wins its own tie), the 5-step xor tree with the
//     `ov > bv || (ov == bv && oe < be)` rule (so a global tie goes to the smaller expert id), the winner's
//     UNbiased probability summed in round order, and one final divide by (wsum + 1e-20) * route_scale;
//   * the routed MoE walks the same chain the device walks: the activation is rounded to bf16 first, every
//     row dot calls the VQ header's one lane-partial function for each of the 32 lanes, the xor tree is
//     replayed in the same order over those 32 values, the gain is applied last in the reference's grouping,
//     gate and up are each rounded to bf16 before SwiGLU, the mid is stored as packed bf16 exactly as the
//     kernel leaves it for the down dot, and the down row is rounded to bf16;
//   * the tail folds sum_k w[k] * partial[k] + shared in ascending k and rounds once.
//
// The one function shared with the device on purpose rather than re-derived: the SwiGLU is
// deepseek4_swiglu_host, the same expression as the dense kernel, so the clamp and the single bf16 round can
// not drift between the two.  The libm's expf is the only value the host and the device may round differently;
// the gate's tolerances cover that one rounding (and the reference fixture's one-hot cases make the chain
// single-term, so even it cannot move a bit there).
#include "strata/kernels/deepseek4_moe.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "strata/kernels/deepseek4_dense.hpp"
#include "strata/kernels/deepseek4_vq.hpp"

namespace strata::kernels {

namespace {

// The high half of an f32 that is already on the bf16 grid (deepseek4_dense_bf16r put it there).
uint16_t bf16_high_bits(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return static_cast<uint16_t>(bits >> 16);
}

// One row dot with the device's lane map and reduction order.  A lane's partial comes from the VQ header's
// shared function; the tree below is the kernel's `for (o = 16; o; o >>= 1) acc += __shfl_xor_sync(..., o)`
// with the old values read before any lane updates, which is what the shuffle does.
template <int NBIT, int V3, int EXT>
float row_dot_host_typed(const Deepseek4VqMatrix& m, uint32_t r, const uint32_t* x) {
    float acc[32];
    for (uint32_t l = 0; l < 32u; l++) acc[l] = deepseek4_vq_lane_partial<NBIT, V3, EXT>(m, r, l, x);
    for (int o = 16; o > 0; o >>= 1) {
        float old[32];
        for (uint32_t l = 0; l < 32u; l++) old[l] = acc[l];
        for (uint32_t l = 0; l < 32u; l++) acc[l] += old[l ^ static_cast<uint32_t>(o)];
    }
    return deepseek4_vq_apply_gain(acc[0], m, r);
}

// The runtime dispatch the launchers also take: v3 is 12 bits without the plane or 13 with it; v2 is 8..12.
// Returns false for a payload with no compiled instance (the caller then treats the matrix as bad).
bool row_dot_host(const Deepseek4VqMatrix& m, uint32_t r, const uint32_t* x, float* out) {
    if (!m.ok) return false;
    if (m.v3) {
        if (m.nbit == 13u && m.plane != nullptr) *out = row_dot_host_typed<13, 1, 1>(m, r, x);
        else if (m.nbit == 12u && m.plane == nullptr) *out = row_dot_host_typed<12, 1, 0>(m, r, x);
        else return false;
        return true;
    }
    switch (m.nbit) {
        case 8: *out = row_dot_host_typed<8, 0, 0>(m, r, x); return true;
        case 9: *out = row_dot_host_typed<9, 0, 0>(m, r, x); return true;
        case 10: *out = row_dot_host_typed<10, 0, 0>(m, r, x); return true;
        case 11: *out = row_dot_host_typed<11, 0, 0>(m, r, x); return true;
        case 12: *out = row_dot_host_typed<12, 0, 0>(m, r, x); return true;
        default: return false;
    }
}

}  // namespace

bool deepseek4_router_host(int32_t* sel, float* wts, const float* logits, const float* bias, const float* bias_vl,
                           const int32_t* vl_rows, uint32_t n_tok, uint32_t n_expert, uint32_t topk,
                           float route_scale) {
    if (sel == nullptr || wts == nullptr || logits == nullptr || bias == nullptr || topk == 0) return false;
    if (topk > kDeepseek4MoeMaxTopk || n_expert > kDeepseek4MoeMaxExperts || n_expert == 0) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float* lg = logits + static_cast<size_t>(t) * n_expert;
        const float* b = (bias_vl != nullptr && vl_rows != nullptr && vl_rows[t] >= 0) ? bias_vl : bias;
        // softplus with torch's threshold 20, then the selection score sqrt(softplus) + bias.
        std::vector<float> pr(n_expert), sc(n_expert);
        for (uint32_t e = 0; e < n_expert; e++) {
            const float z = lg[e];
            const float sp = z > 20.0f ? z : std::log1p(std::exp(z));
            pr[e] = std::sqrt(sp);
            sc[e] = pr[e] + b[e];
        }
        std::vector<uint8_t> used(n_expert, 0);
        float wsum = 0.0f;
        for (uint32_t round = 0; round < topk; round++) {
            // Per lane: its 12 experts, the largest unused selection score, strict > so the smaller of its
            // own ids keeps a tie.
            float bv[32];
            int be[32];
            for (uint32_t lane = 0; lane < 32u; lane++) {
                bv[lane] = -std::numeric_limits<float>::infinity();
                be[lane] = -1;
                for (uint32_t k = 0; k < kDeepseek4RouterPerLane; k++) {
                    const uint32_t e = lane + 32u * k;
                    if (e < n_expert && used[e] == 0 && sc[e] > bv[lane]) {
                        bv[lane] = sc[e];
                        be[lane] = static_cast<int>(e);
                    }
                }
            }
            // The warp's xor tree: larger score wins, equal scores go to the smaller expert id, and a lane
            // with nothing usable yields to any lane that has something.
            for (int off = 16; off > 0; off >>= 1) {
                float obv[32];
                int obe[32];
                for (uint32_t lane = 0; lane < 32u; lane++) {
                    obv[lane] = bv[lane];
                    obe[lane] = be[lane];
                }
                for (uint32_t lane = 0; lane < 32u; lane++) {
                    const float ov = obv[lane ^ static_cast<uint32_t>(off)];
                    const int oe = obe[lane ^ static_cast<uint32_t>(off)];
                    if (oe >= 0 && (be[lane] < 0 || ov > bv[lane] ||
                                    (ov == bv[lane] && oe < be[lane]))) {
                        bv[lane] = ov;
                        be[lane] = oe;
                    }
                }
            }
            const int winner = be[0];
            float myp = 0.0f;
            if (winner >= 0) {
                used[static_cast<uint32_t>(winner)] = 1;
                myp = pr[static_cast<uint32_t>(winner)];   // the unbiased probability, as the kernel sends it
            }
            sel[static_cast<size_t>(t) * topk + round] = winner;   // -1 when topk > n_expert
            wts[static_cast<size_t>(t) * topk + round] = myp;
            wsum += myp;
        }
        for (uint32_t round = 0; round < topk; round++)
            wts[static_cast<size_t>(t) * topk + round] =
                wts[static_cast<size_t>(t) * topk + round] / (wsum + 1e-20f) * route_scale;
    }
    return true;
}

bool deepseek4_routed_moe_host(float* out, const uint8_t* blob, uint64_t blob_size, uint32_t in_dim,
                               uint32_t mid_dim, uint32_t out_dim, const int32_t* sel, const float* w,
                               uint32_t topk, float clamp, const float* x, uint32_t n_tok,
                               const float* gain_override) {
    if (out == nullptr || blob == nullptr || sel == nullptr || w == nullptr || x == nullptr) return false;
    if (n_tok == 0 || topk == 0 || in_dim == 0 || mid_dim == 0 || out_dim == 0) return false;
    if ((in_dim % 8u) != 0 || (mid_dim % 8u) != 0) return false;

    const uint64_t np = static_cast<uint64_t>(n_tok) * topk;
    // The activation, rounded to bf16 and packed two values per word exactly as the xpack kernel leaves it.
    std::vector<uint32_t> xb(static_cast<size_t>(n_tok) * (in_dim / 2u), 0u);
    for (uint32_t t = 0; t < n_tok; t++) {
        const float* xr = x + static_cast<size_t>(t) * in_dim;
        uint32_t* xw = xb.data() + static_cast<size_t>(t) * (in_dim / 2u);
        for (uint32_t i = 0; i < in_dim; i++) {
            const uint16_t bits = bf16_high_bits(deepseek4_dense_bf16r(xr[i]));
            if ((i & 1u) == 0) xw[i >> 1] = bits;
            else xw[i >> 1] |= static_cast<uint32_t>(bits) << 16;
        }
    }
    std::vector<uint16_t> h(static_cast<size_t>(np) * mid_dim, 0u);
    std::vector<float> part(static_cast<size_t>(np) * out_dim, 0.0f);

    for (uint64_t pair = 0; pair < np; pair++) {
        const uint32_t t = static_cast<uint32_t>(pair / topk);
        const int32_t e = sel[pair];
        uint16_t* hp = h.data() + pair * mid_dim;
        float* pp = part.data() + pair * out_dim;
        // A negative selection or a refused payload leaves zeros (our determinism choice; a valid model
        // reaches neither, and the kernel writes the same zeros).
        if (e < 0) continue;
        const Deepseek4VqMatrix mg = deepseek4_vq_parse(blob, blob_size, static_cast<uint32_t>(e), 0, mid_dim, in_dim);
        const Deepseek4VqMatrix mu = deepseek4_vq_parse(blob, blob_size, static_cast<uint32_t>(e), 1, mid_dim, in_dim);
        const uint32_t* xs = xb.data() + static_cast<size_t>(t) * (in_dim / 2u);
        bool gate_ok = mg.ok != 0 && mu.ok != 0;
        if (gate_ok) {
            for (uint32_t r = 0; r < mid_dim && gate_ok; r++) {
                float gv, ui;
                gate_ok = row_dot_host(mg, r, xs, &gv) && row_dot_host(mu, r, xs, &ui);
                if (!gate_ok) break;
                gv = deepseek4_dense_bf16r(gv);
                ui = deepseek4_dense_bf16r(ui);
                float mid;
                deepseek4_swiglu_host(&mid, &gv, &ui, 1, clamp);   // bf16-rounded already
                hp[r] = bf16_high_bits(mid);
            }
        }
        if (!gate_ok) {
            std::memset(hp, 0, static_cast<size_t>(mid_dim) * sizeof(uint16_t));
            continue;   // a zero mid makes a zero partial below, but skip the dot entirely
        }
        Deepseek4VqMatrix md = deepseek4_vq_parse(blob, blob_size, static_cast<uint32_t>(e), 2, out_dim, mid_dim);
        md.gain_override = gain_override != nullptr ? gain_override + static_cast<size_t>(e) * out_dim : nullptr;
        bool down_ok = md.ok != 0;
        const uint32_t* hs = reinterpret_cast<const uint32_t*>(hp);
        for (uint32_t o = 0; o < out_dim && down_ok; o++) {
            float v;
            down_ok = row_dot_host(md, o, hs, &v);
            if (down_ok) pp[o] = deepseek4_dense_bf16r(v);
        }
        if (!down_ok) std::memset(pp, 0, static_cast<size_t>(out_dim) * sizeof(float));
    }

    // out[t][o] = sum_k w[t][k] * partial[t*topk + k][o], ascending k (the official y += weights * expert_out).
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float a = 0.0f;
            for (uint32_t k = 0; k < topk; k++) {
                const uint64_t pair = static_cast<uint64_t>(t) * topk + k;
                a += w[pair] * part[pair * out_dim + o];
            }
            out[static_cast<size_t>(t) * out_dim + o] = a;
        }
    }
    return true;
}

void deepseek4_moe_tail_host(float* y, const float* partial, const float* w, const float* so, uint32_t n_tok,
                             uint32_t topk, uint32_t out_dim) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float a = 0.0f;
            for (uint32_t k = 0; k < topk; k++) {
                const uint64_t pair = static_cast<uint64_t>(t) * topk + k;
                a += w[pair] * partial[pair * out_dim + o];
            }
            y[static_cast<size_t>(t) * out_dim + o] =
                deepseek4_dense_bf16r(a + so[static_cast<size_t>(t) * out_dim + o]);
        }
    }
}

}  // namespace strata::kernels
