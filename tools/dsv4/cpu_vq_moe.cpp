// tools/dsv4/cpu_vq_moe.cpp - the AVX2 kernel behind cpu_vq_moe.hpp.
//
// The measured shape (tools/dsv4/cpu_expert_bench.cpp, 2x Xeon E5-2680 v4, 28 threads, 2026-10-18):
// 13-bit layers 42.9 tok/s, 12-bit layers 52.8 tok/s of routed experts, both with every expert of a
// 1024-token block processed.  What got it there, in order of what each was worth:
//   - eight tokens to a tile, so one codeword fetch feeds eight FMAs;
//   - a 64-codeword column block copied once into a 16 KB scratch, so the eight x loads are constant
//     displacements off one base register instead of eight runtime addresses (25.4 -> 34.1 tok/s);
//   - the row stream prefetched six rows ahead: a row is 32 pairs of work, far more than the OOO
//     window covers, and its 96-byte block is a cold miss otherwise (34.1 -> 42.9 tok/s).
#include "cpu_vq_moe.hpp"

#include "vq_decode.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace strata::dsv4cpu {
namespace {

constexpr uint32_t kColBlock = 64;  // codewords, i.e. 512 columns: 16 KB of activation per 8-token tile
constexpr uint32_t kTileCols = kColBlock * 8;
constexpr uint32_t kPrefetchRows = 6;

inline uint32_t f32_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}
inline float bits_f32(uint32_t u) {
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// RNE f32 -> bf16 -> f32, the engine's v41_bf16r bit for bit (Inf and NaN pass through).
inline float bf16r(float x) {
    uint32_t u = f32_bits(x);
    if ((u & 0x7f800000u) != 0x7f800000u) {
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
    }
    return bits_f32(u);
}
inline uint16_t bf16_bits(float x) { return static_cast<uint16_t>(f32_bits(bf16r(x)) >> 16); }

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

inline float f16_at(const uint8_t* p, uint32_t i) { return dsv4::f16_to_f32(dsv4::rd_u16(p, static_cast<size_t>(i) * 2)); }

// Copy one column block of the tile's activations into the compact scratch, rounding to bf16 as the
// device's xpack does.  The token stride is then a compile-time constant, so the eight dot-product loads
// are constant displacements off one base register instead of eight runtime addresses.
template <int N>
inline void gather_tile(const float* __restrict__ xb, size_t xs, const uint32_t* __restrict__ toks,
                        uint32_t i0, uint32_t i1, float* __restrict__ xblk) {
    const uint32_t n = (i1 - i0) * 8;
    for (int t = 0; t < N; t++) {
        const float* src = xb + xs * toks[t] + static_cast<size_t>(i0) * 8;
        float* dst = xblk + static_cast<size_t>(t) * kTileCols;
        for (uint32_t j = 0; j < n; j++) dst[j] = bf16r(src[j]);
    }
}

// The stream word for a row is only 1.5 lines and the next row is a fixed stride away, so a row starts
// with a cold miss the out-of-order window cannot cover.  Touch the row kPrefetchRows ahead.
inline void prefetch_row(const Matrix& m, uint32_t row, uint32_t i0) {
    const char* p = reinterpret_cast<const char*>(m.stream + static_cast<size_t>(row) * m.mrow + (i0 >> 1) * 3);
    _mm_prefetch(p, _MM_HINT_T1);
    _mm_prefetch(p + 64, _MM_HINT_T1);
    if (m.plane != nullptr)
        _mm_prefetch(reinterpret_cast<const char*>(m.plane + static_cast<size_t>(row) * m.prow + (i0 >> 3)),
                     _MM_HINT_T1);
}

// N tokens' dot with one matrix row over codewords [i0, i1), accumulated into out[t * os].
template <int N, bool Plane>
inline void rowdot_block(const Matrix& m, uint32_t row, uint32_t i0, uint32_t i1,
                         const float* __restrict__ xblk, float* __restrict__ out, size_t os) {
    __m256 acc[N];
#pragma GCC unroll 8
    for (int t = 0; t < N; t++) acc[t] = _mm256_setzero_ps();

    const uint8_t* __restrict__ rs = m.stream + static_cast<size_t>(row) * m.mrow;
    const uint8_t* __restrict__ rp = Plane ? m.plane + static_cast<size_t>(row) * m.prow : nullptr;
    (void) rp;

    for (uint32_t ip = i0; ip < i1; ip += 2) {
        uint32_t w;
        std::memcpy(&w, rs + (ip >> 1) * 3, 4);  // 24 bits wanted; the fourth is masked by the shifts
        uint32_t ia = w & 0xFFFu;
        uint32_t ib = (w >> 12) & 0xFFFu;
        if (Plane) {
            const uint32_t bits = (rp[ip >> 3] >> (ip & 7u)) & 3u;  // ip is even, so both bits share a byte
            ia |= (bits & 1u) << 12;
            ib |= (bits >> 1) << 12;
        }

        const float* __restrict__ xg = xblk + static_cast<size_t>(ip - i0) * 8;
        const __m256 wa = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(m.cb + static_cast<size_t>(ia) * 8))),
            16));
#pragma GCC unroll 8
        for (int t = 0; t < N; t++)
            acc[t] = _mm256_fmadd_ps(wa, _mm256_loadu_ps(xg + static_cast<size_t>(t) * kTileCols), acc[t]);

        const __m256 wb = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(m.cb + static_cast<size_t>(ib) * 8))),
            16));
        xg += 8;
#pragma GCC unroll 8
        for (int t = 0; t < N; t++)
            acc[t] = _mm256_fmadd_ps(wb, _mm256_loadu_ps(xg + static_cast<size_t>(t) * kTileCols), acc[t]);
    }

#pragma GCC unroll 8
    for (int t = 0; t < N; t++) out[static_cast<size_t>(t) * os] += hsum256(acc[t]);
}

struct Scratch {
    std::vector<float> gb, ub, hb, ob, xblk, o8;
};

void ensure_scratch(Scratch& s, uint32_t mid, uint32_t out_dim) {
    if (s.gb.size() != static_cast<size_t>(8) * mid) {
        s.gb.assign(static_cast<size_t>(8) * mid, 0.0f);
        s.ub.assign(static_cast<size_t>(8) * mid, 0.0f);
        s.hb.assign(static_cast<size_t>(8) * mid, 0.0f);
        s.ob.assign(static_cast<size_t>(8) * out_dim, 0.0f);
        s.o8.assign(static_cast<size_t>(8) * out_dim, 0.0f);
        s.xblk.assign(static_cast<size_t>(8) * kTileCols, 0.0f);  // eight tokens by one 64-codeword block
    }
}

// The full expert for N <= 8 tokens: gate and up sweeps, SwiGLU (bf16 mid), down sweep, with the
// device's rounding at each step.
template <int N, bool Plane>
void tile_run(const Matrix& cg, const Matrix& cu, const Matrix& cd, const float* __restrict__ xb, size_t xs,
              const uint32_t* __restrict__ toks, float* __restrict__ out, size_t os, float clamp,
              Scratch& s) {
    const uint32_t mid = cg.rows;
    std::fill(s.gb.begin(), s.gb.begin() + static_cast<size_t>(N) * mid, 0.0f);
    std::fill(s.ub.begin(), s.ub.begin() + static_cast<size_t>(N) * cu.rows, 0.0f);
    for (uint32_t i0 = 0; i0 < cg.nidx; i0 += kColBlock) {
        const uint32_t i1 = std::min(i0 + kColBlock, cg.nidx);
        gather_tile<N>(xb, xs, toks, i0, i1, s.xblk.data());
        // gather_tile writes at stride kTileCols for N tokens only into one block; the sweep needs all N
        // blocks per token, so the tile scratch is 8 tokens wide and filled here instead.
        for (uint32_t r = 0; r < mid; r++) {
            if (r + kPrefetchRows < mid) {
                prefetch_row(cg, r + kPrefetchRows, i0);
                prefetch_row(cu, r + kPrefetchRows, i0);
            }
            rowdot_block<N, Plane>(cg, r, i0, i1, s.xblk.data(), s.gb.data() + r, mid);
            rowdot_block<N, Plane>(cu, r, i0, i1, s.xblk.data(), s.ub.data() + r, cu.rows);
        }
    }
    for (int t = 0; t < N; t++) {
        for (uint32_t r = 0; r < mid; r++) {
            float g = bf16r(s.gb[static_cast<size_t>(t) * mid + r] * cg.gain(r));
            float u = bf16r(s.ub[static_cast<size_t>(t) * cu.rows + r] * cu.gain(r));
            if (clamp > 0.0f) {
                g = std::fmin(g, clamp);
                u = std::fmin(std::fmax(u, -clamp), clamp);
            }
            s.hb[static_cast<size_t>(t) * cd.cols + r] = bf16r((g / (1.0f + std::exp(-g))) * u);
        }
    }
    std::fill(s.ob.begin(), s.ob.begin() + static_cast<size_t>(N) * cd.rows, 0.0f);
    // hb is per-tile: the down sweep indexes it by the token's position in the tile, not its global id
    uint32_t local[8];
    for (int t = 0; t < N; t++) local[t] = static_cast<uint32_t>(t);
    for (uint32_t i0 = 0; i0 < cd.nidx; i0 += kColBlock) {
        const uint32_t i1 = std::min(i0 + kColBlock, cd.nidx);
        gather_tile<N>(s.hb.data(), cd.cols, local, i0, i1, s.xblk.data());
        for (uint32_t o = 0; o < cd.rows; o++) {
            if (o + kPrefetchRows < cd.rows) prefetch_row(cd, o + kPrefetchRows, i0);
            rowdot_block<N, Plane>(cd, o, i0, i1, s.xblk.data(), s.ob.data() + o, cd.rows);
        }
    }
    for (int t = 0; t < N; t++)
        for (uint32_t o = 0; o < cd.rows; o++)
            out[static_cast<size_t>(t) * os + o] = bf16r(s.ob[static_cast<size_t>(t) * cd.rows + o] * cd.gain(o));
}

template <bool Plane>
void tile_dispatch(const Matrix& cg, const Matrix& cu, const Matrix& cd, const float* xb, size_t xs,
                   const uint32_t* toks, uint32_t n, float* out, size_t os, float clamp, Scratch& s) {
    switch (n) {
        case 1: tile_run<1, Plane>(cg, cu, cd, xb, xs, toks, out, os, clamp, s); break;
        case 2: tile_run<2, Plane>(cg, cu, cd, xb, xs, toks, out, os, clamp, s); break;
        case 3: tile_run<3, Plane>(cg, cu, cd, xb, xs, toks, out, os, clamp, s); break;
        case 4: tile_run<4, Plane>(cg, cu, cd, xb, xs, toks, out, os, clamp, s); break;
        case 5: tile_run<5, Plane>(cg, cu, cd, xb, xs, toks, out, os, clamp, s); break;
        case 6: tile_run<6, Plane>(cg, cu, cd, xb, xs, toks, out, os, clamp, s); break;
        case 7: tile_run<7, Plane>(cg, cu, cd, xb, xs, toks, out, os, clamp, s); break;
        default: tile_run<8, Plane>(cg, cu, cd, xb, xs, toks, out, os, clamp, s); break;
    }
}

void tile_any(const Matrix& cg, const Matrix& cu, const Matrix& cd, const float* xb, size_t xs,
              const uint32_t* toks, uint32_t n, float* out, size_t os, float clamp, Scratch& s) {
    if (cg.plane != nullptr)
        tile_dispatch<true>(cg, cu, cd, xb, xs, toks, n, out, os, clamp, s);
    else
        tile_dispatch<false>(cg, cu, cd, xb, xs, toks, n, out, os, clamp, s);
}

// One job, eight tokens a tile, outputs packed to bf16 bits.
void run_job(const Job& job, const float* x, size_t xs, float clamp, Scratch& s) {
    ensure_scratch(s, job.gate->rows, job.down->rows);
    for (uint32_t t0 = 0; t0 < job.n_tok; t0 += 8) {
        const uint32_t n = std::min(8u, job.n_tok - t0);
        tile_any(*job.gate, *job.up, *job.down, x, xs, job.toks + t0, n, s.o8.data(), job.down->rows, clamp, s);
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t o = 0; o < job.down->rows; o++)
                job.out[static_cast<size_t>(t0 + t) * job.out_stride + o] =
                    bf16_bits(s.o8[static_cast<size_t>(t) * job.down->rows + o]);
    }
}

}  // namespace

float Matrix::gain(uint32_t row) const { return f16_at(gains, row); }

void expand_codebook(const uint8_t* codebook, uint32_t nc, bool e4m3, uint16_t* out) {
    const uint32_t n = nc * 8;
    for (uint32_t i = 0; i < n; i++)
        out[i] = e4m3 ? bf16_bits(dsv4::e4m3fn_to_f32(codebook[i]))
                      : bf16_bits(dsv4::f16_to_f32(dsv4::rd_u16(codebook, static_cast<size_t>(i) * 2)));
}

void open_matrix(const Layout& l, const uint16_t* cb16, Matrix* m) {
    m->stream = l.stream;
    m->plane = l.plane;
    m->gains = l.gains;
    m->cb = cb16;
    m->rows = l.rows;
    m->cols = l.cols;
    m->nidx = l.cols / 8;
    m->mrow = (m->nidx * 12 + 7) / 8;
    m->prow = (m->nidx + 7) / 8;
}

void expert_tile(const Matrix& gate, const Matrix& up, const Matrix& down, const float* x, size_t xs,
                 const uint32_t* toks, uint32_t n_tok, float* out, size_t os, float clamp) {
    Scratch s;
    ensure_scratch(s, gate.rows, down.rows);
    tile_any(gate, up, down, x, xs, toks, std::min(n_tok, 8u), out, os, clamp, s);
}

float row_dot(const Matrix& m, uint32_t row, const float* x, size_t xs, uint32_t tok) {
    Scratch s;
    ensure_scratch(s, 8, 8);
    float acc = 0.0f;
    const uint32_t one = tok;
    for (uint32_t i0 = 0; i0 < m.nidx; i0 += kColBlock) {
        const uint32_t i1 = std::min(i0 + kColBlock, m.nidx);
        gather_tile<1>(x, xs, &one, i0, i1, s.xblk.data());
        if (m.plane != nullptr)
            rowdot_block<1, true>(m, row, i0, i1, s.xblk.data(), &acc, 1);
        else
            rowdot_block<1, false>(m, row, i0, i1, s.xblk.data(), &acc, 1);
    }
    return acc * m.gain(row);
}

void run_jobs(const Job* jobs, uint32_t n_jobs, const float* x, size_t xs, uint32_t threads, float clamp) {
    if (n_jobs == 0) return;
    const uint32_t nw = std::max(1u, std::min(threads, n_jobs));
    if (nw == 1) {
        Scratch s;
        for (uint32_t j = 0; j < n_jobs; j++) run_job(jobs[j], x, xs, clamp, s);
        return;
    }

    std::atomic<uint32_t> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> pool;
    pool.reserve(nw);
    for (uint32_t tid = 0; tid < nw; tid++) {
        pool.emplace_back([&, tid] {
            Scratch s;
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) _mm_pause();
            for (uint32_t j = tid; j < n_jobs; j += nw) run_job(jobs[j], x, xs, clamp, s);
        });
    }
    while (ready.load(std::memory_order_acquire) != nw) _mm_pause();
    go.store(true, std::memory_order_release);
    for (auto& th : pool) th.join();
}

// ---- the persistent pool ------------------------------------------------------------------------------

struct JobPool::Impl {
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    const Job* jobs = nullptr;
    uint32_t n_jobs = 0;
    const float* x = nullptr;
    size_t xs = 0;
    float clamp = 0.0f;
    uint64_t gen = 0;
    uint32_t done = 0;
    bool quit = false;
    uint32_t nw = 0;
    std::chrono::steady_clock::time_point t0;
    double last = 0.0;
    std::vector<std::thread> threads;
};

JobPool::JobPool(uint32_t threads) : impl_(new Impl) {
    impl_->nw = std::max(1u, threads);
    impl_->done = impl_->nw;   // idle
    impl_->threads.reserve(impl_->nw);
    for (uint32_t tid = 0; tid < impl_->nw; tid++) {
        impl_->threads.emplace_back([this, tid] {
            Impl& p = *impl_;
            Scratch s;
            uint64_t seen = 0;
            for (;;) {
                const Job* jobs;
                uint32_t n_jobs, nw;
                const float* x;
                size_t xs;
                float clamp;
                {
                    std::unique_lock<std::mutex> lk(p.mu);
                    p.cv_work.wait(lk, [&] { return p.quit || p.gen != seen; });
                    if (p.quit) return;
                    seen = p.gen;
                    jobs = p.jobs;
                    n_jobs = p.n_jobs;
                    nw = p.nw;
                    x = p.x;
                    xs = p.xs;
                    clamp = p.clamp;
                }
                for (uint32_t j = tid; j < n_jobs; j += nw) run_job(jobs[j], x, xs, clamp, s);
                std::lock_guard<std::mutex> lk(p.mu);
                if (++p.done == p.nw) {
                    p.last = std::chrono::duration<double>(std::chrono::steady_clock::now() - p.t0).count();
                    p.cv_done.notify_one();
                }
            }
        });
    }
}

JobPool::~JobPool() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->quit = true;
    }
    impl_->cv_work.notify_all();
    for (auto& th : impl_->threads) th.join();
    delete impl_;
}

void JobPool::run(const Job* jobs, uint32_t n_jobs, const float* x, size_t xs, float clamp) {
    if (n_jobs == 0) return;
    Impl& p = *impl_;
    std::lock_guard<std::mutex> lk(p.mu);
    p.jobs = jobs;
    p.n_jobs = n_jobs;
    p.x = x;
    p.xs = xs;
    p.clamp = clamp;
    p.done = 0;
    p.t0 = std::chrono::steady_clock::now();
    ++p.gen;
    p.cv_work.notify_all();
}

void JobPool::wait() {
    Impl& p = *impl_;
    std::unique_lock<std::mutex> lk(p.mu);
    p.cv_done.wait(lk, [&] { return p.done == p.nw; });
}

double JobPool::last_sec() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->last;
}

}  // namespace strata::dsv4cpu
