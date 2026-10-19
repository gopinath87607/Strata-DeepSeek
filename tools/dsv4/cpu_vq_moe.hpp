// tools/dsv4/cpu_vq_moe.hpp - the routed VQ experts on the CPU, AVX2, eight tokens to a tile.
//
// This is the DeepSeek arm of the same split Qwen and Maya use: while the GPUs stage and compute their
// share of a layer's experts, the host computes the rest straight out of the resident RAM image.  The
// arithmetic mirrors the device arm (src/kernels/cuda/deepseek4_moe.cu): the activation is rounded to
// bf16 before the dot, each gate/up row dot is rounded to bf16 before SwiGLU, the mid is bf16, and the
// down row dot is rounded to bf16.  The accumulation order is not the device's xor tree, so results
// agree to rounding, not bit for bit.
//
// The kernel is the measured one (tools/dsv4/cpu_expert_bench.cpp): a 64-codeword column block of the
// activations is copied once into a 16 KB scratch for the tile, so the eight token loads become constant
// displacements off one register, and the stream rows are prefetched ahead of the sweep.
//
// v3 payloads only (the fixed 12-bit main stream plus the optional plane).  The artifact's routed experts
// are v3; a v2 container is refused by the caller rather than decoded as a different packing.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::dsv4cpu {

// One expert matrix, in host memory, with its layer's codebook expanded to bf16 (the decode multiplies
// bf16 weights; the codebook values are exact in bf16).
struct Matrix {
    const uint8_t* stream = nullptr;   // 12-bit main index stream, row-major
    const uint8_t* plane = nullptr;    // 13th-bit plane, or null
    const uint8_t* gains = nullptr;    // f16 per row
    const uint16_t* cb = nullptr;      // expanded codebook, nc * 8 bf16
    uint32_t rows = 0;                 // output rows
    uint32_t cols = 0;                 // input columns
    uint32_t nidx = 0;                 // cols / 8 codewords per row
    uint32_t mrow = 0;                 // bytes of main stream per row
    uint32_t prow = 0;                 // bytes of plane per row
    float gain(uint32_t row) const;
};

// What the parsers hand over, independent of which one opened the payload.
struct Layout {
    const uint8_t* codebook = nullptr;  // nc*8 E4M3 (v3) or nc*8 f16 (v2)
    const uint8_t* gains = nullptr;
    const uint8_t* stream = nullptr;
    const uint8_t* plane = nullptr;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t nc = 0;
    bool v3 = true;
};

// Expand a layer codebook into nc * 8 bf16 words (E4M3 exactly, or f16 exactly).
void expand_codebook(const uint8_t* codebook, uint32_t nc, bool e4m3, uint16_t* out);

// Fill `m` from `l`; `cb16` is expand_codebook's output and must outlive every job that uses `m`.
void open_matrix(const Layout& l, const uint16_t* cb16, Matrix* m);

// One expert for up to 8 tokens: out[t][o] is the bf16-rounded down row, stored as f32 (the device's
// partial).  x rows are x + toks[t] * xs floats, and are rounded to bf16 inside, as the device's xpack.
// `clamp` is the artifact's 10; a value <= 0 skips the clamp as the device does.
void expert_tile(const Matrix& gate, const Matrix& up, const Matrix& down, const float* x, size_t xs,
                 const uint32_t* toks, uint32_t n_tok, float* out, size_t os, float clamp);

// One gate row dot for one token, bf16-rounded as the device returns it.  For self-tests.
float row_dot(const Matrix& m, uint32_t row, const float* x, size_t xs, uint32_t tok);

// One expert for one group of tokens, output packed to bf16 bits.
struct Job {
    const Matrix* gate = nullptr;
    const Matrix* up = nullptr;
    const Matrix* down = nullptr;
    const uint32_t* toks = nullptr;  // token ids into x
    uint32_t n_tok = 0;
    uint16_t* out = nullptr;         // [n_tok][out_stride], bf16 bits
    uint32_t out_stride = 0;
};

// Run every job on `threads` workers and return when all are done.  Workers are spread over the host's
// cores, each with its own scratch; `x` is shared and read-only.
void run_jobs(const Job* jobs, uint32_t n_jobs, const float* x, size_t xs, uint32_t threads, float clamp);

// The same work on a persistent worker set, split into a start and a wait so the driver can issue the
// device arm in between: run() returns once the workers are awake, wait() blocks until every job is done.
// The host thread is then free to enqueue the GPU's arm (which blocks the stream), and the two overlap.
// The workers keep their scratch between calls.  One run at a time: call wait() before the next run().
class JobPool {
public:
    explicit JobPool(uint32_t threads);
    ~JobPool();
    JobPool(const JobPool&) = delete;
    JobPool& operator=(const JobPool&) = delete;
    void run(const Job* jobs, uint32_t n_jobs, const float* x, size_t xs, float clamp);
    void wait();
    // Wall time of the most recently completed run: run() to the last worker's finish, seconds.
    double last_sec();

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace strata::dsv4cpu
