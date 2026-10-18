// tools/dsv4/dsv4_run.cu - DeepSeek-V4.1-Flash forward driver (docs/DEEPSEEK4.md, P3 onward).
//
// One process, one or more GPUs (P7's layer split), greedy decode.  It reads the 40-part GGUF through a
// parts-aware reader (the header comes from part 1's map; every payload read is an offset into the assembled
// image), keeps the q4_K/BF16/F32 skeleton and the two engram wkv matrices resident, and streams each layer's
// selected routed experts from its DQVL blob into a compact device container per block (P7's layer split, auto
// chunk and opt-in expert cache are here; the loan and ring are not built yet; this is the correctness driver).
//
// The layer body is the reference's, call for call (reference/ds4-dfm-rs/ds4_ds41_forward.inc): engram ->
// hc_half(attn) -> attn_in -> attn_cache (compress/index source, sparse attention, ring commit) -> attn_out ->
// hc_post -> hc_half(ffn) -> MoE -> hc_post; head = hc_pre -> rms(output_norm) -> output projection -> argmax.
// The direct (non-graph) route's host-side counters are reproduced: compressed caches are written by absolute
// group number, a query only sees the groups completed before its block, and the topk used at a layer is the
// last index source's, at that source's row stride.
//
// Usage:
//   dsv4_run --ids 1,2,3 [--gen 32] [--model BASE] [--engram-dir DIR] [--chunk 8] [--ctx N] [--gpu 0]
//            [--devices 0,1,2,3] [--ledger] [--expert-cache-mib N[,N...]]
//            [--dspark [--dspark-verify K] [--dspark-trace]]
//            [--temperature T] [--top-p P] [--min-p M] [--top-k K] [--seed S] [--stop-eos] [--stream-ids]
//            [--engram-feed] [--ram-image]
//
// The runner takes token ids, not text: the tokenizer is Python (tools/strata_tokenizer.py); see
// tools/dsv4/dsv4_prompt.py for the text wrapper.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "strata/artifact/deepseek4.hpp"
#include "strata/artifact/deepseek4_engram.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/deepseek4_attn.hpp"
#include "strata/kernels/deepseek4_dense.hpp"
#include "strata/kernels/deepseek4_engram.hpp"
#include "strata/kernels/deepseek4_fp8.hpp"
#include "strata/kernels/deepseek4_hc.hpp"
#include "strata/kernels/deepseek4_moe.hpp"
#include "strata/kernels/deepseek4_vq.hpp"

namespace d4 = strata::deepseek4;
namespace k = strata::kernels;
using strata::GgufFile;
using strata::TensorInfo;

namespace {

constexpr uint32_t kQ4K = 12;    // ggml type ids this artifact uses
constexpr uint32_t kI32 = 26;
constexpr uint32_t kI64 = 27;
constexpr uint32_t kBF16 = 30;
constexpr uint32_t kF32 = 0;
constexpr uint32_t kFp8 = 44;
constexpr uint32_t kNExpertMax = 4096;   // the DQVL parser's own bound
constexpr uint32_t kDecRows = 10;        // DS4_MTP_MAX_BLOCK + 2: the index scorer's minimum row block
constexpr uint32_t kGemvMaxTok = 8;      // DS4_V41_GEMV_MAX_TOK: the MoE tail arm

double now_sec() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
uint64_t rd64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}
void wr32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, 4); }
void wr64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 8); }

// ---- The 40-part image ----------------------------------------------------------------------------------------

struct Part {
    std::string path;
    uint64_t base = 0;
    uint64_t size = 0;
    int fd = -1;
};

class Parts {
public:
    ~Parts() {
        for (auto& p : parts_)
            if (p.fd >= 0) ::close(p.fd);
        if (ram_ != nullptr) {
            if (ram_pinned_) cudaFreeHost(ram_);
            else std::free(ram_);
        }
    }

    // `model` is the base name (with or without the suffix) or a single assembled file.
    bool open(const std::string& model, std::string& err) {
        std::string first = model;
        const std::string suffix = ".part01-of-40";
        if (first.size() < suffix.size() || first.compare(first.size() - suffix.size(), suffix.size(), suffix) != 0) {
            first = model + suffix;
        }
        struct stat st {};
        if (::stat(first.c_str(), &st) == 0) {
            const std::string base = first.substr(0, first.size() - suffix.size());
            char name[64];
            uint64_t at = 0;
            for (uint32_t i = 1; i <= 40; ++i) {
                std::snprintf(name, sizeof name, ".part%02u-of-40", i);
                Part p;
                p.path = base + name;
                p.base = at;
                p.fd = ::open(p.path.c_str(), O_RDONLY | O_CLOEXEC);
                if (p.fd < 0) {
                    err = "cannot open " + p.path + ": " + std::strerror(errno);
                    return false;
                }
                struct stat ps {};
                if (::fstat(p.fd, &ps) != 0) {
                    err = "cannot stat " + p.path;
                    return false;
                }
                p.size = static_cast<uint64_t>(ps.st_size);
                at += p.size;
                parts_.push_back(p);
            }
            total_ = at;
            return true;
        }
        // A single assembled file.
        Part p;
        p.path = model;
        p.fd = ::open(model.c_str(), O_RDONLY | O_CLOEXEC);
        if (p.fd < 0) {
            err = "cannot open " + model + ": " + std::strerror(errno);
            return false;
        }
        struct stat ps {};
        ::fstat(p.fd, &ps);
        p.size = static_cast<uint64_t>(ps.st_size);
        total_ = p.size;
        parts_.push_back(p);
        return true;
    }

    uint64_t size() const { return total_; }   // the assembled image length
    size_t count() const { return parts_.size(); }
    const std::string& gguf_path() const { return parts_.front().path; }

    bool read(uint64_t abs, void* dst, uint64_t len, std::string& err) const {
        if (ram_ != nullptr) {
            if (abs > total_ || len > total_ - abs) {
                err = "read past the image end (offset " + std::to_string(abs) + ", " + std::to_string(len) + " bytes)";
                return false;
            }
            std::memcpy(dst, ram_ + abs, static_cast<size_t>(len));
            return true;
        }
        return read_disk(abs, dst, len, err);
    }

    // ---- The RAM image (--ram-image, opt-in) ----
    // The whole assembled image in host memory, read once at start.  Every Parts read becomes a memcpy, and
    // stage_blob_b copies an expert's slots straight from it to the device instead of assembling them in the
    // pageable staging vector.  Pinned when cudaHostAlloc takes it (a direct DMA source, the case that saves a
    // copy per miss); pageable otherwise, where the reads are still served from RAM.
    bool load_ram(std::string& err) {
        if (ram_ != nullptr) return true;
        const double t0 = now_sec();
        void* mem = nullptr;
        const cudaError_t ce = cudaHostAlloc(&mem, static_cast<size_t>(total_), cudaHostAllocPortable);
        ram_pinned_ = ce == cudaSuccess;
        if (!ram_pinned_) {
            (void) cudaGetLastError();
            mem = std::malloc(static_cast<size_t>(total_));
            if (mem == nullptr) {
                err = "RAM image: cannot allocate " + std::to_string(total_ / (1ull << 20)) + " MiB of host memory";
                return false;
            }
        }
        ram_ = static_cast<uint8_t*>(mem);
        std::atomic<bool> ok{true};
        std::atomic<uint64_t> next{0};
        std::string first_err;
        std::mutex mu;
        constexpr uint64_t kSlice = 64ull << 20;
        const uint32_t nthr = std::min<uint32_t>(8u, std::max<uint32_t>(1u, std::thread::hardware_concurrency() / 4));
        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < nthr; ++t) {
            pool.emplace_back([&]() {
                while (ok.load(std::memory_order_relaxed)) {
                    const uint64_t at = next.fetch_add(kSlice);
                    if (at >= total_) break;
                    const uint64_t n = std::min(kSlice, total_ - at);
                    std::string e;
                    if (!read_disk(at, ram_ + at, n, e)) {
                        std::lock_guard<std::mutex> g(mu);
                        if (ok.exchange(false)) first_err = e;
                        break;
                    }
                }
            });
        }
        for (auto& th : pool) th.join();
        if (!ok.load()) {
            err = first_err;
            return false;
        }
        const double secs = now_sec() - t0;
        std::fprintf(stderr, "[v41] ram image: %.1f GiB %s in %.2fs (%.0f MB/s)\n",
                     static_cast<double>(total_) / 1073741824.0,
                     ram_pinned_ ? "pinned" : "pageable (cudaHostAlloc refused)",
                     secs, static_cast<double>(total_) / 1e6 / (secs > 1e-9 ? secs : 1e-9));
        return true;
    }

    const uint8_t* ram_base() const { return ram_; }   // the RAM image, or null when the tier is off

private:
    bool read_disk(uint64_t abs, void* dst, uint64_t len, std::string& err) const {
        uint8_t* out = static_cast<uint8_t*>(dst);
        if (abs > total_ || len > total_ - abs) {
            err = "read past the image end (offset " + std::to_string(abs) + ", " + std::to_string(len) + " bytes)";
            return false;
        }
        while (len > 0) {
            size_t i = 0;
            while (i + 1 < parts_.size() && abs >= parts_[i + 1].base) ++i;
            const Part& p = parts_[i];
            const uint64_t off = abs - p.base;
            const uint64_t n = std::min(len, p.size - off);
            size_t got = 0;
            while (got < n) {
                const ssize_t r = ::pread(p.fd, out + got, static_cast<size_t>(n - got), static_cast<off_t>(off + got));
                if (r < 0) {
                    if (errno == EINTR) continue;
                    err = "read " + p.path + " failed: " + std::strerror(errno);
                    return false;
                }
                if (r == 0) {
                    err = "short read in " + p.path;
                    return false;
                }
                got += static_cast<size_t>(r);
            }
            out += n;
            abs += n;
            len -= n;
        }
        return true;
    }

public:
    std::string where(uint64_t abs) const {
        size_t i = 0;
        while (i + 1 < parts_.size() && abs >= parts_[i + 1].base) ++i;
        const std::string& path = parts_[i].path;
        const size_t slash = path.find_last_of('/');
        return slash == std::string::npos ? path : path.substr(slash + 1);
    }

private:
    std::vector<Part> parts_;
    uint64_t total_ = 0;
    uint8_t* ram_ = nullptr;      // the RAM image (--ram-image), or null
    bool ram_pinned_ = false;     // cudaHostAlloc took it: a direct DMA source for stage_blob_b
};

// ---- The expert blob ------------------------------------------------------------------------------------------

struct Blob {
    uint64_t abs = 0;          // payload offset in the assembled image
    uint64_t size = 0;         // the container's payload length
    uint32_t experts = 0;
    uint32_t ver = 3;          // container version (2 or 3; the tower blobs are the same format)
    bool v3 = true;            // payload magic DQV3 (shared codebook) or DQVQ (own codebook)
    uint32_t nc = 0;
    uint64_t cb_off = 0;       // shared codebook offset (v3: 16 + experts*24)
    uint64_t cb_len = 0;       // 0 on a DQVQ container
    std::vector<uint64_t> slot;   // experts * 3 payload offsets (0 = absent)
    std::vector<uint64_t> span;   // the packed payload length of each slot
};

// ---- Per-layer resident weights -------------------------------------------------------------------------------

struct LayerW {
    void* hc_attn_fn = nullptr;
    void* hc_attn_scale = nullptr;
    void* hc_attn_base = nullptr;
    void* attn_norm = nullptr;
    void* q_a = nullptr;
    void* q_a_norm = nullptr;
    void* q_b = nullptr;
    void* kv = nullptr;
    void* kv_a_norm = nullptr;
    void* sinks = nullptr;
    void* out_a = nullptr;
    void* out_b = nullptr;
    void* hc_ffn_fn = nullptr;
    void* hc_ffn_scale = nullptr;
    void* hc_ffn_base = nullptr;
    void* ffn_norm = nullptr;
    void* gate_inp = nullptr;   // BF16 [NE][E]
    void* exp_bias = nullptr;   // F32 [NE]
    void* shexp_gate = nullptr;
    void* shexp_up = nullptr;
    void* shexp_down = nullptr;
    // Compressor and indexer (present at the source layers).
    void* comp_kv = nullptr;    // BF16 [HD][E]
    void* comp_gate = nullptr;  // BF16 [HD][E]
    void* comp_norm = nullptr;  // F32 [HD]
    void* idx_proj = nullptr;   // BF16 [IH][E]
    void* idx_q_b = nullptr;    // Q4_K [Q][IH*IK]
    void* idx_wk = nullptr;     // BF16 [IK][HD]
    void* idx_k_norm = nullptr; // F32 [IK]
    // Engram (present at the engram layers).
    void* eng_q = nullptr;      // F32 [HC][E]
    void* eng_k = nullptr;      // F32 [HC][E]
    void* eng_wkv = nullptr;    // fp8_32x32: in_dim*out_dim e4m3 then the tile scales
};

// splitmix64, the seed expander for the sampler's xorshift128+ (P6).
uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Options {
    std::string model = "/mnt/nvme/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf";
    std::string engram_dir = "/mnt/nvme";
    std::vector<int32_t> ids;
    std::string ids_file;
    uint32_t gen = 0;
    uint32_t chunk = 8;
    uint32_t ctx = 0;   // 0: the model's own context (no reduction)
    int gpu = 0;
    // Layer split (P7): one gpu per stage, layers in order.  Empty keeps the single-device path on --gpu.  The
    // split only cuts at a legal cache seam (see open()); the run must return the same ids as the one-device run.
    std::vector<int> devices;
    bool trace = false;
    bool engram = true;
    // Engram feed (P7, opt-in): the reference's prepare/wait/prefetch_next (ds4_ds41_forward.inc).  The row ids
    // depend only on token ids and positions, so a worker thread reads the next block's rows from the shards
    // while the current block computes, and each engram layer then only waits for its buffer and uploads it.
    // Off by default: the synchronous path is unchanged and the default run stays byte-identical.
    bool engram_feed = false;
    bool check = true;   // the inventory check (header only)
    std::string logits_out;
    // DSpark: the MTP towers draft, the main forward verifies, the cache rolls back.  Off unless asked, and the
    // gate that matters is generated= byte-identical to this same run without it.
    bool dspark = false;
    uint32_t dspark_verify = 0;   // 0: verify the tower's whole block; else pin k (the trace gate)
    bool dspark_trace = false;    // print each round's k, accepted count, confidence and both id lists
    // Sampling (P6).  temperature <= 0 is the greedy path, byte-identical to the runs before this switch; a
    // positive temperature runs the server's chain (temperature, top_k, min_p, top_p) over the logits row the
    // argmax path already reads back, draws with a seeded xorshift128+.  The DSpark verify is greedy-only.
    float temperature = 0.0f;
    float top_p = 1.0f;
    float min_p = 0.0f;
    uint32_t top_k = 0;   // 0: off
    uint64_t seed = 0;
    bool stop_eos = false;   // stop after emitting the model's EOS (id 1 here); off keeps every gate run's ids
    bool stream_ids = false; // print each generated id as a line as it is produced, before the final generated=
    bool ledger = false;     // print the per-stage time/bytes ledger (P7); changes nothing else
    // Resident serving (P8, opt-in): load the weights once, size the state for --serve-ctx, then read requests
    // from stdin, one per line, and answer each with the same code path as a one-shot run.  Off by default; a
    // one-shot run is unchanged.  The expert cache (when asked for) also lives across requests here.
    bool serve = false;
    uint32_t serve_ctx = 16384;   // tokens the resident state is sized for (requests above it are refused by name)
    // Expert cache (P7, opt-in): device slots that keep each staged expert payload, keyed by (layer, expert).
    // A hit copies the payload from its slot into the container instead of reading it back from the parts.  One
    // number applies to every stage; a comma list gives one budget per stage (see --devices).  Empty is off and
    // changes nothing: the default path stays byte-identical to the run without this switch.
    std::vector<uint32_t> expert_cache_mib;
    // RAM image (opt-in): the whole assembled image in host memory, read once at start.  Off by default, and the
    // default path stays byte-identical: every read still goes to the parts.  On, an expert-cache miss is copied
    // straight from the (pinned, when the driver allows it) image to the device container instead of being read
    // from disk and assembled in the pageable staging buffer first.
    bool ram_image = false;
};

// ---- The runner -----------------------------------------------------------------------------------------------

class Runner {
public:
    ~Runner();
    bool open(const Options& o, std::string& err);
    bool load_weights(std::string& err);
    bool load_engram(std::string& err);
    bool run(const std::vector<int32_t>& prompt, uint32_t gen, std::string& err);
    // ---- Resident serving (--serve) ----
    // warm() sizes the state once; each request then resets the per-request state instead of reallocating it.
    bool warm(std::string& err);
    bool set_request_sampling(float temperature, float top_p, float min_p, uint32_t top_k, uint64_t seed,
                              bool stop_eos, std::string& err);
    uint32_t context_tokens() const { return ctx_; }

    d4::Geometry geo;
    std::string last_error;

private:
    // One stage of the layer split (P7).  Every shared forward buffer is allocated once per stage: the kernels
    // of stage s run on stage s's device and only touch that stage's copy.  The weights of a stage's layers and
    // their window/compressed-KV buffers live here too, bound per layer by load_weights and alloc_state.
    struct Stage {
        int dev = 0;
        cudaStream_t stream = nullptr;
        int32_t *d_tok = nullptr, *d_pos = nullptr;
        float* d_prem = nullptr;
        float *x = nullptr, *hc = nullptr, *hc2 = nullptr, *xn = nullptr, *mix = nullptr, *pre = nullptr;
        float *post = nullptr, *comb = nullptr, *qr = nullptr, *qrn = nullptr, *q = nullptr, *kvv = nullptr;
        float *kvn = nullptr, *o = nullptr, *low = nullptr, *attn_out = nullptr, *glog = nullptr, *rw = nullptr;
        float *partial = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr, *so = nullptr, *y = nullptr;
        float *logits = nullptr, *pooled = nullptr, *latent = nullptr, *ktmp = nullptr, *cke = nullptr;
        float *ckv = nullptr, *csc = nullptr, *iq = nullptr, *iw = nullptr, *iscore = nullptr, *iscratch = nullptr;
        int32_t *sel = nullptr, *idx = nullptr, *cand = nullptr, *posg = nullptr;
        uint32_t iscap = 0, isrows = 0, nbc = 0;
        uint8_t* eraw = nullptr;
        float* erows = nullptr;
        uint16_t* x16 = nullptr;
        float* ekv = nullptr;
        uint64_t blob_dcap = 0;
        void* blob_dev = nullptr;
        uint64_t stage_used = 0;
    };
    void bind_stage(uint32_t s);
    void save_stage();
    uint32_t stage_of(uint32_t il) const {
        uint32_t s = 0;
        while (s + 1 < nstages_ && il >= stage_first_[s + 1]) ++s;
        return s;
    }

    // device memory
    void* dev(uint64_t bytes, const char* what);
    void* upload(const void* host, uint64_t bytes, const char* what);
    void* load_tensor(const std::string& name, uint32_t type, const char* what);

    bool alloc_state(uint32_t total, std::string& err);
    bool prepare_request(uint32_t total, std::string& err);
    void reseed();
    bool ensure_index_scratch(uint32_t rows, uint32_t ng, std::string& err);
    bool load_blob(uint32_t il, std::string& err);
    bool load_blob_named(const std::string& name, bool tower, std::string& err, Blob& b);
    bool stage_blob(uint32_t il, const int32_t* sel, uint32_t n_pairs, std::string& err);
    bool stage_blob_b(const Blob& b, uint32_t layer_id, const int32_t* sel, uint32_t n_pairs, std::string& err);

    bool block(const int32_t* ids, uint32_t n, bool head, bool head_all, std::string& err);
    bool layer(uint32_t il, std::string& err);
    bool hc_half(uint32_t il, bool attn_half, std::string& err);
    bool attn(uint32_t il, std::string& err);
    bool compress_source(uint32_t il, uint32_t ratio, std::string& err);
    bool index_source(uint32_t il, uint32_t ratio, std::string& err);
    bool moe(uint32_t il, std::string& err);
    bool engram_layer(uint32_t il, std::string& err);
    // ---- Engram feed (P7, opt-in): the reference's prepare / wait / prefetch_next ----
    bool engram_feed_prepare(uint32_t pos0, const int32_t* ids, uint32_t n, std::string& err);
    void engram_feed_prefetch(uint32_t pos0, const int32_t* hist, const int32_t* ids, uint32_t n);
    bool engram_feed_wait(uint32_t ei, std::string& err);
    bool engram_feed_start(uint32_t ei, int slot, uint32_t pos0, const int32_t* hist, size_t hist_len,
                           const int32_t* ids, uint32_t n);
    void engram_feed_close();
    int32_t pick_token(std::string& err);
    int32_t sample_host(std::string& err);
    // ---- DSpark (P5): the draft towers, the verify batch and its rollback ----
    bool load_towers(std::string& err);
    bool draft_alloc(std::string& err);
    bool tower_half(uint32_t T, bool attn_half, std::string& err);
    bool tower_attn(uint32_t T, std::string& err);
    bool tower_moe(uint32_t T, std::string& err);
    bool tower_layer(uint32_t T, std::string& err);
    bool draft_block(uint32_t pos0, std::string& err);
    int draft_round(int32_t tok, uint32_t pos_main, std::string& err);
    bool spec_snapshot(uint32_t n, std::string& err);
    bool spec_rollback(uint32_t keep, std::string& err);
    void probe(const char* tag, void* d, uint64_t n_floats);
    float theta_of(uint32_t ratio) const {
        return ratio ? static_cast<float>(geo.compress_rope_freq_base) : static_cast<float>(geo.rope_freq_base);
    }
    uint32_t osl_of(uint32_t ratio) const {
        return ratio ? static_cast<uint32_t>(geo.rope_original_context_length) : 0u;
    }

    Options opt_;
    Parts parts_;
    std::unique_ptr<GgufFile> gguf_;
    uint32_t E_ = 0, V_ = 0, H_ = 0, HD_ = 0, Q_ = 0, OL_ = 0, NG_ = 0, NE_ = 0, KU_ = 0, FF_ = 0, HC_ = 0;
    uint32_t IH_ = 0, IK_ = 0, SWA_ = 0, nlayers_ = 0;
    uint32_t eos_ = 1;   // tokenizer.ggml.eos_token_id, read by open(); --stop-eos stops after emitting it
    uint32_t cap_ = 0, ctx_ = 0, topk_cap_ = 0;
    float rms_eps_ = 0, hc_eps_ = 0, route_scale_ = 1.5f, clamp_ = 10.0f;
    uint32_t sinkhorn_ = 0, nrot_ = 0;
    float rope_factor_ = 1, beta_fast_ = 32, beta_slow_ = 1;

    uint32_t kv_src_[64] {}, idx_src_[64] {};   // per layer
    bool is_kv_src_[64] {}, is_idx_src_[64] {}; // per layer
    uint32_t ng_src_[64] {};                    // per kv source layer: groups written so far
    uint32_t cpend_[64] {};                     // per ratio>1 kv source: pending rows
    int engram_of_[64];                         // per layer: engram table index, or -1
    uint32_t kcap_ = 0, cand_bs_ = 0, cand_src_layer_ = 0;
    int cand_owner_ = -1, idx_owner_ = -1;
    uint32_t idx_topk_ = 0;
    uint32_t n_past_ = 0;
    bool state_ready_ = false;    // --serve: the one state allocation is in place (see prepare_request)
    uint32_t state_total_ = 0;    // the token count it was sized for

    LayerW L_[64];
    void* tok_embd_ = nullptr;
    void* out_w_ = nullptr;
    void* out_norm_ = nullptr;
    Blob blob_[64];
    uint64_t blob_dcap_ = 0;
    void* blob_dev_ = nullptr;
    uint64_t stage_used_ = 0;
    std::vector<uint8_t> stage_;

    // state
    uint32_t cur_n_ = 0, cur_pos0_ = 0;
    int32_t* d_tok_ = nullptr;
    int32_t* d_pos_ = nullptr;
    float* d_prem_ = nullptr;
    float* x_ = nullptr;
    float *hc_ = nullptr, *hc2_ = nullptr;
    float *xn_ = nullptr, *mix_ = nullptr, *pre_ = nullptr, *post_ = nullptr, *comb_ = nullptr;
    float* qr_ = nullptr;
    float* qrn_ = nullptr;
    float* q_ = nullptr;
    float* kvv_ = nullptr;
    float* kvn_ = nullptr;
    float* o_ = nullptr;
    float* low_ = nullptr;
    float* attn_out_ = nullptr;
    float* glog_ = nullptr;
    int32_t* sel_ = nullptr;
    float* rw_ = nullptr;
    float* partial_ = nullptr;
    float* sg_ = nullptr;
    float* su_ = nullptr;
    float* sh_ = nullptr;
    float* so_ = nullptr;
    float* y_ = nullptr;
    float* logits_ = nullptr;
    std::vector<float*> win_;
    std::vector<uint8_t*> comp_kv_;
    std::vector<uint8_t*> index_k_;
    std::vector<float*> cpre_kv_;
    std::vector<float*> cpre_sc_;
    float *pooled_ = nullptr, *latent_ = nullptr, *ktmp_ = nullptr, *cke_ = nullptr, *ckv_ = nullptr;
    float* csc_ = nullptr;
    float* iq_ = nullptr;
    float* iw_ = nullptr;
    float* iscore_ = nullptr;
    int32_t* idx_ = nullptr;
    int32_t* cand_ = nullptr;
    float* iscratch_ = nullptr;
    int32_t* posg_ = nullptr;
    uint32_t iscap_ = 0, isrows_ = 0, nbc_ = 0;
    // engram device buffers
    uint8_t* eraw_ = nullptr;
    float* erows_ = nullptr;
    uint16_t* x16_ = nullptr;
    float* ekv_ = nullptr;
    uint32_t eng_cols_ = 0, eng_hd_ = 0, eng_in_dim_ = 0, eng_out_dim_ = 0;
    d4::EngramHash hash_;
    std::unique_ptr<d4::EngramShard> shard_[2];
    bool engram_on_ = false;

    // ---- DSpark (P5) ----
    bool mtp_on_ = false;
    uint32_t mtp_towers_ = 0, mtp_experts_ = 0, mtp_used_ = 0, mtp_block_ = 0, mtp_rank_ = 0;
    uint32_t mtp_noise_ = 0, mtp_n_target_ = 0, cap2_ = 0, prod_cap_ = 0, mainh_cap_ = 0;
    int mtp_slot_[64] {};        // per main layer: the ring slot of its capture, or -1
    LayerW TW_[8];               // per tower
    Blob tblob_[8];              // per tower expert container
    void* mtp_main_proj_ = nullptr;    // fp8_32x32 [n_target*E][E]
    void* mtp_main_norm_ = nullptr;    // f32 [E]
    void* mtp_markov_embd_ = nullptr;  // bf16 [R][V]
    void* mtp_markov_head_ = nullptr;  // bf16 [V][R]
    void* mtp_conf_ = nullptr;         // bf16 [1][E+R]
    void* mtp_out_norm_ = nullptr;     // f32 [E]
    bool tw_gate_f32_[8] {};           // the tower router gate's registered type (f32 or bf16)
    bool mtp_conf_f32_ = false;        // the confidence matrix's registered type (f32 or bf16)
    float* mainh_ = nullptr;           // [mainh_cap][n_target][E] f32 ring
    float* t_mainh_lin_ = nullptr;     // [SWA][n_target][E] the fill's contiguous run
    float* t_mainx_raw_ = nullptr;     // [prod_cap][E] main_proj output
    float* t_main_x_ = nullptr;        // [prod_cap][E] main_norm output
    uint32_t mainh_n_ = 0, mainh_wrote_ = 0;
    int64_t mainh_end_ = -1, win_end_ = -1;
    float* snap_win_ = nullptr;        // [nlayers][cap2][HD] the cells the batch overwrites
    float* snap_cpre_kv_[64] {};       // per ratio>1 kv source: [ratio+cap2][HD]
    float* snap_cpre_sc_[64] {};
    uint32_t snap_cpend_[64] {};
    uint32_t snap_n_ = 0, snap_past_ = 0;
    bool snap_on_ = false;
    // draft device state
    int32_t *t_tok_ = nullptr, *t_pos_ = nullptr, *t_sel_ = nullptr;
    int32_t *t_ids_ = nullptr, *t_ids_next_ = nullptr, *t_want_ = nullptr;
    float *t_premix_ = nullptr, *t_pre_ = nullptr, *t_hc_ = nullptr, *t_hc2_ = nullptr, *t_mix_ = nullptr;
    float *t_post_ = nullptr, *t_comb_ = nullptr, *t_x_ = nullptr, *t_xn_ = nullptr;
    float *t_qr_ = nullptr, *t_qrn_ = nullptr, *t_q_ = nullptr, *t_kv_ = nullptr, *t_kvn_ = nullptr;
    float *t_o_ = nullptr, *t_low_ = nullptr, *t_attn_ = nullptr;
    float *t_glog_ = nullptr, *t_rw_ = nullptr, *t_partial_ = nullptr, *t_routed_ = nullptr, *t_so_ = nullptr;
    float *t_sg_ = nullptr, *t_su_ = nullptr, *t_sh_ = nullptr;
    float* t_y_ = nullptr;
    float* t_logits_ = nullptr;        // [cap2][V] the draft block's per-row logits
    float* vlogits_ = nullptr;         // [cap2][V] the verify batch's per-row head output
    uint16_t* t_x16_ = nullptr;        // the fp8 towers' bf16 activation copy
    std::vector<float*> twin_;         // per tower [SWA + prod_cap][HD]
    float* t_wintmp_ = nullptr;        // [SWA][HD] the push's shift scratch
    float *t_mk_embed_ = nullptr, *t_mk_cur_ = nullptr, *t_mk_bias_ = nullptr;
    float *t_conf_in_ = nullptr, *t_conf_ = nullptr;
    std::vector<int32_t> t_tok_host_, t_bpos_host_, t_wpos_host_, t_want_host_, t_id_host_;
    std::vector<float> t_conf_host_;

    std::vector<int32_t> hist_;
    std::vector<int32_t> pm_host_;
    std::vector<float> probe_host_;
    std::vector<int64_t> erow_ids_;
    std::vector<uint8_t> eraw_host_;
    // The feed's two host buffers per engram layer, the row ids computed on the main thread (the hash is
    // stateful and stays there), and the payload the worker reads.  One job per (layer, slot); an event per
    // (layer, slot), recorded after the slot's upload, keeps a worker from overwriting a buffer whose H2D may
    // still be reading it.  `cur` is the slot the block in hand uses, -1 until the first prepare.
    struct FeedJob {
        std::thread th;
        bool running = false;
        bool done = false;   // `pos0`/`n` rows are in the slot
        uint32_t pos0 = 0, n = 0;
        bool ok = true;
        std::string err;
    };
    std::vector<std::array<FeedJob, 2>> efeed_;
    std::vector<std::array<std::vector<uint8_t>, 2>> efeed_raw_;
    std::vector<std::array<std::vector<int64_t>, 2>> efeed_ids_;
    std::vector<std::array<cudaEvent_t, 2>> efeed_ev_;
    std::vector<std::array<bool, 2>> efeed_ev_set_;
    std::vector<int> efeed_cur_;
    std::vector<int> efeed_dev_;
    std::vector<int32_t> pos_host_;
    std::vector<int32_t> dpos_host_;   // the block's positions, kept apart from pos_host_ (compress_source's scratch)
    std::vector<int32_t> sel_host_;
    std::vector<uint32_t> uniq_;
    std::vector<float> logits_host_;
    // P6 sampling: the sorted (score, id) work list and the seeded xorshift128+ state.  The readback that feeds
    // it is the one the argmax path already does, so a sampled step costs the sort, not another transfer.
    std::vector<std::pair<float, uint32_t>> samp_work_;
    uint64_t rng_s0_ = 0, rng_s1_ = 0;
    uint64_t rng_next() {
        uint64_t x = rng_s0_, y = rng_s1_;
        rng_s0_ = y;
        x ^= x << 23;
        rng_s1_ = x ^ y ^ (x >> 17) ^ (y >> 26);
        return rng_s1_ + y;
    }
    double rng_uniform() { return static_cast<double>(rng_next() >> 11) * 0x1.0p-53; }

    cudaStream_t stream_ = nullptr;

    // ---- P7 layer split ----
    std::vector<Stage> st_;
    uint32_t nstages_ = 1;
    uint32_t stage_cur_ = 0;
    std::vector<uint32_t> stage_first_;   // stage s owns layers [stage_first_[s], stage_first_[s + 1])
    std::vector<float> handoff_hc_;       // the stage boundary's host staging (hidden state)
    std::vector<float> handoff_prem_;     // ... and the evolved premix
    // --ledger (P7): prefill and decode seconds per stage, the hand-off, and what each stage staged.
    double st_pre_sec_[8] {}, st_dec_sec_[8] {}, st_hand_sec_[8] {};
    uint64_t st_bytes_[8] {}, st_experts_[8] {}, st_slots_[8] {}, st_new_[8] {};
    std::vector<uint8_t> seen_exp_;   // (layer, expert) slots staged at least once (--ledger's repeat count)
    uint64_t st_blocks_pre_ = 0, st_blocks_dec_ = 0;
    bool in_decode_ = false;

    // ---- P7 expert cache (opt-in) ----
    // Fixed-size slots per stage, one slot per cached expert payload.  `exp_cache_.size() == nstages_`; an empty
    // map is a stage with the cache off.  A miss that does not fit evicts the least recently used slot that this
    // container does not already point at.  Key is layer * NE_ + expert.
    struct CacheEnt { uint64_t len = 0, stamp = 0; uint32_t slot = 0; };
    std::vector<std::unordered_map<uint64_t, CacheEnt>> exp_cache_;
    std::vector<uint8_t*> exp_cache_dev_;      // slot base per stage
    std::vector<uint64_t> exp_cache_slots_;    // slots allocated per stage
    std::vector<uint64_t> exp_cache_next_;     // bump cursor per stage
    uint64_t exp_cache_stride_ = 0;            // bytes per slot (the largest expert payload)
    uint64_t cache_tick_ = 0;
    // --ledger: what the cache answered, and the bytes those hits did not have to read.
    uint64_t st_cache_hit_[8] {}, st_cache_miss_[8] {}, st_cache_bytes_[8] {};
};

void* Runner::dev(uint64_t bytes, const char* what) {
    void* p = nullptr;
    if (bytes == 0) return nullptr;
    const cudaError_t e = cudaMalloc(&p, static_cast<size_t>(bytes));
    if (e != cudaSuccess) {
        last_error = std::string("cudaMalloc ") + what + " (" + std::to_string(bytes) + " bytes): " +
                     cudaGetErrorString(e);
        return nullptr;
    }
    return p;
}

void* Runner::upload(const void* host, uint64_t bytes, const char* what) {
    void* d = dev(bytes, what);
    if (d == nullptr) return nullptr;
    const cudaError_t e = cudaMemcpy(d, host, static_cast<size_t>(bytes), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        last_error = std::string("upload ") + what + ": " + cudaGetErrorString(e);
        cudaFree(d);
        return nullptr;
    }
    return d;
}

void* Runner::load_tensor(const std::string& name, uint32_t type, const char* what) {
    const TensorInfo* t = gguf_->find(name);
    if (t == nullptr) {
        last_error = "missing tensor '" + name + "'";
        return nullptr;
    }
    if (t->type != type) {
        last_error = "tensor '" + name + "' is type " + std::to_string(t->type) + ", expected " + std::to_string(type);
        return nullptr;
    }
    const uint64_t bytes = d4::payload_bytes(*t);
    if (bytes == 0) {
        last_error = "tensor '" + name + "' has no payload size";
        return nullptr;
    }
    std::vector<uint8_t> host(static_cast<size_t>(bytes));
    std::string err;
    if (!parts_.read(gguf_->data_start() + t->offset, host.data(), bytes, err)) {
        last_error = "tensor '" + name + "': " + err;
        return nullptr;
    }
    return upload(host.data(), bytes, what);
}

void Runner::probe(const char* tag, void* d, uint64_t n_floats) {
    if (probe_host_.size() < n_floats) probe_host_.resize(static_cast<size_t>(n_floats));
    const cudaError_t e = cudaMemcpyAsync(probe_host_.data(), d, n_floats * sizeof(float),
                                          cudaMemcpyDeviceToHost, stream_);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[trace] %s copy failed\n", tag);
        return;
    }
    cudaStreamSynchronize(stream_);
    float mx = 0.f;
    uint64_t nf = 0;
    for (uint64_t i = 0; i < n_floats; ++i) {
        const float v = probe_host_[static_cast<size_t>(i)];
        if (!(v == v) || v == INFINITY || v == -INFINITY) ++nf;
        const float a = v < 0 ? -v : v;
        if (a > mx) mx = a;
    }
    std::fprintf(stderr, "  [trace] %s max %.6g nonfinite %" PRIu64 "\n", tag, static_cast<double>(mx), nf);
}

// Point every shared forward buffer at stage s's copies and make s's device current.  The swap-carrying members
// (hc_/hc2_, pre_/d_prem_) always come back to their stage's saved roles between stages: a layer swaps each pair
// twice, and alloc_state lays each stage out in the same roles.
void Runner::bind_stage(uint32_t s) {
    const Stage& g = st_[s];
    stage_cur_ = s;
    cudaSetDevice(g.dev);
    stream_ = g.stream;
    d_tok_ = g.d_tok;
    d_pos_ = g.d_pos;
    d_prem_ = g.d_prem;
    x_ = g.x;
    hc_ = g.hc;
    hc2_ = g.hc2;
    xn_ = g.xn;
    mix_ = g.mix;
    pre_ = g.pre;
    post_ = g.post;
    comb_ = g.comb;
    qr_ = g.qr;
    qrn_ = g.qrn;
    q_ = g.q;
    kvv_ = g.kvv;
    kvn_ = g.kvn;
    o_ = g.o;
    low_ = g.low;
    attn_out_ = g.attn_out;
    glog_ = g.glog;
    sel_ = g.sel;
    rw_ = g.rw;
    partial_ = g.partial;
    sg_ = g.sg;
    su_ = g.su;
    sh_ = g.sh;
    so_ = g.so;
    y_ = g.y;
    logits_ = g.logits;
    pooled_ = g.pooled;
    latent_ = g.latent;
    ktmp_ = g.ktmp;
    cke_ = g.cke;
    ckv_ = g.ckv;
    csc_ = g.csc;
    iq_ = g.iq;
    iw_ = g.iw;
    iscore_ = g.iscore;
    iscratch_ = g.iscratch;
    idx_ = g.idx;
    cand_ = g.cand;
    posg_ = g.posg;
    iscap_ = g.iscap;
    isrows_ = g.isrows;
    nbc_ = g.nbc;
    eraw_ = g.eraw;
    erows_ = g.erows;
    x16_ = g.x16;
    ekv_ = g.ekv;
    blob_dcap_ = g.blob_dcap;
    blob_dev_ = g.blob_dev;
    stage_used_ = g.stage_used;
}

// The inverse: stage_cur_'s record takes the members' current values (buffer growth, index scratch sizing, the
// expert container).  Called on the way out of a stage, so the next entry reuses instead of reallocating.
void Runner::save_stage() {
    Stage& g = st_[stage_cur_];
    g.d_tok = d_tok_;
    g.d_pos = d_pos_;
    g.d_prem = d_prem_;
    g.x = x_;
    g.hc = hc_;
    g.hc2 = hc2_;
    g.xn = xn_;
    g.mix = mix_;
    g.pre = pre_;
    g.post = post_;
    g.comb = comb_;
    g.qr = qr_;
    g.qrn = qrn_;
    g.q = q_;
    g.kvv = kvv_;
    g.kvn = kvn_;
    g.o = o_;
    g.low = low_;
    g.attn_out = attn_out_;
    g.glog = glog_;
    g.sel = sel_;
    g.rw = rw_;
    g.partial = partial_;
    g.sg = sg_;
    g.su = su_;
    g.sh = sh_;
    g.so = so_;
    g.y = y_;
    g.logits = logits_;
    g.pooled = pooled_;
    g.latent = latent_;
    g.ktmp = ktmp_;
    g.cke = cke_;
    g.ckv = ckv_;
    g.csc = csc_;
    g.iq = iq_;
    g.iw = iw_;
    g.iscore = iscore_;
    g.iscratch = iscratch_;
    g.idx = idx_;
    g.cand = cand_;
    g.posg = posg_;
    g.iscap = iscap_;
    g.isrows = isrows_;
    g.nbc = nbc_;
    g.eraw = eraw_;
    g.erows = erows_;
    g.x16 = x16_;
    g.ekv = ekv_;
    g.blob_dcap = blob_dcap_;
    g.blob_dev = blob_dev_;
    g.stage_used = stage_used_;
}

bool Runner::open(const Options& o, std::string& err) {
    opt_ = o;
    if (!parts_.open(o.model, err)) return false;
    std::fprintf(stderr, "[v41] image %s: %" PRIu64 " bytes in %zu part(s)\n", parts_.gguf_path().c_str(),
                 parts_.size(), parts_.count());
    if (opt_.ram_image && !parts_.load_ram(err)) return false;
    try {
        gguf_ = std::make_unique<GgufFile>(parts_.gguf_path());
    } catch (const std::exception& e) {
        err = std::string("GGUF header: ") + e.what();
        return false;
    }
    if (!(err = d4::read_geometry(*gguf_, geo)).empty()) return false;

    if (opt_.check) {
        const d4::Inventory inv = d4::check_inventory(*gguf_, geo, true, parts_.size());
        if (!inv.errors.empty()) {
            err = "inventory: " + std::to_string(inv.errors.size()) + " error(s); first: " + inv.errors[0];
            return false;
        }
        std::fprintf(stderr, "[v41] inventory: %" PRIu64 " tensors, %.1f GiB, all payloads inside the image\n",
                     inv.tensors, static_cast<double>(inv.bytes) / 1073741824.0);
    }

    nlayers_ = static_cast<uint32_t>(geo.block_count);
    E_ = static_cast<uint32_t>(geo.embedding_length);
    V_ = static_cast<uint32_t>(geo.vocab_size);
    H_ = static_cast<uint32_t>(geo.head_count);
    HD_ = static_cast<uint32_t>(geo.key_length);
    Q_ = static_cast<uint32_t>(geo.q_lora_rank);
    OL_ = static_cast<uint32_t>(geo.output_lora_rank);
    NG_ = static_cast<uint32_t>(geo.output_group_count);
    NE_ = static_cast<uint32_t>(geo.expert_count);
    KU_ = static_cast<uint32_t>(geo.expert_used_count);
    FF_ = static_cast<uint32_t>(geo.expert_feed_forward_length);
    HC_ = static_cast<uint32_t>(geo.hyper_connection_count);
    IH_ = static_cast<uint32_t>(geo.indexer_head_count);
    IK_ = static_cast<uint32_t>(geo.indexer_key_length);
    SWA_ = static_cast<uint32_t>(geo.sliding_window);
    rms_eps_ = static_cast<float>(geo.rms_epsilon);
    hc_eps_ = static_cast<float>(geo.hyper_connection_epsilon);
    route_scale_ = static_cast<float>(geo.expert_weights_scale);
    sinkhorn_ = static_cast<uint32_t>(geo.sinkhorn_iterations);
    nrot_ = static_cast<uint32_t>(geo.rope_dimension_count);
    rope_factor_ = static_cast<float>(geo.rope_scaling_factor);
    beta_fast_ = static_cast<float>(geo.yarn_beta_fast);
    beta_slow_ = static_cast<float>(geo.yarn_beta_slow);
    cand_bs_ = static_cast<uint32_t>(geo.candidate_block_size);
    kcap_ = static_cast<uint32_t>(geo.candidate_topk_blocks);
    cand_src_layer_ = static_cast<uint32_t>(geo.candidate_source_layer);
    clamp_ = static_cast<float>(geo.swiglu_clamp_exp[0]);
    if (geo.swiglu_clamp_exp.size() < nlayers_) { err = "swiglu_clamp_exp is short"; return false; }
    if (geo.compress_ratios.size() < nlayers_) { err = "compress_ratios is short"; return false; }
    for (uint32_t l = 1; l < nlayers_; ++l)
        if (geo.swiglu_clamp_exp[l] != geo.swiglu_clamp_exp[0]) {
            err = "the swiglu clamp is not uniform across layers; the driver uses one limit";
            return false;
        }
    if (nlayers_ > 64) { err = "more than 64 layers"; return false; }
    if (NE_ > kNExpertMax) { err = "expert count above the parser's bound"; return false; }
    if (KU_ > k::kDeepseek4MoeMaxTopk) { err = "top-k above the kernel's bound"; return false; }
    if (H_ % 8 != 0 || HD_ % 32 != 0 || HD_ > 512) { err = "head shape outside the kernels' rules"; return false; }
    if (H_ % k::kDeepseek4AttnHeadsPerBlock != 0) { err = "head count is not a multiple of 8"; return false; }
    if (IK_ % 32 != 0 || IK_ > 128) { err = "indexer key length outside the kernel's rules"; return false; }
    if (H_ % NG_ != 0) { err = "output_group_count does not divide head_count"; return false; }
    if (cand_bs_ == 0) { err = "candidate block size is zero"; return false; }

    // Wiring: a layer's compressed-KV source and index source are the nearest source at or before it; a kv source
    // is also an index source (the reference's table, ds41_forward.h).
    int kv_cur = -1, idx_cur = -1;
    for (uint32_t il = 0; il < nlayers_; ++il) {
        is_kv_src_[il] = d4::detail::contains(geo.kv_source_layers, il);
        is_idx_src_[il] = d4::detail::contains(geo.index_source_layers, il);
        if (is_kv_src_[il]) kv_cur = static_cast<int>(il);
        if (is_idx_src_[il]) idx_cur = static_cast<int>(il);
        kv_src_[il] = kv_cur < 0 ? 0u : static_cast<uint32_t>(kv_cur);
        idx_src_[il] = idx_cur < 0 ? 0u : static_cast<uint32_t>(idx_cur);
        const uint32_t ratio = static_cast<uint32_t>(geo.compress_ratios[il]);
        if (ratio > 0 && kv_cur < 0) { err = "compressed layer before any kv source"; return false; }
        if (is_kv_src_[il] && !is_idx_src_[il]) { err = "a kv source is not an index source"; return false; }
        if (ratio > 0 && static_cast<uint32_t>(geo.compress_ratios[kv_src_[il]]) != ratio) {
            err = "layer " + std::to_string(il) + ": its ratio differs from its kv source's";
            return false;
        }
        if (ratio > 1 && ratio != 2) {
            err = "layer " + std::to_string(il) + ": a compressor ratio other than 2 is not ported";
            return false;
        }
    }
    for (uint32_t il = 0; il < 64; ++il) engram_of_[il] = -1;
    for (size_t i = 0; i < geo.engram_layer_ids.size(); ++i) {
        if (geo.engram_layer_ids[i] >= nlayers_) { err = "engram layer outside the block count"; return false; }
        engram_of_[geo.engram_layer_ids[i]] = static_cast<int>(i);
    }

    ctx_ = opt_.ctx == 0 ? static_cast<uint32_t>(geo.context_length) : opt_.ctx;
    if (ctx_ > geo.context_length) ctx_ = static_cast<uint32_t>(geo.context_length);
    cap_ = std::max(1u, opt_.chunk);
    // `--chunk 0` is auto: the block is picked in alloc_state, after the weights and the engram are resident and
    // each stage's free memory is known.  Until then cap_ stays 1; only prod_cap_ (max(cap_, SWA_)) is read before
    // then, and it is SWA either way.
    if (cap_ > ctx_) cap_ = ctx_;

    // DSpark: the draft towers' metadata.  The path is opt-in; without --dspark the run is the plain one.
    if (opt_.dspark) {
        constexpr uint32_t kMtpMaxTowers = 8, kMtpMaxBlock = 8;   // the reference's DS4_MTP_MAX_TOWERS / _BLOCK
        if (geo.mtp_tower_count == 0 || geo.mtp_target_layers.empty()) {
            err = "--dspark: the image carries no MTP tensors";
            return false;
        }
        if (geo.mtp_tower_count > kMtpMaxTowers) { err = "--dspark: more than 8 towers"; return false; }
        if (geo.mtp_block_size == 0 || geo.mtp_block_size > kMtpMaxBlock) {
            err = "--dspark: the block size is 0 or above 8";
            return false;
        }
        if (geo.mtp_expert_count == 0 || geo.mtp_expert_count > kNExpertMax) {
            err = "--dspark: the tower expert count is outside the parser's bound";
            return false;
        }
        if (geo.mtp_expert_used_count == 0 ||
            geo.mtp_expert_used_count > k::kDeepseek4MoeMaxTopk) {
            err = "--dspark: the tower top-k is outside the kernel's bound";
            return false;
        }
        if (geo.mtp_markov_rank == 0) { err = "--dspark: markov_rank is zero"; return false; }
        mtp_towers_ = static_cast<uint32_t>(geo.mtp_tower_count);
        mtp_experts_ = static_cast<uint32_t>(geo.mtp_expert_count);
        mtp_used_ = static_cast<uint32_t>(geo.mtp_expert_used_count);
        mtp_block_ = static_cast<uint32_t>(geo.mtp_block_size);
        mtp_noise_ = static_cast<uint32_t>(geo.mtp_noise_token_id);
        mtp_rank_ = static_cast<uint32_t>(geo.mtp_markov_rank);
        mtp_n_target_ = mtp_towers_;
        for (uint32_t il = 0; il < 64; ++il) mtp_slot_[il] = -1;
        for (uint32_t T = 0; T < mtp_towers_; ++T) {
            const uint64_t tl = geo.mtp_target_layers[T];
            if (tl >= nlayers_) { err = "--dspark: an mtp target layer is not a block"; return false; }
            mtp_slot_[tl] = static_cast<int>(T);
        }
        cap2_ = mtp_block_ + 1;   // the verify batch runs up to 1+block rows
        if (cap2_ > ctx_) { err = "--dspark: the verify batch is wider than the context"; return false; }
        if (cap2_ > cap_) cap_ = cap2_;   // every per-block buffer must hold the verify batch
        if (cap_ > ctx_) cap_ = ctx_;
        mtp_on_ = true;
    }
    prod_cap_ = std::max(cap_, SWA_);

    // P6 sampling: the wire accepts the server's defaults; the engine refuses what its chain cannot mean.
    if (opt_.temperature < 0.0f) { err = "--temperature is negative"; return false; }
    if (!(opt_.top_p > 0.0f && opt_.top_p <= 1.0f)) { err = "--top-p is outside (0, 1]"; return false; }
    if (!(opt_.min_p >= 0.0f && opt_.min_p < 1.0f)) { err = "--min-p is outside [0, 1)"; return false; }
    if (opt_.top_k > V_) opt_.top_k = V_;
    if (opt_.temperature > 0.0f && opt_.dspark) {
        err = "--dspark verifies by argmax; drafting and sampling cannot be combined yet";
        return false;
    }
    reseed();
    if (const strata::MetaValue* e = gguf_->get("tokenizer.ggml.eos_token_id")) {
        if (e->is_num() && e->num() >= 0.0 && e->num() < static_cast<double>(V_))
            eos_ = static_cast<uint32_t>(e->num());
    }
    if (opt_.stop_eos)
        std::fprintf(stderr, "[v41] stop at EOS (id %u); sampling temperature %.3f top_p %.3f min_p %.3f top_k %u\n",
                     eos_, static_cast<double>(opt_.temperature), static_cast<double>(opt_.top_p),
                     static_cast<double>(opt_.min_p), opt_.top_k);

    std::fprintf(stderr,
                 "[v41] model: %u layers, E %u, V %u, H %u x HD %u, Q %u, OL %u x NG %u, experts %u top %u FF %u,\n"
                 "      HC %u sinkhorn %u, indexer %u x %u top %" PRIu64 ", sources",
                 nlayers_, E_, V_, H_, HD_, Q_, OL_, NG_, NE_, KU_, FF_, HC_, sinkhorn_, IH_, IK_,
                 geo.indexer_top_k);
    for (size_t i = 0; i < geo.kv_source_layers.size(); ++i)
        std::fprintf(stderr, " %" PRIu64, geo.kv_source_layers[i]);
    std::fprintf(stderr, " (ratios");
    for (size_t i = 0; i < geo.kv_source_layers.size(); ++i)
        std::fprintf(stderr, " %" PRIu64, geo.compress_ratios[geo.kv_source_layers[i]]);
    char chunk_txt[24];
    if (opt_.chunk == 0) std::snprintf(chunk_txt, sizeof chunk_txt, "auto");
    else std::snprintf(chunk_txt, sizeof chunk_txt, "%u", cap_);
    std::fprintf(stderr, "), candidate L%u x %u blocks of %u,\n      ctx %u (model %" PRIu64
                         "), chunk %s, engram %s\n",
                 cand_src_layer_, kcap_, cand_bs_, ctx_, geo.context_length, chunk_txt,
                 opt_.engram ? "on" : "off");

    // ---- The stage plan (P7) -------------------------------------------------------------------------------
    // A boundary is legal only at a cache seam: no layer at or after it may take its compressed-KV or index
    // source from before it, and the candidate block's owner must sit in the same stage as its users.  On this
    // model the legal boundaries are 2, 8, 14 and 20 (each kv source starts a stage).  Among the legal sets we
    // take the one with the smallest largest stage, so no card carries more layers than it must.
    if (opt_.devices.empty()) opt_.devices.push_back(opt_.gpu);
    nstages_ = static_cast<uint32_t>(opt_.devices.size());
    if (nstages_ > 8) { err = "--devices: more than 8 stages"; return false; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess) { err = "cudaGetDeviceCount failed"; return false; }
    for (uint32_t s = 0; s < nstages_; ++s) {
        if (opt_.devices[s] < 0 || opt_.devices[s] >= ndev) {
            err = "--devices: gpu " + std::to_string(opt_.devices[s]) + " is outside 0.." +
                  std::to_string(ndev - 1);
            return false;
        }
        for (uint32_t q = 0; q < s; ++q)
            if (opt_.devices[q] == opt_.devices[s]) {
                err = "--devices repeats gpu " + std::to_string(opt_.devices[s]);
                return false;
            }
    }
    if (opt_.dspark && nstages_ > 1) {
        err = "--dspark runs on one device: its capture and rollback span every layer";
        return false;
    }
    stage_first_.assign(nstages_ + 1, 0);
    stage_first_[nstages_] = nlayers_;
    if (nstages_ > 1) {
        std::vector<uint32_t> legal;
        for (uint32_t b = 1; b < nlayers_; ++b) {
            bool ok = true;
            for (uint32_t il = b; il < nlayers_ && ok; ++il) {
                if (kv_src_[il] < b || idx_src_[il] < b) ok = false;
                else if (il > cand_src_layer_ && cand_src_layer_ < b) ok = false;
            }
            if (ok) legal.push_back(b);
        }
        const uint32_t K = nstages_ - 1;
        if (legal.size() < K) {
            err = "--devices: the model's compressed-cache layout allows at most " +
                  std::to_string(legal.size() + 1) + " stage(s)";
            return false;
        }
        // dp[j][i]: the smallest largest stage when j boundaries are placed and legal[i] is the last of them.
        const uint64_t kInf = ~0ull;
        std::vector<std::vector<uint64_t>> dp(K + 1, std::vector<uint64_t>(legal.size(), kInf));
        std::vector<std::vector<int>> par(K + 1, std::vector<int>(legal.size(), -1));
        for (size_t i = 0; i < legal.size(); ++i) dp[1][i] = legal[i];
        for (uint32_t j = 2; j <= K; ++j)
            for (size_t i = 0; i < legal.size(); ++i)
                for (size_t p = 0; p < i; ++p) {
                    if (dp[j - 1][p] == kInf) continue;
                    const uint64_t m = std::max(dp[j - 1][p], static_cast<uint64_t>(legal[i] - legal[p]));
                    if (m < dp[j][i]) { dp[j][i] = m; par[j][i] = static_cast<int>(p); }
                }
        uint64_t best = kInf;
        int bi = -1;
        for (size_t i = 0; i < legal.size(); ++i) {
            if (dp[K][i] == kInf) continue;
            const uint64_t m = std::max(dp[K][i], static_cast<uint64_t>(nlayers_) - legal[i]);
            if (m < best) { best = m; bi = static_cast<int>(i); }
        }
        if (bi < 0) { err = "--devices: no legal layer split for this stage count"; return false; }
        for (uint32_t j = K; j >= 1; --j) {
            stage_first_[j] = legal[static_cast<size_t>(bi)];
            bi = par[j][bi];
        }
    }

    st_.assign(nstages_, Stage{});
    for (uint32_t s = 0; s < nstages_; ++s) {
        st_[s].dev = opt_.devices[s];
        const cudaError_t ce = cudaSetDevice(st_[s].dev);
        if (ce != cudaSuccess) {
            err = "cudaSetDevice " + std::to_string(st_[s].dev) + ": " + cudaGetErrorString(ce);
            return false;
        }
        if (cudaStreamCreate(&st_[s].stream) != cudaSuccess) { err = "cudaStreamCreate failed"; return false; }
    }
    if (nstages_ > 1) {
        std::string map;
        for (uint32_t s = 0; s < nstages_; ++s)
            map += (s ? ", " : "") + std::to_string(stage_first_[s]) + ".." +
                   std::to_string(stage_first_[s + 1] - 1) + " on gpu " + std::to_string(st_[s].dev);
        std::fprintf(stderr, "[v41] stages: %s\n", map.c_str());
    }
    bind_stage(0);
    return true;
}

bool Runner::load_weights(std::string& err) {
    for (uint32_t il = 0; il < nlayers_; ++il) {
        bind_stage(stage_of(il));
        const std::string p = "blk." + std::to_string(il) + ".";
        LayerW& l = L_[il];
        auto need = [&](const char* suffix, uint32_t type) -> void* {
            void* w = load_tensor(p + suffix, type, suffix);
            if (w == nullptr) std::fprintf(stderr, "  L%u: %s\n", il, last_error.c_str());
            return w;
        };
        l.hc_attn_fn = need("hc_attn_fn.weight", kF32);
        l.hc_attn_scale = need("hc_attn_scale.weight", kF32);
        l.hc_attn_base = need("hc_attn_base.weight", kF32);
        l.attn_norm = need("attn_norm.weight", kF32);
        l.q_a = need("attn_q_a.weight", kQ4K);
        l.q_a_norm = need("attn_q_a_norm.weight", kF32);
        l.q_b = need("attn_q_b.weight", kQ4K);
        l.kv = need("attn_kv.weight", kQ4K);
        l.kv_a_norm = need("attn_kv_a_norm.weight", kF32);
        l.sinks = need("attn_sinks.weight", kF32);
        l.out_a = need("attn_output_a.weight", kQ4K);
        l.out_b = need("attn_output_b.weight", kQ4K);
        l.hc_ffn_fn = need("hc_ffn_fn.weight", kF32);
        l.hc_ffn_scale = need("hc_ffn_scale.weight", kF32);
        l.hc_ffn_base = need("hc_ffn_base.weight", kF32);
        l.ffn_norm = need("ffn_norm.weight", kF32);
        l.gate_inp = need("ffn_gate_inp.weight", kBF16);
        l.exp_bias = need("exp_probs_b.bias", kF32);
        l.shexp_gate = need("ffn_gate_shexp.weight", kQ4K);
        l.shexp_up = need("ffn_up_shexp.weight", kQ4K);
        l.shexp_down = need("ffn_down_shexp.weight", kQ4K);
        if (!l.hc_attn_fn || !l.hc_attn_scale || !l.hc_attn_base || !l.attn_norm || !l.q_a || !l.q_a_norm ||
            !l.q_b || !l.kv || !l.kv_a_norm || !l.sinks || !l.out_a || !l.out_b || !l.hc_ffn_fn ||
            !l.hc_ffn_scale || !l.hc_ffn_base || !l.ffn_norm || !l.gate_inp || !l.exp_bias ||
            !l.shexp_gate || !l.shexp_up || !l.shexp_down) {
            err = "layer " + std::to_string(il) + ": " + last_error;
            return false;
        }
        if (is_kv_src_[il]) {
            l.comp_kv = need("attn_compressor_kv.weight", kBF16);
            l.comp_norm = need("attn_compressor_norm.weight", kF32);
            const uint32_t ratio = static_cast<uint32_t>(geo.compress_ratios[il]);
            if (ratio == 2) l.comp_gate = need("attn_compressor_gate.weight", kBF16);
            if (!l.comp_kv || !l.comp_norm || (ratio == 2 && !l.comp_gate)) {
                err = "layer " + std::to_string(il) + ": " + last_error;
                return false;
            }
        }
        if (is_idx_src_[il]) {
            l.idx_proj = need("indexer.proj.weight", kBF16);
            l.idx_q_b = need("indexer.attn_q_b.weight", kQ4K);
            if (is_kv_src_[il]) {
                l.idx_wk = need("indexer.wk.weight", kBF16);
                l.idx_k_norm = need("indexer.k_norm.weight", kF32);
            }
            if (!l.idx_proj || !l.idx_q_b || (is_kv_src_[il] && (!l.idx_wk || !l.idx_k_norm))) {
                err = "layer " + std::to_string(il) + ": " + last_error;
                return false;
            }
        }
        if (engram_of_[il] >= 0 && opt_.engram) {
            l.eng_q = need("engram_q.weight", kF32);
            l.eng_k = need("engram_k.weight", kF32);
            l.eng_wkv = need("engram_wkv.weight", kFp8);
            if (!l.eng_q || !l.eng_k || !l.eng_wkv) {
                err = "layer " + std::to_string(il) + ": " + last_error;
                return false;
            }
        }
    }
    // The embedding runs at the start of stage 0; the head closes the last stage.
    bind_stage(0);
    tok_embd_ = load_tensor("token_embd.weight", kQ4K, "token_embd");
    bind_stage(nstages_ - 1);
    out_w_ = load_tensor("output.weight", kQ4K, "output");
    out_norm_ = load_tensor("output_norm.weight", kF32, "output_norm");
    if (!tok_embd_ || !out_w_ || !out_norm_) { err = last_error; return false; }

    // The expert containers: header, slot table and one payload header per layer (the payload bytes stream later).
    for (uint32_t il = 0; il < nlayers_; ++il)
        if (!load_blob(il, err)) return false;

    // DSpark (opt-in): the three towers, then the draft state.  Both read the geometry open() validated.
    if (opt_.dspark) {
        if (!load_towers(err)) return false;
        if (!draft_alloc(err)) return false;
    }

    if (nstages_ == 1) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::fprintf(stderr, "[v41] weights resident; device %.1f GiB used, %.1f GiB free\n",
                     static_cast<double>(total_b - free_b) / 1073741824.0,
                     static_cast<double>(free_b) / 1073741824.0);
    } else {
        for (uint32_t s = 0; s < nstages_; ++s) {
            bind_stage(s);
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            std::fprintf(stderr,
                         "[v41] stage %u (gpu %d) layers %u..%u: weights %.1f GiB used, %.1f GiB free\n", s,
                         st_[s].dev, stage_first_[s], stage_first_[s + 1] - 1,
                         static_cast<double>(total_b - free_b) / 1073741824.0,
                         static_cast<double>(free_b) / 1073741824.0);
        }
    }
    return true;
}

bool Runner::load_blob(uint32_t il, std::string& err) {
    return load_blob_named("blk." + std::to_string(il) + ".ffn_exps_vq.blob", false, err, blob_[il]);
}

// One expert container: the main layer's or a tower's (the same format, a different expert count).  Container
// versions 2 (each payload carries its own codebook, payload magic DQVQ) and 3 (one shared codebook behind the
// slot table, payload magic DQV3) are both accepted, mirroring the parser's own contract.
bool Runner::load_blob_named(const std::string& name, bool tower, std::string& err, Blob& b) {
    const TensorInfo* t = gguf_->find(name);
    if (t == nullptr) { err = "missing expert blob '" + name + "'"; return false; }
    b = Blob{};
    b.abs = gguf_->data_start() + t->offset;
    b.size = d4::payload_bytes(*t);
    if (b.size == 0) { err = "'" + name + "': the expert blob has no payload size"; return false; }
    uint8_t hdr[16];
    if (!parts_.read(b.abs, hdr, sizeof hdr, err)) return false;
    b.ver = rd32(hdr + 4);
    if (rd32(hdr) != d4::kBlobMagic || b.ver < 2 || b.ver > 3) {
        err = "'" + name + "': expert blob header unreadable in " + parts_.where(b.abs) +
              " (a hole in the download reads as zeros)";
        return false;
    }
    b.experts = rd32(hdr + 12);
    const uint32_t expect_experts = tower ? mtp_experts_ : NE_;
    if (b.experts != expect_experts) {
        err = "'" + name + "': blob carries " + std::to_string(b.experts) + " experts, expected " +
              std::to_string(expect_experts);
        return false;
    }

    b.slot.resize(static_cast<size_t>(b.experts) * 3);
    std::vector<uint8_t> table(static_cast<size_t>(b.experts) * 24);
    if (!parts_.read(b.abs + 16, table.data(), table.size(), err)) return false;
    for (size_t i = 0; i < b.slot.size(); ++i) b.slot[i] = rd64(table.data() + i * 8);

    // Payload metadata from the first present slot (a v3 container shares one codebook).
    uint64_t first = 0;
    for (size_t i = 0; i < b.slot.size(); ++i)
        if (b.slot[i] != 0) { first = b.slot[i]; break; }
    if (first == 0) { err = "'" + name + "': no expert payloads"; return false; }
    uint8_t ph[32];
    if (!parts_.read(b.abs + first, ph, sizeof ph, err)) return false;
    const uint32_t magic = rd32(ph);
    b.v3 = magic == d4::kMatV3Magic;
    if (!b.v3 && magic != d4::kMatV2Magic) {
        err = "'" + name + "': first expert payload header unreadable in " + parts_.where(b.abs + first);
        return false;
    }
    if (b.v3 != (b.ver == 3)) { err = "'" + name + "': container version and payload magic disagree"; return false; }
    b.nc = static_cast<uint32_t>(ph[6]) | (static_cast<uint32_t>(ph[7]) << 8);
    if (b.v3) {
        b.cb_off = rd64(ph + 24);
        b.cb_len = static_cast<uint64_t>(b.nc) * 8;
        const uint64_t expect_cb = 16 + static_cast<uint64_t>(b.experts) * 24;
        if (b.cb_off != expect_cb) { err = "'" + name + "': codebook offset is not the container's"; return false; }
        if (b.cb_off + b.cb_len > b.size) { err = "'" + name + "': codebook outside the blob"; return false; }
    }

    // The packed slot spans: the next greater slot offset, or the container end for the last.
    std::vector<uint64_t> offs;
    offs.reserve(b.slot.size());
    for (uint64_t s : b.slot)
        if (s != 0) offs.push_back(s);
    std::sort(offs.begin(), offs.end());
    b.span.assign(b.slot.size(), 0);
    for (size_t i = 0; i < b.slot.size(); ++i) {
        const uint64_t s = b.slot[i];
        if (s == 0) continue;
        const auto it = std::upper_bound(offs.begin(), offs.end(), s);
        const uint64_t end = it == offs.end() ? b.size : *it;
        if (end <= s || end > b.size) { err = "'" + name + "': slot table is not packed"; return false; }
        b.span[i] = end - s;
    }
    return true;
}

bool Runner::stage_blob(uint32_t il, const int32_t* sel, uint32_t n_pairs, std::string& err) {
    return stage_blob_b(blob_[il], il, sel, n_pairs, err);
}

bool Runner::stage_blob_b(const Blob& b, uint32_t layer_id, const int32_t* sel, uint32_t n_pairs, std::string& err) {
    std::vector<bool> used(b.experts, false);
    for (uint32_t i = 0; i < n_pairs; ++i) {
        const int32_t e = sel[i];
        if (e < 0 || static_cast<uint32_t>(e) >= b.experts) {
            err = "router selected expert " + std::to_string(e) + " outside 0.." + std::to_string(b.experts);
            return false;
        }
        used[static_cast<size_t>(e)] = true;
    }
    uniq_.clear();
    for (uint32_t e = 0; e < b.experts; ++e)
        if (used[e]) uniq_.push_back(e);
    if (opt_.ledger && b.experts == NE_ && layer_id < nlayers_) {
        // Repeat accounting for a possible expert cache: how many of this container's slots were staged by an
        // earlier block.  New slots are the bytes a cache would still have to read.
        for (uint32_t e : uniq_) {
            const size_t k = static_cast<size_t>(layer_id) * NE_ + e;
            if (k < seen_exp_.size() && !seen_exp_[k]) {
                seen_exp_[k] = 1;
                ++st_new_[stage_cur_];
            }
            ++st_slots_[stage_cur_];
        }
    }

    const uint64_t table_end = 16 + static_cast<uint64_t>(b.experts) * 24;
    uint64_t need = table_end + b.cb_len;
    for (uint32_t e : uniq_)
        for (uint32_t w = 0; w < 3; ++w) need += b.span[static_cast<size_t>(e) * 3 + w];
    if (stage_.size() < need) stage_.resize(static_cast<size_t>(need));
    uint8_t* p = stage_.data();
    std::memset(p, 0, static_cast<size_t>(table_end));
    wr32(p, d4::kBlobMagic);
    wr32(p + 4, b.ver);
    wr32(p + 8, layer_id);
    wr32(p + 12, b.experts);
    if (b.cb_len != 0 && !parts_.read(b.abs + b.cb_off, p + table_end, b.cb_len, err)) return false;

    // ---- the expert cache (opt-in): a hit skips the parts read and is copied from its slot after the upload ----
    const bool caching = !opt_.expert_cache_mib.empty() && b.experts == NE_ && layer_id < nlayers_ &&
                         stage_cur_ < exp_cache_dev_.size() && exp_cache_dev_[stage_cur_] != nullptr &&
                         exp_cache_slots_[stage_cur_] != 0;
    uint8_t* const cbase = caching ? exp_cache_dev_[stage_cur_] : nullptr;
    auto& cmap = exp_cache_[stage_cur_];
    std::vector<uint64_t> pinned;   // slots this container points at: never evicted by its own misses
    struct Move { uint64_t dst = 0, src = 0, len = 0; bool to_cache = false; };
    std::vector<Move> moves;
    // --ram-image: a miss's slots are queued here and copied from the RAM image straight to the container
    // (dst is the container offset, src the absolute image offset), so the bytes are never assembled into the
    // pageable staging vector.  Empty when the tier is off, and then the path below is the old one.
    const uint8_t* const ram = parts_.ram_base();
    std::vector<Move> pulls;
    // Least recently used slot the container does not point at; -1 when every slot is pinned.
    auto evict_one = [&]() -> int64_t {
        uint64_t victim = 0, best = ~uint64_t{0};
        int64_t slot = -1;
        for (const auto& kv : cmap) {
            if (kv.second.stamp >= best) continue;
            bool keep = false;
            for (uint64_t k : pinned)
                if (k == kv.first) { keep = true; break; }
            if (keep) continue;
            best = kv.second.stamp;
            victim = kv.first;
            slot = static_cast<int64_t>(kv.second.slot);
        }
        if (slot >= 0) cmap.erase(victim);
        return slot;
    };

    uint64_t cur = table_end + b.cb_len;
    uint32_t holes = 0;
    for (uint32_t e : uniq_) {
        const size_t i0 = static_cast<size_t>(e) * 3;
        const uint64_t len = b.span[i0] + b.span[i0 + 1] + b.span[i0 + 2];
        uint64_t at = cur;
        for (uint32_t w = 0; w < 3; ++w) {
            const uint64_t sl = b.span[i0 + w];
            if (sl == 0) {
                err = "expert " + std::to_string(e) + " slot " + std::to_string(w) + " is absent";
                return false;
            }
            wr64(p + 16 + (i0 + w) * 8, at);   // slot table entry i is at 16 + i*8
            at += sl;
        }
        const uint64_t key = static_cast<uint64_t>(layer_id) * NE_ + e;
        auto it = caching ? cmap.find(key) : cmap.end();
        const bool hit = it != cmap.end() && it->second.len == len;
        if (hit) {
            pinned.push_back(key);
            moves.push_back({cur, it->second.slot * exp_cache_stride_, len, false});
            it->second.stamp = ++cache_tick_;
            if (opt_.ledger) {
                ++st_cache_hit_[stage_cur_];
                st_cache_bytes_[stage_cur_] += len;
            }
        } else {
            uint64_t rd = cur;
            bool payload_ok = true;
            if (ram != nullptr) {
                uint64_t dst = cur;
                for (uint32_t w = 0; w < 3; ++w) {
                    const size_t i = i0 + w;
                    const uint64_t src = b.abs + b.slot[i];
                    if (src > parts_.size() || b.span[i] > parts_.size() - src) {
                        err = "read past the image end (offset " + std::to_string(src) + ", " +
                              std::to_string(b.span[i]) + " bytes)";
                        return false;
                    }
                    // The three slots are usually adjacent in both the image and the container; one copy then
                    // carries all three and a prefill layer issues a third of the calls.
                    if (!pulls.empty() && pulls.back().src + pulls.back().len == src &&
                        pulls.back().dst + pulls.back().len == dst)
                        pulls.back().len += b.span[i];
                    else
                        pulls.push_back({dst, src, b.span[i], false});
                    const uint32_t magic = rd32(ram + src);
                    if (magic != d4::kMatV3Magic && magic != d4::kMatV2Magic) { ++holes; payload_ok = false; }
                    dst += b.span[i];
                }
            } else {
                for (uint32_t w = 0; w < 3; ++w) {
                    const size_t i = i0 + w;
                    if (!parts_.read(b.abs + b.slot[i], p + rd, b.span[i], err)) return false;
                    const uint32_t magic = rd32(p + rd);
                    if (magic != d4::kMatV3Magic && magic != d4::kMatV2Magic) { ++holes; payload_ok = false; }
                    rd += b.span[i];
                }
            }
            if (opt_.ledger) ++st_cache_miss_[stage_cur_];
            if (caching && payload_ok) {
                int64_t slot = -1;
                if (exp_cache_next_[stage_cur_] < exp_cache_slots_[stage_cur_])
                    slot = static_cast<int64_t>(exp_cache_next_[stage_cur_]++);
                else
                    slot = evict_one();
                if (slot >= 0) {
                    cmap[key] = CacheEnt{len, ++cache_tick_, static_cast<uint32_t>(slot)};
                    moves.push_back({static_cast<uint64_t>(slot) * exp_cache_stride_, cur, len, true});
                }
            }
        }
        cur += len;
    }
    if (holes != 0)
        std::fprintf(stderr, "[v41] L%02u: %u of %u selected expert payloads are unreadable (a holey part); "
                             "the kernel skips them\n", layer_id, holes, static_cast<uint32_t>(uniq_.size()) * 3u);
    if (cur > blob_dcap_) {
        if (blob_dev_ != nullptr) cudaFree(blob_dev_);
        blob_dev_ = dev(cur, "expert container");
        if (blob_dev_ == nullptr) { err = last_error; return false; }
        blob_dcap_ = cur;
    }
    // The header (and the shared codebook) always rides the staging vector; with the RAM image the experts'
    // bytes are pulled straight from it below, so only [0, table_end + cb_len) is uploaded here.
    const uint64_t upload = ram != nullptr ? table_end + b.cb_len : cur;
    const cudaError_t ce = cudaMemcpyAsync(blob_dev_, p, upload, cudaMemcpyHostToDevice, stream_);
    if (ce != cudaSuccess) { err = std::string("expert container upload: ") + cudaGetErrorString(ce); return false; }
    for (const Move& m : pulls) {
        const cudaError_t pe = cudaMemcpyAsync(static_cast<uint8_t*>(blob_dev_) + m.dst, ram + m.src, m.len,
                                               cudaMemcpyHostToDevice, stream_);
        if (pe != cudaSuccess) { err = std::string("expert container pull: ") + cudaGetErrorString(pe); return false; }
    }
    // The moves ride the container's stream after the upload, so a hit's bytes land before the MoE reads them and
    // a fresh slot copies the payload that was just uploaded.
    for (const Move& m : moves) {
        uint8_t* dst = m.to_cache ? cbase + m.dst : static_cast<uint8_t*>(blob_dev_) + m.dst;
        const uint8_t* src = m.to_cache ? static_cast<const uint8_t*>(blob_dev_) + m.src : cbase + m.src;
        const cudaError_t me = cudaMemcpyAsync(dst, src, m.len, cudaMemcpyDeviceToDevice, stream_);
        if (me != cudaSuccess) { err = std::string("expert cache move: ") + cudaGetErrorString(me); return false; }
    }
    stage_used_ = cur;
    if (opt_.ledger) {
        st_bytes_[stage_cur_] += cur;
        st_experts_[stage_cur_] += uniq_.size();
    }
    return true;
}

bool Runner::load_engram(std::string& err) {
    if (!opt_.engram || geo.engram_layer_ids.empty()) return true;
    const size_t layers = geo.engram_layer_ids.size();
    if (layers > 2) { err = "more than two engram tables are not laid out here"; return false; }
    // The hash tables are read through the parts reader: the header's tensor map only covers part 1.
    hash_.max_ngram = static_cast<uint32_t>(geo.engram_max_ngram_size);
    hash_.heads = static_cast<uint32_t>(geo.engram_head_count);
    hash_.pad = static_cast<int32_t>(geo.engram_pad_id_compressed);
    hash_.n_vocab = static_cast<uint32_t>(geo.vocab_size);
    hash_.token_map.resize(static_cast<size_t>(geo.vocab_size));
    hash_.multipliers.resize(static_cast<size_t>(layers * hash_.max_ngram));
    hash_.primes.resize(static_cast<size_t>(layers * (hash_.max_ngram - 1) * hash_.heads));
    hash_.offsets.resize(hash_.primes.size());

    auto fetch = [&](const char* name, uint32_t type, uint64_t elems, uint32_t elem_bytes, void* dst) -> bool {
        const TensorInfo* t = gguf_->find(name);
        if (t == nullptr || t->type != type || d4::payload_bytes(*t) != elems * elem_bytes) {
            err = std::string("engram hash tensor '") + name + "' is missing or the wrong size";
            return false;
        }
        return parts_.read(gguf_->data_start() + t->offset, dst, elems * elem_bytes, err);
    };
    if (!fetch("engram.token_map", kI32, geo.vocab_size, 4, hash_.token_map.data()) ||
        !fetch("engram.multipliers", kI64, layers * hash_.max_ngram, 8, hash_.multipliers.data()) ||
        !fetch("engram.primes", kI64, layers * (hash_.max_ngram - 1) * hash_.heads, 8, hash_.primes.data()) ||
        !fetch("engram.offsets", kI64, layers * (hash_.max_ngram - 1) * hash_.heads, 8, hash_.offsets.data()))
        return false;
    std::string bad;
    if (!hash_.valid(static_cast<uint32_t>(layers), bad)) { err = bad; return false; }

    eng_cols_ = hash_.cols();
    eng_hd_ = static_cast<uint32_t>(geo.engram_head_dim);
    if (eng_hd_ % 32 != 0) { err = "engram head dim is not a multiple of 32"; return false; }
    eng_in_dim_ = eng_cols_ * eng_hd_;
    eng_out_dim_ = (HC_ + 1) * E_;
    for (size_t i = 0; i < layers; ++i) {
        shard_[i] = std::make_unique<d4::EngramShard>();
        std::string e;
        if (!d4::open_engram_table(geo, i, opt_.engram_dir, *shard_[i], e)) {
            err = "engram table " + std::to_string(i) + ": " + e;
            return false;
        }
    }
    // The four cap-sized row buffers are allocated by alloc_state, once the auto chunk has picked the block: the
    // engram layers (1 and 14 here) sit in different stages under a split, and only a stage that owns one needs
    // them.  The hash tables and the shard readers stay on the host, shared.
    engram_on_ = true;
    std::fprintf(stderr, "[v41] engram: %zu table(s), cols %u, head dim %u, in %u out %u\n", layers, eng_cols_,
                 eng_hd_, eng_in_dim_, eng_out_dim_);
    return true;
}

// ---- DSpark (P5): the draft towers, the verify batch and its rollback -----------------------------------------

bool Runner::load_towers(std::string& err) {
    auto any = [&](const std::string& name, std::initializer_list<uint32_t> types) -> void* {
        const TensorInfo* t = gguf_->find(name);
        if (t == nullptr) { last_error = "missing tensor '" + name + "'"; return nullptr; }
        bool ok = false;
        for (uint32_t ty : types) ok = ok || t->type == ty;
        if (!ok) {
            last_error = "tensor '" + name + "' is type " + std::to_string(t->type) + " (an accepted type is f32/bf16)";
            return nullptr;
        }
        return load_tensor(name, t->type, name.c_str());
    };
    mtp_main_proj_ = any("mtp.main_proj.weight", {kFp8});
    mtp_main_norm_ = any("mtp.main_norm.weight", {kF32});
    mtp_markov_embd_ = any("mtp.markov_embd.weight", {kBF16});
    mtp_markov_head_ = any("mtp.markov_head.weight", {kBF16});
    mtp_conf_ = any("mtp.confidence.weight", {kBF16, kF32});
    if (mtp_conf_ != nullptr) {
        const TensorInfo* ct = gguf_->find("mtp.confidence.weight");
        mtp_conf_f32_ = ct != nullptr && ct->type == kF32;
    }
    mtp_out_norm_ = any("mtp.out_norm.weight", {kF32});
    if (!mtp_main_proj_ || !mtp_main_norm_ || !mtp_markov_embd_ || !mtp_markov_head_ || !mtp_conf_ ||
        !mtp_out_norm_) {
        err = "DSpark: " + last_error;
        return false;
    }
    for (uint32_t T = 0; T < mtp_towers_; ++T) {
        const std::string p = "mtp." + std::to_string(T) + ".";
        LayerW& l = TW_[T];
        auto need = [&](const char* suffix, uint32_t type) -> void* {
            void* w = load_tensor(p + suffix, type, suffix);
            if (w == nullptr) std::fprintf(stderr, "  tower %u: %s\n", T, last_error.c_str());
            return w;
        };
        l.hc_attn_fn = need("hc_attn_fn.weight", kF32);
        l.hc_attn_scale = need("hc_attn_scale.weight", kF32);
        l.hc_attn_base = need("hc_attn_base.weight", kF32);
        l.attn_norm = need("attn_norm.weight", kF32);
        l.q_a = need("attn_q_a.weight", kFp8);
        l.q_a_norm = need("attn_q_a_norm.weight", kF32);
        l.q_b = need("attn_q_b.weight", kFp8);
        l.kv = need("attn_kv.weight", kFp8);
        l.kv_a_norm = need("attn_kv_a_norm.weight", kF32);
        l.sinks = need("attn_sinks.weight", kF32);
        l.out_a = need("attn_output_a.weight", kFp8);
        l.out_b = need("attn_output_b.weight", kFp8);
        l.hc_ffn_fn = need("hc_ffn_fn.weight", kF32);
        l.hc_ffn_scale = need("hc_ffn_scale.weight", kF32);
        l.hc_ffn_base = need("hc_ffn_base.weight", kF32);
        l.ffn_norm = need("ffn_norm.weight", kF32);
        l.exp_bias = need("exp_probs_b.bias", kF32);
        l.shexp_gate = need("ffn_gate_shexp.weight", kFp8);
        l.shexp_up = need("ffn_up_shexp.weight", kFp8);
        l.shexp_down = need("ffn_down_shexp.weight", kFp8);
        // The router gate is bf16 in this artifact; the reference also accepts an f32 registration.
        const TensorInfo* gt = gguf_->find(p + "ffn_gate_inp.weight");
        if (gt == nullptr || (gt->type != kBF16 && gt->type != kF32)) {
            err = "tower " + std::to_string(T) + ": ffn_gate_inp.weight is missing or neither bf16 nor f32";
            return false;
        }
        tw_gate_f32_[T] = gt->type == kF32;
        l.gate_inp = need("ffn_gate_inp.weight", gt->type);
        if (!l.hc_attn_fn || !l.hc_attn_scale || !l.hc_attn_base || !l.attn_norm || !l.q_a || !l.q_a_norm ||
            !l.q_b || !l.kv || !l.kv_a_norm || !l.sinks || !l.out_a || !l.out_b || !l.hc_ffn_fn ||
            !l.hc_ffn_scale || !l.hc_ffn_base || !l.ffn_norm || !l.gate_inp || !l.exp_bias || !l.shexp_gate ||
            !l.shexp_up || !l.shexp_down) {
            err = "tower " + std::to_string(T) + ": " + last_error;
            return false;
        }
        if (!load_blob_named(p + "ffn_exps_vq.blob", true, err, tblob_[T])) return false;
    }
    std::fprintf(stderr,
                 "[v41] DSpark: %u tower(s) x %u experts top-%u, block %u, markov rank %u, noise %u, targets",
                 mtp_towers_, mtp_experts_, mtp_used_, mtp_block_, mtp_rank_, mtp_noise_);
    for (uint32_t T = 0; T < mtp_towers_; ++T) {
        for (uint32_t il = 0; il < nlayers_; ++il)
            if (mtp_slot_[il] == static_cast<int>(T)) std::fprintf(stderr, " L%02u", il);
    }
    std::fprintf(stderr, "\n");
    return true;
}

// The tower state and the draft round's buffers.  cap2_ = block + 1 is the verify batch's widest row count (the
// reference's `cap`); prod_cap_ = max(cap_, SWA_) is what the window top-up needs (up to SWA rows at once).
bool Runner::draft_alloc(std::string& err) {
    const uint64_t cap2 = cap2_, prod = prod_cap_, R = mtp_rank_;
    auto dv = [&](uint64_t bytes, const char* what) -> void* {
        void* p = dev(bytes, what);
        if (p == nullptr) err = last_error;
        return p;
    };
    // The hidden states and the hc block.
    t_tok_ = static_cast<int32_t*>(dv(cap2 * 4, "draft tokens"));
    t_pos_ = static_cast<int32_t*>(dv(prod * 4, "draft positions"));
    t_premix_ = static_cast<float*>(dv(cap2 * HC_ * 4, "draft pre_mix"));
    t_pre_ = static_cast<float*>(dv(cap2 * HC_ * 4, "draft pre"));
    t_hc_ = static_cast<float*>(dv(cap2 * HC_ * E_ * 4, "draft hc"));
    t_hc2_ = static_cast<float*>(dv(cap2 * HC_ * E_ * 4, "draft hc2"));
    t_mix_ = static_cast<float*>(dv(cap2 * (2u * HC_ + HC_ * HC_) * 4, "draft mix"));
    t_post_ = static_cast<float*>(dv(cap2 * HC_ * 4, "draft post"));
    t_comb_ = static_cast<float*>(dv(cap2 * HC_ * HC_ * 4, "draft comb"));
    t_x_ = static_cast<float*>(dv(cap2 * E_ * 4, "draft x"));
    t_xn_ = static_cast<float*>(dv(cap2 * E_ * 4, "draft xn"));
    // The tower's attention path.  The kv pair holds the block's rows in tower_attn and the fill's window rows
    // (up to SWA at a time), so it carries the wider of the two.
    t_qr_ = static_cast<float*>(dv(cap2 * Q_ * 4, "draft qr"));
    t_qrn_ = static_cast<float*>(dv(cap2 * Q_ * 4, "draft qrn"));
    t_q_ = static_cast<float*>(dv(cap2 * H_ * HD_ * 4, "draft q"));
    t_kv_ = static_cast<float*>(dv(prod * HD_ * 4, "draft kv"));
    t_kvn_ = static_cast<float*>(dv(prod * HD_ * 4, "draft kvn"));
    t_o_ = static_cast<float*>(dv(cap2 * H_ * HD_ * 4, "draft o"));
    t_low_ = static_cast<float*>(dv(cap2 * NG_ * OL_ * 4, "draft low"));
    t_attn_ = static_cast<float*>(dv(cap2 * E_ * 4, "draft attn_out"));
    // The tower's MoE.
    t_glog_ = static_cast<float*>(dv(cap2 * mtp_experts_ * 4, "tower router logits"));
    t_sel_ = static_cast<int32_t*>(dv(cap2 * mtp_used_ * 4, "tower router sel"));
    t_rw_ = static_cast<float*>(dv(cap2 * mtp_used_ * 4, "tower router w"));
    t_partial_ = static_cast<float*>(dv(cap2 * mtp_used_ * E_ * 4, "tower moe partial"));
    t_routed_ = static_cast<float*>(dv(cap2 * E_ * 4, "tower routed"));
    t_sg_ = static_cast<float*>(dv(cap2 * FF_ * 4, "tower shared gate"));
    t_su_ = static_cast<float*>(dv(cap2 * FF_ * 4, "tower shared up"));
    t_sh_ = static_cast<float*>(dv(cap2 * FF_ * 4, "tower shared h"));
    t_so_ = static_cast<float*>(dv(cap2 * E_ * 4, "tower shared out"));
    t_y_ = static_cast<float*>(dv(cap2 * E_ * 4, "tower y"));
    // The draft block's exits.
    t_logits_ = static_cast<float*>(dv(cap2 * V_ * 4, "draft logits"));
    vlogits_ = static_cast<float*>(dv(cap2 * V_ * 4, "verify logits"));
    t_mk_embed_ = static_cast<float*>(dv(cap2 * R * 4, "markov embed"));
    t_mk_cur_ = static_cast<float*>(dv(R * 4, "markov cur"));
    t_mk_bias_ = static_cast<float*>(dv(uint64_t(V_) * 4, "markov bias"));
    t_conf_in_ = static_cast<float*>(dv(cap2 * (E_ + R) * 4, "confidence in"));
    t_conf_ = static_cast<float*>(dv(cap2 * 4, "confidence"));
    t_ids_ = static_cast<int32_t*>(dv(cap2 * 4, "draft ids"));
    t_ids_next_ = static_cast<int32_t*>(dv(16, "draft ids next"));
    t_want_ = static_cast<int32_t*>(dv(cap2 * 4, "verify picks"));
    // The bf16 activation copy every fp8 matmul with n > 1 reads.  Its widest user is the fill's main_proj
    // (SWA rows x n_target*E); the block's widest is out_a's H*HD input.
    uint64_t t16 = uint64_t(SWA_) * mtp_n_target_ * E_;
    const uint64_t block16 = uint64_t(cap2) * H_ * HD_;
    if (block16 > t16) t16 = block16;
    const uint64_t e16 = uint64_t(cap2) * E_;
    if (e16 > t16) t16 = e16;
    t_x16_ = static_cast<uint16_t*>(dv(t16 * 2, "draft x bf16"));
    // The main-hidden ring and the fill's staging.
    mainh_cap_ = SWA_;
    mainh_ = static_cast<float*>(dv(uint64_t(mainh_cap_) * mtp_n_target_ * E_ * 4, "main hidden ring"));
    t_mainh_lin_ = static_cast<float*>(dv(uint64_t(SWA_) * mtp_n_target_ * E_ * 4, "main hidden rows"));
    t_mainx_raw_ = static_cast<float*>(dv(prod * E_ * 4, "main x raw"));
    t_main_x_ = static_cast<float*>(dv(prod * E_ * 4, "main x"));
    t_wintmp_ = static_cast<float*>(dv(uint64_t(SWA_) * HD_ * 4, "tower window shift"));
    twin_.assign(mtp_towers_, nullptr);
    for (uint32_t T = 0; T < mtp_towers_; ++T) {
        twin_[T] = static_cast<float*>(dv(uint64_t(SWA_ + prod) * HD_ * 4, "tower window"));
        // The reference zeroes its windows: below the window the linear segment's leftmost cells are never
        // written, and the attention must not read cudaMalloc's leftovers.
        if (twin_[T] != nullptr) cudaMemset(twin_[T], 0, static_cast<size_t>(uint64_t(SWA_ + prod) * HD_ * 4));
    }
    // The verify batch's snapshot.
    snap_win_ = static_cast<float*>(dv(uint64_t(nlayers_) * cap2 * HD_ * 4, "spec window snapshot"));
    for (uint32_t il = 0; il < nlayers_; ++il) {
        if (!is_kv_src_[il]) continue;
        const uint32_t ratio = static_cast<uint32_t>(geo.compress_ratios[il]);
        if (ratio <= 1) continue;
        snap_cpre_kv_[il] = static_cast<float*>(dv(uint64_t(ratio + cap2) * HD_ * 4, "spec pending kv"));
        snap_cpre_sc_[il] = static_cast<float*>(dv(uint64_t(ratio + cap2) * HD_ * 4, "spec pending score"));
        if (snap_cpre_kv_[il] == nullptr || snap_cpre_sc_[il] == nullptr) return false;
    }
    if (err.empty() == false) return false;
    // Host slots.
    t_tok_host_.assign(cap2, 0);
    t_bpos_host_.assign(cap2, 0);
    t_wpos_host_.assign(SWA_, 0);
    t_want_host_.assign(cap2, 0);
    t_id_host_.assign(cap2, 0);
    t_conf_host_.assign(cap2, 0.f);
    if (sel_host_.size() < cap2 * mtp_used_) sel_host_.resize(static_cast<size_t>(cap2) * mtp_used_);
    std::fprintf(stderr, "[v41] DSpark state: batch cap %u, window cap %u, ring %u x %u x %u\n", cap2_,
                 static_cast<unsigned>(prod), mainh_cap_, mtp_n_target_, E_);
    return true;
}

bool Runner::tower_half(uint32_t T, bool attn_half, std::string& err) {
    const LayerW& l = TW_[T];
    const float* fn = static_cast<const float*>(attn_half ? l.hc_attn_fn : l.hc_ffn_fn);
    const float* sc = static_cast<const float*>(attn_half ? l.hc_attn_scale : l.hc_ffn_scale);
    const float* bs = static_cast<const float*>(attn_half ? l.hc_attn_base : l.hc_ffn_base);
    const float* nm = static_cast<const float*>(attn_half ? l.attn_norm : l.ffn_norm);
    if (!k::deepseek4_hc_mix(fn, t_hc_, t_mix_, E_, HC_, cur_n_, rms_eps_, stream_)) {
        err = "tower hc_mix failed";
        return false;
    }
    if (!k::deepseek4_hc_fused(t_mix_, sc, bs, t_premix_, t_hc_, nm, t_pre_, t_post_, t_comb_, t_x_, t_xn_, E_, HC_,
                               sinkhorn_, hc_eps_, rms_eps_, cur_n_, stream_)) {
        err = "tower hc_fused failed";
        return false;
    }
    std::swap(t_pre_, t_premix_);
    return true;
}

bool Runner::tower_attn(uint32_t T, std::string& err) {
    const LayerW& l = TW_[T];
    const uint32_t n = cur_n_;
    const uint16_t* x16 = n > 1 ? t_x16_ : nullptr;
    const float theta = theta_of(0);
    const uint32_t osl = osl_of(0);
    const uint32_t grp = H_ / NG_;
    // The reference's v41_draft_attention: q_a -> norm -> q_b -> rope, kv -> norm -> rope -> fp8 grid -> the
    // linear window's block region, the full-block attention, then the inverse rope and the grouped output.
    if ((n > 1 && !k::deepseek4_engram_x_to_bf16(t_xn_, t_x16_, uint64_t(n) * E_, stream_)) ||
        !k::deepseek4_matmul_fp8(t_qr_, static_cast<const uint8_t*>(l.q_a), t_xn_, x16, E_, Q_, n, true, stream_) ||
        !k::deepseek4_rms_norm(t_qrn_, t_qr_, static_cast<const float*>(l.q_a_norm), Q_, n, rms_eps_, stream_)) {
        err = "tower q_a path failed";
        return false;
    }
    if ((n > 1 && !k::deepseek4_engram_x_to_bf16(t_qrn_, t_x16_, uint64_t(n) * Q_, stream_)) ||
        !k::deepseek4_matmul_fp8(t_q_, static_cast<const uint8_t*>(l.q_b), t_qrn_, x16, Q_,
                                 uint64_t(H_) * HD_, n, true, stream_) ||
        !k::deepseek4_rope(t_q_, t_pos_, n, H_, HD_, nrot_, theta, osl, rope_factor_, beta_fast_, beta_slow_,
                           false, stream_)) {
        err = "tower q_b/rope failed";
        return false;
    }
    if ((n > 1 && !k::deepseek4_engram_x_to_bf16(t_xn_, t_x16_, uint64_t(n) * E_, stream_)) ||
        !k::deepseek4_matmul_fp8(t_kv_, static_cast<const uint8_t*>(l.kv), t_xn_, x16, E_, HD_, n, true,
                                 stream_) ||
        !k::deepseek4_rms_norm(t_kvn_, t_kv_, static_cast<const float*>(l.kv_a_norm), HD_, n, rms_eps_,
                               stream_) ||
        !k::deepseek4_rope(t_kvn_, t_pos_, n, 1, HD_, nrot_, theta, osl, rope_factor_, beta_fast_, beta_slow_,
                           false, stream_) ||
        !k::deepseek4_act_quant(t_kvn_, n, HD_, 32, k::kDeepseek4QuantFp8E8m0, stream_)) {
        err = "tower kv path failed";
        return false;
    }
    cudaError_t ce = cudaMemcpyAsync(twin_[T] + uint64_t(SWA_) * HD_, t_kvn_, uint64_t(n) * HD_ * 4,
                                     cudaMemcpyDeviceToDevice, stream_);
    if (ce != cudaSuccess) { err = std::string("tower window append: ") + cudaGetErrorString(ce); return false; }
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(HD_)));
    // full_block = n: every position of the block sees the whole block (the API takes the length where the
    // reference flag means n_tok); ring = false: this window is the linear segment push_main maintains.
    if (!k::deepseek4_sparse_attn(t_o_, t_q_, twin_[T], nullptr, nullptr, static_cast<const float*>(l.sinks), n,
                                  cur_pos0_, SWA_, 0, 0, H_, HD_, scale, n, false, 0, stream_)) {
        err = "tower sparse attention failed";
        return false;
    }
    if (!k::deepseek4_rope(t_o_, t_pos_, n, H_, HD_, nrot_, theta, osl, rope_factor_, beta_fast_, beta_slow_, true,
                           stream_)) {
        err = "tower inverse rope failed";
        return false;
    }
    if ((n > 1 && !k::deepseek4_engram_x_to_bf16(t_o_, t_x16_, uint64_t(n) * H_ * HD_, stream_)) ||
        !k::deepseek4_matmul_fp8_grouped(t_low_, static_cast<const uint8_t*>(l.out_a), t_o_, x16, NG_,
                                         uint64_t(grp) * HD_, OL_, n, true, stream_)) {
        err = "tower attn_output_a failed";
        return false;
    }
    if ((n > 1 && !k::deepseek4_engram_x_to_bf16(t_low_, t_x16_, uint64_t(n) * NG_ * OL_, stream_)) ||
        !k::deepseek4_matmul_fp8(t_attn_, static_cast<const uint8_t*>(l.out_b), t_low_, x16, uint64_t(NG_) * OL_,
                                 E_, n, true, stream_)) {
        err = "tower attn_output_b failed";
        return false;
    }
    return true;
}

bool Runner::tower_moe(uint32_t T, std::string& err) {
    const LayerW& l = TW_[T];
    const uint32_t n = cur_n_;
    const uint16_t* x16 = n > 1 ? t_x16_ : nullptr;
    // The reference's v41_moe with st->draft set: shared expert, then the 128-expert router and the routed VQ
    // experts, and never the decode tail arm (`!st->draft && n <= 8` is false for a tower).
    if ((n > 1 && !k::deepseek4_engram_x_to_bf16(t_xn_, t_x16_, uint64_t(n) * E_, stream_)) ||
        !k::deepseek4_matmul_fp8(t_sg_, static_cast<const uint8_t*>(l.shexp_gate), t_xn_, x16, E_, FF_, n, true,
                                 stream_) ||
        !k::deepseek4_matmul_fp8(t_su_, static_cast<const uint8_t*>(l.shexp_up), t_xn_, x16, E_, FF_, n, true,
                                 stream_) ||
        !k::deepseek4_swiglu(t_sh_, t_sg_, t_su_, uint64_t(n) * FF_, clamp_, stream_)) {
        err = "tower shared expert failed";
        return false;
    }
    if ((n > 1 && !k::deepseek4_engram_x_to_bf16(t_sh_, t_x16_, uint64_t(n) * FF_, stream_)) ||
        !k::deepseek4_matmul_fp8(t_so_, static_cast<const uint8_t*>(l.shexp_down), t_sh_, x16, FF_, E_, n, true,
                                 stream_)) {
        err = "tower shared down failed";
        return false;
    }
    // Router gate: the registered type decides f32 vs bf16 (the reference's v41_small_matmul rule).
    const bool gate_ok = tw_gate_f32_[T]
                             ? k::deepseek4_matmul_f32(t_glog_, static_cast<const float*>(l.gate_inp), t_xn_, E_,
                                                       mtp_experts_, n, false, stream_)
                             : k::deepseek4_matmul_bf16(t_glog_, static_cast<const uint16_t*>(l.gate_inp), t_xn_,
                                                        E_, mtp_experts_, n, false, stream_);
    if (!gate_ok) { err = "tower router gate failed"; return false; }
    if (!k::deepseek4_router(t_sel_, t_rw_, t_glog_, static_cast<const float*>(l.exp_bias), nullptr, nullptr, n,
                             mtp_experts_, mtp_used_, route_scale_, stream_)) {
        err = "tower deepseek4_router failed";
        return false;
    }
    cudaError_t ce = cudaMemcpyAsync(sel_host_.data(), t_sel_, uint64_t(n) * mtp_used_ * 4, cudaMemcpyDeviceToHost,
                                     stream_);
    if (ce != cudaSuccess) { err = std::string("tower selection readback: ") + cudaGetErrorString(ce); return false; }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) { err = "tower selection readback sync failed"; return false; }
    if (!stage_blob_b(tblob_[T], T, sel_host_.data(), n * mtp_used_, err)) return false;
    if (!k::deepseek4_routed_moe(t_y_, t_partial_, blob_dev_, stage_used_, E_, FF_, E_, t_sel_, t_rw_, mtp_used_,
                                 clamp_, t_xn_, n, tblob_[T].nc, tblob_[T].v3 ? 1u : 0u, nullptr, stream_)) {
        err = "tower deepseek4_routed_moe failed";
        return false;
    }
    if (!k::deepseek4_add(t_y_, t_so_, uint64_t(n) * E_, stream_) ||
        !k::deepseek4_round_bf16(t_y_, uint64_t(n) * E_, stream_)) {
        err = "tower moe add/round failed";
        return false;
    }
    return true;
}

bool Runner::tower_layer(uint32_t T, std::string& err) {
    if (!tower_half(T, true, err)) return false;
    if (!tower_attn(T, err)) return false;
    if (!k::deepseek4_hc_post(t_attn_, t_hc_, t_post_, t_comb_, t_hc2_, E_, HC_, cur_n_, stream_)) {
        err = "tower hc_post (attn) failed";
        return false;
    }
    std::swap(t_hc_, t_hc2_);
    if (!tower_half(T, false, err)) return false;
    if (!tower_moe(T, err)) return false;
    if (!k::deepseek4_hc_post(t_y_, t_hc_, t_post_, t_comb_, t_hc2_, E_, HC_, cur_n_, stream_)) {
        err = "tower hc_post (ffn) failed";
        return false;
    }
    std::swap(t_hc_, t_hc2_);
    return true;
}

// One draft block forward (the reference's v41_draft_block): the hills of the three towers, the borrowed head,
// the per-position markov greedy chain and the confidence head.  ids[0] is the real token; ids[1..B] come out of
// the chain.  All host reads happen in one sync at the end.
bool Runner::draft_block(uint32_t pos0, std::string& err) {
    const uint32_t B = mtp_block_;
    cur_n_ = B;
    cur_pos0_ = pos0;
    for (uint32_t i = 0; i < B; ++i) t_bpos_host_[i] = static_cast<int32_t>(pos0 + i);
    for (uint32_t i = 0; i < B; ++i)
        for (uint32_t c = 0; c < HC_; ++c)
            pm_host_[static_cast<size_t>(i) * HC_ + c] = (c == 0) ? 1.0f : 0.0f;
    cudaError_t ce = cudaMemcpyAsync(t_tok_, t_tok_host_.data(), uint64_t(B) * 4, cudaMemcpyHostToDevice, stream_);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(t_pos_, t_bpos_host_.data(), uint64_t(B) * 4, cudaMemcpyHostToDevice, stream_);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(t_premix_, pm_host_.data(), uint64_t(B) * HC_ * 4, cudaMemcpyHostToDevice, stream_);
    // Only ids[0] is a host input: the chain writes ids[i+1] from row i's argmax on the device.
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(t_ids_, t_tok_host_.data(), 4, cudaMemcpyHostToDevice, stream_);
    if (ce != cudaSuccess) { err = std::string("draft block inputs: ") + cudaGetErrorString(ce); return false; }
    if (!k::deepseek4_embed_q4k(t_x_, t_tok_, tok_embd_, V_, B, E_, stream_) ||
        !k::deepseek4_expand_hc(t_hc_, t_x_, E_, HC_, B, stream_)) {
        err = "draft embed/expand failed";
        return false;
    }
    for (uint32_t T = 0; T < mtp_towers_; ++T)
        if (!tower_layer(T, err)) return false;
    // Exit: mtp.out_norm over the last half's hc_pre, then the main model's own output matrix.
    if (!k::deepseek4_hc_pre(t_hc_, t_premix_, static_cast<const float*>(mtp_out_norm_), t_x_, t_xn_, E_, HC_,
                             rms_eps_, B, stream_)) {
        err = "draft exit hc_pre failed";
        return false;
    }
    if (!k::deepseek4_matmul_q4k(t_logits_, out_w_, t_xn_, E_, V_, B, false, stream_)) {
        err = "draft head projection failed";
        return false;
    }
    const uint16_t* mk = static_cast<const uint16_t*>(mtp_markov_embd_);
    const uint16_t* mh = static_cast<const uint16_t*>(mtp_markov_head_);
    for (uint32_t i = 0; i < B; ++i) {
        // Row i's embedding feeds the confidence input; the same row goes through markov_head into row i's logits.
        if (!k::deepseek4_row_gather_bf16(t_mk_embed_, mk, V_, mtp_rank_, t_ids_, i, i, stream_) ||
            !k::deepseek4_row_gather_bf16(t_mk_cur_, mk, V_, mtp_rank_, t_ids_, i, 0, stream_) ||
            !k::deepseek4_matmul_bf16(t_mk_bias_, mh, t_mk_cur_, mtp_rank_, V_, 1, false, stream_) ||
            !k::deepseek4_row_add(t_logits_ + uint64_t(i) * V_, t_mk_bias_, V_, stream_) ||
            !k::deepseek4_argmax(t_ids_next_, t_logits_ + uint64_t(i) * V_, V_, stream_)) {
            err = "draft markov chain failed at row " + std::to_string(i);
            return false;
        }
        ce = cudaMemcpyAsync(t_ids_ + i + 1, t_ids_next_, 4, cudaMemcpyDeviceToDevice, stream_);
        if (ce != cudaSuccess) { err = std::string("draft ids copy: ") + cudaGetErrorString(ce); return false; }
    }
    // confidence = proj([h_i; markov_embed_i])
    const uint64_t er = uint64_t(E_) + mtp_rank_;
    for (uint32_t i = 0; i < B; ++i) {
        ce = cudaMemcpyAsync(t_conf_in_ + uint64_t(i) * er, t_x_ + uint64_t(i) * E_, uint64_t(E_) * 4,
                             cudaMemcpyDeviceToDevice, stream_);
        if (ce == cudaSuccess)
            ce = cudaMemcpyAsync(t_conf_in_ + uint64_t(i) * er + E_, t_mk_embed_ + uint64_t(i) * mtp_rank_,
                                 uint64_t(mtp_rank_) * 4, cudaMemcpyDeviceToDevice, stream_);
        if (ce != cudaSuccess) { err = std::string("confidence input copy: ") + cudaGetErrorString(ce); return false; }
    }
    const bool conf_ok =
        mtp_conf_f32_ ? k::deepseek4_matmul_f32(t_conf_, static_cast<const float*>(mtp_conf_), t_conf_in_,
                                                E_ + mtp_rank_, 1, B, false, stream_)
                      : k::deepseek4_matmul_bf16(t_conf_, static_cast<const uint16_t*>(mtp_conf_), t_conf_in_,
                                                 E_ + mtp_rank_, 1, B, false, stream_);
    if (!conf_ok) {
        err = "confidence projection failed";
        return false;
    }
    ce = cudaMemcpyAsync(t_id_host_.data(), t_ids_, uint64_t(B + 1) * 4, cudaMemcpyDeviceToHost, stream_);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(t_conf_host_.data(), t_conf_, uint64_t(B) * 4, cudaMemcpyDeviceToHost, stream_);
    if (ce != cudaSuccess) { err = std::string("draft readback: ") + cudaGetErrorString(ce); return false; }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) { err = "draft readback sync failed"; return false; }
    return true;
}

// One draft round (the reference's v41_draft_launch, direct arm): top the tower windows up from the main-hidden
// ring, then the block.  Returns 0 when the inputs are missing (the caller then verifies a single token).
int Runner::draft_round(int32_t tok, uint32_t pos_main, std::string& err) {
    if (!mtp_on_ || mainh_ == nullptr) return 0;
    if (mainh_end_ != static_cast<int64_t>(pos_main)) {
        std::fprintf(stderr,
                     "[dspark] the main-hidden ring ends at %lld != pos_main %u, no draft this round\n",
                     static_cast<long long>(mainh_end_), pos_main);
        return 0;
    }
    // Top up (win_end_, pos_main]: rounds that sat out are all caught here; a gap wider than the window rebuilds it.
    int64_t first = win_end_ + 1;
    if (static_cast<int64_t>(pos_main) - first + 1 > static_cast<int64_t>(SWA_))
        first = static_cast<int64_t>(pos_main) - static_cast<int64_t>(SWA_) + 1;
    if (first < 0) first = 0;
    const uint32_t rows = static_cast<uint32_t>(static_cast<int64_t>(pos_main) - first + 1);
    if (rows) {
        const int64_t avail_first = mainh_end_ - static_cast<int64_t>(mainh_n_) + 1;
        if (first < avail_first) {
            std::fprintf(stderr,
                         "[dspark] the window needs %lld..%u, the ring's contiguous run starts at %lld (%u rows), "
                         "no draft this round\n",
                         static_cast<long long>(first), pos_main, static_cast<long long>(avail_first), mainh_n_);
            return 0;
        }
    }
    const uint32_t pos0 = pos_main + 1;
    if (rows) {
        for (uint32_t i = 0; i < rows; ++i) t_wpos_host_[i] = static_cast<int32_t>(first + i);
        cudaError_t ce = cudaMemcpyAsync(t_pos_, t_wpos_host_.data(), uint64_t(rows) * 4, cudaMemcpyHostToDevice,
                                         stream_);
        if (ce != cudaSuccess) { err = std::string("fill positions: ") + cudaGetErrorString(ce); return -1; }
        if (!k::deepseek4_ring_rows(t_mainh_lin_, mainh_, mtp_n_target_ * E_, mainh_cap_,
                                    static_cast<uint32_t>(first % mainh_cap_), rows, stream_)) {
            err = "the main-hidden ring rows failed";
            return -1;
        }
        const uint16_t* x16 = rows > 1 ? t_x16_ : nullptr;
        if ((rows > 1 && !k::deepseek4_engram_x_to_bf16(t_mainh_lin_, t_x16_,
                                                        uint64_t(rows) * mtp_n_target_ * E_, stream_)) ||
            !k::deepseek4_matmul_fp8(t_mainx_raw_, static_cast<const uint8_t*>(mtp_main_proj_), t_mainh_lin_, x16,
                                     mtp_n_target_ * E_, E_, rows, true, stream_) ||
            !k::deepseek4_rms_norm(t_main_x_, t_mainx_raw_, static_cast<const float*>(mtp_main_norm_), E_, rows,
                                   rms_eps_, stream_)) {
            err = "main_proj/main_norm failed";
            return -1;
        }
        const float theta = theta_of(0);
        const uint32_t osl = osl_of(0);
        for (uint32_t T = 0; T < mtp_towers_; ++T) {
            const LayerW& l = TW_[T];
            const uint16_t* px16 = rows > 1 ? t_x16_ : nullptr;
            if ((rows > 1 && !k::deepseek4_engram_x_to_bf16(t_main_x_, t_x16_, uint64_t(rows) * E_, stream_)) ||
                !k::deepseek4_matmul_fp8(t_kv_, static_cast<const uint8_t*>(l.kv), t_main_x_, px16, E_, HD_, rows,
                                         true, stream_) ||
                !k::deepseek4_rms_norm(t_kvn_, t_kv_, static_cast<const float*>(l.kv_a_norm), HD_, rows, rms_eps_,
                                       stream_) ||
                !k::deepseek4_rope(t_kvn_, t_pos_, rows, 1, HD_, nrot_, theta, osl, rope_factor_, beta_fast_,
                                   beta_slow_, false, stream_) ||
                !k::deepseek4_act_quant(t_kvn_, rows, HD_, 32, k::kDeepseek4QuantFp8E8m0, stream_)) {
                err = "tower window push projection failed";
                return -1;
            }
            // The linear window stays right-aligned: new rows land at [SWA, SWA+rows), then the whole thing shifts
            // left by rows (the reference's v41_draft_push_main).
            ce = cudaMemcpyAsync(twin_[T] + uint64_t(SWA_) * HD_, t_kvn_, uint64_t(rows) * HD_ * 4,
                                 cudaMemcpyDeviceToDevice, stream_);
            if (ce == cudaSuccess)
                ce = cudaMemcpyAsync(t_wintmp_, twin_[T] + uint64_t(rows) * HD_, uint64_t(SWA_) * HD_ * 4,
                                     cudaMemcpyDeviceToDevice, stream_);
            if (ce == cudaSuccess)
                ce = cudaMemcpyAsync(twin_[T], t_wintmp_, uint64_t(SWA_) * HD_ * 4, cudaMemcpyDeviceToDevice,
                                     stream_);
            if (ce != cudaSuccess) { err = std::string("tower window shift: ") + cudaGetErrorString(ce); return -1; }
        }
    }
    // The block input: the real token then noise, at positions pos0..pos0+B-1.
    t_tok_host_[0] = tok;
    for (uint32_t i = 1; i <= mtp_block_; ++i) t_tok_host_[i] = static_cast<int32_t>(mtp_noise_);
    if (!draft_block(pos0, err)) return -1;
    if (rows) win_end_ = pos_main;
    return 1;
}

// The spec verify's snapshot (the reference's v41_spec_snapshot): only the ring cells the batch is about to
// overwrite, per layer.  The compressor's pending rows are snapshotted inside compress_source, before its shift.
bool Runner::spec_snapshot(uint32_t n, std::string& err) {
    if (n == 0 || n > cap2_) { err = "spec snapshot batch " + std::to_string(n) + " over the cap"; return false; }
    for (uint32_t il = 0; il < nlayers_; ++il) {
        if (!k::deepseek4_win_ring_snap(win_[il], snap_win_ + uint64_t(il) * cap2_ * HD_, n_past_, 0, n, SWA_, HD_,
                                        false, nullptr, stream_)) {
            err = "spec window snapshot failed at L" + std::to_string(il);
            return false;
        }
    }
    snap_n_ = n;
    snap_past_ = n_past_;
    snap_on_ = true;
    return true;
}

// keep = the accepted positions in this batch (the confirmed head included): restore the window cells the
// rejected rows overwrote, rewind the group counts, and put the accepted tokens' compressor pending rows back.
bool Runner::spec_rollback(uint32_t keep, std::string& err) {
    if (!snap_on_) { err = "spec rollback without a snapshot"; return false; }
    if (keep == 0 || keep > snap_n_) { err = "spec rollback keep outside the batch"; return false; }
    const uint32_t pos0 = snap_past_, np = pos0 + keep;
    for (uint32_t il = 0; il < nlayers_; ++il) {
        if (keep < snap_n_ &&
            !k::deepseek4_win_ring_snap(win_[il], snap_win_ + uint64_t(il) * cap2_ * HD_, pos0, keep, snap_n_, SWA_,
                                        HD_, true, nullptr, stream_)) {
            err = "spec window restore failed at L" + std::to_string(il);
            return false;
        }
        if (!is_kv_src_[il]) continue;
        const uint32_t ratio = static_cast<uint32_t>(geo.compress_ratios[il]);
        if (ratio == 0) continue;
        ng_src_[il] = np / ratio;
        if (ratio == 1 || cpre_kv_[il] == nullptr) { cpend_[il] = 0; continue; }
        const uint32_t pend2 = np % ratio;
        if (pend2) {
            // The last accepted tokens' inputs are rows (snap_cpend + keep - pend2) onward of the pre-shift snapshot.
            const uint64_t src = uint64_t(snap_cpend_[il] + keep - pend2) * HD_;
            cudaError_t ce = cudaMemcpyAsync(cpre_kv_[il], snap_cpre_kv_[il] + src, uint64_t(pend2) * HD_ * 4,
                                             cudaMemcpyDeviceToDevice, stream_);
            if (ce == cudaSuccess)
                ce = cudaMemcpyAsync(cpre_sc_[il], snap_cpre_sc_[il] + src, uint64_t(pend2) * HD_ * 4,
                                     cudaMemcpyDeviceToDevice, stream_);
            if (ce != cudaSuccess) { err = std::string("spec pending restore: ") + cudaGetErrorString(ce); return false; }
        }
        cpend_[il] = pend2;
    }
    n_past_ = np;
    snap_on_ = false;
    // Rejected rows leave the ring's contiguous run; their cells are overwritten next step, and until then the
    // window fill must not read them.
    const uint32_t dropped = snap_n_ - keep;
    mainh_n_ = mainh_n_ > dropped ? mainh_n_ - dropped : 0;
    mainh_end_ = static_cast<int64_t>(np) - 1;
    return true;
}

// ---- State buffers --------------------------------------------------------------------------------------------

bool Runner::alloc_state(uint32_t total, std::string& err) {
    const uint64_t ngg = static_cast<uint64_t>(total) + 2;   // groups cap, ratio 1
    topk_cap_ = static_cast<uint32_t>(std::min<uint64_t>(geo.indexer_top_k, ngg));
    if (topk_cap_ == 0) { err = "indexer_top_k is zero"; return false; }

    win_.assign(nlayers_, nullptr);
    comp_kv_.assign(nlayers_, nullptr);
    index_k_.assign(nlayers_, nullptr);
    cpre_kv_.assign(nlayers_, nullptr);
    cpre_sc_.assign(nlayers_, nullptr);
    // The expert cache's slot size is the largest selected-expert payload in the main layers: the three payload
    // spans of one expert.  (The container's table and codebook are small and are always read.)
    exp_cache_stride_ = 0;
    for (uint32_t il = 0; il < nlayers_; ++il) {
        const Blob& b = blob_[il];
        if (b.experts != NE_) continue;
        for (uint32_t e = 0; e < b.experts; ++e) {
            const size_t i = static_cast<size_t>(e) * 3;
            exp_cache_stride_ = std::max(exp_cache_stride_, b.span[i] + b.span[i + 1] + b.span[i + 2]);
        }
    }
    exp_cache_.assign(nstages_, {});
    exp_cache_dev_.assign(nstages_, nullptr);
    exp_cache_slots_.assign(nstages_, 0);
    exp_cache_next_.assign(nstages_, 0);
    for (uint32_t s = 0; s < 8; ++s) st_cache_hit_[s] = st_cache_miss_[s] = st_cache_bytes_[s] = 0;

    // Auto chunk (`--chunk 0`): the largest block that fits on every stage, picked here because the fit needs the
    // free memory after the weights and the engram are resident.  The ceiling is one block for the whole run: the
    // MoE kernel's grid allows n_tok * topk <= 65535, and 4096 rows is the widest block measured on the four-card
    // split.  Per row a stage carries the sum of every cap-scaled buffer allocated below, its layers' windows and
    // (if it owns one) its engram row buffers; beside that it must hold one layer's expert container -- the union
    // of the block's selected experts, which the measured runs put near a quarter of the rows and never above all
    // 384 -- plus the expert cache when one was asked for.  10% of free memory is left over.
    if (opt_.chunk == 0) {
        uint64_t limit = std::min<uint64_t>(total, 4096);
        limit = std::min<uint64_t>(limit, 65535u / std::max(1u, KU_));
        uint64_t floor_cap = 1;
        if (opt_.dspark) floor_cap = std::max<uint64_t>(floor_cap, cap2_);   // the verify batch's rows
        if (limit < floor_cap) limit = floor_cap;
        limit = std::min<uint64_t>(limit, ctx_);
        const uint64_t eng_stride = eng_hd_ + eng_hd_ / 32;
        uint64_t pick = limit;
        for (uint32_t s = 0; s < nstages_; ++s) {
            const uint32_t first = stage_first_[s], last = stage_first_[s + 1];
            uint64_t row = 0;
            row += 4 + 4;                                             // tokens, positions
            row += uint64_t(HC_) * 4;                                 // pre_mix
            row += uint64_t(E_) * 4 * 2;                              // x, xn
            row += uint64_t(HC_) * E_ * 4 * 2;                        // hc, hc2
            row += uint64_t(2u * HC_ + HC_ * HC_) * 4;                // mix
            row += uint64_t(2u * HC_ + HC_ * HC_) * 4;                // pre, post, comb
            row += uint64_t(Q_) * 4 * 2;                              // qr, qrn
            row += uint64_t(H_) * HD_ * 4 * 2;                        // q, o
            row += uint64_t(HD_) * 4 * 2;                             // kv, kvn
            row += uint64_t(NG_) * OL_ * 4;                           // low
            row += uint64_t(E_) * 4 * 3;                              // attn_out, shared out, moe y
            row += uint64_t(NE_) * 4;                                 // router logits
            row += uint64_t(KU_) * 4 * 2;                             // router sel, w
            row += uint64_t(KU_) * E_ * 4;                            // moe partial
            row += uint64_t(FF_) * 4 * 3;                             // shared gate, up, h
            row += uint64_t(HD_) * 4 * 4;                             // pooled, latent, compressor kv/score
            row += uint64_t(IK_) * 4 * 2;                             // ktmp, index key
            row += uint64_t(IH_) * IK_ * 4 + uint64_t(IH_) * 4;       // indexer q, w
            row += uint64_t(topk_cap_) * 4 + 4;                       // indexer topk, group positions
            row += uint64_t(last - first) * HD_ * 4;                  // one window a layer
            for (uint32_t il = first; il < last; ++il)
                if (is_kv_src_[il] && geo.compress_ratios[il] > 1) row += uint64_t(HD_) * 4 * 2;
            bool owns_engram = false;
            for (uint32_t il = first; il < last; ++il)
                if (engram_of_[il] >= 0 && engram_on_) owns_engram = true;
            if (owns_engram)   // row bytes + bf16 copy + f32 kv output
                row += uint64_t(eng_cols_) * eng_stride + uint64_t(eng_in_dim_) * 6 + uint64_t(eng_out_dim_) * 4;
            uint64_t fixed = uint64_t(V_) * 4;                        // logits
            for (uint32_t il = first; il < last; ++il)
                if (is_kv_src_[il]) fixed += ngg * (k::kDeepseek4CkvBytes + k::kDeepseek4IdxkBytes);
            uint64_t cache_b = 0;
            if (!opt_.expert_cache_mib.empty() && exp_cache_stride_ != 0) {
                const size_t pickm = opt_.expert_cache_mib.size() == 1
                                         ? 0
                                         : std::min<size_t>(s, opt_.expert_cache_mib.size() - 1);
                cache_b = static_cast<uint64_t>(opt_.expert_cache_mib[pickm]) << 20;
            }
            if (cudaSetDevice(st_[s].dev) != cudaSuccess) { err = last_error; return false; }
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { err = last_error; return false; }
            const uint64_t budget = static_cast<uint64_t>(static_cast<double>(free_b) * 0.9);
            auto fits = [&](uint64_t cap) {
                const uint64_t uniq = std::min<uint64_t>(NE_, 24 + cap / 3);   // the container's experts
                return cap * row + uniq * exp_cache_stride_ + fixed + cache_b <= budget;
            };
            if (!fits(floor_cap)) {
                std::fprintf(stderr,
                             "[v41] auto chunk: stage %u cannot hold %" PRIu64 " row(s) beside the weights and "
                             "the expert container; the run will fail to allocate\n",
                             s, floor_cap);
                pick = floor_cap;
                continue;
            }
            uint64_t lo = floor_cap, hi = limit;
            while (lo < hi) {
                const uint64_t mid = lo + (hi - lo + 1) / 2;
                if (fits(mid)) lo = mid; else hi = mid - 1;
            }
            std::fprintf(stderr,
                         "[v41] auto chunk: stage %u caps at %" PRIu64 " row(s) (%" PRIu64 " B/row, %.0f MiB "
                         "budget, %.0f MiB cache)\n",
                         s, lo, row, static_cast<double>(budget) / (1024.0 * 1024.0),
                         static_cast<double>(cache_b) / (1024.0 * 1024.0));
            pick = std::min(pick, lo);
        }
        cap_ = static_cast<uint32_t>(std::max<uint64_t>(1, pick));
    }
    // Every stage gets the whole shared scratch set, and the layers it owns get their window and source caches
    // on its device.  A layer's window is only ever touched by that layer, and a source's compressed-KV cache is
    // only touched by layers in the same stage (open() cut only at cache seams), so no buffer crosses a device.
    for (uint32_t s = 0; s < nstages_; ++s) {
        bind_stage(s);
        d_tok_ = static_cast<int32_t*>(dev(uint64_t(cap_) * 4, "tokens"));
        d_pos_ = static_cast<int32_t*>(dev(uint64_t(cap_) * 4, "positions"));
        d_prem_ = static_cast<float*>(dev(uint64_t(cap_) * HC_ * 4, "pre_mix"));
        x_ = static_cast<float*>(dev(uint64_t(cap_) * E_ * 4, "x"));
        hc_ = static_cast<float*>(dev(uint64_t(cap_) * HC_ * E_ * 4, "hc"));
        hc2_ = static_cast<float*>(dev(uint64_t(cap_) * HC_ * E_ * 4, "hc2"));
        xn_ = static_cast<float*>(dev(uint64_t(cap_) * E_ * 4, "xn"));
        const uint32_t mixd = 2u * HC_ + HC_ * HC_;
        mix_ = static_cast<float*>(dev(uint64_t(cap_) * mixd * 4, "mix"));
        pre_ = static_cast<float*>(dev(uint64_t(cap_) * HC_ * 4, "pre"));
        post_ = static_cast<float*>(dev(uint64_t(cap_) * HC_ * 4, "post"));
        comb_ = static_cast<float*>(dev(uint64_t(cap_) * HC_ * HC_ * 4, "comb"));
        qr_ = static_cast<float*>(dev(uint64_t(cap_) * Q_ * 4, "qr"));
        qrn_ = static_cast<float*>(dev(uint64_t(cap_) * Q_ * 4, "qrn"));
        q_ = static_cast<float*>(dev(uint64_t(cap_) * H_ * HD_ * 4, "q"));
        kvv_ = static_cast<float*>(dev(uint64_t(cap_) * HD_ * 4, "kv"));
        kvn_ = static_cast<float*>(dev(uint64_t(cap_) * HD_ * 4, "kvn"));
        o_ = static_cast<float*>(dev(uint64_t(cap_) * H_ * HD_ * 4, "o"));
        low_ = static_cast<float*>(dev(uint64_t(cap_) * NG_ * OL_ * 4, "low"));
        attn_out_ = static_cast<float*>(dev(uint64_t(cap_) * E_ * 4, "attn_out"));
        glog_ = static_cast<float*>(dev(uint64_t(cap_) * NE_ * 4, "router logits"));
        sel_ = static_cast<int32_t*>(dev(uint64_t(cap_) * KU_ * 4, "router sel"));
        rw_ = static_cast<float*>(dev(uint64_t(cap_) * KU_ * 4, "router w"));
        partial_ = static_cast<float*>(dev(uint64_t(cap_) * KU_ * E_ * 4, "moe partial"));
        sg_ = static_cast<float*>(dev(uint64_t(cap_) * FF_ * 4, "shared gate"));
        su_ = static_cast<float*>(dev(uint64_t(cap_) * FF_ * 4, "shared up"));
        sh_ = static_cast<float*>(dev(uint64_t(cap_) * FF_ * 4, "shared h"));
        so_ = static_cast<float*>(dev(uint64_t(cap_) * E_ * 4, "shared out"));
        y_ = static_cast<float*>(dev(uint64_t(cap_) * E_ * 4, "moe y"));
        logits_ = static_cast<float*>(dev(uint64_t(V_) * 4, "logits"));
        pooled_ = static_cast<float*>(dev(uint64_t(cap_) * HD_ * 4, "pooled"));
        latent_ = static_cast<float*>(dev(uint64_t(cap_) * HD_ * 4, "latent"));
        ktmp_ = static_cast<float*>(dev(uint64_t(cap_) * IK_ * 4, "ktmp"));
        cke_ = static_cast<float*>(dev(uint64_t(cap_) * IK_ * 4, "index key"));
        ckv_ = static_cast<float*>(dev(uint64_t(cap_) * HD_ * 4, "compressor kv"));
        csc_ = static_cast<float*>(dev(uint64_t(cap_) * HD_ * 4, "compressor score"));
        iq_ = static_cast<float*>(dev(uint64_t(cap_) * IH_ * IK_ * 4, "indexer q"));
        iw_ = static_cast<float*>(dev(uint64_t(cap_) * IH_ * 4, "indexer w"));
        idx_ = static_cast<int32_t*>(dev(uint64_t(cap_) * topk_cap_ * 4, "indexer topk"));
        posg_ = static_cast<int32_t*>(dev(uint64_t(cap_) * 4, "group positions"));
        // A stage that owns an engram layer (1 and 14 here) holds its four row buffers; no other stage touches
        // them (see load_engram).
        eraw_ = nullptr;
        erows_ = nullptr;
        x16_ = nullptr;
        ekv_ = nullptr;
        bool owns_engram = false;
        for (uint32_t il = stage_first_[s]; il < stage_first_[s + 1]; ++il)
            if (engram_of_[il] >= 0 && engram_on_) owns_engram = true;
        if (owns_engram) {
            const uint64_t eng_stride = eng_hd_ + eng_hd_ / 32;
            eraw_ = static_cast<uint8_t*>(dev(uint64_t(cap_) * eng_cols_ * eng_stride, "eraw"));
            erows_ = static_cast<float*>(dev(uint64_t(cap_) * eng_in_dim_ * 4, "erows"));
            x16_ = static_cast<uint16_t*>(dev(uint64_t(cap_) * eng_in_dim_ * 2, "engram x bf16"));
            ekv_ = static_cast<float*>(dev(uint64_t(cap_) * eng_out_dim_ * 4, "engram kv"));
            if (eraw_ == nullptr || erows_ == nullptr || x16_ == nullptr || ekv_ == nullptr) {
                err = last_error;
                return false;
            }
        }

        for (uint32_t il = stage_first_[s]; il < stage_first_[s + 1]; ++il) {
            win_[il] = static_cast<float*>(dev(uint64_t(SWA_ + cap_) * HD_ * 4, "window"));
            if (is_kv_src_[il]) {
                comp_kv_[il] = static_cast<uint8_t*>(dev(ngg * k::kDeepseek4CkvBytes, "compressed kv cache"));
                index_k_[il] = static_cast<uint8_t*>(dev(ngg * k::kDeepseek4IdxkBytes, "index key cache"));
                const uint32_t ratio = static_cast<uint32_t>(geo.compress_ratios[il]);
                if (ratio > 1) {
                    cpre_kv_[il] = static_cast<float*>(dev(uint64_t(ratio + cap_) * HD_ * 4, "compressor pending kv"));
                    cpre_sc_[il] = static_cast<float*>(dev(uint64_t(ratio + cap_) * HD_ * 4, "compressor pending score"));
                    if (cpre_kv_[il] == nullptr || cpre_sc_[il] == nullptr) { err = last_error; return false; }
                }
                if (comp_kv_[il] == nullptr || index_k_[il] == nullptr) { err = last_error; return false; }
            }
            if (win_[il] == nullptr) { err = last_error; return false; }
        }
        if (d_tok_ == nullptr || d_pos_ == nullptr || d_prem_ == nullptr || x_ == nullptr || hc_ == nullptr ||
            hc2_ == nullptr || xn_ == nullptr || mix_ == nullptr || pre_ == nullptr || post_ == nullptr ||
            comb_ == nullptr || qr_ == nullptr || qrn_ == nullptr || q_ == nullptr || kvv_ == nullptr ||
            kvn_ == nullptr || o_ == nullptr || low_ == nullptr || attn_out_ == nullptr || glog_ == nullptr ||
            sel_ == nullptr || rw_ == nullptr || partial_ == nullptr || sg_ == nullptr || su_ == nullptr ||
            sh_ == nullptr || so_ == nullptr || y_ == nullptr || logits_ == nullptr || pooled_ == nullptr ||
            latent_ == nullptr || ktmp_ == nullptr || cke_ == nullptr || ckv_ == nullptr || csc_ == nullptr ||
            iq_ == nullptr ||
            iw_ == nullptr || idx_ == nullptr || posg_ == nullptr) {
            err = last_error;
            return false;
        }
        // The expert cache (opt-in) rides on top of the state: whole payload slots, so a hit is one copy.
        if (!opt_.expert_cache_mib.empty() && exp_cache_stride_ != 0) {
            const size_t pick = opt_.expert_cache_mib.size() == 1
                                    ? 0
                                    : std::min<size_t>(s, opt_.expert_cache_mib.size() - 1);
            const uint32_t mib = opt_.expert_cache_mib[pick];
            const uint64_t slots = (static_cast<uint64_t>(mib) << 20) / exp_cache_stride_;
            if (slots != 0) {
                exp_cache_dev_[s] = static_cast<uint8_t*>(dev(slots * exp_cache_stride_, "expert cache"));
                if (exp_cache_dev_[s] == nullptr) { err = last_error; return false; }
                exp_cache_slots_[s] = slots;
                std::fprintf(stderr, "[v41] expert cache: stage %u %u MiB = %" PRIu64 " slot(s) of %" PRIu64 " B\n",
                             s, mib, slots, exp_cache_stride_);
            }
        }
        save_stage();
    }
    pm_host_.resize(static_cast<size_t>(cap_) * HC_);
    pos_host_.resize(cap_);
    dpos_host_.resize(cap_);
    if (engram_on_) {
        const uint64_t eng_stride = eng_hd_ + eng_hd_ / 32;
        eraw_host_.resize(static_cast<size_t>(uint64_t(cap_) * eng_cols_ * eng_stride));
        erow_ids_.resize(static_cast<size_t>(uint64_t(cap_) * eng_cols_));
        if (opt_.engram_feed && efeed_.empty()) {
            const size_t layers = geo.engram_layer_ids.size();
            const size_t raw_sz = static_cast<size_t>(uint64_t(cap_) * eng_cols_ * eng_stride);
            const size_t id_sz = static_cast<size_t>(uint64_t(cap_) * eng_cols_);
            efeed_.resize(layers);
            efeed_raw_.resize(layers);
            efeed_ids_.resize(layers);
            efeed_ev_.resize(layers);
            efeed_ev_set_.resize(layers);
            efeed_cur_.assign(layers, -1);
            efeed_dev_.resize(layers);
            for (size_t ei = 0; ei < layers; ++ei) {
                const uint32_t il = static_cast<uint32_t>(geo.engram_layer_ids[ei]);
                efeed_dev_[ei] = st_[stage_of(il)].dev;
                for (int s = 0; s < 2; ++s) {
                    efeed_raw_[ei][s].resize(raw_sz);
                    efeed_ids_[ei][s].resize(id_sz);
                    efeed_ev_set_[ei][s] = false;
                    cudaSetDevice(efeed_dev_[ei]);
                    if (cudaEventCreateWithFlags(&efeed_ev_[ei][s], cudaEventDisableTiming) != cudaSuccess) {
                        err = "engram feed: cannot create the buffer event";
                        return false;
                    }
                }
            }
            cudaSetDevice(st_[0].dev);
            std::fprintf(stderr, "[v41] engram feed: %zu layer(s), two %u-row buffers each, prefetch on\n", layers,
                         cap_);
        }
    }
    sel_host_.resize(static_cast<size_t>(cap_) * KU_);
    logits_host_.resize(V_);
    if (opt_.ledger) seen_exp_.assign(static_cast<size_t>(nlayers_) * NE_, 0);
    if (nstages_ > 1) {
        handoff_hc_.resize(static_cast<size_t>(cap_) * HC_ * E_);
        handoff_prem_.resize(static_cast<size_t>(cap_) * HC_);
    }
    std::fprintf(stderr, "[v41] state: chunk %u, groups cap %" PRIu64 ", topk cap %u, window %u\n", cap_, ngg,
                 topk_cap_, SWA_);
    if (opt_.chunk == 0)
        std::fprintf(stderr, "[v41] auto chunk: %u row(s) for a %u-row run\n", cap_, total);
    return true;
}

bool Runner::ensure_index_scratch(uint32_t rows, uint32_t ng, std::string& err) {
    if (iscore_ != nullptr && ng <= iscap_ && rows <= isrows_ && (cand_ != nullptr || kcap_ == 0)) return true;
    const uint32_t want = std::min<uint32_t>(std::max<uint32_t>(iscap_ ? iscap_ * 2u : ng, ng), ctx_);
    const uint32_t rr = std::max<uint32_t>(rows, kDecRows);
    if (iscore_ != nullptr) cudaFree(iscore_);
    if (cand_ != nullptr) cudaFree(cand_);
    if (iscratch_ != nullptr) cudaFree(iscratch_);
    iscore_ = static_cast<float*>(dev(uint64_t(rr) * want * 4, "indexer scores"));
    cand_ = kcap_ != 0 ? static_cast<int32_t*>(dev(uint64_t(rr) * (1u + kcap_) * 4, "candidate blocks")) : nullptr;
    nbc_ = static_cast<uint32_t>((uint64_t(want) + cand_bs_ - 1) / cand_bs_) + 1;
    iscratch_ = static_cast<float*>(dev(uint64_t(rr) * nbc_ * 5, "candidate scratch"));
    if (iscore_ == nullptr || (kcap_ != 0 && cand_ == nullptr) || iscratch_ == nullptr) { err = last_error; return false; }
    iscap_ = want;
    isrows_ = rr;
    return true;
}

// ---- The layer ------------------------------------------------------------------------------------------------

bool Runner::engram_layer(uint32_t il, std::string& err) {
    const int ei = engram_of_[il];
    if (ei < 0 || !engram_on_) return true;
    const uint32_t n = cur_n_;
    const uint32_t cols = eng_cols_;
    const uint32_t stride = eng_hd_ + eng_hd_ / 32;
    const uint64_t raw_bytes = uint64_t(n) * cols * stride;
    cudaError_t ce = cudaSuccess;
    if (opt_.engram_feed) {
        // The rows were read by the feed (prepare started them, or a previous block's prefetch did); this layer
        // only waits for them and uploads.  The event marks the slot free for the worker's next read.
        if (!engram_feed_wait(static_cast<uint32_t>(ei), err)) return false;
        const size_t slot = static_cast<size_t>(efeed_cur_[static_cast<size_t>(ei)]);
        ce = cudaMemcpyAsync(eraw_, efeed_raw_[static_cast<size_t>(ei)][slot].data(),
                             static_cast<size_t>(raw_bytes), cudaMemcpyHostToDevice, stream_);
        if (ce == cudaSuccess) ce = cudaEventRecord(efeed_ev_[static_cast<size_t>(ei)][slot], stream_);
        efeed_ev_set_[static_cast<size_t>(ei)][slot] = true;
    } else {
        for (uint32_t i = 0; i < n; ++i) {
            hash_.rows_masked(hist_.data(), hist_.size(), nullptr, 0, static_cast<int64_t>(cur_pos0_ + i), ei,
                              erow_ids_.data() + static_cast<size_t>(i) * cols);
        }
        const d4::EngramTable& t = geo.engram_tables[static_cast<size_t>(ei)];
        if (!shard_[ei]->rows(t.weight_offset, t.scale_offset, erow_ids_.data(), static_cast<size_t>(n) * cols,
                              eng_hd_, eraw_host_.data(), err))
            return false;
        ce = cudaMemcpyAsync(eraw_, eraw_host_.data(), static_cast<size_t>(raw_bytes), cudaMemcpyHostToDevice,
                             stream_);
    }
    if (ce != cudaSuccess) { err = std::string("engram rows upload: ") + cudaGetErrorString(ce); return false; }
    if (!k::deepseek4_engram_rows(eraw_, erows_, n * cols, eng_hd_, stream_)) {
        err = "deepseek4_engram_rows failed";
        return false;
    }
    if (n > 1 && !k::deepseek4_engram_x_to_bf16(erows_, x16_, uint64_t(n) * eng_in_dim_, stream_)) {
        err = "deepseek4_engram_x_to_bf16 failed";
        return false;
    }
    if (!k::deepseek4_engram_wkv(static_cast<const uint8_t*>(L_[il].eng_wkv), erows_, n > 1 ? x16_ : nullptr, ekv_,
                                 eng_in_dim_, eng_out_dim_, n, stream_)) {
        err = "deepseek4_engram_wkv failed";
        return false;
    }
    if (!k::deepseek4_round_bf16(ekv_, uint64_t(n) * eng_out_dim_, stream_) ||
        !k::deepseek4_engram_gate(hc_, ekv_, static_cast<const float*>(L_[il].eng_q),
                                  static_cast<const float*>(L_[il].eng_k), E_, HC_, n, rms_eps_, stream_)) {
        err = "engram gate failed";
        return false;
    }
    return true;
}

// ---- The engram feed (P7, opt-in) -----------------------------------------------------------------------------
//
// The reference's feed (reference/ds4-dfm-rs/ds4_ds41_forward.inc, ds4_v41_feed): a block's engram row ids depend
// only on token ids and positions, so the reads for the next block can start while the current block computes.
// The hash is stateful and stays on the main thread; a worker per (engram layer, buffer slot) only reads the
// shard.  Two buffers per layer alternate: the block in hand reads from one while a prefetch fills the other, so
// each engram layer is a join plus the H2D instead of a wait for the disk.  The device path is unchanged (the
// rows land in the same `eraw_` and go through the same kernels), so --engram-feed must return the same ids.
// Two slots of one layer can read at once (the first block's prepare and its prefetch): EngramShard's Unix
// path is pread-only, so that is safe; its Windows FILE* path would need a lock and the feed stays off there.

Runner::~Runner() { engram_feed_close(); }

void Runner::engram_feed_close() {
    for (auto& jobs : efeed_)
        for (auto& j : jobs) {
            if (j.running) {
                j.th.join();
                j.running = false;
            }
        }
    for (size_t ei = 0; ei < efeed_ev_.size(); ++ei)
        for (int s = 0; s < 2; ++s) {
            if (efeed_ev_[ei][s] == nullptr) continue;
            cudaSetDevice(efeed_dev_[ei]);
            cudaEventDestroy(efeed_ev_[ei][s]);
            efeed_ev_[ei][s] = nullptr;
        }
    efeed_ev_set_.clear();
}

bool Runner::engram_feed_start(uint32_t ei, int slot, uint32_t pos0, const int32_t* hist, size_t hist_len,
                               const int32_t* ids, uint32_t n) {
    if (n == 0 || pos0 > hist_len) return false;
    // The hasher reads its history by absolute position; hand it the n-gram window plus this block's tokens at
    // the positions they will have.  Positions before 0 stay the pad id, exactly as the full-history call has it.
    const uint32_t cols = eng_cols_;
    const uint32_t g = hash_.max_ngram;
    const uint32_t back = std::min<uint32_t>(g > 0 ? g - 1u : 0u, pos0);
    std::vector<int32_t> win(static_cast<size_t>(back) + n);
    if (back != 0) std::memcpy(win.data(), hist + (pos0 - back), static_cast<size_t>(back) * 4);
    std::memcpy(win.data() + back, ids, static_cast<size_t>(n) * 4);
    int64_t* out = efeed_ids_[ei][static_cast<size_t>(slot)].data();
    for (uint32_t i = 0; i < n; ++i)
        hash_.rows_masked(win.data(), win.size(), nullptr, 0, static_cast<int64_t>(back) + i,
                          static_cast<uint32_t>(ei), out + static_cast<size_t>(i) * cols);
    FeedJob& j = efeed_[ei][static_cast<size_t>(slot)];
    j.ok = true;
    j.err.clear();
    j.pos0 = pos0;
    j.n = n;
    j.done = false;
    const size_t rows = static_cast<size_t>(n) * cols;
    j.th = std::thread([this, ei, slot, rows]() {
        FeedJob& jj = efeed_[ei][static_cast<size_t>(slot)];
        // The slot may still be the source of an upload recorded a block or two ago.
        if (efeed_ev_set_[ei][static_cast<size_t>(slot)]) {
            cudaSetDevice(efeed_dev_[ei]);
            if (cudaEventSynchronize(efeed_ev_[ei][static_cast<size_t>(slot)]) != cudaSuccess) {
                jj.ok = false;
                jj.err = "buffer event: " + std::string(cudaGetErrorString(cudaGetLastError()));
                return;
            }
        }
        const d4::EngramTable& t = geo.engram_tables[ei];
        std::string e;
        if (!shard_[ei]->rows(t.weight_offset, t.scale_offset, efeed_ids_[ei][static_cast<size_t>(slot)].data(),
                              rows, eng_hd_, efeed_raw_[ei][static_cast<size_t>(slot)].data(), e)) {
            jj.ok = false;
            jj.err = e;
        }
    });
    j.running = true;
    return true;
}

bool Runner::engram_feed_prepare(uint32_t pos0, const int32_t* ids, uint32_t n, std::string& err) {
    if (!opt_.engram_feed || !engram_on_) return true;
    for (size_t ei = 0; ei < efeed_.size(); ++ei) {
        auto& jobs = efeed_[ei];
        int slot = -1;
        for (int s = 0; s < 2; ++s) {
            const FeedJob& j = jobs[static_cast<size_t>(s)];
            if ((j.running || j.done) && j.pos0 == pos0 && j.n == n) { slot = s; break; }
        }
        if (slot >= 0) {
            FeedJob& j = jobs[static_cast<size_t>(slot)];
            if (j.running) {
                j.th.join();
                j.running = false;
                if (!j.ok) { err = "engram feed: " + j.err; return false; }
            }
            j.done = true;
        } else {
            // No prefetched rows for this block: read them here.  The read overlaps the layers before the
            // engram layer; wait() joins it there.
            const int cur = efeed_cur_[ei];
            slot = cur < 0 ? 0 : 1 - cur;
            FeedJob& j = jobs[static_cast<size_t>(slot)];
            if (j.running) {   // a stale read of another block; its rows are not this block's
                j.th.join();
                j.running = false;
            }
            j.done = false;
            if (!engram_feed_start(static_cast<uint32_t>(ei), slot, pos0, hist_.data(), hist_.size(), ids, n)) {
                err = "engram feed: cannot start the read";
                return false;
            }
        }
        efeed_cur_[ei] = slot;
    }
    return true;
}

void Runner::engram_feed_prefetch(uint32_t pos0, const int32_t* hist, const int32_t* ids, uint32_t n) {
    if (!opt_.engram_feed || !engram_on_ || n == 0) return;
    for (size_t ei = 0; ei < efeed_.size(); ++ei) {
        const int cur = efeed_cur_[ei];
        if (cur < 0) continue;   // prepare has not run yet
        const int slot = 1 - cur;
        FeedJob& j = efeed_[ei][static_cast<size_t>(slot)];
        if (j.running) continue;   // the other slot is still being read; this block will be read at prepare
        if (j.done && j.pos0 == pos0 && j.n == n) continue;
        j.done = false;
        // A failure leaves the job object clean and the next prepare reads the block synchronously.
        (void)engram_feed_start(static_cast<uint32_t>(ei), slot, pos0, hist, pos0, ids, n);
    }
}

bool Runner::engram_feed_wait(uint32_t ei, std::string& err) {
    FeedJob& j = efeed_[ei][static_cast<size_t>(efeed_cur_[ei])];
    if (j.running) {
        j.th.join();
        j.running = false;
        if (j.ok) j.done = true;
    }
    if (!j.ok) { err = "engram feed: " + j.err; return false; }
    if (!j.done || j.pos0 != cur_pos0_ || j.n != cur_n_) {
        err = "engram feed: the rows in hand are not this block's";
        return false;
    }
    return true;
}

bool Runner::hc_half(uint32_t il, bool attn_half, std::string& err) {
    const LayerW& l = L_[il];
    const float* fn = static_cast<const float*>(attn_half ? l.hc_attn_fn : l.hc_ffn_fn);
    const float* sc = static_cast<const float*>(attn_half ? l.hc_attn_scale : l.hc_ffn_scale);
    const float* bs = static_cast<const float*>(attn_half ? l.hc_attn_base : l.hc_ffn_base);
    const float* nm = static_cast<const float*>(attn_half ? l.attn_norm : l.ffn_norm);
    if (!k::deepseek4_hc_mix(fn, hc_, mix_, E_, HC_, cur_n_, rms_eps_, stream_)) {
        err = "deepseek4_hc_mix failed";
        return false;
    }
    if (!k::deepseek4_hc_fused(mix_, sc, bs, d_prem_, hc_, nm, pre_, post_, comb_, x_, xn_, E_, HC_, sinkhorn_,
                               hc_eps_, rms_eps_, cur_n_, stream_)) {
        err = "deepseek4_hc_fused failed";
        return false;
    }
    std::swap(pre_, d_prem_);
    return true;
}

bool Runner::compress_source(uint32_t il, uint32_t ratio, std::string& err) {
    const LayerW& l = L_[il];
    const uint32_t n = cur_n_;
    const uint32_t g0 = cur_pos0_ / ratio;
    uint32_t ng_new = 0;
    if (ratio > 1) {
        if (!k::deepseek4_matmul_bf16(ckv_, static_cast<const uint16_t*>(l.comp_kv), xn_, E_, HD_, n, false,
                                      stream_)) {
            err = "compressor kv projection failed";
            return false;
        }
        if (!k::deepseek4_matmul_bf16(csc_, static_cast<const uint16_t*>(l.comp_gate), xn_, E_, HD_, n, false,
                                      stream_)) {
            err = "compressor gate projection failed";
            return false;
        }
        const uint32_t pend = cpend_[il];
        const uint64_t rowb = uint64_t(HD_) * 4;
        cudaError_t ce = cudaMemcpyAsync(cpre_kv_[il] + uint64_t(pend) * HD_, ckv_, n * rowb,
                                         cudaMemcpyDeviceToDevice, stream_);
        if (ce == cudaSuccess)
            ce = cudaMemcpyAsync(cpre_sc_[il] + uint64_t(pend) * HD_, csc_, n * rowb, cudaMemcpyDeviceToDevice,
                                 stream_);
        if (ce != cudaSuccess) { err = std::string("compressor append: ") + cudaGetErrorString(ce); return false; }
        const uint32_t tot = pend + n;
        if (snap_on_ && snap_cpre_kv_[il] != nullptr) {
            // The verify batch's pending rows, before the pool and the shift: the rollback takes the accepted
            // tokens' inputs back out of this (the reference snapshots inside its compress_source, same point).
            const uint64_t bytes = uint64_t(tot) * HD_ * 4;
            ce = cudaMemcpyAsync(snap_cpre_kv_[il], cpre_kv_[il], bytes, cudaMemcpyDeviceToDevice, stream_);
            if (ce == cudaSuccess)
                ce = cudaMemcpyAsync(snap_cpre_sc_[il], cpre_sc_[il], bytes, cudaMemcpyDeviceToDevice, stream_);
            if (ce != cudaSuccess) {
                err = std::string("compressor snapshot: ") + cudaGetErrorString(ce);
                return false;
            }
            snap_cpend_[il] = pend;
        }
        const uint32_t rem = tot % ratio;
        ng_new = tot / ratio;
        if (ng_new) {
            if (!k::deepseek4_compress_pool(pooled_, cpre_kv_[il], cpre_sc_[il], tot, ratio, HD_, stream_)) {
                err = "deepseek4_compress_pool failed";
                return false;
            }
            if (rem) {
                ce = cudaMemcpyAsync(cpre_kv_[il], cpre_kv_[il] + uint64_t(ng_new) * ratio * HD_,
                                     uint64_t(rem) * rowb, cudaMemcpyDeviceToDevice, stream_);
                if (ce == cudaSuccess)
                    ce = cudaMemcpyAsync(cpre_sc_[il], cpre_sc_[il] + uint64_t(ng_new) * ratio * HD_,
                                         uint64_t(rem) * rowb, cudaMemcpyDeviceToDevice, stream_);
                if (ce != cudaSuccess) {
                    err = std::string("compressor remainder move: ") + cudaGetErrorString(ce);
                    return false;
                }
            }
        }
        cpend_[il] = rem;
    } else {
        if (!k::deepseek4_matmul_bf16(pooled_, static_cast<const uint16_t*>(l.comp_kv), xn_, E_, HD_, n, true,
                                      stream_)) {
            err = "compressor projection failed";
            return false;
        }
        ng_new = n;
    }
    ng_src_[il] = g0 + ng_new;
    if (ng_new == 0) return true;

    for (uint32_t g = 0; g < ng_new; ++g) pos_host_[g] = static_cast<int32_t>((g0 + g) * ratio);
    cudaError_t ce = cudaMemcpyAsync(posg_, pos_host_.data(), uint64_t(ng_new) * 4, cudaMemcpyHostToDevice, stream_);
    if (ce != cudaSuccess) { err = std::string("group positions upload: ") + cudaGetErrorString(ce); return false; }

    const float theta = theta_of(ratio);
    const uint32_t osl = osl_of(ratio);
    if (!k::deepseek4_rms_norm(latent_, pooled_, static_cast<const float*>(l.comp_norm), HD_, ng_new, rms_eps_,
                               stream_)) {
        err = "compressor norm failed";
        return false;
    }
    if (!k::deepseek4_matmul_bf16(ktmp_, static_cast<const uint16_t*>(l.idx_wk), latent_, HD_, IK_, ng_new, true,
                                  stream_)) {
        err = "indexer wk failed";
        return false;
    }
    if (!k::deepseek4_rms_norm(cke_, ktmp_, static_cast<const float*>(l.idx_k_norm), IK_, ng_new, rms_eps_,
                               stream_)) {
        err = "indexer k norm failed";
        return false;
    }
    if (!k::deepseek4_rope(cke_, posg_, ng_new, 1, IK_, nrot_, theta, osl, rope_factor_, beta_fast_, beta_slow_,
                           false, stream_)) {
        err = "indexer key rope failed";
        return false;
    }
    if (!k::deepseek4_idxk_pack(index_k_[il], g0, cke_, ng_new, stream_)) {
        err = "deepseek4_idxk_pack failed";
        return false;
    }
    if (!k::deepseek4_rope(latent_, posg_, ng_new, 1, HD_, nrot_, theta, osl, rope_factor_, beta_fast_, beta_slow_,
                           false, stream_)) {
        err = "compressed kv rope failed";
        return false;
    }
    if (!k::deepseek4_ckv_pack(comp_kv_[il], g0, latent_, ng_new, stream_)) {
        err = "deepseek4_ckv_pack failed";
        return false;
    }
    return true;
}

bool Runner::index_source(uint32_t il, uint32_t ratio, std::string& err) {
    const LayerW& l = L_[il];
    const uint32_t n = cur_n_;
    const uint32_t src = kv_src_[il];
    const uint32_t ng = ng_src_[src];
    idx_owner_ = static_cast<int>(il);
    if (ng == 0) {
        idx_topk_ = 0;
        return true;
    }
    uint32_t rb = n <= kDecRows ? n : n / 32u;
    if (rb < kDecRows) rb = kDecRows < n ? kDecRows : n;
    if (!ensure_index_scratch(std::max(rb, n), ng, err)) return false;

    if (!k::deepseek4_matmul_q4k(iq_, l.idx_q_b, qrn_, Q_, uint64_t(IH_) * IK_, n, true, stream_)) {
        err = "indexer q_b failed";
        return false;
    }
    const float theta = theta_of(ratio);
    const uint32_t osl = osl_of(ratio);
    if (!k::deepseek4_rope(iq_, d_pos_, n, IH_, IK_, nrot_, theta, osl, rope_factor_, beta_fast_, beta_slow_, false,
                           stream_)) {
        err = "indexer q rope failed";
        return false;
    }
    if (!k::deepseek4_act_quant(iq_, n * IH_, IK_, 32, k::kDeepseek4QuantFp4E8m0, stream_)) {
        err = "indexer q quant failed";
        return false;
    }
    if (!k::deepseek4_matmul_bf16(iw_, static_cast<const uint16_t*>(l.idx_proj), xn_, E_, IH_, n, false, stream_) ||
        !k::deepseek4_round_bf16(iw_, uint64_t(n) * IH_, stream_)) {
        err = "indexer proj failed";
        return false;
    }
    const float ws = static_cast<float>(1.0 / std::sqrt(static_cast<double>(IK_)) / std::sqrt(static_cast<double>(IH_)));
    if (!k::deepseek4_scale_round(iw_, uint64_t(n) * IH_, ws, stream_)) {
        err = "indexer weight scale failed";
        return false;
    }
    const bool uses_cand = kcap_ > 0 && il > cand_src_layer_ && cand_owner_ >= 0 && cand_ != nullptr;
    const bool is_cand = kcap_ > 0 && il == cand_src_layer_ && cand_ != nullptr;
    const uint32_t topk = std::min<uint32_t>(static_cast<uint32_t>(geo.indexer_top_k), ng);
    for (uint32_t b0 = 0; b0 < n; b0 += rb) {
        const uint32_t nr = std::min(rb, n - b0);
        const float* iq_b = iq_ + uint64_t(b0) * IH_ * IK_;
        const float* iw_b = iw_ + uint64_t(b0) * IH_;
        int32_t* cd_b = cand_ != nullptr ? cand_ + uint64_t(b0) * (1u + kcap_) : nullptr;
        int32_t* ix_b = idx_ + uint64_t(b0) * topk;
        const int32_t* cl = uses_cand ? cd_b : nullptr;
        if (!k::deepseek4_indexer_score(iscore_, iq_b, index_k_[src], iw_b, cl, cur_pos0_ + b0, ng, nr, IH_, IK_,
                                        ratio, cand_bs_, kcap_, nullptr, stream_)) {
            err = "deepseek4_indexer_score failed";
            return false;
        }
        if (is_cand && !k::deepseek4_candidate_blocks(cd_b, iscore_, iscratch_, nr, cur_pos0_ + b0, ng, ratio, kcap_,
                                                      cand_bs_, nbc_, nullptr, stream_)) {
            err = "deepseek4_candidate_blocks failed";
            return false;
        }
        if (!k::deepseek4_indexer_topk(ix_b, iscore_, nr, ng, topk, ratio, cl, cand_bs_, kcap_, nullptr, stream_)) {
            err = "deepseek4_indexer_topk failed";
            return false;
        }
    }
    if (is_cand) cand_owner_ = static_cast<int>(il);
    idx_topk_ = topk;
    return true;
}

bool Runner::attn(uint32_t il, std::string& err) {
    const LayerW& l = L_[il];
    const uint32_t n = cur_n_;
    const uint32_t ratio = static_cast<uint32_t>(geo.compress_ratios[il]);
    const float theta = theta_of(ratio);
    const uint32_t osl = osl_of(ratio);

    // in: window kv then the q path
    if (!k::deepseek4_matmul_q4k(kvv_, l.kv, xn_, E_, HD_, n, true, stream_)) {
        err = "attn kv failed";
        return false;
    }
    if (!k::deepseek4_rms_norm(kvn_, kvv_, static_cast<const float*>(l.kv_a_norm), HD_, n, rms_eps_, stream_) ||
        !k::deepseek4_rope(kvn_, d_pos_, n, 1, HD_, nrot_, theta, osl, rope_factor_, beta_fast_, beta_slow_, false,
                           stream_) ||
        !k::deepseek4_act_quant(kvn_, n, HD_, 32, k::kDeepseek4QuantFp8E8m0, stream_)) {
        err = "attn kv norm/rope/quant failed";
        return false;
    }
    if (!k::deepseek4_matmul_q4k(qr_, l.q_a, xn_, E_, Q_, n, true, stream_) ||
        !k::deepseek4_rms_norm(qrn_, qr_, static_cast<const float*>(l.q_a_norm), Q_, n, rms_eps_, stream_) ||
        !k::deepseek4_matmul_q4k(q_, l.q_b, qrn_, Q_, uint64_t(H_) * HD_, n, true, stream_) ||
        !k::deepseek4_rope(q_, d_pos_, n, H_, HD_, nrot_, theta, osl, rope_factor_, beta_fast_, beta_slow_, false,
                           stream_)) {
        err = "attn q path failed";
        return false;
    }
    // cache: the batch's window rows, the sources, the sparse attention, then the ring commit
    cudaError_t ce = cudaMemcpyAsync(win_[il] + uint64_t(SWA_) * HD_, kvn_, uint64_t(n) * HD_ * 4,
                                     cudaMemcpyDeviceToDevice, stream_);
    if (ce != cudaSuccess) { err = std::string("window append: ") + cudaGetErrorString(ce); return false; }
    uint32_t ng = 0, topk = 0;
    if (ratio) {
        if (is_kv_src_[il] && !compress_source(il, ratio, err)) return false;
        if (is_idx_src_[il] && !index_source(il, ratio, err)) return false;
        ng = ng_src_[kv_src_[il]];
        if (ng) {
            if (idx_owner_ < 0) { err = "compressed layer has no topk"; return false; }
            topk = idx_topk_;
        }
    }
    const bool compressed = ng != 0 && topk != 0;
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(HD_)));
    if (!k::deepseek4_sparse_attn(o_, q_, win_[il], compressed ? comp_kv_[kv_src_[il]] : nullptr,
                                  compressed ? idx_ : nullptr, static_cast<const float*>(l.sinks), n, cur_pos0_,
                                  SWA_, ng, topk, H_, HD_, scale, 0, true, 0, stream_)) {
        err = "deepseek4_sparse_attn failed";
        return false;
    }
    if (!k::deepseek4_win_commit(win_[il], cur_pos0_, n, SWA_, HD_, nullptr, stream_)) {
        err = "deepseek4_win_commit failed";
        return false;
    }
    // out: inverse rope, the grouped output, then wo_b
    const uint32_t grp = H_ / NG_;
    if (!k::deepseek4_rope(o_, d_pos_, n, H_, HD_, nrot_, theta, osl, rope_factor_, beta_fast_, beta_slow_, true,
                           stream_) ||
        !k::deepseek4_matmul_q4k_grouped(low_, l.out_a, o_, NG_, grp * HD_, OL_, n, true, stream_) ||
        !k::deepseek4_matmul_q4k(attn_out_, l.out_b, low_, uint64_t(NG_) * OL_, E_, n, true, stream_)) {
        err = "attn output path failed";
        return false;
    }
    return true;
}

bool Runner::moe(uint32_t il, std::string& err) {
    const LayerW& l = L_[il];
    const uint32_t n = cur_n_;
    // shared expert: w1/w3 -> swiglu(clamp) -> w2
    if (!k::deepseek4_matmul_q4k(sg_, l.shexp_gate, xn_, E_, FF_, n, true, stream_) ||
        !k::deepseek4_matmul_q4k(su_, l.shexp_up, xn_, E_, FF_, n, true, stream_) ||
        !k::deepseek4_swiglu(sh_, sg_, su_, uint64_t(n) * FF_, clamp_, stream_) ||
        !k::deepseek4_matmul_q4k(so_, l.shexp_down, sh_, FF_, E_, n, true, stream_)) {
        err = "shared expert failed";
        return false;
    }
    // router (bf16 gate), then the routed VQ experts
    if (!k::deepseek4_matmul_bf16(glog_, static_cast<const uint16_t*>(l.gate_inp), xn_, E_, NE_, n, false, stream_)) {
        err = "router gate failed";
        return false;
    }
    if (!k::deepseek4_router(sel_, rw_, glog_, static_cast<const float*>(l.exp_bias), nullptr, nullptr, n, NE_, KU_,
                             route_scale_, stream_)) {
        err = "deepseek4_router failed";
        return false;
    }
    cudaError_t ce = cudaMemcpyAsync(sel_host_.data(), sel_, uint64_t(n) * KU_ * 4, cudaMemcpyDeviceToHost, stream_);
    if (ce != cudaSuccess) { err = std::string("selection readback: ") + cudaGetErrorString(ce); return false; }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) { err = "selection readback sync failed"; return false; }
    if (!stage_blob(il, sel_host_.data(), n * KU_, err)) return false;
    const bool tail = n <= kGemvMaxTok;
    if (!k::deepseek4_routed_moe(tail ? nullptr : y_, partial_, blob_dev_, stage_used_, E_, FF_, E_, sel_, rw_, KU_,
                                 clamp_, xn_, n, blob_[il].nc, blob_[il].v3 ? 1u : 0u, nullptr, stream_)) {
        err = "deepseek4_routed_moe failed";
        return false;
    }
    if (tail) {
        if (!k::deepseek4_moe_tail(y_, partial_, rw_, so_, n, KU_, E_, stream_)) {
            err = "deepseek4_moe_tail failed";
            return false;
        }
    } else {
        if (!k::deepseek4_add(y_, so_, uint64_t(n) * E_, stream_) ||
            !k::deepseek4_round_bf16(y_, uint64_t(n) * E_, stream_)) {
            err = "moe add/round failed";
            return false;
        }
    }
    return true;
}

bool Runner::layer(uint32_t il, std::string& err) {
    if (engram_of_[il] >= 0 && engram_on_ && !engram_layer(il, err)) return false;
    // DSpark: capture the target layers' attention input (the hc four-route mean, after the engram and before the
    // layer) into the main-hidden ring.  A wrong capture point does not error; acceptance collapses to about 1.
    if (mtp_on_ && mtp_slot_[il] >= 0) {
        const uint32_t rows = cur_n_ < mainh_cap_ ? cur_n_ : mainh_cap_;
        const uint32_t src0 = cur_n_ - rows;
        if (!k::deepseek4_hc_mean(mainh_, hc_, E_, HC_, rows, src0, static_cast<uint32_t>(mtp_slot_[il]),
                                  mtp_n_target_, cur_pos0_ + src0, mainh_cap_, stream_)) {
            err = "main-hidden capture failed";
            return false;
        }
        mainh_wrote_ = rows;
    }
    if (opt_.trace) probe(("L" + std::to_string(il) + " hc_in").c_str(), hc_, uint64_t(cur_n_) * HC_ * E_);
    if (!hc_half(il, true, err)) return false;
    if (!attn(il, err)) return false;
    if (!k::deepseek4_hc_post(attn_out_, hc_, post_, comb_, hc2_, E_, HC_, cur_n_, stream_)) {
        err = "hc_post (attn) failed";
        return false;
    }
    std::swap(hc_, hc2_);
    if (!hc_half(il, false, err)) return false;
    if (!moe(il, err)) return false;
    if (!k::deepseek4_hc_post(y_, hc_, post_, comb_, hc2_, E_, HC_, cur_n_, stream_)) {
        err = "hc_post (ffn) failed";
        return false;
    }
    std::swap(hc_, hc2_);
    if (opt_.trace) probe(("L" + std::to_string(il) + " hc_out").c_str(), hc_, uint64_t(cur_n_) * HC_ * E_);
    return true;
}

void Runner::reseed() {
    uint64_t seedsrc = opt_.seed;   // splitmix64 advances its argument
    rng_s0_ = splitmix64(seedsrc);
    rng_s1_ = splitmix64(seedsrc);
    if (rng_s0_ == 0 && rng_s1_ == 0) rng_s0_ = 1;   // xorshift128+ cannot start all-zero
}

int32_t Runner::pick_token(std::string& err) {
    const cudaError_t ce = cudaMemcpyAsync(logits_host_.data(), logits_, uint64_t(V_) * 4, cudaMemcpyDeviceToHost,
                                           stream_);
    if (ce != cudaSuccess) { err = std::string("logits readback: ") + cudaGetErrorString(ce); return -1; }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) { err = "logits readback sync failed"; return -1; }
    if (opt_.temperature <= 0.0f) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < V_; ++i)
            if (logits_host_[i] > logits_host_[best]) best = i;   // strict >: ties take the lower index
        return static_cast<int32_t>(best);
    }
    return sample_host(err);
}

// The server's sampling chain over the row pick_token read back: temperature, top_k, min_p (relative to the
// maximum probability), top_p, then one draw.  Ties sort by ascending id, matching the argmax's lower-id rule.
// Non-finite logits are excluded rather than poisoning the sort (a valid model writes none).
int32_t Runner::sample_host(std::string& err) {
    const float inv = 1.0f / opt_.temperature;
    samp_work_.resize(V_);
    bool any = false;
    for (uint32_t i = 0; i < V_; ++i) {
        const float s = logits_host_[i];
        const bool ok = std::isfinite(s);
        samp_work_[i] = {ok ? s * inv : -INFINITY, i};
        any = any || ok;
    }
    if (!any) { err = "sampling: no finite logit"; return -1; }
    auto& w = samp_work_;
    std::sort(w.begin(), w.end(), [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    uint32_t n = V_;
    if (opt_.top_k > 0 && opt_.top_k < n) n = opt_.top_k;
    const double m = w[0].first;
    double sum = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const double e = std::exp(static_cast<double>(w[i].first) - m);
        w[i].first = static_cast<float>(e);
        sum += e;
    }
    if (opt_.min_p > 0.0f) {   // p_i / p_max = exp(s_i - s_max), so the floor is min_p itself
        uint32_t k = 1;
        while (k < n && static_cast<double>(w[k].first) >= static_cast<double>(opt_.min_p)) ++k;
        n = k;
    }
    if (opt_.top_p < 1.0f) {
        double cum = 0;
        uint32_t k = 0;
        while (k < n && cum < static_cast<double>(opt_.top_p)) {
            cum += static_cast<double>(w[k].first) / sum;
            ++k;
        }
        n = k ? k : 1;
    }
    double total = 0;
    for (uint32_t i = 0; i < n; ++i) total += static_cast<double>(w[i].first);
    if (!(total > 0.0)) { err = "sampling: empty distribution"; return -1; }
    const double u = rng_uniform() * total;
    double cum = 0;
    for (uint32_t i = 0; i < n; ++i) {
        cum += static_cast<double>(w[i].first);
        if (u < cum) return static_cast<int32_t>(w[i].second);
    }
    return static_cast<int32_t>(w[n - 1].second);
}

bool Runner::block(const int32_t* ids, uint32_t n, bool head, bool head_all, std::string& err) {
    if (n == 0 || n > cap_) { err = "block size outside the chunk cap"; return false; }
    if (uint64_t(n_past_) + n > ctx_) {
        err = "context full (" + std::to_string(n_past_) + "+" + std::to_string(n) + " > " + std::to_string(ctx_) + ")";
        return false;
    }
    const uint32_t pos0 = n_past_;
    cur_n_ = n;
    cur_pos0_ = pos0;
    mainh_wrote_ = 0;

    for (uint32_t i = 0; i < n; ++i) dpos_host_[i] = static_cast<int32_t>(pos0 + i);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t c = 0; c < HC_; ++c)
            pm_host_[static_cast<size_t>(i) * HC_ + c] = (c == 0) ? 1.0f : 0.0f;
    if (hist_.size() < size_t(pos0) + n) hist_.resize(size_t(pos0) + n, 0);
    std::memcpy(hist_.data() + pos0, ids, size_t(n) * 4);

    bind_stage(0);
    cudaError_t ce = cudaMemcpyAsync(d_tok_, ids, uint64_t(n) * 4, cudaMemcpyHostToDevice, stream_);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(d_pos_, dpos_host_.data(), uint64_t(n) * 4, cudaMemcpyHostToDevice, stream_);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(d_prem_, pm_host_.data(), uint64_t(n) * HC_ * 4, cudaMemcpyHostToDevice, stream_);
    if (ce != cudaSuccess) { err = std::string("block inputs upload: ") + cudaGetErrorString(ce); return false; }

    if (!k::deepseek4_embed_q4k(x_, d_tok_, tok_embd_, V_, n, E_, stream_) ||
        !k::deepseek4_expand_hc(hc_, x_, E_, HC_, n, stream_)) {
        err = "embed/expand failed";
        return false;
    }
    idx_owner_ = -1;
    cand_owner_ = -1;
    idx_topk_ = 0;
    double t_stage = now_sec();
    for (uint32_t s = 0; s < nstages_; ++s) {
        if (s > 0) {
            // Stage hand-off (P7).  The hidden state and the evolved premix move to the next stage through the
            // host: a sync, two reads and two writes.  Stages run in order for now; a later change can overlap
            // stage s + 1's entry with stage s's tail.
            if (cudaStreamSynchronize(stream_) != cudaSuccess) { err = "stage hand-off sync failed"; return false; }
            const uint64_t hc_bytes = uint64_t(n) * HC_ * E_ * 4;
            const uint64_t prem_bytes = uint64_t(n) * HC_ * 4;
            cudaError_t he = cudaMemcpy(handoff_hc_.data(), hc_, hc_bytes, cudaMemcpyDeviceToHost);
            if (he == cudaSuccess)
                he = cudaMemcpy(handoff_prem_.data(), d_prem_, prem_bytes, cudaMemcpyDeviceToHost);
            if (he != cudaSuccess) {
                err = std::string("stage hand-off read: ") + cudaGetErrorString(he);
                return false;
            }
            save_stage();
            bind_stage(s);
            he = cudaMemcpy(hc_, handoff_hc_.data(), hc_bytes, cudaMemcpyHostToDevice);
            if (he == cudaSuccess)
                he = cudaMemcpy(d_prem_, handoff_prem_.data(), prem_bytes, cudaMemcpyHostToDevice);
            if (he == cudaSuccess)
                he = cudaMemcpy(d_pos_, dpos_host_.data(), uint64_t(n) * 4, cudaMemcpyHostToDevice);
            if (he != cudaSuccess) {
                err = std::string("stage hand-off write: ") + cudaGetErrorString(he);
                return false;
            }
            if (opt_.ledger) {
                st_hand_sec_[s] += now_sec() - t_stage;
                t_stage = now_sec();
            }
        }
        for (uint32_t il = stage_first_[s]; il < stage_first_[s + 1]; ++il)
            if (!layer(il, err)) {
                err = "L" + std::to_string(il) + ": " + err;
                return false;
            }
        if (opt_.ledger) {
            const double dt = now_sec() - t_stage;
            if (in_decode_) st_dec_sec_[s] += dt; else st_pre_sec_[s] += dt;
            t_stage = now_sec();
        }
        if (s + 1 < nstages_) save_stage();
    }
    if (head) {
        // head: hc_pre (fused with the output norm) -> output projection.  The generate path projects the last
        // row only; a spec verify batch projects every row into vlogits_ (the reference's full-GEMM arm).
        if (!k::deepseek4_hc_pre(hc_, d_prem_, static_cast<const float*>(out_norm_), x_, xn_, E_, HC_, rms_eps_, n,
                                 stream_)) {
            err = "head hc_pre failed";
            return false;
        }
        if (head_all) {
            if (!k::deepseek4_matmul_q4k(vlogits_, out_w_, xn_, E_, V_, n, false, stream_)) {
                err = "head projection (all rows) failed";
                return false;
            }
        } else {
            const float* last = xn_ + uint64_t(n - 1) * E_;
            if (!k::deepseek4_matmul_q4k(logits_, out_w_, last, E_, V_, 1, false, stream_)) {
                err = "head projection failed";
                return false;
            }
        }
        if (opt_.ledger) {
            const double dt = now_sec() - t_stage;
            if (in_decode_) st_dec_sec_[nstages_ - 1] += dt; else st_pre_sec_[nstages_ - 1] += dt;
        }
    }
    if (in_decode_) ++st_blocks_dec_; else ++st_blocks_pre_;
    // The last stage's record goes back too: its expert container and index scratch must survive into the next
    // block instead of being reallocated (and leaked) on every entry.
    save_stage();
    // The main-hidden ring's accounting (the reference's, core_v41_forward.c:391-400): continue the run when
    // this block touches the previous one, restart it otherwise.  Only the contiguous run ending at mainh_end_
    // is real; the draft window fill may read nothing else.  Must happen before n_past_ advances.
    if (mainh_ != nullptr && mainh_wrote_) {
        const int64_t first = static_cast<int64_t>(pos0) + static_cast<int64_t>(n - mainh_wrote_);
        const bool contig = mainh_n_ > 0 && mainh_end_ == first - 1;
        const uint32_t nn = contig ? mainh_n_ + mainh_wrote_ : mainh_wrote_;
        mainh_n_ = nn > mainh_cap_ ? mainh_cap_ : nn;
        mainh_end_ = static_cast<int64_t>(pos0) + static_cast<int64_t>(n) - 1;
    }
    n_past_ += n;
    return true;
}

// One allocation for a resident process, a fresh per-request state for every run.  The one-shot path allocates
// exactly as before; --serve sizes the state once (see warm) and refuses a request above --serve-ctx.
bool Runner::prepare_request(uint32_t total, std::string& err) {
    if (opt_.serve) {
        if (!state_ready_) {
            const uint32_t want = std::min<uint32_t>(ctx_, opt_.serve_ctx);
            if (total > want) {
                err = "a " + std::to_string(total) + "-token request does not fit the resident state (" +
                      std::to_string(want) + " tokens; --serve-ctx)";
                return false;
            }
            if (!alloc_state(want, err)) return false;
            state_ready_ = true;
            state_total_ = want;
        } else if (total > state_total_) {
            err = "a " + std::to_string(total) + "-token request does not fit the resident state (" +
                  std::to_string(state_total_) + " tokens; --serve-ctx)";
            return false;
        }
    } else if (!alloc_state(total, err)) {
        return false;
    }
    // The KV rows are addressed from position 0 again, and the engram history is this request's tokens only:
    // a longer previous prompt must not leak into the n-gram window.
    n_past_ = 0;
    in_decode_ = false;
    for (uint32_t i = 0; i < 64; ++i) {
        ng_src_[i] = 0;
        cpend_[i] = 0;
        snap_cpend_[i] = 0;
    }
    snap_n_ = 0;
    snap_past_ = 0;
    hist_.clear();
    for (auto& jobs : efeed_)   // normal completion joins every read; a failed request may not have
        for (auto& j : jobs)
            if (j.running) {
                j.th.join();
                j.running = false;
            }
    for (auto& c : efeed_cur_) c = -1;
    return true;
}

bool Runner::warm(std::string& err) {
    if (!opt_.serve) return true;
    if (!prepare_request(0, err)) return false;
    std::fprintf(stderr, "[v41] resident: state sized for %u token(s)\n", state_total_);
    return true;
}

bool Runner::set_request_sampling(float temperature, float top_p, float min_p, uint32_t top_k, uint64_t seed,
                                  bool stop_eos, std::string& err) {
    if (!std::isfinite(temperature) || temperature < 0.0f) { err = "temperature is negative or not finite"; return false; }
    if (!std::isfinite(top_p) || !(top_p > 0.0f && top_p <= 1.0f)) { err = "top-p is outside (0, 1]"; return false; }
    if (!std::isfinite(min_p) || !(min_p >= 0.0f && min_p < 1.0f)) { err = "min-p is outside [0, 1)"; return false; }
    opt_.temperature = temperature;
    opt_.top_p = top_p;
    opt_.min_p = min_p;
    opt_.top_k = std::min<uint32_t>(top_k, V_);
    opt_.seed = seed;
    opt_.stop_eos = stop_eos;
    reseed();
    return true;
}

bool Runner::run(const std::vector<int32_t>& prompt, uint32_t gen, std::string& err) {
    if (prompt.empty()) { err = "the prompt is empty"; return false; }
    const uint64_t total = uint64_t(prompt.size()) + gen;
    if (total > ctx_) {
        err = "prompt + generation (" + std::to_string(total) + ") exceeds the context (" + std::to_string(ctx_) + ")";
        return false;
    }
    if (!prepare_request(static_cast<uint32_t>(total), err)) return false;

    const uint32_t np = static_cast<uint32_t>(prompt.size());
    const double t0 = now_sec();
    for (uint32_t c0 = 0; c0 < np;) {
        uint32_t nc = std::min<uint32_t>(np - c0, cap_);
        const uint32_t rest = np - c0 - nc;
        if (rest > 0 && rest < SWA_ && nc > SWA_) nc -= SWA_ - rest;
        const bool last = c0 + nc >= np;
        // The feed: this block's rows are already in hand (prefetched during the previous one); start the next
        // block's reads now, so they run under this block's compute (the reference's prefetch_next(c1, ...)).
        if (!engram_feed_prepare(c0, prompt.data() + c0, nc, err)) return false;
        if (!last) {
            const uint32_t c1 = c0 + nc;
            uint32_t n1 = std::min<uint32_t>(np - c1, cap_);
            const uint32_t rest1 = np - c1 - n1;
            if (rest1 > 0 && rest1 < SWA_ && n1 > SWA_) n1 -= SWA_ - rest1;
            engram_feed_prefetch(c1, prompt.data(), prompt.data() + c1, n1);
        }
        if (!block(prompt.data() + c0, nc, last, false, err)) return false;
        c0 += nc;
        if (opt_.trace || np >= 64) std::fprintf(stderr, "[v41] prefill %u/%u (%.1fs)\r", c0, np, now_sec() - t0);
    }
    if (np >= 64 || opt_.trace) std::fprintf(stderr, "\n");
    const double tp = now_sec() - t0;
    std::fprintf(stderr, "[v41] prefill %u tokens in %.2fs (%.1f tok/s)\n", np, tp,
                 tp > 0 ? static_cast<double>(np) / tp : 0.0);

    if (!opt_.logits_out.empty()) {
        std::vector<float> buf(V_);
        cudaMemcpy(buf.data(), logits_, uint64_t(V_) * 4, cudaMemcpyDeviceToHost);
        std::FILE* f = std::fopen(opt_.logits_out.c_str(), "wb");
        if (f != nullptr) {
            std::fwrite(buf.data(), 4, V_, f);
            std::fclose(f);
            std::fprintf(stderr, "[v41] logits written to %s (%u floats)\n", opt_.logits_out.c_str(), V_);
        } else {
            std::fprintf(stderr, "[v41] cannot write %s\n", opt_.logits_out.c_str());
        }
    }

    std::vector<int32_t> generated;
    if (gen > 0) {
        in_decode_ = true;
        int32_t t = pick_token(err);
        if (t < 0) return false;
        const double td = now_sec();
        // The DSpark summary counters (report-only; the token stream is what the gate reads).
        uint64_t rounds = 0, accepted = 0, drafted = 0, plain_steps = 0;
        for (uint32_t g = 0; g < gen;) {
            // The loop-top emit (the reference's core_v41_api.c:246): this token's successor is already decided.
            generated.push_back(t);
            ++g;
            if (opt_.stream_ids) {
                std::printf("%d\n", t);
                std::fflush(stdout);
            }
            if (opt_.stop_eos && static_cast<uint32_t>(t) == eos_) break;
            if (g >= gen) break;
            if (n_past_ + 1 > ctx_) {
                err = "context full (" + std::to_string(n_past_) + "+1 > " + std::to_string(ctx_) + ")";
                return false;
            }
            if (!opt_.dspark) {
                // Plain decode: feed the token, pick its successor.
                if (!engram_feed_prepare(n_past_, &t, 1, err)) return false;
                if (!block(&t, 1, true, false, err)) return false;
                t = pick_token(err);
                if (t < 0) return false;
                continue;
            }
            // ---- one DSpark round (the reference's core_v41_api.c:272-390, direct arm) ----
            const uint32_t pos_round = n_past_;
            const int dr = draft_round(t, n_past_ - 1, err);
            if (dr < 0) return false;
            uint32_t k = 0;
            if (dr == 1) {
                k = mtp_block_;
                if (opt_.dspark_verify != 0 && opt_.dspark_verify < k) k = opt_.dspark_verify;
                // Only positions inside the real context can be verified.
                const uint32_t room_k = ctx_ - n_past_ - 1u;
                if (k > room_k) k = room_k;
            }
            if (k == 0) {
                // No draft, or a round the scheduler would have sat out: one direct token step.
                ++plain_steps;
                if (!engram_feed_prepare(n_past_, &t, 1, err)) return false;
                if (!block(&t, 1, true, false, err)) return false;
                t = pick_token(err);
                if (t < 0) return false;
                continue;
            }
            int32_t batch[9];   // DS4_MTP_MAX_BLOCK + 1
            batch[0] = t;
            for (uint32_t i = 0; i < k; ++i) batch[i + 1] = t_id_host_[i + 1];
            const uint32_t nb = 1 + k;
            // Snapshot -> verify forward -> per-row argmax.  The snapshot is only the ring cells this batch
            // overwrites; the compressor's pending rows are captured inside compress_source.
            if (!spec_snapshot(nb, err)) return false;
            if (!engram_feed_prepare(n_past_, batch, nb, err)) return false;
            if (!block(batch, nb, true, true, err)) return false;
            for (uint32_t i = 0; i < nb; ++i) {
                if (!k::deepseek4_argmax(t_want_ + i, vlogits_ + uint64_t(i) * V_, V_, stream_)) {
                    err = "verify argmax failed at row " + std::to_string(i);
                    return false;
                }
            }
            cudaError_t ce = cudaMemcpyAsync(t_want_host_.data(), t_want_, uint64_t(nb) * 4,
                                             cudaMemcpyDeviceToHost, stream_);
            if (ce != cudaSuccess) { err = std::string("verify readback: ") + cudaGetErrorString(ce); return false; }
            if (cudaStreamSynchronize(stream_) != cudaSuccess) { err = "verify readback sync failed"; return false; }
            uint32_t a = 0;
            while (a < k && t_want_host_[a] == batch[a + 1]) ++a;   // the longest matching prefix
            if (opt_.dspark_trace) {
                std::fprintf(stderr, "[dspark] pos %u k %u acc %u conf", pos_round, k, a);
                for (uint32_t i = 0; i < mtp_block_; ++i) {
                    const float c = t_conf_host_[i];
                    std::fprintf(stderr, " %.3f", std::isfinite(c) ? 1.0 / (1.0 + std::exp(-static_cast<double>(c)))
                                                                   : 0.0);
                }
                std::fprintf(stderr, " | draft");
                for (uint32_t i = 0; i < k; ++i) std::fprintf(stderr, " %d", batch[i + 1]);
                std::fprintf(stderr, " | main");
                for (uint32_t i = 0; i < nb; ++i) std::fprintf(stderr, " %d", t_want_host_[i]);
                std::fprintf(stderr, "\n");
            }
            ++rounds;
            drafted += k;
            accepted += a;
            // Roll back to the accepted prefix: keep = the confirmed head + the accepted draft rows.
            if (!spec_rollback(1u + a, err)) return false;
            // The earned tokens, then the next round's head token.
            for (uint32_t i = 0; i < a && g < gen; ++i) {
                generated.push_back(batch[i + 1]);
                ++g;
                if (opt_.stream_ids) {
                    std::printf("%d\n", batch[i + 1]);
                    std::fflush(stdout);
                }
            }
            if (opt_.stop_eos && static_cast<uint32_t>(generated.back()) == eos_) break;
            t = t_want_host_[a];
        }
        const double dt = now_sec() - td;
        const uint64_t n_emitted = generated.size();
        std::fprintf(stderr, "[v41] generated %llu tokens in %.2fs (%.1f tok/s)\n",
                     static_cast<unsigned long long>(n_emitted), dt,
                     dt > 0 ? static_cast<double>(n_emitted) / dt : 0.0);
        if (rounds || plain_steps) {
            std::fprintf(stderr, "[v41] DSpark: %llu round(s), %.2f accepted of %llu drafted (%.1f%%), %llu plain step(s)%s\n",
                         static_cast<unsigned long long>(rounds),
                         rounds ? static_cast<double>(accepted) / static_cast<double>(rounds) : 0.0,
                         static_cast<unsigned long long>(drafted),
                         drafted ? 100.0 * static_cast<double>(accepted) / static_cast<double>(drafted) : 0.0,
                         static_cast<unsigned long long>(plain_steps),
                         opt_.dspark ? "" : " (dspark off)");
        }
    }
    if (opt_.ledger) {
        std::fprintf(stderr, "[v41] ledger: prefill %llu block(s), decode %llu block(s)\n",
                     static_cast<unsigned long long>(st_blocks_pre_),
                     static_cast<unsigned long long>(st_blocks_dec_));
        for (uint32_t s = 0; s < nstages_; ++s)
            std::fprintf(stderr,
                         "[v41]   stage %u (gpu %d) L%u..%u: prefill %.2fs, decode %.2fs, hand-off %.2fs, "
                         "staged %.1f MiB in %llu expert slot(s), %.0f%% of them repeats\n",
                         s, st_[s].dev, stage_first_[s], stage_first_[s + 1] - 1, st_pre_sec_[s], st_dec_sec_[s],
                         st_hand_sec_[s], static_cast<double>(st_bytes_[s]) / 1048576.0,
                         static_cast<unsigned long long>(st_slots_[s]),
                         st_slots_[s] ? 100.0 * static_cast<double>(st_slots_[s] - st_new_[s]) /
                                            static_cast<double>(st_slots_[s])
                                      : 0.0);
        if (!opt_.expert_cache_mib.empty()) {
            for (uint32_t s = 0; s < nstages_; ++s) {
                const uint64_t lookups = st_cache_hit_[s] + st_cache_miss_[s];
                std::fprintf(stderr,
                             "[v41]   stage %u cache: %llu hit, %llu miss (%.0f%%), %.1f MiB not read, "
                             "%llu slot(s) of %llu\n",
                             s, static_cast<unsigned long long>(st_cache_hit_[s]),
                             static_cast<unsigned long long>(st_cache_miss_[s]),
                             lookups ? 100.0 * static_cast<double>(st_cache_hit_[s]) /
                                           static_cast<double>(lookups)
                                     : 0.0,
                             static_cast<double>(st_cache_bytes_[s]) / 1048576.0,
                             static_cast<unsigned long long>(exp_cache_next_[s]),
                             static_cast<unsigned long long>(exp_cache_slots_[s]));
            }
        }
    }
    std::printf("generated=");
    for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : "", generated[i]);
    std::printf("\n");
    if (opt_.serve) std::printf("done\n");   // the resident protocol's per-request end marker
    std::fflush(stdout);
    return true;
}

std::vector<int32_t> parse_ids(const std::string& text) {
    std::vector<int32_t> ids;
    int64_t cur = -1;
    for (char ch : text) {
        if (ch >= '0' && ch <= '9') {
            cur = (cur < 0 ? 0 : cur) * 10 + (ch - '0');
        } else if (cur >= 0) {
            ids.push_back(static_cast<int32_t>(cur));
            cur = -1;
        }
    }
    if (cur >= 0) ids.push_back(static_cast<int32_t>(cur));
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "dsv4_run: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") o.model = next("--model");
        else if (a == "--engram-dir") o.engram_dir = next("--engram-dir");
        else if (a == "--ids") o.ids = parse_ids(next("--ids"));
        else if (a == "--ids-file") o.ids_file = next("--ids-file");
        else if (a == "--gen") o.gen = static_cast<uint32_t>(std::strtoul(next("--gen"), nullptr, 10));
        else if (a == "--chunk") o.chunk = static_cast<uint32_t>(std::strtoul(next("--chunk"), nullptr, 10));
        else if (a == "--ctx") o.ctx = static_cast<uint32_t>(std::strtoul(next("--ctx"), nullptr, 10));
        else if (a == "--gpu") o.gpu = std::atoi(next("--gpu"));
        else if (a == "--devices") {
            const std::string v = next("--devices");
            o.devices.clear();
            size_t p = 0;
            while (p < v.size()) {
                while (p < v.size() && (v[p] == ',' || v[p] == ' ')) ++p;
                if (p >= v.size()) break;
                char* end = nullptr;
                const long d = std::strtol(v.c_str() + p, &end, 10);
                if (end == v.c_str() + p) {
                    std::fprintf(stderr, "dsv4_run: --devices: '%s' is not a gpu index list\n", v.c_str());
                    return 2;
                }
                o.devices.push_back(static_cast<int>(d));
                p = static_cast<size_t>(end - v.c_str());
            }
            if (o.devices.empty()) {
                std::fprintf(stderr, "dsv4_run: --devices needs at least one gpu index\n");
                return 2;
            }
        }
        else if (a == "--logits-out") o.logits_out = next("--logits-out");
        else if (a == "--trace") o.trace = true;
        else if (a == "--dspark") o.dspark = true;
        else if (a == "--dspark-verify") {
            o.dspark_verify = static_cast<uint32_t>(std::strtoul(next("--dspark-verify"), nullptr, 10));
            o.dspark = true;
        } else if (a == "--dspark-trace") {
            o.dspark_trace = true;
            o.dspark = true;
        }
        else if (a == "--temperature") o.temperature = std::strtof(next("--temperature"), nullptr);
        else if (a == "--top-p") o.top_p = std::strtof(next("--top-p"), nullptr);
        else if (a == "--min-p") o.min_p = std::strtof(next("--min-p"), nullptr);
        else if (a == "--top-k") o.top_k = static_cast<uint32_t>(std::strtoul(next("--top-k"), nullptr, 10));
        else if (a == "--seed") o.seed = std::strtoull(next("--seed"), nullptr, 10);
        else if (a == "--stop-eos") o.stop_eos = true;
        else if (a == "--stream-ids") o.stream_ids = true;
        else if (a == "--ledger") o.ledger = true;
        else if (a == "--expert-cache-mib") {
            const std::string v = next("--expert-cache-mib");
            o.expert_cache_mib.clear();
            size_t p = 0;
            while (p < v.size()) {
                while (p < v.size() && (v[p] == ',' || v[p] == ' ')) ++p;
                if (p >= v.size()) break;
                char* end = nullptr;
                const unsigned long m = std::strtoul(v.c_str() + p, &end, 10);
                if (end == v.c_str() + p) {
                    std::fprintf(stderr, "dsv4_run: --expert-cache-mib: '%s' is not a MiB list\n", v.c_str());
                    return 2;
                }
                o.expert_cache_mib.push_back(static_cast<uint32_t>(m));
                p = static_cast<size_t>(end - v.c_str());
            }
            if (o.expert_cache_mib.empty()) {
                std::fprintf(stderr, "dsv4_run: --expert-cache-mib needs at least one budget\n");
                return 2;
            }
        }
        else if (a == "--no-engram") o.engram = false;
        else if (a == "--engram-feed") o.engram_feed = true;
        else if (a == "--ram-image") o.ram_image = true;
        else if (a == "--serve") o.serve = true;
        else if (a == "--serve-ctx") o.serve_ctx = static_cast<uint32_t>(std::strtoul(next("--serve-ctx"), nullptr, 10));
        else if (a == "--no-check") o.check = false;
        else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                         "usage: dsv4_run --ids 1,2,3 [--gen N] [--model BASE] [--engram-dir DIR] [--chunk 8]\n"
                         "               [--ctx N] [--gpu 0] [--devices 0,1,2,3] [--trace] [--no-engram] [--no-check]\n"
                         "               [--logits-out F]\n"
                         "               [--dspark] [--dspark-verify K] [--dspark-trace]\n"
                         "               [--temperature T] [--top-p P] [--min-p M] [--top-k K] [--seed S]\n"
                         "               [--stop-eos] [--stream-ids] [--ledger] [--engram-feed]\n"
                         "               [--expert-cache-mib N[,N...]] [--ram-image]\n"
                         "               [--serve --serve-ctx N]\n"
                         "  --devices        split the layers over these gpus, one stage each, layers in order;\n"
                         "                   the split only cuts at a compressed-cache seam and must return the\n"
                         "                   same ids as the one-gpu run (--dspark is one device only)\n"
                         "  --chunk 0        auto: one block for the whole run when it fits.  The driver picks the\n"
                         "                   largest block that leaves every stage's state, windows, engram rows and\n"
                         "                   expert container inside 90%% of its free memory after the weights (at\n"
                         "                   most 4096 rows, the MoE grid's bound); the default stays 8\n"
                         "  --ledger         report prefill/decode seconds, the hand-off and staged bytes per stage\n"
                         "  --expert-cache-mib\n"
                         "                   keep staged expert payloads in N MiB of device slots per stage (a comma\n"
                         "                   list gives one budget per stage); a hit copies from its slot instead of\n"
                         "                   reading the parts again.  Off by default.\n"
                         "  --ram-image      read the whole image into host RAM once at start (pinned when the\n"
                         "                   driver allows it, so the copy is a direct DMA) and stage expert misses\n"
                         "                   from there instead of the disk.  Off by default.\n"
                         "  --dspark         the MTP towers draft, the main forward verifies, the cache rolls back;\n"
                         "                   the greedy output must be byte-identical to the same run without it\n"
                         "  --dspark-verify  pin the verified rows per round (1..block, default: the whole block)\n"
                         "  --dspark-trace   print each round's k, accepted count, confidence and both id lists\n"
                         "  --temperature    <= 0 (the default) is greedy; above 0 runs the sampling chain\n"
                         "  --top-p/--min-p/--top-k  the chain's filters (default 1.0, 0.0, 0 = off)\n"
                         "  --seed           the sampler's seed (default 0; the server picks one per request)\n"
                         "  --stop-eos       stop after emitting the model's EOS id\n"
                         "  --stream-ids     print each generated id as a line as it is produced\n"
                         "  --engram-feed    read the next block's engram rows while the current block computes\n"
                         "                   (the reference's prepare/wait/prefetch_next); off by default, and\n"
                         "                   the ids must not move\n"
                         "  --serve          resident: load the weights once, size the state for --serve-ctx, print\n"
                         "                   ready, then answer one request per stdin line: ids=1,2,3 gen=16\n"
                         "                   temperature=1 top_p=1 min_p=0.05 top_k=0 seed=1 stop_eos=1.  Each\n"
                         "                   answer streams ids, prints generated=..., then done.  Off by default;\n"
                         "                   a failed forward exits 1 so the supervisor starts a clean process\n"
                         "  --serve-ctx      the resident state's token budget (default 16384)\n");
            return 0;
        } else {
            std::fprintf(stderr, "dsv4_run: unknown argument '%s'\n", a.c_str());
            return 2;
        }
    }
    if (!o.ids_file.empty()) {
        std::FILE* f = std::fopen(o.ids_file.c_str(), "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "dsv4_run: cannot open %s\n", o.ids_file.c_str());
            return 2;
        }
        std::string text;
        char buf[4096];
        size_t got;
        while ((got = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, got);
        std::fclose(f);
        o.ids = parse_ids(text);
    }
    if (o.ids.empty() && !o.serve) {
        std::fprintf(stderr, "dsv4_run: a prompt is required (--ids or --ids-file)\n");
        return 2;
    }
    if (o.serve && o.serve_ctx == 0) {
        std::fprintf(stderr, "dsv4_run: --serve-ctx must be at least 1\n");
        return 2;
    }

    Runner r;
    std::string err;
    if (!r.open(o, err)) {
        std::fprintf(stderr, "dsv4_run: %s\n", err.c_str());
        return 2;
    }
    if (!r.load_weights(err)) {
        std::fprintf(stderr, "dsv4_run: %s\n", err.c_str());
        return 2;
    }
    if (!r.load_engram(err)) {
        std::fprintf(stderr, "dsv4_run: %s\n", err.c_str());
        return 2;
    }
    if (o.serve) {
        if (o.dspark) {
            std::fprintf(stderr, "dsv4_run: --serve and --dspark cannot be combined\n");
            return 2;
        }
        if (!r.warm(err)) {
            std::fprintf(stderr, "dsv4_run: %s\n", err.c_str());
            return 2;
        }
        std::printf("ready\n");
        std::fflush(stdout);
        auto digits = [](const std::string& s) {
            if (s.empty()) return false;
            for (char c : s)
                if (c < '0' || c > '9') return false;
            return true;
        };
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::vector<int32_t> ids;
            bool have_ids = false, bad = false, stop_eos = true;
            uint32_t gen = o.gen, top_k = o.top_k;
            float temperature = o.temperature, top_p = o.top_p, min_p = o.min_p;
            uint64_t seed = o.seed;
            std::string perr;
            size_t p = 0;
            while (p < line.size() && !bad) {
                while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
                if (p >= line.size()) break;
                size_t q = p;
                while (q < line.size() && line[q] != ' ' && line[q] != '\t') ++q;
                const std::string kv = line.substr(p, q - p);
                p = q;
                const size_t eq = kv.find('=');
                if (eq == std::string::npos) {
                    perr = "malformed request field '" + kv + "'";
                    bad = true;
                    break;
                }
                const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
                char* end = nullptr;
                if (k == "ids") {
                    ids = parse_ids(v);
                    have_ids = true;
                } else if (k == "gen" && digits(v)) {
                    gen = static_cast<uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
                } else if (k == "top_k" && digits(v)) {
                    top_k = static_cast<uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
                } else if (k == "seed" && digits(v)) {
                    seed = std::strtoull(v.c_str(), nullptr, 10);
                } else if (k == "stop_eos" && digits(v)) {
                    stop_eos = v != "0";
                } else if (k == "temperature" && !v.empty()) {
                    temperature = std::strtof(v.c_str(), &end);
                    if (end == v.c_str() || *end != '\0') { perr = "temperature is not a number"; bad = true; }
                } else if (k == "top_p" && !v.empty()) {
                    top_p = std::strtof(v.c_str(), &end);
                    if (end == v.c_str() || *end != '\0') { perr = "top_p is not a number"; bad = true; }
                } else if (k == "min_p" && !v.empty()) {
                    min_p = std::strtof(v.c_str(), &end);
                    if (end == v.c_str() || *end != '\0') { perr = "min_p is not a number"; bad = true; }
                } else {
                    perr = "unknown or malformed request field '" + k + "'";
                    bad = true;
                }
            }
            if (!bad && (!have_ids || ids.empty())) {
                perr = "the request has no ids";
                bad = true;
            }
            if (!bad && !r.set_request_sampling(temperature, top_p, min_p, top_k, seed, stop_eos, perr)) bad = true;
            if (!bad) {
                const uint64_t req_total = uint64_t(ids.size()) + gen;
                if (req_total > r.context_tokens()) {
                    perr = "prompt + generation (" + std::to_string(req_total) + ") exceeds the context (" +
                           std::to_string(r.context_tokens()) + ")";
                    bad = true;
                }
            }
            if (bad) {
                // A rejected request changes no model state: say why and keep serving.
                std::printf("error=%s\ndone\n", perr.c_str());
                std::fflush(stdout);
                continue;
            }
            err.clear();
            if (!r.run(ids, gen, err)) {
                // A forward that failed may have left CUDA state unsafe to reuse; the supervisor restarts.
                std::printf("error=%s\ndone\n", err.c_str());
                std::fflush(stdout);
                return 1;
            }
        }
        return 0;
    }
    if (!r.run(o.ids, o.gen, err)) {
        std::fprintf(stderr, "dsv4_run: %s\n", err.c_str());
        return 1;
    }
    return 0;
}
