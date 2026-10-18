// tools/dsv4/dsv4_run.cu - DeepSeek-V4.1-Flash forward driver (docs/DEEPSEEK4.md, P3 onward).
//
// One process, one GPU, greedy decode.  It reads the 40-part GGUF through a parts-aware reader (the header comes
// from part 1's map; every payload read is an offset into the assembled image), keeps the q4_K/BF16/F32 skeleton
// and the two engram wkv matrices resident, and streams each layer's selected routed experts from its DQVL blob
// into a compact device container per block (P7's cache/ring machinery is not built here; this is the
// correctness driver).
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
//
// The runner takes token ids, not text: the tokenizer is Python (tools/strata_tokenizer.py); see
// tools/dsv4/dsv4_prompt.py for the text wrapper.

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "strata/artifact/deepseek4.hpp"
#include "strata/artifact/deepseek4_engram.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/deepseek4_attn.hpp"
#include "strata/kernels/deepseek4_dense.hpp"
#include "strata/kernels/deepseek4_engram.hpp"
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
};

// ---- The expert blob ------------------------------------------------------------------------------------------

struct Blob {
    uint64_t abs = 0;          // payload offset in the assembled image
    uint64_t size = 0;         // the container's payload length
    uint32_t experts = 0;
    uint32_t nc = 0;
    uint64_t cb_off = 0;       // shared codebook offset (== 16 + experts*24)
    uint64_t cb_len = 0;
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

struct Options {
    std::string model = "/mnt/nvme/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf";
    std::string engram_dir = "/mnt/nvme";
    std::vector<int32_t> ids;
    std::string ids_file;
    uint32_t gen = 0;
    uint32_t chunk = 8;
    uint32_t ctx = 0;   // 0: the model's own context (no reduction)
    int gpu = 0;
    bool trace = false;
    bool engram = true;
    bool check = true;   // the inventory check (header only)
    std::string logits_out;
};

// ---- The runner -----------------------------------------------------------------------------------------------

class Runner {
public:
    bool open(const Options& o, std::string& err);
    bool load_weights(std::string& err);
    bool load_engram(std::string& err);
    bool run(const std::vector<int32_t>& prompt, uint32_t gen, std::string& err);

    d4::Geometry geo;
    std::string last_error;

private:
    // device memory
    void* dev(uint64_t bytes, const char* what);
    void* upload(const void* host, uint64_t bytes, const char* what);
    void* load_tensor(const std::string& name, uint32_t type, const char* what);

    bool alloc_state(uint32_t total, std::string& err);
    bool ensure_index_scratch(uint32_t rows, uint32_t ng, std::string& err);
    bool load_blob(uint32_t il, std::string& err);
    bool stage_blob(uint32_t il, const int32_t* sel, uint32_t n_pairs, std::string& err);

    bool block(const int32_t* ids, uint32_t n, bool head, std::string& err);
    bool layer(uint32_t il, std::string& err);
    bool hc_half(uint32_t il, bool attn_half, std::string& err);
    bool attn(uint32_t il, std::string& err);
    bool compress_source(uint32_t il, uint32_t ratio, std::string& err);
    bool index_source(uint32_t il, uint32_t ratio, std::string& err);
    bool moe(uint32_t il, std::string& err);
    bool engram_layer(uint32_t il, std::string& err);
    int32_t argmax_logits(std::string& err);
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

    std::vector<int32_t> hist_;
    std::vector<int32_t> pm_host_;
    std::vector<float> probe_host_;
    std::vector<int64_t> erow_ids_;
    std::vector<uint8_t> eraw_host_;
    std::vector<int32_t> pos_host_;
    std::vector<int32_t> sel_host_;
    std::vector<uint32_t> uniq_;
    std::vector<float> logits_host_;

    cudaStream_t stream_ = nullptr;
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

bool Runner::open(const Options& o, std::string& err) {
    opt_ = o;
    if (!parts_.open(o.model, err)) return false;
    std::fprintf(stderr, "[v41] image %s: %" PRIu64 " bytes in %zu part(s)\n", parts_.gguf_path().c_str(),
                 parts_.size(), parts_.count());
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
    if (cap_ > ctx_) cap_ = ctx_;

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
    std::fprintf(stderr, "), candidate L%u x %u blocks of %u,\n      ctx %u (model %" PRIu64
                         "), chunk %u, engram %s\n",
                 cand_src_layer_, kcap_, cand_bs_, ctx_, geo.context_length, cap_, opt_.engram ? "on" : "off");

    const cudaError_t ce = cudaSetDevice(opt_.gpu);
    if (ce != cudaSuccess) { err = std::string("cudaSetDevice: ") + cudaGetErrorString(ce); return false; }
    if (cudaStreamCreate(&stream_) != cudaSuccess) { err = "cudaStreamCreate failed"; return false; }
    return true;
}

bool Runner::load_weights(std::string& err) {
    for (uint32_t il = 0; il < nlayers_; ++il) {
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
    tok_embd_ = load_tensor("token_embd.weight", kQ4K, "token_embd");
    out_w_ = load_tensor("output.weight", kQ4K, "output");
    out_norm_ = load_tensor("output_norm.weight", kF32, "output_norm");
    if (!tok_embd_ || !out_w_ || !out_norm_) { err = last_error; return false; }

    // The expert containers: header, slot table and one payload header per layer (the payload bytes stream later).
    for (uint32_t il = 0; il < nlayers_; ++il)
        if (!load_blob(il, err)) return false;

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "[v41] weights resident; device %.1f GiB used, %.1f GiB free\n",
                 static_cast<double>(total_b - free_b) / 1073741824.0, static_cast<double>(free_b) / 1073741824.0);
    return true;
}

bool Runner::load_blob(uint32_t il, std::string& err) {
    const TensorInfo* t = gguf_->find("blk." + std::to_string(il) + ".ffn_exps_vq.blob");
    if (t == nullptr) { err = "missing expert blob at layer " + std::to_string(il); return false; }
    Blob& b = blob_[il];
    b.abs = gguf_->data_start() + t->offset;
    b.size = d4::payload_bytes(*t);
    if (b.size == 0) { err = "layer " + std::to_string(il) + ": the expert blob has no payload size"; return false; }
    uint8_t hdr[16];
    if (!parts_.read(b.abs, hdr, sizeof hdr, err)) return false;
    if (rd32(hdr) != d4::kBlobMagic || rd32(hdr + 4) != 3) {
        err = "layer " + std::to_string(il) + ": expert blob header unreadable in " + parts_.where(b.abs) +
              " (a hole in the download reads as zeros)";
        return false;
    }
    b.experts = rd32(hdr + 12);
    if (b.experts != NE_) { err = "layer " + std::to_string(il) + ": blob carries a different expert count"; return false; }

    b.slot.resize(static_cast<size_t>(b.experts) * 3);
    std::vector<uint8_t> table(static_cast<size_t>(b.experts) * 24);
    if (!parts_.read(b.abs + 16, table.data(), table.size(), err)) return false;
    for (size_t i = 0; i < b.slot.size(); ++i) b.slot[i] = rd64(table.data() + i * 8);

    // Payload metadata from the first present slot (the container shares one codebook).
    uint64_t first = 0;
    for (size_t i = 0; i < b.slot.size(); ++i)
        if (b.slot[i] != 0) { first = b.slot[i]; break; }
    if (first == 0) { err = "layer " + std::to_string(il) + ": no expert payloads"; return false; }
    uint8_t ph[32];
    if (!parts_.read(b.abs + first, ph, sizeof ph, err)) return false;
    if (rd32(ph) != d4::kMatV3Magic) {
        err = "layer " + std::to_string(il) + ": first expert payload header unreadable in " + parts_.where(b.abs + first);
        return false;
    }
    b.nc = static_cast<uint32_t>(ph[6]) | (static_cast<uint32_t>(ph[7]) << 8);
    b.cb_off = rd64(ph + 24);
    b.cb_len = static_cast<uint64_t>(b.nc) * 8;
    const uint64_t expect_cb = 16 + static_cast<uint64_t>(b.experts) * 24;
    if (b.cb_off != expect_cb) {
        err = "layer " + std::to_string(il) + ": codebook offset is not the container's";
        return false;
    }
    if (b.cb_off + b.cb_len > b.size) {
        err = "layer " + std::to_string(il) + ": codebook outside the blob";
        return false;
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
        if (end <= s || end > b.size) { err = "layer " + std::to_string(il) + ": slot table is not packed"; return false; }
        b.span[i] = end - s;
    }
    return true;
}

bool Runner::stage_blob(uint32_t il, const int32_t* sel, uint32_t n_pairs, std::string& err) {
    const Blob& b = blob_[il];
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

    const uint64_t table_end = 16 + static_cast<uint64_t>(b.experts) * 24;
    uint64_t need = table_end + b.cb_len;
    for (uint32_t e : uniq_)
        for (uint32_t w = 0; w < 3; ++w) need += b.span[static_cast<size_t>(e) * 3 + w];
    if (stage_.size() < need) stage_.resize(static_cast<size_t>(need));
    uint8_t* p = stage_.data();
    std::memset(p, 0, static_cast<size_t>(table_end));
    wr32(p, d4::kBlobMagic);
    wr32(p + 4, 3);
    wr32(p + 8, il);
    wr32(p + 12, b.experts);
    if (!parts_.read(b.abs + b.cb_off, p + table_end, b.cb_len, err)) return false;

    uint64_t cur = table_end + b.cb_len;
    uint32_t holes = 0;
    for (uint32_t e : uniq_) {
        for (uint32_t w = 0; w < 3; ++w) {
            const size_t i = static_cast<size_t>(e) * 3 + w;
            const uint64_t len = b.span[i];
            if (len == 0) {
                err = "expert " + std::to_string(e) + " slot " + std::to_string(w) + " is absent";
                return false;
            }
            wr64(p + 16 + i * 8, cur);   // slot table entry i is at 16 + i*8
            if (!parts_.read(b.abs + b.slot[i], p + cur, len, err)) return false;
            const uint32_t magic = rd32(p + cur);
            if (magic != d4::kMatV3Magic && magic != d4::kMatV2Magic) ++holes;
            cur += len;
        }
    }
    if (holes != 0)
        std::fprintf(stderr, "[v41] L%02u: %u of %u selected expert payloads are unreadable (a holey part); "
                             "the kernel skips them\n", il, holes, static_cast<uint32_t>(uniq_.size()) * 3u);
    if (cur > blob_dcap_) {
        if (blob_dev_ != nullptr) cudaFree(blob_dev_);
        blob_dev_ = dev(cur, "expert container");
        if (blob_dev_ == nullptr) { err = last_error; return false; }
        blob_dcap_ = cur;
    }
    const cudaError_t ce = cudaMemcpyAsync(blob_dev_, p, cur, cudaMemcpyHostToDevice, stream_);
    if (ce != cudaSuccess) { err = std::string("expert container upload: ") + cudaGetErrorString(ce); return false; }
    stage_used_ = cur;
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
    const uint64_t stride = eng_hd_ + eng_hd_ / 32;
    eraw_ = static_cast<uint8_t*>(dev(static_cast<uint64_t>(cap_) * eng_cols_ * stride, "eraw"));
    erows_ = static_cast<float*>(dev(static_cast<uint64_t>(cap_) * eng_in_dim_ * 4, "erows"));
    x16_ = static_cast<uint16_t*>(dev(static_cast<uint64_t>(cap_) * eng_in_dim_ * 2, "engram x bf16"));
    ekv_ = static_cast<float*>(dev(static_cast<uint64_t>(cap_) * eng_out_dim_ * 4, "engram kv"));
    if (eraw_ == nullptr || erows_ == nullptr || x16_ == nullptr || ekv_ == nullptr) { err = last_error; return false; }
    eraw_host_.resize(static_cast<size_t>(cap_) * eng_cols_ * stride);
    erow_ids_.resize(static_cast<size_t>(cap_) * eng_cols_);
    engram_on_ = true;
    std::fprintf(stderr, "[v41] engram: %zu table(s), cols %u, head dim %u, in %u out %u\n", layers, eng_cols_,
                 eng_hd_, eng_in_dim_, eng_out_dim_);
    return true;
}

// ---- State buffers --------------------------------------------------------------------------------------------

bool Runner::alloc_state(uint32_t total, std::string& err) {
    const uint64_t ngg = static_cast<uint64_t>(total) + 2;   // groups cap, ratio 1
    topk_cap_ = static_cast<uint32_t>(std::min<uint64_t>(geo.indexer_top_k, ngg));
    if (topk_cap_ == 0) { err = "indexer_top_k is zero"; return false; }

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

    win_.assign(nlayers_, nullptr);
    comp_kv_.assign(nlayers_, nullptr);
    index_k_.assign(nlayers_, nullptr);
    cpre_kv_.assign(nlayers_, nullptr);
    cpre_sc_.assign(nlayers_, nullptr);
    for (uint32_t il = 0; il < nlayers_; ++il) {
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
    pm_host_.resize(static_cast<size_t>(cap_) * HC_);
    pos_host_.resize(cap_);
    sel_host_.resize(static_cast<size_t>(cap_) * KU_);
    logits_host_.resize(V_);
    std::fprintf(stderr, "[v41] state: chunk %u, groups cap %" PRIu64 ", topk cap %u, window %u\n", cap_, ngg,
                 topk_cap_, SWA_);
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
    for (uint32_t i = 0; i < n; ++i) {
        hash_.rows_masked(hist_.data(), hist_.size(), nullptr, 0, static_cast<int64_t>(cur_pos0_ + i), ei,
                          erow_ids_.data() + static_cast<size_t>(i) * cols);
    }
    const d4::EngramTable& t = geo.engram_tables[static_cast<size_t>(ei)];
    if (!shard_[ei]->rows(t.weight_offset, t.scale_offset, erow_ids_.data(), static_cast<size_t>(n) * cols,
                          eng_hd_, eraw_host_.data(), err))
        return false;
    const uint64_t raw_bytes = uint64_t(n) * cols * stride;
    cudaError_t ce = cudaMemcpyAsync(eraw_, eraw_host_.data(), static_cast<size_t>(raw_bytes), cudaMemcpyHostToDevice,
                                     stream_);
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
                                 clamp_, xn_, n, blob_[il].nc, 1, nullptr, stream_)) {
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
    if (opt_.trace) probe(("L" + std::to_string(il) + " hc_in").c_str(), hc_, uint64_t(cur_n_) * HC_ * E_);
    if (!hc_half(il, true, err)) return false;
    if (!attn(il, err)) return false;
    if (!k::deepseek4_hc_post(hc2_, attn_out_, hc_, post_, comb_, E_, HC_, cur_n_, stream_)) {
        err = "hc_post (attn) failed";
        return false;
    }
    std::swap(hc_, hc2_);
    if (!hc_half(il, false, err)) return false;
    if (!moe(il, err)) return false;
    if (!k::deepseek4_hc_post(hc2_, y_, hc_, post_, comb_, E_, HC_, cur_n_, stream_)) {
        err = "hc_post (ffn) failed";
        return false;
    }
    std::swap(hc_, hc2_);
    if (opt_.trace) probe(("L" + std::to_string(il) + " hc_out").c_str(), hc_, uint64_t(cur_n_) * HC_ * E_);
    return true;
}

int32_t Runner::argmax_logits(std::string& err) {
    const cudaError_t ce = cudaMemcpyAsync(logits_host_.data(), logits_, uint64_t(V_) * 4, cudaMemcpyDeviceToHost,
                                           stream_);
    if (ce != cudaSuccess) { err = std::string("logits readback: ") + cudaGetErrorString(ce); return -1; }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) { err = "logits readback sync failed"; return -1; }
    uint32_t best = 0;
    for (uint32_t i = 1; i < V_; ++i)
        if (logits_host_[i] > logits_host_[best]) best = i;   // strict >: ties take the lower index
    return static_cast<int32_t>(best);
}

bool Runner::block(const int32_t* ids, uint32_t n, bool head, std::string& err) {
    if (n == 0 || n > cap_) { err = "block size outside the chunk cap"; return false; }
    if (uint64_t(n_past_) + n > ctx_) {
        err = "context full (" + std::to_string(n_past_) + "+" + std::to_string(n) + " > " + std::to_string(ctx_) + ")";
        return false;
    }
    const uint32_t pos0 = n_past_;
    cur_n_ = n;
    cur_pos0_ = pos0;

    for (uint32_t i = 0; i < n; ++i) pos_host_[i] = static_cast<int32_t>(pos0 + i);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t c = 0; c < HC_; ++c)
            pm_host_[static_cast<size_t>(i) * HC_ + c] = (c == 0) ? 1.0f : 0.0f;
    if (hist_.size() < size_t(pos0) + n) hist_.resize(size_t(pos0) + n, 0);
    std::memcpy(hist_.data() + pos0, ids, size_t(n) * 4);

    cudaError_t ce = cudaMemcpyAsync(d_tok_, ids, uint64_t(n) * 4, cudaMemcpyHostToDevice, stream_);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(d_pos_, pos_host_.data(), uint64_t(n) * 4, cudaMemcpyHostToDevice, stream_);
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
    for (uint32_t il = 0; il < nlayers_; ++il)
        if (!layer(il, err)) {
            err = "L" + std::to_string(il) + ": " + err;
            return false;
        }
    if (head) {
        // head: hc_pre (fused with the output norm) -> output projection of the last row only
        if (!k::deepseek4_hc_pre(hc_, d_prem_, static_cast<const float*>(out_norm_), x_, xn_, E_, HC_, rms_eps_, n,
                                 stream_)) {
            err = "head hc_pre failed";
            return false;
        }
        const float* last = xn_ + uint64_t(n - 1) * E_;
        if (!k::deepseek4_matmul_q4k(logits_, out_w_, last, E_, V_, 1, false, stream_)) {
            err = "head projection failed";
            return false;
        }
    }
    n_past_ += n;
    return true;
}

bool Runner::run(const std::vector<int32_t>& prompt, uint32_t gen, std::string& err) {
    if (prompt.empty()) { err = "the prompt is empty"; return false; }
    const uint64_t total = uint64_t(prompt.size()) + gen;
    if (total > ctx_) {
        err = "prompt + generation (" + std::to_string(total) + ") exceeds the context (" + std::to_string(ctx_) + ")";
        return false;
    }
    if (!alloc_state(static_cast<uint32_t>(total), err)) return false;
    n_past_ = 0;

    const uint32_t np = static_cast<uint32_t>(prompt.size());
    const double t0 = now_sec();
    for (uint32_t c0 = 0; c0 < np;) {
        uint32_t nc = std::min<uint32_t>(np - c0, cap_);
        const uint32_t rest = np - c0 - nc;
        if (rest > 0 && rest < SWA_ && nc > SWA_) nc -= SWA_ - rest;
        const bool last = c0 + nc >= np;
        if (!block(prompt.data() + c0, nc, last, err)) return false;
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
        int32_t t = argmax_logits(err);
        if (t < 0) return false;
        const double td = now_sec();
        for (uint32_t g = 0; g < gen; ++g) {
            generated.push_back(t);
            if (g + 1 < gen) {
                if (!block(&t, 1, true, err)) return false;
                t = argmax_logits(err);
                if (t < 0) return false;
            }
        }
        const double dt = now_sec() - td;
        std::fprintf(stderr, "[v41] generated %u tokens in %.2fs (%.1f tok/s)\n", gen, dt,
                     dt > 0 ? static_cast<double>(gen) / dt : 0.0);
    }
    std::printf("generated=");
    for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : "", generated[i]);
    std::printf("\n");
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
        else if (a == "--logits-out") o.logits_out = next("--logits-out");
        else if (a == "--trace") o.trace = true;
        else if (a == "--no-engram") o.engram = false;
        else if (a == "--no-check") o.check = false;
        else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                         "usage: dsv4_run --ids 1,2,3 [--gen N] [--model BASE] [--engram-dir DIR] [--chunk 8]\n"
                         "               [--ctx N] [--gpu 0] [--trace] [--no-engram] [--no-check] [--logits-out F]\n");
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
    if (o.ids.empty()) {
        std::fprintf(stderr, "dsv4_run: a prompt is required (--ids or --ids-file)\n");
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
    if (!r.run(o.ids, o.gen, err)) {
        std::fprintf(stderr, "dsv4_run: %s\n", err.c_str());
        return 1;
    }
    return 0;
}
