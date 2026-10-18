# DeepSeek-V4.1-Flash on Strata

Status: fork `deepseek4`, cut from upstream main `477ac13c`. P0 is done. P1's
loader half has landed — the family, its type table, geometry, blob-container
checks and the tensor inventory are in `include/strata/artifact/deepseek4.hpp`
with a synthetic test suite — and was checked against the real file's 64 MiB
header: 1,000 tensors, 0 errors. P1's tokenizer has landed too: the
`joyai-llm` pre-tokenizer, the family's control-token table and the pack's
`pre` dispatch are in `tools/strata_tokenizer.py`, and the
`encode`/rendered-chat ids agree with the reference tokenizer oracle on every
corpus string tried except one literal whose engine entry postdates this
checkout's C source (see §8). P1's chat template has landed as the reference's
V4.1 DSML head, byte-pinned against its renderer and the official
`encoding.py` vectors; the pack writer installs it. P2's CUDA half has landed:
the GPU decode is bit-identical to the CPU decoder on the engine's v2 fixtures
and on all 35.4M values of real expert 0, and the row dot reads the packed bit
stream directly; HIP is wired but has no device here to run on. The
blob-payload gate is still open (it waits for the full GGUF). P3's trunk
pieces have landed and are gated too — hyper-connections, the dense
primitives, RoPE, the quantized-KV group, sparse attention, the window ring,
the sparse indexer, and now the router and the routed VQ MoE with the
reference engine's own MoE fixture bit-exact on its one-hot cases; the trunk
forward driver is written, builds, and has answered real prompts on the real
weights (`tools/dsv4/dsv4_run.cu`: the reference's layer order over the 40-part
image, the engram feed, one GPU, greedy decode, the selected experts streamed per
layer). The 25 parts found damaged on 2026-10-18 were re-downloaded and swapped
for copies checked against the tree API's size and SHA-256 (0 of 40 parts now
hold a hole; the parts checker reads all 43 expert blobs, `OK`). Two chat runs on
the repaired image (chunk 8, RTX 3060): "Hello! In one short sentence, what are
you?" (15 prompt tokens) prefilled in 7.99 s, generated at 1.3 tok/s and stopped
at end-of-sentence with "I'm an AI assistant created by DeepSeek to help you with
questions and tasks."; "What is the capital of France? Answer in one word."
(16 tokens) answered "Paris". The first longer prompt exposed an argument-order
bug in the driver's two hc_post calls (the output pointer was passed where the
API takes y, so the kernel wrote into the 128-float comb_ buffer, up to ~650 KB
out of bounds); the earlier 3-token smoke run had stayed inside the arena and
silently corrupted the hyper-connection state, so its ids were not meaningful.
Fixed in `e466ec94`. P3's temperature-0 gate against the reference engine
remains open. This file is the state carrier for the port.

Goal: serve the published DeepSeek-V4.1-Flash-1017 VQ artifact on one normal
PC — NVIDIA or AMD cards plus system RAM — with Strata's layer split, expert
cache, prompt-chunk and loan machinery. The reference runtime
(`Baekpica/ds4-dfm-rs`) runs the model on one device. Strata's reason to exist
is the multi-GPU, normal-PC path, so this is a port of a capability neither
upstream Strata nor the reference has: the model across four consumer cards.

## 1. Why a fork, not a patch to upstream

Upstream Strata opens two families: `qwen4exp` (`include/strata/artifact/gguf_reader.hpp`)
and `qwen35moe` (`include/strata/artifact/model_family.hpp`). The artifact's
`general.architecture` is `deepseek4`, and it carries two tensor types Strata
does not know. A default-path change would touch the loading of every shipped
model, against `AGENTS.md`'s opt-in rule. The fork adds a family behind the
architecture string; the existing path stays byte-identical.

## 2. The artifact, measured here

Read from the first 64 MiB of part 01 with `/tmp/ds41-tools/gguf_dump.py`; the
dump is kept in `reference/gguf-header.txt`.

| item | value |
| --- | --- |
| files | `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf`, 40 parts |
| total size | 113,556,639,424 B |
| header | 4,787,722 B of metadata and tensor descriptors; data starts at 4,787,744 |
| tensors | 1,000 |
| metadata | 73 keys |
| architecture / variant | `deepseek4` / `v41_flash` |
| recipe string | `routed=vq8x8192/4096 sharedcb cbfp8 skeleton=q4_K mtp=vq8x4096 engram=sidecar nocal` |
| tokenizer | `gpt2`, pre `joyai-llm`, vocab 129,280 |
| layers | 40 |

Geometry, all from the metadata:

- Width 5,120; vocab 129,280; 64 attention heads, 1 KV head, key and value
  length 512; q LoRA 1,280; output LoRA 1,024 in 8 groups.
- 384 routed experts per layer, top 6, one shared expert, FFN width 2,304,
  `sqrtsoftplus` gating, routed weights scale 1.5 with norm.
- Hyper-connections: 4 routes, 20 Sinkhorn iterations, epsilon 1e-6.
- RMS epsilon 1e-20; attention sinks; sliding window 128.
- Sparse indexer: 32 heads x 128, top-k 512; KV source layers 2, 8, 14, 20;
  index source layers 2, 8, 14, 20, 24, 28, 32, 36; candidate source layer 20,
  2,048 blocks of 8.
- Compress ratios: 0,0, then 2 for layers 2-19, 1 for layers 20-39, then 0,0,0
  for the three draft towers.
- RoPE base 10,000, factor 16, YaRN 32/1, original context 65,536; compressor
  RoPE base 160,000. SwiGLU clamp exponent 10 on every layer.
- Engram on layers 1 and 14: 4-gram, 8 heads, head dim 256, pad id 2; the
  129,280 model tokens map into a 99,092-entry compressed vocabulary; the
  metadata's engram vocabulary is 16,000,000.

### Tensor types

| type | count | meaning |
| --- | --- | --- |
| F32 | 531 | norms, router bias, hyper-connection tables, sinks |
| Q4_K | 330 | attention and shared-expert weights |
| BF16 | 65 | router, indexer projections, embeddings |
| 42 (`vqblob`) | 43 | one routed-expert blob per layer, plus 3 for the towers |
| 44 (`fp8_32x32`) | 27 | two engram `wkv` matrices and 25 MTP tensors |
| I32 / I64 | 4 | the four engram hash tables |

`fp8_32x32` is measured, not assumed: 1,025 bytes per 1,024 values (a 32x32
E4M3 block and one ue8m0 scale). Every one of the 27 tensors is exactly that
size in the file, with no remainder. Its 32x32 tiles span the whole matrix, so
dim 0 must be a multiple of 32 but not of 1,024; rows such as 1,280 and 2,304
are therefore valid.

### What the loader enforces

The family table is separate from the shared one, because type 42 is Q2_0
(64 values / 18 bytes) to the Qwen models and a DQVL container (1 byte / 1
byte) here. The shared table is not changed, so the default path stays
byte-identical (a test asserts both readings). For `deepseek4`, type 42 is `vqblob` (1/1), 44 is
`fp8_32x32` (1,024/1,025), 40/41 are go1b/go2b (256/34, 256/68), 43 is
fp4x32 (32/17), and 26/27 are I32/I64 (1/4, 1/8). A type-44 tensor's payload
is sized over the whole matrix, not per row.

The inventory is generated from the metadata rather than listed: 22 tensors in
every block and tower (the blob included), compressor `kv`+norm on
`kv_source_layers`, compressor gate additionally only where the compress ratio
is 2, indexer `proj`+`attn_q_b` on `index_source_layers`, indexer `wk`+`k_norm`
only on the intersection of the two source lists, engram `k`/`q`/`wkv` on
layers 1 and 14, 7 globals, and 6 MTP globals. That yields exactly the 1,000
names the file carries; any other name, type or shape is reported by name.

The blob check reads every slot of every container: offsets must be nonzero,
outside the slot table and unique; each payload is bounded by the next slot's
offset; the shared v3 codebook must sit between the slot table and the first
payload; and each payload's header, gains and index stream (including the
3-byte window at the last index) must fit. A container whose declared length
ends inside a payload is refused, not reported as partial.

### Where the bytes are

| part | bytes |
| --- | --- |
| 40 routed-expert blobs | 105,197,126,272 |
| non-expert weights in the 40 layers | 4,340,347,328 |
| three MTP towers | 3,269,184,440 |
| `token_embd` + `output` | 744,673,280 |
| engram hash tables | 517,952 |
| header (4,787,744) and inter-tensor alignment (2,408) | 4,790,152 |
| **sum** | **113,556,639,424** |

The routed experts are 92.6% of the model. That is the part Strata's expert
cache exists to place, and it is why this model fits a normal PC at all.

## 3. The expert blob (GGUF type 42)

One `DQVL` container per layer holds all 384 experts' gate/up/down matrices,
vector-quantized to about 1.6 bits per weight. The format is defined by the C
engine and is ported here from `Baekpica/ds4-dfm-rs`
`crates/ds4-core/src/vq.rs` (MIT), which cites the engine line by line.

```
container  [u32 'DQVL'][u32 version][u32 layer][u32 nexp]
           [nexp*3 u64 payload offsets]          0 = absent
slot       expert e, which 0 = gate, 1 = up, 2 = down
value      value[row][col] = codebook[idx][col % 8] * gain[row]
```

Two payload versions occur. v2 (`DQVQ`) carries a codebook per matrix and is
the older artifact. This model is v3 (`DQV3`): every layer blob and every tower
blob's length closes against the v3 layout with one codebook per container, and
layer 0 was read directly. v3 carries one codebook per layer at `cb_off`, E4M3
codewords, f16 per-row gains, a 12-bit main stream packed per row, and an
optional 1-bit plane that supplies bit 12 of each index. Layers 0-13 use `nc`
8,192 (13-bit, plane present); layers 14-39 and the three towers use `nc`
4,096 (12-bit, no plane).

### Measured against the real file

`/tmp/dsv4_vq_info` on the first 8 MiB of layer 0's blob (fetched from part 01
at offset 456,249,888):

```
DQVL v3, layer 0, 384 experts
expert 0: nc 8192 -> 13-bit indices
  w1 gate: offset 74768  rows 2304 cols 5120 flags 3 mnb 12 cb_off 9232
  w3 up:   offset 2475576 rows 2304 cols 5120
  w2 down: offset 4876384 rows 5120 cols 2304
```

All three matrices decode to 11,796,480 finite values each; ranges are
-0.1246..0.1233 (gate), -0.1227..0.1227 (up), -0.1191..0.1154 (down); the first
four row gains are 0.0201..0.0255. The byte arithmetic closes exactly:

| layers | slot table | layer codebook | per expert | 384 experts + overhead |
| --- | --- | --- | --- | --- |
| 0-13 | 9,232 | 65,536 | 7,208,056 | 2,767,968,272 |
| 14-39 | 9,232 | 32,768 | 6,655,096 | 2,555,598,864 |

Both totals equal the tensor lengths in the GGUF header to the byte, so the
container is fully accounted for. Expert 1 starts at 7,282,824 = 74,768 +
7,208,056, so the slots are packed with no per-expert header.

The decoder in `tools/dsv4/vq_decode.hpp` is bit-exact against the reference
engine's own v2 fixtures (`v2_nc256`, `v2_nc4096`) and covers the v3 12-bit and
13-bit cases synthetically. That is format correctness, not model correctness;
no forward pass has run.

## 4. Engram sidecar

Not in the GGUF. Two safetensors shards hold one table each; the metadata
records the paths, row offsets and embed counts.

| item | value |
| --- | --- |
| rows, table 0 (layer 1) | 384,006,168 |
| rows, table 1 (layer 14) | 384,016,682 |
| row | 264 B: 256 E4M3 weights + 8 ue8m0 scale bytes (one per 32 values) |
| weight plane / scale plane, table 0 | offsets 664 / 98,305,579,672 |
| weight plane / scale plane, table 1 | offsets 672 / 98,308,271,264 |
| shard sizes, reference measurement | 101,535,150,936 and 101,537,926,640 B |

The offsets satisfy `weight_offset + rows*256 = scale_offset` exactly, which is
the row layout the reference reader implements. Per token each engram layer
reads 24 rows (3 n-grams x 8 heads), 6,336 B, through `O_DIRECT`; the reference
falls back to a plain read plus `FADV_RANDOM` on filesystems that refuse it, and
notes that 264-byte random reads through the page cache evict the model mapping
and make temperature-0 runs non-reproducible.

Both shards are on `/mnt/nvme` and their safetensors headers were read on
2026-10-17; they match this table tensor for tensor. Each shard holds, after
the embed planes, `q_weight` and `k_weight` BF16 `[4, 5120]` and `wkv` F8_E4M3
`[25600, 6144]` with E8M0 scales `[800, 192]`, so P4's projections need no
other download. The format is the easy one: an 8-byte little-endian header
length, a JSON map of name → dtype, shape and byte offsets, then raw planes —
no pickle. A row is a pread at `data_start + row*264`. The 264-byte row and
the 6,336-byte per-token block are not `O_DIRECT`-aligned, so the reader needs
aligned window reads or the reference's fallback.

The hash is fully specified by the four GGUF tables: compressed id through
`token_map` (I32, 129,280), four multipliers per engram layer, rolling XOR, then
per-head `rolling % prime + offset`. The tables must be read in the engine's
flat layer-major order, not by their GGUF dimensions; reading by dimension
produces different rows silently. `reference/ds4-raw/src_engram.rs` is the
reference implementation with those caveats in its comments.

Both shards are on `/mnt/nvme`, and both halves of P4 are written and gated.
The host half (hash tables, mask, O_DIRECT row reader, gate emulation) runs
135 checks with 0 failures: the hash matches the reference's
`gen_engram_ref.c` output on its 24-row fixture; the golden prompt positions
captured from this checkpoint decode to the same rows through O_DIRECT,
buffered reads and `FILE*`; and the reference's three gate cases are bit-exact
against its f64 emulation. The device half (rows, gate and wkv in
`src/kernels/cuda/deepseek4_engram.cu`) runs 109 checks with 0 failures: the
rows kernel is bit-identical to the host decode on the fixture and on 192
golden rows per layer read from the real shards; the gate matches the
emulation within the bound (the dense case differs on 2 of 40,960 values, both
within 2 bf16 ulps; absorb and zero are bit-exact); and the real
`[25600, 6144]` wkv agrees with a float64 dot to 1.16e-7 of its scale, for one
token and for two. The measured absolute wkv offsets are 101,377,710,936
(weights) and 101,534,997,336 (scales) in shard 47, and 101,380,486,640 and
101,537,773,040 in shard 48; each scale plane ends exactly at its shard's end.

The wiring is landed: `tools/dsv4/dsv4_run.cu` calls the engram at layers 1
and 14 in the reference's order, and `--no-engram` is the reference's
`--v41-no-engram` compare mode. On the 11-token prompt the last-position
logits differ between on and off on all 129,280 values (max absolute
difference 7.14) while the first 8 greedy ids coincide, so the path is live
and the argmax simply does not move on that content. Still open for P4: the
paired on/off comparison against the reference engine, which is blocked on
this hardware the same way P3's temperature-0 gate is, and the io_uring
overlap if prefill wants it.

## 5. Fit on the test machine

| item | value |
| --- | --- |
| RAM | 247 GiB (134 free + 103 cache at the last check) |
| VRAM | 40 GB: 2 x RTX 3060 12 GB and 2 x RTX 5060 8 GB |
| fast disk | /mnt/nvme 916 G |
| base GGUF | 113.6 GB (105.8 GiB) |
| Engram | 203.1 GB (189.2 GiB) |

All of the base model fits in RAM with roughly 100 GiB to spare for the OS and
page cache, so expert payloads can be read from RAM rather than disk;
`--ram-image` is that tier (see "The RAM image, measured" below). The
non-expert trunk, the three towers and the embeddings together are 8.36 GB;
on the 40 GB of VRAM that leaves about 31.6 GB for a compressed expert cache,
or roughly 4,600 of the 15,360 routed expert payloads (30%) at 6.7-7.2 MB each.
A decode token touches 6 experts in each of 40 layers, so its routed-expert
payload traffic is about 1.63 GB. No speed number is claimed: none has been
measured.

The Engram tables do not fit VRAM or comfortably in RAM alongside the model
(203 GB + 105.8 GiB is over 247 GiB once the OS is counted), so they stay on
NVMe and are read per token. That is the reference design and the reason its
reader is O_DIRECT.

## 6. Port phases and gates

Each phase has a gate that is a comparison, not an inspection.

| phase | work | gate |
| --- | --- | --- |
| P0 (done) | artifact opened; types, geometry and both hard formats read; VQ decoder written and checked | fixtures bit-exact; real layer-0 slot arithmetic closes; expert 0 decodes finite |
| P1 (loader, tokenizer and template landed) | family loader in Strata: `deepseek4` metadata, tensor types 42/44, inventory, tokenizer and chat template | the loader reports the 1,000-tensor inventory and every rejected tensor by name; malformed blobs are refused with a named error; the tokenizer's ids match the reference oracle; the template's bytes match the reference renderer. Met for metadata, types, inventory and blob containers on the real header, for `encode` and rendered-chat ids on the tokenizer corpus, and for the V4.1 DSML template on the official and differential vectors; the 43 blob payloads await the full file |
| P2 (CUDA done; HIP waits for AMD hardware) | VQ decode on CUDA, then HIP: row-dot kernels that read the bit stream directly | GPU output bit-identical to the CPU decoder on the fixtures and on real expert 0. Met on CUDA: 0 mismatching bytes on both v2 fixtures (also against the engine's own dumps) and on all 35.4M decoded values of the real layer-0 expert 0, whose three matrices are v3 13-bit; the row dot agrees with a float64 reference to 9.1e-9 of the term scale. The same source is HIP-translated by the gfx906 build, but no ROCm device exists here to run it |
| P3 (hyper-connections, dense primitives, RoPE, the quantized-KV group, sparse attention, the window ring, the sparse indexer and the router + routed VQ MoE landed and gated) | trunk forward: MLA q/kv LoRAs, compressor source steps, grouped output projection, inverse RoPE, sparse indexer, sinks, sliding window, hyper-connections, RoPE both bases, MoE 384/top-6 + shared, fused VQ MoE, the forward driver | temperature-0 token ids and logits match the reference engine per position. Met for hyper-connections: mix, the Sinkhorn split, hc_pre + the RMS norm and hc_post match the host oracle within summation-order tolerances on 5120x4 at one and five tokens, 300x2 and 128x8; every bf16 output is on the bf16 grid; fused and unfused agree bitwise (364 checks, RTX 3060). Met for the dense primitives: rms_norm, SwiGLU, add/round/scale/expand, the q4_K embedding and the f32/bf16/q4_K matmuls including the grouped output form match their host oracles; the shared q4_K decoder produces dequant_bf16's exact bits on exact headers and stays within one bf16 ulp on varied fp16 ones (171 checks, RTX 3060). Met for RoPE: forward and inverse, both bases, the YaRN ramp and the three head geometries (512/64 at 64 heads, compressor 512/64, indexer 128/64) agree with the host oracle and only the last n_rot values move, bitwise (70 checks, RTX 3060). Met for the quantized-KV group: the compressor pool matches its host oracle, an independent double-precision reference and the bf16 grid (and bitwise when the scores are equal); act_quant's three modes and the ckv (288 B) and idxk (72 B) packs agree with their host oracles bit for bit, the read-back equals act_quant's own write-back at every element, and the idxk pad is zero (201 checks, RTX 3060). Met for sparse attention: the window ring's row mapping and the DSpark draft block match an independent double-precision softmax over the intended rows, an invalid topk entry is bitwise the same output as topk = 0 across an 8-key tile boundary, and the kernel agrees with its host oracle (which writes out the device's 32-lane xor tree) across head widths 64 and 512, 8 and 64 heads, a win_lo clamp that bites and a window-less compressed-only layer (179 checks, RTX 3060). Met for the window ring: the commit and its spec-decode snapshot match their host oracles bitwise over the whole buffer, including the n > window aliasing case, the device position override and the no-rows no-op (77 checks, RTX 3060). Met for the sparse indexer: the score agrees bitwise with its host oracle and a hand-written expectation on integer fixtures (per-lane partials, the xor tree, relu and the bf16 rounding points included) over 32- and 128-dim keys and the compact candidate row; the candidate list's pin, equal-key fill, fewer-visible-than-kk fallback and kk >= nb paths match the oracle bitwise and by hand, including a random block set; the topk output matches by hand and against the oracle with ties, a tail of -1, NaN-treated-as-invisible, the compact candidate mapping and the graph position recompute of ng and topk (320 checks, RTX 3060). Met for the MoE: the router matches its host oracle and the known cases - all logits equal picks experts 0..topk-1 in order, a six-way tie at the top-k boundary picks exactly those six ascending, z = 100 takes softplus's threshold-20 branch (weight exactly 1, not nan), topk > n_expert leaves -1 rounds with weight 0 exactly, and an image token selects on bias_vl - with every id identical and weights within 3.0e-8; the routed chain matches the reference engine's own MoE fixture, all six one-hot cases bit-exact against its f32 dump and the two random cases within 2.7e-4 of the vector's own scale (the reference's own measured worst was 2.6e-4), and the 2-token call and the partials + tail path pass on the same fixture (174 checks, RTX 3060). The shared expert is deliberately not in this module: it is the q4_K dense path, folded by the tail, exactly as the reference composes it |
| P4 (hash, reader, projections, wiring and a live on/off check done; the reference-side on/off comparison waits for the reference engine on this hardware) | Engram: hash, O_DIRECT row reader, q/k/wkv projections, io_uring overlap if prefill needs it | hash rows match the reference implementation on its fixture, then on rows captured from the same checkpoint; output matches with engram on and off. Met for the hash and the row decode: 0 differing rows or values on the reference's hash fixture and on the captured golden positions through all four read paths, and 0 bit-diffs on the GPU row decode of the real shards. The gate and wkv meet their float64 bounds on the fixtures and the real weights. Wired into the driver: `--no-engram` mirrors the reference's compare mode, and on the 11-token prompt the last-position logits differ on all 129,280 values between on and off (max abs difference 7.14) while the first 8 greedy ids coincide, so the path is live. The paired on/off comparison against the reference engine is blocked on this hardware like P3's |
| P5 (towers, the window fill and the verify loop landed and gated; the acceptance numbers are below) | three DSpark towers and verify/rollback | greedy output byte-identical with drafting on and off; cache frontier exact. Met: the same 16 greedy ids from an 11-token chat prompt with drafting off, `--dspark-verify 1`, `--dspark-verify 2` and the full block 5, and with `--chunk 4` (narrower than the 6-row verify batch), and the same 12 ids from a 151-token prompt (whose first window fill takes the full 128 rows) with drafting off and on. The frontier is exact by observation: every later round's argmax reads the rolled-back window, the restored compressor pending rows and the rewound group counts, and the ids match the plain run |
| P6 (serving landed and gated; residency is P7) | serving: OpenAI and Anthropic surfaces, sampling defaults, thinking toggle | one served request per contract. Met on 127.0.0.1 against the real 40-part image: `POST /v1/chat/completions` buffered answers "Paris" with `finish_reason` stop and usage 16 + 2; its SSE form sends the role chunk, the text delta, the finish chunk, the usage chunk and `[DONE]`, then closes; `POST /v1/messages` buffered returns one text block with `stop_reason` end_turn and the same usage; its SSE form sends `message_start`, block start/delta/stop, `message_delta`, `message_stop`, then closes. The thinking toggle routes the pre-`</think>` text to `reasoning_content` (a live 16-token run under `enable_thinking` came back all reasoning, no answer yet, `finish_reason` length). The wire defaults are the reference host's: temperature 1.0, top_p 1.0, min_p 0.05 (the engine's own default is 0.0); temperature 0 is the greedy path |
| P7 (layer split, auto chunk, expert cache and engram feed landed and gated; the loan and ring are open) | Strata integration: layer split, compressed expert cache, prompt chunk, loan and ring | four-GPU run returns the same tokens as one GPU; then a measured performance ledger. Met for the split and the chunk: the 79-token prompt returns `33001,14,2513,734,4189,1910,270,3417` on one RTX 5060 at chunk 8, on four cards at chunk 8 and on four cards in one auto block; the 1017-token prompt returns `671,3051,440,4496,344,260,8837,294` on one 5060 (auto 593 rows, two blocks) and on four cards (auto 672 rows, two blocks, cache on). The ledger is measured: the split costs prefill because the stages serialize (226.1 s on four cards vs 178.5 s on one 5060 at 1017 tokens; 38.2 s vs 36.2 s at 79), the auto chunk takes a 79-token run from 10 blocks to 1 (prefill 38.2 -> 26.2 s, chunk 8 vs one block, four cards) and sizes itself from free VRAM (593 rows on a 2.3 GiB-free 8 GB card, 3360 of 4061 on the four-card chain), and the cache trades 0.25 s of prefill for 1.65 s of decode at 79 tokens (5.63 -> 3.98 s, 17.8% hits, 6.9 GiB not read). The engram feed (`--engram-feed`) returns the same ids and takes 12.5 s off the two stages that own an engram layer (1017 tokens: 225.82 -> 213.37 s; 79 tokens: 38.58 -> 36.89 s). A 4,053-token run picked 3,360 rows and took 832.6 s (4.9 tok/s, 81.2 GiB staged): the block cut bytes a token 3.7x but time a row barely moved, so the dial is flat on this rig. The loan and the ring are not built; the 1k ledger (52.4 GiB staged, I/O about 47-50% of prefill, reads dependency-bound across layers) is in the section below |
| P8 | AMD HIP and Windows builds | P3's gate passes on the same hardware |

P1's synthetic suite and P2's fixtures need no downloads. P1's blob gate and
P3 need the 113.6 GB GGUF; P4 needs the two engram shards.

## 7. Open questions

1. **Expert cache contents.** Storing compressed payloads (6.7-7.2 MB) fits 30%
   of the experts in 40 GB and decodes on use, like the reference; storing
   decoded bf16 tiles (about 70.8 MB per expert) fits under 450 experts but
   removes decode from the hot path. Both are implementable; the decision is a
   measurement, and the answer may differ between prefill and decode.
2. **Hyper-connections.** The 4-route mix and 20-iteration Sinkhorn run per
   layer; the reference binds them as `hc_*` tensors and a `v41_hc_mixes`
   routine. The exact normalization and ordering must be ported from the
   reference, not inferred, or logits will drift.
3. **Sparse indexer and candidate blocks.** These are part of the model, not an
   optimization to add later; skipping them changes outputs. Their source
   layers and block geometry are in the metadata above.
4. **Oracle.** The reference's golden set was captured on a DGX Spark. The
   safest oracle for this machine is the reference runtime built here with
   `make cuda-generic`, or its CPU oracle for the formats. A token match against
   another machine's capture is still valid at temperature 0 if the kernels are
   deterministic, but the same-machine build is the stronger gate.
5. **Engram I/O at prefill.** A decode token needs about 96 small reads across
   the two tables; a prefill chunk multiplies that by its token count. The
   reference overlaps it with io_uring. Whether this machine's NVMe keeps up is
   not measured.
6. **License of the C engine.** The reference runtime is MIT. The engine it
   documents (`JLFN/YoungAi`) needs its license checked before any source is
   copied; the decoder here was written from the reference's MIT code and the
   documented byte layout, with attribution in the header.

## 8. Working files

`reference/` is excluded from git: it holds the GGUF header dump, the raw
reference sources, the v2 fixtures and the 8 MiB layer-0 blob head. It is
research material, not part of the build.

```
tools/dsv4/vq_decode.hpp        header-only decoder, v2 and v3
tools/dsv4/vq_decode_test.cpp   89 checks: synthetic + engine fixtures
tools/dsv4/vq_blob_info.cpp     geometry and decode stats for a real blob

include/strata/artifact/deepseek4.hpp   family-scoped type table, geometry, blob check, inventory
tools/dsv4/gguf_inspect.cpp             checks a real GGUF's header, inventory and blob containers
tools/dsv4/make_tokenizer_gguf.py       writes a metadata-only GGUF from the file's header
tests/core/deepseek4_test.cpp           50 checks: family, types, geometry, blobs, inventory
tests/core/gguf_fixture.hpp             changed: a tensor can declare its own payload length
tools/strata_tokenizer.py               changed: `pre` dispatch, the joyai-llm byte scan,
                                        the family control table, special-token decode,
                                        the DSML template in the pack
tools/test_tokenizer_joyai.py           18 tests: split rules, dispatch, goldens, oracle differential
serve/chat_template_deepseek4.jinja     the V4.1 DSML chat head (shipped into the pack)
serve/test_chat_template_deepseek4.py   16 tests: official goldens, renderer differential, request wiring
tools/dsv4/dsml_render.py               the reference V4.1 render as a Python function (the template's oracle)
tools/dsv4/test_dsml_render.py          14 tests: the reference's own golden vectors
include/strata/kernels/deepseek4_vq.hpp  host/device payload parser, the two GPU entries, and the row dot's shared arithmetic
src/kernels/cuda/deepseek4_vq.cu         P2: whole-matrix VQ decode and the packed-stream row dot
src/kernels/cuda/deepseek4_vq_device.hpp the one-warp row dot, shared by the P2 GEMV and the P3 MoE kernels
tests/core/deepseek4_vq_cuda_test.cpp    the P2 gate: GPU decode vs the CPU decoder, bitwise; row dot vs f64
include/strata/kernels/deepseek4_moe.hpp    P3: the router and routed VQ MoE entries, the tail, the oracle's contract
src/kernels/deepseek4_moe_host.cpp        P3 host oracle: the router lane for lane, the routed chain, the tail
src/kernels/cuda/deepseek4_moe.cu         P3 device half: router, xpack, fused gate/up, down, reduce, tail
tests/core/deepseek4_moe_cuda_test.cpp    the MoE gate: router cases, the reference's own MoE fixture, the oracle, tail
include/strata/kernels/deepseek4_hc.hpp  P3: hyper-connection semantics, host oracle and device entries
src/kernels/deepseek4_hc_host.cpp        P3 host oracle: mix, Sinkhorn split, hc_pre + norm, hc_post
src/kernels/cuda/deepseek4_hc.cu         P3 device half: the five kernels incl. the fused split/pre/norm
tests/core/deepseek4_hc_cuda_test.cpp    the P3 gate: GPU vs oracle on four shapes, bf16 grid, fused bitwise
include/strata/kernels/deepseek4_dense.hpp  P3: dense semantics + the shared q4_K decoder, host oracles, device entries
src/kernels/deepseek4_dense_host.cpp     P3 host oracle: norm, swiglu, elementwise, embed, three matmuls and grouped
src/kernels/cuda/deepseek4_dense.cu      P3 device half: the elementwise kernels, embed and row-per-block matmuls
tests/core/deepseek4_dense_cuda_test.cpp the dense gate: decoder vs dequant_bf16, kernels vs oracle, exact where exact
include/strata/kernels/deepseek4_attn.hpp   P3: attention primitives - the shared rope frequency, block quant, pack formats, window rows, sparse attention, the window ring and the sparse indexer
src/kernels/deepseek4_attn_host.cpp       P3 host oracle: RoPE, the compressor pool, block quant, KV pack and unpack, sparse attention, the window ring, the indexer's score, candidate list and topk
src/kernels/cuda/deepseek4_attn.cu        P3 device half: the rope, pool, quant, pack, unpack, sparse attention, window ring and indexer kernels
tests/core/deepseek4_rope_cuda_test.cpp   the RoPE gate: six geometries vs oracle, bf16 grid, the prefix bitwise
tests/core/deepseek4_attn_kv_cuda_test.cpp the quantized-KV gate: pool, three quant modes, ckv/idxk pack and unpack
tests/core/deepseek4_sparse_attn_cuda_test.cpp the sparse attention gate: exact fixtures, ring and draft-block geometry, host-vs-device
tests/core/deepseek4_win_commit_cuda_test.cpp the window ring gate: commit and snapshot vs oracle, whole buffer bitwise
tests/core/deepseek4_indexer_cuda_test.cpp the sparse indexer gate: exact score fixtures, the candidate list's four selection paths, topk ties/tail/compact/pos
include/strata/kernels/deepseek4_fp8.hpp  P5: the towers' fp8 skeleton matmuls and the draft's small helpers
src/kernels/cuda/deepseek4_fp8.cu        P5 device half. The single-matrix GEMV is the engram's wkv (gated in P4); the grouped form is the same kernel with a grid.y offset. No separate unit test: P5's gate is the driver's byte identity
tools/dsv4/dsv4_run.cu                   the forward driver: 40-part reader, blob staging, the reference's layer order, greedy decode; P5 adds the DSpark towers, the window fill and the verify loop behind `--dspark`; P6 adds host sampling, `--stop-eos` and `--stream-ids`
tools/dsv4/dsv4_prompt.py                text in front of it: the DSML head, the artifact tokenizer, the ids file, decoding back
tools/dsv4/dsv4_server.py                P6 serving: OpenAI Chat and Anthropic Messages, buffered and SSE, one driver process per request behind a GPU lock

include/strata/artifact/deepseek4_engram.hpp       P4 host half: hash, O_DIRECT row reader, gate emulation
include/strata/artifact/deepseek4_engram_bits.hpp  the fp8/bf16 conversions, host and device, shared by both
include/strata/kernels/deepseek4_engram.hpp        P4 device entries: rows, gate, bf16 copy, wkv
src/kernels/cuda/deepseek4_engram.cu               P4 device half
tests/core/deepseek4_engram_test.cpp               135 checks: hash vs the reference, reader paths, gate, real shards
tests/core/deepseek4_engram_cuda_test.cpp          109 checks: GPU rows bitwise, gate bound, wkv vs f64, real shards
```

Build and run:

```
g++ -O2 -std=c++17 tools/dsv4/vq_decode_test.cpp -o /tmp/dsv4_vq_test
/tmp/dsv4_vq_test reference/ds4-raw
g++ -O2 -std=c++17 tools/dsv4/vq_blob_info.cpp -o /tmp/dsv4_vq_info
/tmp/dsv4_vq_info reference/layer0-blob-head.bin 0
g++ -O2 -std=c++17 -Iinclude -Itests/core tests/core/deepseek4_test.cpp -o /tmp/deepseek4_test
/tmp/deepseek4_test
g++ -O2 -std=c++17 -Iinclude tools/dsv4/gguf_inspect.cpp -o /tmp/dsv4_gguf_inspect
/tmp/dsv4_gguf_inspect reference/head.bin --partial
python3 tools/dsv4/make_tokenizer_gguf.py reference/head.bin reference/ds4-tokenizer.gguf
python3 tools/strata_tokenizer.py --gguf reference/ds4-tokenizer.gguf --out /tmp/ds4-pack --check
DS4_TOK_ORACLE=/tmp/ds4_tok_oracle python3 -m unittest tools.test_tokenizer_joyai -v
python3 -m unittest tools.dsv4.test_dsml_render serve.test_chat_template_deepseek4 -v
/usr/local/cuda-13.2/bin/nvcc -O2 -std=c++17 -Iinclude -Itools/dsv4 \
  -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 \
  tests/core/deepseek4_vq_cuda_test.cpp src/kernels/cuda/deepseek4_vq.cu -o /tmp/dsv4_vq_cuda_test
CUDA_VISIBLE_DEVICES=1 /tmp/dsv4_vq_cuda_test reference
cmake -S . -B /tmp/dsv4-build -DSTRATA_BUILD_TESTS=ON
cmake --build /tmp/dsv4-build --target deepseek4_engram_test
/tmp/dsv4-build/deepseek4_engram_test reference /mnt/nvme
/usr/local/cuda-13.2/bin/nvcc -O2 -std=c++20 -Iinclude \
  tests/core/deepseek4_engram_cuda_test.cpp src/kernels/cuda/deepseek4_engram.cu -o /tmp/dsv4_engram_cuda_test
/tmp/dsv4_engram_cuda_test reference /mnt/nvme
cmake -S . -B /tmp/dsv4-build -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
cmake --build /tmp/dsv4-build --target deepseek4_hc_cuda_test
/tmp/dsv4-build/deepseek4_hc_cuda_test
cmake --build /tmp/dsv4-build --target deepseek4_dense_cuda_test
/tmp/dsv4-build/deepseek4_dense_cuda_test
cmake --build /tmp/dsv4-build --target deepseek4_rope_cuda_test
/tmp/dsv4-build/deepseek4_rope_cuda_test
cmake --build /tmp/dsv4-build --target deepseek4_attn_kv_cuda_test
/tmp/dsv4-build/deepseek4_attn_kv_cuda_test
cmake --build /tmp/dsv4-build --target deepseek4_sparse_attn_cuda_test
/tmp/dsv4-build/deepseek4_sparse_attn_cuda_test
cmake --build /tmp/dsv4-build --target deepseek4_win_commit_cuda_test
/tmp/dsv4-build/deepseek4_win_commit_cuda_test
cmake --build /tmp/dsv4-build --target deepseek4_indexer_cuda_test
/tmp/dsv4-build/deepseek4_indexer_cuda_test
cmake --build /tmp/dsv4-build --target deepseek4_moe_cuda_test
/tmp/dsv4-build/deepseek4_moe_cuda_test reference
cmake --build /tmp/dsv4-build --target dsv4_run
python3 tools/dsv4/dsv4_prompt.py --prompt "Hello" --gen 32 --binary /tmp/dsv4-build/dsv4_run
/tmp/dsv4-build/dsv4_run --ids 1,2,3 --gen 8 --chunk 8 --trace
```

Expected: `89 checks, 0 failures`, two fixture lines with 0 mismatches, three
2304x5120 / 5120x2304 matrices with all values finite, `PASS: 50 checks, 0
failures`, and for a header prefix or the full file, `OK` with the inventory
histogram (`F32 x531 Q4_K x330 I32 x1 I64 x3 BF16 x65 vqblob x43 fp8_32x32
x27`). Without `--partial`, every payload must also be inside the file. The
test is also a CMake target, `deepseek4_test`, under `STRATA_BUILD_TESTS`.
The tokenizer extraction prints `vocab 129280, merges 127741, pre joyai-llm`
and `round trip: 16 strings, 0 failed`; the unit suite runs 18 tests, with the
five `Real` ones skipped when `reference/ds4-tokenizer.gguf` is absent and the
oracle differential skipped without `DS4_TOK_ORACLE`. The template suite runs
16 tests and the renderer suite 14; both need no artifact. The GPU test prints
one line per matrix, all with `0 mismatches`, the row dot's worst error (about
`1e-8` of the term scale on the real payloads), and `PASS: 101 checks, 0
failures`; without `reference/` it prints SKIP and exits 77 (ctest treats that
as skipped). The engram host test prints, with the shard directory, `PASS: 135
checks, 0 failures` (without it, `PASS: 134 checks`; the two real-shard lines
and the last check are skipped). The engram GPU test prints `PASS: 109 checks,
0 failures` with the shard directory and `PASS: 67 checks` without it: the rows
line reports `0 bit-diffs`, the dense gate case `2 differ, 0 over bound`, and
each real wkv line its `max rel` (about `1.2e-7`) and `0 over 1e-3*scale`.
Both engram tests are CMake targets under `STRATA_BUILD_TESTS`; the host test
runs its synthetic shard section even with no fixtures, and the GPU test exits
77 when no CUDA device is present. The P3 gates print `PASS: 364 checks, 0
failures` (hyper-connections) and `PASS: 171 checks, 0 failures` (dense
primitives, whose decoder lines report `0 differ` on the exact headers and a
worst of 0.951 x the one-bf16-ulp bound on varied fp16 ones) and `PASS: 70
checks, 0 failures` (RoPE, whose prefix lines report `0 moved` on every
geometry) and `PASS: 201 checks, 0 failures` (the quantized-KV group, whose
lines report `0 bits differ` for act_quant and both packs, `0 differ from
act_quant's write-back` for the read-back identity, and `0 nonzero` for the
idxk pad) and `PASS: 179 checks, 0 failures` (sparse attention, whose lines
report `0 bits differ` for the exact fixtures and the invalid-topk equivalence,
`0 differ between replicated queries` in the draft block, and `0 off the bf16
grid` everywhere) and `PASS: 77 checks, 0 failures` (the window ring, whose
lines report `0 bits differ` and `0 misplaced`) and `PASS: 320 checks, 0
failures` (the sparse indexer, whose score lines report `0 bits differ` - the
device against the host oracle and against the hand expectation - and whose
candidate-list and topk lines report `0 differ`, ties and tails included) and
`PASS: 174 checks, 0 failures` (the MoE, whose six router lines report `0 ids
differ` and a worst weight difference of `3.0e-8`, whose six one-hot fixture
lines report `0 diffs`, whose random fixture line reports `2.7e-4 of scale`
against the reference engine's own f32 dump - the reference measured `2.6e-4`
for its own build - whose host-oracle lines report at most `9.1e-12 of scale`,
and whose tail and 2-token lines report `0 differ`); all exit 77 without a CUDA
device. Without `reference/ds4-dfm-rs/tests/fixtures/ds41/vq/` the MoE test
prints one SKIP line for the fixture and still runs the router, guard and tail
sections.

The forward driver is not a test and is not in ctest: it wants the 113.6 GB
image and both engram shards. It prints the image (`113556639424 bytes in 40
part(s)`), the geometry and wiring line, `weights resident`, `state: chunk 8,
groups cap ...`, a prefill line with its rate, `generated=...` on stdout, and
the decode rate. `--trace` adds `[trace]` max and non-finite counts per layer
half; `--chunk` is the block cap (the prefill chunk rule from the reference),
`--gen` is greedy decode one token at a time, and `--logits-out` writes the
last prefill chunk's logits as f32. A payload slot that sits in a hole reads as
zeros and the driver names the part that holds it.

### The tokenizer, measured

`tokenizer.ggml.pre` is `joyai-llm`, and the model's tokenizer.json shape (the
one llama.cpp carries as a triplet of patterns) is **not** what the reference
engine runs. The engine has a hand-written byte scan (`ds4.c`
`bpe_tokenize_text_joyai`, ported 1:1 in `ds4-core/src/tok.rs`), and the two
regex readings of the triplet each disagree with it on spaces before CJK: a
single alternation merges the last space into the CJK letters, while llama.cpp's
sequential application has already cut the CJK out and cannot. The engine also
counts every non-ASCII byte as a letter, so `😀é` is one piece. The Python scan
is a branch-for-branch port; measured against the C oracle built from
`tests/parity/tokenizer_c_oracle.c`, it gives the same ids on 459 corpus
strings in both plain-encode and rendered-chat modes.

Two special-token facts come from the engine, not from the GGUF (it has no
`token_type` array): the control literals are BOS/EOS, `<｜System｜>`,
`<｜User｜>`, `<｜Assistant｜>`, `<think>`, `</think>`, `｜DSML｜` and
`<｜deepseek_image｜>`, and the ASCII spellings `<|endoftext|>`,
`<|beginoftext|>` and `<|startoftext|>` map to EOS/BOS even though they are not
vocabulary tokens. `<｜System｜>` (id 128799) is the one known disagreement with
this checkout's C oracle: its `special_token_at` table predates the engine's
V4.1 row, which the reference port's plan records as a found-and-fixed gap
(`docs/deepseek41-port-plan.md` §6.17 item 4). The Rust port and the V4.1
renderer both have it, so this port keeps it.

### The chat template, measured

The artifact carries no `tokenizer.chat_template`. The format is the reference
server's V4.1 DSML head (`render_dsml_chat_choice_ex`, `DsmlHead::V41 {
system_token: true }`), which is the official `encoding.py`: its own golden
test records that case 1 is byte-identical to `test_output_1.txt`. The head is

```
<｜begin▁of▁sentence｜>
[<｜System｜> if a system body or a tools prompt is present]
[effort line: "Reasoning Effort: 75 ..." (high) or 100 (max)]
system messages joined by a blank line, then the tools prompt
<｜User｜>first user block, a blank line for an adjacent user-side block
<｜Assistant｜><think>reasoning</think>content + DSML calls + <｜end▁of▁sentence｜>
...
tail: <｜Assistant｜> then <think> or </think>
```

Every DSML tag in this generation opens with a space (`<｜DSML｜ calls>`,
`<｜DSML｜ invoke`, `<｜DSML｜ parameter>`), string parameters are raw, other
parameters are `json.dumps`-shaped, and a closing parameter tag inside a value
is the one thing escaped (its `<` becomes `&lt;`; inside a non-string value,
`<`). Thinking follows the engine's request defaults: **off** unless the
request asks, effort HIGH when it does. A user-only chat with no effort writes
no `<｜System｜>`; a tools prompt alone does.

Two renderings pin it. `tools/dsv4/dsml_render.py` ports the reference
renderer and its 14 tests transcribe the reference's own golden vectors
(`crates/ds4-server/tests/render.rs`): the four official head conversations
and the two spaced-tools conversations, including parallel calls, a
zero-parameter call, `a<b & c`, a nested object, a `\u` escape in a schema and
two merged tool results. `serve/chat_template_deepseek4.jinja` is the shipped
Jinja template; its 16 tests compare it to that renderer over 18 conversations
× 4 tool sets × 4 think modes × 2 `add_generation_prompt` values (576 renders,
0 differing bytes), then repeat the official vectors through the server's own
`openai_to_messages`/`effort_kwargs` path. `python3
tools/strata_tokenizer.py --gguf ... --out <pack>` writes the template into
the pack as `chat_template.jinja`, which `serve/server.py` already prefers
over its own.

Two boundaries, recorded rather than hidden. Strata's shared effort table maps
every client's `max`/`maximum` to `xhigh`, so the 100 line is unreachable from
a request until that table grows a family hook; the template itself accepts
`max` and the renderer's 100-line vector covers it. And `ChatTemplate.render`
drops completed empty assistant turns (#843, a server-wide policy the
reference does not have); the differential disables that filter to compare the
template itself, and `EmptyAssistantTurns` pins the filter as served.

### The VQ decode on the GPU, measured

P2's gate is byte equality, and it is met. Measured on an RTX 3060 (sm_86)
with the CUDA 13.2 toolkit, `tests/core/deepseek4_vq_cuda_test.cpp`:

| input | decoded values | mismatches |
| --- | --- | --- |
| `v2_nc256` fixture (8-bit) | 1,024 | 0 (also 0 against the engine's `ref.f32`) |
| `v2_nc4096` fixture (12-bit) | 1,024 | 0 (also 0 against the engine's `ref.f32`) |
| real layer-0 expert 0, w1 2304x5120 | 11,796,480 | 0 |
| real layer-0 expert 0, w3 2304x5120 | 11,796,480 | 0 |
| real layer-0 expert 0, w2 5120x2304 | 11,796,480 | 0 |

The real payloads are v3, `nc` 8192, 13-bit: the 12-bit main stream plus the
1-bit plane, the path the reference's own comment records as having run 42%
PPL high when the plane advanced per block instead of per 32-index group. The
decode is bit-identical because it is the CPU decoder's arithmetic, not a
hardware shortcut: the same three-byte window, the same integer f16 and E4M3
conversions (0x7f/0xff become positive quiet NaN in both), and one final
multiply by the gain.

The row dot reads the packed stream directly and never materialises a matrix.
One warp per row, lane j takes index groups `j, j+32, ...` (the reference's
single-token mapping), each group is eight values times the same eight bf16
activation values, the gain is applied last and the reduction is the reference's
five-step xor tree. Against a float64 dot of the decoded values and the same
bf16 activation, the worst error is 6.0e-9 (w1), 6.8e-9 (w3) and 9.1e-9 (w2)
of `1 + sum|term|` - the fixtures' 16-row dots are exact. That comparison is a
tolerance on purpose: a bit claim for the dot belongs to P3, against the
reference engine, where the accumulation order and FMA contraction can be
compared like for like.

The E4M3 conversion is integer/float arithmetic rather than
`cvt.rn.f16x2.e4m3x2`. The instruction is exact - E4M3 is a subset of f16 - but
it needs sm_89+, and this fork builds for sm_86 and HIP too; the engine's
measured cost of the scalar table (26% per decode step, 20% prefill) is a
reason to add the instruction behind an arch guard when P7 has a number, not a
reason for two different conversions now. HIP is wired (the gfx906 build
translates every `src/*.cu`, and the compat shim supplies the runtime calls),
but there is no ROCm device on this machine, so the HIP run stays open.

### The DSpark towers and the verify loop, measured

P5 adds the draft path behind `--dspark`; without the flag the driver takes the
plain P3 path unchanged. The driver carries the towers' forward (the fp8
skeleton matrices, the routed VQ experts, the bf16 gates, the borrowed head,
the per-position markov greedy chain, the confidence head), the main-hidden
capture, the tower window's right-aligned fill, and the verify batch with its
rollback. The kernel entries the towers added live in
`include/strata/kernels/deepseek4_fp8.hpp` (CUDA source
`src/kernels/cuda/deepseek4_fp8.cu`): `matmul_fp8` and `matmul_fp8_grouped`
for the towers' type-44 matrices, then `hc_mean` (the capture), `ring_rows`
(the fill's row gather), `row_gather_bf16` (the markov embedding), `argmax`
and `row_add` (the greedy chain and the head bias), each against its reference
kernel.

The gate is byte identity of the greedy stream, not acceptance:

| prompt | drafting | rounds | generated ids |
| --- | --- | --- | --- |
| 11-token chat, `--gen 16` | off | - | `671,6102,294,8760,344,2619,51119,42499,1,0,30,94,8277,19704,19053,94` |
| 11-token chat | `--dspark` (block 5) | 11 | same |
| 11-token chat | `--dspark-verify 2` | 11 | same |
| 11-token chat | `--dspark-verify 1` | 12 | same |
| 11-token chat, `--chunk 4`, `--gen 10` | off | - | `671,6102,294,8760,344,2619,51119,42499,1,0` |
| 11-token chat, `--chunk 4`, `--gen 10` | `--dspark` | 5 | same |
| 151-token prompt, `--chunk 64`, `--gen 12` | off | - | `2389,43,3018,343,82845,60845,14362,17399,305,14733,11,1` |
| 151-token prompt, `--chunk 64`, `--gen 12` | `--dspark` | 10 | same |

`--chunk 4` makes the prefill chunk narrower than the 6-row verify batch (the
runner raises its per-block cap to `block + 1`). The 151-token prompt is
longer than the 128-row tower window, so its first window fill takes 128 rows
and every later one a single row; the 11-token prompt's first fill takes 11.
Both are in the table. `--dspark-verify K` pins the batch to 1+K rows, which
walks a rollback at every width from 2 to 6.

The frontier is exact by construction and by observation. Each round
snapshots the window cells the batch will overwrite (per layer) and the
compressor's pending rows; the verify runs the plain block at
`pos0 = n_past - 1`; `a` is the longest matching prefix; rollback restores
the window, rewinds the group counts to `n_past / ratio`, puts the last
`n_past % ratio` pending rows back and drops the rejected main-hidden cells.
Every later round's argmax reads that rolled-back cache, so the ids staying
equal to the plain run's is the observation.

On acceptance, measured on the RTX 3060 (sm_86) with the 40-part VQ image,
16 generated tokens from the 11-token chat prompt, over the generation span
the driver prints (after prefill):

| drafting | time | tokens/s | rounds | accepted/round | of drafted |
| --- | --- | --- | --- | --- | --- |
| off | 11.87 s | 1.3 | - | - | - |
| `--dspark-verify 1` | 17.48 s | 0.9 | 12 | 0.25 | 25.0% |
| `--dspark-verify 2` | 21.81 s | 0.7 | 11 | 0.36 | 18.2% |
| `--dspark` (block 5) | 35.93 s | 0.4 | 11 | 0.36 | 7.3% |

The drafter does not pay for itself here: a verify step costs a 2- to 6-row
batch and this content accepts 0.25-0.36 tokens per round. That is a property
of this artifact on this content, not a performance claim; P7 owns any speed
number. The reference's own table (M3 Ultra, Q4-imatrix target, public 2-bit
Q2K drafter, temperature 0, yield guard off) is content-dependent in the same
way: ~68% repetitive with ~2.4 tokens/step, ~22% stepwise arithmetic at
~1.3x, ~20% code at ~1.2x, ~13% free prose at ~1.2x, and its yield guard
shuts the drafter off per sequence when it loses. The 151-token prose prompt
lands at 0.10 accepted per round (2.0% of drafted).

`--dspark-trace` prints each round's `k`, `a`, the five confidence sigmoids
and both id lists. It is what to read when acceptance moves: on the prose run
the single accepted first position carried confidence 0.650 and the misses
carried 0.11-0.35, so the head's ordering is meaningful. The towers' small
matrices are read by their registered GGUF type, never assumed: the probe
reports `mtp.markov_embd`, `mtp.markov_head` and `mtp.confidence` bf16,
`mtp.main_norm`/`mtp.out_norm` f32 and `mtp.main_proj` type 44, which is what
the reference's 2026-09-15 crash (f32 read as bf16, acceptance 0.14/5)
requires. That signature is not what the numbers above are.

Two things from the port worth recording:

- The towers' kv buffers are `prod = max(cap_, SWA)` rows, not
  `cap2 = block + 1`. The first run died with `tower window shift: invalid
  argument`: the verify batch is 6 rows but the fill writes up to 128 into
  that same buffer.
- The reference's fill ends with a copy of `main_x` row 0 <- row `rows - 1`.
  It runs after `push_main` (the only reader), and the next fill's
  `main_norm` rewrites row 0, so it is a dead store and is not ported.

```
cmake --build /tmp/dsv4-build --target dsv4_run
/tmp/dsv4-build/dsv4_run --ids-file ids.txt --gen 16 --gpu 0 --dspark --dspark-trace
```

### Serving, measured

`tools/dsv4/dsv4_server.py` is the P6 front end: Python's standard library
only, one `ThreadingHTTPServer`, and a single lock that a request holds from
prompt rendering through the last token (the engine owns the GPU). It serves
two contracts:

- OpenAI Chat, `POST /v1/chat/completions`: `chat.completion`, or
  `chat.completion.chunk` events ending in `data: [DONE]`;
  `stream_options.include_usage` adds the usage chunk. `max_tokens` /
  `max_completion_tokens` default to 4096 here (the reference host's own
  default is 393216).
- Anthropic Messages, `POST /v1/messages`: `message`, or `message_start` /
  `content_block_start` / `content_block_delta` / `content_block_stop` /
  `message_delta` / `message_stop`. `max_tokens` is required, as the
  contract says.

`GET /v1/models` and `GET /health` are there for the usual clients.

The same server also serves the checkout's own `serve/web` chat page at `/`
(`--web-dir` points elsewhere; a missing directory leaves the API only), so
the page known from Strata talks to this engine.  `/health` reports the model
id and context, and `/metrics` feeds the page's Monitor tab: the live request
(`reading`, then `generating` with its decode rate and the prefill rate),
per-GPU utilisation, memory, temperature and power from `nvidia-smi`, host
CPU, RAM and disk rates from `/proc/stat`, `/proc/meminfo` and `/sys/block`,
the engine's model, context, expert-cache MiB and slots, a 60-point history,
and the finished-request table with its totals.  `prompt_ms` is the time to
the first token, so it holds the prefill plus the first decode step; a
request that had to start (or restart) the driver also holds the model open
there (measured: 2.82 s prefill reported for a request whose first token came
7.32 s after it was sent, the difference being the reload).
`decode_tok_s` counts `(tokens - 1)` over the span from the first token to the
last; `reused` is 0 and `hit_rate` null because prefix reuse is not built.
`/config`, `/settings` and `/mcp` are 404s - the page already treats those as
an older server, hides those cards and keeps the chat.
`image_generation` is false, so the Images tab stays hidden.  A
thinking-high web request ("What is the capital of France? Answer in one
short sentence.", 96 max tokens) streamed 31 events and answered
`The capital of France is Paris.` in 34.64 s through the resident server.

Each request renders the DSML head with `dsml_render.render`, encodes with the
artifact's tokenizer, writes the ids file, runs the driver, and reads the id
stream back. The driver grew three flags for this: the sampler
(`--temperature`, `--top-p`, `--min-p`, `--top-k`, `--seed`), `--stop-eos`
(stop after the EOS id; off by default so every P5 gate stays byte-identical)
and `--stream-ids` (one id per stdout line). A `</think>` in the stream splits
the reply: the OpenAI surface puts the part before it in `reasoning_content`,
the Anthropic surface in a `{"type": "thinking"}` block. The EOS id and the
`<｜User｜>`/`<｜Assistant｜>` heads end the turn and are not sent; an
incremental UTF-8 decoder keeps a character split across two ids intact, and
stop strings hold back up to their own length so a stop that straddles two ids
is cut cleanly.

The gate, run on 127.0.0.1 against the real image (RTX 3060, chunk 8, the
40-part VQ artifact):

| request | result |
| --- | --- |
| OpenAI Chat, buffered, temperature 0, max_tokens 8 | `"Paris"`, `finish_reason` stop, usage 16 prompt + 2 completion |
| OpenAI Chat, SSE, same | role chunk, `"Paris"` delta, finish chunk, usage chunk, `[DONE]`, connection close |
| Anthropic Messages, buffered, same | one text block `"Paris"`, `stop_reason` end_turn, usage 16 + 2 |
| Anthropic Messages, SSE, same | `message_start`, text block start/delta/stop, `message_delta`, `message_stop`, close |
| OpenAI Chat, `enable_thinking: true`, max_tokens 16 | `reasoning_content` `"We need answer briefly. 2+2=4. Just respond."`, content empty, `finish_reason` length |

The thinking row is the split working, not a failed answer: 16 tokens was too
small to reach `</think>`, so the whole reply was still reasoning and the
visible answer would follow in `content`.

The sampler is host-side over the logits row the argmax path already reads
back: temperature, top_k (ties to the lower id), min_p relative to the row
maximum, top_p on the normalized prefix, then a seeded xorshift128+ draw.
Checked on the P5 gate's chat prompt (whose greedy output is the 8 ids
`671,6102,294,8760,344,2619,51119,42499`):

| run | ids |
| --- | --- |
| default (temperature 0) | the greedy 8, unchanged |
| `--temperature 0.8 --top-p 0.95 --min-p 0.05 --seed 42`, twice | identical both times: `671,6102,294,8760,344,11111,16,1` |
| the same with `--seed 43` | the greedy 8 |
| `--temperature 0.8 --top-k 1 --seed 7` | the greedy 8 |
| `--temperature 0.8 --dspark` | refused by name: the DSpark verify is argmax-only |

`--stop-eos` appends the EOS id and stops (`...,42499,1` with `--gen 16`);
`--stream-ids` prints each id before the `generated=` line.

The cost is the process. One request at 16 prompt tokens took 12.3-12.8 s
wall: about 3.9-4.2 s of model open, 7.6-7.8 s of prefill and 0.8 s of decode
(2.5 tok/s). The thinking request at 40 prompt tokens took 37.75 s (20.7 s
prefill, 12.8 s for 16 tokens). This path pays a fresh load and a cold prefill
per request; residency landed afterwards under "Residency, measured" below,
and prefix reuse and batching remain open.

Only text is served. `tools`, images and tool results are refused with a 400
by name rather than dropped. There is no API key unless `--api-key` is given,
and a non-loopback `--host` without one is refused.

```
python3 tools/dsv4/dsv4_server.py --port 8000
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":8,"temperature":0}'
```

One report fix came out of this: the driver's `generated N tokens` line
printed the requested cap, not the number emitted, so a `--stop-eos` run that
stopped after 2 of 8 read as 8. It now prints `generated.size()`. The DSpark
arm also now streams accepted tokens and honours `--stop-eos` inside a round,
which it did not before (no gate ran that combination).

### The layer split, the prompt chunk and the expert cache, measured (P7)

Three switches, all in `tools/dsv4/dsv4_run.cu`.  `--devices` and
`--expert-cache-mib` are opt-in and change nothing when absent; the server
passes `--chunk 0` (auto) by default.

- **`--devices 0,1,2,3`** puts one stage per GPU, in layer order.  A cut is
  legal only at a compressed-cache seam - no layer at or after it may take its
  compressed-KV or index source from before it, and the candidate block's
  owner must sit in the same stage as its users - which on this model means
  {2, 8, 14, 20}.  Four stages choose {8, 14, 20}: L0..7 on gpu 0, L8..13 on
  gpu 1, L14..19 on gpu 2, L20..39 on gpu 3.  Each stage holds only its own
  layers' window and source caches and its own copy of every scratch buffer,
  with its own stream; only `hc_` (f32, cap x 4 x 5120), the evolved
  `pre_mix` and the positions cross a boundary, through host memory.
- **`--chunk 0`** is auto: the driver picks the block after the weights and
  the engram are resident, from each stage's free memory.  It takes the
  largest block whose cap-scaled state (about 0.88 MiB a row on an 8-layer
  stage, windows and a stage's engram row buffers included) plus one layer's
  expert container - the union of the block's selected experts, reserved at no
  more than all 384 - stays inside 90% of free memory, up to 4096 rows (the
  MoE grid takes `n_tok * topk <= 65535`) and the run's own length.  A stage
  that owns no engram layer holds no engram row buffers.  Each stage prints
  its own cap, so the binding card is visible: at 1,017 tokens with the
  3072/3072/6144/6144 MiB cache the caps are 743, 992, 1,025 and 672 rows
  (stage 3, the 20-layer 3060), and the block is 672 for every stage.
- **`--expert-cache-mib`** keeps staged expert payloads in device slots keyed
  by `(layer, expert)`.  A hit copies the 6.7-7.2 MB payload from its slot
  into the layer's container instead of reading it back from the parts; a miss
  reads the three spans and inserts, evicting least-recently-used slots and
  never the current container's own hits.

The identity gate is the same tokens through every plan - one card and four,
chunk 8 and a whole-run block, one block and two:

| prompt | plan | prefill | decode (8) | ids |
| --- | --- | --- | --- | --- |
| 79 tokens | chunk 8, one RTX 5060 | 36.24 s | 5.62 s | `33001,14,2513,734,4189,1910,270,3417` |
| 79 tokens | chunk 8, four cards | 38.17 s | 5.64 s | same |
| 79 tokens | auto = 87 rows, one block, four cards | 26.45 s | 3.98 s | same |
| 1017 tokens | auto = 593 rows, two blocks, one RTX 5060 | 178.53 s | 6.56 s | `671,3051,440,4496,344,260,8837,294` |
| 1017 tokens | auto = 672 rows, two blocks, four cards, cache on | 226.11 s | 4.91 s | same |

The 1017-token pair is the wider check: different card count, a different
block size (593 vs 672) and different block boundaries, the same eight ids.
The four-card row carried `--expert-cache-mib 3072,3072,6144,6144`; the fit
includes the cache budget, which is why its block is 672 rather than the
run's 1025 rows.

On this machine the split is capacity, not throughput.  The stages run one
after another - a layer's router needs the previous layer's output - so the
chain costs the sum of its stages: at 1017 tokens the four cards spent
40.18 + 24.17 + 42.09 + 119.53 s of prefill, 75.1 GiB staged, 11,804 expert
slots over 2 blocks (43-46% repeats).  One 5060 did the same prompt in
178.53 s with 2 blocks and 75.3 GiB staged (11,846 slots, 45% repeats).  The
3060s are slower at compute than the 5060s and four contexts add their own
cost.  What the split buys here is room for state and for the cache: on the
one 8 GB card, free memory after the weights (2.3 GiB) caps auto at 593 rows
and leaves no space for cache slots, while the four-card chain held
3072-6144 MiB of cache slots beside a 672-row block at 1017 tokens.

The cache is a decode trade.  The clean pair is the 79-token run at one block:
prefill 26.20 s without and 26.45 s with it (the insert copies), decode
5.63 s without and 3.98 s with.  It hit 1,081 of 6,077 lookups (17.8%) and did
not read 6.9 GiB.  Per stage, with 3072, 3072, 6144 and 6144 MiB:

| stage | layers | slots | hits / misses | not read |
| --- | --- | --- | --- | --- |
| 0 | L0..7 | 446 | 208 / 1016 (17%) | 1429.8 MiB |
| 1 | L8..13 | 446 | 186 / 689 (21%) | 1278.6 MiB |
| 2 | L14..19 | 893 | 210 / 725 (22%) | 1332.8 MiB |
| 3 | L20..39 | 893 | 477 / 2566 (16%) | 3027.4 MiB |

At 1017 tokens over two blocks the same cache avoided 9.9 GiB of reads (hits
202/179/759/424, 10%/11%/42%/7%).  Stage 3 is capacity-bound: 893 slots
against 6,381 lookups over 20 layers.  In the 79-token run the hits are the
decode blocks': the single prefill block fills the slots and every later
block reads them, which is why the cache earns its space there and costs
prefill a fraction of a second.

Auto chunk, measured: 87 rows for the 79-token run; 593 for a 1025-row run on
one 5060 (the old fixed clamp would have asked for 1025 rows, which does not
fit in the 2.3 GiB free after the weights); 672 on the four cards with a
3072 MiB cache on stage 0; 3360 for a 4061-row prompt on the four cards
without one.

The 4,053-token prompt is the block-size check, and it says the dial is
flat: 3,360 rows, 832.6 s prefill (4.9 tok/s), 81.2 GiB staged in 12,767
expert slots, stage 3 alone 437.5 s.  Per token the block cut the staged
bytes 3.7x (20.1 MiB against 75.6 at 1,017 tokens over two blocks) and the
time per row barely moved (stage 3: 0.108 s against 0.116; stage 0: 0.0355
against 0.0392), so throughput rose 8% (4.87 against 4.50 tok/s).  At 672
rows a layer already routes to most of the 384 experts, so the block's reads
are near the full set at either size; the part that grows with rows is
compute, and it grows linearly.  Stage 3 reads about 357 MB/s while it is
I/O-bound at 1,017 tokens and about 104 MB/s at 4,053, where its per-token
compute is the long pole.  With every stage serial there is no other stage
to fill the gap.

```
/tmp/dsv4-build/dsv4_run --ids-file ids.txt --gen 8 --devices 0,1,2,3 --chunk 0 --ledger \
    --expert-cache-mib 3072,3072,6144,6144
/tmp/dsv4-build/dsv4_run --ids-file ids.txt --gen 8 --gpu 0 --chunk 0 --ledger
```

The loan and ring are not built, and the 1k ledger says why they are not
free.  A 1017-token prompt at chunk 1024 staged 52.4 GiB through the parts
reader - about 47-50% of prefill at the measured 250-330 MB/s effective
rate (550 MB/s direct) - and the reads are dependency-bound: layer `il+1`'s
router cannot run before layer `il`'s MoE output exists, so a read of
`il+1`'s experts cannot start earlier than that.  What remains available is
the same layer's own spans.  A fixed ring of prompt rows, as upstream uses,
does not map onto this path either: the block's state is per row and per
stage (about 0.88 MiB a row), so the chunk is bounded by free VRAM, not by a
byte budget chosen in advance.  Both stay open, and this port claims no
number for either.

The overlap that was built instead is the reference's engram feed
(`--engram-feed`, opt-in): `prepare` / `wait` / `prefetch_next` from
`ds4_ds41_forward.inc`.  A block's engram row ids depend only on token ids
and positions, so while a prefill block computes, a worker reads the next
block's rows from the two shards into the other of two host buffers per
engram layer; each engram layer then joins the read and uploads its rows.
The hash stays on the main thread, and one event per (layer, buffer) keeps a
worker from overwriting a buffer whose upload may still be in flight.  It is
worth exactly what the synchronous reads cost: at 1,017 tokens prefill goes
225.82 -> 213.37 s (4.5 -> 4.8 tok/s) and the whole 12.5 s comes off the two
stages that own an engram layer (stage 0: 40.23 -> 35.22 s for L1, stage 2:
42.31 -> 34.71 s for L14); stages 1 and 3 do not move (24.18 and 119.18 s).
At 79 tokens over ten 8-row blocks it is 38.58 -> 36.89 s, decode 5.69 ->
5.52 s.  Both pairs return the gate ids above, and without the flag the
synchronous path runs unchanged.

### Residency, measured (P7)

`--serve` on the driver, and on by default in the server (`--no-resident`
keeps the P6 process-per-request path).  The driver loads the weights and the
engrams once, sizes the state once for `min(--ctx, --serve-ctx)` tokens, prints
`ready`, and then answers one request per stdin line:

```
ids=1,2,3 gen=16 temperature=0 top_p=1 min_p=0 top_k=0 seed=42 stop_eos=1
```

With `--stream-ids` it prints one id per line, then `generated=...`, then
`done`.  A line that does not parse, or a request longer than the resident
state, prints `error=...` and `done` and the process keeps serving; a failed
forward exits 1, and the server's next request starts a fresh driver.
`--dspark` is refused with `--serve`.  The server's `--serve-ctx` defaults to
16384; the context you actually want to serve has to be passed.

Resident mode resets only what belongs to a request: `n_past_`, `in_decode_`,
the group-completion counters and snapshots, the engram n-gram history
(`hist_` - a longer previous prompt must not feed the next prompt's window),
the engram feed buffers (running reads are joined first) and the RNG (reseeded
from `seed`).  The weights, the state allocation and the expert cache slots
persist.  Temperature 0 stays argmax.

The gate is byte identity against a cold process, in one resident process that
alternates a 30-token and an 11-token prompt, greedy, on one card and on four:

| prompt | ids |
| --- | --- |
| long (30 tokens) | `51119,344,270,6102,294,8760,16,455` |
| short (11 tokens) | `32,3476,477,260,11502,22896,16,11992` |
| short, T 1.0 / top_p 1.0 / min_p 0 / seed 42 | `5957,54921,10240,54921,539,31683,54921,331` |

Every occurrence of an arm in the process matched the fresh-process ids.  The
four-card run sent long, short, long, short, and all four matched.

Same HTTP request before and after, same four cards, same flags (`--devices
0,1,2,3 --chunk 0 --engram-feed --expert-cache-mib 1024,1024,4096,4096`):
"What is the capital of France?", max_tokens 8, temperature 0.

| server | request 1 | request 2 | answer |
| --- | --- | --- | --- |
| `--no-resident` (P6 path) | 13.79 s | 13.79 s | `The capital of France is **Paris**.` |
| resident | 9.32 s | 8.13 s | same, usage 11 + 8 |

The non-resident request pays model open and a cold prefill every time; the
resident first request pays them once at startup, and the second finds the
expert cache warm.  A 21-token prose prompt with 48 generated tokens took
34.71 s through the resident server.

The expert cache earns more resident than it did cold, because it survives
between requests.  Isolated driver runs (`--serve-ctx 256`, 8 generated
tokens):

| plan | prefill | decode 8 | cache |
| --- | --- | --- | --- |
| one RTX 5060, no cache | 4.81 s short; 10.38 / 9.98 s long | 5.7-5.8 s | - |
| one RTX 3060, 4096 MiB | unchanged | 4.22 s against 5.73 s with no slots | 595 slots, 28-29% hits, 11.0 GiB not read on request 2 |
| four cards, 1024/1024/4096/4096 MiB | 5.13 s then 4.44 s | 3.80 s then 3.68 s | 148/148/595/595 slots; hits 31-43 / 31-43 / 38-69 / 34-37%; not read 6.6 then 15.9 GiB |

State size costs the prompt chunk: the same four-card plan with the 1/1/4/4
GiB caches picks auto chunks 2191/3491/3261/2130 rows for a 16,384-token state
and 1787/3009/2857/1664 for the full 1,048,576-token one (stage 3 binds either
way).  Each stage owns one KV source layer, so a row costs 360 B there; the
drop is the whole price of serving the full context resident.

```
python3 tools/dsv4/dsv4_server.py --port 8080 --devices 0,1,2,3 --chunk 0 \
    --engram-feed --ram-image --expert-cache-mib 1100,1950,5000,4096 --serve-ctx 1048576
```

### The RAM image, measured (`--ram-image`)

`--ram-image` (opt-in; off by default) reads the whole assembled image into one
host buffer at start and serves every read from it.  The buffer is pinned with
`cudaHostAlloc` when the driver allows it - on this machine it did - which makes
it a direct DMA source: an expert-cache miss is copied straight into the layer's
device container with one `cudaMemcpyAsync` per slot, instead of the old path's
disk read into the pageable staging vector followed by the driver's bounce copy.
Without the flag every read still goes to the parts and the path is unchanged.

Measured on 2026-10-18 on the four-card rig (gen 2, prompt ids 1,2,3): 105.8 GiB
pinned in 141.50 s (803 MB/s; the load is disk-bound - a direct read of part 1
measured 785 MB/s).  The same run without the flag returned the same ids:
`generated=11992,3` both ways (with the image 1.48 s prefill and 0.34 s for 2
tokens, without it 2.30 s and 0.66 s - a tiny run whose first pass includes the
driver's own loads, so it is a correctness pair, not a speed claim).  The load
is one-time: after it no expert read touches the disk.  The two engram shards
are not in it - 203.1 GB does not fit beside 105.8 GiB in 247 GiB - so they
stay O_DIRECT on NVMe.

The resident server warms the driver in the background while the port opens, so
a request sent before the load finishes waits on the engine lock; with the RAM
image that wait is about 2.5 minutes on first start, and the log's
`[v41] ram image: ... pinned in ... s` line says when it is done.

The 1100/1950/5000/4096 MiB cache is the most the fit takes without pulling
stage 3's cap below the other stages': each budget is solved from its stage's
own fit (`cap * row + 384 * slot + fixed + cache <= 0.9 * free`).  On the
server start of 2026-10-18 the budgets were 5405/6049/9425/8169 MiB and the
caps 1484/1492/1627/1412 rows, so the run's chunk was stage 3's 1412.  Stage
3's cap follows the VRAM free when the driver starts - it had been 1663 in a
run earlier the same day - and the caches above leave stage 3 at its old 4096
MiB, so the 1412 was that start's free VRAM, not these caches.  Against the
earlier 1/1/4/4 GiB plan the caches add +76/+926/+904/+0 MiB and +12/+135/+132/+0
slots (1,765 in all, against 1,486).

The documented request (11 tokens in, 8 greedily, through the server) on this
plan took 5.31 s end to end: 3.32 s to first token and 2.00 s for the 8 tokens
(4.0 tok/s in the driver's log; 3.5 tok/s over the 7 inter-token intervals the
metrics count).  The same request under the older plan (no RAM image, 1/1/4/4
cache, chunk 1663) took 8.13 s on its second request earlier the same day.
Two things changed with the plan, so that pair is not an A/B.

## 9. Downloads for the next phases

| file | size | target |
| --- | --- | --- |
| 40 GGUF parts + vision mmproj | 113.6 GB + 971 MB | / (456 G free at the last check) |
| `model-00047-of-00048.safetensors` | 101,535,150,936 B | /mnt/nvme (257 G free) |
| `model-00048-of-00048.safetensors` | 101,537,926,640 B | /mnt/nvme |

Verify against `SHA256SUMS`. Use the 1017 engram shards; the V4.1 shards are a
different model. The reader should take a directory and match the shard names
the metadata records, as the reference's `--engram-dir` does.
