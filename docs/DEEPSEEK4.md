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
blob-payload gate is still open (it waits for the full GGUF). This file is the
state carrier for the port.

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

Still open for P4: wiring the three pieces into a forward pass (that needs
P3), the io_uring overlap if prefill wants it, and the on/off output gate,
which needs the GGUF.

## 5. Fit on the test machine

| item | value |
| --- | --- |
| RAM | 247 GiB (134 free + 103 cache at the last check) |
| VRAM | 40 GB: 2 x RTX 3060 12 GB and 2 x RTX 5060 8 GB |
| fast disk | /mnt/nvme 916 G |
| base GGUF | 113.6 GB (105.8 GiB) |
| Engram | 203.1 GB (189.2 GiB) |

All of the base model fits in RAM with roughly 100 GiB to spare for the OS and
page cache, so expert payloads can be read from RAM rather than disk. The
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
| P3 | trunk forward: MLA q/kv LoRAs, compressors, sparse indexer, sinks, sliding window, hyper-connections, RoPE both bases, MoE 384/top-6 + shared, fused VQ MoE | temperature-0 token ids and logits match the reference engine per position |
| P4 (hash, reader, projections and their GPU gates done; wiring and the on/off gate await P3) | Engram: hash, O_DIRECT row reader, q/k/wkv projections, io_uring overlap if prefill needs it | hash rows match the reference implementation on its fixture, then on rows captured from the same checkpoint; output matches with engram on and off. Met for the hash and the row decode: 0 differing rows or values on the reference's hash fixture and on the captured golden positions through all four read paths, and 0 bit-diffs on the GPU row decode of the real shards. The gate and wkv meet their float64 bounds on the fixtures and the real weights. The on/off output gate needs the forward pass |
| P5 | three DSpark towers and verify/rollback | greedy output byte-identical with drafting on and off; cache frontier exact |
| P6 | serving: OpenAI and Anthropic surfaces, sampling defaults, thinking toggle | one served request per contract |
| P7 | Strata integration: layer split, compressed expert cache, prompt chunk, loan and ring | four-GPU run returns the same tokens as one GPU; then a measured performance ledger |
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
include/strata/kernels/deepseek4_vq.hpp  host/device payload parser and the two GPU entries
src/kernels/cuda/deepseek4_vq.cu         P2: whole-matrix VQ decode and the packed-stream row dot
tests/core/deepseek4_vq_cuda_test.cpp    the P2 gate: GPU decode vs the CPU decoder, bitwise; row dot vs f64

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
77 when no CUDA device is present.

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

## 9. Downloads for the next phases

| file | size | target |
| --- | --- | --- |
| 40 GGUF parts + vision mmproj | 113.6 GB + 971 MB | / (456 G free at the last check) |
| `model-00047-of-00048.safetensors` | 101,535,150,936 B | /mnt/nvme (257 G free) |
| `model-00048-of-00048.safetensors` | 101,537,926,640 B | /mnt/nvme |

Verify against `SHA256SUMS`. Use the 1017 engram shards; the V4.1 shards are a
different model. The reader should take a directory and match the shard names
the metadata records, as the reference's `--engram-dir` does.
