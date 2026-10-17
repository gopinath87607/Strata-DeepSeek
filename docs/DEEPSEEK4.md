# DeepSeek-V4.1-Flash on Strata

Status: fork `deepseek4`, cut from upstream main `477ac13c`. Nothing under
`src/` or `include/` has been changed yet. This file is the state carrier for
the port.

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
size in the file, with no remainder.

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

The hash is fully specified by the four GGUF tables: compressed id through
`token_map` (I32, 129,280), four multipliers per engram layer, rolling XOR, then
per-head `rolling % prime + offset`. The tables must be read in the engine's
flat layer-major order, not by their GGUF dimensions; reading by dimension
produces different rows silently. `reference/ds4-raw/src_engram.rs` is the
reference implementation with those caveats in its comments.

This port has not downloaded the shards and has not run a single hash lookup.

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
| P1 | family loader in Strata: `deepseek4` metadata, tensor types 42/44, inventory, tokenizer and chat template | the loader reports the 1,000-tensor inventory and every rejected tensor by name; malformed blobs are refused with a named error |
| P2 | VQ decode on CUDA, then HIP: row-dot kernels that read the bit stream directly | GPU output bit-identical to the CPU decoder on the fixtures and on real expert 0 |
| P3 | trunk forward: MLA q/kv LoRAs, compressors, sparse indexer, sinks, sliding window, hyper-connections, RoPE both bases, MoE 384/top-6 + shared, fused VQ MoE | temperature-0 token ids and logits match the reference engine per position |
| P4 | Engram: hash, O_DIRECT row reader, q/k/wkv projections, io_uring overlap if prefill needs it | hash rows match the reference implementation on its fixture, then on rows captured from the same checkpoint; output matches with engram on and off |
| P5 | three DSpark towers and verify/rollback | greedy output byte-identical with drafting on and off; cache frontier exact |
| P6 | serving: OpenAI and Anthropic surfaces, sampling defaults, thinking toggle | one served request per contract |
| P7 | Strata integration: layer split, compressed expert cache, prompt chunk, loan and ring | four-GPU run returns the same tokens as one GPU; then a measured performance ledger |
| P8 | AMD HIP and Windows builds | P3's gate passes on the same hardware |

P1 and P2 need no downloads. P3 needs the 113.6 GB GGUF; P4 needs the two
engram shards.

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
```

Build and run:

```
g++ -O2 -std=c++17 tools/dsv4/vq_decode_test.cpp -o /tmp/dsv4_vq_test
/tmp/dsv4_vq_test reference/ds4-raw
g++ -O2 -std=c++17 tools/dsv4/vq_blob_info.cpp -o /tmp/dsv4_vq_info
/tmp/dsv4_vq_info reference/layer0-blob-head.bin 0
```

Expected: `89 checks, 0 failures`, two fixture lines with 0 mismatches, and
three 2304x5120 / 5120x2304 matrices with all values finite.

## 9. Downloads for the next phases

| file | size | target |
| --- | --- | --- |
| 40 GGUF parts + vision mmproj | 113.6 GB + 971 MB | / (456 G free at the last check) |
| `model-00047-of-00048.safetensors` | 101,535,150,936 B | /mnt/nvme (257 G free) |
| `model-00048-of-00048.safetensors` | 101,537,926,640 B | /mnt/nvme |

Verify against `SHA256SUMS`. Use the 1017 engram shards; the V4.1 shards are a
different model. The reader should take a directory and match the shard names
the metadata records, as the reference's `--engram-dir` does.
