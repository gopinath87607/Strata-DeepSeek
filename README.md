<h1 align="center">DeepSeek V4.1 Flash on Strata</h1>

<p align="center"><b>The full 40-layer, 384-expert DeepSeek V4.1 Flash on one normal PC</b><br>
Four consumer graphics cards · routed experts in host RAM · CUDA · Linux (measured)</p>

This repository is a fork of [Strata](https://github.com/Niko1221/Strata) that runs **DeepSeek V4.1
Flash**. It adds a C++/CUDA driver (`tools/dsv4/dsv4_run.cu`), an OpenAI- and Anthropic-compatible
server with a web chat page (`tools/dsv4/dsv4_server.py`), and the host CPU-expert arm
(`tools/dsv4/cpu_vq_moe.cpp`). The 105 GB of routed-expert weights live in host RAM and stream through
a 40 GB VRAM window; the cards hold the trunk and a decode expert cache.

The Qwen3.8-Flash-Next engine and its documentation still exist in this tree (they are what Strata
upstream is), but this README covers only DeepSeek. Measured speeds, every flag, and the exact
commands are below. The engineering record is [docs/DEEPSEEK4.md](docs/DEEPSEEK4.md).

## The model and the quant we used

All measurements below use this exact artifact:

| item | value |
| --- | --- |
| file | `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf`, 40 parts |
| size | **113,556,639,424 B = 113.6 GB (105.8 GiB)** |
| recipe string in the file's metadata | `routed=vq8x8192/4096 sharedcb cbfp8 skeleton=q4_K mtp=vq8x4096 engram=sidecar nocal` |
| architecture / variant | `deepseek4` / `v41_flash` |
| layers / vocab | 40 / 129,280 |
| engram sidecar (separate) | two shards, **203.1 GB (189.2 GiB)** |

The recipe, in plain words:

- **Routed experts - vector-quantized (type 42 `vqblob`), about 1.6 bits per weight.** Layers 0-13
  use 8,192-entry E4M3 codebooks with 13-bit indices (a 12-bit main stream plus a 1-bit plane);
  layers 14-39 and the three MTP towers use 4,096-entry codebooks with 12-bit indices. One expert's
  gate/up/down payload is 7,208,056 B on layers 0-13 and 6,655,096 B on layers 14-39. Each layer has
  384 experts, top-6 per token plus one shared expert.
- **Skeleton - Q4_K (330 tensors):** the attention and shared-expert weights.
- **BF16 (65 tensors):** the router, indexer projections and embeddings.
- **F32 (531 tensors):** norms, router bias, hyper-connection tables and attention sinks.
- **fp8_32x32 (27 tensors, type 44):** the engram `wkv` matrices and 25 MTP tensors - one E4M3
  32x32 block with a ue8m0 scale, 1,025 bytes per 1,024 values.
- **I32 / I64 (4 tensors):** the four engram hash tables.

Where the bytes are:

| part | bytes |
| --- | ---: |
| 40 routed-expert blobs | 105,197,126,272 (92.6%) |
| non-expert weights in the 40 layers | 4,340,347,328 |
| three MTP towers | 3,269,184,440 |
| `token_embd` + `output` | 744,673,280 |
| engram hash tables | 517,952 |
| header and alignment | 4,790,152 |
| **total** | **113,556,639,424** |

## The machine

Every number in this README was measured on this machine. They are not estimates for other hardware.

| item | value |
| --- | --- |
| GPUs | 2 x RTX 3060 12 GB (PCIe Gen3 x16) + 2 x RTX 5060 8 GB (Gen3 x8), 40 GB together |
| CUDA device order | the two RTX 5060 are devices 0 and 1, the two RTX 3060 are devices 2 and 3 |
| RAM | 247 GiB over two NUMA nodes, 56 hardware threads |
| Disk | /mnt/nvme, 916 GB |
| CUDA | 13.2, `CMAKE_CUDA_ARCHITECTURES=86;120` |

The split used here is four stages of ten layers: layers 0-9 on GPU 0, 10-19 on GPU 1, 20-29 on
GPU 2, 30-39 on GPU 3. With `--stage-pipeline` a source layer's compressed KV and top-k selection
cross the seam, and each stage reads its source layer's mirror.

## Build

```sh
cmake -S . -B build-check -DSTRATA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="86;120"
cmake --build build-check -j8 --target dsv4_run
```

The server is plain Python 3 (standard library plus the checkout's `strata_tokenizer`); nothing else
is installed. `--binary` must point at the driver, because the server does not look in
`build-check/` by default.

The CPU MoE source (`tools/dsv4/cpu_vq_moe.cpp`) is compiled with `-mavx2 -mfma` (MSVC:
`/arch:AVX2`). `tools/dsv4/cpu_expert_bench.cpp` is its standalone benchmark and is not part of the
driver.

## How to run it

### The server (recommended)

This is the exact configuration the final measurements used. It is loopback-only; the server refuses
a non-loopback `--host` without `--api-key`.

```sh
cd /home/gopi/strata-deepseek4 && numactl --interleave=0,1 python3 tools/dsv4/dsv4_server.py \
  --binary /home/gopi/strata-deepseek4/build-check/dsv4_run \
  --model /mnt/nvme/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf --engram-dir /mnt/nvme \
  --devices 0,1,2,3 --ram-image --chunk 1024 --serve-ctx 8192 \
  --moe-prefill --dense-batch --tc-gemm --attn-mma \
  --stage-pipeline --engram-feed --engram-scale-cache \
  --expert-cache-mib 1500,1500,5000,5000 --expert-cache-decode-only --reserve-container \
  --port 8080 --open
```

The first start takes about two and a half minutes: 83-89 s pins the 105.8 GiB image in host RAM,
then the weights load on the four cards and the state is sized. The port opens at once and a request
sent during the load waits on the engine lock. When the log prints

```
[v41] resident: state sized for 8192 token(s)
```

the engine is ready. The web chat page is `http://127.0.0.1:8080/`; `/health`, `/metrics` and
`/v1/models` are there too. `/health` reports `max_context: 8192` - the resident state is
`min(--ctx, --serve-ctx)`, and a longer prompt is refused with a 400 before the engine is touched.

Stop it with Ctrl+C. After a client disconnects mid-answer, the engine stops at the next token
(`[cancelled]`) and stays resident for the next request.

### Call it

OpenAI-compatible:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Anthropic-compatible:

```sh
curl -s http://127.0.0.1:8080/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","max_tokens":64,"messages":[{"role":"user","content":"Hello"}]}'
```

Any OpenAI or Anthropic client works: point OpenAI clients at `http://127.0.0.1:8080/v1` and
Anthropic clients at `http://127.0.0.1:8080` (their SDK appends `/v1/messages`). Sampling defaults
follow the reference host: temperature 1.0, top_p 1.0, min_p 0.05; temperature 0 is the greedy path.
`max_tokens` defaults to 4096. One request runs at a time (the engine lock serializes them).

### The driver directly (scripted, one-shot)

The driver takes token ids, not text:

```sh
./build-check/dsv4_run --ids 1,2,3 --gen 32 \
  --model /mnt/nvme/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf --engram-dir /mnt/nvme \
  --devices 0,1,2,3 --ram-image --chunk 0 \
  --moe-prefill --dense-batch --tc-gemm --attn-mma --stage-pipeline --engram-feed
```

`--ids-file FILE` reads the ids from a file instead. For a text prompt there is a small wrapper that
renders the chat head, encodes, runs the driver and decodes:

```sh
python3 tools/dsv4/dsv4_prompt.py --prompt "Explain what a hash table is" --gen 64 \
  --model /mnt/nvme/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf --engram-dir /mnt/nvme --gpu 0
```

That wrapper forwards only `--gpu` and `--chunk`; it is a one-card smoke test, not the fast path.

## Measured speeds

### Reading the prompt (prefill)

Four cards, `numactl --interleave=0,1`, `--ram-image`, `--stage-pipeline`, `--engram-feed`,
`--moe-prefill --dense-batch --tc-gemm --attn-mma`, `--chunk 1024`, unless the row says otherwise.
Measured 2026-10-19.

| prompt tokens | configuration | prefill | seconds |
| ---: | --- | ---: | ---: |
| 8,160 | `--engram-scale-cache` | **200.8 tok/s** | 40.64 |
| 8,160 | without the scale cache | 193.5 tok/s | 42.16 |
| 8,160 | `--no-engram` control (not a usable model) | 243.4 tok/s | 33.52 |
| 4,687 | final serving plan below | **203.7-204.3 tok/s** | 22.94-23.01 |
| 4,096 | scale cache | 182.6 tok/s | 22.43 |
| 2,048 | serving plan, caches 2500,2500,5000,5000 | 152.2 tok/s | 13.46 |
| 1,024 | serving plan, same caches | 116.4 tok/s | 8.80 |
| 1,024 | pipeline + feed, before the cache plan | 106.9 tok/s | 9.58 |
| 4,053 | serial staging path, 2026-10-18 (before the pipeline) | 4.9 tok/s | 832.6 |

The engram costs about 7 s at 8,160 tokens even with the feed: the no-engram control is 33.52 s
against 40.64 s with it. The scale cache paid 3.6% at 8,160 (40.64 s against 42.16 s). For
comparison, Qwen3.8-Flash-Next on the same box measured 382-390 tok/s at about 8,200 tokens.

### Writing the answer (decode)

| configuration | decode |
| --- | ---: |
| four cards, final serving plan, 48-token reply (37-token prompt) | **3.57 tok/s** |
| four cards, warmed resident, caches 2500,2500,5000,5000, 1,024/2,048 context | 3.6-4.5 tok/s |
| four cards, before the expert cache | 2.2 tok/s |
| four cards + host experts, `--cpu-moe 2 --cpu-moe-share 40 --cpu-moe-decode` | 1.74 tok/s |
| Qwen3.8-Flash-Next on the same box | 25-34 tok/s |

81% of decode expert picks hit the cache (4,085 of 5,040 lookups), which is what took decode from
2.2 to 3.6-4.5 tok/s. Decode is a serial 40-layer chain: only one stage is busy at a time, and the
host's VQ kernel ran at about 605 pairs/s in the serving path, so adding the CPU made the token
slower. That is why `--cpu-moe` stays off by default.

### Drafting (`--dspark`, one device only)

Measured on one RTX 3060, 16 generated tokens from the 11-token chat prompt (generation span after
prefill):

| drafting | time | tokens/s | rounds | accepted per round |
| --- | ---: | ---: | ---: | ---: |
| off | 11.87 s | 1.3 | - | - |
| `--dspark-verify 1` | 17.48 s | 0.9 | 12 | 0.25 |
| `--dspark-verify 2` | 21.81 s | 0.7 | 11 | 0.36 |
| `--dspark` (block 5) | 35.93 s | 0.4 | 11 | 0.36 |

The greedy output is byte-identical with drafting on and off, but the drafter does not pay for itself
on this content. `--dspark` is refused with `--serve` and with more than one stage (its capture and
rollback span every layer), so it is not part of the serving plan.

### Startup and memory

| item | measured |
| --- | ---: |
| full resident start | 123-150 s |
| of which pins the 105.8 GiB image in RAM | 83-89 s (1.27-1.4 GB/s) |
| `--engram-scale-cache` adds to startup | about 23 s, 6 GB host RAM |

## Driver flags (`tools/dsv4/dsv4_run.cu`)

Run `./build-check/dsv4_run --help` for the same list. Every off-by-default flag leaves the default
path byte-identical unless the text says otherwise.

### Input and model

| flag | meaning |
| --- | --- |
| `--ids 1,2,3` | the token ids to run (a chat prompt's encoded ids) |
| `--ids-file FILE` | read the ids from a file instead (one per line or comma-separated) |
| `--gen N` | generate N tokens after the prompt; default 0 |
| `--model BASE` | the base path of the GGUF parts (`...part01-of-40` and the rest are found beside it) |
| `--engram-dir DIR` | where the two engram shards are |
| `--ctx N` | context limit for the run; default 0 = the model's own limit (1,048,576) |

### Cards and blocks

| flag | meaning |
| --- | --- |
| `--gpu N` | run every layer on one card |
| `--devices 0,1,2,3` | split the layers over these cards, one stage each, in order; the split must return the same ids as one card |
| `--chunk N` | tokens per prompt block. `0` is auto: one block for the whole run when it fits, otherwise the largest block that keeps every stage's state, windows, engram rows and expert container inside 90% of its free memory after the weights (at most 4096 rows). The default is 8. An explicit value skips the fit check, so pair a large explicit chunk with `--reserve-container` |

### Prefill speed (opt-in)

| flag | meaning |
| --- | --- |
| `--dense-batch` | prompt-width dense matmuls use the token-tiled kernels instead of one block per (row, token) |
| `--tc-gemm` | those tiles run as bf16 tensor-core MMA (128 rows x 128 tokens); implies `--dense-batch`; small f32 rounding differences from the SIMT tiles |
| `--attn-mma` | the prompt's sparse attention runs the tensor-core route (16 heads x 16 keys a block); not bit-equal; batches under 64 tokens keep SIMT |
| `--moe-prefill` | the routed experts' block runs the batched bf16 tensor-core arm (sort by expert, 128-row x 128-token tiles) instead of one warp per (token, expert) pair; blocks of 8 tokens or fewer keep the per-pair arm |
| `--moe-prefill-check` | with `--moe-prefill`: also run the per-pair arm on the first 8 tokens of the first three layers and print the deltas |
| `--stage-pipeline` | with 2+ stages, the chunks flow through the stages at once (a stage's attention and router are issued, then its experts while the others compute); hand-offs use pinned ring slots; needs `--engram-feed` |
| `--engram-feed` | read the next block's engram rows while the current block computes |
| `--engram-scale-cache` | read each engram table's scale plane into host RAM once (3.1 GB a table), so a prompt row costs one random pread instead of two |

### Memory and the expert cache (opt-in)

| flag | meaning |
| --- | --- |
| `--ram-image` | read the whole 105.8 GiB image into pinned host RAM once and stage expert misses from there instead of the disk |
| `--expert-cache-mib N[,N...]` | keep staged expert payloads in N MiB of device slots per stage; a hit is copied from its slot; off by default |
| `--expert-cache-decode-only` | with the cache: look up and admit slots only in decode-sized blocks, so a prompt block does not evict the set decode reuses; refused together with `--expert-loan` |
| `--expert-loan` | with the cache: the pool is the layer's expert container (no second copy, and the chunk reserves the pool once instead of the container plus the cache); off by default, and not used in the serving plan here |
| `--reserve-container` | size every stage's worst-case expert container and its index scratch for a full chunk during startup, before the cache, so no request ever allocates; the fit is printed per stage |

`--reserve-container` is the fix for the long-prompt failure of 2026-10-19. Without it, a prompt
block that selects all 384 experts of a stage tries to grow the container by up to 2,639.7 MiB while
the state and cache are resident; a 4,667-token request died 2.18 s in with `cudaMalloc candidate
blocks (8392704 bytes): out of memory` on the first 8 GB card. With it, startup prints

```
[v41] expert container: stage 0 reserved 2639.7 MiB (all 384 experts)
[v41] stage 0: 2151 MiB free before the expert cache (state + full-chunk reserve)
[v41] stage 0: state + cache + full-chunk reserve + moe scratch resident, 535 MiB free
```

and the same prompt completes twice, 22.99 s and 22.94 s. With the caches `1500,1500,5000,5000`
the four stages keep 535 / 915 / 1660 / 1747 MiB free (the first two are the 8 GB cards). A larger
cache plan fails at startup with the stage and the flag to lower, instead of during a request.

### CPU experts (`--cpu-moe`, opt-in)

| flag | meaning |
| --- | --- |
| `--cpu-moe N` | each prefill layer's N most-selected routed experts are computed on the host out of the RAM image while the GPUs compute the rest; needs `--ram-image` |
| `--cpu-moe-share PCT` | take the hottest experts until PCT (0-100) of the block's pairs are the host's; N stays a cap |
| `--cpu-moe-threads T` | host worker threads; 0 = half the hardware threads |
| `--cpu-moe-check` | on the first three prefill layers, recompute up to two GPU-owned experts on the host for their first eight tokens and print the deltas |
| `--cpu-moe-decode` | hand the host its share in decode and in blocks of 8 tokens or fewer too; required with `--stage-pipeline`, where the pipelined prefill has no host arm |

Measured on this box, the host is slower than the GPUs in the serving path: `--cpu-moe 2
--cpu-moe-share 40 --cpu-moe-threads 28` took 2 of 6 experts per layer and decoded at 1.74 tok/s
against 3.57 tok/s without it. The host computed 1,555 of 8,880 prefill pairs in 2.57 s (about 605
pairs/s) while the GPU arm took 0.41 s. Decode is a serial chain, so the host's work lands on the
critical path. The option is kept, and off by default.

### Drafting (opt-in, one device only)

| flag | meaning |
| --- | --- |
| `--dspark` | the three MTP towers draft a block and the main forward verifies it (greedy only); refused with `--serve` and with more than one stage |
| `--dspark-verify K` | verify at most K drafted tokens per round; 0 (default) verifies the tower's whole block |
| `--dspark-trace` | print each round's k, accepted count, confidence and both id lists |

### Sampling and output

| flag | meaning |
| --- | --- |
| `--temperature T` | 0 or less is greedy; above 0 runs the sampler |
| `--top-p P`, `--min-p M`, `--top-k K` | the sampler's filters; defaults 1.0, 0.0, 0 (off) |
| `--seed S` | sampler seed; default 0, the server picks one per request |
| `--stop-eos` | stop after the model's EOS id |
| `--stream-ids` | print each generated id on its own line as it is produced |

### Serving

| flag | meaning |
| --- | --- |
| `--serve` | resident mode: load once, size the state for `--serve-ctx`, print `ready`, then answer one request per stdin line (`ids=... gen=... temperature=... top_p=... min_p=... top_k=... seed=... stop_eos=...`); a failed forward exits 1 so the supervisor starts a clean process |
| `--serve-ctx N` | the resident state's token budget; default 16384 |

### Diagnostics

| flag | meaning |
| --- | --- |
| `--trace` | dump intermediate tensors for inspection |
| `--no-check` | skip the load-time checks |
| `--no-engram` | run without the engram tables (a control, not the real model) |
| `--logits-out FILE` | write the final logits row to a file |
| `--ledger` | report prefill/decode seconds and the staged/hand-off bytes per stage |
| `--pf-time` | per-layer staging and MoE timings (accepted by the parser; not listed in `--help`) |
| `--help` | the full usage text |

## Server flags (`tools/dsv4/dsv4_server.py`)

`python3 tools/dsv4/dsv4_server.py --help` prints the same list.

| flag | meaning |
| --- | --- |
| `--host HOST` | bind address; default `127.0.0.1`. A non-loopback host is refused without `--api-key` |
| `--port PORT` | default 8000; the examples here use 8080 |
| `--open` | open the chat page in a browser at start |
| `--api-key KEY` | require this key (Bearer or `x-api-key`) |
| `--model PATH` | the base part of the GGUF image; the default is this machine's `/mnt/nvme/...mtpnative.gguf` |
| `--model-id ID` | the model id the wire answers with; default `deepseek-v4.1-flash` |
| `--tokenizer-gguf PATH` | tokenizer source; default: part 1 of `--model` |
| `--engram-dir DIR` | the engram shards |
| `--binary PATH` | the `dsv4_run` binary |
| `--chunk N` | tokens per block; default auto (0), as in the driver |
| `--ctx N` | the model context limit; default 1,048,576 |
| `--devices LIST` | the layer split, e.g. `0,1,2,3` |
| `--gpu N` | one card |
| `--expert-cache-mib LIST` | the per-stage cache budgets |
| `--expert-cache-decode-only` | admit cache slots only in decode-sized blocks |
| `--reserve-container` | reserve each stage's worst-case container and index scratch at startup |
| `--no-resident` | one driver process per request (the old P6 path) |
| `--serve-ctx N` | the resident state's token budget; default 16384, `min(--ctx, --serve-ctx)` is advertised |
| `--timeout S` | request timeout; default 1800 s |
| `--no-engram` | run without the engram tables |
| `--engram-feed` | overlap the next block's engram reads |
| `--engram-scale-cache` | cache the engram scale planes in host RAM |
| `--moe-prefill` | the batched expert arm for prompt blocks |
| `--moe-prefill-check` | compare it against the per-pair arm on the first three layers |
| `--stage-pipeline` | the chunk pipeline across stages; implies `--engram-feed` |
| `--pf-time` | ask the driver for per-layer timings |
| `--cpu-moe N`, `--cpu-moe-share PCT`, `--cpu-moe-threads T`, `--cpu-moe-decode`, `--cpu-moe-check` | as in the driver; with `--stage-pipeline`, `--cpu-moe-decode` is required |
| `--dense-batch`, `--tc-gemm`, `--attn-mma` | the prompt kernels as in the driver |
| `--expert-loan` | the loan cache layout |
| `--ram-image` | keep the whole image in host RAM |
| `--web-dir DIR` | the chat page's files; default the checkout's `serve/web` |

## API

| endpoint | what it is |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions, buffered or SSE (`stream_options.include_usage` supported) |
| `POST /v1/messages` | Anthropic Messages, buffered or SSE |
| `GET /v1/models` | the served model id |
| `GET /health` | readiness, `max_context` and the engine state |
| `GET /metrics` | request/token counters and the last request's timing |
| `GET /` | the web chat page |

## Known limits

- **Decode is 3.6-4.5 tok/s.** The 40 layers run one after another, so at any instant one stage works
  and the other three wait. The expert cache (81% hits) is what made it usable. Adding the host
  (`--cpu-moe`) measured slower; the lever that would help is speculative decode across all four
  stages, which is not built (today `--dspark` is one-device-only and refused with `--serve`).
- **The engram costs time.** Even with `--engram-feed`, the 8,160-token run takes 40.64 s against
  33.52 s with `--no-engram`.
- **Context in the resident server is `--serve-ctx`, 8,192 in this plan.** The state can be sized
  larger, but 1,048,576-token serving was not measured after the pipeline changes.
- **An explicit `--chunk` skips the fit check.** Pair it with `--reserve-container` or keep `--chunk 0`.
- **Only the CUDA build was tested.** The driver is not part of the HIP or SYCL builds; no AMD card,
  Windows build, or other GPU was tested.
- **One request at a time.** The engine lock serializes requests.
- **The artifact is not redistributed here.** You need the 40-part GGUF and the two engram shards;
  the paths above are the ones this build was measured with.

## What was not tested

AMD (HIP) and SYCL builds of this driver; Windows; graphics cards other than the two models above;
DeepSeek artifacts other than the VQ8/shared-14/Q4K MTP-native image; `--cpu-moe` shares other than 2
experts at 40% (the estimate at one expert is arithmetic from the measured 605 pairs/s, not a run);
serving above 8,192 tokens; `--expert-loan` in the final serving plan; DSpark with serving or with
more than one stage.

The long-form record - the port's phase gates, the kernels, every measurement with its configuration,
and the places where it was checked against the reference - is [docs/DEEPSEEK4.md](docs/DEEPSEEK4.md).
