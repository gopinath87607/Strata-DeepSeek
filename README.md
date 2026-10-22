# Strata-DeepSeek

A small, independent engine for **DeepSeek-V4.1-Flash** on your own machine — the DeepSeek engine taken out of
[Project Maya](https://github.com/mw00/project-maya) / [Strata](https://github.com/Niko1221/Strata) and made
standalone, so the repository is only the engine, its packer and its web server.

It runs a **GGUF** of the model (IQ2_XS, IQ3_XXS, MXFP4, K-quants, BF16, Q8_0 …) across **any number of NVIDIA
GPUs of different kinds**, lending VRAM between the prompt buffers and the expert banks, decoding ahead with the
**DSpark** drafter, and answering through an OpenAI- and Anthropic-compatible HTTP server with a live dashboard.

MIT licensed. Copyright © 2026 Gopi (gopinath87607). Built on Maya and Strata (both MIT), whose notices this
repository keeps — see [Credits](#credits).

---

## What is in the box

| Path | What it is |
|---|---|
| `src/ds4/` | The engine: model, GGUF/i-quant readers, placement, expert lending, DSpark, Engram, vision |
| `src/kernels/` | The CUDA kernels (decode, prompt GEMMs, i-quants) and the CPU lane (AVX-512/AVX2 + ggml-cpu dots) |
| `src/core/`, `src/platform/`, `src/ngram/` | The GPU runtime and file/platform helpers the engine needs |
| `serve/` | The HTTP server, the tokenizer/chat templates, MCP tools, telemetry and the web dashboard |
| `tools/ds4/` | `pack_gguf.py` (GGUF → engine pack), `drafter_pack.py` (DSpark), `vision_pack.py` (vision tower) |
| `third_party/ggml/` | ggml's block layouts/codebooks (MIT), the rest of ggml is built from llama.cpp |
| `docs/DS4-WORK.md` | The work log: every DS4 flag, prompt path and measurement to date, with the dead ends |

Everything else from Maya (the GLM and Qwen engines, their tools, tests and data) is deliberately **not** here:
the checkout is a few megabytes and builds one binary, `strata-ds4`.

## Features

- **Multi-GPU, mixed cards.** The engine splits the layers by each card's free VRAM, not by a fixed plan: two
  GPUs, four, ten, or an RTX 3060 beside an RTX 5060. `--gpus` picks which cards, `--reserve-pct` keeps a
  per-card percentage free (the display cards need a little more than headless ones).
- **Expert lending and borrowing.** Each card's prompt buffers (chunk buffers + the CPU lane's ring) are lent to
  the expert banks while decoding and taken back when a prompt arrives: the same bytes are the prompt's scratch
  during prefill and resident experts during decode. `--no-expert-lend` turns it off.
- **Two expert tiers.** The hottest experts stay in VRAM, the rest live in RAM or are read in place from the
  model files (`--mmap-experts auto`); when VRAM is short, the CPU lane answers the remaining picks with AVX-512
  kernels that match ggml's arithmetic bit for bit.
- **DSpark speculative decoding.** The model's `mtp.*` blocks are loaded as a drafter, verified while confident
  (`--dspark-conf`), and reported live (draft acceptance, draft and verify time).
- **Engram.** The hash tables are read from the pack and used by the layers that have them.
- **Chunked prefill, with a decoder tail.** The prompt is processed in chunks sized per GPU (`--prefill-chunk`,
  `--prefill-chunk-max`), with the cross-GPU copies nested so the parts can use different chunk sizes. Layers after
  the candidate-source layer only see the last rows a prompt can still reach, so they run over those alone
  (`DS4_TAIL_PREFILL=1`).
- **Two more prompt paths, each behind its own flag.** `--ep-prefill` gives every GPU a share of every layer's
  non-resident experts and has the layer's owner sum them; `--row-split-prefill` gives every GPU the same layer
  group and a chunk of the prompt's rows, ordered by a wavefront. Both are correct and **off by default**: on
  this host the chunked path is faster (EP re-reads experts the owner already holds, row-split multiplies the
  expert bytes by the number of cards). [The work log](docs/DS4-WORK.md) has the numbers and why.
- **Source-driven dashboard.** The Monitor tabs (GPUs, Loan, Prefill, Storage, Engram, Draft) appear only when
  the engine actually reports that data — one GPU or ten, one CPU or two, SSD or RAM tier.
- **Vision.** The model's vision tower (when the pack has one) reads images in the engine itself.

## Tested on

The development machine is a normal desktop: 2 × RTX 5060 8 GB (CUDA devices 0 and 1) + 2 × RTX 3060 12 GB
(devices 2 and 3), 40 GB of VRAM together, 247 GiB of RAM over two NUMA nodes, and `/mnt/nvme` for the model.
CUDA 13.2, `-DCMAKE_CUDA_ARCHITECTURES="86;120"`. The display cards are the 3060s, which is what the example
`--reserve-pct 1,1,2,2` encodes (1% on the headless 5060s, 2% on the display 3060s).

## Performance

Measured on the development machine above with DeepSeek-V4.1-Flash-1017: a **111,124-token prompt is read at 576
tokens/s** (about 193 s), against **344 tokens/s** before the encoder/decoder split. The split is what buys it: layers
0-20 (the encoder) run over every token, layers 21-39 (the decoder) only over the last 2,560 rows, with the layer
split at 5,5,6,24 (`DS4_TAIL_PREFILL=1`, `DS4_SPLIT=5,5,6,24`).

Two further prompt paths are in the engine but off by default, and measured slower on this host: `--ep-prefill`
reads a 2,282-token prompt in 17.0 s against the chunked path's 15.7 s, and `--row-split-prefill` reads 16k
tokens in 130 s against 52 s. Both produced the same generated tokens as the chunked path in their checks; the
measurements, the reasons, and every other flag are in [docs/DS4-WORK.md](docs/DS4-WORK.md).

One more experiment is measured and off by default: `DS4_PASS_LEND=1` lends the idle expert banks between the two
passes (the tail's banks to the encoders, then the encoders' to the tail). A 116,228-token prompt reads in 219.6 s
with it against 189.0 s without: the stage fills (10.5 s) and the doubled exit refill (26.6 GB against 9.4 GB)
cost more than the streamed experts they save. Chunk sizes above 8,192 pay the same way - 16,384 rows shrink the
8 GB cards' prompt-resident expert slots from 140/156 to 4/37 and read at 486 tok/s, and 24,576 does not load at
all (the prompt arena alone would need 5,986 MiB on an 8 GB card).

## Requirements

- Linux or Windows, x86-64 with AVX2 (AVX-512 recommended for the CPU lane).
- An NVIDIA GPU of compute capability **7.5 or newer** (RTX 20 / 30 / 40 / 50 series); several may be mixed.
- CUDA toolkit **12.8 or newer** for Blackwell (RTX 50); 12.4+ for Ada/Ampere.
- CMake ≥ 3.24, a C++20 compiler, Python 3.10+.
- `jinja2` (the chat templates) and `regex` (the tokenizer). `psutil` enables CPU/RAM/disk readings in the
  Monitor; `numpy` and `Pillow` are needed by the packing tools and picture inputs.
- Enough RAM/VRAM for the quant you choose. The pack directory is beside the GGUFs, no weights are copied.

## Build

```bash
git clone https://github.com/gopinath87607/Strata-DeepSeek.git
cd Strata-DeepSeek
python3 -m pip install -r requirements.txt    # jinja2 + regex are required; psutil/numpy/pillow recommended

# A llama.cpp checkout for ggml (any recent one; the ggml headers in third_party/ggml are pinned):
#   git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_GGML_DIR=$HOME/llama.cpp \
      -DCMAKE_CUDA_ARCHITECTURES="86;120"
cmake --build build -j
```

Without `-DSTRATA_GGML_DIR`, CMake fetches the pinned llama.cpp commit
(`3cf03257f219afbe7334045ff7c6a06ac68c627d`, `third_party/ggml/VERSION.txt`) on first configure.

Set `CMAKE_CUDA_ARCHITECTURES` to your cards: `75` (Turing), `80` (A100), `86` (RTX 30), `89` (RTX 40), `120`
(RTX 50 / Blackwell). Several may be listed (`"86;120"` builds for both). A build for one architecture does not
run on another.

The result is `build/strata-ds4`. `--help` prints the engine's usage.

## Make a pack

The engine reads a **pack** made from a GGUF — a few small text/binary files beside the model, no weights copied:

```bash
# 1. Get the GGUF. The manifest tools/ds4/download_gguf.json pins every file's size and sha256.
python3 -m pip install huggingface_hub
huggingface-cli download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
    --revision c991b28e7c109eed10a55075ddb46f830b5b3059 \
    --include "UD-Q8_K_XL/*" --local-dir ~/models/DeepSeek-V4.1-Flash

# 2. The DSpark drafter (optional, recommended): from DeepSeek's own checkpoint, the three shards that hold it.
python3 tools/ds4/drafter_pack.py --dir ~/models/DeepSeek-V4.1-Flash-1017/shard-44-46

# 3. The vision tower (optional): from shard 1 + config.json of DeepSeek's checkpoint.
python3 tools/ds4/vision_pack.py --dir ~/models/DeepSeek-V4.1-Flash-1017/shard-1

# 4. The pack itself (writes <model>/strata-ds4/ with ds4.txt, tensors.txt, engram.bin, tokenizer/):
python3 tools/ds4/pack_gguf.py --model ~/models/DeepSeek-V4.1-Flash/UD-Q8_K_XL
```

`pack_gguf.py` takes the drafter from `<model>/../dspark/` and the vision tower from `<model>/../vision/` when
they are present (`--drafter` / `--vision` point elsewhere).

### Quant types the packer accepts

`F32`, `F16`, `BF16`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`,
`IQ2_S`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, `MXFP8`.

The engine's own low-bit expert kernels cover the IQ2/IQ3 families (`IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`),
MXFP4 and the K-quants; anything else is dequantized by the GGUF path. A mixed file (dense BF16 + MXFP4
experts + MXFP8 Engram tables) is the usual shape of unsloth's quants.

> **Maya quants** (`peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF`, the IQ2_XXS / IQ3_XXS expert sets) work
> the same way; `tools/ds4/download_gguf.json` lists them with each file's SHA-256.

## Run

### The web server (recommended)

Copy the example config and edit the two paths:

```bash
cp strata-ds4.example.json strata-ds4.json
#   "exe":  "<repo>/build/strata-ds4"
#   "args": ["--pack", "<model>/strata-ds4", ...]
#   "tokenizer": "<model>/strata-ds4/tokenizer"

python3 -m serve.server --engine strata --config strata-ds4.json --port 8080 --open
```

Then open <http://localhost:8080>. The server starts the engine, waits for it, and restarts it if it dies. It
speaks OpenAI (`/v1/chat/completions`), Anthropic (`/v1/messages`), `/health`, `/metrics`, `/metrics/prometheus`
and `/v1/status`; the dashboard is the page at `/`.

`strata-ds4.example.json` keys:

| Key | Meaning |
|---|---|
| `exe` | The engine binary (`build/strata-ds4`) |
| `args` | Engine flags, passed through unchanged (see the table below) |
| `cwd` | Working directory for the engine process (optional) |
| `tokenizer` | `<model>/strata-ds4/tokenizer`, written by `pack_gguf.py` |
| `model_name` | The name the API reports and the chat shows |
| `sampling` | `temperature`, `top_p`, `top_k`, `min_p`, … defaults for the API |
| `fit_max_tokens` | Size the context to the request (`true`) instead of a fixed `--max-context` |
| `log` | The engine's log file (the dashboard's report and restart lines read it) |
| `env` | Extra environment variables for the engine (optional) |

### The engine directly

```bash
# tokens by id, greedy, 16 new tokens:
build/strata-ds4 --pack ~/models/DeepSeek-V4.1-Flash/UD-Q8_K_XL/strata-ds4 \
                 --ids 0,128803,42 --n 16 -v

# the stdin/stdout protocol the server uses:
build/strata-ds4 --pack ... --serve
```

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--pack DIR` | — | The pack directory (`<model>/strata-ds4`), required |
| `--gpus I,J,...` | every visible GPU | Which CUDA devices to use, in CUDA's order |
| `--max-context N` | `32768` | Context window in tokens |
| `--reserve-mib N` | `600` | VRAM kept free on **each** GPU; wins over `--reserve-pct` |
| `--reserve-pct P0,P1,...` | — | Per-card percentage kept free, in the split's (CUDA) order. One value applies to every card; a short list repeats its last; floor 64 MiB. Example: `1,1,2,2` keeps 1% on two headless cards and 2% on two display cards |
| `--no-expert-lend` | lending on | Keep the prompt buffers and expert banks in separate allocations (the pre-lending placement; useful for A/B) |
| `--vram-experts F` | `1.0` | Share of the expert budget to use (tests: lower, forcing the CPU lane) |
| `--cpu-threads N` | physical cores | Threads of the CPU expert lane |
| `--expert-profile FILE` | `<pack>/expert-profile.bin` | Hit-count profile used to place the hottest experts in VRAM; `none` disables reading **and** writing it |
| `--no-expert-cache` | cache on | Stop moving experts between VRAM and the CPU lane as they are used |
| `--tp N` | `2` | Tensor parallelism: runs of N same-kind GPUs take the same layers, each a share of heads, output projection and FFN, summed through host-mapped memory. A card unlike its neighbour stays alone; `--tp 1 --threads off` is layer split only |
| `--threads on\|off` | `off` | One host thread per GPU enqueues its layers (needed for `--tp > 1`) |
| `--prefill-chunk N` | `-1` | Prompt chunk in tokens: `0` = windows of 16, `-1` = 2048 (8192 when experts are read in place), `N` = N tokens. The whole split's ceiling |
| `--prefill-chunk-max N0,N1,...` | — | Per-part ceiling on the prompt chunk, in the split's (CUDA) order; `0`/absent = no ceiling. The steps nest: each part's step is the largest whole multiple of its feeder's step below its ceiling |
| `--ep-prefill` | off | Expert-parallel prefill: every GPU streams and computes a share of each layer's non-resident experts, the owner summing them. One-chunk encoder pass; never with the decoder tail. Slower on this host, kept for the packs and splits it targets |
| `--row-split-prefill` | off | Every GPU holds the same layer group and runs its own chunk of the prompt's rows through it, group by group, ordered by a wavefront; the plan is automatic and any prompt size runs in rounds. Slower on this host |
| `--dspark on\|off\|none` | `on` | DSpark speculative decoding: `off` loads the drafter but does not draft, `none` does not load it |
| `--dspark-conf F` | `0.5` | A draft is verified while its confidence (sigmoid) is at least F |
| `--vision on\|off` | `on` | Run the model's vision tower (when the pack has one) |
| `--mmap-experts on\|off\|auto` | `auto` | The CPU lane reads experts in place from the model files (`on`), copies them into RAM (`off`), or reads in place only when RAM cannot hold them (`auto`) |
| `--ram-resident` | — | The bare form of `--mmap-experts off`: every CPU-lane expert is copied into anonymous RAM up to the copy budget, the coldest staying in the file only when RAM cannot hold them all |
| `--verbose`, `-v` | off | Per-request and per-load detail, including the LEND lines the Monitor reads |
| `--serve` | — | The server's stdin/stdout protocol (`GEN`, `GENI`, `STOP`, `QUIT`) |
| `--ids I,J,...` | — | CLI mode: prompt token ids |
| `--images FILE` | — | CLI mode: pictures for the prompt (PNG/JPEG/BMP …) |
| `--n N` | `0` | CLI mode: tokens to generate |
| `--temperature F`, `--seed N` | `0`, `1` | CLI sampling (0 = greedy) |
| `--dump-logits FILE` | — | CLI mode: write every position's logits (the oracle check) |
| `--encode-image F.rgb --size WxH --dump-image OUT` | — | Run the vision tower alone on raw RGB |
| `--window N` | `16` | Decode window (CLI); above 16 takes the chunked prompt path |
| `--eos ID` | `1` | End-of-sequence token id (CLI) |
| `--layer-split N` | — | Accepted and ignored: this engine balances its own split |

The Maya tree's `--chunk-mmq` (the prompt's MXFP4 experts through llama.cpp's MMQ kernels) is not in this
standalone tree: it needs the Maya tree's prefill MMQ library, which this repository does not carry.

Environment variables keep the `MAYA_*` names they had upstream (the server's settings and `tools/maya_names.py`
map them); nothing in the flags was renamed.

## The Monitor

The dashboard's **Monitor** tab is built from what the engine reports, not from a fixed layout:

| Tab | Shown when | What it shows |
|---|---|---|
| **Overview** | always | tok/s, VRAM/RAM tiers, CPU lane, prompt/decode phase, pipeline |
| **GPUs** | any GPU is visible | Each card's name, PCIe link, load, VRAM, power, temperature |
| **Loan** | any card reports a lend/arena | The per-card split: weights / caches / arena / permanent experts / lent experts / MTP / free, in MiB, from the engine's `LEND` lines |
| **Prefill** | any card reports a chunk | Chunk size per part, the nesting chain, prompt throughput |
| **Storage** | a pack/mmap path is known | Free/used space where the model lives and disk read rate |
| **Engram** | the model has Engram layers | Layer count, rows, hit rate, cache |
| **Draft** | DSpark is loaded | Drafts per token, acceptance, draft and verify time |

Tabs whose data the engine does not report stay hidden, so a one-GPU machine, a ten-GPU machine, a two-socket
server and a laptop all get a truthful page. Direct links work too: `http://localhost:8080/?mtab=lend`.

`/metrics` is the same data as JSON; `/metrics/prometheus` exports the totals plus every numeric `STAT` key the
engine emits (`strata:engine_*`), alongside llama.cpp's `llamacpp:*` names so existing dashboards work.

## How the VRAM is split

At load the engine asks each card what is free after its reserve, places the permanent expert banks there, and
allocates one arena per card for the prompt path. While decoding, the arena's bytes become extra expert banks
(lent experts, refilled with the hottest not-resident experts at each prompt's end); while a prompt runs, the
same bytes are its chunk buffers and the CPU lane's ring. The display cards keep `--reserve-pct` free so the
desktop does not lose its compositor. `--no-expert-lend` keeps both allocations apart for comparison.

## License

MIT. See [LICENSE](LICENSE): Copyright © 2026 Gopi (gopinath87607) and the Strata-DeepSeek contributors, plus
the upstream Project Maya (Peasant Smith) and Strata (Niko1221) notices this work is derived from. ggml and
llama.cpp are MIT by the ggml authors (`third_party/ggml/LICENSE`).

## Credits

This engine is the DeepSeek-V4.1 engine of [Project Maya](https://github.com/mw00/project-maya) (Peasant Smith
and contributors) and [Strata](https://github.com/Niko1221/Strata) (Niko1221 and contributors), extracted into a
standalone MIT repository by Gopi. The quant formats and CPU dot products are matched to llama.cpp/ggml. The
model itself is DeepSeek's; the GGUFs are unsloth's and peasantsmith's.
