# inference

Turn a [Radxa Orion O6](https://radxa.com/products/orion/o6/) (or any Debian Bookworm box) into a local LLM inference server — one command, many models.

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/gregology/inference/refs/heads/main/install.sh | sudo bash
```

This will:

1. Install system packages (build tools, Vulkan, etc.)
2. Create a dedicated `llm` system user under `/srv/llm`
3. Grant GPU render-node access
4. Build Vulkan-Headers, glslc and llama.cpp from source
5. Set up the Hugging Face CLI
6. Download every model listed in `models.toml`
7. Generate a `models.ini` router config
8. Install and enable a `llama-router` systemd service

The server exposes an OpenAI-compatible API on port 8000 with automatic model loading.

## Re-running

Run the same command again to pick up changes. The installer is idempotent — completed steps are skipped.

To add a model, add a block to `models.toml` and re-run. To remove a model, delete its block and re-run with `--prune`:

```bash
curl -fsSL ... | sudo bash -s -- --prune
```

## Options

```
--prune         Remove model files for entries no longer in models.toml
--dry-run       Show what would be done without making changes
--only STEP     Run a single step (packages, user, gpu-permissions,
                vulkan-headers, spirv-headers, shaderc, build-llama,
                huggingface, models, router-config, systemd,
                resident, services)
--host HOST     Listen address (default: 0.0.0.0)
--port PORT     Listen port (default: 8000)
--models-toml   Path to models.toml (default: repo's models.toml)
```

## models.toml

The model manifest. Each model has a Hugging Face source and one or more router profiles:

```toml
[models.qwen3_8-27b]
repo = "lmstudio-community/Qwen3.8-27B-GGUF"
file = "Qwen3.8-27B-Q4_K_M.gguf"

[models.qwen3_8-27b.profiles.default]
name = "qwen3.8:27b"
ctx_size = 262144
device = "none"
gpu_layers = 0
extra = { threads = 8, threads_batch = 12, reasoning_budget = 1024 }

[models.qwen3_8-27b.profiles.gpu]
name = "qwen3.8-gpu:27b"
ctx_size = 262144
gpu_layers = 999
extra = { threads = 1, reasoning_budget = 1024, ubatch_size = 64 }
```

`extra` keys are passed to llama-server as `models.ini` options (underscores become hyphens), so `reasoning_budget = 1024` becomes `--reasoning-budget 1024`.

For models that exceed available RAM (large MoE models), use `include` instead of `file` and set `large = true`. The pattern can name a shard directory (`IQ3_XXS/*`) or root-level shards (`Name-IQ4_XS-*.gguf`). Run them on the CPU (`device = "none"`) and force mmap so the experts page in from NVMe:

```toml
[models.qwen3_8-flash-next-uncensored]
repo = "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF"
include = "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-*.gguf"
large = true

[models.qwen3_8-flash-next-uncensored.profiles.default]
name = "qwen3.8-flash-next-uncensored:125b"
ctx_size = 524288
device = "none"
gpu_layers = 0
extra = { load_mode = "mmap", no_repack = true, threads = 8, ctx_checkpoints = 8, reasoning_budget = 2048, parallel = 2 }
```

### Resident profiles

A profile with a `port` runs in its own always-on llama-server (`llama-<name>.service`) instead of the router, so the router's least-recently-used eviction never unloads it and its prompt cache stays warm. Use it for interactive clients that can't wait for a cold load, like the voice assistant:

```toml
[models.qwen3_5-9b-uncensored.profiles.voice]
name = "voice:9b"
port = 8001
ctx_size = 32768
device = "none"
gpu_layers = 0
extra = { threads = 8, threads_batch = 12, jinja = true, reasoning_budget = 0 }
```

The `[defaults]` and the profile's keys become llama-server flags (`true` is a bare flag, `false` is left off). A resident model holds its RAM and competes with the router's CPU models for cores and memory bandwidth whenever both are busy.

### Services

`[services.<name>]` blocks install Python network services, each in its own venv under `/srv/llm/services/<name>`, run as the `llm` user by `inference-<name>.service`. They provide the voice assistant's speech-to-text and text-to-speech (Wyoming, for Home Assistant's Assist pipeline) and its web search (MCP):

| Service | Port | What | Home Assistant integration |
|---|---|---|---|
| `stt` | 10300 | Qwen3-ASR 0.6B via wyoming-faster-whisper | Wyoming Protocol |
| `tts` | 10200 | Piper, `en_US-lessac-medium` | Wyoming Protocol |
| `search` | 10500 | DeepSeek Search/Fetch as MCP tools | Model Context Protocol, SSE URL `http://<host>:10500/sse` |

Keys: `package` (pinned pip spec), `module` or `script` (a file in this repo, copied next to the venv), `port`, `args` (may use `{port}`, `{data}` and `{download}`), optional `env_file` and `download` (a Hugging Face repo fetched into the data dir). The search service needs `/srv/llm/etc/deepseek.env` with `DEEPSEEK_API_KEY=…` (owner `llm`, mode 0600) and only accepts Home Assistant and loopback, since DeepSeek bills per request.

### Gated repos

Some repos (all the `orcarouter/*` ones) are gated. Accept the terms on each repo's Hugging Face page, then log in once as the `llm` user so the installer's downloads can authenticate:

```bash
sudo -u llm env HF_HOME=/srv/llm/cache/huggingface /srv/llm/venv/bin/hf auth login
```

## Usage

Once running, use the OpenAI-compatible API:

```bash
curl -s http://YOUR_IP:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8:27b",
    "messages": [{"role":"user","content":"Hello!"}]
  }' | jq .
```

List available models:

```bash
curl -s http://YOUR_IP:8000/v1/models | jq '.data[].id'
```

llama.cpp lists profile names with an upper-cased tag (`qwen3.8:27B`); the names as written in `models.toml` work too, via an alias.

The router loads models on demand and keeps up to `models_max` of them loaded (`[router]` in `models.toml`, 2 by default); requesting another unloads the least recently used one.

## Performance on the O6

Measured through the router with llama.cpp f3363593f and Mali driver r53p0 (tokens/s; prompt measured on a ~1.2K-token prompt, generation at short context):

| Profile | Runs on | Prompt | Generation |
|---|---|---|---|
| `qwen3.8:27b`, `qwen3.8-uncensored:27b` | CPU | ~9.6 | ~2.2 |
| `qwen3.8-gpu:27b`, `qwen3.8-uncensored-gpu:27b` | GPU | ~4.3 | ~2.4 |
| `orcasaq-2-cyber:27b` | CPU | ~5.4 | ~2.4–2.5 |
| `qwen3.8-flash-next-uncensored:125b` | CPU | ~11 | ~4.7–5.9 |
| `glm-5.3-flash-uncensored:320b` | CPU | ~2.7 | ~1.1 |

Prompt processing dominates: a 4K-token prompt takes ~7 minutes on the 27B models. The router's prompt cache (`cache_ram`) makes follow-up turns cheap: only new tokens are processed, even when the client drops the model's reasoning from the history. A 23-token follow-up takes ~4 s on the CPU but ~20 s on the GPU, where ggml-vulkan runs batches of 9–32 tokens on its 32-wide matmul tile, which is slow on the Mali. Models serve one request at a time (`parallel = 1` in `[defaults]`) and queue the rest; Flash-Next has two slots (`extra = { parallel = 2 }`), so two requests each get ~3.5 tok/s.

The CPU and the GPU share ~40 GB/s of memory bandwidth, so the GPU is only worth using for a second model. With a `-gpu` profile and a CPU model loaded side by side, both answer at once: the 27B on the GPU at ~2.1 tok/s and Flash-Next at ~3.5, against ~1.6 and ~1.4 with both on the CPU. Splitting one model between the two is slower than either alone (Flash-Next with experts on the CPU and the rest on the GPU: 5.3/2.7 tok/s).

What made the difference:

- **An explicit Arm target for the CPU backend.** llama.cpp's `GGML_NATIVE` probes the CPU's extensions with `-mcpu=native+<ext>`, which Bookworm's GCC 12 rejects for the O6's Cortex-A720, so every probe failed and the CPU kernels were built for plain armv8-a, without dot-product or int8 matmul instructions. The `build-llama` step now passes the extensions from `/proc/cpuinfo` (`GGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod+i8mm+bf16`). Prompts got 1.4–3.7x faster (27B: 2.8 → 10.1 tok/s) and generation up to 20% faster.
- **No weight repacking for mmap'd models.** llama.cpp copies the tensors it can repack for its int8 kernels out of the page cache into RAM. For Flash-Next that cost ~25% of its generation speed, so the large models set `no_repack = true`.
- **glslc from shaderc.** Bookworm's `glslc` is too old for ggml-vulkan's int8 dot-product shaders, which the Mali supports. The `shaderc` step builds a current one.
- **Mali environment.** The router unit sets `GGML_VK_DISABLE_COOPMAT=1` (the driver's cooperative-matrix path is ~4x slower than int8 dot-product) and `GGML_VK_FORCE_MMVQ=1` (~20% faster generation).
- **GPU profiles: 64-token ubatches and one CPU thread.** ggml-vulkan sizes its matmul tile from the batch, and the 64-wide tile is the Mali's fastest (`ubatch_size = 64`, 13–19% faster prompts). With the default 12 threads, a GPU model's idle CPU threads slowed a CPU model running beside it by ~20%, so GPU profiles set `threads = 1`.
- **Plain K-quants on the GPU.** IQ4_XS-heavy files (unsloth's UD quants, OrcaSAQ) are ~35% slower on the Mali.
- **CPU for MoE.** Splitting experts (CPU) from the rest (GPU) is slower than running the whole model on the 8 big cores.
- **12 threads for prompts.** CPU profiles generate on the 8 big cores, but single-slot ones process prompts on all 12 (`threads_batch = 12`; 6–16% faster). Flash-Next doesn't, because llama.cpp also uses those threads when its two slots generate together, which made each 24% slower.
- **Not worth it:** MTP speculative decoding (net slower), f16 KV cache, larger batch sizes, more than 8 generation threads, SVE kernels (no faster than NEON on the A720's 128-bit SVE).

### Two models at once

Real-world runs on the production router (llama.cpp 7182ef71, `models_max = 2`) with `qwen3.8-flash-next-uncensored:125b` (CPU) and `qwen3.8-uncensored-gpu:27b` (GPU) both loaded. The requests in each row start together. Each one asks for a 256-token answer with thinking off and has its own text, so nothing comes from the prompt cache. In the "only" rows the other model is loaded but idle. Each cell gives the speed and the elapsed time: prompt speed and the wait for the first token, then answer speed and the time until the answer is complete.

Chat question (~40-token prompt):

| Running at once | Flash-Next: first token | Flash-Next: answer | 27B GPU: first token | 27B GPU: answer |
|---|---|---|---|---|
| Flash-Next only | 8 s | 5.2 tok/s, done in 57 s | | |
| 27B only | | | 15 s | 2.4 tok/s, 120 s |
| 1 Flash-Next + 1 27B | 10 s | 3.5 tok/s, 83 s | 16 s | 1.9 tok/s, 150 s |
| 2 Flash-Next + 1 27B | 15 s | 1.9 tok/s each, 148 s | 16 s | 1.7 tok/s, 168 s |

Document (1.1–1.4K-token prompt):

| Running at once | Flash-Next: prompt | Flash-Next: answer | 27B GPU: prompt | 27B GPU: answer |
|---|---|---|---|---|
| Flash-Next only | 8.1 tok/s, first token at 2.2 min | 4.3 tok/s, done at 3.2 min | | |
| 27B only | | | 4.4 tok/s, 4.5 min | 2.3 tok/s, 6.3 min |
| 1 Flash-Next + 1 27B | 7.5 tok/s, 3.0 min | 3.6 tok/s, 4.2 min | 4.4 tok/s, 4.1 min | 2.3 tok/s, 6.0 min |
| 2 Flash-Next + 1 27B | 3.5–3.9 tok/s each, 5.3 min | 1.9 tok/s each, 7.6 min | 4.3 tok/s, 5.5 min | 1.7 tok/s, 8.0 min |

Long document (4.1–5.1K-token prompt):

| Running at once | Flash-Next: prompt | Flash-Next: answer | 27B GPU: prompt | 27B GPU: answer |
|---|---|---|---|---|
| Flash-Next only | 7.5 tok/s, first token at 9.5 min | 4.0 tok/s, done at 10.6 min | | |
| 27B only | | | 4.3 tok/s, 19.1 min | 2.0 tok/s, 21.2 min |
| 1 Flash-Next + 1 27B | 7.4 tok/s, 11.6 min | 3.3 tok/s, 12.9 min | 4.2 tok/s, 17.6 min | 2.1 tok/s, 19.6 min |
| 2 Flash-Next + 1 27B | 4.1–6.1 tok/s each, 19.3–20.5 min | 1.3–2.1 tok/s, 22.6 min | 4.3 tok/s, 16.9 min | 1.8 tok/s, 19.3 min |

A long document (~5.3K tokens) beside chats:

| Running at once | Request | First token | Answer, done |
|---|---|---|---|
| Document on the 27B GPU, two chats on Flash-Next | 27B document | 20.8 min | 2.0 tok/s, 22.9 min |
| | each Flash-Next chat | 20 s | 2.0 tok/s, 2.5 min |
| Document on Flash-Next, a chat on each model | Flash-Next document | 12.4 min | 2.0 tok/s, 14.5 min |
| | Flash-Next chat | 9.3 min | 0.8 tok/s, 14.5 min |
| | 27B GPU chat | 15 s | 2.1 tok/s, 2.3 min |

What this means in practice:

- **Running requests together beats queueing them.** Three chats at once (two on Flash-Next, one on the 27B) were all done in 2.8 minutes, against ~3.9 minutes one after another. Three long documents were done in 22.6 minutes, against ~42.
- **The GPU 27B processes documents at the same speed whatever Flash-Next is doing** (4.2–4.4 tok/s in every document run). Its answers slow from 2.0–2.4 to 1.7–1.9 tok/s while Flash-Next is generating too.
- **Flash-Next answers about a third slower while the 27B generates** (5.2 → 3.5 tok/s), and at ~1.9 tok/s per request when both of its slots are busy too.
- **Keeping the 27B loaded slows Flash-Next by ~25%, even while the 27B is idle.** Its ~22 GB come out of the page cache that holds Flash-Next's experts. With the 27B unloaded, 1.1–1.3K-token documents ran on Flash-Next at 10.1–11.2 tok/s prompt and 5.7–5.8 tok/s answer, against 8.1 and 4.3 with it loaded. Unload the GPU model when you're done with it (`POST /models/unload`), or set `models_max = 1` if you rarely run two models at once.
- **A long document on Flash-Next blocks its other slot.** A chat sent to Flash-Next during a 5.3K-token document waited 9.3 minutes for its first token, while a chat sent to the GPU 27B at the same moment answered in 15 s. Send chats to the model that isn't busy with a long document.
- **A long document on the GPU 27B doesn't block Flash-Next chats** (first token in 20 s), though it slows their answers to ~2 tok/s.

## Tests

```bash
# All tests (including HF network checks)
python3 -m pytest tests/ -v

# Offline only (fast, no network)
python3 -m pytest tests/test_config.py tests/test_router_config.py tests/test_systemd.py tests/test_shaderc.py -v

# HF manifest validation only
python3 -m pytest tests/test_huggingface.py -v
```

The HF tests verify that every repo, file, and include pattern in `models.toml` actually exists on Hugging Face. Run these after editing the manifest to catch typos before deploying.

## Architecture

```
install.sh            Shell bootstrap (ensures git + python3, clones repo)
  └─ python3 -m installer
       ├── config.py       Load + validate models.toml (stdlib tomllib)
       ├── hardware.py     Detect GPU, NPU, Coral TPU, PCIe devices
       ├── runner.py       Iterate steps, skip completed, report errors
       └── steps/
            ├── packages.py        apt packages
            ├── user.py            llm user + dirs
            ├── gpu_permissions.py render node access
            ├── vulkan_headers.py  build newer headers
            ├── shaderc.py         build a current glslc
            ├── build_llama.py     cmake llama.cpp
            ├── huggingface.py     HF CLI venv
            ├── models.py          download / prune models
            ├── router_config.py   generate models.ini
            ├── systemd.py         llama-router.service
            ├── resident.py        llama-<name>.service per resident profile
            └── services.py        inference-<name>.service per [services] entry
services/
  └── deepseek_search_mcp.py       MCP server for the search service
```

Zero Python dependencies beyond the standard library. The HF CLI gets its own venv under `/srv/llm/venv`, and each service its own under `/srv/llm/services/<name>`.

## Hardware

The Radxa Orion O6 provides:

- **GPU** — Arm Immortalis G720 (Vulkan), used for inference
- **NPU** — available but not currently used for LLM workloads
- **TPU** — Coral M.2 Accelerator (if installed)
- **PCIe x16** — free slot for a discrete GPU

The installer auto-detects available hardware via `hardware.py` and configures accordingly. Extension points exist for future NPU/TPU/vision/speech capabilities.
