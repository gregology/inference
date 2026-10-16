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
                huggingface, models, router-config, systemd)
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
ctx_size = 65536
gpu_layers = 999
extra = { reasoning_budget = 1024 }
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
ctx_size = 65536
device = "none"
gpu_layers = 0
extra = { load_mode = "mmap", threads = 8, ctx_checkpoints = 8, reasoning_budget = 2048 }
```

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

The router loads models on demand. With `--models-max 1`, requesting a different model unloads the current one first.

## Performance on the O6

Measured with llama.cpp 620d2225c and Mali driver r53p0 (tokens/s; prompt measured on 512–2048-token prompts, generation at short context):

| Profile | Runs on | Prompt | Generation |
|---|---|---|---|
| `qwen3.8:27b`, `qwen3.8-uncensored:27b` | GPU | ~3.9 | ~2.4 |
| `orcasaq-2-cyber:27b` | CPU | ~2.5 | ~2.1 |
| `qwen3.8-flash-next-uncensored:125b` | CPU | ~7–8 | ~5.0 |
| `glm-5.3-flash-uncensored:320b` | CPU | ~2 | ~1.0 |

Prompt processing dominates: a 4K-token prompt takes ~19 minutes on the 27B models. The router's prompt cache (`cache_ram`) makes follow-up turns cheap: only new tokens are processed. Concurrent requests don't add much total throughput (CPU and GPU both top out at ~40 GB/s of memory bandwidth and are compute-bound), so each extra request roughly divides the speed. Models serve one request at a time (`parallel = 1` in `[defaults]`) and queue the rest; Flash-Next has two slots (`extra = { parallel = 2 }`), so two requests each get ~2.9 tok/s.

What made the difference:

- **glslc from shaderc.** Bookworm's `glslc` is too old for ggml-vulkan's int8 dot-product shaders, which the Mali supports. The `shaderc` step builds a current one.
- **Mali environment.** The router unit sets `GGML_VK_DISABLE_COOPMAT=1` (the driver's cooperative-matrix path is ~4x slower than int8 dot-product) and `GGML_VK_FORCE_MMVQ=1` (~20% faster generation).
- **Plain K-quants on the GPU.** IQ4_XS-heavy files (unsloth's UD quants, OrcaSAQ) are ~35% slower on the Mali.
- **CPU for MoE.** Splitting experts (CPU) from the rest (GPU) is slower than running the whole model on the 8 big cores.
- **Not worth it:** MTP speculative decoding (net slower), f16 KV cache, larger batch sizes, more than 8 threads.

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
            └── systemd.py         llama-router.service
```

Zero Python dependencies beyond the standard library. The HF CLI gets its own venv under `/srv/llm/venv`.

## Hardware

The Radxa Orion O6 provides:

- **GPU** — Arm Immortalis G720 (Vulkan), used for inference
- **NPU** — available but not currently used for LLM workloads
- **TPU** — Coral M.2 Accelerator (if installed)
- **PCIe x16** — free slot for a discrete GPU

The installer auto-detects available hardware via `hardware.py` and configures accordingly. Extension points exist for future NPU/TPU/vision/speech capabilities.
