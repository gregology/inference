# inference

Turn any recent Debian or Ubuntu server into a local LLM inference server. Auto-detects GPU hardware and builds llama.cpp with the right backend (Vulkan, CUDA, or CPU-only).

I built this for my Radax Orion O6 with 64Gb of RAM but it might work for you ¯\\\_(ツ)\_/¯

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/gregology/inference/refs/heads/main/install.sh | sudo bash
```

The installer auto-detects your GPU and selects the best backend. To override:

```bash
curl -fsSL ... | sudo bash -s -- --backend cuda
```

This will:

1. Install system packages (build tools, GPU libraries for your backend)
2. Create a dedicated `llm` system user under `/srv/llm`
3. Grant GPU render-node access (if applicable)
4. Build llama.cpp from source with the detected backend
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
--backend B     GPU backend: vulkan, cuda, cpu (default: auto-detect)
--prune         Remove model files for entries no longer in models.toml
--dry-run       Show what would be done without making changes
--only STEP     Run a single step (packages, user, gpu-permissions,
                vulkan-headers, build-llama, huggingface, models,
                router-config, systemd)
--host HOST     Listen address (default: 0.0.0.0)
--port PORT     Listen port (default: 8000)
--models-toml   Path to models.toml (default: repo's models.toml)
```

## models.toml

The model manifest. Each model has a Hugging Face source and one or more router profiles:

```toml
[models.glm-4_7-flash]
repo = "unsloth/GLM-4.7-Flash-GGUF"
file = "GLM-4.7-Flash-Q4_K_M.gguf"

[models.glm-4_7-flash.profiles.gpu]
name = "glm-4.7-flash:30b"
ctx_size = 65536
gpu_layers = 999
extra = { jinja = true, temp = 0.7 }

[models.glm-4_7-flash.profiles.cpu]
name = "glm-4.7-flash:30b-cpu"
ctx_size = 65536
gpu_layers = 0
device = "none"
extra = { jinja = true, temp = 0.7 }
```

For models that exceed available RAM (large MoE models), use `include` instead of `file` and set `large = true`:

```toml
[models.qwen3_5-397b]
repo = "unsloth/Qwen3.5-397B-A17B-GGUF"
include = "Q4_K_M/*"
large = true
```

## Usage

Once running, use the OpenAI-compatible API:

```bash
curl -s http://YOUR_IP:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-4.7-flash:30b",
    "messages": [{"role":"user","content":"Hello!"}]
  }' | jq .
```

List available models:

```bash
curl -s http://YOUR_IP:8000/v1/models | jq '.data[].id'
```

The router loads models on demand. With `--models-max 1`, requesting a different model unloads the current one first.

## Tests

```bash
# All tests (including HF network checks)
python3 -m pytest tests/ -v

# Offline only (fast, no network)
python3 -m pytest tests/test_backend.py tests/test_config.py tests/test_router_config.py -v

# HF manifest validation only
python3 -m pytest tests/test_huggingface.py -v
```

The HF tests verify that every repo, file, and include pattern in `models.toml` actually exists on Hugging Face. Run these after editing the manifest to catch typos before deploying.

## Architecture

```
install.sh            Shell bootstrap (ensures git + python3, clones repo)
  └─ python3 -m installer
       ├── backend.py     Backend detection and configuration (vulkan/cuda/cpu)
       ├── config.py      Load + validate models.toml (stdlib tomllib)
       ├── hardware.py    Detect GPU, NPU, Coral TPU, PCIe devices
       ├── runner.py      Iterate steps, skip completed, report errors
       └── steps/
            ├── packages.py        apt packages (common + backend-specific)
            ├── user.py            llm user + dirs
            ├── gpu_permissions.py render node access
            ├── vulkan_headers.py  build newer headers (Vulkan only)
            ├── build_llama.py     cmake llama.cpp (backend-aware)
            ├── huggingface.py     HF CLI venv
            ├── models.py          download / prune models
            ├── router_config.py   generate models.ini
            └── systemd.py         llama-router.service

installer/backend.py defines three backends:
  vulkan — Arm Mali, AMD, Intel integrated GPUs via Vulkan
  cuda   — NVIDIA GPUs with the proprietary driver
  cpu    — no GPU acceleration
```

Zero Python dependencies beyond the standard library. The HF CLI gets its own venv under `/srv/llm/venv`.

## Backends

The installer auto-detects your hardware and selects the best backend:

| Priority | Condition | Backend |
|----------|-----------|---------|
| 1 | NVIDIA driver loaded | `cuda` |
| 2 | Vulkan render nodes present | `vulkan` |
| 3 | Fallback | `cpu` |

Override with `--backend vulkan|cuda|cpu`. Each backend determines which packages to install, which cmake flags to pass to llama.cpp, and which installer steps to skip.

Adding a new backend (e.g., ROCm for AMD) requires only adding a new entry in `installer/backend.py` — no step code changes needed.

## Hardware

The installer detects available hardware via `hardware.py`:

- **GPU** — Vulkan render nodes, NVIDIA driver, PCIe GPU vendor
- **NPU** — Rockchip RKNPU (detected, not yet used for LLM workloads)
- **TPU** — Google Coral M.2 Accelerator (detected if installed)
- **PCIe** — discrete NVIDIA or AMD GPUs via lspci

Originally built for the [Radxa Orion O6](https://radxa.com/products/orion/o6/) (Arm Immortalis G720, 64GB unified RAM) but works on any Debian/Ubuntu server with or without a GPU.
