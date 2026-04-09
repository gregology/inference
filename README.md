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
4. Build Vulkan-Headers and llama.cpp from source
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
python3 -m pytest tests/test_config.py tests/test_router_config.py -v

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
