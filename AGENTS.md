# inference

Automated installer and model manager for running local LLM inference on any recent Debian or Ubuntu server using llama.cpp. Auto-detects GPU hardware and selects the best backend (Vulkan, CUDA, or CPU-only).

## How it works

A thin shell bootstrap (`install.sh`) ensures git and Python 3 exist, clones this repo, then hands off to a Python installer (`python3 -m installer`). The installer detects hardware, selects a GPU backend, and runs a sequence of idempotent steps — each has a `check()` that skips it if already done, and a `run()` that applies it.

`models.toml` is the single source of truth. It defines which models to download from Hugging Face, how to expose them as router profiles, and what defaults to apply. Both the download step and the `models.ini` generation step read from this one file. To add a model, add a TOML block. To remove one, delete the block and re-run with `--prune`.

## Architecture

```
install.sh → python3 -m installer
                ├── backend.py       backend detection + config (vulkan/cuda/cpu)
                ├── config.py        loads models.toml (stdlib tomllib)
                ├── hardware.py      detects GPU, NPU, Coral, PCIe
                ├── runner.py        iterates steps
                └── steps/           one module per step, executed in order
```

The installer has zero Python dependencies beyond the standard library. The Hugging Face CLI gets its own venv under `/srv/llm/venv`.

## Backends

`backend.py` separates detection (what hardware exists) from policy (what to use). Three backends are supported:

- **vulkan** — Arm Mali, AMD, Intel integrated GPUs via Vulkan render nodes
- **cuda** — NVIDIA GPUs with the proprietary driver loaded
- **cpu** — no GPU acceleration, pure CPU inference

Auto-detection priority: NVIDIA driver loaded -> cuda, Vulkan render nodes -> vulkan, otherwise -> cpu. Override with `--backend`.

Each backend defines: packages to install, cmake flags for llama.cpp, build directory name, default device string, and which steps to skip. Adding a new backend (e.g., ROCm) means adding one entry in `backend.py` — no step code changes needed.

### Mali G720 workaround

On Vulkan systems with an Arm Mali/Immortalis GPU, the backend automatically pins llama.cpp to commit `914dde72b` to avoid a Vulkan descriptor set regression (ggml-org/llama.cpp#21483). This pin does not apply to other Vulkan hardware or other backends.

## Design principles

- **Single source of truth** — `models.toml` drives both downloads and router configuration. No model-specific data lives in Python code.
- **Idempotent** — every step checks whether it's already been applied. Re-running the installer is safe and expected.
- **Minimal bootstrap** — the shell script is ~30 lines. Heavy lifting happens in Python where it's testable.
- **Zero external Python deps** — `tomllib` (stdlib 3.11+) for config, string formatting for templates. No pip install needed for the installer itself.
- **Backend-aware** — GPU-specific logic lives in `backend.py`. Steps consume `self.backend` generically.

## Models

Prefer `unsloth` quantizations when available. Unsloth provides well-optimized GGUF quantizations (particularly Q4_K_M) that balance quality against memory pressure. Check [unsloth's Hugging Face org](https://huggingface.co/unsloth) first before looking at other quantization sources.

For models that exceed available RAM (large MoE models like Qwen 3.5 397B or MiniMax M2.5), set `large = true` in the TOML block. The installer doesn't enforce specific behavior for this flag yet, but it signals that the model relies on mmap paging from NVMe rather than fitting in memory.

## Hardware

`hardware.py` detects available hardware passively:

- **GPU** — Vulkan render nodes via `/dev/dri`, device name via `vulkaninfo`
- **NVIDIA** — driver loaded via `/proc/driver/nvidia/version`, vendor via `lspci`
- **NPU** — Rockchip RKNPU (detected, not currently used for LLM inference)
- **TPU** — Google Coral M.2 Accelerator (detected via `/dev/apex_*`)
- **PCIe** — discrete GPU vendor (NVIDIA/AMD) via `lspci`

Steps and the backend system query the `Hardware` dataclass to adapt behavior.

## Adding a step

1. Create `installer/steps/your_step.py` with a class that inherits from `Step`
2. Set `name` and `description`
3. Implement `check()` (return True if already done) and `run()`
4. Use `self.backend` for any GPU/backend-specific logic
5. Add it to `ALL_STEPS` in `installer/steps/__init__.py`

## Adding a backend

1. Add a value to `BackendType` in `installer/backend.py`
2. Add a `_get_<name>_config()` function returning a `BackendConfig`
3. Add it to `_BACKEND_FACTORIES`
4. Add detection logic in `detect_backend()`
5. Add the string to `--backend` choices (automatic via the enum)

## File layout

```
install.sh                  Shell bootstrap
models.toml                 Model manifest (single source of truth)
installer/
  __main__.py               CLI entry point
  backend.py                Backend detection + configuration
  config.py                 TOML loader, dataclasses
  hardware.py               Hardware detection
  runner.py                 Step runner
  step.py                   Base class (config, hw, backend)
  steps/
    packages.py             apt packages (common + backend-specific)
    user.py                 llm user + directories
    gpu_permissions.py      render node access
    vulkan_headers.py       build newer Vulkan-Headers (Vulkan only)
    build_llama.py          build llama.cpp (backend-aware cmake flags)
    huggingface.py          HF CLI venv
    models.py               download / prune models
    router_config.py        generate models.ini
    systemd.py              llama-router.service (backend-aware binary path)
```
