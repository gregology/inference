# inference

Automated installer and model manager for running local LLM inference on a Radxa Orion O6 (or any Debian Bookworm machine) using llama.cpp with Vulkan GPU offload.

## How it works

A thin shell bootstrap (`install.sh`) ensures git and Python 3 exist, clones this repo, then hands off to a Python installer (`python3 -m installer`). The installer runs a sequence of idempotent steps — each has a `check()` that skips it if already done, and a `run()` that applies it.

`models.toml` is the single source of truth. It defines which models to download from Hugging Face, how to expose them as router profiles, and what defaults to apply. Both the download step and the `models.ini` generation step read from this one file. To add a model, add a TOML block. To remove one, delete the block and re-run with `--prune`.

## Architecture

```
install.sh → python3 -m installer
                ├── config.py        loads models.toml (stdlib tomllib)
                ├── hardware.py      detects GPU, NPU, Coral, PCIe
                ├── runner.py        iterates steps
                └── steps/           one module per step, executed in order
```

The installer has zero Python dependencies beyond the standard library. The Hugging Face CLI gets its own venv under `/srv/llm/venv`.

## Design principles

- **Single source of truth** — `models.toml` drives both downloads and router configuration. No model-specific data lives in Python code.
- **Idempotent** — every step checks whether it's already been applied. Re-running the installer is safe and expected.
- **Minimal bootstrap** — the shell script is ~30 lines. Heavy lifting happens in Python where it's testable.
- **Zero external Python deps** — `tomllib` (stdlib 3.11+) for config, string formatting for templates. No pip install needed for the installer itself.

## Models

Prefer `unsloth` quantizations when available. Unsloth provides well-optimized GGUF quantizations (particularly Q4_K_M) that balance quality against memory pressure on the O6's 64GB unified RAM. Check [unsloth's Hugging Face org](https://huggingface.co/unsloth) first before looking at other quantization sources.

For models that exceed available RAM (large MoE models like Qwen 3.5 397B or MiniMax M2.5), set `large = true` in the TOML block. The installer doesn't enforce specific behavior for this flag yet, but it signals that the model relies on mmap paging from NVMe rather than fitting in memory.

## Hardware

The Radxa Orion O6 provides:

- **GPU** — Arm Immortalis G720, exposed via Vulkan. This is the primary inference accelerator.
- **NPU** — Rockchip RKNPU. Not currently used for LLM inference but detected by `hardware.py`.
- **TPU** — Google Coral M.2 Accelerator (if installed in the free M.2 slot). Detected via `/dev/apex_*`.
- **PCIe x16** — free slot that accepts a full-sized discrete GPU.

`hardware.py` detects all of these. Steps and future extensions can query the `Hardware` dataclass to adapt behavior.

## Adding a step

1. Create `installer/steps/your_step.py` with a class that inherits from `Step`
2. Set `name` and `description`
3. Implement `check()` (return True if already done) and `run()`
4. Add it to `ALL_STEPS` in `installer/steps/__init__.py`

## File layout

```
install.sh                  Shell bootstrap
models.toml                 Model manifest (single source of truth)
installer/
  __main__.py               CLI entry point
  config.py                 TOML loader, dataclasses
  hardware.py               Hardware detection
  runner.py                 Step runner
  step.py                   Base class
  steps/
    packages.py             apt packages
    user.py                 llm user + directories
    gpu_permissions.py      render node access
    vulkan_headers.py       build newer Vulkan-Headers
    build_llama.py          build llama.cpp
    huggingface.py          HF CLI venv
    models.py               download / prune models
    router_config.py        generate models.ini
    systemd.py              llama-router.service
```
