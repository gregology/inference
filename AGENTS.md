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

Q4_K_M balances quality against memory pressure on the O6's 64GB unified RAM. For models that run on the GPU, prefer plain K-quant files (Q4_K/Q6_K tensors only): the Mali's IQ kernels are slow, and IQ4_XS-heavy quants (including unsloth's UD-* files) run ~35% slower there. Check a file's tensor types before adding it. Check [unsloth's Hugging Face org](https://huggingface.co/unsloth) first, but fall back to sources such as lmstudio-community for plain Q4_K_M. Models that are only available as IQ-heavy quants should run on the CPU (`device = "none"`, `gpu_layers = 0`, `threads = 8`, `threads_batch = 12`).

For models that exceed available RAM (large MoE models like GLM-5.3-Flash 320B or Qwen3.8-Flash-Next 125B), set `large = true` in the TOML block. The installer doesn't enforce specific behavior for this flag yet, but it signals that the model relies on mmap paging from NVMe rather than fitting in memory. Run such profiles on the CPU with `device = "none"`, `gpu_layers = 0` and `extra = { load_mode = "mmap", threads = 8, threads_batch = 12 }`: splitting experts (CPU) from the rest (GPU) is slower on the O6 than running everything on the 8 big cores, and more generation threads pull in the slower A520 cores.

Generation runs at 1-5 tok/s, so give thinking models a `reasoning_budget` sized for a few minutes of generation.

Before adding a GGUF, check that its `general.architecture` matches the name mainline llama.cpp registers. Quants made with an unmerged fork can carry a different name (e.g. `glm5next` vs mainline `glm5-next`) and will fail to load.

## Hardware

The Radxa Orion O6 provides:

- **GPU** — Arm Immortalis G720, exposed via Vulkan. Used for dense models with plain K-quants. ggml-vulkan needs a current glslc (the `shaderc` step) to use its int8 dot-product shaders, and the router runs with `GGML_VK_DISABLE_COOPMAT=1` and `GGML_VK_FORCE_MMVQ=1` when `hardware.py` detects a Mali. GPU profiles set `ubatch_size = 64`: ggml-vulkan's 64-wide matmul tile is the fastest on the Mali (13-19% faster 27B prompts than the default 512 ubatch). CPU profiles keep the default, since the CPU backend has no tile effect (Flash-Next: 8.01 vs 7.95 tok/s).
- **CPU** — 8x Cortex-A720 plus 4x Cortex-A520. Runs the large MoE models and IQ-heavy quants. Generate on the 8 big cores (`threads = 8`) and process prompts on all 12 (`threads_batch = 12`, 6-16% faster), but only in single-slot profiles: llama.cpp uses the batch threads whenever several slots generate at once, which slowed Flash-Next's two slots by 24%.
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
    spirv_headers.py        build SPIRV-Headers (ggml-vulkan dependency)
    shaderc.py              build glslc (Bookworm's lacks int8 dot-product support)
    build_llama.py          build llama.cpp
    huggingface.py          HF CLI venv
    models.py               download / prune models
    router_config.py        generate models.ini
    systemd.py              llama-router.service
```
