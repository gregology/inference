"""GPU backend detection and configuration.

hardware.py detects what is present; this module decides what to use.
Each BackendType maps to a BackendConfig that tells steps which packages
to install, which cmake flags to pass, and which steps to skip.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hardware import Hardware


class BackendType(enum.Enum):
    VULKAN = "vulkan"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass
class BackendConfig:
    type: BackendType

    # Packages to install beyond the common base set.
    packages: list[str] = field(default_factory=list)

    # cmake flags for the llama.cpp build (e.g. {"-DGGML_VULKAN": "ON"}).
    cmake_flags: dict[str, str] = field(default_factory=dict)

    # Build directory name under /srv/llm/src/llama.cpp/.
    build_dir_name: str = "build"

    # Default device string for models.ini (e.g. "Vulkan0", "CUDA0", "none").
    default_device: str = "none"

    # Step names to skip entirely for this backend.
    skip_steps: set[str] = field(default_factory=set)

    # If set, overrides models.toml [build] llama_cpp_ref.
    llama_cpp_ref_override: str | None = None


# ── Mali G720 Vulkan regression pin ────────────────────────────────
# Builds after 914dde72b trigger a Vulkan descriptor set assertion on
# the Mali G720 Immortalis GPU.  Tracked: ggml-org/llama.cpp#21483
_MALI_LLAMA_CPP_PIN = "914dde72b"


def _is_mali_gpu(hw: Hardware) -> bool:
    """Return True if the detected GPU is a Mali / Arm Immortalis."""
    if not hw.gpu_device:
        return False
    lower = hw.gpu_device.lower()
    return "mali" in lower or "immortalis" in lower


def _get_vulkan_config(hw: Hardware) -> BackendConfig:
    return BackendConfig(
        type=BackendType.VULKAN,
        packages=[
            "libvulkan1", "libvulkan-dev", "vulkan-tools",
            "mesa-vulkan-drivers", "glslc",
        ],
        cmake_flags={"-DGGML_VULKAN": "ON"},
        build_dir_name="build-vulkan",
        default_device="Vulkan0",
        skip_steps=set(),
        llama_cpp_ref_override=_MALI_LLAMA_CPP_PIN if _is_mali_gpu(hw) else None,
    )


def _get_cuda_config(hw: Hardware) -> BackendConfig:
    return BackendConfig(
        type=BackendType.CUDA,
        packages=["nvidia-cuda-toolkit"],
        cmake_flags={"-DGGML_CUDA": "ON"},
        build_dir_name="build-cuda",
        default_device="CUDA0",
        skip_steps={"vulkan-headers"},
    )


def _get_cpu_config(hw: Hardware) -> BackendConfig:
    return BackendConfig(
        type=BackendType.CPU,
        packages=[],
        cmake_flags={},
        build_dir_name="build-cpu",
        default_device="none",
        skip_steps={"vulkan-headers", "gpu-permissions"},
    )


_BACKEND_FACTORIES = {
    BackendType.VULKAN: _get_vulkan_config,
    BackendType.CUDA: _get_cuda_config,
    BackendType.CPU: _get_cpu_config,
}


def get_backend_config(backend_type: BackendType, hw: Hardware) -> BackendConfig:
    """Return the fully populated config for the given backend."""
    return _BACKEND_FACTORIES[backend_type](hw)


def detect_backend(hw: Hardware) -> BackendType:
    """Auto-detect the best backend from the hardware.

    Priority:
      1. NVIDIA GPU with driver loaded  -> CUDA
      2. Vulkan render nodes present     -> VULKAN
      3. Otherwise                       -> CPU
    """
    if hw.nvidia_driver_loaded:
        return BackendType.CUDA

    if hw.render_nodes:
        return BackendType.VULKAN

    return BackendType.CPU
