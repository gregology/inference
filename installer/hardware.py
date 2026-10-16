"""Detect available hardware on the host."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Hardware:
    gpu_device: str | None = None
    has_mali: bool = False
    render_nodes: list[str] = field(default_factory=list)
    render_group: str = "render"
    has_npu: bool = False
    has_coral: bool = False
    has_pcie_gpu: bool = False
    cpu_count: int = 4
    cpu_arch: str | None = None


# GCC extension for each /proc/cpuinfo feature that ggml's Arm kernels use.
ARM_EXTENSIONS = {
    "asimdhp": "fp16",
    "asimddp": "dotprod",
    "i8mm": "i8mm",
    "bf16": "bf16",
}


def arm_march(cpuinfo: str) -> str | None:
    """Return a -march value with the CPU's Arm extensions, or None off Arm.

    llama.cpp's GGML_NATIVE probes each extension with -mcpu=native+<ext>.
    GCC 12 rejects that for CPUs it doesn't know, such as the O6's
    Cortex-A720, so every probe fails and the CPU backend is built for
    plain armv8-a, without the dot-product and int8 matmul kernels.
    """
    for line in cpuinfo.splitlines():
        if line.startswith("Features"):
            flags = set(line.split(":", 1)[1].split())
            break
    else:
        return None
    exts = [ext for flag, ext in ARM_EXTENSIONS.items() if flag in flags]
    # fp16 and dot product need an Armv8.2-A base; older cores get plain armv8-a.
    base = "armv8.2-a" if exts else "armv8-a"
    return "+".join([base, *exts])


def detect_hardware() -> Hardware:
    hw = Hardware()

    # CPU count
    try:
        import os
        hw.cpu_count = os.cpu_count() or 4
    except Exception:
        pass

    # Arm extensions for the llama.cpp build
    try:
        hw.cpu_arch = arm_march(Path("/proc/cpuinfo").read_text())
    except OSError:
        pass

    # Render nodes
    dri = Path("/dev/dri")
    if dri.exists():
        hw.render_nodes = sorted(str(p) for p in dri.glob("renderD*"))
        if hw.render_nodes:
            # Determine group ownership
            try:
                import stat
                import grp
                st = Path(hw.render_nodes[0]).stat()
                hw.render_group = grp.getgrgid(st.st_gid).gr_name
            except Exception:
                pass

    # GPU device via vulkaninfo
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "deviceName" in line:
                hw.gpu_device = line.split("=")[-1].strip()
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Arm Mali (kbase driver). Checked via sysfs because vulkaninfo can
    # crash on headless boards that ship several Vulkan ICDs.
    hw.has_mali = (
        Path("/sys/class/misc/mali0").exists()
        or "mali" in (hw.gpu_device or "").lower()
    )

    # Coral TPU. /dev/apex_* only exists once the out-of-tree gasket driver
    # is installed, so also look for the card itself on the PCIe bus.
    hw.has_coral = Path("/dev/apex_0").exists() or bool(list(Path("/dev").glob("apex_*")))

    # NPU — Rockchip RKNPU, or the CIX P1's Arm China Zhouyi (aipu driver)
    hw.has_npu = (
        Path("/dev/rknpu").exists()
        or Path("/sys/class/misc/rknpu").exists()
        or Path("/dev/aipu").exists()
    )

    # Discrete PCIe GPU (beyond the integrated Mali)
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            lower = line.lower()
            if "coral edge tpu" in lower:
                hw.has_coral = True
            if ("vga" in lower or "3d controller" in lower or "display" in lower):
                if "nvidia" in lower or "amd" in lower or "radeon" in lower:
                    hw.has_pcie_gpu = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return hw
