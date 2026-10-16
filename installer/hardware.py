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


def detect_hardware() -> Hardware:
    hw = Hardware()

    # CPU count
    try:
        import os
        hw.cpu_count = os.cpu_count() or 4
    except Exception:
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

    # Coral TPU
    hw.has_coral = Path("/dev/apex_0").exists() or bool(list(Path("/dev").glob("apex_*")))

    # NPU — Rockchip RKNPU or generic
    hw.has_npu = (
        Path("/dev/rknpu").exists()
        or Path("/sys/class/misc/rknpu").exists()
    )

    # Discrete PCIe GPU (beyond the integrated Mali)
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            lower = line.lower()
            if ("vga" in lower or "3d controller" in lower or "display" in lower):
                if "nvidia" in lower or "amd" in lower or "radeon" in lower:
                    hw.has_pcie_gpu = True
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return hw
