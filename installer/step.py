"""Base class for installer steps."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import BackendConfig
    from .config import Config
    from .hardware import Hardware


class Step:
    """A single idempotent installer step.

    Subclasses must set `name` and `description` and implement
    `check()` and `run()`.
    """

    name: str = ""
    description: str = ""

    def __init__(self, config: Config, hw: Hardware, backend: BackendConfig) -> None:
        self.config = config
        self.hw = hw
        self.backend = backend

    def check(self) -> bool:
        """Return True if this step has already been applied."""
        return False

    def run(self) -> None:
        """Apply this step. May raise on failure."""
        raise NotImplementedError

    # ── helpers ──────────────────────────────────────────────────

    def sh(self, cmd: str, **kwargs) -> subprocess.CompletedProcess:
        """Run a shell command, raising on failure."""
        return subprocess.run(
            cmd, shell=True, check=True,
            text=True, capture_output=True,
            **kwargs,
        )

    def sh_ok(self, cmd: str) -> bool:
        """Return True if a shell command exits 0."""
        try:
            subprocess.run(
                cmd, shell=True, check=True,
                text=True, capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def sh_output(self, cmd: str) -> str:
        """Return stdout of a shell command, or empty string on failure."""
        try:
            r = subprocess.run(
                cmd, shell=True, check=True,
                text=True, capture_output=True,
            )
            return r.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    def sh_live(self, cmd: str, **kwargs) -> None:
        """Run a shell command with live stdout/stderr (no capture)."""
        subprocess.run(cmd, shell=True, check=True, **kwargs)
