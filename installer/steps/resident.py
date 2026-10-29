"""Run resident profiles (those with a port) in their own always-on llama-server."""

import re
from pathlib import Path

from ..step import Step
from .systemd import MALI_ENV, SERVER_BIN

UNIT_DIR = Path("/etc/systemd/system")
MARKER = "# Managed by the inference installer: resident profile"


def unit_name(profile) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", profile.name.lower()).strip("-")
    return f"llama-{slug}.service"


def server_args(config, model, profile) -> list[str]:
    """llama-server flags for a resident profile: [defaults] overlaid by the profile.

    The router reads these as models.ini keys; a standalone server takes the
    same names as flags. A true value becomes a bare flag (--jinja) and a
    false one is left off, since llama-server's boolean flags are opt-in.
    """
    d = config.defaults
    opts = {
        "load-mode": d.load_mode,
        "parallel": d.parallel,
        "cache-type-k": d.cache_type_k,
        "cache-type-v": d.cache_type_v,
        "device": profile.device if profile.device is not None else d.device,
        "n-gpu-layers": profile.gpu_layers,
        "ctx-size": profile.ctx_size,
        "no-warmup": d.no_warmup,
        "cache-ram": d.cache_ram,
    }
    for key, val in profile.extra.items():
        opts[key.replace("_", "-")] = val

    args = [f"--model {model.first_gguf}", f"--alias {profile.name}"]
    for key, val in opts.items():
        if val is True:
            args.append(f"--{key}")
        elif val is not False:
            args.append(f"--{key} {val}")
    return args


def render_unit(config, model, profile, mali: bool = False) -> str:
    gpu_env = MALI_ENV if mali else ""
    flags = " \\\n  ".join(server_args(config, model, profile))
    return f"""\
{MARKER}
[Unit]
Description=llama.cpp resident model {profile.name} (port {profile.port})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=llm
Group=llm
WorkingDirectory=/srv/llm
Environment=HOME=/srv/llm
{gpu_env}
ExecStart={SERVER_BIN} \\
  --host {config.host} --port {profile.port} \\
  {flags}

Restart=on-failure
RestartSec=2
LimitNOFILE=1048576

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""


class ResidentStep(Step):
    name = "resident"
    description = "Install always-on llama-servers for resident profiles"

    def _desired(self) -> dict[str, str]:
        units = {}
        for model in self.config.models:
            for profile in model.profiles:
                if profile.resident:
                    units[unit_name(profile)] = render_unit(
                        self.config, model, profile, mali=self.hw.has_mali,
                    )
        return units

    def _stale(self, desired) -> list[Path]:
        if not UNIT_DIR.exists():
            return []
        return [
            p for p in sorted(UNIT_DIR.glob("llama-*.service"))
            if p.name not in desired and MARKER in p.read_text()
        ]

    def check(self) -> bool:
        desired = self._desired()
        if self._stale(desired):
            return False
        for name, unit in desired.items():
            path = UNIT_DIR / name
            if not (path.exists() and path.read_text() == unit):
                return False
            if not self.sh_ok(f"systemctl is-enabled --quiet {name}"):
                return False
            if not self.sh_ok(f"systemctl is-active --quiet {name}"):
                return False
        return True

    def run(self) -> None:
        desired = self._desired()
        for path in self._stale(desired):
            print(f"   Removing {path.name} (profile no longer resident)")
            self.sh(f"systemctl disable --now {path.name}")
            path.unlink()
        changed = set()
        for name, unit in desired.items():
            path = UNIT_DIR / name
            if not (path.exists() and path.read_text() == unit):
                path.write_text(unit)
                changed.add(name)
        self.sh("systemctl daemon-reload")
        for name in desired:
            self.sh(f"systemctl enable {name}")
            self.sh(f"systemctl {'restart' if name in changed else 'start'} {name}")
            print(f"   {name} enabled and running")
        if not desired:
            print("   No resident profiles")
