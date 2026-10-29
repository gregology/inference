"""Install the Python network services listed under [services] in models.toml.

Each service gets a venv, a data dir and a systemd unit under
/srv/llm/services/<name>, and runs as the llm user.
"""

from pathlib import Path

from ..step import Step
from .models import HF_BIN, HF_HOME

UNIT_DIR = Path("/etc/systemd/system")
MARKER = "# Managed by the inference installer: service"


def unit_name(service) -> str:
    return f"inference-{service.name}.service"


def _systemd_quote(arg: str) -> str:
    # ExecStart expands % specifiers and $VARS; neither belongs in our args.
    arg = arg.replace("%", "%%").replace("$", "$$")
    if not arg or any(c in arg for c in " \t\"'\\"):
        arg = '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return arg


def exec_args(service) -> list[str]:
    """The service's command line, with {port}, {data} and {download} filled in."""
    values = {"port": service.port, "data": service.data_dir, "download": service.download_dir or ""}
    args = [a.format(**values) for a in service.args]
    python = f"{service.root}/venv/bin/python"
    if service.module:
        return [python, "-m", service.module, *args]
    return [python, f"{service.root}/{Path(service.script).name}", *args]


def _command_lines(args: list[str]) -> list[str]:
    """Group the command line one option (with its value) per line."""
    lines = []
    for arg in map(_systemd_quote, args):
        if lines and not arg.startswith("-") and lines[-1].startswith("-") and " " not in lines[-1]:
            lines[-1] += f" {arg}"
        else:
            lines.append(arg)
    return lines


def render_unit(service) -> str:
    env_file = f"EnvironmentFile={service.env_file}\n" if service.env_file else ""
    cmd = " \\\n  ".join(_command_lines(exec_args(service)))
    return f"""\
{MARKER}
[Unit]
Description=inference service {service.name} ({service.package})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=llm
Group=llm
WorkingDirectory={service.root}
Environment=HOME=/srv/llm
Environment=HF_HOME={HF_HOME}
{env_file}ExecStart={cmd}

Restart=on-failure
RestartSec=5

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""


class ServicesStep(Step):
    name = "services"
    description = "Install Python network services (speech, tools)"

    def _script_source(self, service) -> Path:
        return self.config.repo_dir / service.script

    def _current(self, service) -> bool:
        root = Path(service.root)
        unit = UNIT_DIR / unit_name(service)
        marker = root / ".package"
        if not (unit.exists() and unit.read_text() == render_unit(service)):
            return False
        if not (marker.exists() and marker.read_text() == service.package):
            return False
        if service.script:
            installed = root / Path(service.script).name
            if not (installed.exists()
                    and installed.read_bytes() == self._script_source(service).read_bytes()):
                return False
        if service.download_dir and not any(Path(service.download_dir).iterdir()
                                            if Path(service.download_dir).exists() else []):
            return False
        return True

    def _stale(self) -> list[Path]:
        desired = {unit_name(s) for s in self.config.services}
        if not UNIT_DIR.exists():
            return []
        return [
            p for p in sorted(UNIT_DIR.glob("inference-*.service"))
            if p.name not in desired and MARKER in p.read_text()
        ]

    def check(self) -> bool:
        if self._stale():
            return False
        for service in self.config.services:
            name = unit_name(service)
            if not self._current(service):
                return False
            if not self.sh_ok(f"systemctl is-enabled --quiet {name}"):
                return False
            if not self.sh_ok(f"systemctl is-active --quiet {name}"):
                return False
        return True

    def _install(self, service) -> None:
        root = Path(service.root)
        venv = root / "venv"
        Path(service.data_dir).mkdir(parents=True, exist_ok=True)
        self.sh(f"chown -R llm:llm {root}")
        if not venv.exists():
            self.sh(f"sudo -u llm python3 -m venv {venv}")
        marker = root / ".package"
        if not (marker.exists() and marker.read_text() == service.package):
            print(f"   Installing {service.package} ...")
            self.sh(f"sudo -u llm {venv}/bin/pip install --upgrade pip '{service.package}'")
            marker.write_text(service.package)
        if service.script:
            installed = root / Path(service.script).name
            installed.write_bytes(self._script_source(service).read_bytes())
        if service.download_dir:
            target = Path(service.download_dir)
            if not (target.exists() and any(target.iterdir())):
                # --local-dir writes real files: onnxruntime refuses external
                # data reached through the HF cache's blob symlinks.
                print(f"   Downloading {service.download} ...")
                self.sh_live(
                    f"sudo -u llm env HF_HOME={HF_HOME} {HF_BIN} download "
                    f"{service.download} --local-dir {target}"
                )
        self.sh(f"chown -R llm:llm {root}")

    def run(self) -> None:
        for path in self._stale():
            print(f"   Removing {path.name} (no longer in models.toml)")
            self.sh(f"systemctl disable --now {path.name}")
            path.unlink()

        missing_env = []
        changed = set()
        for service in self.config.services:
            name = unit_name(service)
            if not self._current(service):
                self._install(service)
                unit = UNIT_DIR / name
                if not (unit.exists() and unit.read_text() == render_unit(service)):
                    unit.write_text(render_unit(service))
                changed.add(name)
            if service.env_file and not Path(service.env_file).exists():
                missing_env.append(service)
        self.sh("systemctl daemon-reload")

        for service in self.config.services:
            name = unit_name(service)
            if service in missing_env:
                print(f"   {name}: not started, {service.env_file} is missing")
                continue
            self.sh(f"systemctl enable {name}")
            self.sh(f"systemctl {'restart' if name in changed else 'start'} {name}")
            print(f"   {name} enabled and running on port {service.port}")

        if missing_env:
            raise RuntimeError(
                "missing env file(s): "
                + ", ".join(f"{s.env_file} (for {s.name})" for s in missing_env)
                + " — create them (owner llm, mode 0600) and re-run"
            )
