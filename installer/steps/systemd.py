"""Install and enable the llama-router systemd service."""

from pathlib import Path

from ..step import Step

SERVICE = "llama-router.service"
SERVICE_PATH = Path("/etc/systemd/system") / SERVICE
SERVER_BIN = "/srv/llm/src/llama.cpp/build-vulkan/bin/llama-server"
MODELS_INI = "/srv/llm/etc/models.ini"


# Mali tuning for ggml-vulkan. The driver's cooperative-matrix support is
# ~4x slower for matmuls than ggml's int8 dot-product shaders, and the int8
# mat-vec kernel speeds up generation ~20%. Child servers spawned by the
# router inherit these.
MALI_ENV = """\
Environment=GGML_VK_DISABLE_COOPMAT=1
Environment=GGML_VK_FORCE_MMVQ=1
"""


def render_unit(host: str, port: int, mali: bool = False) -> str:
    gpu_env = MALI_ENV if mali else ""
    return f"""\
[Unit]
Description=llama.cpp router (multi-model, Vulkan)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=llm
Group=llm
WorkingDirectory=/srv/llm
Environment=HOME=/srv/llm
Environment=HF_HOME=/srv/llm/cache/huggingface
{gpu_env}
ExecStart={SERVER_BIN} \\
  --host {host} --port {port} \\
  --models-preset {MODELS_INI} \\
  --models-max 1 \\
  --parallel 1 \\
  --timeout 1800

Restart=on-failure
RestartSec=2
LimitNOFILE=1048576

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""


class SystemdStep(Step):
    name = "systemd"
    description = "Install and enable llama-router service"

    def _unit(self) -> str:
        return render_unit(self.config.host, self.config.port, mali=self.hw.has_mali)

    def _unit_current(self) -> bool:
        return SERVICE_PATH.exists() and SERVICE_PATH.read_text() == self._unit()

    def check(self) -> bool:
        # A current unit is not enough: the service may have been stopped by
        # hand, and an install run should leave the router up.
        return (
            self._unit_current()
            and self.sh_ok(f"systemctl is-enabled --quiet {SERVICE}")
            and self.sh_ok(f"systemctl is-active --quiet {SERVICE}")
        )

    def run(self) -> None:
        changed = not self._unit_current()
        if changed:
            SERVICE_PATH.write_text(self._unit())
            self.sh("systemctl daemon-reload")
        self.sh(f"systemctl enable {SERVICE}")
        # restart, not start: a running service only picks up unit changes
        # (e.g. new Environment= lines) on restart.
        self.sh(f"systemctl {'restart' if changed else 'start'} {SERVICE}")
        print("   Service enabled and running")
        print("   Logs: journalctl -u llama-router.service -f")
