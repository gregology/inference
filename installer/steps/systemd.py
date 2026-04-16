"""Install and enable the llama-router systemd service."""

from pathlib import Path

from ..step import Step

SERVICE_PATH = Path("/etc/systemd/system/llama-router.service")
MODELS_INI = "/srv/llm/etc/models.ini"


def render_unit(host: str, port: int, server_bin: str, backend_label: str) -> str:
    return f"""\
[Unit]
Description=llama.cpp router (multi-model, {backend_label})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=llm
Group=llm
WorkingDirectory=/srv/llm
Environment=HOME=/srv/llm
Environment=HF_HOME=/srv/llm/cache/huggingface

ExecStart={server_bin} \\
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

    @property
    def server_bin(self) -> str:
        return f"/srv/llm/src/llama.cpp/{self.backend.build_dir_name}/bin/llama-server"

    def _render(self) -> str:
        return render_unit(
            self.config.host, self.config.port,
            self.server_bin, self.backend.type.value,
        )

    def check(self) -> bool:
        if not SERVICE_PATH.exists():
            return False
        # Check if the unit content matches what we'd generate.
        return SERVICE_PATH.read_text() == self._render()

    def run(self) -> None:
        SERVICE_PATH.write_text(self._render())
        self.sh("systemctl daemon-reload")
        self.sh("systemctl enable --now llama-router.service")
        print("   Service enabled and started")
        print("   Logs: journalctl -u llama-router.service -f")
