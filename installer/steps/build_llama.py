"""Build llama.cpp from source."""

from pathlib import Path

from ..step import Step

LLAMA_DIR = Path("/srv/llm/src/llama.cpp")


class BuildLlamaStep(Step):
    name = "build-llama"
    description = "Build llama.cpp"

    @property
    def build_dir(self) -> Path:
        return LLAMA_DIR / self.backend.build_dir_name

    @property
    def server_bin(self) -> Path:
        return self.build_dir / "bin" / "llama-server"

    @property
    def build_hash(self) -> Path:
        return self.build_dir / ".build-hash"

    def _desired_ref(self) -> str:
        """Return the full commit hash for the desired build."""
        ref = self.config.llama_cpp_ref
        if ref == "latest":
            # Resolve origin's default branch.
            for branch in ("master", "main"):
                h = self.sh_output(f"git -C {LLAMA_DIR} rev-parse origin/{branch}")
                if h:
                    return h
            return ""
        # Pinned ref — resolve to full hash.
        return self.sh_output(f"git -C {LLAMA_DIR} rev-parse {ref}")

    def check(self) -> bool:
        if not self.server_bin.exists():
            return False
        if not LLAMA_DIR.exists():
            return False
        self.sh_ok(f"sudo -u llm git -C {LLAMA_DIR} fetch --quiet")
        built = self.build_hash.read_text().strip() if self.build_hash.exists() else ""
        desired = self._desired_ref()
        if not built or not desired or built != desired:
            return False
        return True

    def run(self) -> None:
        if not LLAMA_DIR.exists():
            self.sh_live(
                "sudo -u llm git clone https://github.com/ggml-org/llama.cpp "
                f"{LLAMA_DIR}"
            )
        else:
            self.sh_live(f"sudo -u llm git -C {LLAMA_DIR} fetch --all")

        # Checkout the desired ref.
        ref = self.config.llama_cpp_ref
        if ref == "latest":
            self.sh_live(f"sudo -u llm git -C {LLAMA_DIR} pull")
        else:
            self.sh(f"sudo -u llm git -C {LLAMA_DIR} checkout {ref}")
            print(f"   Pinned to {ref}")

        # Clean previous build
        if self.build_dir.exists():
            self.sh(f"rm -rf {self.build_dir}")

        cmake_flags = " ".join(
            f"{k}={v}" for k, v in self.backend.cmake_flags.items()
        )
        nproc = self.sh_output("nproc") or "4"
        self.sh_live(
            f"sudo -u llm bash -lc '"
            f"cmake -S {LLAMA_DIR} -B {self.build_dir} -G Ninja "
            f"-DCMAKE_BUILD_TYPE=Release "
            f"-DLLAMA_CURL=OFF "
            f"{cmake_flags} "
            f"&& cmake --build {self.build_dir} -j {nproc}'"
        )

        # Record which commit we built so check() can compare next run.
        head = self.sh_output(f"git -C {LLAMA_DIR} rev-parse HEAD")
        if head:
            self.build_hash.write_text(head + "\n")

        # Restart the service if it's running so it picks up the new binary.
        if self.sh_ok("systemctl is-active --quiet llama-router.service"):
            self.sh("systemctl restart llama-router.service")
            print("   Restarted llama-router.service")
