"""Build llama.cpp with Vulkan support."""

from pathlib import Path

from ..step import Step

LLAMA_DIR = Path("/srv/llm/src/llama.cpp")
BUILD_DIR = LLAMA_DIR / "build-vulkan"
SERVER_BIN = BUILD_DIR / "bin" / "llama-server"
BUILD_HASH = BUILD_DIR / ".build-hash"


class BuildLlamaStep(Step):
    name = "build-llama"
    description = "Build llama.cpp (Vulkan, curl disabled)"

    def _default_branch(self) -> str:
        for branch in ("master", "main"):
            if self.sh_ok(
                f"sudo -u llm git -C {LLAMA_DIR} rev-parse --verify --quiet origin/{branch}"
            ):
                return branch
        raise RuntimeError(f"Could not determine default branch for {LLAMA_DIR}")

    def _desired_ref(self) -> str:
        """Return the full commit hash for the desired build."""
        ref = self.config.llama_cpp_ref
        if ref == "latest":
            # Resolve origin's default branch.
            return self.sh_output(f"git -C {LLAMA_DIR} rev-parse origin/{self._default_branch()}")
        # Pinned ref — resolve to full hash.
        return self.sh_output(f"git -C {LLAMA_DIR} rev-parse {ref}")

    def check(self) -> bool:
        if not SERVER_BIN.exists():
            return False
        if not LLAMA_DIR.exists():
            return False
        self.sh_ok(f"sudo -u llm git -C {LLAMA_DIR} fetch --quiet")
        built = BUILD_HASH.read_text().strip() if BUILD_HASH.exists() else ""
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
            # A previous pinned build may have left the checkout in detached
            # HEAD, where `git pull` fails ("You are not currently on a
            # branch"). Reset the local default branch onto origin's instead —
            # this works from detached HEAD or any stale branch state.
            branch = self._default_branch()
            self.sh_live(f"sudo -u llm git -C {LLAMA_DIR} checkout -B {branch} origin/{branch}")
        else:
            self.sh(f"sudo -u llm git -C {LLAMA_DIR} checkout {ref}")
            print(f"   Pinned to {ref}")

        # Clean previous build
        if BUILD_DIR.exists():
            self.sh(f"rm -rf {BUILD_DIR}")

        nproc = self.sh_output("nproc") or "4"
        self.sh_live(
            f"sudo -u llm bash -lc '"
            f"cmake -S {LLAMA_DIR} -B {BUILD_DIR} -G Ninja "
            f"-DCMAKE_BUILD_TYPE=Release "
            f"-DLLAMA_CURL=OFF "
            f"-DGGML_VULKAN=ON "
            f"&& cmake --build {BUILD_DIR} -j {nproc}'"
        )

        # Record which commit we built so check() can compare next run.
        head = self.sh_output(f"git -C {LLAMA_DIR} rev-parse HEAD")
        if head:
            BUILD_HASH.write_text(head + "\n")

        # Restart the service if it's running so it picks up the new binary.
        if self.sh_ok("systemctl is-active --quiet llama-router.service"):
            self.sh("systemctl restart llama-router.service")
            print("   Restarted llama-router.service")
