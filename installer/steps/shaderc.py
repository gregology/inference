"""Build glslc from shaderc (Bookworm's is too old for ggml-vulkan's fast paths)."""

from pathlib import Path

from ..step import Step

# Bookworm's glslc (shaderc 2023.2) predates GL_EXT_integer_dot_product and
# GL_KHR_cooperative_matrix. ggml-vulkan then silently compiles without its
# int8 dot-product matmul shaders, even though the O6's Mali driver supports
# them; on the O6 those shaders make prompt processing ~5x and generation
# ~2.5x faster than the fallback.
SHADERC_TAG = "v2026.4"
GLSLC_PATH = Path("/usr/local/bin/glslc")
SRC_DIR = "/tmp/shaderc"


class ShadercStep(Step):
    name = "shaderc"
    description = "Build glslc from shaderc " + SHADERC_TAG

    def check(self) -> bool:
        if not GLSLC_PATH.exists():
            return False
        # First line looks like "shaderc v2026.4 v2026.4".
        version = self.sh_output(f"{GLSLC_PATH} --version").split("\n")[0]
        return f"shaderc {SHADERC_TAG} " in f"{version} "

    def run(self) -> None:
        self.sh(f"rm -rf {SRC_DIR}")
        self.sh(
            f"git clone --depth 1 --branch {SHADERC_TAG} "
            f"https://github.com/google/shaderc.git {SRC_DIR}"
        )
        # Fetches glslang, SPIRV-Tools and SPIRV-Headers at the commits the
        # tag was released with.
        self.sh_live(f"python3 {SRC_DIR}/utils/git-sync-deps")
        self.sh(
            f"cmake -S {SRC_DIR} -B {SRC_DIR}/build -G Ninja "
            "-DCMAKE_BUILD_TYPE=Release "
            "-DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON "
            "-DSHADERC_SKIP_COPYRIGHT_CHECK=ON "
            "-DSPIRV_SKIP_TESTS=ON -DSPIRV_SKIP_EXECUTABLES=ON "
            "-DENABLE_GLSLANG_BINARIES=OFF"
        )
        # Only the glslc binary is needed; it links everything statically.
        self.sh_live(f"cmake --build {SRC_DIR}/build --target glslc_exe")
        self.sh(f"install -m 0755 {SRC_DIR}/build/glslc/glslc {GLSLC_PATH}")
        self.sh(f"rm -rf {SRC_DIR}")
