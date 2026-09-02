"""Install SPIRV-Headers (current ggml-vulkan requires them; Bookworm's are too old)."""

from pathlib import Path

from ..step import Step

# No vulkan-sdk-1.4.339.x tag exists for SPIRV-Headers; 1.4.335.0 is the
# closest tag to the Vulkan-Headers v1.4.339 pin in vulkan_headers.py.
SPIRV_HEADERS_TAG = "vulkan-sdk-1.4.335.0"
CMAKE_CONFIG = Path("/usr/local/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake")


class SpirvHeadersStep(Step):
    name = "spirv-headers"
    description = "Install SPIRV-Headers " + SPIRV_HEADERS_TAG

    def check(self) -> bool:
        return CMAKE_CONFIG.exists()

    def run(self) -> None:
        self.sh("rm -rf /tmp/SPIRV-Headers")
        self.sh(
            f"git clone --depth 1 --branch {SPIRV_HEADERS_TAG} "
            "https://github.com/KhronosGroup/SPIRV-Headers.git /tmp/SPIRV-Headers"
        )
        self.sh(
            "cmake -S /tmp/SPIRV-Headers -B /tmp/SPIRV-Headers/build "
            "-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local"
        )
        self.sh("cmake --build /tmp/SPIRV-Headers/build")
        self.sh("cmake --install /tmp/SPIRV-Headers/build")
        self.sh("rm -rf /tmp/SPIRV-Headers")
