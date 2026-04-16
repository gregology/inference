"""Install system packages."""

from ..step import Step

COMMON_PACKAGES = [
    # Build tools
    "git", "cmake", "ninja-build", "build-essential", "pkg-config",
    # Python
    "python3", "python3-pip", "python3-venv",
    # Utilities
    "curl", "ca-certificates", "jq", "xz-utils",
    # Math / threading
    "libopenblas-dev", "libomp-dev",
    # Build dep (not linked, but needed for cmake find)
    "libcurl4-openssl-dev",
]


class PackagesStep(Step):
    name = "packages"
    description = "Install system packages"

    def check(self) -> bool:
        # Check a representative subset — if these are present, apt ran.
        probes = ["cmake", "ninja-build"]
        # Also probe the first backend-specific package, if any.
        if self.backend.packages:
            probes.append(self.backend.packages[0])
        for pkg in probes:
            if not self.sh_ok(f"dpkg -s {pkg} 2>/dev/null"):
                return False
        return True

    def run(self) -> None:
        all_pkgs = COMMON_PACKAGES + self.backend.packages
        self.sh("apt-get update")
        self.sh(f"apt-get install -y {' '.join(all_pkgs)}")
