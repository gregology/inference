"""Entry point for python3 -m installer."""

import argparse
import sys

from .config import load_config
from .hardware import detect_hardware
from .runner import run_steps
from .steps import ALL_STEPS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install and maintain the O6 inference stack.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove models that are no longer in models.toml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--only",
        help="Run only the named step (e.g. --only models)",
    )
    parser.add_argument(
        "--models-toml",
        default=None,
        help="Path to models.toml (default: models.toml next to install.sh)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Listen address for llama-server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Listen port for llama-server (default: 8000)",
    )
    args = parser.parse_args()

    config = load_config(args)
    hw = detect_hardware()

    print(f"\n{'=' * 60}")
    print("  inference installer")
    print(f"{'=' * 60}")
    ref = config.llama_cpp_ref
    ref_label = f"{ref} (pinned)" if ref != "latest" else "latest"
    print(f"  Models file : {config.models_toml}")
    print(f"  Models count: {len(config.models)}")
    print(f"  llama.cpp   : {ref_label}")
    print(f"  GPU device  : {hw.gpu_device or 'none detected'}")
    print(f"  Dry run     : {config.dry_run}")
    print(f"  Prune       : {config.prune}")
    print(f"{'=' * 60}\n")

    return run_steps(ALL_STEPS, config, hw)


sys.exit(main())
