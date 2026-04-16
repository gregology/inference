"""Step runner — iterates steps, checks idempotency, applies."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import BackendConfig
    from .config import Config
    from .hardware import Hardware
    from .step import Step


def run_steps(
    steps: list[type[Step]],
    config: Config,
    hw: Hardware,
    backend: BackendConfig,
) -> int:
    only = None
    for arg in sys.argv:
        if arg.startswith("--only"):
            break
    else:
        arg = ""
    # Parse --only from config isn't stored, re-check argv
    # Actually, let's just accept it via a simpler path
    import argparse
    # We already parsed; check sys.argv directly for --only value
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            only = sys.argv[idx + 1]

    errors = 0
    for step_cls in steps:
        step = step_cls(config, hw, backend)

        if only and step.name != only:
            continue

        if step.name in backend.skip_steps:
            print(f"\n── {step.name}: {step.description} ──")
            print(f"   (not needed for {backend.type.value} backend, skipping)")
            continue

        print(f"\n── {step.name}: {step.description} ──")

        if step.check():
            print(f"   ✓ already done, skipping")
            continue

        if config.dry_run:
            print(f"   → would run: {step.description}")
            continue

        try:
            step.run()
            print(f"   ✓ done")
        except Exception as e:
            print(f"   ✗ FAILED: {e}", file=sys.stderr)
            errors += 1
            # Continue with remaining steps rather than aborting

    print()
    if errors:
        print(f"Completed with {errors} error(s).")
        return 1

    print("All steps completed successfully.")
    return 0
