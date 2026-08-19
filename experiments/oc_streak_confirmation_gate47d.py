"""Gate 47d: fresh-seed confirmation after the persistent SMP writeback repair.

This gate keeps the Gate47b execution matrix and protocol unchanged while
using a new layer-2 seed and output directory.  Historical Gate47b/Gate47c
artifacts are therefore immutable evidence and cannot be mistaken for this
repair's result.
"""

from __future__ import annotations

from pathlib import Path

from experiments import oc_streak_confirmation_gate47b as gate47b


gate47b.LAYER2_SEED = 20260838
gate47b.OUTPUT_ROOT = Path("artifacts/oc_streak_confirmation_gate47d")


def main() -> int:
    return gate47b.main()


if __name__ == "__main__":
    raise SystemExit(main())
