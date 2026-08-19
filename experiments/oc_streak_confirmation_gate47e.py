"""Gate 47e: fresh-seed confirmation of the ordered SMP writeback repair.

The matrix and checks are inherited without modification from Gate47b.  A
new layer-2 seed and output directory keep Gate47b/Gate47c/Gate47d artifacts
immutable and make this run an independent confirmation of the repaired
ARAC-OC lifecycle.
"""

from __future__ import annotations

from pathlib import Path

from experiments import oc_streak_confirmation_gate47b as gate47b


gate47b.LAYER2_SEED = 20260839
gate47b.OUTPUT_ROOT = Path("artifacts/oc_streak_confirmation_gate47e")


def main() -> int:
    return gate47b.main()


if __name__ == "__main__":
    raise SystemExit(main())
