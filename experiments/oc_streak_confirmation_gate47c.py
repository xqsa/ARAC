"""Gate 47c: full fresh-seed confirmation after the budget-lifecycle repair.

The execution matrix is intentionally inherited from Gate47b.  Only the output
directory and the layer-2 seed change, so the historical Gate47b artifacts stay
immutable while the repaired loop is evaluated under a new seed.
"""

from __future__ import annotations

from pathlib import Path

from experiments import oc_streak_confirmation_gate47b as gate47b


gate47b.LAYER2_SEED = 20260837
gate47b.OUTPUT_ROOT = Path("artifacts/oc_streak_confirmation_gate47c")


def main() -> int:
    return gate47b.main()


if __name__ == "__main__":
    raise SystemExit(main())
