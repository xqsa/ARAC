"""Gate 47f: final fresh-seed confirmation of the complete SMP lanes.

This gate inherits the unchanged Gate47b matrix and checks while exercising
the ordered context/neighborhood writeback and unified sense headroom repair.
Its seed and output directory are new; all earlier gate artifacts remain
immutable.
"""

from __future__ import annotations

from pathlib import Path

from experiments import oc_streak_confirmation_gate47b as gate47b


gate47b.LAYER2_SEED = 20260840
gate47b.OUTPUT_ROOT = Path("artifacts/oc_streak_confirmation_gate47f")


def main() -> int:
    return gate47b.main()


if __name__ == "__main__":
    raise SystemExit(main())
