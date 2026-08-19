"""Gate 47i: fresh-seed confirmation of shared-core budget feasibility."""

from __future__ import annotations

from pathlib import Path

from experiments import oc_streak_confirmation_gate47b as gate47b


gate47b.LAYER2_SEED = 20260843
gate47b.OUTPUT_ROOT = Path("artifacts/oc_streak_confirmation_gate47i")


def main() -> int:
    return gate47b.main()


if __name__ == "__main__":
    raise SystemExit(main())
