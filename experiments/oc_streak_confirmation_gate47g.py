"""Gate 47g: fresh-seed confirmation of the arbitration value gate.

Gate47f remains immutable.  This wrapper reuses its paired matrix protocol,
changes only the fresh Layer-2 seed and writes a new artifact directory so the
value-gate result is independently auditable.
"""

from __future__ import annotations

from pathlib import Path

from experiments import oc_streak_confirmation_gate47b as gate47b


gate47b.LAYER2_SEED = 20260841
gate47b.OUTPUT_ROOT = Path("artifacts/oc_streak_confirmation_gate47g")


def main() -> int:
    return gate47b.main()


if __name__ == "__main__":
    raise SystemExit(main())
