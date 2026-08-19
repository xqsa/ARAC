"""Gate 47h: fresh-seed confirmation of both value gates.

Gate47g remains immutable. This wrapper changes only the fresh Layer-2 seed
and output directory after the post-operator archive value gate was added.
"""

from __future__ import annotations

from pathlib import Path

from experiments import oc_streak_confirmation_gate47b as gate47b


gate47b.LAYER2_SEED = 20260842
gate47b.OUTPUT_ROOT = Path("artifacts/oc_streak_confirmation_gate47h")


def main() -> int:
    return gate47b.main()


if __name__ == "__main__":
    raise SystemExit(main())
