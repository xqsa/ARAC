"""Run the paired E3/A4/S5 shared-variable repair pilot."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from experiments.pilots.paired_overlap_action_runner import (
    main as paired_action_main,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "exp_021_shared_variable_repair_pilot"


def main(argv: Sequence[str] | None = None) -> int:
    forwarded_args = sys.argv[1:] if argv is None else argv
    return paired_action_main(
        [
            "--config",
            str(DEFAULT_CONFIG_PATH),
            "--output-root",
            str(DEFAULT_OUTPUT_ROOT),
            *forwarded_args,
        ],
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
