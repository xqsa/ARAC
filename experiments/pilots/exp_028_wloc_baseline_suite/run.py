"""CLI for matrix emission and implementation-only WLOC mechanical smokes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from .protocol import CASE_IDS, METHODS, build_task_matrix, load_protocol  # noqa: E402
from .runner import run_mechanical_smoke, write_task_matrix  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen continuous WLOC baseline suite.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-matrix", type=Path)
    mode.add_argument("--mechanical-smoke", action="store_true")
    parser.add_argument("--case", choices=CASE_IDS, default="WLOC01")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results") / "exp_028_wloc_baseline_suite" / "mechanical_smoke",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[Path, ...] | Path:
    args = parse_args(argv)
    config = load_protocol()
    if args.emit_matrix is not None:
        return write_task_matrix(args.emit_matrix, config, build_task_matrix(config))
    methods = config.methods if args.method == "all" else (args.method,)
    return run_mechanical_smoke(
        config,
        args.output_root,
        case_id=args.case,
        methods=methods,
    )


if __name__ == "__main__":
    main()
