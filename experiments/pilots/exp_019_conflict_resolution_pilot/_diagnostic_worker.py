"""Private child-process entry point for the exp_019 conflict benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import ConflictBenchmarkFactory, VENDOR_DATA_DIR


CASE_FUNCTIONS = {
    "E3": ("elliptic", 3),
    "A4": ("ackley", 4),
    "S5": ("schwefel", 5),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Private exp_019 HCC worker.")
    parser.add_argument("--case", required=True, choices=tuple(CASE_FUNCTIONS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-fes", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timestamp", required=True)
    return parser.parse_args(argv)


def build_runner_args(args: argparse.Namespace) -> list[str]:
    function_name, function_id = CASE_FUNCTIONS[str(args.case)]
    return [
        "--functions",
        function_name,
        "--ids",
        str(function_id),
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--aob-data-root",
        str(VENDOR_DATA_DIR.resolve()),
        "--timestamp",
        str(args.timestamp),
        "--seed",
        str(args.seed),
        "--max-fes",
        str(args.max_fes),
        "--arac-action",
        "arac_evidence_action_controller_v37",
        "--budget-accounting",
        "strict",
        "--search-state-backend",
        "phase_i_mmes",
        "--enable-relation-dispatch",
        "--relation-policy",
        "controller_v31",
        "--evidence-overlay-mode",
        "paired_owner",
        "--skip-plots",
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.max_fes <= 0:
        raise ValueError("max_fes must be positive")

    from scripts import hcc_smoke_runner

    hcc_smoke_runner.Benchmark = ConflictBenchmarkFactory
    hcc_smoke_runner.main(build_runner_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
