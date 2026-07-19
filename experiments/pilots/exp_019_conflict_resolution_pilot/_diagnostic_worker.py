"""Private HCC child entry point for the exp019 action-ceiling audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import ConflictBenchmarkFactory, VENDOR_DATA_DIR


CASE_FUNCTIONS = {
    "E1": ("elliptic", 1),
    "E3": ("elliptic", 3),
    "A4": ("ackley", 4),
    "R4": ("rastrigin", 4),
    "S5": ("schwefel", 5),
}
COHORT_CASES = {
    "real_aob": frozenset(CASE_FUNCTIONS),
    "synthetic_conflict": frozenset({"E3", "A4", "S5"}),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Private exp019 action-ceiling worker.")
    parser.add_argument("--cohort", required=True, choices=tuple(COHORT_CASES))
    parser.add_argument("--case", required=True, choices=tuple(CASE_FUNCTIONS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-fes", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timestamp", required=True)
    return parser.parse_args(argv)


def build_runner_args(args: argparse.Namespace) -> list[str]:
    cohort = str(args.cohort)
    problem_id = str(args.case)
    if problem_id not in COHORT_CASES[cohort]:
        raise ValueError(f"{problem_id} is not available in {cohort}")
    function_name, function_id = CASE_FUNCTIONS[problem_id]
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
        "action_ceiling",
        "--evidence-overlay-mode",
        "paired_owner",
        "--action-ceiling-capture",
        "--action-ceiling-cohort",
        cohort,
        "--skip-plots",
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.max_fes <= 0:
        raise ValueError("max_fes must be positive")
    if args.case not in COHORT_CASES[args.cohort]:
        raise ValueError(f"{args.case} is not available in {args.cohort}")

    from scripts import hcc_smoke_runner

    if args.cohort == "synthetic_conflict":
        hcc_smoke_runner.Benchmark = ConflictBenchmarkFactory
    hcc_smoke_runner.main(build_runner_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
