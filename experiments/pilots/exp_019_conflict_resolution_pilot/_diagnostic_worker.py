"""Private HCC child entry point for the exp019 action-ceiling audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from arac.policy.action_ceiling import (
    ACTION_CEILING_ARM_RESULT_FIELDS,
    ACTION_CEILING_CONTEXT_FIELDS,
    ACTION_CEILING_HORIZONS,
    ACTION_CEILING_FULL_MATRIX_PROFILE,
    ACTION_CEILING_PROFILES,
    RS_FAMILY_TARGET_PROFILE,
    S_FAMILY_BUDGET_PULSE_PROFILE,
    action_ceiling_capture_contract,
)

from .benchmark import ConflictBenchmarkFactory, VENDOR_DATA_DIR


CASE_FUNCTIONS = {
    "E1": ("elliptic", 1),
    "E3": ("elliptic", 3),
    "A4": ("ackley", 4),
    **{f"R{function_id}": ("rastrigin", function_id) for function_id in range(1, 7)},
    **{f"S{function_id}": ("schwefel", function_id) for function_id in range(1, 7)},
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
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(ACTION_CEILING_PROFILES)),
        default=ACTION_CEILING_FULL_MATRIX_PROFILE,
    )
    return parser.parse_args(argv)


def build_runner_args(args: argparse.Namespace) -> list[str]:
    cohort = str(args.cohort)
    problem_id = str(args.case)
    if problem_id not in COHORT_CASES[cohort]:
        raise ValueError(f"{problem_id} is not available in {cohort}")
    profile = str(
        getattr(args, "profile", ACTION_CEILING_FULL_MATRIX_PROFILE)
    )
    if profile == RS_FAMILY_TARGET_PROFILE:
        if cohort != "real_aob":
            raise ValueError("rs_family_target only supports real AOB")
        if problem_id[0] not in {"R", "S"}:
            raise ValueError("rs_family_target requires a Rastrigin or Schwefel case")
    function_name, function_id = CASE_FUNCTIONS[problem_id]
    runner_args = [
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
    if profile != ACTION_CEILING_FULL_MATRIX_PROFILE:
        runner_args.extend(("--action-ceiling-profile", profile))
    return runner_args


def _read_rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise RuntimeError(f"action-ceiling worker CSV schema mismatch: {path}")
        return list(reader)


def _require_profile_artifacts(args: argparse.Namespace, profile: str) -> None:
    problem_id = str(args.case)
    function_name, _ = CASE_FUNCTIONS[problem_id]
    base = Path(args.output_root).resolve() / str(args.timestamp) / function_name
    contexts = _read_rows(
        base / f"{problem_id}_action_ceiling_contexts.csv",
        ACTION_CEILING_CONTEXT_FIELDS,
    )
    arm_rows = _read_rows(
        base / f"{problem_id}_action_ceiling_arm_results.csv",
        ACTION_CEILING_ARM_RESULT_FIELDS,
    )
    contract = action_ceiling_capture_contract(profile, problem_id)
    expected_contexts = 4
    expected_arm_rows = expected_contexts * len(contract.arms) * len(
        ACTION_CEILING_HORIZONS
    )
    if len(contexts) != expected_contexts or len(arm_rows) != expected_arm_rows:
        raise RuntimeError(
            "R/S target worker produced incomplete action-ceiling artifacts: "
            f"contexts={len(contexts)}/{expected_contexts}, "
            f"arm_rows={len(arm_rows)}/{expected_arm_rows}"
        )


def require_rs_target_artifacts(args: argparse.Namespace) -> None:
    _require_profile_artifacts(args, RS_FAMILY_TARGET_PROFILE)


def require_s_budget_pulse_artifacts(args: argparse.Namespace) -> None:
    _require_profile_artifacts(args, S_FAMILY_BUDGET_PULSE_PROFILE)


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
    if args.profile == RS_FAMILY_TARGET_PROFILE:
        require_rs_target_artifacts(args)
    elif args.profile == S_FAMILY_BUDGET_PULSE_PROFILE:
        require_s_budget_pulse_artifacts(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
