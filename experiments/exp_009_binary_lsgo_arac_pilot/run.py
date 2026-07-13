from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from arac.backends.binary_lsgo import BinaryLsgoExecutionRequest, run_binary_lsgo
from arac.benchmarks.binary_lsgo import BinaryLsgoProblem, generate_binary_lsgo, standard_binary_lsgo_specs
from arac.evaluation import classify_utility, relative_gain
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS

RUN_ID = "exp_009_binary_lsgo_arac_pilot"
DEFAULT_TOTAL_FES = 2_000
PHASE_ONE_FRACTION = 0.20
OPTIMIZER_SEED_BASE = 20_260_713
LANES = (
    "native_baseline",
    "arac_policy",
    "shuffled_evidence_negative_control",
)
ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _problem_hash(problem: BinaryLsgoProblem) -> str:
    payload = {
        "spec": asdict(problem.spec),
        "template": problem.template,
        "groups": problem.topology.groups,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_row(result) -> dict[str, object]:
    evidence = asdict(result.evidence)
    forbidden = sorted(FORBIDDEN_RUNTIME_FIELDS.intersection(evidence))
    return {
        "run_id": result.run_id,
        "lane_id": result.lane_id,
        "problem_id": result.problem_id,
        **evidence,
        "forbidden_runtime_fields": ";".join(forbidden),
        "runtime_dispatch_allowed": int(not forbidden),
    }


def _trace_row(result) -> dict[str, object]:
    trace = asdict(result.action_trace)
    semantics = asdict(result.semantics)
    return {
        "run_id": result.run_id,
        "lane_id": result.lane_id,
        "problem_id": result.problem_id,
        **trace,
        **{f"{key}": int(value) for key, value in semantics.items()},
        "backend_semantics_changed": int(result.semantics.changed),
    }


def _ledger_row(result) -> dict[str, object]:
    return {
        "run_id": result.run_id,
        "lane_id": result.lane_id,
        "problem_id": result.problem_id,
        "optimizer_seed": result.optimizer_seed,
        "phase_i_fe": result.ledger.phase_i_fe,
        "phase_ii_fe": result.ledger.phase_ii_fe,
        "total_fe": result.ledger.total_fe,
        "budget_limit": result.ledger.budget_limit,
        "same_budget_violation": int(result.ledger.violation),
        "fresh_execution": int(result.ledger.fresh_execution),
    }


def _result_rows(results_by_case: dict[str, list]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for problem_id, results in results_by_case.items():
        baseline = next(result for result in results if result.lane_id == "native_baseline")
        for result in results:
            gain = relative_gain(baseline.final_objective, result.final_objective)
            utility = classify_utility(baseline.final_objective, result.final_objective)
            rows.append(
                {
                    "run_id": result.run_id,
                    "lane_id": result.lane_id,
                    "problem_id": problem_id,
                    "optimizer_seed": result.optimizer_seed,
                    "initial_vector_hash": result.initial_vector_hash,
                    "phase_one_objective": f"{result.phase_one_objective:.12g}",
                    "final_objective": f"{result.final_objective:.12g}",
                    "selected_action_name": result.decision.action_name,
                    "selected_action_family": result.decision.action_family.value,
                    "decision": result.decision.decision,
                    "optimizer_consumed": int(result.optimizer_consumed),
                    "offline_relative_gain_vs_native": f"{gain:.12g}",
                    "utility_label_vs_native": utility,
                    "catastrophic_loss": int(utility == "catastrophic_loss"),
                    "claim_allowed": 0,
                    "claim_blocker": "pilot_single_seed_not_final_claim",
                }
            )
    return rows


def _manifest(
    output: Path,
    *,
    total_fes: int,
    input_hashes: dict[str, str],
) -> None:
    payload = {
        "run_id": RUN_ID,
        "executor": "Codex",
        "benchmark": "deterministic binary overlapping LSGO",
        "benchmark_case_count": len(input_hashes),
        "lane_count": len(LANES),
        "lanes": list(LANES),
        "total_fes": total_fes,
        "phase_one_fraction": PHASE_ONE_FRACTION,
        "optimizer_seed_base": OPTIMIZER_SEED_BASE,
        "claim_level": "single-seed same-budget pilot; offline evidence only",
        "input_hashes": input_hashes,
        "artifacts": [
            "execution_results.csv",
            "action_trace.csv",
            "same_budget_ledger.csv",
            "runtime_evidence.csv",
            "manifest.json",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_pilot(
    output_dir: Path | str = Path("results/exp_009_binary_lsgo_arac_pilot"),
    *,
    total_fes: int = DEFAULT_TOTAL_FES,
) -> Path:
    if isinstance(total_fes, bool) or not isinstance(total_fes, int) or total_fes < 2:
        raise ValueError("total_fes must be an integer >= 2")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_by_case: dict[str, list] = {}
    input_hashes: dict[str, str] = {}

    for case_index, spec in enumerate(standard_binary_lsgo_specs(), start=1):
        problem = generate_binary_lsgo(spec)
        input_hashes[spec.problem_id] = _problem_hash(problem)
        case_results = []
        optimizer_seed = OPTIMIZER_SEED_BASE + case_index
        for lane_id in LANES:
            result = run_binary_lsgo(
                BinaryLsgoExecutionRequest(
                    problem,
                    optimizer_seed=optimizer_seed,
                    total_fes=total_fes,
                    phase_one_fraction=PHASE_ONE_FRACTION,
                    run_id=RUN_ID,
                    lane_id=lane_id,
                )
            )
            case_results.append(result)
        initial_hashes = {result.initial_vector_hash for result in case_results}
        phase_one_objectives = {result.phase_one_objective for result in case_results}
        if len(initial_hashes) != 1 or len(phase_one_objectives) != 1:
            raise RuntimeError(f"lane initialization diverged for {spec.problem_id}")
        results_by_case[spec.problem_id] = case_results

    all_results = [result for results in results_by_case.values() for result in results]
    _write_csv(
        output / "execution_results.csv",
        _result_rows(results_by_case),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "optimizer_seed",
            "initial_vector_hash",
            "phase_one_objective",
            "final_objective",
            "selected_action_name",
            "selected_action_family",
            "decision",
            "optimizer_consumed",
            "offline_relative_gain_vs_native",
            "utility_label_vs_native",
            "catastrophic_loss",
            "claim_allowed",
            "claim_blocker",
        ],
    )
    _write_csv(
        output / "action_trace.csv",
        [_trace_row(result) for result in all_results],
        [
            "run_id",
            "lane_id",
            "problem_id",
            "action_name",
            "decision",
            "trigger_reason",
            "phase",
            "affected_group_count",
            "affected_shared_variable_count",
            "allocated_fe",
            "consumed_fe",
            "variable_owner_changed",
            "relation_handling_changed",
            "coordination_mode_changed",
            "budget_allocation_changed",
            "backend_semantics_changed",
        ],
    )
    _write_csv(
        output / "same_budget_ledger.csv",
        [_ledger_row(result) for result in all_results],
        [
            "run_id",
            "lane_id",
            "problem_id",
            "optimizer_seed",
            "phase_i_fe",
            "phase_ii_fe",
            "total_fe",
            "budget_limit",
            "same_budget_violation",
            "fresh_execution",
        ],
    )
    _write_csv(
        output / "runtime_evidence.csv",
        [_evidence_row(result) for result in all_results],
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "unit_type",
            "unit_id",
            "feature_coverage",
            "overlap_degree",
            "shared_var_support_ratio",
            "direction_disagreement",
            "harmful_coord_score",
            "group_gain_asymmetry",
            "priority_spread",
            "rank_stability",
            "budget_remaining_ratio",
            "fallback_margin_proxy",
            "forbidden_runtime_fields",
            "runtime_dispatch_allowed",
        ],
    )
    _manifest(output, total_fes=total_fes, input_hashes=input_hashes)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the binary LSGO ARAC pilot.")
    parser.add_argument(
        "--output-dir",
        default="results/exp_009_binary_lsgo_arac_pilot",
        help="Directory where pilot artifacts are written.",
    )
    parser.add_argument("--total-fes", type=int, default=DEFAULT_TOTAL_FES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_pilot(args.output_dir, total_fes=args.total_fes)


if __name__ == "__main__":
    main()
