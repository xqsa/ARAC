from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

from arac.backends.binary_lsgo import (
    BinaryLsgoExecutionRequest,
    BinaryLsgoExecutionResult,
    run_binary_lsgo,
)
from arac.benchmarks.binary_lsgo import (
    BinaryLsgoProblem,
    generate_binary_lsgo,
    standard_binary_lsgo_specs,
)
from arac.evaluation import classify_utility, relative_gain
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS

RUN_ID = "exp_010_binary_lsgo_focused_3seed"
FOCUSED_PROBLEM_IDS = (
    "BLSGO-F07",
    "BLSGO-F08",
    "BLSGO-F09",
    "BLSGO-F14",
    "BLSGO-F15",
)
TARGET_PROBLEM_IDS = ("BLSGO-F08", "BLSGO-F15")
OPTIMIZER_SEEDS = (20_260_713, 20_260_714, 20_260_715)
LANES = (
    "native_baseline",
    "arac_policy",
    "shuffled_evidence_negative_control",
)
CANONICAL_TOTAL_FES = 2_000
PHASE_ONE_FRACTION = 0.20
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FocusedRunRecord:
    result: BinaryLsgoExecutionResult
    input_hash: str
    offline_gain_vs_native: float
    utility_label_vs_native: str
    forbidden_runtime_fields: tuple[str, ...]
    negative_evidence_changed: bool
    claim_allowed: bool = False


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


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_fixed_specs() -> dict[str, object]:
    specs = {spec.problem_id: spec for spec in standard_binary_lsgo_specs()}
    missing = [problem_id for problem_id in FOCUSED_PROBLEM_IDS if problem_id not in specs]
    if missing:
        raise ValueError(f"focused binary LSGO cases are missing: {', '.join(missing)}")
    return {problem_id: specs[problem_id] for problem_id in FOCUSED_PROBLEM_IDS}


def _forbidden_fields(result: BinaryLsgoExecutionResult) -> tuple[str, ...]:
    return tuple(sorted(FORBIDDEN_RUNTIME_FIELDS.intersection(asdict(result.evidence))))


def execute_focused_matrix(
    total_fes: int,
) -> tuple[list[FocusedRunRecord], dict[str, str]]:
    specs = _validate_fixed_specs()
    records: list[FocusedRunRecord] = []
    input_hashes: dict[str, str] = {}
    for problem_id in FOCUSED_PROBLEM_IDS:
        problem = generate_binary_lsgo(specs[problem_id])
        input_hashes[problem_id] = _problem_hash(problem)
        for seed in OPTIMIZER_SEEDS:
            results: dict[str, BinaryLsgoExecutionResult] = {}
            for lane_id in LANES:
                results[lane_id] = run_binary_lsgo(
                    BinaryLsgoExecutionRequest(
                        problem,
                        optimizer_seed=seed,
                        total_fes=total_fes,
                        phase_one_fraction=PHASE_ONE_FRACTION,
                        run_id=RUN_ID,
                        lane_id=lane_id,
                    )
                )
            initial_hashes = {result.initial_vector_hash for result in results.values()}
            phase_one_objectives = {result.phase_one_objective for result in results.values()}
            if len(initial_hashes) != 1 or len(phase_one_objectives) != 1:
                raise RuntimeError(f"lane initialization diverged for {problem_id} seed {seed}")
            baseline = results["native_baseline"]
            policy_evidence = asdict(results["arac_policy"].evidence)
            negative_evidence = asdict(results["shuffled_evidence_negative_control"].evidence)
            evidence_changed = any(
                policy_evidence[field] != negative_evidence[field]
                for field in ("priority_spread",)
            )
            for lane_id in LANES:
                result = results[lane_id]
                gain = relative_gain(baseline.final_objective, result.final_objective)
                records.append(
                    FocusedRunRecord(
                        result=result,
                        input_hash=input_hashes[problem_id],
                        offline_gain_vs_native=gain,
                        utility_label_vs_native=classify_utility(
                            baseline.final_objective,
                            result.final_objective,
                        ),
                        forbidden_runtime_fields=_forbidden_fields(result),
                        negative_evidence_changed=(
                            evidence_changed
                            if lane_id == "shuffled_evidence_negative_control"
                            else False
                        ),
                    )
                )
    return records, input_hashes


RUN_RESULT_FIELDS = (
    "run_id",
    "problem_id",
    "optimizer_seed",
    "lane_id",
    "input_hash",
    "initial_vector_hash",
    "phase_one_objective",
    "final_objective",
    "selected_action_name",
    "selected_action_family",
    "decision",
    "trigger_reason",
    "optimizer_consumed",
    "variable_owner_changed",
    "relation_handling_changed",
    "coordination_mode_changed",
    "budget_allocation_changed",
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
    "phase_i_fe",
    "phase_ii_fe",
    "total_fe",
    "budget_limit",
    "same_budget_violation",
    "offline_gain_vs_native",
    "utility_label_vs_native",
    "catastrophic_loss",
    "forbidden_runtime_fields",
    "runtime_dispatch_allowed",
    "negative_evidence_changed",
    "claim_allowed",
)


def run_record_to_row(record: FocusedRunRecord) -> dict[str, object]:
    result = record.result
    evidence = asdict(result.evidence)
    semantics = asdict(result.semantics)
    return {
        "run_id": result.run_id,
        "problem_id": result.problem_id,
        "optimizer_seed": result.optimizer_seed,
        "lane_id": result.lane_id,
        "input_hash": record.input_hash,
        "initial_vector_hash": result.initial_vector_hash,
        "phase_one_objective": f"{result.phase_one_objective:.12g}",
        "final_objective": f"{result.final_objective:.12g}",
        "selected_action_name": result.decision.action_name,
        "selected_action_family": result.decision.action_family.value,
        "decision": result.decision.decision,
        "trigger_reason": result.decision.trigger_reason,
        "optimizer_consumed": int(result.optimizer_consumed),
        **{key: int(value) for key, value in semantics.items()},
        **{
            key: evidence[key]
            for key in (
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
            )
        },
        "phase_i_fe": result.ledger.phase_i_fe,
        "phase_ii_fe": result.ledger.phase_ii_fe,
        "total_fe": result.ledger.total_fe,
        "budget_limit": result.ledger.budget_limit,
        "same_budget_violation": int(result.ledger.violation),
        "offline_gain_vs_native": f"{record.offline_gain_vs_native:.12g}",
        "utility_label_vs_native": record.utility_label_vs_native,
        "catastrophic_loss": int(
            result.lane_id == "arac_policy" and record.offline_gain_vs_native <= -0.20
        ),
        "forbidden_runtime_fields": ";".join(record.forbidden_runtime_fields),
        "runtime_dispatch_allowed": int(not record.forbidden_runtime_fields),
        "negative_evidence_changed": int(record.negative_evidence_changed),
        "claim_allowed": int(record.claim_allowed),
    }


CASE_SUMMARY_FIELDS = (
    "problem_id",
    "seed_count",
    "action_consumed_count",
    "mean_relative_gain",
    "median_relative_gain",
    "minimum_relative_gain",
    "action_consumed_median_gain",
    "catastrophic_loss_count",
    "negative_evidence_changed_count",
    "all_same_budget",
    "runtime_boundary_pass",
)


def build_case_summaries(records: list[FocusedRunRecord]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for problem_id in FOCUSED_PROBLEM_IDS:
        case_records = [record for record in records if record.result.problem_id == problem_id]
        policy_records = [
            record for record in case_records if record.result.lane_id == "arac_policy"
        ]
        negative_records = [
            record
            for record in case_records
            if record.result.lane_id == "shuffled_evidence_negative_control"
        ]
        if len(case_records) != 9 or len(policy_records) != 3 or len(negative_records) != 3:
            raise ValueError(f"incomplete focused record set for {problem_id}")
        policy_seeds = {record.result.optimizer_seed for record in policy_records}
        if policy_seeds != set(OPTIMIZER_SEEDS):
            raise ValueError(f"unexpected optimizer seeds for {problem_id}")
        gains = [record.offline_gain_vs_native for record in policy_records]
        action_gains = [
            record.offline_gain_vs_native
            for record in policy_records
            if record.result.optimizer_consumed
        ]
        summaries.append(
            {
                "problem_id": problem_id,
                "seed_count": len(policy_records),
                "action_consumed_count": len(action_gains),
                "mean_relative_gain": mean(gains),
                "median_relative_gain": median(gains),
                "minimum_relative_gain": min(gains),
                "action_consumed_median_gain": (
                    median(action_gains) if action_gains else ""
                ),
                "catastrophic_loss_count": sum(gain <= -0.20 for gain in gains),
                "negative_evidence_changed_count": sum(
                    record.negative_evidence_changed for record in negative_records
                ),
                "all_same_budget": int(
                    all(
                        record.result.ledger.total_fe == record.result.ledger.budget_limit
                        and not record.result.ledger.violation
                        for record in case_records
                    )
                ),
                "runtime_boundary_pass": int(
                    all(not record.forbidden_runtime_fields for record in case_records)
                ),
            }
        )
    return summaries


def _gate_result(
    passed: bool,
    *,
    observed: object,
    threshold: object,
    failure_reason: str,
) -> dict[str, object]:
    return {
        "passed": passed,
        "observed": observed,
        "threshold": threshold,
        "reason": "" if passed else failure_reason,
    }


def build_promotion_gate(
    records: list[FocusedRunRecord],
    summaries: list[dict[str, object]],
    *,
    configured_total_fes: int,
) -> dict[str, object]:
    if len(records) != 45:
        raise ValueError("promotion gate requires exactly 45 focused execution records")
    summary_by_problem = {str(row["problem_id"]): row for row in summaries}
    if set(summary_by_problem) != set(FOCUSED_PROBLEM_IDS):
        raise ValueError("promotion gate requires all five focused case summaries")

    canonical_budget_pass = configured_total_fes == CANONICAL_TOTAL_FES
    action_counts = {
        problem_id: int(summary_by_problem[problem_id]["action_consumed_count"])
        for problem_id in TARGET_PROBLEM_IDS
    }
    target_action_frequency_pass = all(count >= 2 for count in action_counts.values())
    action_medians = {
        problem_id: summary_by_problem[problem_id]["action_consumed_median_gain"]
        for problem_id in TARGET_PROBLEM_IDS
    }
    target_action_median_pass = all(
        value != "" and float(value) >= 0.0 for value in action_medians.values()
    )
    policy_records = [record for record in records if record.result.lane_id == "arac_policy"]
    catastrophic_count = sum(
        record.offline_gain_vs_native <= -0.20 for record in policy_records
    )
    same_budget_violations = sum(
        record.result.ledger.total_fe != configured_total_fes
        or record.result.ledger.budget_limit != configured_total_fes
        or record.result.ledger.violation
        for record in records
    )
    forbidden_record_count = sum(bool(record.forbidden_runtime_fields) for record in records)
    negative_records = [
        record
        for record in records
        if record.result.lane_id == "shuffled_evidence_negative_control"
    ]
    changed_negative_count = sum(record.negative_evidence_changed for record in negative_records)
    negative_claim_count = sum(record.claim_allowed for record in negative_records)
    negative_control_pass = (
        len(negative_records) == 15
        and changed_negative_count == 15
        and negative_claim_count == 0
    )

    gate = {
        "canonical_budget": _gate_result(
            canonical_budget_pass,
            observed=configured_total_fes,
            threshold=CANONICAL_TOTAL_FES,
            failure_reason="configured_total_fes_is_not_2000",
        ),
        "target_action_frequency": _gate_result(
            target_action_frequency_pass,
            observed=action_counts,
            threshold={problem_id: 2 for problem_id in TARGET_PROBLEM_IDS},
            failure_reason="target_action_consumed_in_fewer_than_2_seeds",
        ),
        "target_action_median_gain": _gate_result(
            target_action_median_pass,
            observed=action_medians,
            threshold={problem_id: 0.0 for problem_id in TARGET_PROBLEM_IDS},
            failure_reason="target_action_median_gain_is_negative_or_missing",
        ),
        "no_catastrophic_loss": _gate_result(
            catastrophic_count == 0 and len(policy_records) == 15,
            observed={"policy_rows": len(policy_records), "catastrophic_count": catastrophic_count},
            threshold={"policy_rows": 15, "catastrophic_count": 0},
            failure_reason="catastrophic_loss_detected_or_policy_rows_missing",
        ),
        "same_budget": _gate_result(
            same_budget_violations == 0,
            observed={"execution_rows": len(records), "violation_count": same_budget_violations},
            threshold={"execution_rows": 45, "violation_count": 0},
            failure_reason="same_budget_violation_detected",
        ),
        "runtime_boundary": _gate_result(
            forbidden_record_count == 0,
            observed={"forbidden_record_count": forbidden_record_count},
            threshold={"forbidden_record_count": 0},
            failure_reason="forbidden_runtime_field_detected",
        ),
        "negative_control": _gate_result(
            negative_control_pass,
            observed={
                "negative_rows": len(negative_records),
                "evidence_changed_rows": changed_negative_count,
                "claim_allowed_rows": negative_claim_count,
            },
            threshold={
                "negative_rows": 15,
                "evidence_changed_rows": 15,
                "claim_allowed_rows": 0,
            },
            failure_reason="negative_control_missing_unchanged_or_claim_allowed",
        ),
    }
    gate["overall_pass"] = all(bool(value["passed"]) for value in gate.values())
    return gate


def _manifest(
    output: Path,
    *,
    total_fes: int,
    input_hashes: dict[str, str],
) -> None:
    payload = {
        "run_id": RUN_ID,
        "date": "2026-07-13",
        "executor": "Codex",
        "claim_level": "focused 3-seed pilot",
        "focused_problem_ids": list(FOCUSED_PROBLEM_IDS),
        "target_problem_ids": list(TARGET_PROBLEM_IDS),
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "lanes": list(LANES),
        "execution_count": len(FOCUSED_PROBLEM_IDS) * len(OPTIMIZER_SEEDS) * len(LANES),
        "total_fes": total_fes,
        "phase_one_fraction": PHASE_ONE_FRACTION,
        "phase_one_fes": round(total_fes * PHASE_ONE_FRACTION),
        "phase_two_fes": total_fes - round(total_fes * PHASE_ONE_FRACTION),
        "input_hashes": input_hashes,
        "code_hashes": {
            "benchmark": _file_hash(ROOT / "src/arac/benchmarks/binary_lsgo.py"),
            "backend": _file_hash(ROOT / "src/arac/backends/binary_lsgo.py"),
            "runner": _file_hash(Path(__file__)),
        },
        "artifacts": [
            "run_results.csv",
            "case_summary.csv",
            "promotion_gate.json",
            "manifest.json",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_focused_pilot(
    output_dir: Path | str = Path("results/exp_010_binary_lsgo_focused_3seed"),
    *,
    total_fes: int = CANONICAL_TOTAL_FES,
) -> Path:
    if isinstance(total_fes, bool) or not isinstance(total_fes, int) or total_fes < 2:
        raise ValueError("total_fes must be an integer >= 2")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records, input_hashes = execute_focused_matrix(total_fes)
    summaries = build_case_summaries(records)
    gate = build_promotion_gate(records, summaries, configured_total_fes=total_fes)
    _write_csv(
        output / "run_results.csv",
        [run_record_to_row(record) for record in records],
        list(RUN_RESULT_FIELDS),
    )
    _write_csv(
        output / "case_summary.csv",
        summaries,
        list(CASE_SUMMARY_FIELDS),
    )
    (output / "promotion_gate.json").write_text(
        json.dumps(gate, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _manifest(output, total_fes=total_fes, input_hashes=input_hashes)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the focused binary LSGO 3-seed pilot.")
    parser.add_argument(
        "--output-dir",
        default="results/exp_010_binary_lsgo_focused_3seed",
    )
    parser.add_argument("--total-fes", type=int, default=CANONICAL_TOTAL_FES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_focused_pilot(args.output_dir, total_fes=args.total_fes)


if __name__ == "__main__":
    main()
