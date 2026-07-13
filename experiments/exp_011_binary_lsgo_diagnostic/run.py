from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from arac.action_space import ActionFamily
from arac.backends.binary_lsgo import (
    BinaryLsgoExecutionRequest,
    BinaryLsgoExecutionResult,
    run_binary_lsgo,
)
from arac.benchmarks.binary_lsgo import (
    BinaryLsgoProblem,
    BinaryLsgoSpec,
    generate_binary_lsgo,
    standard_binary_lsgo_specs,
)
from arac.evaluation import classify_utility, relative_gain
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS
from arac.policy import ActionDecision

RUN_ID = "exp_011_binary_lsgo_diagnostic"
DIAGNOSTIC_PROBLEM_IDS = ("BLSGO-F08", "BLSGO-F15")
OPTIMIZER_SEEDS = (20_260_713, 20_260_714, 20_260_715, 20_260_716, 20_260_717)
LANES = (
    "native_single_bit",
    "native_group_block",
    "forced_isolate",
    "arac_policy",
)
CANONICAL_TOTAL_FES = 2_000
PHASE_ONE_FRACTION = 0.20
SIGNAL_SEED_THRESHOLD = 3
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DiagnosticSignals:
    optimizer_signal: bool
    policy_signal: bool
    label: str


@dataclass(frozen=True)
class DiagnosticRunRecord:
    result: BinaryLsgoExecutionResult
    input_hash: str
    offline_gain_vs_native: float
    utility_label_vs_native: str
    forbidden_runtime_fields: tuple[str, ...]
    claim_allowed: bool = False


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _fixed_specs() -> dict[str, BinaryLsgoSpec]:
    specs = {spec.problem_id: spec for spec in standard_binary_lsgo_specs()}
    missing = [problem_id for problem_id in DIAGNOSTIC_PROBLEM_IDS if problem_id not in specs]
    if missing:
        raise ValueError(f"diagnostic binary LSGO cases are missing: {', '.join(missing)}")
    return {problem_id: specs[problem_id] for problem_id in DIAGNOSTIC_PROBLEM_IDS}


def _forbidden_fields(result: BinaryLsgoExecutionResult) -> tuple[str, ...]:
    return tuple(sorted(FORBIDDEN_RUNTIME_FIELDS.intersection(asdict(result.evidence))))


def _native_decision() -> ActionDecision:
    return ActionDecision(
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "fallback",
        "diagnostic_native_lane",
        0.0,
    )


def _forced_isolate_decision() -> ActionDecision:
    return ActionDecision(
        ActionFamily.ISOLATE,
        "isolate_conflicting_relation",
        "allow",
        "diagnostic_forced_isolate",
        1.0,
    )


def _lane_configuration(lane_id: str) -> tuple[str, ActionDecision | None]:
    if lane_id == "native_single_bit":
        return "single_bit", _native_decision()
    if lane_id == "native_group_block":
        return "group_block", _native_decision()
    if lane_id == "forced_isolate":
        return "single_bit", _forced_isolate_decision()
    if lane_id == "arac_policy":
        return "single_bit", None
    raise ValueError(f"unsupported diagnostic lane: {lane_id}")


def classify_diagnostic_signals(
    block_improved_seed_count: int,
    block_accepted_seed_count: int,
    forced_isolate_improved_seed_count: int,
    policy_isolate_not_consumed_seed_count: int,
) -> DiagnosticSignals:
    counts = (
        block_improved_seed_count,
        block_accepted_seed_count,
        forced_isolate_improved_seed_count,
        policy_isolate_not_consumed_seed_count,
    )
    if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
        raise ValueError("diagnostic signal counts must be integers")
    if any(not 0 <= count <= len(OPTIMIZER_SEEDS) for count in counts):
        raise ValueError("diagnostic signal counts must be within the fixed seed count")

    optimizer_signal = (
        block_improved_seed_count >= SIGNAL_SEED_THRESHOLD
        and block_accepted_seed_count >= SIGNAL_SEED_THRESHOLD
    )
    policy_signal = (
        forced_isolate_improved_seed_count >= SIGNAL_SEED_THRESHOLD
        and policy_isolate_not_consumed_seed_count >= SIGNAL_SEED_THRESHOLD
    )
    if optimizer_signal and policy_signal:
        label = "mixed"
    elif optimizer_signal:
        label = "optimizer_limited"
    elif policy_signal:
        label = "policy_limited"
    else:
        label = "inconclusive"
    return DiagnosticSignals(optimizer_signal, policy_signal, label)


def execute_diagnostic_matrix(
    total_fes: int,
) -> tuple[list[DiagnosticRunRecord], dict[str, str]]:
    specs = _fixed_specs()
    records: list[DiagnosticRunRecord] = []
    input_hashes: dict[str, str] = {}
    for problem_id in DIAGNOSTIC_PROBLEM_IDS:
        problem = generate_binary_lsgo(specs[problem_id])
        input_hashes[problem_id] = _problem_hash(problem)
        for seed in OPTIMIZER_SEEDS:
            results: dict[str, BinaryLsgoExecutionResult] = {}
            for lane_id in LANES:
                operator, decision_override = _lane_configuration(lane_id)
                results[lane_id] = run_binary_lsgo(
                    BinaryLsgoExecutionRequest(
                        problem,
                        optimizer_seed=seed,
                        total_fes=total_fes,
                        phase_one_fraction=PHASE_ONE_FRACTION,
                        run_id=RUN_ID,
                        lane_id=lane_id,
                        phase_two_operator=operator,
                    ),
                    decision_override=decision_override,
                )

            initial_hashes = {result.initial_vector_hash for result in results.values()}
            phase_one_objectives = {result.phase_one_objective for result in results.values()}
            if len(initial_hashes) != 1 or len(phase_one_objectives) != 1:
                raise RuntimeError(f"lane initialization diverged for {problem_id} seed {seed}")

            baseline = results["native_single_bit"]
            for lane_id in LANES:
                result = results[lane_id]
                gain = relative_gain(baseline.final_objective, result.final_objective)
                records.append(
                    DiagnosticRunRecord(
                        result=result,
                        input_hash=input_hashes[problem_id],
                        offline_gain_vs_native=gain,
                        utility_label_vs_native=classify_utility(
                            baseline.final_objective,
                            result.final_objective,
                        ),
                        forbidden_runtime_fields=_forbidden_fields(result),
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
    "proposal_operator",
    "proposed_count",
    "accepted_count",
    "multi_bit_proposed_count",
    "multi_bit_accepted_count",
    "maximum_accepted_flip_width",
    "selected_action_name",
    "selected_action_family",
    "decision",
    "trigger_reason",
    "optimizer_consumed",
    "variable_owner_changed",
    "relation_handling_changed",
    "coordination_mode_changed",
    "budget_allocation_changed",
    "direction_disagreement",
    "harmful_coord_score",
    "rank_stability",
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
    "claim_allowed",
)


def run_record_to_row(record: DiagnosticRunRecord) -> dict[str, object]:
    result = record.result
    proposal = asdict(result.proposal_trace)
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
        "proposal_operator": proposal["operator"],
        "proposed_count": proposal["proposed_count"],
        "accepted_count": proposal["accepted_count"],
        "multi_bit_proposed_count": proposal["multi_bit_proposed_count"],
        "multi_bit_accepted_count": proposal["multi_bit_accepted_count"],
        "maximum_accepted_flip_width": proposal["maximum_accepted_flip_width"],
        "selected_action_name": result.decision.action_name,
        "selected_action_family": result.decision.action_family.value,
        "decision": result.decision.decision,
        "trigger_reason": result.decision.trigger_reason,
        "optimizer_consumed": int(result.optimizer_consumed),
        **{key: int(value) for key, value in semantics.items()},
        **{
            key: evidence[key]
            for key in (
                "direction_disagreement",
                "harmful_coord_score",
                "rank_stability",
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
        "catastrophic_loss": int(record.offline_gain_vs_native <= -0.20),
        "forbidden_runtime_fields": ";".join(record.forbidden_runtime_fields),
        "runtime_dispatch_allowed": int(not record.forbidden_runtime_fields),
        "claim_allowed": int(record.claim_allowed),
    }


CASE_SUMMARY_FIELDS = (
    "problem_id",
    "seed_count",
    "block_improved_seed_count",
    "block_accepted_seed_count",
    "forced_isolate_improved_seed_count",
    "policy_isolate_not_consumed_seed_count",
    "block_median_gain",
    "forced_isolate_median_gain",
    "policy_median_gain",
    "optimizer_signal",
    "policy_signal",
    "diagnosis_label",
    "catastrophic_loss_count",
    "all_same_budget",
    "runtime_boundary_pass",
)


def build_case_summaries(records: list[DiagnosticRunRecord]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for problem_id in DIAGNOSTIC_PROBLEM_IDS:
        case_records = [record for record in records if record.result.problem_id == problem_id]
        by_lane = {
            lane_id: [
                record for record in case_records if record.result.lane_id == lane_id
            ]
            for lane_id in LANES
        }
        if len(case_records) != len(OPTIMIZER_SEEDS) * len(LANES):
            raise ValueError(f"incomplete diagnostic record set for {problem_id}")
        if any(len(lane_records) != len(OPTIMIZER_SEEDS) for lane_records in by_lane.values()):
            raise ValueError(f"incomplete diagnostic lane set for {problem_id}")
        if any(
            {record.result.optimizer_seed for record in lane_records} != set(OPTIMIZER_SEEDS)
            for lane_records in by_lane.values()
        ):
            raise ValueError(f"unexpected optimizer seeds for {problem_id}")

        block = by_lane["native_group_block"]
        forced = by_lane["forced_isolate"]
        policy = by_lane["arac_policy"]
        block_improved_seed_count = sum(
            record.offline_gain_vs_native > 0.0 for record in block
        )
        block_accepted_seed_count = sum(
            record.result.proposal_trace.multi_bit_accepted_count > 0 for record in block
        )
        forced_isolate_improved_seed_count = sum(
            record.offline_gain_vs_native > 0.0 for record in forced
        )
        policy_isolate_not_consumed_seed_count = sum(
            record.result.decision.action_name != "isolate_conflicting_relation"
            or not record.result.optimizer_consumed
            for record in policy
        )
        signals = classify_diagnostic_signals(
            block_improved_seed_count,
            block_accepted_seed_count,
            forced_isolate_improved_seed_count,
            policy_isolate_not_consumed_seed_count,
        )
        summaries.append(
            {
                "problem_id": problem_id,
                "seed_count": len(OPTIMIZER_SEEDS),
                "block_improved_seed_count": block_improved_seed_count,
                "block_accepted_seed_count": block_accepted_seed_count,
                "forced_isolate_improved_seed_count": forced_isolate_improved_seed_count,
                "policy_isolate_not_consumed_seed_count": (
                    policy_isolate_not_consumed_seed_count
                ),
                "block_median_gain": median(
                    record.offline_gain_vs_native for record in block
                ),
                "forced_isolate_median_gain": median(
                    record.offline_gain_vs_native for record in forced
                ),
                "policy_median_gain": median(
                    record.offline_gain_vs_native for record in policy
                ),
                "optimizer_signal": int(signals.optimizer_signal),
                "policy_signal": int(signals.policy_signal),
                "diagnosis_label": signals.label,
                "catastrophic_loss_count": sum(
                    record.offline_gain_vs_native <= -0.20
                    for record in case_records
                    if record.result.lane_id != "native_single_bit"
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


def _diagnosis_payload(summaries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "claim_level": "mechanism_diagnosis_only",
        "claim_allowed": False,
        "signal_seed_threshold": SIGNAL_SEED_THRESHOLD,
        "negative_control_status": "not_applicable_causal_four_lane_diagnostic",
        "case_diagnoses": summaries,
    }


def _manifest(
    output: Path,
    *,
    total_fes: int,
    input_hashes: dict[str, str],
) -> None:
    phase_one_fes = round(total_fes * PHASE_ONE_FRACTION)
    payload = {
        "run_id": RUN_ID,
        "date": "2026-07-13",
        "executor": "Codex",
        "claim_level": "mechanism_diagnosis_only",
        "claim_allowed": False,
        "negative_control_status": "not_applicable_causal_four_lane_diagnostic",
        "diagnostic_problem_ids": list(DIAGNOSTIC_PROBLEM_IDS),
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "lanes": list(LANES),
        "execution_count": len(DIAGNOSTIC_PROBLEM_IDS) * len(OPTIMIZER_SEEDS) * len(LANES),
        "total_fes": total_fes,
        "phase_one_fraction": PHASE_ONE_FRACTION,
        "phase_one_fes": phase_one_fes,
        "phase_two_fes": total_fes - phase_one_fes,
        "signal_seed_threshold": SIGNAL_SEED_THRESHOLD,
        "input_hashes": input_hashes,
        "code_hashes": {
            "benchmark": _file_hash(ROOT / "src/arac/benchmarks/binary_lsgo.py"),
            "backend": _file_hash(ROOT / "src/arac/backends/binary_lsgo.py"),
            "runner": _file_hash(Path(__file__)),
        },
        "artifacts": [
            "run_results.csv",
            "case_summary.csv",
            "diagnosis.json",
            "manifest.json",
        ],
    }
    _write_json(output / "manifest.json", payload)


def run_diagnostic(
    output_dir: Path | str = Path("results/exp_011_binary_lsgo_diagnostic"),
    *,
    total_fes: int = CANONICAL_TOTAL_FES,
) -> Path:
    if isinstance(total_fes, bool) or not isinstance(total_fes, int) or total_fes < 2:
        raise ValueError("total_fes must be an integer >= 2")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records, input_hashes = execute_diagnostic_matrix(total_fes)
    summaries = build_case_summaries(records)
    _write_csv(
        output / "run_results.csv",
        [run_record_to_row(record) for record in records],
        list(RUN_RESULT_FIELDS),
    )
    _write_csv(output / "case_summary.csv", summaries, list(CASE_SUMMARY_FIELDS))
    _write_json(output / "diagnosis.json", _diagnosis_payload(summaries))
    _manifest(output, total_fes=total_fes, input_hashes=input_hashes)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed binary LSGO F08/F15 mechanism diagnosis."
    )
    parser.add_argument(
        "--output-dir",
        default="results/exp_011_binary_lsgo_diagnostic",
    )
    parser.add_argument("--total-fes", type=int, default=CANONICAL_TOTAL_FES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_diagnostic(args.output_dir, total_fes=args.total_fes)


if __name__ == "__main__":
    main()
