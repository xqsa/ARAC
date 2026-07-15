"""Audit the frozen v41 runtime component-lease pilot."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ARAC_REPO_ROOT = Path(__file__).resolve().parents[1]
ARAC_SRC_ROOT = ARAC_REPO_ROOT / "src"
if str(ARAC_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(ARAC_SRC_ROOT))

from arac.policy.component_delayed_credit import (
    SchedulerRevisitCap,
    calculate_scheduler_revisit_cap,
    decide_component_lease,
)


MIN_SELECTED_RUNS = 6
MIN_SELECTED_CASES = 3
MIN_CASES_WITH_TWO_SELECTED_SEEDS = 2
MIN_CHANGED_RUNS = 3
CATASTROPHIC_RELATIVE_GAIN = -0.20


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty v41 audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _int(value: object, *, field: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer field {field}: {value!r}") from exc


def _float(value: object, *, field: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float field {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite float field {field}: {value!r}")
    return number


def _run_key(row: dict[str, str]) -> tuple[str, int]:
    problem_id = row.get("problem_id", "")
    if not problem_id:
        raise ValueError("missing problem_id")
    return problem_id, _int(row.get("seed"), field="seed")


def _group_by_run(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_run_key(row)].append(row)
    return grouped


def _one_by_run(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int], dict[str, str]]:
    grouped = _group_by_run(rows)
    if any(len(values) != 1 for values in grouped.values()):
        raise ValueError("expected exactly one row per run")
    return {key: values[0] for key, values in grouped.items()}


def _cap_from_trace(row: dict[str, str]) -> tuple[SchedulerRevisitCap, list[str]]:
    populations = tuple(
        _int(value, field="component_scheduler_population_sizes")
        for value in row.get("component_scheduler_population_sizes", "").split(";")
        if value
    )
    cap = calculate_scheduler_revisit_cap(
        sweep_start_fe=_int(
            row.get("component_scheduler_sweep_start_fe"),
            field="component_scheduler_sweep_start_fe",
        ),
        decision_fe=_int(
            row.get("component_decision_fe"), field="component_decision_fe"
        ),
        cc_budget_limit_fe=_int(
            row.get("component_scheduler_cc_budget_limit_fe"),
            field="component_scheduler_cc_budget_limit_fe",
        ),
        current_group_index=_int(row.get("group_index"), field="group_index"),
        current_sweep_group_budget_fe=_int(
            row.get("component_scheduler_group_budget_fe"),
            field="component_scheduler_group_budget_fe",
        ),
        current_optimizer_budget_fe=_int(
            row.get("component_scheduler_optimizer_budget_fe"),
            field="component_scheduler_optimizer_budget_fe",
        ),
        group_population_sizes=populations,
    )
    blockers: list[str] = []
    serialized_cap = row.get("component_scheduler_revisit_cap_fe", "")
    expected_cap = "" if cap.cap_fe is None else str(cap.cap_fe)
    if serialized_cap != expected_cap:
        blockers.append("scheduler_revisit_cap_mismatch")
    if row.get("component_scheduler_revisit_reachable") != str(int(cap.reachable)):
        blockers.append("scheduler_revisit_reachability_mismatch")
    if row.get("component_scheduler_revisit_reason") != cap.reason:
        blockers.append("scheduler_revisit_reason_mismatch")
    return cap, blockers


def replay_runtime_run(
    trace_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    attempts = [row for row in trace_rows if row.get("component_lease_decision")]
    attempts.sort(
        key=lambda row: (
            _int(row.get("component_decision_fe"), field="component_decision_fe"),
            _int(row.get("outer_iter"), field="outer_iter"),
            _int(row.get("group_index"), field="group_index"),
        )
    )
    active_by_component: dict[str, dict[str, str]] = {}
    output: list[dict[str, object]] = []
    for row in attempts:
        component_id = row.get("component_id", "")
        if not component_id:
            raise ValueError("lease attempt is missing component_id")
        decision_fe = _int(
            row.get("component_decision_fe"), field="component_decision_fe"
        )
        active = active_by_component.get(component_id)
        released_action_id = ""
        if active is not None and active.get("component_credit_status") == "resolved":
            resolution_fe = _int(
                active.get("component_resolution_fe"),
                field="component_resolution_fe",
            )
            if resolution_fe <= decision_fe:
                released_action_id = active.get("component_action_id", "")
                active_by_component.pop(component_id, None)
                active = None
        active_action_id = active.get("component_action_id", "") if active else ""
        cap, cap_blockers = _cap_from_trace(row)
        eligibility = decide_component_lease(
            scheduler_revisit_cap=cap,
            active_component_action_id=active_action_id,
        )
        runtime_selected = row.get("component_lease_decision") == "selected"
        decision_matches = runtime_selected == eligibility.selected
        reason_matches = row.get("component_lease_reason") == eligibility.reason
        active_matches = (
            row.get("component_active_lease_action_id")
            == eligibility.active_component_action_id
        )
        consumed_matches = row.get("component_precision_consumed") == str(
            int(runtime_selected)
        ) and row.get("downstream_consumed") == str(int(runtime_selected))
        overlap = int(runtime_selected and bool(active_action_id))
        status = row.get("component_credit_status", "") if runtime_selected else ""
        resolution_fe = (
            _int(row.get("component_resolution_fe"), field="component_resolution_fe")
            if runtime_selected and row.get("component_resolution_fe", "")
            else None
        )
        actual_delay = (
            resolution_fe - decision_fe
            if status == "resolved" and resolution_fe is not None
            else None
        )
        underprediction = (
            int(actual_delay > cap.cap_fe)
            if actual_delay is not None and cap.cap_fe is not None
            else ""
        )
        action_id = row.get("component_action_id", "")
        if runtime_selected:
            if not action_id:
                cap_blockers.append("selected_lease_missing_action_id")
            active_by_component[component_id] = row
        output.append(
            {
                "problem_id": row.get("problem_id", ""),
                "seed": row.get("seed", ""),
                "outer_iter": row.get("outer_iter", ""),
                "group_index": row.get("group_index", ""),
                "component_id": component_id,
                "component_action_id": action_id,
                "component_decision_fe": decision_fe,
                "active_lease_action_id_before": active_action_id,
                "released_lease_action_id": released_action_id,
                "expected_decision": (
                    "selected" if eligibility.selected else "abstained"
                ),
                "runtime_decision": row.get("component_lease_decision", ""),
                "expected_reason": eligibility.reason,
                "runtime_reason": row.get("component_lease_reason", ""),
                "decision_match": int(decision_matches),
                "reason_match": int(reason_matches),
                "active_lease_match": int(active_matches),
                "consumption_match": int(consumed_matches),
                "cap_contract_status": "pass" if not cap_blockers else "fail",
                "cap_contract_blockers": ";".join(cap_blockers),
                "overlap_violation": overlap,
                "selected_credit_status": status,
                "selected_resolution_fe": (
                    "" if resolution_fe is None else resolution_fe
                ),
                "selected_resolution_delay_fe": (
                    "" if actual_delay is None else actual_delay
                ),
                "selected_cap_underprediction": underprediction,
            }
        )
    return output


def summarize_runtime_run(
    problem_id: str,
    seed: int,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    selected = [row for row in rows if row["runtime_decision"] == "selected"]
    return {
        "problem_id": problem_id,
        "seed": seed,
        "attempt_count": len(rows),
        "selected_action_count": len(selected),
        "abstained_action_count": len(rows) - len(selected),
        "selected_resolved_count": sum(
            row["selected_credit_status"] == "resolved" for row in selected
        ),
        "selected_unresolved_count": sum(
            row["selected_credit_status"] != "resolved" for row in selected
        ),
        "overlap_violation_count": sum(
            int(row["overlap_violation"]) for row in selected
        ),
        "cap_underprediction_count": sum(
            int(row["selected_cap_underprediction"])
            for row in selected
            if row["selected_cap_underprediction"] != ""
        ),
        "contract_failure_count": sum(
            row["cap_contract_status"] != "pass" for row in rows
        ),
        "decision_mismatch_count": sum(
            not (
                row["decision_match"]
                and row["reason_match"]
                and row["active_lease_match"]
                and row["consumption_match"]
            )
            for row in rows
        ),
    }


def _aob_signature(rows: list[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((row.get("file", ""), row.get("sha256_before", "")) for row in rows)
    )


def audit_root(
    root: Path,
    expected_keys: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int], dict[str, str]], list[str]]:
    results = _one_by_run(_read_csv(root / "our_result_by_case.csv"))
    ledgers = _one_by_run(_read_csv(root / "same_budget_ledger.csv"))
    aob = _group_by_run(_read_csv(root / "aob_input_manifest.csv"))
    anti = _read_csv(root / "anti_leakage_audit.csv")
    blockers: list[str] = []
    if set(results) != expected_keys or set(ledgers) != expected_keys or set(aob) != expected_keys:
        blockers.append("run_matrix_mismatch")
    if not anti or any(row.get("audit_status") != "pass" for row in anti):
        blockers.append("anti_leakage_failed")
    for key in expected_keys.intersection(results, ledgers, aob):
        result = results[key]
        ledger = ledgers[key]
        if result.get("hcc_smoke_status") != "completed":
            blockers.append(f"run_not_completed:{key[0]}:{key[1]}")
        if result.get("fresh_optimizer_execution") != "1" or ledger.get(
            "fresh_execution"
        ) != "1":
            blockers.append(f"run_not_fresh:{key[0]}:{key[1]}")
        actual = _int(ledger.get("actual_fe_used"), field="actual_fe_used")
        budget = _int(ledger.get("budget_limit"), field="budget_limit")
        if actual > budget or ledger.get("same_budget_violation") != "0":
            blockers.append(f"fe_overspend:{key[0]}:{key[1]}")
        if any(
            row.get("unchanged") != "1"
            or row.get("sha256_before") != row.get("sha256_after")
            for row in aob[key]
        ):
            blockers.append(f"aob_input_changed:{key[0]}:{key[1]}")
    return results, sorted(set(blockers))


def build_performance_rows(
    *,
    v41_results: dict[tuple[str, int], dict[str, str]],
    v38_results: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, object]]:
    if set(v41_results) != set(v38_results):
        raise ValueError("v38/v41 result matrices differ")
    rows: list[dict[str, object]] = []
    for problem_id, seed in sorted(v41_results):
        v41_error = _float(
            v41_results[(problem_id, seed)].get("hcc_smoke_final_error"),
            field="v41_final_error",
        )
        v38_error = _float(
            v38_results[(problem_id, seed)].get("hcc_smoke_final_error"),
            field="v38_final_error",
        )
        relative_gain = (v38_error - v41_error) / max(abs(v38_error), 1e-300)
        log_advantage = math.log(
            max(v38_error, 1e-300) / max(v41_error, 1e-300)
        )
        rows.append(
            {
                "problem_id": problem_id,
                "seed": seed,
                "v38_final_error": f"{v38_error:.12e}",
                "v41_final_error": f"{v41_error:.12e}",
                "paired_log_advantage": f"{log_advantage:.12e}",
                "relative_gain": f"{relative_gain:.12e}",
                "changed": int(v38_error != v41_error),
                "win": int(v41_error < v38_error),
                "loss": int(v41_error > v38_error),
                "catastrophic_loss": int(
                    relative_gain <= CATASTROPHIC_RELATIVE_GAIN
                ),
            }
        )
    return rows


def build_gate(
    *,
    run_summaries: list[dict[str, object]],
    performance_rows: list[dict[str, object]],
    integrity_blockers: list[str],
    aob_equal: bool,
) -> dict[str, object]:
    selected_runs = [row for row in run_summaries if row["selected_action_count"]]
    selected_cases = {str(row["problem_id"]) for row in selected_runs}
    seeds_by_case: dict[str, set[int]] = defaultdict(set)
    for row in selected_runs:
        seeds_by_case[str(row["problem_id"])].add(int(row["seed"]))
    cross_seed_cases = sum(len(seeds) >= 2 for seeds in seeds_by_case.values())
    selected = sum(int(row["selected_action_count"]) for row in run_summaries)
    unresolved = sum(int(row["selected_unresolved_count"]) for row in run_summaries)
    overlap = sum(int(row["overlap_violation_count"]) for row in run_summaries)
    underprediction = sum(
        int(row["cap_underprediction_count"]) for row in run_summaries
    )
    contract_failures = sum(
        int(row["contract_failure_count"]) for row in run_summaries
    )
    decision_mismatches = sum(
        int(row["decision_mismatch_count"]) for row in run_summaries
    )
    log_advantages = [
        _float(row["paired_log_advantage"], field="paired_log_advantage")
        for row in performance_rows
    ]
    changed = sum(int(row["changed"]) for row in performance_rows)
    wins = sum(int(row["win"]) for row in performance_rows)
    losses = sum(int(row["loss"]) for row in performance_rows)
    catastrophic = sum(int(row["catastrophic_loss"]) for row in performance_rows)
    mean_log = statistics.fmean(log_advantages)
    median_log = statistics.median(log_advantages)
    blockers = list(integrity_blockers)
    if not aob_equal:
        blockers.append("v38_v41_aob_manifest_mismatch")
    if contract_failures:
        blockers.append("scheduler_cap_contract_failed")
    if decision_mismatches:
        blockers.append("runtime_lease_decision_mismatch")
    if overlap:
        blockers.append("selected_lease_overlap")
    if not selected or unresolved:
        blockers.append("selected_horizon_not_closed")
    if underprediction:
        blockers.append("scheduler_cap_underpredicted")
    if len(selected_runs) < MIN_SELECTED_RUNS:
        blockers.append("selected_run_coverage_lt_6")
    if len(selected_cases) < MIN_SELECTED_CASES:
        blockers.append("selected_case_coverage_lt_3")
    if cross_seed_cases < MIN_CASES_WITH_TWO_SELECTED_SEEDS:
        blockers.append("selected_cross_seed_case_coverage_lt_2")
    if changed < MIN_CHANGED_RUNS:
        blockers.append("changed_run_count_lt_3")
    if catastrophic:
        blockers.append("catastrophic_loss")
    if mean_log <= 0.0:
        blockers.append("mean_log_advantage_not_positive")
    if median_log < 0.0:
        blockers.append("median_log_advantage_negative")
    if wins < losses:
        blockers.append("changed_run_wins_lt_losses")
    return {
        "overall_status": "pass" if not blockers else "fail",
        "evidence_scope": "v41_runtime_component_lease_pilot",
        "current_action_outcome_used_for_eligibility": 0,
        "run_count": len(run_summaries),
        "selected_action_count": selected,
        "selected_resolved_count": selected - unresolved,
        "selected_unresolved_count": unresolved,
        "selected_run_count": len(selected_runs),
        "selected_case_count": len(selected_cases),
        "cases_with_two_selected_seeds": cross_seed_cases,
        "overlap_violation_count": overlap,
        "cap_underprediction_count": underprediction,
        "cap_contract_failure_count": contract_failures,
        "decision_mismatch_count": decision_mismatches,
        "paired_changed_run_count": changed,
        "paired_win_count": wins,
        "paired_loss_count": losses,
        "catastrophic_loss_count": catastrophic,
        "mean_paired_log_advantage": f"{mean_log:.12e}",
        "median_paired_log_advantage": f"{median_log:.12e}",
        "blockers": ";".join(sorted(set(blockers))),
    }


def write_reports(
    *,
    v41_root: Path,
    v38_root: Path,
    problems: tuple[str, ...],
    seeds: tuple[int, ...],
    output_root: Path,
) -> tuple[Path, ...]:
    expected_keys = {(problem, seed) for problem in problems for seed in seeds}
    v41_results, v41_blockers = audit_root(v41_root, expected_keys)
    v38_results, v38_blockers = audit_root(v38_root, expected_keys)
    v41_aob = _group_by_run(_read_csv(v41_root / "aob_input_manifest.csv"))
    v38_aob = _group_by_run(_read_csv(v38_root / "aob_input_manifest.csv"))
    aob_equal = set(v41_aob) == set(v38_aob) and all(
        _aob_signature(v41_aob[key]) == _aob_signature(v38_aob[key])
        for key in set(v41_aob).intersection(v38_aob)
    )
    traces = _group_by_run(_read_csv(v41_root / "action_trace.csv"))
    if set(traces) != expected_keys:
        raise ValueError("v41 trace matrix mismatch")
    replay_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for problem_id, seed in sorted(expected_keys):
        run_rows = replay_runtime_run(traces[(problem_id, seed)])
        replay_rows.extend(run_rows)
        summaries.append(summarize_runtime_run(problem_id, seed, run_rows))
    performance = build_performance_rows(
        v41_results=v41_results,
        v38_results=v38_results,
    )
    gate = build_gate(
        run_summaries=summaries,
        performance_rows=performance,
        integrity_blockers=[
            *(f"v41:{value}" for value in v41_blockers),
            *(f"v38:{value}" for value in v38_blockers),
        ],
        aob_equal=aob_equal,
    )
    paths = (
        output_root / "runtime_component_lease_action_audit.csv",
        output_root / "runtime_component_lease_run_summary.csv",
        output_root / "runtime_component_lease_paired_performance.csv",
        output_root / "runtime_component_lease_gate.csv",
    )
    for path, rows in zip(
        paths,
        (replay_rows, summaries, performance, [gate]),
        strict=True,
    ):
        _write_csv(path, rows)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v41-dir", type=Path, required=True)
    parser.add_argument("--v38-dir", type=Path, required=True)
    parser.add_argument("--problems", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = write_reports(
        v41_root=args.v41_dir,
        v38_root=args.v38_dir,
        problems=tuple(args.problems),
        seeds=tuple(args.seeds),
        output_root=args.output_dir,
    )
    gate = _read_csv(paths[-1])[0]
    if gate["overall_status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
