"""Audit deterministic scheduler revisit caps and replay component leases."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ARAC_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ARAC_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(ARAC_SRC_ROOT))

from arac.policy.component_delayed_credit import (
    SchedulerRevisitCap,
    calculate_scheduler_revisit_cap,
)


PRECISION_ACTION = "post_retirement_precision_reanchor"
MIN_SELECTED_RUNS = 6
MIN_SELECTED_CASES = 3
MIN_CASES_WITH_TWO_SELECTED_SEEDS = 2


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty scheduler-cap audit: {path}")
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


def _float(value: object) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def audit_cap_contract(
    row: dict[str, str],
) -> tuple[SchedulerRevisitCap, list[str]]:
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


def _release_observed_lease(
    active: dict[str, str] | None,
    current_fe: int,
) -> tuple[dict[str, str] | None, str]:
    if active is None:
        return None, ""
    resolution_fe = _int(
        active.get("component_resolution_fe"), field="component_resolution_fe"
    )
    if (
        active.get("component_credit_status") == "resolved"
        and resolution_fe <= current_fe
    ):
        return None, active.get("component_action_id", "")
    return active, ""


def replay_run(trace_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    actions = [
        row
        for row in trace_rows
        if row.get("selected_action_name") == PRECISION_ACTION
    ]
    actions.sort(
        key=lambda row: (
            _int(row.get("component_decision_fe"), field="component_decision_fe"),
            row.get("component_action_id", ""),
        )
    )
    active_by_component: dict[str, dict[str, str]] = {}
    output: list[dict[str, object]] = []
    for action in actions:
        component_id = action.get("component_id", "")
        action_id = action.get("component_action_id", "")
        if not component_id or not action_id:
            raise ValueError("precision action is missing component identity")
        decision_fe = _int(
            action.get("component_decision_fe"), field="component_decision_fe"
        )
        active, released_action_id = _release_observed_lease(
            active_by_component.get(component_id), decision_fe
        )
        if active is None:
            active_by_component.pop(component_id, None)
        else:
            active_by_component[component_id] = active
        active_action_id = active.get("component_action_id", "") if active else ""

        cap, cap_blockers = audit_cap_contract(action)
        if cap_blockers:
            abstain_reason = "abstain_invalid_cap_contract"
        elif not cap.reachable:
            abstain_reason = "abstain_scheduler_unreachable"
        elif active is not None:
            abstain_reason = "abstain_component_mutex"
        else:
            abstain_reason = ""
        selected = not abstain_reason
        if selected:
            active_by_component[component_id] = action

        status = action.get("component_credit_status", "") if selected else ""
        resolution_fe = (
            _int(action.get("component_resolution_fe"), field="component_resolution_fe")
            if selected
            else None
        )
        actual_delay = (
            resolution_fe - decision_fe
            if selected and status == "resolved" and resolution_fe is not None
            else None
        )
        cap_underprediction = (
            int(actual_delay > cap.cap_fe)
            if actual_delay is not None and cap.cap_fe is not None
            else ""
        )
        neighbor_gain = (
            _float(action.get("component_neighbor_gain")) if selected else None
        )
        overwrite = (
            _float(action.get("shared_var_overwrite_rate")) if selected else None
        )
        survival = (
            _float(action.get("shared_var_survival_rate")) if selected else None
        )
        output.append(
            {
                "problem_id": action.get("problem_id", ""),
                "seed": action.get("seed", ""),
                "component_action_id": action_id,
                "component_id": component_id,
                "outer_iter": action.get("outer_iter", ""),
                "group_index": action.get("group_index", ""),
                "component_decision_fe": decision_fe,
                "scheduler_revisit_cap_fe": "" if cap.cap_fe is None else cap.cap_fe,
                "scheduler_revisit_reachable": int(cap.reachable),
                "scheduler_revisit_reason": cap.reason,
                "cap_contract_status": "pass" if not cap_blockers else "fail",
                "cap_contract_blockers": ";".join(cap_blockers),
                "active_lease_action_id_before": active_action_id,
                "released_lease_action_id": released_action_id,
                "replay_decision": "selected" if selected else "abstained",
                "abstain_reason": abstain_reason,
                "overlap_violation": int(selected and active is not None),
                "selected_observed_credit_status": status,
                "selected_observed_resolution_fe": (
                    "" if resolution_fe is None else resolution_fe
                ),
                "selected_actual_resolution_delay_fe": (
                    "" if actual_delay is None else actual_delay
                ),
                "selected_cap_slack_fe": (
                    ""
                    if actual_delay is None or cap.cap_fe is None
                    else cap.cap_fe - actual_delay
                ),
                "selected_cap_underprediction": cap_underprediction,
                "selected_neighbor_harm": (
                    "" if neighbor_gain is None else int(neighbor_gain < 0.0)
                ),
                "selected_shared_var_overwrite_rate": (
                    "" if overwrite is None else overwrite
                ),
                "selected_shared_var_survival_rate": (
                    "" if survival is None else survival
                ),
            }
        )
    return output


def summarize_run(
    problem_id: str,
    seed: int,
    replay_rows: list[dict[str, object]],
) -> dict[str, object]:
    selected = [row for row in replay_rows if row["replay_decision"] == "selected"]
    resolved = [
        row for row in selected if row["selected_observed_credit_status"] == "resolved"
    ]
    return {
        "problem_id": problem_id,
        "seed": seed,
        "original_action_count": len(replay_rows),
        "selected_action_count": len(selected),
        "abstained_action_count": len(replay_rows) - len(selected),
        "abstain_scheduler_unreachable": sum(
            row["abstain_reason"] == "abstain_scheduler_unreachable"
            for row in replay_rows
        ),
        "abstain_component_mutex": sum(
            row["abstain_reason"] == "abstain_component_mutex"
            for row in replay_rows
        ),
        "selected_resolved_count": len(resolved),
        "selected_unresolved_count": len(selected) - len(resolved),
        "selected_overlap_violation_count": sum(
            int(row["overlap_violation"]) for row in selected
        ),
        "selected_cap_underprediction_count": sum(
            int(row["selected_cap_underprediction"])
            for row in resolved
            if row["selected_cap_underprediction"] != ""
        ),
        "cap_contract_failure_count": sum(
            row["cap_contract_status"] != "pass" for row in replay_rows
        ),
        "selected_neighbor_harm_count": sum(
            int(row["selected_neighbor_harm"])
            for row in selected
            if row["selected_neighbor_harm"] != ""
        ),
        "selected_overwrite_observation_count": sum(
            row["selected_shared_var_overwrite_rate"] != "" for row in selected
        ),
        "selected_cap_slack_fe_min": min(
            (
                int(row["selected_cap_slack_fe"])
                for row in resolved
                if row["selected_cap_slack_fe"] != ""
            ),
            default="",
        ),
    }


def validate_input_integrity(audit_root: Path) -> bool:
    integrity = _read_csv(audit_root / "component_credit_integrity_audit.csv")
    parity = _read_csv(audit_root / "component_credit_parity_audit.csv")
    gates = _read_csv(audit_root / "component_credit_gate.csv")
    return bool(
        integrity
        and parity
        and len(gates) == 1
        and all(row.get("status") == "pass" for row in integrity)
        and all(row.get("status") == "pass" for row in parity)
        and gates[0].get("anti_leakage_pass") == "1"
        and _int(gates[0].get("integrity_pass_count"), field="integrity_pass_count")
        == len(integrity)
        and _int(gates[0].get("parity_pass_count"), field="parity_pass_count")
        == len(parity)
    )


def build_gate(
    *,
    run_summaries: list[dict[str, object]],
    input_integrity_pass: bool,
) -> dict[str, object]:
    selected_runs = [row for row in run_summaries if row["selected_action_count"]]
    selected_cases = {str(row["problem_id"]) for row in selected_runs}
    seeds_by_case: dict[str, set[int]] = defaultdict(set)
    for row in selected_runs:
        seeds_by_case[str(row["problem_id"])].add(int(row["seed"]))
    cases_with_two_seeds = sum(len(values) >= 2 for values in seeds_by_case.values())
    selected = sum(int(row["selected_action_count"]) for row in run_summaries)
    resolved = sum(int(row["selected_resolved_count"]) for row in run_summaries)
    unresolved = sum(int(row["selected_unresolved_count"]) for row in run_summaries)
    overlap = sum(
        int(row["selected_overlap_violation_count"]) for row in run_summaries
    )
    underprediction = sum(
        int(row["selected_cap_underprediction_count"]) for row in run_summaries
    )
    contract_failures = sum(
        int(row["cap_contract_failure_count"]) for row in run_summaries
    )
    blockers: list[str] = []
    if not input_integrity_pass:
        blockers.append("input_integrity_failed")
    if contract_failures:
        blockers.append("scheduler_cap_contract_failed")
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
    if cases_with_two_seeds < MIN_CASES_WITH_TWO_SELECTED_SEEDS:
        blockers.append("selected_cross_seed_case_coverage_lt_2")
    return {
        "overall_status": "pass" if not blockers else "fail",
        "evidence_scope": "scheduler_feasibility_only",
        "input_integrity_pass": int(input_integrity_pass),
        "current_action_outcome_used_for_eligibility": 0,
        "run_count": len(run_summaries),
        "selected_action_count": selected,
        "selected_resolved_count": resolved,
        "selected_unresolved_count": unresolved,
        "selected_run_count": len(selected_runs),
        "selected_case_count": len(selected_cases),
        "cases_with_two_selected_seeds": cases_with_two_seeds,
        "selected_overlap_violation_count": overlap,
        "selected_cap_underprediction_count": underprediction,
        "cap_contract_failure_count": contract_failures,
        "blockers": ";".join(blockers),
    }


def write_reports(
    *,
    v40_root: Path,
    credit_audit_root: Path,
    output_root: Path,
) -> tuple[Path, ...]:
    traces = _group_by_run(_read_csv(v40_root / "action_trace.csv"))
    ledgers = _group_by_run(_read_csv(v40_root / "same_budget_ledger.csv"))
    if set(traces) != set(ledgers) or any(len(rows) != 1 for rows in ledgers.values()):
        raise ValueError("trace and FE-ledger matrices differ")
    replay_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for problem_id, seed in sorted(ledgers):
        run_replay = replay_run(traces[(problem_id, seed)])
        replay_rows.extend(run_replay)
        summaries.append(summarize_run(problem_id, seed, run_replay))
    gate = build_gate(
        run_summaries=summaries,
        input_integrity_pass=validate_input_integrity(credit_audit_root),
    )
    paths = (
        output_root / "scheduler_revisit_cap_action_replay.csv",
        output_root / "scheduler_revisit_cap_run_summary.csv",
        output_root / "scheduler_revisit_cap_gate.csv",
    )
    for path, rows in zip(paths, (replay_rows, summaries, [gate]), strict=True):
        _write_csv(path, rows)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v40-dir", type=Path, required=True)
    parser.add_argument("--credit-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = write_reports(
        v40_root=args.v40_dir,
        credit_audit_root=args.credit_audit_dir,
        output_root=args.output_dir,
    )
    gate = _read_csv(paths[-1])[0]
    if gate["overall_status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
