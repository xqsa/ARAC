"""Replay component-lease feasibility from frozen v40 traces.

This script is exploratory and offline-only. It tests whether a component
mutex and a prior-cycle horizon estimate would have produced non-overlapping,
resolvable leases. It does not estimate counterfactual optimizer utility and
must never be imported by runtime dispatch code.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import NamedTuple


PRECISION_ACTION = "post_retirement_precision_reanchor"
RELATION_OBSERVATION = "relation_observation"

MIN_SELECTED_RUNS = 6
MIN_SELECTED_CASES = 3
MIN_CASES_WITH_TWO_SELECTED_SEEDS = 2

ELIGIBILITY_INPUT_FIELDS = frozenset(
    {
        "component_id",
        "outer_iter",
        "group_index",
        "component_decision_fe",
        "budget_limit",
        "prior_relation_cycle_fe",
        "prior_selected_resolution_event",
    }
)
FORBIDDEN_ELIGIBILITY_FIELDS = frozenset(
    {
        "problem_id",
        "seed",
        "function_family",
        "case_label",
        "paper_best",
        "historical_best",
        "final_error",
        "component_credit_status_of_current_action",
        "component_resolution_fe_of_current_action",
        "component_gain",
        "component_neighbor_gain",
        "shared_var_overwrite_rate",
        "shared_var_survival_rate",
    }
)


class CycleProjection(NamedTuple):
    projected_fe: int | None
    history_group_count: int
    completed_cycle_count: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty replay artifact: {path}")
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


def _mean(values: list[float]) -> float | str:
    return fmean(values) if values else ""


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


def _ledger_by_run(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int], dict[str, str]]:
    grouped = _group_by_run(rows)
    duplicates = sorted(key for key, values in grouped.items() if len(values) != 1)
    if duplicates:
        raise ValueError(f"expected one FE ledger row per run: {duplicates}")
    return {key: values[0] for key, values in grouped.items()}


def project_next_revisit_fe(
    *,
    trace_rows: list[dict[str, str]],
    component_id: str,
    action_outer_iter: int,
    action_decision_fe: int,
) -> CycleProjection:
    """Project one revisit using only completed cycles before the action sweep."""
    observations: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in trace_rows:
        if (
            row.get("component_credit_status") != RELATION_OBSERVATION
            or row.get("component_id") != component_id
        ):
            continue
        outer_iter = _int(row.get("outer_iter"), field="outer_iter")
        decision_fe = _int(
            row.get("component_decision_fe"), field="component_decision_fe"
        )
        if outer_iter >= action_outer_iter or decision_fe >= action_decision_fe:
            continue
        group_index = _int(row.get("group_index"), field="group_index")
        observations[group_index].append((outer_iter, decision_fe))

    latest_intervals: list[int] = []
    completed_cycle_count = 0
    for group_rows in observations.values():
        ordered = sorted(set(group_rows))
        group_intervals: list[int] = []
        for previous, current in zip(ordered, ordered[1:], strict=False):
            previous_outer, previous_fe = previous
            current_outer, current_fe = current
            if current_outer != previous_outer + 1:
                continue
            interval = current_fe - previous_fe
            if interval <= 0:
                raise ValueError("non-positive prior relation cycle interval")
            group_intervals.append(interval)
        completed_cycle_count += len(group_intervals)
        if group_intervals:
            latest_intervals.append(group_intervals[-1])

    return CycleProjection(
        projected_fe=max(latest_intervals) if latest_intervals else None,
        history_group_count=len(latest_intervals),
        completed_cycle_count=completed_cycle_count,
    )


def _release_observed_lease(
    *,
    active_lease: dict[str, str] | None,
    current_fe: int,
) -> tuple[dict[str, str] | None, str]:
    if active_lease is None:
        return None, ""
    status = active_lease.get("component_credit_status")
    resolution_fe = _int(
        active_lease.get("component_resolution_fe"),
        field="component_resolution_fe",
    )
    if status == "resolved" and resolution_fe <= current_fe:
        return None, active_lease.get("component_action_id", "")
    return active_lease, ""


def replay_run(
    *,
    trace_rows: list[dict[str, str]],
    budget_limit: int,
) -> list[dict[str, object]]:
    """Replay precision-action eligibility for one run in decision-FE order."""
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
    replay_rows: list[dict[str, object]] = []

    for action in actions:
        component_id = action.get("component_id", "")
        action_id = action.get("component_action_id", "")
        if not component_id or not action_id:
            raise ValueError("precision action is missing component identity")
        decision_fe = _int(
            action.get("component_decision_fe"), field="component_decision_fe"
        )
        outer_iter = _int(action.get("outer_iter"), field="outer_iter")
        group_index = _int(action.get("group_index"), field="group_index")
        remaining_fe = budget_limit - decision_fe
        if remaining_fe < 0:
            raise ValueError("action decision exceeds FE budget")

        active, released_action_id = _release_observed_lease(
            active_lease=active_by_component.get(component_id),
            current_fe=decision_fe,
        )
        if active is None:
            active_by_component.pop(component_id, None)
        else:
            active_by_component[component_id] = active
        active_action_id = active.get("component_action_id", "") if active else ""

        projection = project_next_revisit_fe(
            trace_rows=trace_rows,
            component_id=component_id,
            action_outer_iter=outer_iter,
            action_decision_fe=decision_fe,
        )
        if projection.projected_fe is None:
            abstain_reason = "abstain_insufficient_history"
        elif projection.projected_fe > remaining_fe:
            abstain_reason = "abstain_unresolvable_horizon"
        elif active is not None:
            abstain_reason = "abstain_component_mutex"
        else:
            abstain_reason = ""
        selected = not abstain_reason
        overlap_violation = int(selected and active is not None)
        if selected:
            active_by_component[component_id] = action

        observed_status = action.get("component_credit_status", "") if selected else ""
        observed_resolved = int(observed_status == "resolved") if selected else ""
        resolution_fe = (
            _int(action.get("component_resolution_fe"), field="component_resolution_fe")
            if selected
            else None
        )
        actual_delay = (
            resolution_fe - decision_fe
            if selected and observed_status == "resolved" and resolution_fe is not None
            else None
        )
        component_gain = _float(action.get("component_gain")) if selected else None
        neighbor_gain = (
            _float(action.get("component_neighbor_gain")) if selected else None
        )
        overwrite_rate = (
            _float(action.get("shared_var_overwrite_rate")) if selected else None
        )
        survival_rate = (
            _float(action.get("shared_var_survival_rate")) if selected else None
        )
        replay_rows.append(
            {
                "problem_id": action.get("problem_id", ""),
                "seed": action.get("seed", ""),
                "component_action_id": action_id,
                "component_id": component_id,
                "outer_iter": outer_iter,
                "group_index": group_index,
                "component_decision_fe": decision_fe,
                "budget_limit": budget_limit,
                "remaining_fe": remaining_fe,
                "projected_next_revisit_fe": (
                    "" if projection.projected_fe is None else projection.projected_fe
                ),
                "history_group_count": projection.history_group_count,
                "completed_cycle_count": projection.completed_cycle_count,
                "active_lease_action_id_before": active_action_id,
                "released_lease_action_id": released_action_id,
                "replay_decision": "selected" if selected else "abstained",
                "abstain_reason": abstain_reason,
                "overlap_violation": overlap_violation,
                "selected_observed_credit_status": observed_status,
                "selected_observed_resolved": observed_resolved,
                "selected_observed_resolution_fe": (
                    "" if resolution_fe is None else resolution_fe
                ),
                "selected_actual_resolution_delay_fe": (
                    "" if actual_delay is None else actual_delay
                ),
                "selected_projection_error_fe": (
                    ""
                    if actual_delay is None or projection.projected_fe is None
                    else actual_delay - projection.projected_fe
                ),
                "selected_component_gain": (
                    "" if component_gain is None else component_gain
                ),
                "selected_neighbor_gain": (
                    "" if neighbor_gain is None else neighbor_gain
                ),
                "selected_neighbor_harm": (
                    "" if neighbor_gain is None else int(neighbor_gain < 0.0)
                ),
                "selected_shared_var_overwrite_rate": (
                    "" if overwrite_rate is None else overwrite_rate
                ),
                "selected_shared_var_survival_rate": (
                    "" if survival_rate is None else survival_rate
                ),
            }
        )
    return replay_rows


def _summarize_run(
    *,
    problem_id: str,
    seed: int,
    replay_rows: list[dict[str, object]],
) -> dict[str, object]:
    selected = [row for row in replay_rows if row["replay_decision"] == "selected"]
    resolved = [
        row for row in selected if row["selected_observed_credit_status"] == "resolved"
    ]
    overwrite_rates = [
        float(row["selected_shared_var_overwrite_rate"])
        for row in selected
        if row["selected_shared_var_overwrite_rate"] != ""
    ]
    survival_rates = [
        float(row["selected_shared_var_survival_rate"])
        for row in selected
        if row["selected_shared_var_survival_rate"] != ""
    ]
    projection_errors = [
        float(row["selected_projection_error_fe"])
        for row in resolved
        if row["selected_projection_error_fe"] != ""
    ]
    reason_counts = {
        reason: sum(row["abstain_reason"] == reason for row in replay_rows)
        for reason in (
            "abstain_insufficient_history",
            "abstain_unresolvable_horizon",
            "abstain_component_mutex",
        )
    }
    return {
        "problem_id": problem_id,
        "seed": seed,
        "original_action_count": len(replay_rows),
        "selected_action_count": len(selected),
        "abstained_action_count": len(replay_rows) - len(selected),
        **reason_counts,
        "selected_resolved_count": len(resolved),
        "selected_unresolved_count": len(selected) - len(resolved),
        "selected_horizon_closure_rate": len(resolved) / len(selected) if selected else "",
        "selected_overlap_violation_count": sum(
            int(row["overlap_violation"]) for row in selected
        ),
        "selected_neighbor_harm_count": sum(
            int(row["selected_neighbor_harm"])
            for row in selected
            if row["selected_neighbor_harm"] != ""
        ),
        "selected_overwrite_observation_count": len(overwrite_rates),
        "selected_full_overwrite_count": sum(rate == 1.0 for rate in overwrite_rates),
        "selected_full_survival_count": sum(rate == 1.0 for rate in survival_rates),
        "selected_overwrite_rate_mean": _mean(overwrite_rates),
        "selected_survival_rate_mean": _mean(survival_rates),
        "selected_projection_underestimate_count": sum(
            error > 0.0 for error in projection_errors
        ),
        "selected_projection_error_fe_mean": _mean(projection_errors),
        "selected_projection_error_fe_max": (
            max(projection_errors) if projection_errors else ""
        ),
    }


def validate_s20_integrity(audit_root: Path) -> dict[str, object]:
    integrity_rows = _read_csv(audit_root / "component_credit_integrity_audit.csv")
    parity_rows = _read_csv(audit_root / "component_credit_parity_audit.csv")
    gate_rows = _read_csv(audit_root / "component_credit_gate.csv")
    if len(gate_rows) != 1:
        raise ValueError("expected one S20 gate row")
    gate = gate_rows[0]
    blockers: list[str] = []
    if not integrity_rows or any(row.get("status") != "pass" for row in integrity_rows):
        blockers.append("s20_run_integrity_failed")
    if not parity_rows or any(row.get("status") != "pass" for row in parity_rows):
        blockers.append("s20_parity_failed")
    if gate.get("anti_leakage_pass") != "1":
        blockers.append("s20_anti_leakage_failed")
    if _int(gate.get("run_count"), field="run_count") != len(integrity_rows):
        blockers.append("s20_integrity_count_mismatch")
    if _int(gate.get("integrity_pass_count"), field="integrity_pass_count") != len(
        integrity_rows
    ):
        blockers.append("s20_integrity_pass_count_mismatch")
    if _int(gate.get("parity_pass_count"), field="parity_pass_count") != len(
        parity_rows
    ):
        blockers.append("s20_parity_pass_count_mismatch")
    return {
        "s20_integrity_pass": int(not blockers),
        "s20_integrity_blockers": ";".join(blockers),
    }


def build_gate(
    *,
    run_summaries: list[dict[str, object]],
    s20_integrity: dict[str, object],
) -> dict[str, object]:
    selected_runs = [row for row in run_summaries if row["selected_action_count"]]
    selected_cases = {str(row["problem_id"]) for row in selected_runs}
    seeds_by_case: dict[str, set[int]] = defaultdict(set)
    for row in selected_runs:
        seeds_by_case[str(row["problem_id"])].add(int(row["seed"]))
    cases_with_two_seeds = sum(len(seeds) >= 2 for seeds in seeds_by_case.values())
    original_actions = sum(int(row["original_action_count"]) for row in run_summaries)
    selected_actions = sum(int(row["selected_action_count"]) for row in run_summaries)
    resolved_actions = sum(int(row["selected_resolved_count"]) for row in run_summaries)
    unresolved_actions = sum(
        int(row["selected_unresolved_count"]) for row in run_summaries
    )
    overlap_violations = sum(
        int(row["selected_overlap_violation_count"]) for row in run_summaries
    )
    neighbor_harm = sum(
        int(row["selected_neighbor_harm_count"]) for row in run_summaries
    )
    overwrite_observations = sum(
        int(row["selected_overwrite_observation_count"]) for row in run_summaries
    )
    full_overwrites = sum(
        int(row["selected_full_overwrite_count"]) for row in run_summaries
    )
    full_survivals = sum(
        int(row["selected_full_survival_count"]) for row in run_summaries
    )
    projection_underestimates = sum(
        int(row["selected_projection_underestimate_count"])
        for row in run_summaries
    )
    eligibility_boundary_pass = not (
        ELIGIBILITY_INPUT_FIELDS & FORBIDDEN_ELIGIBILITY_FIELDS
    )

    blockers: list[str] = []
    if s20_integrity["s20_integrity_pass"] != 1:
        blockers.append("s20_integrity_failed")
    if not eligibility_boundary_pass:
        blockers.append("forbidden_eligibility_input")
    if overlap_violations:
        blockers.append("selected_lease_overlap")
    if selected_actions == 0 or unresolved_actions:
        blockers.append("selected_horizon_not_closed")
    if len(selected_runs) < MIN_SELECTED_RUNS:
        blockers.append("selected_run_coverage_lt_6")
    if len(selected_cases) < MIN_SELECTED_CASES:
        blockers.append("selected_case_coverage_lt_3")
    if cases_with_two_seeds < MIN_CASES_WITH_TWO_SELECTED_SEEDS:
        blockers.append("selected_cross_seed_case_coverage_lt_2")
    return {
        "overall_status": "pass" if not blockers else "fail",
        "evidence_scope": "exploratory_feasibility_only",
        "s20_integrity_pass": s20_integrity["s20_integrity_pass"],
        "eligibility_boundary_pass": int(eligibility_boundary_pass),
        "current_action_outcome_used_for_eligibility": 0,
        "run_count": len(run_summaries),
        "original_action_count": original_actions,
        "selected_action_count": selected_actions,
        "abstained_action_count": original_actions - selected_actions,
        "selected_run_count": len(selected_runs),
        "selected_case_count": len(selected_cases),
        "cases_with_two_selected_seeds": cases_with_two_seeds,
        "selected_resolved_count": resolved_actions,
        "selected_unresolved_count": unresolved_actions,
        "selected_horizon_closure_rate": (
            resolved_actions / selected_actions if selected_actions else 0.0
        ),
        "selected_overlap_violation_count": overlap_violations,
        "selected_neighbor_harm_count": neighbor_harm,
        "selected_overwrite_observation_count": overwrite_observations,
        "selected_full_overwrite_count": full_overwrites,
        "selected_full_survival_count": full_survivals,
        "selected_projection_underestimate_count": projection_underestimates,
        "selected_projection_underestimate_rate": (
            projection_underestimates / resolved_actions if resolved_actions else 0.0
        ),
        "blockers": ";".join(blockers),
    }


def write_reports(
    *,
    v40_root: Path,
    s20_audit_root: Path,
    output_root: Path,
) -> tuple[Path, ...]:
    traces_by_run = _group_by_run(_read_csv(v40_root / "action_trace.csv"))
    ledgers = _ledger_by_run(_read_csv(v40_root / "same_budget_ledger.csv"))
    if set(traces_by_run) != set(ledgers):
        raise ValueError("trace and FE-ledger run matrices differ")

    all_replay_rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for problem_id, seed in sorted(ledgers):
        budget_limit = _int(
            ledgers[(problem_id, seed)].get("budget_limit"), field="budget_limit"
        )
        replay_rows = replay_run(
            trace_rows=traces_by_run[(problem_id, seed)],
            budget_limit=budget_limit,
        )
        all_replay_rows.extend(replay_rows)
        run_summaries.append(
            _summarize_run(
                problem_id=problem_id,
                seed=seed,
                replay_rows=replay_rows,
            )
        )

    s20_integrity = validate_s20_integrity(s20_audit_root)
    gate = build_gate(run_summaries=run_summaries, s20_integrity=s20_integrity)
    paths = (
        output_root / "component_lease_action_replay.csv",
        output_root / "component_lease_run_summary.csv",
        output_root / "component_lease_feasibility_gate.csv",
    )
    for path, rows in zip(
        paths,
        (all_replay_rows, run_summaries, [gate]),
        strict=True,
    ):
        _write_csv(path, rows)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v40-dir", type=Path, required=True)
    parser.add_argument("--s20-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = write_reports(
        v40_root=args.v40_dir,
        s20_audit_root=args.s20_audit_dir,
        output_root=args.output_dir,
    )
    gate = _read_csv(paths[-1])[0]
    if gate["overall_status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
