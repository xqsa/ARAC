"""Audit held-out v40 component delayed-credit trace coverage.

This script is offline-only. It validates raw artifacts after optimizer
execution and must never be imported by runtime dispatch code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


PRECISION_ACTION = "post_retirement_precision_reanchor"
COMPONENT_STATUSES = {
    "relation_observation",
    "resolved",
    "unresolved_run_end",
}
COMPONENT_TRACE_FIELDS = {
    "component_id",
    "component_group_count",
    "component_shared_var_count",
    "component_action_id",
    "component_action_scope",
    "component_credit_status",
    "component_decision_fe",
    "component_remaining_budget_ratio",
    "component_resolution_fe",
    "component_resolution_delay_fe",
    "component_resolution_window",
    "component_pending_before",
    "component_lock_conflict",
    "component_proposal_disagreement",
    "component_local_gain",
    "component_gain",
    "component_neighbor_gain",
    "component_neighbor_spillover",
    "shared_var_overwrite_rate",
    "shared_var_survival_rate",
    "component_credit_reason",
}

MIN_PRECISION_RUNS = 6
MIN_CASES_WITH_TWO_PRECISION_SEEDS = 2
MIN_RESOLUTION_RATE = 0.90
MIN_OVERWRITE_RUNS = 3
MIN_OVERWRITE_CASES = 2


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> float | str:
    return fmean(values) if values else ""


def _group_by_run(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        seed = _int(row.get("seed"))
        if not row.get("problem_id") or seed is None:
            raise ValueError("run-key fields are missing")
        grouped[(row["problem_id"], seed)].append(row)
    return grouped


def _one_by_run(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    grouped = _group_by_run(rows)
    duplicates = sorted(key for key, values in grouped.items() if len(values) != 1)
    if duplicates:
        raise ValueError(f"expected one row per run: {duplicates}")
    return {key: values[0] for key, values in grouped.items()}


def _plans_by_problem(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        problem_id = row.get("problem_id", "")
        if not problem_id:
            raise ValueError("action execution plan is missing problem_id")
        grouped[problem_id].append(row)
    output: dict[str, dict[str, str]] = {}
    for problem_id, repeated in grouped.items():
        first = repeated[0]
        if any(row != first for row in repeated[1:]):
            raise ValueError(
                f"inconsistent action execution plans for {problem_id}"
            )
        output[problem_id] = first
    return output


def _add(blockers: list[str], condition: bool, blocker: str) -> None:
    if condition and blocker not in blockers:
        blockers.append(blocker)


def _validate_fraction(value: object) -> float | None:
    number = _float(value)
    if number is None or number < 0.0 or number > 1.0:
        return None
    return number


def audit_run(
    *,
    result: dict[str, str],
    ledger: dict[str, str],
    trace_rows: list[dict[str, str]],
    aob_rows: list[dict[str, str]],
    plan: dict[str, str],
) -> tuple[dict[str, object], list[str]]:
    blockers: list[str] = []
    problem_id = result.get("problem_id", "")
    seed = _int(result.get("seed"))
    actual_fe = _int(result.get("hcc_smoke_fe_used"))
    budget = _int(ledger.get("budget_limit"))

    _add(blockers, not problem_id or seed is None, "invalid_run_key")
    _add(blockers, result.get("hcc_smoke_status") != "completed", "run_not_completed")
    _add(
        blockers,
        result.get("fresh_optimizer_execution") != "1",
        "optimizer_execution_not_fresh",
    )
    _add(blockers, ledger.get("fresh_execution") != "1", "ledger_not_fresh")
    _add(blockers, ledger.get("same_budget_violation") != "0", "fe_overspend")
    _add(
        blockers,
        actual_fe is None or budget is None or actual_fe > budget,
        "invalid_fe_ledger",
    )
    _add(
        blockers,
        actual_fe != _int(ledger.get("actual_fe_used"))
        or actual_fe != _int(ledger.get("total_fe")),
        "result_ledger_fe_mismatch",
    )

    _add(blockers, not aob_rows, "missing_aob_manifest")
    _add(
        blockers,
        any(
            row.get("unchanged") != "1"
            or row.get("sha256_before") != row.get("sha256_after")
            for row in aob_rows
        ),
        "aob_input_changed",
    )

    try:
        parameters = json.loads(plan.get("optimizer_consumed_parameters", ""))
    except json.JSONDecodeError:
        parameters = {}
        _add(blockers, True, "invalid_action_plan_parameters")
    _add(blockers, plan.get("optimizer_consumed") != "1", "plan_not_consumed")
    _add(
        blockers,
        plan.get("runtime_dispatch_allowed") != "1",
        "runtime_dispatch_not_allowed",
    )
    _add(
        blockers,
        parameters.get("trace_affects_dispatch") is not False,
        "trace_may_affect_dispatch",
    )
    _add(
        blockers,
        parameters.get("optimizer_runtime_hook") != PRECISION_ACTION,
        "unexpected_optimizer_runtime_hook",
    )
    _add(
        blockers,
        "cross_sweep_cma_terminal_sigma_continuation" in str(parameters),
        "v39_sigma_continuation_present",
    )

    relation_rows = [
        row
        for row in trace_rows
        if row.get("component_credit_status") == "relation_observation"
    ]
    precision_rows = [
        row for row in trace_rows if row.get("selected_action_name") == PRECISION_ACTION
    ]
    action_rows = [row for row in trace_rows if row.get("component_action_id")]
    component_statuses = {
        row.get("component_credit_status", "")
        for row in trace_rows
        if row.get("component_credit_status")
    }
    _add(
        blockers,
        bool(component_statuses - COMPONENT_STATUSES),
        "invalid_component_credit_status",
    )
    _add(
        blockers,
        len(action_rows) != len(precision_rows),
        "precision_action_trace_mismatch",
    )

    proposal_disagreements: list[float] = []
    for row in relation_rows:
        disagreement = _validate_fraction(row.get("component_proposal_disagreement"))
        decision_fe = _int(row.get("component_decision_fe"))
        remaining = _validate_fraction(row.get("component_remaining_budget_ratio"))
        _add(
            blockers,
            not row.get("component_id")
            or _int(row.get("component_group_count")) is None
            or _int(row.get("component_shared_var_count")) is None,
            "invalid_relation_component_topology",
        )
        _add(
            blockers,
            row.get("component_action_scope") != "shared_relation_observation",
            "invalid_relation_action_scope",
        )
        _add(
            blockers,
            disagreement is None,
            "invalid_proposal_disagreement",
        )
        _add(
            blockers,
            decision_fe is None
            or actual_fe is None
            or decision_fe < 0
            or decision_fe > actual_fe,
            "invalid_relation_decision_fe",
        )
        _add(blockers, remaining is None, "invalid_relation_remaining_budget")
        if disagreement is not None:
            proposal_disagreements.append(disagreement)

    resolved_rows = 0
    unresolved_rows = 0
    overwrite_rates: list[float] = []
    survival_rates: list[float] = []
    local_gains: list[float] = []
    component_gains: list[float] = []
    neighbor_gains: list[float] = []
    delays: list[float] = []
    lock_conflicts = 0
    pending_before_values: list[int] = []
    neighbor_harm_rows = 0
    for row in precision_rows:
        status = row.get("component_credit_status")
        decision_fe = _int(row.get("component_decision_fe"))
        resolution_fe = _int(row.get("component_resolution_fe"))
        delay = _int(row.get("component_resolution_delay_fe"))
        pending_before = _int(row.get("component_pending_before"))
        lock_conflict = _int(row.get("component_lock_conflict"))
        local_gain = _float(row.get("component_local_gain"))
        remaining = _validate_fraction(row.get("component_remaining_budget_ratio"))
        _add(
            blockers,
            status not in {"resolved", "unresolved_run_end"},
            "precision_credit_not_finalized",
        )
        _add(
            blockers,
            not row.get("component_action_id", "").startswith(PRECISION_ACTION + ":"),
            "invalid_component_action_id",
        )
        _add(
            blockers,
            not row.get("component_id")
            or _int(row.get("component_group_count")) is None
            or _int(row.get("component_shared_var_count")) is None,
            "invalid_precision_component_topology",
        )
        _add(
            blockers,
            row.get("component_action_scope")
            != "group_search_start_component_credit",
            "invalid_precision_action_scope",
        )
        _add(
            blockers,
            row.get("component_resolution_window")
            != "next_canonical_group_revisit",
            "invalid_resolution_window",
        )
        _add(blockers, remaining is None, "invalid_precision_remaining_budget")
        _add(blockers, local_gain is None, "invalid_local_gain")
        _add(
            blockers,
            decision_fe is None
            or resolution_fe is None
            or delay is None
            or actual_fe is None
            or resolution_fe < decision_fe
            or resolution_fe > actual_fe
            or delay != resolution_fe - decision_fe,
            "non_monotonic_resolution_fe",
        )
        _add(
            blockers,
            pending_before is None
            or pending_before < 0
            or lock_conflict not in {0, 1}
            or lock_conflict != int(pending_before > 0),
            "invalid_component_lock_state",
        )
        if local_gain is not None:
            local_gains.append(local_gain)
        if delay is not None:
            delays.append(float(delay))
        if pending_before is not None:
            pending_before_values.append(pending_before)
        lock_conflicts += int(lock_conflict == 1)

        overwrite = _validate_fraction(row.get("shared_var_overwrite_rate"))
        survival = _validate_fraction(row.get("shared_var_survival_rate"))
        overwrite_present = str(row.get("shared_var_overwrite_rate", "")).strip() != ""
        survival_present = str(row.get("shared_var_survival_rate", "")).strip() != ""
        _add(
            blockers,
            overwrite_present != survival_present,
            "partial_overwrite_survival_pair",
        )
        if overwrite_present and survival_present:
            _add(
                blockers,
                overwrite is None
                or survival is None
                or not math.isclose(overwrite + survival, 1.0, abs_tol=1e-9),
                "overwrite_survival_not_complementary",
            )
            if overwrite is not None and survival is not None:
                overwrite_rates.append(overwrite)
                survival_rates.append(survival)

        if status == "resolved":
            resolved_rows += 1
            component_gain = _float(row.get("component_gain"))
            neighbor_gain = _float(row.get("component_neighbor_gain"))
            spillover = _float(row.get("component_neighbor_spillover"))
            _add(
                blockers,
                component_gain is None
                or neighbor_gain is None
                or spillover is None
                or spillover < 0.0,
                "invalid_resolved_credit",
            )
            _add(
                blockers,
                not overwrite_present
                and row.get("component_credit_reason")
                not in {
                    "resolved_no_shared_variables",
                    "resolved_no_shared_variable_displacement",
                },
                "missing_overwrite_survival_observation",
            )
            if component_gain is not None:
                component_gains.append(component_gain)
            if neighbor_gain is not None:
                neighbor_gains.append(neighbor_gain)
                neighbor_harm_rows += int(neighbor_gain < 0.0)
        elif status == "unresolved_run_end":
            unresolved_rows += 1
            _add(
                blockers,
                any(
                    str(row.get(field, "")).strip()
                    for field in (
                        "component_gain",
                        "component_neighbor_gain",
                        "component_neighbor_spillover",
                    )
                ),
                "fabricated_unresolved_credit",
            )
            _add(
                blockers,
                row.get("component_credit_reason")
                != "budget_ended_before_next_group_revisit",
                "invalid_unresolved_reason",
            )

    summary = {
        "problem_id": problem_id,
        "seed": "" if seed is None else seed,
        "actual_fe": "" if actual_fe is None else actual_fe,
        "budget_limit": "" if budget is None else budget,
        "trace_rows": len(trace_rows),
        "relation_observation_rows": len(relation_rows),
        "precision_rows": len(precision_rows),
        "resolved_rows": resolved_rows,
        "unresolved_rows": unresolved_rows,
        "resolution_rate": (
            resolved_rows / len(precision_rows) if precision_rows else ""
        ),
        "overwrite_observation_rows": len(overwrite_rates),
        "lock_conflict_rows": lock_conflicts,
        "neighbor_harm_rows": neighbor_harm_rows,
        "proposal_disagreement_mean": _mean(proposal_disagreements),
        "local_gain_mean": _mean(local_gains),
        "component_gain_mean": _mean(component_gains),
        "neighbor_gain_mean": _mean(neighbor_gains),
        "overwrite_rate_mean": _mean(overwrite_rates),
        "survival_rate_mean": _mean(survival_rates),
        "resolution_delay_fe_mean": _mean(delays),
        "resolution_delay_fe_max": max(delays) if delays else "",
        "pending_before_max": max(pending_before_values) if pending_before_values else "",
    }
    return summary, blockers


def compare_parity_run(
    *,
    v38_result: dict[str, str],
    v40_result: dict[str, str],
    v38_trace: list[dict[str, str]],
    v40_trace: list[dict[str, str]],
    v38_aob: list[dict[str, str]],
    v40_aob: list[dict[str, str]],
) -> dict[str, object]:
    blockers: list[str] = []
    final_equal = (
        v38_result.get("hcc_smoke_final_error")
        == v40_result.get("hcc_smoke_final_error")
    )
    fe_equal = v38_result.get("hcc_smoke_fe_used") == v40_result.get(
        "hcc_smoke_fe_used"
    )
    _add(blockers, not final_equal, "final_error_mismatch")
    _add(blockers, not fe_equal, "fe_mismatch")
    _add(
        blockers,
        v38_result.get("fresh_optimizer_execution") != "1"
        or v40_result.get("fresh_optimizer_execution") != "1",
        "parity_execution_not_fresh",
    )

    common_fields = (
        [field for field in v38_trace[0] if field != "lane_id"]
        if v38_trace
        else []
    )
    _add(blockers, not common_fields, "missing_v38_trace")
    _add(blockers, not v40_trace, "missing_v40_trace")
    missing_fields = (
        [field for field in common_fields if field not in v40_trace[0]]
        if v40_trace
        else common_fields
    )
    _add(blockers, bool(missing_fields), "missing_common_trace_fields")
    common_differences = abs(len(v38_trace) - len(v40_trace))
    for left, right in zip(v38_trace, v40_trace, strict=False):
        common_differences += int(
            any(left.get(field, "") != right.get(field, "") for field in common_fields)
        )
    _add(blockers, common_differences != 0, "common_trace_mismatch")

    def aob_signature(rows: list[dict[str, str]]) -> list[tuple[str, ...]]:
        return sorted(
            (
                row.get("file", ""),
                row.get("sha256_before", ""),
                row.get("sha256_after", ""),
                row.get("unchanged", ""),
            )
            for row in rows
        )

    aob_equal = bool(v38_aob) and aob_signature(v38_aob) == aob_signature(v40_aob)
    _add(blockers, not aob_equal, "aob_manifest_mismatch")
    return {
        "problem_id": v40_result.get("problem_id", ""),
        "seed": v40_result.get("seed", ""),
        "final_error_equal": int(final_equal),
        "fe_equal": int(fe_equal),
        "common_trace_fields": len(common_fields),
        "v38_trace_rows": len(v38_trace),
        "v40_trace_rows": len(v40_trace),
        "common_trace_differences": common_differences,
        "aob_manifest_equal": int(aob_equal),
        "status": "pass" if not blockers else "fail",
        "blockers": ";".join(blockers),
    }


def build_gate(
    *,
    summaries: list[dict[str, object]],
    integrity_rows: list[dict[str, object]],
    parity_rows: list[dict[str, object]],
    anti_leakage_pass: bool,
) -> dict[str, object]:
    blockers: list[str] = []
    precision_runs = [row for row in summaries if int(row["precision_rows"]) > 0]
    precision_seeds_by_case: dict[str, set[int]] = defaultdict(set)
    overwrite_cases: set[str] = set()
    overwrite_runs = 0
    for row in summaries:
        if int(row["precision_rows"]) > 0:
            precision_seeds_by_case[str(row["problem_id"])].add(int(row["seed"]))
        if int(row["overwrite_observation_rows"]) > 0:
            overwrite_runs += 1
            overwrite_cases.add(str(row["problem_id"]))
    cases_with_two_seeds = sum(
        len(seeds) >= 2 for seeds in precision_seeds_by_case.values()
    )
    precision_rows = sum(int(row["precision_rows"]) for row in summaries)
    resolved_rows = sum(int(row["resolved_rows"]) for row in summaries)
    unresolved_rows = sum(int(row["unresolved_rows"]) for row in summaries)
    resolution_rate = resolved_rows / precision_rows if precision_rows else 0.0
    relation_runs = sum(
        int(row["relation_observation_rows"]) > 0 for row in summaries
    )
    lock_conflicts = sum(int(row["lock_conflict_rows"]) for row in summaries)
    precision_without_resolution = sum(
        int(row["precision_rows"]) > 0 and int(row["resolved_rows"]) == 0
        for row in summaries
    )

    _add(
        blockers,
        not integrity_rows or any(row.get("status") != "pass" for row in integrity_rows),
        "run_integrity_failed",
    )
    _add(blockers, not anti_leakage_pass, "anti_leakage_failed")
    _add(
        blockers,
        not parity_rows or any(row.get("status") != "pass" for row in parity_rows),
        "v38_v40_parity_failed",
    )
    _add(
        blockers,
        len(precision_runs) < MIN_PRECISION_RUNS,
        "insufficient_precision_run_coverage",
    )
    _add(
        blockers,
        cases_with_two_seeds < MIN_CASES_WITH_TWO_PRECISION_SEEDS,
        "insufficient_cross_seed_case_coverage",
    )
    _add(
        blockers,
        relation_runs != len(summaries),
        "missing_relation_component_observation",
    )
    _add(blockers, resolved_rows == 0, "missing_resolved_credit")
    _add(blockers, unresolved_rows == 0, "missing_run_end_unresolved_credit")
    _add(
        blockers,
        resolution_rate < MIN_RESOLUTION_RATE,
        "resolution_rate_below_preregistered_floor",
    )
    _add(
        blockers,
        precision_without_resolution > 0,
        "precision_run_without_resolved_credit",
    )
    _add(
        blockers,
        overwrite_runs < MIN_OVERWRITE_RUNS,
        "insufficient_overwrite_run_coverage",
    )
    _add(
        blockers,
        len(overwrite_cases) < MIN_OVERWRITE_CASES,
        "insufficient_overwrite_case_coverage",
    )
    _add(blockers, lock_conflicts == 0, "missing_lock_conflict_observation")
    return {
        "overall_status": "pass" if not blockers else "fail",
        "run_count": len(summaries),
        "integrity_pass_count": sum(
            row.get("status") == "pass" for row in integrity_rows
        ),
        "parity_pass_count": sum(row.get("status") == "pass" for row in parity_rows),
        "anti_leakage_pass": int(anti_leakage_pass),
        "precision_run_count": len(precision_runs),
        "cases_with_two_precision_seeds": cases_with_two_seeds,
        "relation_observation_run_count": relation_runs,
        "precision_rows": precision_rows,
        "resolved_rows": resolved_rows,
        "unresolved_rows": unresolved_rows,
        "resolution_rate": resolution_rate,
        "overwrite_run_count": overwrite_runs,
        "overwrite_case_count": len(overwrite_cases),
        "lock_conflict_rows": lock_conflicts,
        "blockers": ";".join(blockers),
    }


def _load_root(
    root: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, str]],
    dict[tuple[str, int], dict[str, str]],
    dict[tuple[str, int], list[dict[str, str]]],
    dict[tuple[str, int], list[dict[str, str]]],
    dict[str, dict[str, str]],
]:
    results = _one_by_run(_read_csv(root / "our_result_by_case.csv"))
    ledgers = _one_by_run(_read_csv(root / "same_budget_ledger.csv"))
    traces = _group_by_run(_read_csv(root / "action_trace.csv"))
    aob = _group_by_run(_read_csv(root / "aob_input_manifest.csv"))
    plans = _plans_by_problem(_read_csv(root / "action_execution_plan.csv"))
    return results, ledgers, traces, aob, plans


def audit_v40_root(
    root: Path,
    expected_keys: set[tuple[str, int]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], bool]:
    results, ledgers, traces, aob, plans = _load_root(root)
    observed_keys = set(results)
    if observed_keys != expected_keys:
        raise ValueError(
            f"v40 matrix mismatch: missing={sorted(expected_keys - observed_keys)};"
            f"unexpected={sorted(observed_keys - expected_keys)}"
        )
    summaries: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    for key in sorted(expected_keys):
        if key not in ledgers or key not in traces or key not in aob:
            raise ValueError(f"missing v40 run artifact: {key}")
        if key[0] not in plans:
            raise ValueError(f"missing v40 action plan: {key[0]}")
        summary, blockers = audit_run(
            result=results[key],
            ledger=ledgers[key],
            trace_rows=traces[key],
            aob_rows=aob[key],
            plan=plans[key[0]],
        )
        summaries.append(summary)
        integrity_rows.append(
            {
                "problem_id": key[0],
                "seed": key[1],
                "status": "pass" if not blockers else "fail",
                "blockers": ";".join(blockers),
            }
        )
    anti_rows = _read_csv(root / "anti_leakage_audit.csv")
    anti_pass = bool(anti_rows) and all(
        row.get("audit_status") == "pass"
        and row.get("found_in_runtime_payload") == "0"
        and row.get("runtime_dispatch_allowed") == "1"
        for row in anti_rows
    )
    return summaries, integrity_rows, anti_pass


def compare_parity_roots(
    *,
    v38_root: Path,
    v40_root: Path,
    expected_keys: set[tuple[str, int]],
) -> list[dict[str, object]]:
    v38_results, _v38_ledgers, v38_traces, v38_aob, _v38_plans = _load_root(v38_root)
    v40_results, _v40_ledgers, v40_traces, v40_aob, _v40_plans = _load_root(v40_root)
    if set(v38_results) != expected_keys:
        raise ValueError("v38 parity matrix mismatch")
    if not expected_keys.issubset(v40_results):
        raise ValueError("v40 parity runs are missing")
    return [
        compare_parity_run(
            v38_result=v38_results[key],
            v40_result=v40_results[key],
            v38_trace=v38_traces[key],
            v40_trace=v40_traces[key],
            v38_aob=v38_aob[key],
            v40_aob=v40_aob[key],
        )
        for key in sorted(expected_keys)
    ]


def write_reports(
    *,
    v40_root: Path,
    v38_parity_root: Path,
    expected_problems: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    parity_seed: int,
    output_root: Path,
) -> tuple[Path, ...]:
    expected_keys = {
        (problem_id, seed)
        for problem_id in expected_problems
        for seed in expected_seeds
    }
    summaries, integrity_rows, anti_pass = audit_v40_root(v40_root, expected_keys)
    parity_rows = compare_parity_roots(
        v38_root=v38_parity_root,
        v40_root=v40_root,
        expected_keys={(problem_id, parity_seed) for problem_id in expected_problems},
    )
    gate = build_gate(
        summaries=summaries,
        integrity_rows=integrity_rows,
        parity_rows=parity_rows,
        anti_leakage_pass=anti_pass,
    )
    paths = (
        output_root / "component_credit_run_summary.csv",
        output_root / "component_credit_integrity_audit.csv",
        output_root / "component_credit_parity_audit.csv",
        output_root / "component_credit_gate.csv",
    )
    for path, rows in zip(
        paths,
        (summaries, integrity_rows, parity_rows, [gate]),
        strict=True,
    ):
        _write_csv(path, rows)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v40-dir", type=Path, required=True)
    parser.add_argument("--v38-parity-dir", type=Path, required=True)
    parser.add_argument("--problems", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--parity-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.parity_seed not in args.seeds:
        parser.error("--parity-seed must be one of --seeds")
    paths = write_reports(
        v40_root=args.v40_dir,
        v38_parity_root=args.v38_parity_dir,
        expected_problems=tuple(args.problems),
        expected_seeds=tuple(args.seeds),
        parity_seed=args.parity_seed,
        output_root=args.output_dir,
    )
    gate = _read_csv(paths[-1])[0]
    if gate["overall_status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
