"""Recover SMP's historical long group-visit schedule without an HCC runtime."""

# Thread caps must be applied before importing NumPy or pypop7.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping
import warnings

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from arac.actions._execution import (
    STATE_STALE_WINDOW,
    _PersistentBlockSession,
    _block_population_size,
    _log_improvement,
    terminal_result,
)
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, canonical_sha256
from experiments.historical_recovery.independent_semantic_parity_pilot import (
    REPOSITORY_ROOT,
    _context,
    _file_sha256,
    _hcc_runtime_imports,
    _load_json,
    _write_json_atomic,
)
from experiments.historical_recovery.independent_terminal_parity import nearest_rank_p90
from experiments.historical_recovery.replay import _checkpoint


DEFAULT_PROTOCOL = Path(__file__).with_name(
    "independent_smp_schedule_recovery_protocol_v2.json"
)
TRACE_MANIFEST_SCHEMA = "arac-independent-smp-schedule-trace-manifest-v2"
TRACE_RECEIPT_SCHEMA = "arac-independent-smp-schedule-trace-receipt-v2"
TRACE_SUMMARY_SCHEMA = "arac-independent-smp-schedule-trace-summary-v2"
TERMINAL_MANIFEST_SCHEMA = "arac-independent-smp-schedule-terminal-manifest-v2"
TERMINAL_RECEIPT_SCHEMA = "arac-independent-smp-schedule-terminal-receipt-v2"
TERMINAL_SUMMARY_SCHEMA = "arac-independent-smp-schedule-terminal-summary-v2"
SOURCE_PATHS = (
    "experiments/historical_recovery/independent_smp_schedule_recovery.py",
    "experiments/historical_recovery/independent_smp_schedule_recovery_protocol_v2.json",
    "experiments/historical_recovery/independent_semantic_parity_pilot.py",
    "src/arac/actions/_execution.py",
    "src/arac/benchmarks/aob.py",
    "src/arac/runtime/contracts.py",
    "src/arac/runtime/ledger.py",
)
HISTORICAL_MATERIAL_LOG_GAIN = 0.009950330853168092


def _resolved(relative: str) -> Path:
    return REPOSITORY_ROOT / relative


def _verify_file(relative: str, expected_sha256: str) -> Path:
    path = _resolved(relative)
    if not path.is_file() or _file_sha256(path) != expected_sha256:
        raise ValueError(f"source artifact drifted: {relative}")
    return path


def _verify_payload_hash(payload: Mapping[str, Any], key: str, label: str) -> None:
    body = dict(payload)
    claimed = body.pop(key, None)
    if claimed != canonical_sha256(body):
        raise ValueError(f"{label} hash drifted")


def _git_source_evidence() -> dict[str, Any]:
    target = "ba5f5fa3:scripts/hcc_smoke_runner.py"
    completed = subprocess.run(
        ["git", "show", target],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    source = completed.stdout
    markers = (
        "sub_fes = math.ceil(max(0, cc_budget_limit_fes - current_fes) / sub_num)",
        '"max_function_evaluations": optimizer_budget',
        "persistence_initial_state = previous_distribution_state",
        "persistent_group_state_lifecycle.record_visit(",
        "requested_fes=optimizer_budget",
        "actual_fes=primary_cc_fe",
        "cmaes_restart: bool = True",
        '"is_restart": config.cmaes_restart,',
    )
    missing = [marker for marker in markers if marker not in source]
    if missing:
        raise ValueError(f"historical long-visit source markers drifted: {missing}")
    return {
        "git_object": target,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "markers_verified": list(markers),
    }


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != "arac-independent-smp-schedule-recovery-protocol-v2":
        raise ValueError("SMP schedule recovery protocol schema drifted")
    anchors = {
        "status": "frozen_design_not_run",
        "case_id": "E1",
        "checkpoint_seed": 117,
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "trace_step_fes": 120_000,
        "native_threads": 1,
        "historical_p90": 1.8255606813339802,
        "production_hcc_dependency_allowed": False,
        "production_smp_modification_allowed": False,
        "selector_execution_allowed": False,
    }
    if any(protocol.get(key) != value for key, value in anchors.items()):
        raise ValueError("SMP schedule recovery protocol anchors drifted")
    schedule = protocol.get("schedule_contract", {})
    if (
        schedule.get("early_stopping_evaluations") != 1000
        or schedule.get("material_log_gain") != HISTORICAL_MATERIAL_LOG_GAIN
        or schedule.get("stale_window") != 3
        or schedule.get("mean_rule") != "incumbent_slice_at_visit_start"
    ):
        raise ValueError("SMP schedule contract drifted")
    trace_gate = protocol.get("trace_gate", {})
    if (
        trace_gate.get("minimum_median_population_batches_per_visit") != 10
        or trace_gate.get("maximum_terminal_visits_per_group") != 32
    ):
        raise ValueError("SMP trace gate drifted")
    numerical = protocol.get("numerical_restart_contract", {})
    if (
        numerical.get("enabled") is not True
        or numerical.get("seed_namespace") != "independent-smp-long-visit-v2"
        or numerical.get("maximum_consecutive_zero_fe_restarts") != 3
        or numerical.get("fe_accounting")
        != "retain_consumed_fes_and_never_replay"
        or numerical.get("visit_accounting")
        != "continue_same_group_visit_without_increment"
        or numerical.get("stagnation_accounting")
        != "do_not_change_group_stagnation_streak"
    ):
        raise ValueError("SMP numerical restart contract drifted")
    return protocol


def _historical_facts(protocol: Mapping[str, Any]) -> dict[str, Any]:
    reference = protocol["historical_reference"]
    action_path = _verify_file(reference["action_path"], reference["action_sha256"])
    summary_path = _verify_file(
        reference["run_summary_path"], reference["run_summary_sha256"]
    )
    receipt_path = _verify_file(
        reference["execution_receipt_path"], reference["execution_receipt_sha256"]
    )
    budget_path = _verify_file(
        reference["budget_summary_path"], reference["budget_summary_sha256"]
    )
    action = _load_json(action_path)
    action_contract = action.get("action", {})
    events = action.get("events", [])
    records = [event for event in events if event.get("event") == "record"]
    restores = [event for event in events if event.get("event") == "restore"]
    cold_starts = [event for event in events if event.get("event") == "cold_start"]
    resets = [event for event in records if event.get("state_retained") is False]
    visits_by_group: dict[int, int] = {}
    for event in records:
        group_index = int(event["group_index"])
        visits_by_group[group_index] = visits_by_group.get(group_index, 0) + 1
    required_state_fields = {
        "covariance",
        "eigenvectors",
        "eigenvalues",
        "path_covariance",
        "path_sigma",
        "sigma",
        "mean",
        "generation",
        "rng_state",
    }
    if (
        action_contract.get("name") != "smp"
        or action_contract.get("stale_window") != 3
        or action_contract.get("material_log_gain") != HISTORICAL_MATERIAL_LOG_GAIN
        or action_contract.get("mean_rule") != "incumbent_slice"
        or set(action_contract.get("state_fields", [])) != required_state_fields
        or len(records) != reference["record_count"]
        or len(restores) != reference["restore_count"]
        or len(resets) != reference["reset_count"]
        or len(visits_by_group) != reference["group_count"]
        or min(visits_by_group.values()) != reference["min_visits_per_group"]
        or max(visits_by_group.values()) != reference["max_visits_per_group"]
        or len(cold_starts) + len(restores) != len(records)
        or any(event.get("reset_reason") != "stagnation_window_reached" for event in resets)
    ):
        raise ValueError("EXP-052 SMP lifecycle evidence drifted")
    run_summary = _load_json(summary_path)
    execution_receipt = _load_json(receipt_path)
    if (
        run_summary.get("fitness_evaluations") != 3_000_000
        or run_summary.get("final_error") != reference["seed117_final_error"]
        or execution_receipt.get("runner_sha256") != reference["runner_sha256"]
        or reference.get("exact_runner_source_available") is not False
    ):
        raise ValueError("EXP-052 terminal receipt drifted")
    with budget_path.open("r", encoding="utf-8", newline="") as handle:
        budget_rows = list(csv.DictReader(handle))
    if len(budget_rows) != 1 or int(budget_rows[0]["cc_phase_fe"]) != reference["cc_phase_fes"]:
        raise ValueError("EXP-052 budget summary drifted")

    errors = []
    distribution_hashes = {}
    for row in protocol["historical_distribution"]:
        path = _verify_file(row["path"], row["sha256"])
        payload = _load_json(path)
        if payload.get("seed") != row["seed"] or payload.get("final_error") != row["final_error"]:
            raise ValueError("EXP-052 historical distribution drifted")
        errors.append(float(row["final_error"]))
        distribution_hashes[row["path"]] = row["sha256"]
    p90 = nearest_rank_p90(errors)
    if p90 != protocol["historical_p90"]:
        raise ValueError("EXP-052 historical P90 drifted")
    return {
        "record_count": len(records),
        "restore_count": len(restores),
        "cold_start_count": len(cold_starts),
        "reset_count": len(resets),
        "group_count": len(visits_by_group),
        "visits_per_group": [visits_by_group[index] for index in sorted(visits_by_group)],
        "cc_phase_fes": int(budget_rows[0]["cc_phase_fe"]),
        "mean_cc_fes_per_visit": int(budget_rows[0]["cc_phase_fe"]) / len(records),
        "seed117_final_error": float(run_summary["final_error"]),
        "historical_p90": p90,
        "runner_sha256": execution_receipt["runner_sha256"],
        "exact_runner_source_available": False,
        "distribution_hashes": distribution_hashes,
    }


def _checkpoint_input(protocol: Mapping[str, Any]):
    checkpoint_spec = protocol["checkpoint"]
    path = _verify_file(checkpoint_spec["path"], checkpoint_spec["sha256"])
    wrapper = _load_json(path)
    checkpoint = _checkpoint(wrapper["checkpoint"])
    if (
        wrapper.get("checkpoint_hash") != checkpoint.checkpoint_hash
        or checkpoint.run_seed != protocol["checkpoint_seed"]
        or checkpoint.total_budget_fes != protocol["total_budget_fes"]
        or checkpoint.phase1_fes != protocol["phase1_fes"]
        or len(checkpoint.blocks) != protocol["historical_reference"]["group_count"]
    ):
        raise ValueError("E1 Phase-I checkpoint drifted")
    return checkpoint


def _live_source_hashes() -> dict[str, str]:
    return {path: _file_sha256(_resolved(path)) for path in SOURCE_PATHS}


def _threadpools() -> list[dict[str, Any]]:
    return [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
        }
        for item in threadpool_info()
    ]


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _session_state_digest(session: _PersistentBlockSession) -> str:
    payload = {
        "covariance": session.covariance,
        "eigenvalues": session.eigenvalues,
        "eigenvectors": session.eigenvectors,
        "generation": session.optimizer._n_generations,
        "mean": session.mean,
        "path_covariance": session.path_covariance,
        "path_sigma": session.path_sigma,
        "rng_state": session.optimizer.rng_optimization.bit_generator.state,
        "sigma": session.optimizer.sigma,
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonfinite_session_fields(session: _PersistentBlockSession) -> list[str]:
    state = {
        "covariance": session.covariance,
        "eigenvalues": session.eigenvalues,
        "eigenvectors": session.eigenvectors,
        "mean": session.mean,
        "path_covariance": session.path_covariance,
        "path_sigma": session.path_sigma,
        "sigma": session.optimizer.sigma,
    }
    return [
        name
        for name, value in state.items()
        if not np.all(np.isfinite(np.asarray(value, dtype=float)))
    ]


def _aligned_budget(requested_fes: int, remaining_fes: int, population_size: int) -> int:
    available = min(int(requested_fes), int(remaining_fes))
    return available - available % population_size


def _long_visit(
    session: _PersistentBlockSession,
    requested_fes: int,
    *,
    maximum_consecutive_zero_fe_restarts: int,
) -> tuple[int, bool, list[dict[str, Any]]]:
    aligned = _aligned_budget(
        requested_fes, session.context.ledger.remaining, session.population_size
    )
    if aligned == 0:
        return 0, False, []
    incumbent_slice = session.context.ledger.best_x[session._dimensions]
    session.begin_visit()
    mean_recentered = np.array_equal(session.mean, incumbent_slice)
    consumed_before = session.consumed_fes
    numerical_restarts: list[dict[str, Any]] = []
    consecutive_zero_fe_restarts = 0
    while session.consumed_fes - consumed_before < aligned and not session.early_stopped:
        completes_visit = (
            session.consumed_fes + session.population_size
            >= consumed_before + aligned
        )
        ledger_before = session.context.ledger.count
        session_before = session.consumed_fes
        try:
            with np.errstate(over="raise", invalid="raise"):
                session.advance(adapt_on_early_stop=completes_visit)
                nonfinite = _nonfinite_session_fields(session)
                if nonfinite:
                    raise FloatingPointError(
                        f"non-finite CMA state fields: {','.join(nonfinite)}"
                    )
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            ledger_delta = session.context.ledger.count - ledger_before
            session_delta = session.consumed_fes - session_before
            if ledger_delta != session_delta or ledger_delta not in (
                0,
                session.population_size,
            ):
                raise RuntimeError(
                    "numerical restart encountered ambiguous FE accounting"
                ) from error
            consecutive_zero_fe_restarts = (
                consecutive_zero_fe_restarts + 1 if ledger_delta == 0 else 0
            )
            if (
                consecutive_zero_fe_restarts
                > maximum_consecutive_zero_fe_restarts
            ):
                raise RuntimeError(
                    "numerical restart made no FE progress repeatedly"
                ) from error
            optimizer_before_restart = session.optimizer
            session.restart()
            identity_changed = session.optimizer is not optimizer_before_restart
            finite_after_restart = not _nonfinite_session_fields(session)
            if not identity_changed or not finite_after_restart:
                raise RuntimeError(
                    "numerical restart did not rebuild a finite optimizer"
                ) from error
            numerical_restarts.append(
                {
                    "trigger_fe": session.context.ledger.count,
                    "visit_consumed_fes": session.consumed_fes - consumed_before,
                    "generation_consumed_fes": ledger_delta,
                    "population_size": session.population_size,
                    "exception_type": type(error).__name__,
                    "reason": str(error),
                    "optimizer_identity_changed": identity_changed,
                    "state_finite_after_restart": finite_after_restart,
                }
            )
            continue
        consecutive_zero_fe_restarts = 0
    return (
        session.consumed_fes - consumed_before,
        mean_recentered,
        numerical_restarts,
    )


def execute_schedule(
    context: ActionContext,
    requested_fes: int,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if requested_fes <= 0 or requested_fes > context.ledger.remaining:
        raise ValueError("SMP schedule budget is outside the remaining ledger")
    active_protocol = load_protocol() if protocol is None else protocol
    numerical_contract = active_protocol["numerical_restart_contract"]
    blocks = context.checkpoint.blocks
    populations = tuple(_block_population_size(len(block)) for block in blocks)
    sessions = tuple(
        _PersistentBlockSession(
            context,
            block,
            index,
            requested_fes,
            population_size=populations[index],
            seed_namespace=numerical_contract["seed_namespace"],
        )
        for index, block in enumerate(blocks)
    )
    start_fes = context.ledger.count
    target_fes = start_fes + requested_fes
    stale_streaks = [0] * len(sessions)
    visits_per_group = [0] * len(sessions)
    last_boundary_digests: list[str | None] = [None] * len(sessions)
    events = []
    restart_count = 0
    restart_identity_change_count = 0
    numerical_restart_count = 0
    numerical_restart_identity_change_count = 0
    numerical_restart_affected_visits = 0
    numerical_restart_events: list[dict[str, Any]] = []
    optimizer_identity_preserved_visits = 0
    state_update_count = 0
    cross_visit_boundary_checks = 0
    cross_visit_state_digest_matches = 0
    mean_recenter_match_count = 0
    full_sweeps = 0
    partial_sweeps = 0
    while context.ledger.count < target_fes:
        sweep_remaining = target_fes - context.ledger.count
        requested_per_group = math.ceil(sweep_remaining / len(sessions))
        completed_in_sweep = 0
        for index, session in enumerate(sessions):
            remaining = target_fes - context.ledger.count
            population = populations[index]
            if remaining < population:
                break
            visit_budget = _aligned_budget(
                max(requested_per_group, population), remaining, population
            )
            if visit_budget == 0:
                break
            optimizer_before = session.optimizer
            state_before = _session_state_digest(session)
            if last_boundary_digests[index] is not None:
                cross_visit_boundary_checks += 1
                if state_before == last_boundary_digests[index]:
                    cross_visit_state_digest_matches += 1
            best_before = context.ledger.best_error
            consumed, mean_recentered, visit_numerical_restarts = _long_visit(
                session,
                visit_budget,
                maximum_consecutive_zero_fe_restarts=numerical_contract[
                    "maximum_consecutive_zero_fe_restarts"
                ],
            )
            if consumed <= 0 or consumed > visit_budget:
                raise RuntimeError("SMP long visit returned an invalid FE count")
            state_after = _session_state_digest(session)
            if session.optimizer is optimizer_before:
                optimizer_identity_preserved_visits += 1
            if visit_numerical_restarts:
                numerical_restart_affected_visits += 1
                numerical_restart_count += len(visit_numerical_restarts)
                numerical_restart_identity_change_count += sum(
                    event["optimizer_identity_changed"]
                    for event in visit_numerical_restarts
                )
                numerical_restart_events.extend(
                    {
                        **event,
                        "sweep_index": full_sweeps + partial_sweeps,
                        "group_index": index,
                    }
                    for event in visit_numerical_restarts
                )
            if state_after != state_before:
                state_update_count += 1
            if mean_recentered:
                mean_recenter_match_count += 1
            gain = _log_improvement(best_before, context.ledger.best_error)
            stale_streaks[index] = (
                0 if gain >= HISTORICAL_MATERIAL_LOG_GAIN else stale_streaks[index] + 1
            )
            streak_at_record = stale_streaks[index]
            reset = streak_at_record >= STATE_STALE_WINDOW
            restart_state_digest = None
            if reset:
                optimizer_before_restart = session.optimizer
                session.restart()
                restart_state_digest = _session_state_digest(session)
                if session.optimizer is not optimizer_before_restart:
                    restart_identity_change_count += 1
                stale_streaks[index] = 0
                restart_count += 1
            last_boundary_digests[index] = _session_state_digest(session)
            visits_per_group[index] += 1
            completed_in_sweep += 1
            events.append(
                {
                    "sweep_index": full_sweeps + partial_sweeps,
                    "group_index": index,
                    "start_fe": context.ledger.count - consumed,
                    "requested_fes": visit_budget,
                    "actual_fes": consumed,
                    "population_size": population,
                    "population_batches": consumed // population,
                    "input_state_digest": state_before,
                    "output_state_digest": state_after,
                    "restart_state_digest": restart_state_digest,
                    "log_gain": gain,
                    "material_improvement": gain >= HISTORICAL_MATERIAL_LOG_GAIN,
                    "stagnation_streak": streak_at_record,
                    "state_retained": not reset,
                    "reset_reason": "stagnation_window_reached" if reset else "",
                    "mean_recentered": mean_recentered,
                    "numerical_restart_count": len(visit_numerical_restarts),
                    "numerical_restarts": visit_numerical_restarts,
                }
            )
        if completed_in_sweep == len(sessions):
            full_sweeps += 1
        elif completed_in_sweep:
            partial_sweeps += 1
        else:
            break
    noop_fill_fes = target_fes - context.ledger.count
    if noop_fill_fes:
        incumbent = context.ledger.best_x
        context.ledger.evaluate(np.repeat(incumbent[None, :], noop_fill_fes, axis=0))
    actual_batches = [int(event["population_batches"]) for event in events]
    requested_batches = [
        int(event["requested_fes"]) // int(event["population_size"]) for event in events
    ]
    return {
        "schedule_version": "independent-smp-long-group-visit-v2",
        "requested_fes": requested_fes,
        "actual_fes": context.ledger.count - start_fes,
        "start_fes": start_fes,
        "end_fes": context.ledger.count,
        "group_count": len(sessions),
        "visit_count": len(events),
        "full_sweeps": full_sweeps,
        "partial_sweeps": partial_sweeps,
        "visits_per_group": visits_per_group,
        "restart_count": restart_count,
        "restart_identity_change_count": restart_identity_change_count,
        "stagnation_reset_count": restart_count,
        "stagnation_reset_identity_change_count": restart_identity_change_count,
        "numerical_restart_count": numerical_restart_count,
        "numerical_restart_identity_change_count": (
            numerical_restart_identity_change_count
        ),
        "numerical_restart_affected_visits": numerical_restart_affected_visits,
        "numerical_restart_fe_accounting_verified": True,
        "numerical_restart_events": numerical_restart_events,
        "terminal_group_states_finite": all(
            not _nonfinite_session_fields(session) for session in sessions
        ),
        "optimizer_identity_preserved_visits": optimizer_identity_preserved_visits,
        "state_update_count": state_update_count,
        "cross_visit_boundary_checks": cross_visit_boundary_checks,
        "cross_visit_state_digest_matches": cross_visit_state_digest_matches,
        "mean_recenter_match_count": mean_recenter_match_count,
        "median_actual_population_batches": statistics.median(actual_batches),
        "median_requested_population_batches": statistics.median(requested_batches),
        "minimum_actual_population_batches": min(actual_batches),
        "maximum_actual_population_batches": max(actual_batches),
        "noop_fill_fes": noop_fill_fes,
        "events": events,
    }


def _mechanism_gate(
    schedule: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    terminal: bool,
) -> bool:
    visit_count = int(schedule.get("visit_count", 0))
    gate = protocol["trace_gate"]
    visits_per_group = [int(value) for value in schedule.get("visits_per_group", [])]
    events = schedule.get("events", [])
    numerical_events = schedule.get("numerical_restart_events", [])
    numerical_restart_count = int(schedule.get("numerical_restart_count", -1))
    numerical_restart_affected_visits = int(
        schedule.get("numerical_restart_affected_visits", -1)
    )
    passed = bool(
        visit_count > 0
        and schedule.get("actual_fes") == schedule.get("requested_fes")
        and schedule.get("median_actual_population_batches", 0)
        >= gate["minimum_median_population_batches_per_visit"]
        and schedule.get("optimizer_identity_preserved_visits")
        + numerical_restart_affected_visits
        == visit_count
        and schedule.get("state_update_count") == visit_count
        and schedule.get("cross_visit_state_digest_matches")
        == schedule.get("cross_visit_boundary_checks")
        and schedule.get("mean_recenter_match_count") == visit_count
        and schedule.get("restart_identity_change_count")
        == schedule.get("restart_count")
        and schedule.get("stagnation_reset_count") == schedule.get("restart_count")
        and schedule.get("stagnation_reset_identity_change_count")
        == schedule.get("restart_identity_change_count")
        and numerical_restart_count == len(numerical_events)
        and schedule.get("numerical_restart_identity_change_count")
        == numerical_restart_count
        and numerical_restart_affected_visits
        == sum(bool(event.get("numerical_restart_count")) for event in events)
        and schedule.get("numerical_restart_fe_accounting_verified") is True
        and schedule.get("terminal_group_states_finite") is True
        and schedule.get("noop_fill_fes", math.inf)
        < max(int(event["population_size"]) for event in events)
        and all(
            int(event["actual_fes"]) <= int(event["requested_fes"])
            and int(event["actual_fes"]) % int(event["population_size"]) == 0
            for event in events
        )
        and all(
            int(event["generation_consumed_fes"])
            in (0, int(event["population_size"]))
            and event.get("optimizer_identity_changed") is True
            and event.get("state_finite_after_restart") is True
            for event in numerical_events
        )
    )
    if terminal:
        passed = bool(
            passed
            and visits_per_group
            and min(visits_per_group) > 0
            and max(visits_per_group) <= gate["maximum_terminal_visits_per_group"]
        )
    return passed


def _manifest(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    historical: Mapping[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    schema = TRACE_MANIFEST_SCHEMA if kind == "trace" else TERMINAL_MANIFEST_SCHEMA
    body = {
        "schema_version": schema,
        "kind": kind,
        "protocol_sha256": _file_sha256(protocol_path),
        "source_hashes": _live_source_hashes(),
        "historical_distribution_hashes": historical["distribution_hashes"],
        "historical_runner_sha256": historical["runner_sha256"],
        "historical_exact_runner_source_available": False,
        "git_source_evidence": _git_source_evidence(),
        "production_hcc_runtime_imports": _hcc_runtime_imports(),
        "production_smp_modification_allowed": protocol[
            "production_smp_modification_allowed"
        ],
        "selector_execution_allowed": False,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _write_artifact_root(
    root: Path,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    if root.exists():
        raise ValueError(f"SMP recovery output already exists: {root}")
    root.mkdir(parents=True)
    _write_json_atomic(root / "manifest.json", manifest)
    _write_json_atomic(root / "receipt.json", receipt)
    _write_json_atomic(root / "summary.json", summary)


def preflight(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    historical = _historical_facts(protocol)
    checkpoint = _checkpoint_input(protocol)
    git_evidence = _git_source_evidence()
    if _hcc_runtime_imports():
        raise ValueError("production HCC runtime imports remain")
    trace_root = _resolved(protocol["trace_output_root"])
    terminal_root = _resolved(protocol["terminal_output_root"])
    if trace_root.exists() or terminal_root.exists():
        raise ValueError("fresh SMP recovery output namespace is not empty")
    return {
        "historical_contract_complete": True,
        "historical": historical,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "checkpoint_block_count": len(checkpoint.blocks),
        "git_source_evidence": git_evidence,
        "production_hcc_runtime_imports": [],
        "trace_output_root": str(trace_root),
        "terminal_output_root": str(terminal_root),
        "selector_execution_allowed": False,
    }


def _run(
    protocol_path: Path,
    *,
    kind: str,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    historical = _historical_facts(protocol)
    checkpoint = _checkpoint_input(protocol)
    output_root = _resolved(
        protocol["trace_output_root"] if kind == "trace" else protocol["terminal_output_root"]
    )
    if output_root.exists():
        raise ValueError(f"SMP recovery output already exists: {output_root}")
    if kind == "terminal":
        trace = _validate_artifact_root(protocol_path, protocol, kind="trace")
        if trace["summary"].get("mechanism_gate_passed") is not True:
            raise ValueError("terminal SMP run is blocked by the trace gate")
    with threadpool_limits(limits=1):
        pools = _threadpools()
        if not pools or any(item["num_threads"] != 1 for item in pools):
            raise RuntimeError(f"native thread limit is not one: {pools}")
        problem = AobBenchmark().load(protocol["case_id"])
        context = _context("smp", checkpoint, problem, action_seed=checkpoint.run_seed)
        requested_fes = (
            protocol["trace_step_fes"] if kind == "trace" else context.ledger.remaining
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            schedule = execute_schedule(context, requested_fes, protocol=protocol)
        runtime_warnings = [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ]
    mechanism_passed = _mechanism_gate(
        schedule, protocol, terminal=(kind == "terminal")
    )
    exact_fes = schedule["actual_fes"] == requested_fes
    common = {
        "kind": kind,
        "case_id": protocol["case_id"],
        "run_seed": checkpoint.run_seed,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "phase1_fes": checkpoint.phase1_fes,
        "requested_action_fes": requested_fes,
        "consumed_action_fes": schedule["actual_fes"],
        "terminal_fes": context.ledger.count,
        "final_error": context.ledger.best_error,
        "schedule": schedule,
        "mechanism_gate_passed": mechanism_passed,
        "exact_fes": exact_fes,
        "runtime_warnings": runtime_warnings,
        "threadpools": pools,
        "native_thread_limit_verified": all(item["num_threads"] == 1 for item in pools),
        "historical_p90": protocol["historical_p90"],
        "selector_evaluation_authorized": False,
    }
    receipt_schema = TRACE_RECEIPT_SCHEMA if kind == "trace" else TERMINAL_RECEIPT_SCHEMA
    receipt_body = {"schema_version": receipt_schema, **common}
    if kind == "terminal":
        result = terminal_result(
            context,
            route=(
                f"independent_smp_long_visits_{schedule['visit_count']}_"
                f"stagnation_resets_{schedule['stagnation_reset_count']}_"
                f"numerical_restarts_{schedule['numerical_restart_count']}_"
                f"noop_{schedule['noop_fill_fes']}"
            ),
        )
        receipt_body.update(
            {
                "result_hash": result.result_hash,
                "route": result.route,
                "historical_level_passed": (
                    result.final_error <= protocol["historical_p90"]
                ),
            }
        )
    receipt = {**receipt_body, "receipt_hash": canonical_sha256(receipt_body)}
    integrity_passed = bool(
        mechanism_passed
        and exact_fes
        and not runtime_warnings
        and common["native_thread_limit_verified"]
    )
    if kind == "trace":
        summary_body = {
            "schema_version": TRACE_SUMMARY_SCHEMA,
            "receipt_hash": receipt["receipt_hash"],
            "mechanism_gate_passed": integrity_passed,
            "terminal_run_authorized": integrity_passed,
            "historical_terminal_parity_evaluated": False,
            "selector_evaluation_authorized": False,
        }
    else:
        historical_level_passed = bool(receipt["historical_level_passed"])
        summary_body = {
            "schema_version": TERMINAL_SUMMARY_SCHEMA,
            "receipt_hash": receipt["receipt_hash"],
            "integrity_gate_passed": integrity_passed,
            "final_error": receipt["final_error"],
            "historical_p90": protocol["historical_p90"],
            "historical_level_passed": historical_level_passed,
            "terminal_screen_passed": integrity_passed and historical_level_passed,
            "production_smp_integration_authorized": (
                integrity_passed and historical_level_passed
            ),
            "selector_evaluation_authorized": False,
        }
    summary = {**summary_body, "summary_hash": canonical_sha256(summary_body)}
    manifest = _manifest(
        protocol_path, protocol, historical, kind=kind
    )
    _write_artifact_root(output_root, manifest, receipt, summary)
    return summary


def run_trace(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    return _run(Path(protocol_path), kind="trace")


def run_terminal(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    return _run(Path(protocol_path), kind="terminal")


def _validate_artifact_root(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    *,
    kind: str,
) -> dict[str, dict[str, Any]]:
    root = _resolved(
        protocol["trace_output_root"] if kind == "trace" else protocol["terminal_output_root"]
    )
    manifest = _load_json(root / "manifest.json")
    receipt = _load_json(root / "receipt.json")
    summary = _load_json(root / "summary.json")
    _verify_payload_hash(manifest, "manifest_sha256", f"{kind} manifest")
    _verify_payload_hash(receipt, "receipt_hash", f"{kind} receipt")
    _verify_payload_hash(summary, "summary_hash", f"{kind} summary")
    expected_manifest_schema = (
        TRACE_MANIFEST_SCHEMA if kind == "trace" else TERMINAL_MANIFEST_SCHEMA
    )
    expected_receipt_schema = (
        TRACE_RECEIPT_SCHEMA if kind == "trace" else TERMINAL_RECEIPT_SCHEMA
    )
    expected_summary_schema = (
        TRACE_SUMMARY_SCHEMA if kind == "trace" else TERMINAL_SUMMARY_SCHEMA
    )
    if (
        manifest.get("schema_version") != expected_manifest_schema
        or receipt.get("schema_version") != expected_receipt_schema
        or summary.get("schema_version") != expected_summary_schema
        or manifest.get("kind") != kind
        or receipt.get("kind") != kind
        or summary.get("receipt_hash") != receipt.get("receipt_hash")
        or manifest.get("protocol_sha256") != _file_sha256(protocol_path)
        or manifest.get("source_hashes") != _live_source_hashes()
        or manifest.get("production_hcc_runtime_imports") != []
        or receipt.get("runtime_warnings") != []
        or receipt.get("native_thread_limit_verified") is not True
        or receipt.get("exact_fes") is not True
        or receipt.get("mechanism_gate_passed") is not True
        or not _mechanism_gate(
            receipt["schedule"], protocol, terminal=(kind == "terminal")
        )
    ):
        raise ValueError(f"{kind} SMP recovery artifact failed closed validation")
    return {"manifest": manifest, "receipt": receipt, "summary": summary}


def check(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    _historical_facts(protocol)
    trace = _validate_artifact_root(protocol_path, protocol, kind="trace")
    terminal = _validate_artifact_root(protocol_path, protocol, kind="terminal")
    receipt = terminal["receipt"]
    summary = terminal["summary"]
    expected_historical_passed = receipt["final_error"] <= protocol["historical_p90"]
    if (
        trace["summary"].get("terminal_run_authorized") is not True
        or receipt.get("terminal_fes") != protocol["total_budget_fes"]
        or receipt.get("consumed_action_fes")
        != protocol["total_budget_fes"] - protocol["phase1_fes"]
        or receipt.get("historical_level_passed") != expected_historical_passed
        or summary.get("historical_level_passed") != expected_historical_passed
        or summary.get("terminal_screen_passed") != expected_historical_passed
        or summary.get("selector_evaluation_authorized") is not False
    ):
        raise ValueError("terminal SMP recovery decision drifted")
    return {
        "integrity_gate_passed": True,
        "final_error": receipt["final_error"],
        "historical_p90": protocol["historical_p90"],
        "historical_level_passed": expected_historical_passed,
        "terminal_screen_passed": expected_historical_passed,
        "production_smp_integration_authorized": expected_historical_passed,
        "selector_evaluation_authorized": False,
        "trace_manifest_sha256": trace["manifest"]["manifest_sha256"],
        "terminal_manifest_sha256": terminal["manifest"]["manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "trace", "run", "check"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        payload = preflight(args.protocol)
    elif args.command == "trace":
        payload = run_trace(args.protocol)
    elif args.command == "run":
        payload = run_terminal(args.protocol)
    else:
        payload = check(args.protocol)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
