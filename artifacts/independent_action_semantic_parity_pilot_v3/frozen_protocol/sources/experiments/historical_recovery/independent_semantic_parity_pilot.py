"""Run the fixed-checkpoint independent action semantic-parity mechanism screen."""

# Thread caps must be applied before importing NumPy, pypop7, or ARAC numerical modules.
# ruff: noqa: E402

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import warnings

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from arac.actions._execution import (
    BLOCK_POPULATION_SIZE,
    DEFAULT_SIGMA,
    FULL_SPACE_POPULATION_SIZE,
    STATE_MATERIAL_LOG_GAIN,
    STATE_STALE_WINDOW,
    _PersistentBlockSession,
    _allocate_block_budgets,
    _block_population_size,
    _log_improvement,
    _run_block_visit,
    run_full_space,
    terminal_result,
)
from arac.actions.registry import ActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ACTION_NAMES, ActionContext, ActionResult, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort
from experiments.historical_recovery.audit_independent_action_semantic_parity import (
    _hcc_runtime_imports,
)
from experiments.historical_recovery.replay import _checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("independent_semantic_parity_protocol_v3.json")
MANIFEST_SCHEMA = "arac-independent-semantic-parity-pilot-manifest-v3"
RECEIPT_SCHEMA = "arac-independent-semantic-parity-pilot-receipt-v3"
SUMMARY_SCHEMA = "arac-independent-semantic-parity-pilot-summary-v3"
SOURCE_PATHS = (
    "experiments/historical_recovery/independent_semantic_parity_pilot.py",
    "experiments/historical_recovery/independent_semantic_parity_protocol_v3.json",
    "src/arac/actions/_execution.py",
    "src/arac/actions/aor.py",
    "src/arac/actions/ctp.py",
    "src/arac/actions/gcb.py",
    "src/arac/actions/smp.py",
    "src/arac/actions/registry.py",
    "src/arac/benchmarks/aob.py",
    "src/arac/runtime/contracts.py",
    "src/arac/runtime/ledger.py",
    "src/arac/runtime/optimizers.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_reference_contract(action: str, payload: Mapping[str, Any]) -> None:
    if action == "aor":
        expected = {
            "optimizer_route": "full_space_sep_cmaes",
            "initial_mean": 0.0,
            "sigma": 0.5,
            "population_size": 24,
            "restart": False,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("AOR historical semantic contract drifted")
        return
    action_payload = payload.get("action")
    if not isinstance(action_payload, dict):
        raise ValueError(f"{action.upper()} semantic contract has no action object")
    if action == "ctp":
        if (
            action_payload.get("coverage_sweeps") != 4
            or action_payload.get("group_polish_mode") != "unbounded_group_polish"
            or action_payload.get("restart_policy") != "none"
        ):
            raise ValueError("CTP historical semantic contract drifted")
    elif action == "smp":
        required_state = {"covariance", "path_sigma", "sigma", "mean", "rng_state"}
        if (
            action_payload.get("name") != "smp"
            or action_payload.get("stale_window") != 3
            or not required_state.issubset(action_payload.get("state_fields", []))
        ):
            raise ValueError("SMP historical semantic contract drifted")
    elif action == "gcb":
        raise ValueError("GCB action contract must be validated at the receipt level")


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = _load_json(protocol_path)
    required = {
        "schema_version",
        "status",
        "purpose",
        "historical_reference_policy",
        "production_hcc_dependency_allowed",
        "selector_execution_allowed",
        "common_anchor",
        "lanes",
        "arms",
        "acceptance_gates",
        "promotion_rule",
    }
    if set(protocol) != required:
        raise ValueError("semantic parity protocol keys drifted")
    if protocol["schema_version"] != "arac-independent-action-semantic-parity-protocol-v3":
        raise ValueError("semantic parity protocol schema drifted")
    if protocol["production_hcc_dependency_allowed"] or protocol["selector_execution_allowed"]:
        raise ValueError("semantic parity protocol cannot enable HCC or selector execution")
    if protocol["arms"] != ["current_production", "historical_semantic_port"]:
        raise ValueError("semantic parity protocol arms drifted")
    common = protocol["common_anchor"]
    if (
        common.get("checkpoint_seed") != 117
        or common.get("screen_step_fes") != 120_000
        or common.get("native_threads") != 1
        or common.get("max_workers") != 4
    ):
        raise ValueError("semantic parity common anchor drifted")
    lanes = protocol["lanes"]
    if len(lanes) != 4 or {lane.get("action") for lane in lanes} != set(ACTION_NAMES):
        raise ValueError("semantic parity lanes drifted")
    for lane in lanes:
        checkpoint_path = REPOSITORY_ROOT / str(lane["checkpoint"])
        contract_path = REPOSITORY_ROOT / str(lane["reference_contract"])
        if not checkpoint_path.is_file() or not contract_path.is_file():
            raise ValueError(f"semantic parity input is missing: {lane['action']}")
        wrapper = _load_json(checkpoint_path)
        checkpoint = _checkpoint(wrapper["checkpoint"])
        if wrapper.get("checkpoint_hash") != checkpoint.checkpoint_hash:
            raise ValueError(f"source checkpoint hash drifted: {lane['action']}")
        contract = _load_json(contract_path)
        if lane["action"] == "gcb":
            if (
                contract.get("execution_mode") != "one_native_sweep_burst_then_native"
                or contract.get("native_resumed") is not True
                or not contract.get("source_group_actual_fes")
            ):
                raise ValueError("GCB historical semantic contract drifted")
        else:
            _validate_reference_contract(str(lane["action"]), contract)
    return protocol


def _context(
    action: str,
    checkpoint,
    problem,
    *,
    action_seed: int,
) -> ActionContext:
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    return ActionContext(action, checkpoint, problem, ledger, action_seed=action_seed)


def _fresh_aor(context: ActionContext) -> tuple[ActionResult, dict[str, Any]]:
    budget = context.ledger.remaining
    run = PypopOptimizerPort().run(
        "sepcmaes",
        problem=context.problem,
        ledger=context.ledger,
        initial_mean=np.zeros(context.problem.dimension),
        sigma=DEFAULT_SIGMA,
        seed=context.action_seed,
        budget_fes=budget,
        population_size=FULL_SPACE_POPULATION_SIZE,
        restart=False,
    )
    events = {
        "algorithm": run.algorithm,
        "initial_mean_max_abs": 0.0,
        "sigma": DEFAULT_SIGMA,
        "population_size": FULL_SPACE_POPULATION_SIZE,
        "restart": False,
        "optimizer_fes": run.consumed_fes,
    }
    return terminal_result(context, route=f"historical_fresh_zero_mean_sepcmaes_{budget}"), events


def _unbounded_block_polish(context: ActionContext) -> tuple[int, int]:
    blocks = context.checkpoint.blocks
    aligned = context.ledger.remaining
    aligned -= aligned % (len(blocks) * BLOCK_POPULATION_SIZE)
    if aligned == 0:
        return 0, 0
    budgets = _allocate_block_budgets(blocks, aligned, equal_generations=False)
    count_before = context.ledger.count
    for index, block in enumerate(blocks):
        session = _PersistentBlockSession(context, block, index, budgets[index])
        session.begin_visit()
        while not session.complete:
            session.advance(adapt_on_early_stop=True)
    return context.ledger.count - count_before, len(blocks)


def _historical_ctp(context: ActionContext) -> tuple[ActionResult, dict[str, Any]]:
    blocks = context.checkpoint.blocks
    sweep_count = 4
    per_session_budget = sweep_count * BLOCK_POPULATION_SIZE
    sessions = tuple(
        _PersistentBlockSession(context, block, index, per_session_budget)
        for index, block in enumerate(blocks)
    )
    coverage_before = context.ledger.count
    coverage_sweep_fes = []
    for _ in range(sweep_count):
        sweep_before = context.ledger.count
        for session in sessions:
            session.advance()
        coverage_sweep_fes.append(context.ledger.count - sweep_before)
    coverage_fes = context.ledger.count - coverage_before
    polish_fes, polished_blocks = _unbounded_block_polish(context)
    tail_fes = context.ledger.remaining
    if tail_fes:
        run_full_space(
            context,
            algorithm="mmes",
            budget_fes=tail_fes,
            namespace="historical-ctp-tail",
        )
    events = {
        "coverage_sweeps": len(coverage_sweep_fes),
        "coverage_sweep_fes": coverage_sweep_fes,
        "coverage_session_count": len(sessions),
        "coverage_fes": coverage_fes,
        "polish_mode": "unbounded_group_polish",
        "polish_fes": polish_fes,
        "polished_blocks": polished_blocks,
        "tail_fes": tail_fes,
    }
    route = (
        f"historical_coverage_sweeps_4_fes_{coverage_fes}_"
        f"unbounded_group_polish_{polish_fes}_tail_{tail_fes}"
    )
    return terminal_result(context, route=route), events


def _session_state_fields(session: _PersistentBlockSession) -> set[str]:
    fields = {
        name
        for name in ("covariance", "path_sigma", "sigma", "mean")
        if hasattr(session.optimizer if name == "sigma" else session, name)
    }
    rng = getattr(session.optimizer, "rng_optimization", None)
    if getattr(getattr(rng, "bit_generator", None), "state", None) is not None:
        fields.add("rng_state")
    return fields


def _jsonable_state(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_state(item) for item in value]
    return value


def _session_state_digest(session: _PersistentBlockSession) -> str:
    state = {
        "covariance": session.covariance,
        "generation": session.optimizer._n_generations,
        "mean": session.mean,
        "path_covariance": session.path_covariance,
        "path_sigma": session.path_sigma,
        "rng_state": session.optimizer.rng_optimization.bit_generator.state,
        "sigma": session.optimizer.sigma,
    }
    encoded = json.dumps(
        _jsonable_state(state),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _historical_smp(context: ActionContext) -> tuple[ActionResult, dict[str, Any]]:
    blocks = context.checkpoint.blocks
    populations = tuple(_block_population_size(len(block)) for block in blocks)
    sessions = tuple(
        _PersistentBlockSession(
            context,
            block,
            index,
            context.ledger.remaining,
            population_size=populations[index],
            seed_namespace="historical-smp",
        )
        for index, block in enumerate(blocks)
    )
    stale_streaks = [0] * len(sessions)
    visits_per_session = [0] * len(sessions)
    restart_count = 0
    optimizer_identity_preserved_visits = 0
    restart_identity_change_count = 0
    state_update_count = 0
    cross_visit_boundary_checks = 0
    cross_visit_state_digest_matches = 0
    last_state_digests: list[str | None] = [None] * len(sessions)
    first_state_digests: list[str | None] = [None] * len(sessions)
    full_sweeps = 0
    while context.ledger.remaining >= sum(populations):
        for index, session in enumerate(sessions):
            optimizer_before = session.optimizer
            state_before = _session_state_digest(session)
            if last_state_digests[index] is not None:
                cross_visit_boundary_checks += 1
                if state_before == last_state_digests[index]:
                    cross_visit_state_digest_matches += 1
            before = context.ledger.best_error
            consumed = _run_block_visit(session, populations[index])
            if consumed != populations[index]:
                raise RuntimeError("SMP visit did not consume one complete population")
            state_after = _session_state_digest(session)
            if session.optimizer is optimizer_before:
                optimizer_identity_preserved_visits += 1
            if state_after != state_before:
                state_update_count += 1
            if first_state_digests[index] is None:
                first_state_digests[index] = state_after
            visits_per_session[index] += 1
            gain = _log_improvement(before, context.ledger.best_error)
            stale_streaks[index] = (
                0 if gain >= STATE_MATERIAL_LOG_GAIN else stale_streaks[index] + 1
            )
            if stale_streaks[index] >= STATE_STALE_WINDOW:
                optimizer_before_restart = session.optimizer
                session.restart()
                if session.optimizer is not optimizer_before_restart:
                    restart_identity_change_count += 1
                stale_streaks[index] = 0
                restart_count += 1
            last_state_digests[index] = _session_state_digest(session)
        full_sweeps += 1
    tail_fes = context.ledger.remaining
    if tail_fes:
        run_full_space(
            context,
            algorithm="sepcmaes",
            budget_fes=tail_fes,
            namespace="historical-smp-terminal-alignment",
        )
    required_fields = {"covariance", "path_sigma", "sigma", "mean", "rng_state"}
    fields_present = set.intersection(*(_session_state_fields(session) for session in sessions))
    events = {
        "state_persistence": min(visits_per_session) > 1,
        "stale_window": STATE_STALE_WINDOW,
        "state_fields": sorted(required_fields & fields_present),
        "session_count": len(sessions),
        "visits_per_session": visits_per_session,
        "full_sweeps": full_sweeps,
        "restart_count": restart_count,
        "session_restart_counts": [session.restart_count for session in sessions],
        "optimizer_identity_preserved_visits": optimizer_identity_preserved_visits,
        "restart_identity_change_count": restart_identity_change_count,
        "state_update_count": state_update_count,
        "cross_visit_boundary_checks": cross_visit_boundary_checks,
        "cross_visit_state_digest_matches": cross_visit_state_digest_matches,
        "state_trace": [
            {
                "block_index": index,
                "first_recorded_state": first_state_digests[index],
                "final_state": _session_state_digest(session),
                "restart_count": session.restart_count,
                "visit_count": visits_per_session[index],
            }
            for index, session in enumerate(sessions)
        ],
        "tail_fes": tail_fes,
        "reference_status": "five_seed_mechanism_only",
    }
    route = (
        f"historical_stateful_sessions_{len(sessions)}_sweeps_{full_sweeps}_"
        f"stale_restarts_{restart_count}_tail_{tail_fes}"
    )
    return terminal_result(context, route=route), events


def _scaled_gcb_visit_budgets(
    raw_budgets: list[int],
    populations: tuple[int, ...],
    target: int,
) -> tuple[int, ...]:
    if len(raw_budgets) != len(populations) or sum(raw_budgets) <= 0:
        raise ValueError("GCB source group budgets are invalid")
    total = sum(raw_budgets)
    scaled = []
    for raw, population in zip(raw_budgets, populations, strict=True):
        desired = max(population, math.floor(target * raw / total))
        scaled.append(max(population, desired - desired % population))
    return tuple(scaled)


def _historical_gcb(
    context: ActionContext,
    contract: Mapping[str, Any],
) -> tuple[ActionResult, dict[str, Any]]:
    blocks = context.checkpoint.blocks
    populations = tuple(_block_population_size(len(block)) for block in blocks)
    source_budgets = [int(value) for value in contract["source_group_actual_fes"]]
    reference_total = int(contract["terminal_fe"])
    reference_burst = int(contract["candidate_action_budget_fes"])
    target_native = max(
        sum(populations),
        round(context.checkpoint.remaining_fes * reference_burst / reference_total),
    )
    visit_budgets = _scaled_gcb_visit_budgets(source_budgets, populations, target_native)
    sessions = tuple(
        _PersistentBlockSession(
            context,
            block,
            index,
            context.checkpoint.remaining_fes,
            population_size=populations[index],
            seed_namespace="historical-gcb-native",
        )
        for index, block in enumerate(blocks)
    )
    native_before = sum(
        _run_block_visit(session, budget)
        for session, budget in zip(sessions, visit_budgets, strict=True)
    )
    session_consumed_before_coordination = [session.consumed_fes for session in sessions]
    optimizer_before_coordination = [session.optimizer for session in sessions]
    state_before_coordination = [_session_state_digest(session) for session in sessions]
    coordination_fes = min(native_before, context.ledger.remaining)
    if coordination_fes:
        run_full_space(
            context,
            algorithm="sepcmaes",
            budget_fes=coordination_fes,
            namespace="historical-gcb-phase-boundary",
        )
    coordination_identity_preserved = all(
        session.optimizer is optimizer
        for session, optimizer in zip(sessions, optimizer_before_coordination, strict=True)
    )
    coordination_state_digest_preserved = state_before_coordination == [
        _session_state_digest(session) for session in sessions
    ]
    resume_before = context.ledger.count
    while context.ledger.remaining >= min(populations):
        remaining_before = context.ledger.remaining
        requested_per_block = math.ceil(remaining_before / len(blocks))
        for session, population in zip(sessions, populations, strict=True):
            if context.ledger.remaining < population:
                break
            available = session.budget_fes - session.consumed_fes
            requested = min(max(requested_per_block, population), available)
            if requested >= population:
                _run_block_visit(session, requested)
        if context.ledger.remaining == remaining_before:
            break
    native_resume = context.ledger.count - resume_before
    session_consumed_after_resume = [session.consumed_fes for session in sessions]
    resume_identity_preserved = all(
        session.optimizer is optimizer
        for session, optimizer in zip(sessions, optimizer_before_coordination, strict=True)
    )
    state_after_resume = [_session_state_digest(session) for session in sessions]
    tail_fes = context.ledger.remaining
    if tail_fes:
        run_full_space(
            context,
            algorithm="sepcmaes",
            budget_fes=tail_fes,
            namespace="historical-gcb-terminal-alignment",
        )
    events = {
        "trigger_scope": "phase_boundary",
        "native_before_fes": native_before,
        "coordination_fes": coordination_fes,
        "native_resume_fes": native_resume,
        "tail_fes": tail_fes,
        "session_consumed_before_coordination": session_consumed_before_coordination,
        "session_consumed_after_resume": session_consumed_after_resume,
        "coordination_identity_preserved": coordination_identity_preserved,
        "coordination_state_digest_preserved": coordination_state_digest_preserved,
        "resume_identity_preserved": resume_identity_preserved,
        "state_changed_after_resume_count": sum(
            after != before
            for before, after in zip(
                state_before_coordination,
                state_after_resume,
                strict=True,
            )
        ),
        "state_trace": [
            {
                "block_index": index,
                "before_coordination": state_before_coordination[index],
                "after_resume": state_after_resume[index],
            }
            for index in range(len(sessions))
        ],
        "state_reused_after_coordination": native_resume > 0
        and any(
            after > before
            for before, after in zip(
                session_consumed_before_coordination,
                session_consumed_after_resume,
                strict=True,
            )
        ),
        "session_count": len(sessions),
    }
    route = (
        f"historical_native_{native_before}_phase_boundary_coordination_"
        f"{coordination_fes}_native_resume_{native_resume}_tail_{tail_fes}_state_reused"
    )
    return terminal_result(context, route=route), events


def execute_historical_semantic_port(
    context: ActionContext,
    contract: Mapping[str, Any],
) -> tuple[ActionResult, dict[str, Any]]:
    """Execute one independent port of the historical action JSON semantics."""

    if context.action_name == "aor":
        return _fresh_aor(context)
    if context.action_name == "ctp":
        return _historical_ctp(context)
    if context.action_name == "gcb":
        return _historical_gcb(context, contract)
    if context.action_name == "smp":
        return _historical_smp(context)
    raise ValueError(f"unsupported action: {context.action_name}")


def _candidate_mechanism_passed(action: str, events: Mapping[str, Any]) -> bool:
    if action == "aor":
        return (
            events.get("algorithm") == "sepcmaes"
            and events.get("initial_mean_max_abs") == 0.0
            and events.get("population_size") == 24
            and events.get("restart") is False
            and int(events.get("optimizer_fes", 0)) > 0
        )
    if action == "ctp":
        return (
            events.get("coverage_sweeps") == 4
            and len(events.get("coverage_sweep_fes", [])) == 4
            and all(int(value) > 0 for value in events.get("coverage_sweep_fes", []))
            and events.get("polish_mode") == "unbounded_group_polish"
            and int(events.get("polish_fes", 0)) > 0
        )
    if action == "gcb":
        return (
            events.get("trigger_scope") == "phase_boundary"
            and int(events.get("coordination_fes", 0)) > 0
            and int(events.get("native_resume_fes", 0)) > 0
            and events.get("state_reused_after_coordination") is True
            and events.get("coordination_identity_preserved") is True
            and events.get("coordination_state_digest_preserved") is True
            and events.get("resume_identity_preserved") is True
            and int(events.get("state_changed_after_resume_count", 0)) > 0
            and len(events.get("state_trace", []))
            == int(events.get("session_count", -1))
            and any(
                after > before
                for before, after in zip(
                    events.get("session_consumed_before_coordination", []),
                    events.get("session_consumed_after_resume", []),
                    strict=True,
                )
            )
        )
    return (
        events.get("state_persistence") is True
        and events.get("stale_window") == 3
        and "rng_state" in events.get("state_fields", [])
        and int(events.get("full_sweeps", 0)) > 1
        and int(events.get("restart_count", 0)) > 0
        and int(events.get("optimizer_identity_preserved_visits", -1))
        == sum(events.get("visits_per_session", []))
        and int(events.get("restart_identity_change_count", -1))
        == int(events.get("restart_count", -2))
        and int(events.get("state_update_count", 0)) > 0
        and int(events.get("cross_visit_boundary_checks", 0)) > 0
        and int(events.get("cross_visit_state_digest_matches", -1))
        == int(events.get("cross_visit_boundary_checks", -2))
        and len(events.get("state_trace", [])) == int(events.get("session_count", -1))
        and sum(events.get("session_restart_counts", []))
        == int(events.get("restart_count", -1))
    )


def _threadpool_snapshot() -> list[dict[str, Any]]:
    return [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
            "user_api": item.get("user_api"),
        }
        for item in threadpool_info()
    ]


def _result_payload(
    result: ActionResult,
    *,
    events: Mapping[str, Any],
    warnings_payload: list[dict[str, str]],
    reference_match: bool | None,
    reference_evaluated: bool,
) -> dict[str, Any]:
    return {
        "terminal_fes": result.terminal_fes,
        "consumed_fes": result.consumed_fes,
        "final_error": result.final_error,
        "route": result.route,
        "result_hash": result.result_hash,
        "optimizer_package": result.optimizer_package,
        "optimizer_version": result.optimizer_version,
        "events": dict(events),
        "runtime_warnings": warnings_payload,
        "reference_mechanism_match": reference_match,
        "reference_mechanism_evaluated": reference_evaluated,
    }


def _run_lane_limited(
    lane: Mapping[str, Any],
    screen_fes: int,
    output_root_text: str,
    threadpools: list[dict[str, Any]],
) -> dict[str, Any]:
    action = str(lane["action"])
    case_id = str(lane["case_id_audit_metadata"])
    checkpoint_wrapper = _load_json(REPOSITORY_ROOT / str(lane["checkpoint"]))
    source_checkpoint = _checkpoint(checkpoint_wrapper["checkpoint"])
    screen_checkpoint = replace(
        source_checkpoint,
        total_budget_fes=source_checkpoint.phase1_fes + int(screen_fes),
    )
    contract = _load_json(REPOSITORY_ROOT / str(lane["reference_contract"]))
    problem = AobBenchmark().load(case_id)
    arms: dict[str, Any] = {}

    current_context = _context(
        action,
        screen_checkpoint,
        problem,
        action_seed=source_checkpoint.run_seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        current_result = ActionRegistry().execute(current_context)
    current_warnings = [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    ]
    arms["current_production"] = _result_payload(
        current_result,
        events={"route": current_result.route},
        warnings_payload=current_warnings,
        reference_match=None,
        reference_evaluated=False,
    )

    candidate_context = _context(
        action,
        screen_checkpoint,
        problem,
        action_seed=source_checkpoint.run_seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidate_result, events = execute_historical_semantic_port(
            candidate_context,
            contract,
        )
    candidate_warnings = [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    ]
    candidate_passed = _candidate_mechanism_passed(action, events)
    arms["historical_semantic_port"] = _result_payload(
        candidate_result,
        events=events,
        warnings_payload=candidate_warnings,
        reference_match=candidate_passed,
        reference_evaluated=True,
    )
    exact_fes = all(arm["consumed_fes"] == screen_fes for arm in arms.values())
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "case_id_audit_metadata": case_id,
        "action": action,
        "run_seed": source_checkpoint.run_seed,
        "source_checkpoint_hash": source_checkpoint.checkpoint_hash,
        "screen_checkpoint_hash": screen_checkpoint.checkpoint_hash,
        "source_phase1_fes": source_checkpoint.phase1_fes,
        "screen_step_fes": screen_fes,
        "reference_contract": str(lane["reference_contract"]),
        "reference_contract_sha256": _file_sha256(
            REPOSITORY_ROOT / str(lane["reference_contract"])
        ),
        "arms": arms,
        "same_screen_checkpoint": all(
            result.checkpoint_hash == screen_checkpoint.checkpoint_hash
            for result in (current_result, candidate_result)
        ),
        "exact_screen_fes": exact_fes,
        "candidate_mechanism_passed": candidate_passed,
        "native_thread_limit_verified": all(
            item.get("num_threads") == 1 for item in threadpools
        ),
        "threadpools": threadpools,
        "historical_terminal_parity_evaluated": False,
    }
    payload["receipt_hash"] = canonical_sha256(payload)
    destination = Path(output_root_text) / "receipts" / f"{action}.json"
    _write_json_atomic(destination, payload)
    return payload


def _run_lane(lane: Mapping[str, Any], screen_fes: int, output_root_text: str) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        threadpools = _threadpool_snapshot()
        if not threadpools or any(item.get("num_threads") != 1 for item in threadpools):
            raise RuntimeError(f"native thread limit is not one: {threadpools}")
        return _run_lane_limited(lane, screen_fes, output_root_text, threadpools)


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    contracts = {
        str(lane["action"]): {
            "path": str(lane["reference_contract"]),
            "sha256": _file_sha256(REPOSITORY_ROOT / str(lane["reference_contract"])),
        }
        for lane in protocol["lanes"]
    }
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_sha256": _file_sha256(protocol_path),
        "source_hashes": {
            path: _file_sha256(REPOSITORY_ROOT / path) for path in SOURCE_PATHS
        },
        "reference_contracts": contracts,
        "production_hcc_runtime_imports": _hcc_runtime_imports(),
        "selector_execution_allowed": False,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _prepare_output(protocol_path: Path, protocol: Mapping[str, Any]) -> Path:
    output_root = (REPOSITORY_ROOT / protocol["common_anchor"]["new_output_root"]).resolve()
    if output_root.exists():
        raise ValueError(f"semantic parity output already exists: {output_root}")
    frozen_sources = output_root / "frozen_protocol" / "sources"
    frozen_sources.mkdir(parents=True)
    shutil.copy2(protocol_path, output_root / "frozen_protocol" / "protocol.json")
    for relative in SOURCE_PATHS:
        source = REPOSITORY_ROOT / relative
        destination = frozen_sources / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _write_json_atomic(
        output_root / "frozen_protocol" / "manifest.json",
        _manifest(protocol_path, protocol),
    )
    return output_root


def _summarize(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    receipts.sort(key=lambda item: ACTION_NAMES.index(str(item["action"])))
    candidate_passes = sum(bool(row["candidate_mechanism_passed"]) for row in receipts)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "lane_count": len(receipts),
        "all_exact_screen_fes": all(row["exact_screen_fes"] for row in receipts),
        "all_same_screen_checkpoint": all(row["same_screen_checkpoint"] for row in receipts),
        "current_reference_evaluated_count": sum(
            bool(row["arms"]["current_production"]["reference_mechanism_evaluated"])
            for row in receipts
        ),
        "candidate_mechanism_passed_count": candidate_passes,
        "all_native_thread_limits_verified": all(
            row["native_thread_limit_verified"] for row in receipts
        ),
        "mechanism_screen_passed": candidate_passes == len(receipts) == 4
        and all(row["exact_screen_fes"] for row in receipts)
        and all(row["same_screen_checkpoint"] for row in receipts)
        and all(row["native_thread_limit_verified"] for row in receipts),
        "historical_terminal_parity_evaluated": False,
        "selector_evaluation_authorized": False,
        "lanes": [
            {
                "action": row["action"],
                "case_id_audit_metadata": row["case_id_audit_metadata"],
                "current_reference_evaluated": row["arms"]["current_production"][
                    "reference_mechanism_evaluated"
                ],
                "candidate_mechanism_passed": row["candidate_mechanism_passed"],
                "current_final_error": row["arms"]["current_production"]["final_error"],
                "candidate_final_error": row["arms"]["historical_semantic_port"][
                    "final_error"
                ],
                "receipt_hash": row["receipt_hash"],
            }
            for row in receipts
        ],
    }
    return summary


def run_pilot(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    if _hcc_runtime_imports():
        raise ValueError("production HCC runtime imports remain")
    output_root = _prepare_output(protocol_file, protocol)
    screen_fes = int(protocol["common_anchor"]["screen_step_fes"])
    receipts = []
    with ProcessPoolExecutor(
        max_workers=int(protocol["common_anchor"]["max_workers"])
    ) as executor:
        futures = {
            executor.submit(_run_lane, lane, screen_fes, str(output_root)): lane
            for lane in protocol["lanes"]
        }
        for future in as_completed(futures):
            receipts.append(future.result())
    summary = _summarize(receipts)
    _write_json_atomic(output_root / "summary.json", summary)
    return summary


def _validate_receipt(
    lane: Mapping[str, Any],
    receipt: Mapping[str, Any],
    screen_fes: int,
) -> None:
    action = str(lane["action"])
    checkpoint_wrapper = _load_json(REPOSITORY_ROOT / str(lane["checkpoint"]))
    source_checkpoint = _checkpoint(checkpoint_wrapper["checkpoint"])
    screen_checkpoint = replace(
        source_checkpoint,
        total_budget_fes=source_checkpoint.phase1_fes + screen_fes,
    )
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "action": action,
        "case_id_audit_metadata": str(lane["case_id_audit_metadata"]),
        "run_seed": source_checkpoint.run_seed,
        "source_checkpoint_hash": source_checkpoint.checkpoint_hash,
        "screen_checkpoint_hash": screen_checkpoint.checkpoint_hash,
        "source_phase1_fes": source_checkpoint.phase1_fes,
        "screen_step_fes": screen_fes,
        "reference_contract": str(lane["reference_contract"]),
        "reference_contract_sha256": _file_sha256(
            REPOSITORY_ROOT / str(lane["reference_contract"])
        ),
        "same_screen_checkpoint": True,
        "exact_screen_fes": True,
        "candidate_mechanism_passed": True,
        "native_thread_limit_verified": True,
        "historical_terminal_parity_evaluated": False,
    }
    drifted = [key for key, value in expected.items() if receipt.get(key) != value]
    if drifted:
        raise ValueError(f"semantic parity receipt gate failed for {action}: {drifted}")
    arms = receipt.get("arms")
    if not isinstance(arms, dict) or set(arms) != {
        "current_production",
        "historical_semantic_port",
    }:
        raise ValueError(f"semantic parity arms drifted: {action}")
    current = arms["current_production"]
    candidate = arms["historical_semantic_port"]
    if (
        current.get("reference_mechanism_evaluated") is not False
        or current.get("reference_mechanism_match") is not None
    ):
        raise ValueError(f"current mechanism comparison was not left unevaluated: {action}")
    if (
        candidate.get("reference_mechanism_evaluated") is not True
        or candidate.get("reference_mechanism_match") is not True
        or not _candidate_mechanism_passed(action, candidate.get("events", {}))
    ):
        raise ValueError(f"candidate mechanism evidence failed: {action}")
    expected_terminal = source_checkpoint.phase1_fes + screen_fes
    for arm_name, arm in arms.items():
        if (
            arm.get("consumed_fes") != screen_fes
            or arm.get("terminal_fes") != expected_terminal
            or arm.get("runtime_warnings") != []
        ):
            raise ValueError(f"semantic parity arm gate failed: {action}/{arm_name}")
    threadpools = receipt.get("threadpools")
    if (
        not isinstance(threadpools, list)
        or not threadpools
        or any(item.get("num_threads") != 1 for item in threadpools)
    ):
        raise ValueError(f"native thread evidence failed: {action}")


def check_pilot(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    output_root = (REPOSITORY_ROOT / protocol["common_anchor"]["new_output_root"]).resolve()
    manifest = _load_json(output_root / "frozen_protocol" / "manifest.json")
    claimed_manifest = manifest.pop("manifest_sha256", None)
    if claimed_manifest != canonical_sha256(manifest):
        raise ValueError("semantic parity manifest hash drifted")
    expected = _manifest(protocol_file, protocol)
    if {**manifest, "manifest_sha256": claimed_manifest} != expected:
        raise ValueError("semantic parity manifest inputs drifted")
    receipt_files = {path.stem for path in (output_root / "receipts").glob("*.json")}
    if receipt_files != set(ACTION_NAMES):
        raise ValueError("semantic parity receipt set drifted")
    receipts = []
    screen_fes = int(protocol["common_anchor"]["screen_step_fes"])
    lanes = {str(lane["action"]): lane for lane in protocol["lanes"]}
    for action in ACTION_NAMES:
        receipt = _load_json(output_root / "receipts" / f"{action}.json")
        claimed = receipt.pop("receipt_hash", None)
        if claimed != canonical_sha256(receipt):
            raise ValueError(f"semantic parity receipt hash drifted: {action}")
        receipt["receipt_hash"] = claimed
        _validate_receipt(lanes[action], receipt, screen_fes)
        receipts.append(receipt)
    expected_summary = _summarize(receipts)
    if _load_json(output_root / "summary.json") != expected_summary:
        raise ValueError("semantic parity summary drifted")
    required_summary = {
        "lane_count": 4,
        "all_exact_screen_fes": True,
        "all_same_screen_checkpoint": True,
        "current_reference_evaluated_count": 0,
        "candidate_mechanism_passed_count": 4,
        "all_native_thread_limits_verified": True,
        "mechanism_screen_passed": True,
        "historical_terminal_parity_evaluated": False,
        "selector_evaluation_authorized": False,
    }
    failed = [
        key for key, value in required_summary.items() if expected_summary.get(key) != value
    ]
    if failed:
        raise ValueError(f"semantic parity summary gate failed: {failed}")
    return expected_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "check"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        protocol = load_protocol(args.protocol)
        output_root = REPOSITORY_ROOT / protocol["common_anchor"]["new_output_root"]
        if output_root.exists():
            raise ValueError(f"semantic parity output already exists: {output_root}")
        print(
            json.dumps(
                {
                    "lane_count": len(protocol["lanes"]),
                    "max_workers": protocol["common_anchor"]["max_workers"],
                    "output_root": str(output_root.resolve()),
                    "screen_step_fes": protocol["common_anchor"]["screen_step_fes"],
                    "source_inputs_valid": True,
                },
                sort_keys=True,
            )
        )
        return 0
    summary = run_pilot(args.protocol) if args.command == "run" else check_pilot(args.protocol)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
