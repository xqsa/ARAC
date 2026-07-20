from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from arac.actions.full_space_sep_cma import (
    CANONICAL_SEP_CMA_PARAMETERIZATION,
    CANONICAL_SEP_CMA_PARAMETERS_HASH,
    CANONICAL_SEP_CMA_POPULATION_SIZE,
    CANONICAL_SEP_CMA_REFERENCE_VERSION,
    FULL_SPACE_SEP_CMA_ACTION,
    NO_RESTART_POLICY,
    FullSpaceSepCmaAction,
    FullSpaceSepCmaExecutionState,
    full_space_sep_cma_anchor_hash,
    full_space_vector_hash,
)
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    ACTION_CEILING_HORIZONS,
    ACTION_CEILING_PROTOCOL_VERSION,
    GUARDED_EQ8_PROBE_FES,
    GUARDED_EQ8_WRITEBACK_ACTION,
    actionability_delta,
)
from experiments.pilots.exp_019_conflict_resolution_pilot import _diagnostic_worker
from experiments.pilots.exp_019_conflict_resolution_pilot.benchmark import (
    ConflictBenchmarkFactory,
)
from experiments.pilots.exp_019_conflict_resolution_pilot.diagnostic import (
    ARM_RESULT_FIELDS,
    CONFIG_PATH,
    CONTEXT_FIELDS,
    PILOT_SEEDS,
    REAL_CASES,
    SYNTHETIC_CASES,
    TrajectorySpec,
    _canonical_payload_hash,
    aggregate_action_ceiling,
    build_integrity_gate,
    build_specs,
    load_config,
    summarize_fe_accounting,
    validate_raw_rows,
)


def _context(context_id: str, cohort: str, problem_id: str) -> dict[str, str]:
    initial_mean = (0.0,) * 1000
    full_space_action = FullSpaceSepCmaAction(
        problem_id=problem_id,
        run_seed=117,
        checkpoint_fe=120,
        dispatch_checkpoint_hash="c" * 64,
        trigger_relation_hash="d" * 64,
        anchor_hash=full_space_sep_cma_anchor_hash(problem_id, initial_mean),
        initial_mean=initial_mean,
        initial_mean_hash=full_space_vector_hash(initial_mean),
        initial_state_hash="e" * 64,
        initial_sigma=0.5,
        lower_bound=-100.0,
        upper_bound=100.0,
        acceptance_fitness=50.0,
        population_size=CANONICAL_SEP_CMA_POPULATION_SIZE,
        budget_fes=50,
        parameterization=CANONICAL_SEP_CMA_PARAMETERIZATION,
        canonical_reference_version=CANONICAL_SEP_CMA_REFERENCE_VERSION,
        canonical_parameters_hash=CANONICAL_SEP_CMA_PARAMETERS_HASH,
        optimizer_seed=2026071901,
        seed_namespace=FULL_SPACE_SEP_CMA_ACTION,
        restart_policy=NO_RESTART_POLICY,
        issued_sweep=2,
        target_sweep=3,
        ttl_sweeps=1,
        expires_sweep=3,
    )
    row = {field: "" for field in CONTEXT_FIELDS}
    row.update(
        {
            "protocol_version": ACTION_CEILING_PROTOCOL_VERSION,
            "cohort": cohort,
            "problem_id": problem_id,
            "seed": "117",
            "context_id": context_id,
            "relation_id": "g0-1:v4-5",
            "action_set_hash": "a" * 64,
            "checkpoint_hash": "b" * 64,
            "dispatch_checkpoint_hash": "c" * 64,
            "phase_boundary_fe": "100",
            "dispatch_fe": "120",
            "issued_sweep": "2",
            "target_sweep": "3",
            "group_index": "1",
            "efficiency_ewma": "[0.1, 0.2]",
            "completed_efficiency_sweeps": "3",
            "stagnation_streaks": "[0, 0]",
            "population_sizes": "[2, 2]",
            "uniform_group_budgets": "[4, 4]",
            "horizon_fe": "50",
            "full_space_action_hash": full_space_action.action_hash,
            "full_space_action_payload": json.dumps(
                full_space_action.audit_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "full_space_initial_mean_hash": full_space_action.initial_mean_hash,
            "full_space_parameter_hash": CANONICAL_SEP_CMA_PARAMETERS_HASH,
            "full_space_optimizer_seed": "2026071901",
            "full_space_population_size": "24",
            "full_space_budget_fes": "50",
            "full_space_acceptance_fitness": "50.0",
            "selector_arm": "true_no_writeback",
            "selector_reason": "anchor_mismatch",
            "native_parity": "1",
            "runtime_authorized": "0",
            "status": "complete",
        }
    )
    return row


def _arm_rows(
    context: dict[str, str],
    *,
    winning_arm: str,
) -> list[dict[str, str]]:
    action_payload = json.loads(context["full_space_action_payload"])
    action_payload.pop("action")
    full_space_action = FullSpaceSepCmaAction(**action_payload)
    execution_state = FullSpaceSepCmaExecutionState.for_action(full_space_action)
    execution_state.start(
        full_space_action,
        current_fe=full_space_action.checkpoint_fe,
        current_sweep=full_space_action.target_sweep,
        dispatch_checkpoint_hash=full_space_action.dispatch_checkpoint_hash,
        trigger_relation_hash=full_space_action.trigger_relation_hash,
        anchor_hash=full_space_action.anchor_hash,
    )
    execution_state.complete(
        full_space_action,
        consumed_fes=full_space_action.budget_fes,
        completed_fe=(
            full_space_action.checkpoint_fe + full_space_action.budget_fes
        ),
        final_state_hash="3" * 64,
    )
    lifecycle_payload = json.dumps(
        execution_state.audit_payload(full_space_action),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    lifecycle_hash = execution_state.state_hash(full_space_action)
    rows: list[dict[str, str]] = []
    for arm in ACTION_CEILING_ARMS:
        for horizon_index, horizon in enumerate(ACTION_CEILING_HORIZONS, start=1):
            native_error = 100.0 - horizon_index
            arm_error = native_error
            if arm == "true_no_writeback":
                arm_error = native_error * 1.01
            if arm == winning_arm:
                arm_error = native_error * 0.90
            incumbent_mutated = arm in {
                "native_eq8",
                "exact_left",
                "exact_right",
                "exact_bridge",
                "efficiency_budget_reallocation",
                "delta_priority_scan",
                "stagnation_cross_group_warm_start",
                "full_space_sep_cma",
            }
            continuation_applied = arm in {
                "efficiency_budget_reallocation",
                "delta_priority_scan",
                "stagnation_cross_group_warm_start",
                "full_space_sep_cma",
            }
            warm_start_applied = arm == "stagnation_cross_group_warm_start"
            full_space_sep_cma = arm == "full_space_sep_cma"
            guarded = arm == GUARDED_EQ8_WRITEBACK_ACTION
            if (
                full_space_sep_cma and horizon in {"immediate", "sweep_1"}
            ) or (guarded and horizon == "immediate"):
                sweep_trace = "[]"
                order_trace = "[]"
                budget_trace = "[]"
                start_fe_trace = "[]"
            elif horizon == "immediate":
                sweep_trace = "[3]"
                order_trace = "[0]"
                budget_trace = "[4]"
                start_fe_trace = "[1]"
            else:
                sweep_trace = "[3, 3]"
                order_trace = "[0, 1]"
                budget_trace = "[4, 4]"
                if full_space_sep_cma:
                    start_fe_trace = "[51, 56]"
                elif guarded:
                    start_fe_trace = "[3, 8]"
                else:
                    start_fe_trace = "[1, 6]"
            guarded_payload = {
                "arm": GUARDED_EQ8_WRITEBACK_ACTION,
                "selection": "argmin_fitness",
                "tie_break": "evaluation_order",
                "probe_fes_budget": GUARDED_EQ8_PROBE_FES,
                "probe_fes_actual": GUARDED_EQ8_PROBE_FES,
                "selected_candidate": "previous",
                "accepted": True,
            }
            guarded_payload_json = json.dumps(
                guarded_payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            guarded_hash = _canonical_payload_hash(guarded_payload)
            row = {field: "" for field in ARM_RESULT_FIELDS}
            row.update(
                {
                    "protocol_version": ACTION_CEILING_PROTOCOL_VERSION,
                    "cohort": context["cohort"],
                    "problem_id": context["problem_id"],
                    "seed": context["seed"],
                    "context_id": context["context_id"],
                    "arm": arm,
                    "horizon": horizon,
                    "target_fe": str(
                        120
                        + {"immediate": 1, "sweep_1": 50, "sweep_3": 150}[
                            horizon
                        ]
                    ),
                    "natural_endpoint_fe": "500",
                    "native_error": str(native_error),
                    "arm_error": str(arm_error),
                    "delta": str(actionability_delta(native_error, arm_error)),
                    "action_budget_fes": (
                        "50"
                        if full_space_sep_cma
                        else (str(GUARDED_EQ8_PROBE_FES) if guarded else "0")
                    ),
                    "action_actual_fes": (
                        "50"
                        if full_space_sep_cma
                        else (str(GUARDED_EQ8_PROBE_FES) if guarded else "0")
                    ),
                    "action_instance_hash": (
                        context["full_space_action_hash"]
                        if full_space_sep_cma
                        else ("2" * 64 if guarded else "")
                    ),
                    "action_lifecycle_payload": (
                        lifecycle_payload
                        if full_space_sep_cma
                        else (guarded_payload_json if guarded else "")
                    ),
                    "action_lifecycle_hash": (
                        lifecycle_hash
                        if full_space_sep_cma
                        else (guarded_hash if guarded else "")
                    ),
                    "action_accepted": str(int(full_space_sep_cma or guarded)),
                    "action_candidate_hash": (
                        "1" * 64
                        if full_space_sep_cma
                        else ("2" * 64 if guarded else "")
                    ),
                    "action_candidate_fitness": (
                        "40.0" if full_space_sep_cma or guarded else ""
                    ),
                    "action_post_incumbent_hash": (
                        "1" * 64
                        if full_space_sep_cma
                        else ("2" * 64 if guarded else "")
                    ),
                    "optimizer_scope": (
                        "full_space"
                        if full_space_sep_cma
                        else (
                            "decomposed_groups"
                            if continuation_applied
                            else "relation_writeback"
                        )
                    ),
                    "optimizer_parameter_hash": (
                        context["full_space_parameter_hash"]
                        if full_space_sep_cma
                        else ""
                    ),
                    "optimizer_initial_state_hash": (
                        "e" * 64 if full_space_sep_cma else ""
                    ),
                    "optimizer_final_state_hash": (
                        "3" * 64 if full_space_sep_cma else ""
                    ),
                    "optimizer_population_size": "24" if full_space_sep_cma else "0",
                    "optimizer_generation_count": "2" if full_space_sep_cma else "0",
                    "counterfactual_applied": str(
                        int(incumbent_mutated or continuation_applied or guarded)
                    ),
                    "mutation_norm": str(float(incumbent_mutated)),
                    "optimizer_mean_mutation_norm": "0.0",
                    "continuation_policy_applied": str(
                        int(continuation_applied)
                    ),
                    "execution_sweep_trace": sweep_trace,
                    "execution_order_trace": order_trace,
                    "group_budget_trace": budget_trace,
                    "execution_start_fe_trace": start_fe_trace,
                    "warm_start_trigger_count": str(int(warm_start_applied)),
                    "warm_start_mean_shift_norm": str(float(warm_start_applied)),
                    "selected_candidate": "previous" if guarded else arm,
                    "runtime_authorized": "0",
                    "status": "complete",
                }
            )
            rows.append(row)
    return rows


def test_frozen_config_and_run_matrices() -> None:
    config = load_config()

    assert config["observer_only"] is True
    assert config["continuation_actions"]["efficiency_budget_reallocation"] == {
        "ewma_alpha": 0.3,
        "cold_start_uniform_sweeps": 1,
        "minimum_population_multiples": 1,
        "maximum_uniform_budget_multiples": 3,
        "preserve_total_requested_fes": True,
    }
    assert config["continuation_actions"]["delta_priority_scan"]["tie_break"] == (
        "original_group_index_ascending"
    )
    assert config["continuation_actions"]["stagnation_cross_group_warm_start"][
        "trigger_streak"
    ] == 3
    assert config["continuation_actions"]["full_space_sep_cma"] == {
        "scope": "full_space",
        "dimension": 1000,
        "population_size": 24,
        "parameterization": "ros_hansen_2008_pypop7",
        "canonical_reference_version": (
            "pypop7-sepcmaes@67b29061d121cba9a5715897a2eb5d409df04c2d"
        ),
        "budget_source": "one_actual_native_sweep_horizon",
        "resume_native_after_action": True,
        "restart_policy": "none",
        "acceptance_rule": "strict_improvement",
    }
    smoke = build_specs("smoke")
    assert len(smoke) == 6
    assert {(spec.problem_id, spec.seed, spec.max_fes) for spec in smoke} == {
        (case, seed, 300_000)
        for case in ("E3", "S5")
        for seed in (117, 118, 119)
    }
    assert all(":" not in spec.trajectory_id for spec in smoke)
    pilot = build_specs("pilot")
    assert len(pilot) == 40
    assert {(spec.problem_id, spec.seed) for spec in pilot if spec.cohort == "real_aob"} == {
        (case, seed) for case in REAL_CASES for seed in PILOT_SEEDS
    }
    assert {
        (spec.problem_id, spec.seed)
        for spec in pilot
        if spec.cohort == "synthetic_conflict"
    } == {(case, seed) for case in SYNTHETIC_CASES for seed in PILOT_SEEDS}


def test_legacy_config_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    legacy = copy.deepcopy(config)
    legacy["protocol_version"] = "exp019-action-ceiling-v4"
    path = tmp_path / "diagnostic_config.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy exp019"):
        load_config(path)


def test_raw_rows_require_every_frozen_arm_and_horizon() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")

    observations = validate_raw_rows([context], rows)

    assert len(observations) == len(ACTION_CEILING_ARMS) * len(ACTION_CEILING_HORIZONS)
    with pytest.raises(ValueError, match="every frozen arm and horizon"):
        validate_raw_rows([context], rows[:-1])


def test_incomplete_context_and_inconsistent_mutation_flag_fail_closed() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    invalid_context = {**context, "native_parity": "0", "status": "invalid"}
    with pytest.raises(ValueError, match="native parity"):
        validate_raw_rows([invalid_context], rows)
    rows[0]["counterfactual_applied"] = "0"
    with pytest.raises(ValueError, match="disagrees with branch mutation"):
        validate_raw_rows([context], rows)


def test_full_space_strict_acceptance_is_recomputed_from_fitness() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    sep_rows = [row for row in rows if row["arm"] == "full_space_sep_cma"]

    for row in sep_rows:
        row["action_accepted"] = "0"
        row["action_post_incumbent_hash"] = context[
            "full_space_initial_mean_hash"
        ]
    with pytest.raises(ValueError, match="full-space Sep-CMA arm contract"):
        validate_raw_rows([context], rows)

    for row in sep_rows:
        row["action_candidate_fitness"] = "50.0"
    validate_raw_rows([context], rows)


def test_full_space_parameters_must_match_pinned_snapshot() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    context["full_space_parameter_hash"] = "f" * 64

    with pytest.raises(ValueError, match="full-space Sep-CMA context contract"):
        validate_raw_rows([context], rows)


def test_full_space_candidate_outcome_must_match_across_horizons() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    sweep_3 = next(
        row
        for row in rows
        if row["arm"] == FULL_SPACE_SEP_CMA_ACTION
        and row["horizon"] == "sweep_3"
    )
    sweep_3["action_candidate_fitness"] = "41.0"

    with pytest.raises(ValueError, match="action outcome differs across horizons"):
        validate_raw_rows([context], rows)

    rows = _arm_rows(context, winning_arm="exact_left")
    sweep_3 = next(
        row
        for row in rows
        if row["arm"] == FULL_SPACE_SEP_CMA_ACTION
        and row["horizon"] == "sweep_3"
    )
    sweep_3["action_candidate_hash"] = "2" * 64
    sweep_3["action_post_incumbent_hash"] = "2" * 64

    with pytest.raises(ValueError, match="action outcome differs across horizons"):
        validate_raw_rows([context], rows)


def test_full_space_lifecycle_is_bound_and_identical_across_horizons() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    sweep_3 = next(
        row
        for row in rows
        if row["arm"] == FULL_SPACE_SEP_CMA_ACTION
        and row["horizon"] == "sweep_3"
    )
    action_payload = json.loads(context["full_space_action_payload"])
    action_payload.pop("action")
    action = FullSpaceSepCmaAction(**action_payload)
    lifecycle_payload = json.loads(sweep_3["action_lifecycle_payload"])
    lifecycle_payload.pop("action")
    lifecycle_payload["final_state_hash"] = "4" * 64
    lifecycle = FullSpaceSepCmaExecutionState(**lifecycle_payload)
    lifecycle.validate_for(action)
    sweep_3["optimizer_final_state_hash"] = "4" * 64
    sweep_3["action_lifecycle_payload"] = json.dumps(
        lifecycle.audit_payload(action),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    sweep_3["action_lifecycle_hash"] = lifecycle.state_hash(action)

    with pytest.raises(ValueError, match="action outcome differs across horizons"):
        validate_raw_rows([context], rows)

    rows = _arm_rows(context, winning_arm="exact_left")
    sweep_3 = next(
        row
        for row in rows
        if row["arm"] == FULL_SPACE_SEP_CMA_ACTION
        and row["horizon"] == "sweep_3"
    )
    payload = json.loads(sweep_3["action_lifecycle_payload"])
    payload["completed_fe"] = 169
    sweep_3["action_lifecycle_payload"] = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    with pytest.raises(ValueError, match="lifecycle payload is invalid"):
        validate_raw_rows([context], rows)


def test_legacy_raw_row_fails_closed() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    context["protocol_version"] = "exp019-action-ceiling-v4"

    with pytest.raises(ValueError, match="legacy action-ceiling context row"):
        validate_raw_rows([context], rows)


def test_adaptive_budget_trace_must_preserve_complete_sweep_total() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    budget_row = next(
        row
        for row in rows
        if row["arm"] == "efficiency_budget_reallocation"
        and row["horizon"] == "sweep_1"
    )
    budget_row["group_budget_trace"] = "[5, 4]"

    with pytest.raises(ValueError, match="preserve total requested FEs"):
        validate_raw_rows([context], rows)


def test_integrity_gate_and_fe_summary_cover_complete_stage() -> None:
    spec = TrajectorySpec("smoke", "real_aob", "E3", 117, 300_000)
    contexts = [
        _context(f"real:E3:117:r{index}", "real_aob", "E3")
        for index in range(4)
    ]
    for context in contexts:
        context["seed"] = "117"
    rows = [
        row
        for context in contexts
        for row in _arm_rows(context, winning_arm="exact_left")
    ]

    gate = build_integrity_gate([spec], contexts, rows)
    fe_summary = summarize_fe_accounting([spec], contexts, rows)

    assert gate["passed"] == 1
    assert gate["expected_context_count"] == 1
    assert gate["expected_arm_result_count"] == 1
    assert fe_summary["nominal_trajectory_fe_total"] == 300_000
    assert fe_summary["branch_action_fe_by_arm"]["native_eq8"] == 0
    assert fe_summary["branch_action_fe_by_arm"]["full_space_sep_cma"] == 200


def test_delta_is_recomputed_from_raw_errors() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    rows[6]["delta"] = "999"

    with pytest.raises(ValueError, match="delta does not match"):
        validate_raw_rows([context], rows)


def test_real_and_synthetic_summaries_are_not_pooled() -> None:
    real = _context("real:E3:117:r0", "real_aob", "E3")
    synthetic = _context("synthetic:E3:117:r0", "synthetic_conflict", "E3")
    summaries = aggregate_action_ceiling(
        [real, synthetic],
        _arm_rows(real, winning_arm="exact_left")
        + _arm_rows(synthetic, winning_arm="exact_right"),
    )

    assert {(row["cohort"], row["horizon"]) for row in summaries} == {
        (cohort, horizon)
        for cohort in ("real_aob", "synthetic_conflict")
        for horizon in ACTION_CEILING_HORIZONS
    }
    real_primary = next(
        row for row in summaries if row["cohort"] == "real_aob" and row["horizon"] == "sweep_1"
    )
    synthetic_primary = next(
        row
        for row in summaries
        if row["cohort"] == "synthetic_conflict" and row["horizon"] == "sweep_1"
    )
    assert real_primary["sbs_arm"] == "exact_left"
    assert synthetic_primary["sbs_arm"] == "exact_right"


def test_worker_builds_explicit_action_ceiling_request() -> None:
    args = argparse.Namespace(
        cohort="real_aob",
        case="R4",
        seed=1,
        max_fes=100_000,
        output_root="results/test",
        timestamp="fixed-smoke",
    )
    request = _diagnostic_worker.build_runner_args(args)

    assert request[request.index("--functions") + 1] == "rastrigin"
    assert request[request.index("--ids") + 1] == "4"
    assert request[request.index("--relation-policy") + 1] == "action_ceiling"
    assert request[request.index("--action-ceiling-cohort") + 1] == "real_aob"
    assert "--action-ceiling-capture" in request


def test_synthetic_worker_replaces_factory_only_in_child(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import hcc_smoke_runner

    observed: list[list[str]] = []
    monkeypatch.setattr(hcc_smoke_runner, "main", lambda args: observed.append(args))
    monkeypatch.setattr(hcc_smoke_runner, "Benchmark", object())

    result = _diagnostic_worker.main(
        [
            "--cohort",
            "synthetic_conflict",
            "--case",
            "A4",
            "--seed",
            "1",
            "--max-fes",
            "100000",
            "--output-root",
            "results/test",
            "--timestamp",
            "fixed-smoke",
        ]
    )

    assert result == 0
    assert hcc_smoke_runner.Benchmark is ConflictBenchmarkFactory
    assert len(observed) == 1
