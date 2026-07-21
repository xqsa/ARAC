from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest

from arac.actions.budget_reallocation import (
    FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
    BudgetAllocationExecutionState,
)
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
    ACTION_CEILING_FULL_MATRIX_PROFILE,
    ACTION_CEILING_ARMS,
    ACTION_CEILING_HORIZONS,
    ACTION_CEILING_PROTOCOL_VERSION,
    AUDITED_RELATION_WRITEBACK_ACTIONS,
    GUARDED_EQ8_PROBE_FES,
    GUARDED_EQ8_WRITEBACK_ACTION,
    RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION,
    RS_FAMILY_TARGET_PROFILE,
    actionability_delta,
    relation_writeback_action_parameters,
)
from arac.backends.hcc_action_ceiling import freeze_efficiency_budget_action
from arac.backends.hcc import required_aob_data_files
from experiments.pilots.exp_019_conflict_resolution_pilot import _diagnostic_worker
from experiments.pilots.exp_019_conflict_resolution_pilot.benchmark import (
    ConflictBenchmarkFactory,
    VENDOR_DATA_DIR,
)
from experiments.pilots.exp_019_conflict_resolution_pilot.diagnostic import (
    ARM_RESULT_FIELDS,
    AOB_INPUT_MANIFEST_FIELDS,
    CONFIG_PATH,
    CONTEXT_FIELDS,
    PILOT_SEEDS,
    REAL_CASES,
    RS_NO_RELATION_CONTEXT_CASES,
    RS_SMOKE_CASES,
    RS_VALIDATION_CASES,
    RS_VALIDATION_SEEDS,
    SYNTHETIC_CASES,
    TrajectorySpec,
    _canonical_payload_hash,
    aggregate_action_ceiling,
    aggregate_stage_artifacts,
    build_integrity_gate,
    build_rs_family_integrity_gate,
    build_specs,
    load_config,
    summarize_fe_accounting,
    summarize_rs_family_target,
    validate_raw_rows,
    validate_rs_family_target_rows,
    _trajectory_artifacts,
    _write_csv,
)


def _context(
    context_id: str,
    cohort: str,
    problem_id: str,
    *,
    seed: int = 117,
) -> dict[str, str]:
    initial_mean = (0.0,) * 1000
    action_set_hash = _canonical_payload_hash(
        {"kind": "action_set", "context_id": context_id}
    )
    checkpoint_hash = _canonical_payload_hash(
        {"kind": "checkpoint", "context_id": context_id}
    )
    dispatch_checkpoint_hash = _canonical_payload_hash(
        {"kind": "dispatch_checkpoint", "context_id": context_id}
    )
    full_space_action = FullSpaceSepCmaAction(
        problem_id=problem_id,
        run_seed=seed,
        checkpoint_fe=120,
        dispatch_checkpoint_hash=dispatch_checkpoint_hash,
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
            "seed": str(seed),
            "context_id": context_id,
            "relation_id": "g0-1:v4-5",
            "action_set_hash": action_set_hash,
            "checkpoint_hash": checkpoint_hash,
            "dispatch_checkpoint_hash": dispatch_checkpoint_hash,
            "dispatch_anchor_hash": "4" * 64,
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
            audited_writeback = arm in AUDITED_RELATION_WRITEBACK_ACTIONS
            incumbent_mutated = arm in {
                "native_eq8",
                "exact_left",
                "exact_right",
                "exact_bridge",
                "efficiency_budget_reallocation",
                "delta_priority_scan",
                "stagnation_cross_group_warm_start",
                "full_space_sep_cma",
            } or audited_writeback
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
            writeback_fields: dict[str, str] = {}
            if audited_writeback:
                parameters = relation_writeback_action_parameters(arm)
                candidate_names = {
                    GUARDED_EQ8_WRITEBACK_ACTION: [
                        "current",
                        "previous",
                        "eq8_blend",
                    ],
                    "stagnation_guard_writeback": ["current", "native_eq8"],
                    "contribution_owner_writeback": [
                        "current",
                        "left_owner",
                        "right_owner",
                    ],
                    "contribution_owner_reverse_writeback": [
                        "current",
                        "left_owner",
                        "right_owner",
                    ],
                }[arm]
                selected = {
                    GUARDED_EQ8_WRITEBACK_ACTION: "previous",
                    "stagnation_guard_writeback": "native_eq8",
                    "contribution_owner_writeback": "right_owner",
                    "contribution_owner_reverse_writeback": "left_owner",
                }[arm]
                candidate_hashes = {
                    name: _canonical_payload_hash({"candidate": name})
                    for name in candidate_names
                }
                instance = {
                    "arm": arm,
                    "context_hash": context["dispatch_checkpoint_hash"],
                    "action_set_hash": context["action_set_hash"],
                    "relation": {"owners": [0, 1], "shared": [4, 5]},
                    "dispatch_anchor_hash": "4" * 64,
                    "previous_values_hash": "5" * 64,
                    "current_values_hash": "6" * 64,
                    "previous_delta": 1.0,
                    "current_delta": 3.0,
                    "parameters": parameters,
                    "parameter_hash": _canonical_payload_hash(parameters),
                    "action_budget_fes": int(parameters["probe_fes"]),
                    "candidates": [
                        {"name": name, "values_hash": candidate_hashes[name]}
                        for name in candidate_names
                    ],
                }
                instance_hash = _canonical_payload_hash(instance)
                writeback_payload: dict[str, object] = {
                    "instance": instance,
                    "instance_hash": instance_hash,
                    "action_actual_fes": int(parameters["probe_fes"]),
                    "selected_candidate": selected,
                    "selected_values_hash": candidate_hashes[selected],
                    "post_incumbent_hash": "7" * 64,
                    "accepted": True,
                }
                candidate_fitness = ""
                if guarded:
                    outcomes = [
                        ("current", 50.0),
                        ("previous", 40.0),
                        ("eq8_blend", 45.0),
                    ]
                    writeback_payload["selected_fitness"] = 40.0
                    writeback_payload["probe_outcomes"] = [
                        {
                            "name": name,
                            "values_hash": candidate_hashes[name],
                            "fitness": fitness,
                        }
                        for name, fitness in outcomes
                    ]
                    candidate_fitness = "40.0"
                writeback_payload_json = json.dumps(
                    writeback_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                writeback_fields = {
                    "action_instance_hash": instance_hash,
                    "action_lifecycle_payload": writeback_payload_json,
                    "action_lifecycle_hash": _canonical_payload_hash(
                        writeback_payload
                    ),
                    "action_accepted": "1",
                    "action_candidate_hash": candidate_hashes[selected],
                    "action_candidate_fitness": candidate_fitness,
                    "action_post_incumbent_hash": "7" * 64,
                    "selected_candidate": selected,
                }
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
                        else writeback_fields.get("action_instance_hash", "")
                    ),
                    "action_lifecycle_payload": (
                        lifecycle_payload
                        if full_space_sep_cma
                        else writeback_fields.get("action_lifecycle_payload", "")
                    ),
                    "action_lifecycle_hash": (
                        lifecycle_hash
                        if full_space_sep_cma
                        else writeback_fields.get("action_lifecycle_hash", "")
                    ),
                    "action_accepted": (
                        "1"
                        if full_space_sep_cma
                        else writeback_fields.get("action_accepted", "0")
                    ),
                    "action_candidate_hash": (
                        "1" * 64
                        if full_space_sep_cma
                        else writeback_fields.get("action_candidate_hash", "")
                    ),
                    "action_candidate_fitness": (
                        "40.0"
                        if full_space_sep_cma
                        else writeback_fields.get("action_candidate_fitness", "")
                    ),
                    "action_post_incumbent_hash": (
                        "1" * 64
                        if full_space_sep_cma
                        else writeback_fields.get("action_post_incumbent_hash", "")
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
                    "selected_candidate": writeback_fields.get(
                        "selected_candidate",
                        arm,
                    ),
                    "runtime_authorized": "0",
                    "status": "complete",
                }
            )
            rows.append(row)
    return rows


def _rs_rows(
    problem_id: str,
    context_id: str,
    *,
    seed: int = 117,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    target_arm = (
        "full_space_sep_cma"
        if problem_id.startswith("R")
        else FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
    )
    context = _context(
        context_id,
        "real_aob",
        problem_id,
        seed=seed,
    )
    fixture_arm = (
        target_arm
        if problem_id.startswith("R")
        else "efficiency_budget_reallocation"
    )
    all_rows = _arm_rows(context, winning_arm=fixture_arm)
    context.update(
        {
            "protocol_version": RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION,
            "anchor_values": "[0.0,0.0]",
            "left_values": "[-1.0,-1.0]",
            "right_values": "[1.0,1.0]",
            "bridge_values": "[0.0,0.0]",
            "bridge_weights": '{"left_owner":0.5,"right_owner":0.5}',
        }
    )
    relation_index_text = context_id.rpartition(":r")[2]
    if relation_index_text.isdigit():
        relation_index = int(relation_index_text)
        context["relation_id"] = (
            f"g{relation_index}-{relation_index + 1}:v{4 + relation_index}"
        )
    if problem_id.startswith("S"):
        for field in CONTEXT_FIELDS:
            if field.startswith("full_space_"):
                context[field] = ""
    rows = [
        row
        for row in all_rows
        if row["arm"] in {"native_eq8", fixture_arm}
    ]
    for row in rows:
        row["protocol_version"] = RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
        if row["arm"] == "native_eq8":
            row["action_post_incumbent_hash"] = "7" * 64
    if problem_id.startswith("S"):
        budget_action = freeze_efficiency_budget_action(
            problem_id=problem_id,
            run_seed=seed,
            checkpoint_fe=int(context["dispatch_fe"]),
            dispatch_checkpoint_hash=context["dispatch_checkpoint_hash"],
            source_efficiency_ewma=(0.1, 0.2),
            population_sizes=(2, 2),
            uniform_group_budgets=(4, 4),
            issued_sweep=int(context["target_sweep"]),
            target_sweep=int(context["target_sweep"]) + 1,
        )
        lifecycle = BudgetAllocationExecutionState.for_action(budget_action)
        lifecycle.consume(
            budget_action,
            current_sweep=budget_action.target_sweep,
            application_fe=51,
            dispatch_checkpoint_hash=budget_action.dispatch_checkpoint_hash,
            anchor_hash=budget_action.anchor_hash,
        )
        lifecycle_audit = {
            "action": FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
            "instance": budget_action.audit_payload(),
            "instance_hash": budget_action.action_hash,
            "execution": lifecycle.audit_payload(budget_action),
            "execution_hash": lifecycle.state_hash(budget_action),
        }
        lifecycle_payload = json.dumps(
            lifecycle_audit,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        lifecycle_hash = _canonical_payload_hash(lifecycle_audit)
        target_rows = [row for row in rows if row["arm"] == fixture_arm]
        for row in target_rows:
            row.update(
                {
                    "arm": target_arm,
                    "action_instance_hash": budget_action.action_hash,
                    "action_lifecycle_payload": lifecycle_payload,
                    "action_lifecycle_hash": lifecycle_hash,
                    "action_accepted": "1",
                    "action_candidate_hash": _canonical_payload_hash(
                        {"group_budgets": budget_action.group_budgets}
                    ),
                    "action_post_incumbent_hash": "7" * 64,
                    "optimizer_parameter_hash": budget_action.parameter_hash,
                    "selected_candidate": target_arm,
                }
            )
            if row["horizon"] in {"immediate", "sweep_1"}:
                row["continuation_policy_applied"] = "0"
            else:
                row.update(
                    {
                        "execution_sweep_trace": "[3,3,4,4,5,5]",
                        "execution_order_trace": "[0,1,0,1,0,1]",
                        "group_budget_trace": json.dumps(
                            [4, 4, *budget_action.group_budgets, 4, 4]
                        ),
                        "execution_start_fe_trace": "[1,6,51,56,101,106]",
                        "continuation_policy_applied": "1",
                    }
                )
    return context, rows


def _valid_aob_input_rows(spec: TrajectorySpec) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in required_aob_data_files(VENDOR_DATA_DIR, int(spec.problem_id[1:])):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "problem_id": spec.problem_id,
                "file": path.name,
                "path": str(path.resolve()),
                "sha256_before": digest,
                "sha256_after": digest,
                "unchanged": "1",
            }
        )
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
    real_only = build_specs("pilot", cohort="real_aob")
    assert len(real_only) == 25
    assert all(spec.cohort == "real_aob" for spec in real_only)
    assert {
        (spec.problem_id, spec.seed, spec.max_fes) for spec in real_only
    } == {
        (case, seed, 3_000_000)
        for case in REAL_CASES
        for seed in PILOT_SEEDS
    }
    with pytest.raises(ValueError, match="smoke stage only contains real AOB"):
        build_specs("smoke", cohort="synthetic_conflict")


def test_rs_family_target_config_and_run_matrices_are_separate_from_v6() -> None:
    config = load_config()["rs_family_target_validation"]

    assert config["profile"] == RS_FAMILY_TARGET_PROFILE
    assert config["protocol_version"] == RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
    assert config["cutoff_tie_policy"] == "structural_key"
    assert config["no_relation_context_cases"] == list(
        RS_NO_RELATION_CONTEXT_CASES
    )
    assert config["rs_family_validation"]["jobs"] == 20
    smoke = build_specs("rs_smoke")
    assert len(smoke) == 4
    assert {(spec.problem_id, spec.seed, spec.max_fes) for spec in smoke} == {
        (case, 117, 300_000) for case in RS_SMOKE_CASES
    }
    validation = build_specs("rs_family_validation")
    assert len(validation) == 50
    assert {
        (spec.problem_id, spec.seed, spec.max_fes) for spec in validation
    } == {
        (case, seed, 3_000_000)
        for case in RS_VALIDATION_CASES
        for seed in RS_VALIDATION_SEEDS
    }
    assert all(
        spec.action_ceiling_profile == RS_FAMILY_TARGET_PROFILE
        for spec in (*smoke, *validation)
    )
    assert all(
        spec.action_ceiling_profile == ACTION_CEILING_FULL_MATRIX_PROFILE
        for spec in build_specs("pilot")
    )
    with pytest.raises(ValueError, match="only contain real AOB"):
        build_specs("rs_smoke", cohort="synthetic_conflict")


def test_legacy_config_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    legacy = copy.deepcopy(config)
    legacy["protocol_version"] = "exp019-action-ceiling-v4"
    path = tmp_path / "diagnostic_config.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy exp019"):
        load_config(path)

    stale_schema = copy.deepcopy(config)
    stale_schema["schema_version"] = "exp019-action-ceiling-config-v6"
    path.write_text(json.dumps(stale_schema), encoding="utf-8")
    with pytest.raises(ValueError, match="config schema mismatch"):
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


@pytest.mark.parametrize("tampered_field", ["parameters", "dispatch_anchor_hash"])
def test_relation_writeback_instance_binds_parameters_and_anchor(
    tampered_field: str,
) -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    target = next(
        row
        for row in rows
        if row["arm"] == "contribution_owner_writeback"
        and row["horizon"] == "sweep_1"
    )
    payload = json.loads(target["action_lifecycle_payload"])
    instance = payload["instance"]
    if tampered_field == "parameters":
        instance["parameters"]["winner"] = "smaller_delta_owner"
        instance["parameter_hash"] = _canonical_payload_hash(instance["parameters"])
    else:
        instance["dispatch_anchor_hash"] = "9" * 64
    instance_hash = _canonical_payload_hash(instance)
    payload["instance_hash"] = instance_hash
    target["action_instance_hash"] = instance_hash
    target["action_lifecycle_payload"] = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    target["action_lifecycle_hash"] = _canonical_payload_hash(payload)

    with pytest.raises(ValueError, match="relation writeback arm contract"):
        validate_raw_rows([context], rows)


def test_legacy_raw_row_fails_closed() -> None:
    context = _context("real:E3:117:r0", "real_aob", "E3")
    rows = _arm_rows(context, winning_arm="exact_left")
    context["protocol_version"] = "exp019-action-ceiling-v4"

    with pytest.raises(ValueError, match="legacy action-ceiling context row"):
        validate_raw_rows([context], rows)


@pytest.mark.parametrize(
    ("problem_id", "target_arm"),
    [
        ("R2", "full_space_sep_cma"),
        ("S2", FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION),
    ],
)
def test_rs_target_validator_accepts_only_the_family_pair(
    problem_id: str,
    target_arm: str,
) -> None:
    context, rows = _rs_rows(problem_id, f"rs:{problem_id}:117:r0")

    observations = validate_rs_family_target_rows([context], rows)

    assert len(observations) == 2 * len(ACTION_CEILING_HORIZONS)
    assert {row.arm for row in observations} == {"native_eq8", target_arm}
    with pytest.raises(ValueError, match="legacy action-ceiling context row"):
        validate_raw_rows([context], rows)
    with pytest.raises(ValueError, match="exactly two arms and three horizons"):
        validate_rs_family_target_rows([context], rows[:-1])


def test_rs_target_validator_requires_r_lifecycle_and_forbids_it_for_s() -> None:
    r_context, r_rows = _rs_rows("R2", "rs:R2:117:r0")
    sep_row = next(
        row
        for row in r_rows
        if row["arm"] == "full_space_sep_cma" and row["horizon"] == "sweep_1"
    )
    sep_row["action_lifecycle_hash"] = "f" * 64
    with pytest.raises(ValueError, match="Sep-CMA"):
        validate_rs_family_target_rows([r_context], r_rows)

    s_context, s_rows = _rs_rows("S2", "rs:S2:117:r0")
    s_context["full_space_action_hash"] = "a" * 64
    with pytest.raises(ValueError, match="contains Sep-CMA fields"):
        validate_rs_family_target_rows([s_context], s_rows)


def test_s_target_requires_consumed_frozen_budget_lifecycle() -> None:
    context, rows = _rs_rows("S2", "rs:S2:117:r0")
    target_rows = [
        row
        for row in rows
        if row["arm"] == FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
    ]

    validate_rs_family_target_rows([context], rows)
    assert all(row["action_instance_hash"] for row in target_rows)
    assert all(row["action_lifecycle_hash"] for row in target_rows)

    payload = json.loads(target_rows[0]["action_lifecycle_payload"])
    assert payload["instance"]["checkpoint_fe"] == int(context["dispatch_fe"])
    assert payload["instance"]["checkpoint_fe"] != int(context["phase_boundary_fe"])
    payload["execution"].update(
        {
            "status": "issued",
            "consumed_sweep": None,
            "application_fe": None,
            "applied_group_budgets": [],
        }
    )
    payload["execution_hash"] = _canonical_payload_hash(payload["execution"])
    target_rows[0]["action_lifecycle_payload"] = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    target_rows[0]["action_lifecycle_hash"] = _canonical_payload_hash(payload)
    with pytest.raises(ValueError, match="frozen budget arm contract"):
        validate_rs_family_target_rows([context], rows)


def test_s_target_rejects_non_native_dispatch_order_and_post_state() -> None:
    context, original_rows = _rs_rows("S2", "rs:S2:117:r0")

    non_monotonic = copy.deepcopy(original_rows)
    sweep_3 = next(
        row
        for row in non_monotonic
        if row["arm"] == FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
        and row["horizon"] == "sweep_3"
    )
    sweep_3["execution_sweep_trace"] = "[5,5,4,4,3,3]"
    with pytest.raises(ValueError, match="horizon FE contract"):
        validate_rs_family_target_rows([context], non_monotonic)

    reordered = copy.deepcopy(original_rows)
    sweep_3 = next(
        row
        for row in reordered
        if row["arm"] == FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
        and row["horizon"] == "sweep_3"
    )
    payload = json.loads(sweep_3["action_lifecycle_payload"])
    action_budgets = payload["instance"]["group_budgets"]
    sweep_3["execution_order_trace"] = "[1,0,1,0,1,0]"
    sweep_3["group_budget_trace"] = json.dumps(
        [4, 4, action_budgets[1], action_budgets[0], 4, 4]
    )
    with pytest.raises(ValueError, match="native group order"):
        validate_rs_family_target_rows([context], reordered)

    changed_post_state = copy.deepcopy(original_rows)
    for row in changed_post_state:
        if row["arm"] == FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION:
            row["action_post_incumbent_hash"] = "8" * 64
    with pytest.raises(ValueError, match="non-budget dispatch state"):
        validate_rs_family_target_rows([context], changed_post_state)


def test_rs_target_case_bootstrap_and_catastrophic_gate() -> None:
    contexts: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    for seed in RS_VALIDATION_SEEDS:
        context, case_rows = _rs_rows("S2", f"rs:S2:{seed}:r0", seed=seed)
        contexts.append(context)
        rows.extend(case_rows)
    observations = validate_rs_family_target_rows(contexts, rows)

    summary = summarize_rs_family_target(observations, inferential=True)[0]

    assert summary["cluster_count"] == 5
    assert summary["mean_delta"] > 0.0
    assert summary["min_delta"] > 0.0
    assert summary["max_delta"] > 0.0
    assert summary["delta_lcb"] > 0.0
    assert summary["delta_ucb"] > 0.0
    assert summary["positive_count"] == 5
    assert summary["material_positive_count"] == 5
    assert summary["catastrophic_count"] == 0
    assert summary["gate"] == "target_action_validated"

    target = next(
        row
        for row in rows
        if row["context_id"] == "rs:S2:117:r0"
        and row["arm"] == FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
        and row["horizon"] == "sweep_1"
    )
    target["arm_error"] = str(float(target["native_error"]) * 1.25)
    target["delta"] = str(
        actionability_delta(
            float(target["native_error"]),
            float(target["arm_error"]),
        )
    )
    observations = validate_rs_family_target_rows(contexts, rows)
    summary = summarize_rs_family_target(observations, inferential=True)[0]
    assert summary["catastrophic_count"] == 1
    assert summary["gate"] == "reject_target_action_catastrophic_loss"


def test_rs_integrity_requires_four_unique_relation_checkpoints() -> None:
    spec = build_specs("rs_smoke")[0]
    contexts: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    for relation_index in range(4):
        context, relation_rows = _rs_rows(
            spec.problem_id,
            f"rs:{spec.problem_id}:{spec.seed}:r{relation_index}",
        )
        contexts.append(context)
        rows.extend(relation_rows)

    gate = build_rs_family_integrity_gate([spec], contexts, rows)

    assert gate["unique_relation_ids_per_trajectory"] == 1
    assert gate["unique_action_set_hashes_per_trajectory"] == 1
    assert gate["unique_dispatch_checkpoint_hashes_per_trajectory"] == 1
    contexts[1]["relation_id"] = contexts[0]["relation_id"]
    with pytest.raises(ValueError, match="unique_relation_ids_per_trajectory"):
        build_rs_family_integrity_gate([spec], contexts, rows)


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


@pytest.mark.parametrize(
    ("case", "function_name", "function_id"),
    [("R2", "rastrigin", "2"), ("S6", "schwefel", "6")],
)
def test_worker_builds_rs_family_target_request(
    case: str,
    function_name: str,
    function_id: str,
) -> None:
    args = argparse.Namespace(
        cohort="real_aob",
        case=case,
        seed=117,
        max_fes=300_000,
        output_root="results/test",
        timestamp="rs-smoke",
        profile=RS_FAMILY_TARGET_PROFILE,
    )

    request = _diagnostic_worker.build_runner_args(args)

    assert request[request.index("--functions") + 1] == function_name
    assert request[request.index("--ids") + 1] == function_id
    assert request[request.index("--action-ceiling-profile") + 1] == (
        RS_FAMILY_TARGET_PROFILE
    )


def test_rs_target_worker_rejects_header_only_artifacts(tmp_path: Path) -> None:
    args = argparse.Namespace(
        cohort="real_aob",
        case="R2",
        seed=117,
        max_fes=300_000,
        output_root=str(tmp_path),
        timestamp="rs-smoke",
        profile=RS_FAMILY_TARGET_PROFILE,
    )
    base = tmp_path / args.timestamp / "rastrigin"
    contexts: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    for relation_index in range(4):
        context, relation_rows = _rs_rows(
            "R2",
            f"rs:R2:117:r{relation_index}",
        )
        contexts.append(context)
        rows.extend(relation_rows)
    _write_csv(
        base / "R2_action_ceiling_contexts.csv",
        contexts,
        CONTEXT_FIELDS,
    )
    _write_csv(
        base / "R2_action_ceiling_arm_results.csv",
        rows,
        ARM_RESULT_FIELDS,
    )

    _diagnostic_worker.require_rs_target_artifacts(args)

    _write_csv(base / "R2_action_ceiling_contexts.csv", [], CONTEXT_FIELDS)
    _write_csv(base / "R2_action_ceiling_arm_results.csv", [], ARM_RESULT_FIELDS)
    with pytest.raises(RuntimeError, match="incomplete action-ceiling artifacts"):
        _diagnostic_worker.require_rs_target_artifacts(args)


def test_worker_rejects_rs_family_target_for_synthetic_cohort() -> None:
    args = argparse.Namespace(
        cohort="synthetic_conflict",
        case="S5",
        seed=117,
        max_fes=300_000,
        output_root="results/test",
        timestamp="rs-smoke",
        profile=RS_FAMILY_TARGET_PROFILE,
    )

    with pytest.raises(ValueError, match="only supports real AOB"):
        _diagnostic_worker.build_runner_args(args)


def test_rs_smoke_manifest_records_contract_and_input_hashes(tmp_path: Path) -> None:
    specs = build_specs("rs_smoke")
    for spec in specs:
        contexts: list[dict[str, str]] = []
        rows: list[dict[str, str]] = []
        for relation_index in range(4):
            context, relation_rows = _rs_rows(
                spec.problem_id,
                f"rs:{spec.problem_id}:{spec.seed}:r{relation_index}",
                seed=spec.seed,
            )
            contexts.append(context)
            rows.extend(relation_rows)
        context_path, arm_path = _trajectory_artifacts(spec, tmp_path)
        _write_csv(context_path, contexts, CONTEXT_FIELDS)
        _write_csv(arm_path, rows, ARM_RESULT_FIELDS)
        _write_csv(
            context_path.parent / f"{spec.problem_id}_aob_input_manifest.csv",
            _valid_aob_input_rows(spec),
            AOB_INPUT_MANIFEST_FIELDS,
        )

    manifest_path = aggregate_stage_artifacts(
        "rs_smoke",
        output_root=tmp_path,
        worker_count=4,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["protocol_version"] == (
        RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
    )
    assert manifest["profile"] == RS_FAMILY_TARGET_PROFILE
    assert manifest["cutoff_tie_policy"] == "structural_key"
    assert manifest["worker_count"] == 4
    assert manifest["trajectory_count"] == 4
    assert manifest["context_count"] == 16
    assert manifest["arm_result_count"] == 96
    assert manifest["integrity_gate"]["passed"] == 1
    assert manifest["integrity_gate"]["unique_relation_ids_per_trajectory"] == 1
    assert (
        manifest["integrity_gate"]["unique_action_set_hashes_per_trajectory"]
        == 1
    )
    assert (
        manifest["integrity_gate"][
            "unique_dispatch_checkpoint_hashes_per_trajectory"
        ]
        == 1
    )
    assert manifest["action_gate_passed_all_cases"] == 0
    assert manifest["primary_recommendation"] == "mechanical_smoke_only"
    assert set(manifest["case_target_mapping"]) == set(RS_SMOKE_CASES)
    assert set(manifest["arm_contract_hashes"]) == set(RS_SMOKE_CASES)
    assert set(manifest["inputs"]["trajectory_artifacts"]) == {
        spec.trajectory_id for spec in specs
    }
    assert all(
        "aob_input_manifest_sha256" in artifacts
        for artifacts in manifest["inputs"]["trajectory_artifacts"].values()
    )
    assert manifest["no_relation_context"] == {
        "R1": {"reason": "no_relation_context", "status": "not_run"},
        "S1": {"reason": "no_relation_context", "status": "not_run"},
    }

    first_spec = specs[0]
    first_context_path, _ = _trajectory_artifacts(first_spec, tmp_path)
    input_path = (
        first_context_path.parent
        / f"{first_spec.problem_id}_aob_input_manifest.csv"
    )
    valid_rows = _valid_aob_input_rows(first_spec)

    _write_csv(input_path, [], AOB_INPUT_MANIFEST_FIELDS)
    with pytest.raises(ValueError, match="must contain rows"):
        aggregate_stage_artifacts("rs_smoke", output_root=tmp_path, worker_count=4)

    changed_rows = copy.deepcopy(valid_rows)
    changed_rows[0]["unchanged"] = "0"
    _write_csv(input_path, changed_rows, AOB_INPUT_MANIFEST_FIELDS)
    with pytest.raises(ValueError, match="truth mismatch"):
        aggregate_stage_artifacts("rs_smoke", output_root=tmp_path, worker_count=4)

    changed_rows[0]["unchanged"] = "1"
    changed_rows[0]["sha256_after"] = "0" * 64
    _write_csv(input_path, changed_rows, AOB_INPUT_MANIFEST_FIELDS)
    with pytest.raises(ValueError, match="truth mismatch"):
        aggregate_stage_artifacts("rs_smoke", output_root=tmp_path, worker_count=4)

    _write_csv(input_path, valid_rows[:-1], AOB_INPUT_MANIFEST_FIELDS)
    with pytest.raises(ValueError, match="coverage mismatch"):
        aggregate_stage_artifacts("rs_smoke", output_root=tmp_path, worker_count=4)


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
