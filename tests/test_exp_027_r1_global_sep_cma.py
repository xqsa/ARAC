from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from arac.actions.full_space_sep_cma import (
    FullSpaceSepCmaAction,
    FullSpaceSepCmaExecutionState,
    full_space_sep_cma_anchor_hash,
    full_space_vector_hash,
)
from arac.backends.hcc_persistent_phase2 import (
    full_space_sep_cma_burst_optimizer_seed,
    full_space_sep_cma_phase_boundary_action_source_hash,
    full_space_sep_cma_phase_boundary_checkpoint_hash,
)
from experiments.pilots.exp_027_r1_global_sep_cma import run as exp027
from scripts import hcc_smoke_runner


def _config() -> dict[str, object]:
    return exp027.load_config(exp027.DEFAULT_CONFIG_PATH)


def _spec(tmp_path: Path, seed: int = 117) -> exp027.RunSpec:
    return exp027.RunSpec(seed=seed, output_root=tmp_path)


def _valid_command(tmp_path: Path) -> list[str]:
    config = _config()
    spec = _spec(tmp_path)
    return list(exp027.build_command(spec, config, "python"))[2:]


def _sha(fill: str) -> str:
    return fill * 64


def _write_valid_artifacts(spec: exp027.RunSpec) -> None:
    spec.result_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_fe = 5_805
    action_budget_fes = 1_935
    action_completed_fe = checkpoint_fe + action_budget_fes
    final_error = 42.0
    parameter_hash = "935292123ceeb24517dcb36cf001f10d7a0639fbc28c51f112c2d247a07526c5"
    initial_state_hash = _sha("e")
    final_state_hash = _sha("1")
    topology_hash = _sha("4")
    order_hash = _sha("5")
    issued_sweep = 2
    target_sweep = 3
    initial_mean = tuple(0.0 for _ in range(1000))
    candidate = tuple(0.1 for _ in range(1000))
    fitness_prefix = tuple(1_000.0 - index * 0.01 for index in range(checkpoint_fe))
    fitness_prefix_hash = exp027._canonical_sha256(fitness_prefix)
    source_group_deltas = tuple(0.0 for _ in range(20))
    source_group_actual_fes = (96,) * 19 + (111,)
    action_source_hash = full_space_sep_cma_phase_boundary_action_source_hash(
        problem_id=exp027.CASE,
        run_seed=spec.seed,
        issued_sweep=issued_sweep,
        target_sweep=target_sweep,
        frozen_burst_budget_fes=action_budget_fes,
        topology_hash=topology_hash,
        order_hash=order_hash,
    )
    checkpoint_hash = full_space_sep_cma_phase_boundary_checkpoint_hash(
        problem_id=exp027.CASE,
        run_seed=spec.seed,
        checkpoint_fe=checkpoint_fe,
        issued_sweep=issued_sweep,
        target_sweep=target_sweep,
        incumbent=initial_mean,
        fitness_prefix=fitness_prefix,
        topology_hash=topology_hash,
        order_hash=order_hash,
        action_source_hash=action_source_hash,
        completed_group_deltas=source_group_deltas,
        completed_group_actual_fes=source_group_actual_fes,
        frozen_burst_budget_fes=action_budget_fes,
    )
    action_instance = FullSpaceSepCmaAction(
        problem_id=exp027.CASE,
        run_seed=spec.seed,
        checkpoint_fe=checkpoint_fe,
        dispatch_checkpoint_hash=checkpoint_hash,
        trigger_relation_hash=checkpoint_hash,
        anchor_hash=full_space_sep_cma_anchor_hash(exp027.CASE, initial_mean),
        initial_mean=initial_mean,
        initial_mean_hash=full_space_vector_hash(initial_mean),
        initial_state_hash=initial_state_hash,
        initial_sigma=0.5,
        lower_bound=-100.0,
        upper_bound=100.0,
        acceptance_fitness=10.0,
        population_size=24,
        budget_fes=action_budget_fes,
        parameterization="ros_hansen_2008_pypop7",
        canonical_reference_version="pypop7-sepcmaes@67b29061d121cba9a5715897a2eb5d409df04c2d",
        canonical_parameters_hash=parameter_hash,
        optimizer_seed=full_space_sep_cma_burst_optimizer_seed(checkpoint_hash),
        seed_namespace=exp027.ACTION,
        restart_policy="none",
        issued_sweep=issued_sweep,
        target_sweep=target_sweep,
        ttl_sweeps=1,
        expires_sweep=target_sweep,
        trigger_scope=exp027.TRIGGER_SCOPE,
        acceptance_rule="strict_improvement",
    )
    action = action_instance.audit_payload()
    action_hash = action_instance.action_hash
    lifecycle_state = FullSpaceSepCmaExecutionState(
        action_hash=action_hash,
        initial_state_hash=initial_state_hash,
        status="completed",
        consumed_fes=action_budget_fes,
        started_fe=checkpoint_fe,
        completed_fe=action_completed_fe,
        final_state_hash=final_state_hash,
        invalidation_reason="",
    )
    lifecycle_state.validate_for(action_instance)
    lifecycle_details = lifecycle_state.audit_payload(action_instance)
    lifecycle_state_hash = lifecycle_state.state_hash(action_instance)
    lifecycle = {
        "action_hash": action_hash,
        "status": "completed",
        "consumed_fes": action_budget_fes,
        "started_fe": checkpoint_fe + 1,
        "completed_fe": action_completed_fe,
        "state_hash": lifecycle_state_hash,
        "details": lifecycle_details,
    }
    artifact: dict[str, Any] = {
        "schema_version": exp027.GLOBAL_ACTION_ARTIFACT_SCHEMA,
        "problem_id": exp027.CASE,
        "run_seed": spec.seed,
        "configured_max_fes": exp027.EXACT_MAX_FES,
        "terminal_fe": exp027.EXACT_MAX_FES,
        "selected_action": exp027.ACTION,
        "selection_count": 1,
        "application_count": 1,
        "action_selection_rule": "forced_action_validation",
        "runtime_authorized": True,
        "runtime_consumed": True,
        "status": "completed",
        "trigger_scope": exp027.TRIGGER_SCOPE,
        "relation": None,
        "execution_mode": "one_native_sweep_burst_then_native",
        "budget_source": "previous_complete_native_sweep_actual_fes",
        "native_resumed": True,
        "native_resume_sweeps_planned": exp027.NATIVE_RESUME_SWEEPS,
        "native_resume_sweeps_completed": exp027.NATIVE_RESUME_SWEEPS,
        "topology_hash": topology_hash,
        "order_hash": order_hash,
        "fitness_prefix_hash": fitness_prefix_hash,
        "source_group_deltas": list(source_group_deltas),
        "source_group_actual_fes": list(source_group_actual_fes),
        "action_source_hash": action_source_hash,
        "checkpoint_hash": checkpoint_hash,
        "action_hash": action_hash,
        "parameter_hash": parameter_hash,
        "lifecycle_state_hash": lifecycle_state_hash,
        "candidate": list(candidate),
        "candidate_hash": full_space_vector_hash(candidate),
        "post_incumbent_hash": full_space_vector_hash(candidate),
        "selection_fe": checkpoint_fe,
        "checkpoint_fe": checkpoint_fe,
        "action_start_fe": checkpoint_fe + 1,
        "action_completed_fe": action_completed_fe,
        "action_actual_fes": action_budget_fes,
        "action_budget_fes": action_budget_fes,
        "budget_source_sweep": 2,
        "budget_source_actual_fes": action_budget_fes,
        "action_accepted": True,
        "candidate_fitness": 5.0,
        "native_resume_start_fe": action_completed_fe + 1,
        "post_action_native_fes": exp027.EXACT_MAX_FES - action_completed_fe,
        "start_sweep": 3,
        "final_error": final_error,
        "action": action,
        "lifecycle": lifecycle,
    }
    action_path = spec.result_directory / exp027.GLOBAL_ACTION_ARTIFACT_FILENAME
    action_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_sha = hashlib.sha256(action_path.read_bytes()).hexdigest()
    summary = {
        "protocol_version": exp027.RUN_SUMMARY_PROTOCOL_VERSION,
        "problem_id": exp027.CASE,
        "seed": spec.seed,
        "configured_max_fes": exp027.EXACT_MAX_FES,
        "fitness_evaluations": exp027.EXACT_MAX_FES,
        "comparison_fe": exp027.EXACT_MAX_FES,
        "comparison_error": final_error,
        "final_error": final_error,
        "group_optimizer_mode": "full_cmaes",
        "global_phase2_action": exp027.ACTION,
        "global_phase2_action_artifact": exp027.GLOBAL_ACTION_ARTIFACT_FILENAME,
        "global_phase2_action_artifact_sha256": artifact_sha,
    }
    (spec.result_directory / "run_summary.json").write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )


def test_config_and_run_matrix_freeze_r1_global_contract(tmp_path: Path) -> None:
    config = _config()
    execution = config["execution"]
    assert isinstance(execution, dict)
    assert tuple(execution["cases"]) == ("R1",)
    assert tuple(execution["seeds"]) == exp027.VALIDATION_SEEDS
    assert execution["max_fes"] == exp027.EXACT_MAX_FES
    assert execution["jobs"] == 5
    assert execution["action_trigger_scope"] == exp027.TRIGGER_SCOPE
    assert execution["runner_contract"] == {
        "arac_action": "native_eq8",
        "enable_relation_dispatch": False,
        "relation_policy": "global_phase2",
        "runtime_probe_repair_mode": "hard_repair",
        "evidence_overlay_mode": "off",
        "group_optimizer_mode": "full_cmaes",
    }
    specs = exp027.build_run_matrix(config, tmp_path)
    assert [spec.seed for spec in specs] == list(exp027.VALIDATION_SEEDS)
    assert len({spec.trajectory_id for spec in specs}) == 5


def test_global_command_is_accepted_by_runner_parser(tmp_path: Path) -> None:
    args = hcc_smoke_runner.parse_args(_valid_command(tmp_path))
    assert args.functions == ["rastrigin"]
    assert args.ids == [1]
    assert args.relation_policy == "global_phase2"
    assert args.persistent_phase2_action == exp027.ACTION
    assert args.arac_action == "native_eq8"
    assert args.enable_relation_dispatch is False
    assert args.evidence_overlay_mode == "off"
    assert args.group_optimizer_mode == "full_cmaes"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda command: command + ["--enable-relation-dispatch"],
        lambda command: command[: command.index("off")] + ["paired_owner"] + command[command.index("off") + 1 :],
        lambda command: command[: command.index("1")] + ["2"] + command[command.index("1") + 1 :],
    ],
    ids=["relation-dispatch", "paired-owner", "non-r1"],
)
def test_global_contract_rejects_relation_or_non_r1_mixing(
    tmp_path: Path,
    mutation,
) -> None:
    with pytest.raises(SystemExit):
        hcc_smoke_runner.parse_args(mutation(_valid_command(tmp_path)))


def test_global_artifact_gate_accepts_strict_consistent_artifact(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write_valid_artifacts(spec)
    result = exp027.read_trajectory_artifacts(spec)
    assert result["seed"] == 117
    assert result["final_error"] == 42.0
    assert result["action_consumed_fes"] == 1_935
    assert result["trigger_scope"] == exp027.TRIGGER_SCOPE


@pytest.mark.parametrize(
    "drift, expected_message",
    [
        ("artifact_sha", "SHA-256"),
        ("final_error", "final_error"),
        ("fe", "FE accounting"),
        ("lifecycle", "lifecycle details status"),
    ],
)
def test_global_artifact_gate_rejects_hash_final_fe_and_lifecycle_drift(
    tmp_path: Path,
    drift: str,
    expected_message: str,
) -> None:
    spec = _spec(tmp_path)
    _write_valid_artifacts(spec)
    action_path = spec.result_directory / exp027.GLOBAL_ACTION_ARTIFACT_FILENAME
    summary_path = spec.result_directory / "run_summary.json"
    action = json.loads(action_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if drift == "artifact_sha":
        summary["global_phase2_action_artifact_sha256"] = _sha("9")
    elif drift == "final_error":
        action["final_error"] = 43.0
    elif drift == "fe":
        action["action_actual_fes"] += 1
    else:
        action["lifecycle"]["details"]["status"] = "issued"
    action_path.write_text(json.dumps(action, sort_keys=True), encoding="utf-8")
    if drift != "artifact_sha":
        summary["global_phase2_action_artifact_sha256"] = hashlib.sha256(
            action_path.read_bytes()
        ).hexdigest()
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        exp027.read_trajectory_artifacts(spec)


@pytest.mark.parametrize(
    ("drift", "expected_message"),
    [
        ("action", "action hash"),
        ("source", "source hash"),
        ("checkpoint", "checkpoint hash"),
        ("lifecycle_state", "lifecycle state hash"),
        ("candidate", "candidate hash"),
    ],
)
def test_global_artifact_gate_rejects_fabricated_internal_hashes(
    tmp_path: Path,
    drift: str,
    expected_message: str,
) -> None:
    spec = _spec(tmp_path)
    _write_valid_artifacts(spec)
    action_path = spec.result_directory / exp027.GLOBAL_ACTION_ARTIFACT_FILENAME
    summary_path = spec.result_directory / "run_summary.json"
    artifact = json.loads(action_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fabricated = _sha("9")

    if drift == "action":
        artifact["action_hash"] = fabricated
        artifact["lifecycle"]["action_hash"] = fabricated
        artifact["lifecycle"]["details"]["action_hash"] = fabricated
        state_hash = exp027._canonical_sha256(artifact["lifecycle"]["details"])
        artifact["lifecycle_state_hash"] = state_hash
        artifact["lifecycle"]["state_hash"] = state_hash
    elif drift == "source":
        artifact["action_source_hash"] = fabricated
    elif drift == "checkpoint":
        artifact["checkpoint_hash"] = fabricated
        artifact["action"]["dispatch_checkpoint_hash"] = fabricated
        artifact["action"]["trigger_context_hash"] = fabricated
        artifact["action"]["optimizer_seed"] = (
            full_space_sep_cma_burst_optimizer_seed(fabricated)
        )
        action_hash = exp027._canonical_sha256(artifact["action"])
        artifact["action_hash"] = action_hash
        artifact["lifecycle"]["action_hash"] = action_hash
        artifact["lifecycle"]["details"]["action_hash"] = action_hash
        state_hash = exp027._canonical_sha256(artifact["lifecycle"]["details"])
        artifact["lifecycle_state_hash"] = state_hash
        artifact["lifecycle"]["state_hash"] = state_hash
    elif drift == "lifecycle_state":
        artifact["lifecycle_state_hash"] = fabricated
        artifact["lifecycle"]["state_hash"] = fabricated
    else:
        artifact["candidate_hash"] = fabricated
        artifact["post_incumbent_hash"] = fabricated

    action_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    summary["global_phase2_action_artifact_sha256"] = hashlib.sha256(
        action_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        exp027.read_trajectory_artifacts(spec)
