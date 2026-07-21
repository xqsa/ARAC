"""Run the R1 forced global phase-boundary Sep-CMA validation pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "hcc_smoke_runner.py"
VENDOR_ROOT = REPOSITORY_ROOT / "vendor" / "hcc"
DEFAULT_AOB_DATA_ROOT = VENDOR_ROOT / "AOB" / "AOBG" / "datafile"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "exp_027_r1_global_sep_cma"

PROTOCOL_VERSION = "r1-global-sep-cma-action-validation-v1"
CONFIG_SCHEMA_VERSION = 1
RUN_SUMMARY_PROTOCOL_VERSION = "hcc-run-summary-v3"
GLOBAL_ACTION_ARTIFACT_SCHEMA = "phase2-global-action-v1"
GLOBAL_ACTION_ARTIFACT_FILENAME = "global_phase2_action.json"
EXPERIMENT_ID = "exp_027_r1_global_sep_cma"
CASE = "R1"
FUNCTION_NAME = "rastrigin"
FUNCTION_ID = 1
ACTION = "full_space_sep_cma"
TRIGGER_SCOPE = "phase_boundary"
EXACT_MAX_FES = 3_000_000
PHASE1_COMPLETE_SWEEPS = 3
NATIVE_RESUME_SWEEPS = 3
VALIDATION_SEEDS = (117, 118, 119, 120, 121)
SUBPROCESS_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# The pilot validator must use the same immutable action/hash contracts as the
# runner.  Add the source tree explicitly so ``python path/to/run.py`` works
# even when the repository is not installed as a package.
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arac.actions.full_space_sep_cma import (  # noqa: E402
    FullSpaceSepCmaAction,
    FullSpaceSepCmaExecutionState,
    full_space_vector_hash,
)
from arac.backends.hcc_persistent_phase2 import (  # noqa: E402
    full_space_sep_cma_burst_optimizer_seed,
    full_space_sep_cma_phase_boundary_action_source_hash,
)


PHASE_BOUNDARY_CHECKPOINT_SCHEMA = (
    "full-space-sep-cma-phase-boundary-checkpoint-v1"
)


@dataclass(frozen=True)
class RunSpec:
    seed: int
    output_root: Path
    experiment_id: str = EXPERIMENT_ID

    @property
    def trajectory_id(self) -> str:
        return f"{self.experiment_id}-r1-seed{self.seed}"

    @property
    def run_directory(self) -> Path:
        return self.output_root / "runs" / CASE / f"seed_{self.seed}"

    @property
    def result_directory(self) -> Path:
        return self.run_directory / self.trajectory_id / FUNCTION_NAME


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _as_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _as_non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_non_negative(value: object, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return converted


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_finite_sequence(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty JSON list")
    converted = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in converted):
        raise ValueError(f"{field} must contain finite values")
    return converted


def _read_positive_int_sequence(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty JSON list")
    converted = tuple(_as_positive_int(item, field) for item in value)
    return converted


def _phase_boundary_checkpoint_hash_from_artifact(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    issued_sweep: int,
    target_sweep: int,
    incumbent_hash: str,
    fitness_prefix_hash: str,
    topology_hash: str,
    order_hash: str,
    action_source_hash: str,
    completed_group_deltas: tuple[float, ...],
    completed_group_actual_fes: tuple[int, ...],
    frozen_burst_budget_fes: int,
) -> str:
    """Rebuild the backend checkpoint digest from the persisted provenance.

    The full fitness prefix is intentionally not duplicated in the artifact;
    its canonical digest is persisted instead.  All remaining checkpoint
    fields are reconstructed here, so changing any claimed state invalidates
    the digest even in offline validation.
    """

    return _canonical_sha256(
        {
            "protocol": PHASE_BOUNDARY_CHECKPOINT_SCHEMA,
            "trigger_scope": TRIGGER_SCOPE,
            "problem_id": problem_id,
            "run_seed": run_seed,
            "checkpoint_fe": checkpoint_fe,
            "issued_sweep": issued_sweep,
            "target_sweep": target_sweep,
            "incumbent_hash": incumbent_hash,
            "fitness_prefix_hash": fitness_prefix_hash,
            "topology_hash": topology_hash,
            "order_hash": order_hash,
            "action_source_hash": action_source_hash,
            "completed_group_deltas": completed_group_deltas,
            "completed_group_actual_fes": completed_group_actual_fes,
            "frozen_burst_budget_fes": frozen_burst_budget_fes,
        }
    )


def _action_from_artifact_payload(
    payload: object,
    *,
    artifact_path: Path,
) -> FullSpaceSepCmaAction:
    if not isinstance(payload, dict):
        raise ValueError(f"global action payload missing: {artifact_path}")
    if payload.get("trigger_scope") != TRIGGER_SCOPE:
        raise ValueError(f"global action trigger_scope mismatch: {artifact_path}")
    try:
        action = FullSpaceSepCmaAction(
            problem_id=payload["problem_id"],
            run_seed=payload["run_seed"],
            checkpoint_fe=payload["checkpoint_fe"],
            dispatch_checkpoint_hash=payload["dispatch_checkpoint_hash"],
            trigger_relation_hash=payload["trigger_context_hash"],
            anchor_hash=payload["anchor_hash"],
            initial_mean=tuple(payload["initial_mean"]),
            initial_mean_hash=payload["initial_mean_hash"],
            initial_state_hash=payload["initial_state_hash"],
            initial_sigma=payload["initial_sigma"],
            lower_bound=payload["lower_bound"],
            upper_bound=payload["upper_bound"],
            acceptance_fitness=payload["acceptance_fitness"],
            population_size=payload["population_size"],
            budget_fes=payload["budget_fes"],
            parameterization=payload["parameterization"],
            canonical_reference_version=payload["canonical_reference_version"],
            canonical_parameters_hash=payload["canonical_parameters_hash"],
            optimizer_seed=payload["optimizer_seed"],
            seed_namespace=payload["seed_namespace"],
            restart_policy=payload["restart_policy"],
            issued_sweep=payload["issued_sweep"],
            target_sweep=payload["target_sweep"],
            ttl_sweeps=payload["ttl_sweeps"],
            expires_sweep=payload["expires_sweep"],
            trigger_scope=payload["trigger_scope"],
            acceptance_rule=payload["acceptance_rule"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"global action payload is invalid: {artifact_path}") from error
    _require(
        action.audit_payload() == payload,
        f"global action payload is not canonical: {artifact_path}",
    )
    return action


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing at exact path: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    payload = _read_json(path, "exp027 config")
    for field, expected in {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "stage": "phase2_action_validation",
    }.items():
        _require(payload.get(field) == expected, f"config {field} must be {expected!r}")

    execution = payload.get("execution")
    _require(isinstance(execution, dict), "execution config missing")
    for field, expected in {
        "cases": [CASE],
        "seeds": list(VALIDATION_SEEDS),
        "max_fes": EXACT_MAX_FES,
        "phase1_complete_sweeps": PHASE1_COMPLETE_SWEEPS,
        "native_resume_sweeps": NATIVE_RESUME_SWEEPS,
        "jobs": 5,
        "budget_accounting": "strict",
        "search_state_backend": "phase_i_mmes",
        "cmaes_restart": True,
        "mmes_restart": True,
        "skip_plots": True,
        "action": ACTION,
        "action_trigger_scope": TRIGGER_SCOPE,
    }.items():
        _require(execution.get(field) == expected, f"execution {field} must be {expected!r}")

    contract = execution.get("runner_contract")
    _require(isinstance(contract, dict), "runner_contract missing")
    for field, expected in {
        "arac_action": "native_eq8",
        "enable_relation_dispatch": False,
        "relation_policy": "global_phase2",
        "runtime_probe_repair_mode": "hard_repair",
        "evidence_overlay_mode": "off",
        "group_optimizer_mode": "full_cmaes",
    }.items():
        _require(contract.get(field) == expected, f"runner_contract {field} must be {expected!r}")

    artifact = payload.get("artifact_contract")
    _require(isinstance(artifact, dict), "artifact_contract missing")
    for field, expected in {
        "filename": GLOBAL_ACTION_ARTIFACT_FILENAME,
        "schema_version": GLOBAL_ACTION_ARTIFACT_SCHEMA,
        "run_summary_protocol_version": RUN_SUMMARY_PROTOCOL_VERSION,
    }.items():
        _require(artifact.get(field) == expected, f"artifact_contract {field} must be {expected!r}")

    analysis = payload.get("analysis")
    _require(isinstance(analysis, dict), "analysis config missing")
    for field, expected in {
        "primary_metric": "exact_3000000_fe_best_so_far_error",
        "seed_results_required": True,
        "case_summary": ["mean", "median", "sample_std", "bootstrap_mean_95_ci"],
        "bootstrap_method": "within_case_seed_bootstrap",
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 2026071901,
        "paper_comparison": "out_of_scope",
    }.items():
        _require(analysis.get(field) == expected, f"analysis {field} must be {expected!r}")
    return payload


def build_run_matrix(config: Mapping[str, object], output_root: Path) -> list[RunSpec]:
    _require(config.get("experiment_id") == EXPERIMENT_ID, "experiment id changed")
    specs = [RunSpec(seed=seed, output_root=output_root) for seed in VALIDATION_SEEDS]
    _require(len({spec.seed for spec in specs}) == 5, "exp027 requires five unique seeds")
    return specs


def build_command(
    spec: RunSpec,
    config: Mapping[str, object],
    python_executable: str,
) -> tuple[str, ...]:
    execution = config["execution"]
    assert isinstance(execution, dict)
    contract = execution["runner_contract"]
    assert isinstance(contract, dict)
    data_root = Path(str(execution.get("aob_data_root", DEFAULT_AOB_DATA_ROOT)))
    if not data_root.is_absolute():
        data_root = REPOSITORY_ROOT / data_root
    command = [
        python_executable,
        str(RUNNER_PATH),
        "--functions",
        FUNCTION_NAME,
        "--ids",
        str(FUNCTION_ID),
        "--output-root",
        str(spec.run_directory),
        "--aob-data-root",
        str(data_root.resolve()),
        "--timestamp",
        spec.trajectory_id,
        "--seed",
        str(spec.seed),
        "--max-fes",
        str(EXACT_MAX_FES),
        "--arac-action",
        str(contract["arac_action"]),
        "--budget-accounting",
        str(execution["budget_accounting"]),
        "--search-state-backend",
        str(execution["search_state_backend"]),
        "--relation-policy",
        str(contract["relation_policy"]),
        "--persistent-phase2-action",
        ACTION,
        "--evidence-overlay-mode",
        str(contract["evidence_overlay_mode"]),
        "--runtime-probe-repair-mode",
        str(contract["runtime_probe_repair_mode"]),
        "--group-optimizer-mode",
        str(contract["group_optimizer_mode"]),
    ]
    if contract["enable_relation_dispatch"] is True:
        command.append("--enable-relation-dispatch")
    if execution.get("skip_plots") is True:
        command.append("--skip-plots")
    return tuple(command)


def read_trajectory_artifacts(spec: RunSpec) -> dict[str, object]:
    summary_path = spec.result_directory / "run_summary.json"
    action_path = spec.result_directory / GLOBAL_ACTION_ARTIFACT_FILENAME
    summary = _read_json(summary_path, "runner summary")
    for field, expected in {
        "protocol_version": RUN_SUMMARY_PROTOCOL_VERSION,
        "problem_id": CASE,
        "seed": spec.seed,
        "configured_max_fes": EXACT_MAX_FES,
        "fitness_evaluations": EXACT_MAX_FES,
        "comparison_fe": EXACT_MAX_FES,
        "group_optimizer_mode": "full_cmaes",
        "global_phase2_action": ACTION,
        "global_phase2_action_artifact": GLOBAL_ACTION_ARTIFACT_FILENAME,
    }.items():
        _require(summary.get(field) == expected, f"runner summary {field} mismatch: {summary_path}")
    summary_error = _finite_non_negative(summary.get("final_error"), f"{summary_path} final_error")
    comparison_error = _finite_non_negative(
        summary.get("comparison_error"),
        f"{summary_path} comparison_error",
    )
    _require(comparison_error == summary_error, f"runner summary comparison error drift: {summary_path}")

    artifact_bytes = action_path.read_bytes() if action_path.is_file() else b""
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    _require(
        summary.get("global_phase2_action_artifact_sha256") == artifact_sha256,
        f"runner summary action artifact SHA-256 mismatch: {summary_path}",
    )
    artifact = _read_json(action_path, "global Phase2 action artifact")
    for field, expected in {
        "schema_version": GLOBAL_ACTION_ARTIFACT_SCHEMA,
        "problem_id": CASE,
        "run_seed": spec.seed,
        "configured_max_fes": EXACT_MAX_FES,
        "terminal_fe": EXACT_MAX_FES,
        "selected_action": ACTION,
        "selection_count": 1,
        "application_count": 1,
        "action_selection_rule": "forced_action_validation",
        "runtime_authorized": True,
        "runtime_consumed": True,
        "status": "completed",
        "trigger_scope": TRIGGER_SCOPE,
        "relation": None,
        "execution_mode": "one_native_sweep_burst_then_native",
        "budget_source": "previous_complete_native_sweep_actual_fes",
        "native_resumed": True,
        "native_resume_sweeps_planned": NATIVE_RESUME_SWEEPS,
        "native_resume_sweeps_completed": NATIVE_RESUME_SWEEPS,
    }.items():
        _require(artifact.get(field) == expected, f"global action artifact {field} mismatch: {action_path}")

    artifact_error = _finite_non_negative(artifact.get("final_error"), f"{action_path} final_error")
    _require(artifact_error == summary_error, f"artifact and runner final_error differ: {action_path}")
    for field in (
        "topology_hash",
        "order_hash",
        "fitness_prefix_hash",
        "action_source_hash",
        "checkpoint_hash",
        "action_hash",
        "parameter_hash",
        "lifecycle_state_hash",
        "candidate_hash",
        "post_incumbent_hash",
    ):
        _require(_is_sha256(artifact.get(field)), f"global action artifact {field} invalid: {action_path}")

    checkpoint_fe = _as_positive_int(artifact.get("checkpoint_fe"), "checkpoint_fe")
    selection_fe = _as_positive_int(artifact.get("selection_fe"), "selection_fe")
    action_start_fe = _as_positive_int(artifact.get("action_start_fe"), "action_start_fe")
    action_completed_fe = _as_positive_int(artifact.get("action_completed_fe"), "action_completed_fe")
    action_actual_fes = _as_positive_int(artifact.get("action_actual_fes"), "action_actual_fes")
    action_budget_fes = _as_positive_int(artifact.get("action_budget_fes"), "action_budget_fes")
    source_actual_fes = _as_positive_int(
        artifact.get("budget_source_actual_fes"),
        "budget_source_actual_fes",
    )
    source_sweep = _as_non_negative_int(artifact.get("budget_source_sweep"), "budget_source_sweep")
    _require(selection_fe == checkpoint_fe, f"global action must be selected at checkpoint: {action_path}")
    _require(action_start_fe == checkpoint_fe + 1, f"global action start FE mismatch: {action_path}")
    _require(action_actual_fes == action_budget_fes == source_actual_fes, f"global burst FE accounting mismatch: {action_path}")
    _require(action_completed_fe == checkpoint_fe + action_actual_fes, f"global burst completion FE mismatch: {action_path}")
    _require(action_completed_fe < EXACT_MAX_FES, f"global burst must finish before terminal FE: {action_path}")
    _require(artifact.get("native_resume_start_fe") == action_completed_fe + 1, f"native resume start FE mismatch: {action_path}")
    _require(artifact.get("post_action_native_fes") == EXACT_MAX_FES - action_completed_fe, f"post-action native FE mismatch: {action_path}")

    action = artifact.get("action")
    action_instance = _action_from_artifact_payload(
        action,
        artifact_path=action_path,
    )
    for field, expected in {
        "action": ACTION,
        "problem_id": CASE,
        "run_seed": spec.seed,
        "checkpoint_fe": checkpoint_fe,
        "budget_fes": action_budget_fes,
        "seed_namespace": ACTION,
        "trigger_scope": TRIGGER_SCOPE,
        "dispatch_checkpoint_hash": artifact["checkpoint_hash"],
        "trigger_context_hash": artifact["checkpoint_hash"],
        "acceptance_rule": "strict_improvement",
        "population_size": 24,
        "ttl_sweeps": 1,
    }.items():
        _require(action.get(field) == expected, f"global action payload {field} mismatch: {action_path}")
    for field in (
        "anchor_hash",
        "initial_mean_hash",
        "initial_state_hash",
        "canonical_parameters_hash",
    ):
        _require(_is_sha256(action.get(field)), f"global action payload {field} invalid: {action_path}")
    _require(action.get("canonical_parameters_hash") == artifact.get("parameter_hash"), f"parameter hash mismatch: {action_path}")
    initial_mean = action.get("initial_mean")
    _require(isinstance(initial_mean, list) and len(initial_mean) == 1000, f"global action initial_mean must be 1000D: {action_path}")
    _require(all(math.isfinite(float(value)) for value in initial_mean), f"global action initial_mean must be finite: {action_path}")
    _require(
        action_instance.action_hash == artifact["action_hash"],
        f"global action hash does not match canonical payload: {action_path}",
    )
    _require(
        action_instance.optimizer_seed
        == full_space_sep_cma_burst_optimizer_seed(artifact["checkpoint_hash"]),
        f"global action optimizer seed mismatch: {action_path}",
    )

    issued_sweep = _as_non_negative_int(action.get("issued_sweep"), "issued_sweep")
    target_sweep = _as_positive_int(action.get("target_sweep"), "target_sweep")
    expires_sweep = _as_positive_int(action.get("expires_sweep"), "expires_sweep")
    _require(issued_sweep == PHASE1_COMPLETE_SWEEPS - 1, f"global action issue sweep mismatch: {action_path}")
    _require(target_sweep == issued_sweep + 1 == PHASE1_COMPLETE_SWEEPS, f"global action target sweep mismatch: {action_path}")
    _require(expires_sweep == target_sweep, f"global action expiry mismatch: {action_path}")
    _require(source_sweep == issued_sweep, f"global action budget source sweep mismatch: {action_path}")
    _require(artifact.get("start_sweep") == target_sweep, f"global action start_sweep mismatch: {action_path}")

    source_deltas = _read_finite_sequence(
        artifact.get("source_group_deltas"),
        "source_group_deltas",
    )
    source_group_fes = _read_positive_int_sequence(
        artifact.get("source_group_actual_fes"),
        "source_group_actual_fes",
    )
    _require(
        len(source_deltas) == len(source_group_fes),
        f"source group deltas/FEs are not aligned: {action_path}",
    )
    _require(
        sum(source_group_fes) == action_budget_fes,
        f"source group FEs do not equal action budget: {action_path}",
    )
    topology_hash = artifact["topology_hash"]
    order_hash = artifact["order_hash"]
    fitness_prefix_hash = artifact["fitness_prefix_hash"]
    expected_source_hash = full_space_sep_cma_phase_boundary_action_source_hash(
        problem_id=CASE,
        run_seed=spec.seed,
        issued_sweep=issued_sweep,
        target_sweep=target_sweep,
        frozen_burst_budget_fes=action_budget_fes,
        topology_hash=topology_hash,
        order_hash=order_hash,
    )
    _require(
        expected_source_hash == artifact["action_source_hash"],
        f"global action source hash mismatch: {action_path}",
    )
    expected_checkpoint_hash = _phase_boundary_checkpoint_hash_from_artifact(
        problem_id=CASE,
        run_seed=spec.seed,
        checkpoint_fe=checkpoint_fe,
        issued_sweep=issued_sweep,
        target_sweep=target_sweep,
        incumbent_hash=action_instance.initial_mean_hash,
        fitness_prefix_hash=fitness_prefix_hash,
        topology_hash=topology_hash,
        order_hash=order_hash,
        action_source_hash=artifact["action_source_hash"],
        completed_group_deltas=source_deltas,
        completed_group_actual_fes=source_group_fes,
        frozen_burst_budget_fes=action_budget_fes,
    )
    _require(
        expected_checkpoint_hash == artifact["checkpoint_hash"],
        f"global checkpoint hash mismatch: {action_path}",
    )

    lifecycle = artifact.get("lifecycle")
    _require(isinstance(lifecycle, dict), f"global action lifecycle missing: {action_path}")
    for field, expected in {
        "action_hash": artifact["action_hash"],
        "state_hash": artifact["lifecycle_state_hash"],
        "status": "completed",
        "consumed_fes": action_actual_fes,
        "started_fe": action_start_fe,
        "completed_fe": action_completed_fe,
    }.items():
        _require(lifecycle.get(field) == expected, f"global action lifecycle {field} mismatch: {action_path}")
    details = lifecycle.get("details")
    _require(isinstance(details, dict), f"global action lifecycle details missing: {action_path}")
    for field, expected in {
        "action": ACTION,
        "action_hash": artifact["action_hash"],
        "initial_state_hash": action["initial_state_hash"],
        "status": "completed",
        "consumed_fes": action_actual_fes,
        "started_fe": checkpoint_fe,
        "completed_fe": action_completed_fe,
        "invalidation_reason": "",
    }.items():
        _require(
            details.get(field) == expected,
            f"global lifecycle details {field} mismatch: {action_path}",
        )
    _require(_is_sha256(details.get("final_state_hash")), f"global lifecycle final_state_hash invalid: {action_path}")
    try:
        lifecycle_state = FullSpaceSepCmaExecutionState(
            action_hash=details["action_hash"],
            initial_state_hash=details["initial_state_hash"],
            status=details["status"],
            consumed_fes=details["consumed_fes"],
            started_fe=details["started_fe"],
            completed_fe=details["completed_fe"],
            final_state_hash=details["final_state_hash"],
            invalidation_reason=details["invalidation_reason"],
        )
        lifecycle_state.validate_for(action_instance)
        canonical_lifecycle_details = lifecycle_state.audit_payload(action_instance)
        lifecycle_state_hash = lifecycle_state.state_hash(action_instance)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"global lifecycle state is invalid: {action_path}") from error
    _require(
        details == canonical_lifecycle_details,
        f"global lifecycle details are not canonical: {action_path}",
    )
    _require(
        lifecycle_state_hash == artifact["lifecycle_state_hash"] == lifecycle["state_hash"],
        f"global lifecycle state hash mismatch: {action_path}",
    )

    candidate_fitness = _finite_non_negative(
        artifact.get("candidate_fitness"),
        f"{action_path} candidate_fitness",
    )
    acceptance_fitness = _finite_non_negative(
        action.get("acceptance_fitness"),
        f"{action_path} action acceptance_fitness",
    )
    accepted = artifact.get("action_accepted")
    _require(isinstance(accepted, bool), f"global action_accepted must be boolean: {action_path}")
    _require(accepted == (candidate_fitness < acceptance_fitness), f"strict-improvement decision mismatch: {action_path}")
    candidate_values = _read_finite_sequence(
        artifact.get("candidate"),
        "candidate",
    )
    _require(
        len(candidate_values) == 1000,
        f"global action candidate must be 1000D: {action_path}",
    )
    _require(
        full_space_vector_hash(candidate_values) == artifact["candidate_hash"],
        f"global candidate hash mismatch: {action_path}",
    )
    expected_post_hash = artifact["candidate_hash"] if accepted else action_instance.initial_mean_hash
    _require(artifact.get("post_incumbent_hash") == expected_post_hash, f"post-incumbent hash mismatch: {action_path}")

    return {
        "trajectory_id": spec.trajectory_id,
        "case": CASE,
        "seed": spec.seed,
        "action": ACTION,
        "trigger_scope": TRIGGER_SCOPE,
        "final_error": summary_error,
        "fitness_evaluations": EXACT_MAX_FES,
        "action_accepted": accepted,
        "action_consumed_fes": action_actual_fes,
        "summary_path": str(summary_path),
        "action_artifact_path": str(action_path),
        "action_artifact_sha256": artifact_sha256,
        "action_hash": artifact["action_hash"],
    }


def _validate_existing_trajectory(spec: RunSpec, *, execution_source: str) -> dict[str, object]:
    started = time.perf_counter()
    try:
        result = read_trajectory_artifacts(spec)
    except (OSError, TypeError, ValueError) as error:
        return {
            "trajectory_id": spec.trajectory_id,
            "case": CASE,
            "seed": spec.seed,
            "action": ACTION,
            "ok": False,
            "status": "artifact_gate_failed",
            "execution_source": execution_source,
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(error),
        }
    result.update(
        {
            "ok": True,
            "status": "completed",
            "execution_source": execution_source,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return result


def _read_log_tail(path: Path, max_characters: int = 2000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_characters * 4))
        return handle.read().decode("utf-8", errors="replace")[-max_characters:]


def run_one(
    spec: RunSpec,
    config: Mapping[str, object],
    python_executable: str,
    *,
    run_subprocess: bool = True,
) -> dict[str, object]:
    if not run_subprocess:
        return _validate_existing_trajectory(spec, execution_source="offline_validation")

    spec.run_directory.mkdir(parents=True, exist_ok=True)
    command = build_command(spec, config, python_executable)
    runner_log_path = spec.run_directory / "runner.log"
    started = time.perf_counter()
    with runner_log_path.open("wb") as runner_log:
        completed = subprocess.run(
            command,
            cwd=VENDOR_ROOT,
            stdout=runner_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **SUBPROCESS_ENVIRONMENT},
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            "trajectory_id": spec.trajectory_id,
            "case": CASE,
            "seed": spec.seed,
            "action": ACTION,
            "ok": False,
            "status": f"runner_failed_{completed.returncode}",
            "execution_source": "fresh_execution",
            "elapsed_seconds": elapsed,
            "runner_log_path": str(runner_log_path),
            "stderr_tail": _read_log_tail(runner_log_path),
        }
    result = _validate_existing_trajectory(spec, execution_source="fresh_execution")
    result["elapsed_seconds"] = elapsed
    result["runner_log_path"] = str(runner_log_path)
    return result


def _run_one_resumable(
    spec: RunSpec,
    config: Mapping[str, object],
    python_executable: str,
) -> dict[str, object]:
    existing = _validate_existing_trajectory(
        spec,
        execution_source="reused_valid_artifact",
    )
    if existing["ok"] is True:
        return existing
    result = run_one(spec, config, python_executable)
    result["execution_source"] = "rerun_after_artifact_gate_failure"
    result["resume_gate_error"] = existing["error"]
    return result


def _print_trajectory_progress(result: Mapping[str, object]) -> None:
    suffix = (
        f" final_error={float(result['final_error']):.12e}"
        if result.get("ok") is True
        else ""
    )
    print(
        f"[R1/seed{result['seed']}] {result['status']} "
        f"source={result['execution_source']}{suffix}",
        flush=True,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "quantile requires values")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    _require(len(values) == len(VALIDATION_SEEDS), "R1 summary requires five seeds")
    rng = random.Random(seed)
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    ]
    return _quantile(means, 0.025), _quantile(means, 0.975)


def build_case_summary(
    results: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, object]:
    _require(len(results) == 5, "R1 summary requires exactly five trajectories")
    by_seed = {int(row.get("seed", -1)): row for row in results}
    _require(set(by_seed) == set(VALIDATION_SEEDS), "R1 seed set is incomplete or duplicated")
    _require(
        all(row.get("ok") is True and row.get("status") == "completed" for row in results),
        "only completed trajectories may be summarized",
    )
    values = [
        _finite_non_negative(by_seed[seed].get("final_error"), f"R1/seed{seed} final_error")
        for seed in VALIDATION_SEEDS
    ]
    analysis = config["analysis"]
    assert isinstance(analysis, dict)
    ci_low, ci_high = _bootstrap_mean_ci(
        values,
        replicates=int(analysis["bootstrap_replicates"]),
        seed=int(analysis["bootstrap_seed"]),
    )
    return {
        "case": CASE,
        "action": ACTION,
        "trigger_scope": TRIGGER_SCOPE,
        "seed_count": len(values),
        "seed_final_errors": [
            {"seed": seed, "final_error": by_seed[seed]["final_error"]}
            for seed in VALIDATION_SEEDS
        ],
        "mean_error": statistics.fmean(values),
        "median_error": statistics.median(values),
        "sample_std_error": statistics.stdev(values),
        "bootstrap_mean_95_ci": [ci_low, ci_high],
        "interpretation": (
            "Five-seed forced global phase-boundary action validation; "
            "this is not relation-selected action evidence."
        ),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python_executable: str = sys.executable,
    jobs: int | None = None,
    reuse_existing: bool = False,
    resume: bool = False,
    progress_callback: Callable[[Mapping[str, object]], None] = _print_trajectory_progress,
) -> tuple[list[dict[str, object]], dict[str, object] | None, dict[str, object]]:
    _require(not (reuse_existing and resume), "reuse_existing and resume are mutually exclusive")
    config = load_config(config_path)
    specs = build_run_matrix(config, output_root)
    execution = config["execution"]
    assert isinstance(execution, dict)
    workers = int(execution["jobs"]) if jobs is None else jobs
    _require(workers > 0, "jobs must be positive")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        if resume:
            futures = [
                pool.submit(_run_one_resumable, spec, config, python_executable)
                for spec in specs
            ]
        else:
            futures = [
                pool.submit(
                    run_one,
                    spec,
                    config,
                    python_executable,
                    run_subprocess=not reuse_existing,
                )
                for spec in specs
            ]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress_callback(result)

    results.sort(key=lambda row: int(row["seed"]))
    case_summary = (
        build_case_summary(results, config)
        if all(row.get("ok") is True for row in results)
        else None
    )
    completed_count = sum(row.get("ok") is True for row in results)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trajectory_count": len(results),
        "completed_trajectory_count": completed_count,
        "integrity_gate_passed": completed_count == 5 and case_summary is not None,
        "exact_max_fes": EXACT_MAX_FES,
        "forced_action": ACTION,
        "trigger_scope": TRIGGER_SCOPE,
        "relation_selected_action": False,
        "worker_count": workers,
        "execution_mode": (
            "offline_validation" if reuse_existing else "resume" if resume else "fresh"
        ),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "reused_trajectory_count": sum(
            row.get("execution_source") == "reused_valid_artifact" for row in results
        ),
        "executed_trajectory_count": sum(
            row.get("execution_source")
            in {"fresh_execution", "rerun_after_artifact_gate_failure"}
            for row in results
        ),
    }
    _write_json(
        output_root / "run_summary.json",
        {**manifest, "results": results, "case_summary": case_summary},
    )
    return results, case_summary, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=None)
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--reuse-existing",
        action="store_true",
        help="validate all existing artifacts without launching runner subprocesses",
    )
    execution_mode.add_argument(
        "--resume",
        action="store_true",
        help="reuse strictly valid trajectories and rerun only missing or invalid ones",
    )
    args = parser.parse_args(argv)
    if args.jobs is not None and args.jobs <= 0:
        parser.error("--jobs must be positive")
    _results, case_summary, manifest = run_experiment(
        config_path=args.config,
        output_root=args.output_root,
        python_executable=args.python_executable,
        jobs=args.jobs,
        reuse_existing=args.reuse_existing,
        resume=args.resume,
    )
    if case_summary is not None:
        for row in case_summary["seed_final_errors"]:
            print(
                f"[R1/seed{row['seed']}] final_error={float(row['final_error']):.12e}",
                flush=True,
            )
        print(
            f"[R1] n=5 mean={float(case_summary['mean_error']):.12e} "
            f"median={float(case_summary['median_error']):.12e}",
            flush=True,
        )
    print(f"Summary: {args.output_root / 'run_summary.json'}", flush=True)
    return 0 if manifest["integrity_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
