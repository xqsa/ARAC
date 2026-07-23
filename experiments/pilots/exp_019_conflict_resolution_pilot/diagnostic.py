"""Orchestrate and aggregate the exp019 same-FE action-ceiling audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

from arac.actions.budget_reallocation import (
    FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
    BudgetAllocationAction,
    BudgetAllocationExecutionState,
)
from arac.actions.gcb import (
    CANONICAL_SEP_CMA_PARAMETERIZATION,
    CANONICAL_SEP_CMA_PARAMETERS_HASH,
    CANONICAL_SEP_CMA_POPULATION_SIZE,
    CANONICAL_SEP_CMA_REFERENCE_VERSION,
    FULL_SPACE_DIMENSION,
    GCB_ACTION,
    NO_RESTART_POLICY,
    STRICT_IMPROVEMENT_ACCEPTANCE,
    GcbAction,
    GcbExecutionState,
)
from arac.backends.hcc import required_aob_data_files
from arac.backends.hcc_action_ceiling import freeze_efficiency_budget_action
from arac.policy.action_ceiling import (
    ACTION_CEILING_FULL_MATRIX_PROFILE,
    ACTION_CEILING_ARMS,
    ACTION_CEILING_ARM_RESULT_FIELDS,
    ACTION_CEILING_CONTEXT_FIELDS,
    ACTION_CEILING_HORIZONS,
    ACTION_CEILING_PROTOCOL_VERSION,
    AUDITED_RELATION_WRITEBACK_ACTIONS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BUDGET_MAX_UNIFORM_MULTIPLIER,
    CATASTROPHIC_DELTA,
    EFFICIENCY_EWMA_ALPHA,
    GUARDED_EQ8_PROBE_FES,
    GUARDED_EQ8_WRITEBACK_ACTION,
    MATERIAL_POSITIVE_DELTA,
    PRIMARY_HORIZON,
    RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION,
    RS_FAMILY_RASTRIGIN_ARMS,
    RS_FAMILY_SCHWEFEL_ARMS,
    RS_FAMILY_TARGET_PROFILE,
    SPARSE_POSITIVE_THRESHOLD,
    STAGNATION_EPSILON,
    STAGNATION_TRIGGER_STREAK,
    WARM_START_COOLDOWN_SWEEPS,
    ActionCeilingObservation,
    action_ceiling_capture_contract,
    actionability_delta,
    relation_writeback_action_parameters,
    summarize_action_ceiling,
)
from arac.policy.evidence_overlay import UTILITY_EPSILON

from .benchmark import REPO_ROOT, VENDOR_DATA_DIR, validate_synthetic_bundle


EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIR / "diagnostic_config.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "exp_019_conflict_resolution_pilot"
CONFIG_SCHEMA_VERSION = "exp019-action-ceiling-config-v8"
AOB_INPUT_MANIFEST_FIELDS = (
    "problem_id",
    "file",
    "path",
    "sha256_before",
    "sha256_after",
    "unchanged",
)
SMOKE_CASES = ("E3", "S5")
SMOKE_SEEDS = (117, 118, 119)
SMOKE_JOBS = 6
PILOT_SEEDS = (117, 118, 119, 120, 121)
PILOT_MAX_FES = 3_000_000
SMOKE_MAX_FES = 300_000
RS_SMOKE_CASES = ("R2", "S2", "R6", "S6")
RS_VALIDATION_CASES = (
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
)
RS_NO_RELATION_CONTEXT_CASES = ("R1", "S1")
RS_SMOKE_SEEDS = (117,)
RS_VALIDATION_SEEDS = PILOT_SEEDS
RS_SMOKE_MAX_FES = 300_000
RS_VALIDATION_MAX_FES = 3_000_000
RS_SMOKE_JOBS = 4
RS_VALIDATION_JOBS = 20
REAL_CASES = ("E1", "E3", "A4", "R4", "S5")
SYNTHETIC_CASES = ("E3", "A4", "S5")
CASE_FUNCTIONS = {
    "E1": "elliptic",
    "E3": "elliptic",
    "A4": "ackley",
    "R4": "rastrigin",
    "S5": "schwefel",
    "R2": "rastrigin",
    "R3": "rastrigin",
    "R5": "rastrigin",
    "R6": "rastrigin",
    "S2": "schwefel",
    "S3": "schwefel",
    "S4": "schwefel",
    "S6": "schwefel",
}

CONTEXT_FIELDS = ACTION_CEILING_CONTEXT_FIELDS
ARM_RESULT_FIELDS = ACTION_CEILING_ARM_RESULT_FIELDS
SUMMARY_FIELDS = (
    "protocol_version",
    "cohort",
    "horizon",
    "context_count",
    "cluster_count",
    "sbs_arm",
    "sbs_mean_delta",
    "vbs_mean_delta",
    "vbs_lcb",
    "selector_mean_delta",
    "selector_lcb",
    "selector_vbs_regret",
    "selector_material_positive_count",
    "selector_material_positive_rate",
    "vbs_material_positive_rate",
    "vbs_material_positive_ucb",
    "selector_catastrophic_count",
    "selector_catastrophic_rate",
    "recommendation",
)
RS_SUMMARY_FIELDS = (
    "protocol_version",
    "profile",
    "problem_id",
    "target_arm",
    "horizon",
    "context_count",
    "cluster_count",
    "mean_delta",
    "min_delta",
    "max_delta",
    "delta_lcb",
    "delta_ucb",
    "positive_count",
    "positive_rate",
    "material_positive_count",
    "material_positive_rate",
    "catastrophic_count",
    "catastrophic_rate",
    "gate",
)
_GCB_CONTEXT_FIELDS = tuple(
    field for field in CONTEXT_FIELDS if field.startswith("gcb_")
)


@dataclass(frozen=True)
class TrajectorySpec:
    stage: str
    cohort: str
    problem_id: str
    seed: int
    max_fes: int
    action_ceiling_profile: str = ACTION_CEILING_FULL_MATRIX_PROFILE

    @property
    def function_name(self) -> str:
        return CASE_FUNCTIONS[self.problem_id]

    @property
    def trajectory_id(self) -> str:
        return (
            f"{self.stage}-{self.cohort}-{self.problem_id}-"
            f"seed{self.seed}-{self.max_fes}fe"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"CSV schema mismatch: {path}")
        return list(reader)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("protocol_version") != ACTION_CEILING_PROTOCOL_VERSION:
        raise ValueError("legacy exp019 artifacts/configs cannot be used as G1 results")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("exp019 action-ceiling config schema mismatch")
    if config.get("observer_only") is not True or config.get("runtime_authorized") is not False:
        raise ValueError("action-ceiling must remain offline and runtime-unauthorized")
    if tuple(config.get("arms", ())) != ACTION_CEILING_ARMS:
        raise ValueError("action-ceiling arm contract drifted")
    if tuple(config.get("horizons", ())) != ACTION_CEILING_HORIZONS:
        raise ValueError("action-ceiling horizon contract drifted")
    if config.get("primary_horizon") != PRIMARY_HORIZON:
        raise ValueError("action-ceiling primary horizon drifted")
    if config.get("phase1_top_relations") != 4:
        raise ValueError("action-ceiling must select four Phase1 relations")
    expected_continuation_actions = {
        "efficiency_budget_reallocation": {
            "ewma_alpha": EFFICIENCY_EWMA_ALPHA,
            "cold_start_uniform_sweeps": 1,
            "minimum_population_multiples": 1,
            "maximum_uniform_budget_multiples": BUDGET_MAX_UNIFORM_MULTIPLIER,
            "preserve_total_requested_fes": True,
        },
        "delta_priority_scan": {
            "priority": "descending_previous_sweep_delta",
            "tie_break": "original_group_index_ascending",
            "cold_start_order": "native",
        },
        "stagnation_cross_group_warm_start": {
            "relative_stagnation_epsilon": STAGNATION_EPSILON,
            "trigger_streak": STAGNATION_TRIGGER_STREAK,
            "cooldown_group_visits": WARM_START_COOLDOWN_SWEEPS,
            "perturbation_standard_deviation_sigma_multiple": 1.0,
            "perturb_unique_mean_positions_only": True,
            "preserve_shared_mean_positions": True,
        },
        GCB_ACTION: {
            "scope": "full_space",
            "dimension": FULL_SPACE_DIMENSION,
            "population_size": CANONICAL_SEP_CMA_POPULATION_SIZE,
            "parameterization": CANONICAL_SEP_CMA_PARAMETERIZATION,
            "canonical_reference_version": (
                CANONICAL_SEP_CMA_REFERENCE_VERSION
            ),
            "budget_source": "one_actual_native_sweep_horizon",
            "resume_native_after_action": True,
            "restart_policy": NO_RESTART_POLICY,
            "acceptance_rule": STRICT_IMPROVEMENT_ACCEPTANCE,
        },
    }
    if config.get("continuation_actions") != expected_continuation_actions:
        raise ValueError("action-ceiling continuation action contract drifted")
    expected_writeback_actions = {
        GUARDED_EQ8_WRITEBACK_ACTION: {
            "candidate_set": ["current", "previous", "eq8_blend"],
            "selection": "argmin_fitness",
            "tie_break": "evaluation_order",
            "probe_fes": GUARDED_EQ8_PROBE_FES,
            "probe_fes_source": "same_horizon",
        },
        "stagnation_guard_writeback": {
            "guard": "zero_delta_sum_skips_mean_fallback",
            "fallback": "keep_current_values",
        },
        "contribution_owner_writeback": {
            "winner": "larger_delta_owner",
            "tie": "abstain_keep_current",
        },
        "contribution_owner_reverse_writeback": {
            "winner": "smaller_delta_owner",
            "tie": "abstain_keep_current",
            "role": "directional_control",
        },
    }
    if config.get("writeback_actions") != expected_writeback_actions:
        raise ValueError("action-ceiling writeback action contract drifted")
    statistics = config.get("statistics", {})
    expected_statistics = {
        "epsilon": UTILITY_EPSILON,
        "material_positive_delta": MATERIAL_POSITIVE_DELTA,
        "catastrophic_delta": CATASTROPHIC_DELTA,
        "sparse_positive_threshold": SPARSE_POSITIVE_THRESHOLD,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if statistics != expected_statistics:
        raise ValueError("action-ceiling statistics contract drifted")
    expected_smoke = {
        "cohort": "real_aob",
        "cases": list(SMOKE_CASES),
        "seeds": list(SMOKE_SEEDS),
        "max_fes": SMOKE_MAX_FES,
        "jobs": SMOKE_JOBS,
    }
    if config.get("smoke") != expected_smoke:
        raise ValueError("action-ceiling smoke matrix drifted")
    pilot = config.get("pilot", {})
    if pilot.get("real_aob") != {
        "cases": list(REAL_CASES),
        "seeds": list(PILOT_SEEDS),
        "max_fes": PILOT_MAX_FES,
    }:
        raise ValueError("real AOB pilot matrix drifted")
    if pilot.get("synthetic_conflict") != {
        "cases": list(SYNTHETIC_CASES),
        "seeds": list(PILOT_SEEDS),
        "max_fes": PILOT_MAX_FES,
    }:
        raise ValueError("synthetic pilot matrix drifted")
    if pilot.get("jobs") != 12:
        raise ValueError("action-ceiling pilot jobs drifted")
    rs_validation = config.get("rs_family_target_validation", {})
    expected_target_actions = {
        case: action_ceiling_capture_contract(
            RS_FAMILY_TARGET_PROFILE,
            case,
        ).arms[1]
        for case in RS_VALIDATION_CASES
    }
    if rs_validation.get("profile") != RS_FAMILY_TARGET_PROFILE:
        raise ValueError("R/S target-action profile drifted")
    if (
        rs_validation.get("protocol_version")
        != RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
    ):
        raise ValueError("R/S target-action protocol drifted")
    if rs_validation.get("cutoff_tie_policy") != "structural_key":
        raise ValueError("R/S target-action cutoff tie policy drifted")
    if tuple(rs_validation.get("horizons", ())) != ACTION_CEILING_HORIZONS:
        raise ValueError("R/S target-action horizons drifted")
    if rs_validation.get("case_target_actions") != expected_target_actions:
        raise ValueError("R/S case target-action mapping drifted")
    if tuple(
        rs_validation.get("no_relation_context_cases", ())
    ) != RS_NO_RELATION_CONTEXT_CASES:
        raise ValueError("R/S no-relation cases drifted")
    if rs_validation.get("rs_smoke") != {
        "cohort": "real_aob",
        "cases": list(RS_SMOKE_CASES),
        "seeds": list(RS_SMOKE_SEEDS),
        "max_fes": RS_SMOKE_MAX_FES,
        "jobs": RS_SMOKE_JOBS,
    }:
        raise ValueError("R/S target-action smoke matrix drifted")
    if rs_validation.get("rs_family_validation") != {
        "cohort": "real_aob",
        "cases": list(RS_VALIDATION_CASES),
        "seeds": list(RS_VALIDATION_SEEDS),
        "max_fes": RS_VALIDATION_MAX_FES,
        "jobs": RS_VALIDATION_JOBS,
    }:
        raise ValueError("R/S target-action validation matrix drifted")
    return config


def build_specs(
    stage: str,
    *,
    cohort: str = "all",
) -> tuple[TrajectorySpec, ...]:
    load_config()
    if cohort not in {"all", "real_aob", "synthetic_conflict"}:
        raise ValueError("cohort must be all, real_aob, or synthetic_conflict")
    if stage in {"rs_smoke", "rs_family_validation"}:
        if cohort == "synthetic_conflict":
            raise ValueError("R/S target-action stages only contain real AOB trajectories")
        cases, seeds, max_fes = (
            (RS_SMOKE_CASES, RS_SMOKE_SEEDS, RS_SMOKE_MAX_FES)
            if stage == "rs_smoke"
            else (
                RS_VALIDATION_CASES,
                RS_VALIDATION_SEEDS,
                RS_VALIDATION_MAX_FES,
            )
        )
        return tuple(
            TrajectorySpec(
                stage,
                "real_aob",
                case,
                seed,
                max_fes,
                RS_FAMILY_TARGET_PROFILE,
            )
            for case in cases
            for seed in seeds
        )
    if stage == "smoke":
        if cohort == "synthetic_conflict":
            raise ValueError("the smoke stage only contains real AOB trajectories")
        return tuple(
            TrajectorySpec("smoke", "real_aob", case, seed, SMOKE_MAX_FES)
            for case in SMOKE_CASES
            for seed in SMOKE_SEEDS
        )
    if stage != "pilot":
        raise ValueError(
            "stage must be smoke, pilot, rs_smoke, or rs_family_validation"
        )
    specs: list[TrajectorySpec] = []
    if cohort in {"all", "real_aob"}:
        specs.extend(
            TrajectorySpec("pilot", "real_aob", case, seed, PILOT_MAX_FES)
            for case in REAL_CASES
            for seed in PILOT_SEEDS
        )
    if cohort in {"all", "synthetic_conflict"}:
        specs.extend(
            TrajectorySpec("pilot", "synthetic_conflict", case, seed, PILOT_MAX_FES)
            for case in SYNTHETIC_CASES
            for seed in PILOT_SEEDS
        )
    return tuple(specs)


def validate_raw_rows(
    context_rows: Sequence[Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, str]],
) -> tuple[ActionCeilingObservation, ...]:
    contexts: dict[str, Mapping[str, str]] = {}
    gcb_actions: dict[str, GcbAction] = {}
    for row in context_rows:
        if row.get("protocol_version") != ACTION_CEILING_PROTOCOL_VERSION:
            raise ValueError("legacy action-ceiling context row")
        if row.get("runtime_authorized") != "0":
            raise ValueError("action-ceiling context cannot authorize runtime")
        if (
            row.get("status") != "complete"
            or row.get("invalidation_reason")
            or row.get("native_parity") != "1"
        ):
            raise ValueError("action-ceiling context did not pass native parity")
        context_id = str(row.get("context_id", ""))
        if not context_id or context_id in contexts:
            raise ValueError("action-ceiling context identity is missing or duplicated")
        if not _is_sha256(row.get("dispatch_anchor_hash")):
            raise ValueError("action-ceiling dispatch anchor hash is invalid")
        selector_arm = row.get("selector_arm")
        if selector_arm not in ACTION_CEILING_ARMS:
            raise ValueError("context selector arm is invalid")
        efficiency = json.loads(str(row.get("efficiency_ewma", "")))
        streaks = json.loads(str(row.get("stagnation_streaks", "")))
        populations = json.loads(str(row.get("population_sizes", "")))
        uniform_budgets = json.loads(str(row.get("uniform_group_budgets", "")))
        if (
            not isinstance(efficiency, list)
            or not efficiency
            or len(efficiency) != len(streaks)
            or len(efficiency) != len(populations)
            or len(efficiency) != len(uniform_budgets)
            or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in efficiency)
            or any(
                isinstance(value, bool) or int(value) < 0 or int(value) != value
                for value in streaks
            )
            or any(
                isinstance(value, bool) or int(value) <= 0 or int(value) != value
                for value in populations
            )
            or any(
                isinstance(value, bool) or int(value) <= 0 or int(value) != value
                for value in uniform_budgets
            )
            or any(
                int(budget) < int(population)
                for budget, population in zip(
                    uniform_budgets,
                    populations,
                    strict=True,
                )
            )
            or int(row.get("completed_efficiency_sweeps", "-1")) < 0
        ):
            raise ValueError("action-ceiling continuation context is invalid")
        horizon_fe = int(row.get("horizon_fe", "0"))
        acceptance_fitness = float(
            row.get("gcb_acceptance_fitness", "nan")
        )
        try:
            action_payload = json.loads(
                str(row.get("gcb_action_payload", ""))
            )
            if not isinstance(action_payload, dict):
                raise ValueError("action payload must be an object")
            action_fields = dict(action_payload)
            if action_fields.pop("action", None) != GCB_ACTION:
                raise ValueError("action payload name is invalid")
            gcb_action = GcbAction(**action_fields)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "GCB action payload is invalid"
            ) from error
        if (
            horizon_fe < CANONICAL_SEP_CMA_POPULATION_SIZE
            or int(row.get("gcb_budget_fes", "-1")) != horizon_fe
            or int(row.get("gcb_population_size", "-1"))
            != CANONICAL_SEP_CMA_POPULATION_SIZE
            or int(row.get("gcb_optimizer_seed", "-1")) < 0
            or not _is_sha256(row.get("gcb_action_hash"))
            or not _is_sha256(row.get("gcb_initial_mean_hash"))
            or not _is_sha256(row.get("gcb_parameter_hash"))
            or row.get("gcb_parameter_hash")
            != CANONICAL_SEP_CMA_PARAMETERS_HASH
            or not math.isfinite(acceptance_fitness)
            or _canonical_payload_hash(action_payload)
            != row.get("gcb_action_hash")
            or gcb_action.action_hash
            != row.get("gcb_action_hash")
            or gcb_action.problem_id != row.get("problem_id")
            or gcb_action.run_seed != int(row.get("seed", "-1"))
            or gcb_action.checkpoint_fe
            != int(row.get("dispatch_fe", "-1"))
            or gcb_action.dispatch_checkpoint_hash
            != row.get("dispatch_checkpoint_hash")
            or gcb_action.initial_mean_hash
            != row.get("gcb_initial_mean_hash")
            or gcb_action.canonical_parameters_hash
            != row.get("gcb_parameter_hash")
            or gcb_action.optimizer_seed
            != int(row.get("gcb_optimizer_seed", "-1"))
            or gcb_action.population_size
            != int(row.get("gcb_population_size", "-1"))
            or gcb_action.budget_fes
            != int(row.get("gcb_budget_fes", "-1"))
            or gcb_action.acceptance_fitness != acceptance_fitness
            or gcb_action.issued_sweep
            != int(row.get("issued_sweep", "-1"))
            or gcb_action.target_sweep
            != int(row.get("target_sweep", "-1"))
        ):
            raise ValueError("GCB context contract is invalid")
        contexts[context_id] = row
        gcb_actions[context_id] = gcb_action

    by_context: dict[str, dict[tuple[str, str], Mapping[str, str]]] = {}
    for row in arm_rows:
        if row.get("protocol_version") != ACTION_CEILING_PROTOCOL_VERSION:
            raise ValueError("legacy action-ceiling arm row")
        if row.get("runtime_authorized") != "0":
            raise ValueError("action-ceiling arm cannot authorize runtime")
        if row.get("status") != "complete" or row.get("invalidation_reason"):
            raise ValueError("action-ceiling arm result is incomplete")
        if row.get("counterfactual_applied") not in {"0", "1"}:
            raise ValueError("counterfactual_applied must be a binary truth value")
        if row.get("continuation_policy_applied") not in {"0", "1"}:
            raise ValueError("continuation policy flag must be binary")
        if row.get("action_accepted") not in {"0", "1"}:
            raise ValueError("action_accepted must be a binary truth value")
        action_budget_fes = int(row.get("action_budget_fes", "-1"))
        action_actual_fes = int(row.get("action_actual_fes", "-1"))
        optimizer_population_size = int(
            row.get("optimizer_population_size", "-1")
        )
        optimizer_generation_count = int(
            row.get("optimizer_generation_count", "-1")
        )
        if (
            action_budget_fes < 0
            or action_actual_fes < 0
            or action_actual_fes > action_budget_fes
            or optimizer_population_size < 0
            or optimizer_generation_count < 0
        ):
            raise ValueError("action optimizer accounting is invalid")
        mutation_norm = float(row.get("mutation_norm", "nan"))
        mean_mutation_norm = float(row.get("optimizer_mean_mutation_norm", "nan"))
        warm_start_norm = float(row.get("warm_start_mean_shift_norm", "nan"))
        warm_start_count = int(row.get("warm_start_trigger_count", "-1"))
        if (
            not math.isfinite(mutation_norm)
            or not math.isfinite(mean_mutation_norm)
            or not math.isfinite(warm_start_norm)
            or mutation_norm < 0.0
            or mean_mutation_norm < 0.0
            or warm_start_norm < 0.0
            or warm_start_count < 0
        ):
            raise ValueError("action-ceiling mutation norms must be finite and non-negative")
        if (warm_start_count == 0) != (warm_start_norm == 0.0):
            raise ValueError("warm-start count and mutation norm disagree")
        arm = str(row.get("arm", ""))
        horizon = str(row.get("horizon", ""))
        sweep_trace = json.loads(str(row.get("execution_sweep_trace", "")))
        order_trace = json.loads(str(row.get("execution_order_trace", "")))
        budget_trace = json.loads(str(row.get("group_budget_trace", "")))
        start_fe_trace = json.loads(
            str(row.get("execution_start_fe_trace", ""))
        )
        empty_action_prefix = (
            (
                arm == GCB_ACTION
                and horizon in {"immediate", PRIMARY_HORIZON}
            )
            or (arm == GUARDED_EQ8_WRITEBACK_ACTION and horizon == "immediate")
        )
        if (
            not isinstance(sweep_trace, list)
            or not isinstance(order_trace, list)
            or (not order_trace and not empty_action_prefix)
            or not isinstance(budget_trace, list)
            or not isinstance(start_fe_trace, list)
            or len(sweep_trace) != len(order_trace)
            or len(order_trace) != len(budget_trace)
            or len(order_trace) != len(start_fe_trace)
            or any(
                isinstance(value, bool)
                or int(value) != value
                or int(value) < 0
                for value in sweep_trace
            )
            or any(
                isinstance(value, bool)
                or int(value) != value
                or int(value) < 0
                for value in order_trace
            )
            or any(
                isinstance(value, bool)
                or int(value) != value
                or int(value) <= 0
                for value in budget_trace
            )
            or any(
                isinstance(value, bool)
                or int(value) != value
                or int(value) <= 0
                for value in start_fe_trace
            )
        ):
            raise ValueError("action-ceiling continuation trace is invalid")
        expected_applied = str(
            int(
                mutation_norm > 0.0
                or mean_mutation_norm > 0.0
                or action_actual_fes > 0
                or row["continuation_policy_applied"] == "1"
            )
        )
        if row["counterfactual_applied"] != expected_applied:
            raise ValueError("counterfactual_applied disagrees with branch mutation")
        if not row.get("selected_candidate"):
            raise ValueError("action-ceiling selected candidate is missing")
        context_id = str(row.get("context_id", ""))
        if context_id not in contexts:
            raise ValueError("arm result references an unknown context")
        context = contexts[context_id]
        populations = tuple(
            int(value) for value in json.loads(context["population_sizes"])
        )
        uniform_budgets = tuple(
            int(value) for value in json.loads(context["uniform_group_budgets"])
        )
        group_count = len(populations)
        horizon_fe = int(context["horizon_fe"])
        horizon_targets = {
            "immediate": 1,
            PRIMARY_HORIZON: horizon_fe,
            "sweep_3": 3 * horizon_fe,
        }
        if horizon not in horizon_targets:
            raise ValueError("action-ceiling horizon is invalid")
        target_relative_fe = horizon_targets[horizon]
        dispatch_fe = int(context["dispatch_fe"])
        if (
            int(row.get("target_fe", "-1"))
            != dispatch_fe + target_relative_fe
            or int(row.get("natural_endpoint_fe", "-1"))
            < dispatch_fe + 3 * horizon_fe
            or start_fe_trace != sorted(set(start_fe_trace))
            or any(int(value) > target_relative_fe for value in start_fe_trace)
        ):
            raise ValueError("action-ceiling horizon FE trace is invalid")
        if arm == GCB_ACTION:
            gcb_action = gcb_actions[context_id]
            try:
                lifecycle_payload = json.loads(
                    str(row.get("action_lifecycle_payload", ""))
                )
                if not isinstance(lifecycle_payload, dict):
                    raise ValueError("lifecycle payload must be an object")
                lifecycle_fields = dict(lifecycle_payload)
                if lifecycle_fields.pop("action", None) != GCB_ACTION:
                    raise ValueError("lifecycle action name is invalid")
                execution_state = GcbExecutionState(
                    **lifecycle_fields
                )
                execution_state.validate_for(gcb_action)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "GCB lifecycle payload is invalid"
                ) from error
            candidate_fitness = float(
                row.get("action_candidate_fitness", "nan")
            )
            expected_accepted = str(
                int(
                    candidate_fitness
                    < float(context["gcb_acceptance_fitness"])
                )
            )
            expected_post_hash = (
                row.get("action_candidate_hash")
                if expected_accepted == "1"
                else context.get("gcb_initial_mean_hash")
            )
            if (
                action_budget_fes != horizon_fe
                or action_actual_fes != horizon_fe
                or row.get("action_instance_hash")
                != context.get("gcb_action_hash")
                or not _is_sha256(row.get("action_lifecycle_hash"))
                or _canonical_payload_hash(lifecycle_payload)
                != row.get("action_lifecycle_hash")
                or execution_state.state_hash(gcb_action)
                != row.get("action_lifecycle_hash")
                or execution_state.status != "completed"
                or execution_state.consumed_fes != action_actual_fes
                or execution_state.started_fe != dispatch_fe
                or execution_state.completed_fe != dispatch_fe + horizon_fe
                or execution_state.final_state_hash
                != row.get("optimizer_final_state_hash")
                or row.get("optimizer_parameter_hash")
                != context.get("gcb_parameter_hash")
                or not _is_sha256(row.get("action_candidate_hash"))
                or not math.isfinite(candidate_fitness)
                or row.get("action_accepted") != expected_accepted
                or row.get("action_post_incumbent_hash")
                != expected_post_hash
                or not _is_sha256(row.get("optimizer_initial_state_hash"))
                or row.get("optimizer_initial_state_hash")
                != gcb_action.initial_state_hash
                or not _is_sha256(row.get("optimizer_final_state_hash"))
                or row.get("optimizer_scope") != "full_space"
                or optimizer_population_size
                != CANONICAL_SEP_CMA_POPULATION_SIZE
                or optimizer_generation_count
                != horizon_fe // CANONICAL_SEP_CMA_POPULATION_SIZE
                or row.get("continuation_policy_applied") != "1"
                or any(int(value) <= horizon_fe for value in start_fe_trace)
                or (
                    horizon == "sweep_3"
                    and int(start_fe_trace[0]) != horizon_fe + 1
                )
            ):
                raise ValueError("GCB arm contract is invalid")
        elif arm in AUDITED_RELATION_WRITEBACK_ACTIONS:
            try:
                writeback_payload = json.loads(
                    str(row.get("action_lifecycle_payload", ""))
                )
                if not isinstance(writeback_payload, dict):
                    raise ValueError("writeback payload must be an object")
                instance_payload = writeback_payload.get("instance")
                if not isinstance(instance_payload, dict):
                    raise ValueError("writeback instance must be an object")
                candidates = instance_payload.get("candidates")
                if not isinstance(candidates, list) or not all(
                    isinstance(candidate, dict) for candidate in candidates
                ):
                    raise ValueError("writeback candidates must be objects")
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "relation writeback lifecycle payload is invalid"
                ) from error

            parameters = relation_writeback_action_parameters(arm)
            expected_budget = int(parameters["probe_fes"])
            selected_candidate = str(row.get("selected_candidate", ""))
            expected_accepted = str(int(selected_candidate != "current"))
            expected_candidate_names = {
                GUARDED_EQ8_WRITEBACK_ACTION: ["current", "previous", "eq8_blend"],
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
            candidate_names = [candidate.get("name") for candidate in candidates]
            candidate_hashes = {
                str(candidate.get("name")): candidate.get("values_hash")
                for candidate in candidates
            }
            relation_payload = instance_payload.get("relation")
            if not isinstance(relation_payload, dict):
                raise ValueError("relation writeback instance has no relation")
            owners = relation_payload.get("owners")
            shared = relation_payload.get("shared")
            if not isinstance(owners, list) or not isinstance(shared, list):
                raise ValueError("relation writeback relation is invalid")
            instance_relation_id = "g{}:v{}".format(
                "-".join(str(value) for value in owners),
                "-".join(str(value) for value in shared),
            )
            if (
                action_budget_fes != expected_budget
                or action_actual_fes != expected_budget
                or _canonical_payload_hash(writeback_payload)
                != row.get("action_lifecycle_hash")
                or _canonical_payload_hash(instance_payload)
                != row.get("action_instance_hash")
                or writeback_payload.get("instance_hash")
                != row.get("action_instance_hash")
                or instance_payload.get("arm") != arm
                or instance_payload.get("context_hash")
                != context.get("dispatch_checkpoint_hash")
                or instance_payload.get("action_set_hash")
                != context.get("action_set_hash")
                or instance_relation_id != context.get("relation_id")
                or instance_payload.get("dispatch_anchor_hash")
                != context.get("dispatch_anchor_hash")
                or not _is_sha256(instance_payload.get("previous_values_hash"))
                or not _is_sha256(instance_payload.get("current_values_hash"))
                or not math.isfinite(float(instance_payload.get("previous_delta", "nan")))
                or not math.isfinite(float(instance_payload.get("current_delta", "nan")))
                or instance_payload.get("parameters") != parameters
                or _canonical_payload_hash(parameters)
                != instance_payload.get("parameter_hash")
                or int(instance_payload.get("action_budget_fes", -1))
                != expected_budget
                or candidate_names != expected_candidate_names
                or any(not _is_sha256(value) for value in candidate_hashes.values())
                or selected_candidate not in expected_candidate_names
                or candidate_hashes.get(selected_candidate)
                != row.get("action_candidate_hash")
                or writeback_payload.get("selected_candidate") != selected_candidate
                or writeback_payload.get("selected_values_hash")
                != row.get("action_candidate_hash")
                or writeback_payload.get("post_incumbent_hash")
                != row.get("action_post_incumbent_hash")
                or int(writeback_payload.get("action_actual_fes", -1))
                != expected_budget
                or not isinstance(writeback_payload.get("accepted"), bool)
                or str(int(bool(writeback_payload.get("accepted"))))
                != expected_accepted
                or row.get("action_accepted") != expected_accepted
                or not _is_sha256(row.get("action_candidate_hash"))
                or not _is_sha256(row.get("action_post_incumbent_hash"))
                or row.get("optimizer_parameter_hash")
                or row.get("optimizer_initial_state_hash")
                or row.get("optimizer_final_state_hash")
                or optimizer_population_size != 0
                or optimizer_generation_count != 0
                or row.get("optimizer_scope") != "relation_writeback"
                or (start_fe_trace and int(start_fe_trace[0]) != 1 + expected_budget)
            ):
                raise ValueError("relation writeback arm contract is invalid")

            if arm == GUARDED_EQ8_WRITEBACK_ACTION:
                try:
                    candidate_fitness = float(row.get("action_candidate_fitness", "nan"))
                    selected_fitness = float(
                        writeback_payload.get("selected_fitness", "nan")
                    )
                    probe_outcomes = writeback_payload.get("probe_outcomes")
                    if not isinstance(probe_outcomes, list) or len(probe_outcomes) != 3:
                        raise ValueError("guarded probe outcomes are incomplete")
                    outcome_names = [outcome["name"] for outcome in probe_outcomes]
                    outcome_fitness = [float(outcome["fitness"]) for outcome in probe_outcomes]
                    outcome_hashes = [outcome["values_hash"] for outcome in probe_outcomes]
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("guarded probe outcomes are invalid") from error
                winner = min(range(len(outcome_fitness)), key=outcome_fitness.__getitem__)
                if (
                    outcome_names != expected_candidate_names
                    or outcome_hashes
                    != [candidate_hashes[name] for name in expected_candidate_names]
                    or not all(math.isfinite(value) for value in outcome_fitness)
                    or not math.isfinite(candidate_fitness)
                    or candidate_fitness != selected_fitness
                    or candidate_fitness != outcome_fitness[winner]
                    or selected_candidate != outcome_names[winner]
                ):
                    raise ValueError("guarded probe winner is invalid")
            elif (
                row.get("action_candidate_fitness")
                or "selected_fitness" in writeback_payload
                or "probe_outcomes" in writeback_payload
            ):
                raise ValueError("zero-FE relation writeback contains probe outcomes")
        elif (
            action_budget_fes != 0
            or action_actual_fes != 0
            or row.get("action_instance_hash")
            or row.get("action_lifecycle_payload")
            or row.get("action_lifecycle_hash")
            or row.get("action_accepted") != "0"
            or row.get("action_candidate_hash")
            or row.get("action_candidate_fitness")
            or row.get("action_post_incumbent_hash")
            or row.get("optimizer_parameter_hash")
            or row.get("optimizer_initial_state_hash")
            or row.get("optimizer_final_state_hash")
            or optimizer_population_size != 0
            or optimizer_generation_count != 0
            or row.get("optimizer_scope")
            not in {"relation_writeback", "decomposed_groups"}
            or int(start_fe_trace[0]) != 1
        ):
            raise ValueError("non-Sep arm contains full-space optimizer state")
        for group, budget in zip(order_trace, budget_trace, strict=True):
            group_index = int(group)
            group_budget = int(budget)
            if not 0 <= group_index < group_count:
                raise ValueError("continuation trace group index is invalid")
            if arm == "efficiency_budget_reallocation":
                if not (
                    populations[group_index]
                    <= group_budget
                    <= 3 * uniform_budgets[group_index]
                ):
                    raise ValueError("adaptive group budget violates its frozen bounds")
            elif group_budget != uniform_budgets[group_index]:
                raise ValueError("non-budget arm changed a group budget")
        for sweep in set(int(value) for value in sweep_trace):
            positions = [
                index
                for index, value in enumerate(sweep_trace)
                if int(value) == sweep
            ]
            groups = [int(order_trace[index]) for index in positions]
            if len(set(groups)) != len(groups):
                raise ValueError("continuation sweep dispatched a group more than once")
            if len(groups) == group_count:
                if set(groups) != set(range(group_count)):
                    raise ValueError("complete continuation sweep lost a group")
                if arm == "efficiency_budget_reallocation" and sum(
                    int(budget_trace[index]) for index in positions
                ) != sum(uniform_budgets):
                    raise ValueError("adaptive sweep did not preserve total requested FEs")
        key = (str(row.get("arm", "")), str(row.get("horizon", "")))
        if key in by_context.setdefault(context_id, {}):
            raise ValueError("duplicate context arm horizon row")
        by_context[context_id][key] = row

    observations: list[ActionCeilingObservation] = []
    expected = {(arm, horizon) for arm in ACTION_CEILING_ARMS for horizon in ACTION_CEILING_HORIZONS}
    for context_id, context in contexts.items():
        result_rows = by_context.get(context_id, {})
        if set(result_rows) != expected:
            raise ValueError("context does not contain every frozen arm and horizon")
        sep_rows = [
            result_rows[(GCB_ACTION, horizon)]
            for horizon in ACTION_CEILING_HORIZONS
        ]
        invariant_fields = (
            "action_candidate_fitness",
            "action_candidate_hash",
            "action_accepted",
            "action_post_incumbent_hash",
            "optimizer_final_state_hash",
            "action_lifecycle_payload",
            "action_lifecycle_hash",
        )
        first_sep_row = sep_rows[0]
        if any(
            row.get(field) != first_sep_row.get(field)
            for row in sep_rows[1:]
            for field in invariant_fields
        ):
            raise ValueError(
                "GCB action outcome differs across horizons"
            )
        for arm in AUDITED_RELATION_WRITEBACK_ACTIONS:
            writeback_rows = [
                result_rows[(arm, horizon)] for horizon in ACTION_CEILING_HORIZONS
            ]
            first_writeback_row = writeback_rows[0]
            if any(
                row.get(field) != first_writeback_row.get(field)
                for row in writeback_rows[1:]
                for field in invariant_fields
            ):
                raise ValueError(
                    "relation writeback action outcome differs across horizons"
                )
        for horizon in ACTION_CEILING_HORIZONS:
            native = result_rows[("native_eq8", horizon)]
            native_error = float(native["native_error"])
            if float(native["arm_error"]) != native_error or float(native["delta"]) != 0.0:
                raise ValueError("native arm must be its own zero-delta reference")
            for arm in ACTION_CEILING_ARMS:
                row = result_rows[(arm, horizon)]
                if float(row["native_error"]) != native_error:
                    raise ValueError("paired arms do not share the native reference")
                delta = actionability_delta(native_error, float(row["arm_error"]))
                if abs(delta - float(row["delta"])) > 1e-12:
                    raise ValueError("action-ceiling delta does not match raw errors")
                observations.append(
                    ActionCeilingObservation(
                        context_id=context_id,
                        cohort=str(context["cohort"]),
                        problem_id=str(context["problem_id"]),
                        seed=int(context["seed"]),
                        arm=arm,
                        horizon=horizon,
                        delta=delta,
                        selector_arm=str(context["selector_arm"]),
                    )
                )
    return tuple(observations)


def _rs_context_state(
    row: Mapping[str, str],
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    try:
        efficiency = json.loads(str(row.get("efficiency_ewma", "")))
        streaks = json.loads(str(row.get("stagnation_streaks", "")))
        raw_populations = json.loads(str(row.get("population_sizes", "")))
        raw_uniform_budgets = json.loads(
            str(row.get("uniform_group_budgets", ""))
        )
        populations = tuple(int(value) for value in raw_populations)
        uniform_budgets = tuple(int(value) for value in raw_uniform_budgets)
        shared_values = tuple(
            tuple(float(value) for value in json.loads(str(row.get(field, ""))))
            for field in ("anchor_values", "left_values", "right_values", "bridge_values")
        )
        bridge_weights = json.loads(str(row.get("bridge_weights", "")))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("R/S target-action context payload is invalid") from error
    if (
        not isinstance(efficiency, list)
        or not efficiency
        or not isinstance(streaks, list)
        or not isinstance(raw_populations, list)
        or not isinstance(raw_uniform_budgets, list)
        or len(efficiency) != len(streaks)
        or len(efficiency) != len(populations)
        or len(efficiency) != len(uniform_budgets)
        or any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in efficiency
        )
        or any(
            isinstance(value, bool) or int(value) < 0 or int(value) != value
            for value in streaks
        )
        or any(
            isinstance(value, bool) or int(value) != value or int(value) <= 0
            for value in raw_populations
        )
        or any(
            isinstance(value, bool) or int(value) != value or int(value) <= 0
            for value in raw_uniform_budgets
        )
        or any(
            budget < population
            for budget, population in zip(
                uniform_budgets,
                populations,
                strict=True,
            )
        )
        or not shared_values[0]
        or any(len(values) != len(shared_values[0]) for values in shared_values)
        or any(
            not math.isfinite(value)
            for values in shared_values
            for value in values
        )
        or not isinstance(bridge_weights, dict)
        or set(bridge_weights) != {"left_owner", "right_owner"}
        or any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in bridge_weights.values()
        )
        or not math.isclose(
            math.fsum(float(value) for value in bridge_weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("R/S target-action context state is invalid")
    horizon_fe = int(row.get("horizon_fe", "0"))
    if (
        horizon_fe <= 0
        or int(row.get("phase_boundary_fe", "-1")) < 0
        or int(row.get("dispatch_fe", "-1"))
        < int(row.get("phase_boundary_fe", "-1"))
        or int(row.get("issued_sweep", "-1")) < 0
        or int(row.get("target_sweep", "-1"))
        != int(row.get("issued_sweep", "-1")) + 1
        or not 0 <= int(row.get("group_index", "-1")) < len(populations)
        or int(row.get("completed_efficiency_sweeps", "-1")) < 0
    ):
        raise ValueError("R/S target-action context checkpoint is invalid")
    return populations, uniform_budgets, horizon_fe


def _rs_gcb_action(row: Mapping[str, str]) -> GcbAction:
    try:
        payload = json.loads(str(row.get("gcb_action_payload", "")))
        if not isinstance(payload, dict):
            raise ValueError("action payload must be an object")
        action_fields = dict(payload)
        if action_fields.pop("action", None) != GCB_ACTION:
            raise ValueError("action payload name is invalid")
        action = GcbAction(**action_fields)
        acceptance_fitness = float(row.get("gcb_acceptance_fitness", "nan"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("R/S GCB context action is invalid") from error
    if (
        int(row.get("horizon_fe", "0")) < CANONICAL_SEP_CMA_POPULATION_SIZE
        or int(row.get("gcb_budget_fes", "-1"))
        != int(row.get("horizon_fe", "0"))
        or int(row.get("gcb_population_size", "-1"))
        != CANONICAL_SEP_CMA_POPULATION_SIZE
        or int(row.get("gcb_optimizer_seed", "-1")) < 0
        or not math.isfinite(acceptance_fitness)
        or not all(
            _is_sha256(row.get(field))
            for field in (
                "gcb_action_hash",
                "gcb_initial_mean_hash",
                "gcb_parameter_hash",
            )
        )
        or row.get("gcb_parameter_hash")
        != CANONICAL_SEP_CMA_PARAMETERS_HASH
        or _canonical_payload_hash(payload) != row.get("gcb_action_hash")
        or action.action_hash != row.get("gcb_action_hash")
        or action.problem_id != row.get("problem_id")
        or action.run_seed != int(row.get("seed", "-1"))
        or action.checkpoint_fe != int(row.get("dispatch_fe", "-1"))
        or action.dispatch_checkpoint_hash != row.get("dispatch_checkpoint_hash")
        or action.initial_mean_hash != row.get("gcb_initial_mean_hash")
        or action.canonical_parameters_hash != row.get("gcb_parameter_hash")
        or action.optimizer_seed != int(row.get("gcb_optimizer_seed", "-1"))
        or action.population_size
        != int(row.get("gcb_population_size", "-1"))
        or action.budget_fes != int(row.get("gcb_budget_fes", "-1"))
        or action.acceptance_fitness != acceptance_fitness
        or action.issued_sweep != int(row.get("issued_sweep", "-1"))
        or action.target_sweep != int(row.get("target_sweep", "-1"))
    ):
        raise ValueError("R/S GCB context contract is invalid")
    return action


def _rs_budget_action(row: Mapping[str, str]) -> BudgetAllocationAction:
    try:
        action = freeze_efficiency_budget_action(
            problem_id=str(row.get("problem_id", "")),
            run_seed=int(row.get("seed", "-1")),
            checkpoint_fe=int(row.get("dispatch_fe", "-1")),
            dispatch_checkpoint_hash=str(
                row.get("dispatch_checkpoint_hash", "")
            ),
            source_efficiency_ewma=tuple(
                float(value)
                for value in json.loads(str(row.get("efficiency_ewma", "")))
            ),
            population_sizes=tuple(
                int(value)
                for value in json.loads(str(row.get("population_sizes", "")))
            ),
            uniform_group_budgets=tuple(
                int(value)
                for value in json.loads(
                    str(row.get("uniform_group_budgets", ""))
                )
            ),
            issued_sweep=int(row.get("target_sweep", "-1")),
            target_sweep=int(row.get("target_sweep", "-1")) + 1,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("R/S frozen budget action is invalid") from error
    return action


def _rs_arm_traces(
    row: Mapping[str, str],
    *,
    allow_empty: bool,
) -> tuple[list[int], list[int], list[int], list[int]]:
    try:
        traces = tuple(
            json.loads(str(row.get(field, "")))
            for field in (
                "execution_sweep_trace",
                "execution_order_trace",
                "group_budget_trace",
                "execution_start_fe_trace",
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError("R/S target-action continuation trace is invalid") from error
    if (
        any(not isinstance(values, list) for values in traces)
        or len({len(values) for values in traces}) != 1
        or (not allow_empty and not traces[1])
        or any(
            isinstance(value, bool) or int(value) != value or int(value) < 0
            for value in traces[0]
        )
        or any(
            isinstance(value, bool) or int(value) != value or int(value) < 0
            for value in traces[1]
        )
        or any(
            isinstance(value, bool) or int(value) != value or int(value) <= 0
            for value in traces[2]
        )
        or any(
            isinstance(value, bool) or int(value) != value or int(value) <= 0
            for value in traces[3]
        )
    ):
        raise ValueError("R/S target-action continuation trace is invalid")
    sweep_trace, order_trace, budget_trace, start_fe_trace = traces
    return (
        [int(value) for value in sweep_trace],
        [int(value) for value in order_trace],
        [int(value) for value in budget_trace],
        [int(value) for value in start_fe_trace],
    )


def _validate_rs_gcb_arm(
    row: Mapping[str, str],
    context: Mapping[str, str],
    action: GcbAction,
    *,
    start_fe_trace: Sequence[int],
) -> None:
    try:
        lifecycle_payload = json.loads(str(row.get("action_lifecycle_payload", "")))
        if not isinstance(lifecycle_payload, dict):
            raise ValueError("lifecycle payload must be an object")
        lifecycle_fields = dict(lifecycle_payload)
        if lifecycle_fields.pop("action", None) != GCB_ACTION:
            raise ValueError("lifecycle action name is invalid")
        state = GcbExecutionState(**lifecycle_fields)
        state.validate_for(action)
        candidate_fitness = float(row.get("action_candidate_fitness", "nan"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("R/S GCB lifecycle is invalid") from error
    horizon_fe = int(context["horizon_fe"])
    expected_accepted = str(
        int(candidate_fitness < float(context["gcb_acceptance_fitness"]))
    )
    expected_post_hash = (
        row.get("action_candidate_hash")
        if expected_accepted == "1"
        else context.get("gcb_initial_mean_hash")
    )
    if (
        int(row.get("action_budget_fes", "-1")) != horizon_fe
        or int(row.get("action_actual_fes", "-1")) != horizon_fe
        or row.get("action_instance_hash") != context.get("gcb_action_hash")
        or _canonical_payload_hash(lifecycle_payload)
        != row.get("action_lifecycle_hash")
        or state.state_hash(action) != row.get("action_lifecycle_hash")
        or state.status != "completed"
        or state.consumed_fes != horizon_fe
        or state.started_fe != int(context["dispatch_fe"])
        or state.completed_fe != int(context["dispatch_fe"]) + horizon_fe
        or state.final_state_hash != row.get("optimizer_final_state_hash")
        or row.get("optimizer_parameter_hash")
        != context.get("gcb_parameter_hash")
        or not _is_sha256(row.get("action_candidate_hash"))
        or not math.isfinite(candidate_fitness)
        or row.get("action_accepted") != expected_accepted
        or row.get("action_post_incumbent_hash") != expected_post_hash
        or row.get("optimizer_initial_state_hash") != action.initial_state_hash
        or not _is_sha256(row.get("optimizer_final_state_hash"))
        or row.get("optimizer_scope") != "full_space"
        or row.get("selected_candidate") != GCB_ACTION
        or int(row.get("optimizer_population_size", "-1"))
        != CANONICAL_SEP_CMA_POPULATION_SIZE
        or int(row.get("optimizer_generation_count", "-1"))
        != horizon_fe // CANONICAL_SEP_CMA_POPULATION_SIZE
        or row.get("continuation_policy_applied") != "1"
        or any(fe <= horizon_fe for fe in start_fe_trace)
        or (
            row.get("horizon") == "sweep_3"
            and (not start_fe_trace or start_fe_trace[0] != horizon_fe + 1)
        )
    ):
        raise ValueError("R/S GCB arm contract is invalid")


def _validate_rs_budget_arm(
    row: Mapping[str, str],
    action: BudgetAllocationAction,
    *,
    sweep_trace: Sequence[int],
    order_trace: Sequence[int],
    budget_trace: Sequence[int],
    start_fe_trace: Sequence[int],
) -> None:
    try:
        lifecycle_payload = json.loads(str(row.get("action_lifecycle_payload", "")))
        if (
            not isinstance(lifecycle_payload, dict)
            or set(lifecycle_payload)
            != {
                "action",
                "instance",
                "instance_hash",
                "execution",
                "execution_hash",
            }
            or lifecycle_payload.get("action")
            != FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
        ):
            raise ValueError("lifecycle payload must be an object")
        instance_payload = lifecycle_payload.get("instance")
        execution_payload = lifecycle_payload.get("execution")
        if (
            not isinstance(instance_payload, dict)
            or not isinstance(execution_payload, dict)
        ):
            raise ValueError("budget action lifecycle sections must be objects")
        instance_fields = dict(instance_payload)
        execution_fields = dict(execution_payload)
        if (
            instance_fields.pop("action", None)
            != FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
            or execution_fields.pop("action", None)
            != FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
        ):
            raise ValueError("lifecycle action name is invalid")
        recorded_action = BudgetAllocationAction(**instance_fields)
        recorded_state = BudgetAllocationExecutionState(**execution_fields)
        recorded_state.validate_for(recorded_action)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("R/S frozen budget lifecycle is invalid") from error
    target_positions = [
        index
        for index, sweep in enumerate(sweep_trace)
        if sweep == action.target_sweep
    ]
    expected_candidate_hash = _canonical_payload_hash(
        {"group_budgets": action.group_budgets}
    )
    if (
        recorded_action != action
        or recorded_state.status != "consumed"
        or int(row.get("action_budget_fes", "-1")) != 0
        or int(row.get("action_actual_fes", "-1")) != 0
        or row.get("action_instance_hash") != action.action_hash
        or _canonical_payload_hash(lifecycle_payload)
        != row.get("action_lifecycle_hash")
        or lifecycle_payload.get("instance_hash") != action.action_hash
        or lifecycle_payload.get("execution_hash")
        != recorded_state.state_hash(action)
        or row.get("action_accepted") != "1"
        or row.get("action_candidate_hash") != expected_candidate_hash
        or row.get("action_candidate_fitness")
        or not _is_sha256(row.get("action_post_incumbent_hash"))
        or row.get("optimizer_scope") != "decomposed_groups"
        or row.get("optimizer_parameter_hash") != action.parameter_hash
        or row.get("optimizer_initial_state_hash")
        or row.get("optimizer_final_state_hash")
        or int(row.get("optimizer_population_size", "-1")) != 0
        or int(row.get("optimizer_generation_count", "-1")) != 0
        or row.get("selected_candidate")
        != FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION
        or row.get("continuation_policy_applied") != str(int(bool(target_positions)))
        or (
            target_positions
            and recorded_state.application_fe
            != min(start_fe_trace[index] for index in target_positions)
        )
        or any(
            budget_trace[index] != action.group_budgets[order_trace[index]]
            for index in target_positions
        )
        or (
            row.get("horizon") == "sweep_3"
            and {
                order_trace[index] for index in target_positions
            }
            != set(range(len(action.group_budgets)))
        )
    ):
        raise ValueError("R/S frozen budget arm contract is invalid")


def validate_rs_family_target_rows(
    context_rows: Sequence[Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, str]],
) -> tuple[ActionCeilingObservation, ...]:
    """Validate the two-arm R/S action experiment under the GCB protocol."""

    contexts: dict[str, Mapping[str, str]] = {}
    states: dict[str, tuple[tuple[int, ...], tuple[int, ...], int]] = {}
    gcb_actions: dict[str, GcbAction] = {}
    budget_actions: dict[str, BudgetAllocationAction] = {}
    for row in context_rows:
        problem_id = str(row.get("problem_id", ""))
        if problem_id not in RS_VALIDATION_CASES:
            raise ValueError("R/S target-action context has an unsupported case")
        contract = action_ceiling_capture_contract(
            RS_FAMILY_TARGET_PROFILE,
            problem_id,
        )
        context_id = str(row.get("context_id", ""))
        if (
            row.get("protocol_version") != contract.protocol_version
            or row.get("cohort") != "real_aob"
            or row.get("runtime_authorized") != "0"
            or row.get("status") != "complete"
            or row.get("invalidation_reason")
            or row.get("native_parity") != "1"
            or not context_id
            or context_id in contexts
            or not row.get("relation_id")
            or row.get("selector_arm") not in ACTION_CEILING_ARMS
            or not row.get("selector_reason")
            or any(
                not _is_sha256(row.get(field))
                for field in (
                    "action_set_hash",
                    "checkpoint_hash",
                    "dispatch_checkpoint_hash",
                    "dispatch_anchor_hash",
                )
            )
            or int(row.get("seed", "-1")) < 0
        ):
            raise ValueError("R/S target-action context truth contract is invalid")
        states[context_id] = _rs_context_state(row)
        if contract.arms == RS_FAMILY_RASTRIGIN_ARMS:
            gcb_actions[context_id] = _rs_gcb_action(row)
        elif contract.arms == RS_FAMILY_SCHWEFEL_ARMS:
            if any(row.get(field) for field in _GCB_CONTEXT_FIELDS):
                raise ValueError("Schwefel target context contains GCB fields")
            budget_actions[context_id] = _rs_budget_action(row)
        else:
            raise ValueError("R/S target-action arm contract is unsupported")
        contexts[context_id] = row

    by_context: dict[str, dict[tuple[str, str], Mapping[str, str]]] = {}
    for row in arm_rows:
        context_id = str(row.get("context_id", ""))
        if context_id not in contexts:
            raise ValueError("R/S target-action arm references an unknown context")
        context = contexts[context_id]
        problem_id = str(context["problem_id"])
        contract = action_ceiling_capture_contract(
            RS_FAMILY_TARGET_PROFILE,
            problem_id,
        )
        arm = str(row.get("arm", ""))
        horizon = str(row.get("horizon", ""))
        if (
            row.get("protocol_version") != contract.protocol_version
            or row.get("cohort") != context.get("cohort")
            or row.get("problem_id") != problem_id
            or row.get("seed") != context.get("seed")
            or arm not in contract.arms
            or horizon not in ACTION_CEILING_HORIZONS
            or row.get("runtime_authorized") != "0"
            or row.get("status") != "complete"
            or row.get("invalidation_reason")
            or row.get("counterfactual_applied") not in {"0", "1"}
            or row.get("continuation_policy_applied") not in {"0", "1"}
            or row.get("action_accepted") not in {"0", "1"}
            or not row.get("selected_candidate")
        ):
            raise ValueError("R/S target-action arm truth contract is invalid")
        action_budget_fes = int(row.get("action_budget_fes", "-1"))
        action_actual_fes = int(row.get("action_actual_fes", "-1"))
        population_size = int(row.get("optimizer_population_size", "-1"))
        generation_count = int(row.get("optimizer_generation_count", "-1"))
        mutation_norm = float(row.get("mutation_norm", "nan"))
        mean_mutation_norm = float(row.get("optimizer_mean_mutation_norm", "nan"))
        warm_start_count = int(row.get("warm_start_trigger_count", "-1"))
        warm_start_norm = float(row.get("warm_start_mean_shift_norm", "nan"))
        native_error = float(row.get("native_error", "nan"))
        arm_error = float(row.get("arm_error", "nan"))
        recorded_delta = float(row.get("delta", "nan"))
        if (
            action_budget_fes < 0
            or not 0 <= action_actual_fes <= action_budget_fes
            or population_size < 0
            or generation_count < 0
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (
                    mutation_norm,
                    mean_mutation_norm,
                    warm_start_norm,
                    native_error,
                    arm_error,
                )
            )
            or not math.isfinite(recorded_delta)
            or warm_start_count != 0
            or warm_start_norm != 0.0
        ):
            raise ValueError("R/S target-action arm numeric contract is invalid")
        expected_applied = str(
            int(
                mutation_norm > 0.0
                or mean_mutation_norm > 0.0
                or action_actual_fes > 0
                or row.get("continuation_policy_applied") == "1"
            )
        )
        if row.get("counterfactual_applied") != expected_applied:
            raise ValueError("R/S counterfactual truth flag is invalid")

        horizon_fe = states[context_id][2]
        target_relative_fe = {
            "immediate": 1,
            PRIMARY_HORIZON: horizon_fe,
            "sweep_3": 3 * horizon_fe,
        }[horizon]
        allow_empty = (
            arm == GCB_ACTION
            and horizon in {"immediate", PRIMARY_HORIZON}
        )
        sweep_trace, order_trace, budget_trace, start_fe_trace = _rs_arm_traces(
            row,
            allow_empty=allow_empty,
        )
        if (
            int(row.get("target_fe", "-1"))
            != int(context["dispatch_fe"]) + target_relative_fe
            or int(row.get("natural_endpoint_fe", "-1"))
            < int(context["dispatch_fe"]) + 3 * horizon_fe
            or start_fe_trace != sorted(set(start_fe_trace))
            or any(fe > target_relative_fe for fe in start_fe_trace)
            or any(
                next_sweep not in {sweep, sweep + 1}
                for sweep, next_sweep in zip(
                    sweep_trace,
                    sweep_trace[1:],
                )
            )
        ):
            raise ValueError("R/S target-action horizon FE contract is invalid")

        populations, uniform_budgets, _ = states[context_id]
        for trace_index, (group_index, budget) in enumerate(
            zip(order_trace, budget_trace, strict=True)
        ):
            if not 0 <= group_index < len(populations):
                raise ValueError("R/S continuation group index is invalid")
            if arm == FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION:
                budget_action = budget_actions[context_id]
                expected_budget = (
                    budget_action.group_budgets[group_index]
                    if sweep_trace[trace_index] == budget_action.target_sweep
                    else uniform_budgets[group_index]
                )
                if budget != expected_budget:
                    raise ValueError("R/S frozen budget trace changed its allocation")
            elif budget != uniform_budgets[group_index]:
                raise ValueError("R/S non-budget arm changed a group budget")
        for sweep in set(sweep_trace):
            positions = [
                index for index, value in enumerate(sweep_trace) if value == sweep
            ]
            groups = [order_trace[index] for index in positions]
            if groups != sorted(groups) or len(groups) != len(set(groups)):
                raise ValueError("R/S continuation changed the native group order")
            if len(groups) == len(populations):
                if set(groups) != set(range(len(populations))):
                    raise ValueError("R/S continuation lost a group")
                if arm == FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION and math.fsum(
                    budget_trace[index] for index in positions
                ) != math.fsum(uniform_budgets):
                    raise ValueError("R/S adaptive budget did not preserve sweep FEs")

        if arm == GCB_ACTION:
            _validate_rs_gcb_arm(
                row,
                context,
                gcb_actions[context_id],
                start_fe_trace=start_fe_trace,
            )
        elif arm == FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION:
            _validate_rs_budget_arm(
                row,
                budget_actions[context_id],
                sweep_trace=sweep_trace,
                order_trace=order_trace,
                budget_trace=budget_trace,
                start_fe_trace=start_fe_trace,
            )
        elif (
            action_budget_fes != 0
            or action_actual_fes != 0
            or row.get("action_instance_hash")
            or row.get("action_lifecycle_payload")
            or row.get("action_lifecycle_hash")
            or row.get("action_accepted") != "0"
            or row.get("action_candidate_hash")
            or row.get("action_candidate_fitness")
            or (
                arm == "native_eq8"
                and not _is_sha256(row.get("action_post_incumbent_hash"))
            )
            or (
                arm != "native_eq8"
                and row.get("action_post_incumbent_hash")
            )
            or row.get("optimizer_parameter_hash")
            or row.get("optimizer_initial_state_hash")
            or row.get("optimizer_final_state_hash")
            or population_size != 0
            or generation_count != 0
            or (
                arm == "native_eq8"
                and row.get("optimizer_scope") != "relation_writeback"
            )
            or row.get("selected_candidate") != arm
            or not start_fe_trace
            or start_fe_trace[0] != 1
            or (arm == "native_eq8" and row.get("continuation_policy_applied") != "0")
        ):
            raise ValueError("R/S non-Sep arm contains forbidden optimizer state")

        key = (arm, horizon)
        if key in by_context.setdefault(context_id, {}):
            raise ValueError("duplicate R/S context arm horizon row")
        by_context[context_id][key] = row

    observations: list[ActionCeilingObservation] = []
    invariant_fields = (
        "action_candidate_fitness",
        "action_candidate_hash",
        "action_accepted",
        "action_post_incumbent_hash",
        "optimizer_final_state_hash",
        "action_lifecycle_payload",
        "action_lifecycle_hash",
    )
    for context_id, context in contexts.items():
        contract = action_ceiling_capture_contract(
            RS_FAMILY_TARGET_PROFILE,
            str(context["problem_id"]),
        )
        expected = {
            (arm, horizon)
            for arm in contract.arms
            for horizon in ACTION_CEILING_HORIZONS
        }
        result_rows = by_context.get(context_id, {})
        if set(result_rows) != expected:
            raise ValueError("R/S context does not contain exactly two arms and three horizons")
        if GCB_ACTION in contract.arms:
            sep_rows = [
                result_rows[(GCB_ACTION, horizon)]
                for horizon in ACTION_CEILING_HORIZONS
            ]
            if any(
                row.get(field) != sep_rows[0].get(field)
                for row in sep_rows[1:]
                for field in invariant_fields
            ):
                raise ValueError("R/S GCB outcome differs across horizons")
        if FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION in contract.arms:
            budget_rows = [
                result_rows[
                    (FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION, horizon)
                ]
                for horizon in ACTION_CEILING_HORIZONS
            ]
            if any(
                row.get(field) != budget_rows[0].get(field)
                for row in budget_rows[1:]
                for field in invariant_fields
            ):
                raise ValueError("R/S frozen budget outcome differs across horizons")
        for horizon in ACTION_CEILING_HORIZONS:
            native = result_rows[("native_eq8", horizon)]
            native_error = float(native["native_error"])
            if (
                float(native["arm_error"]) != native_error
                or float(native["delta"]) != 0.0
            ):
                raise ValueError("R/S native arm is not its zero-delta reference")
            if FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION in contract.arms:
                budget = result_rows[
                    (FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION, horizon)
                ]
                native_trace = _rs_arm_traces(native, allow_empty=False)
                budget_trace = _rs_arm_traces(budget, allow_empty=False)
                native_dispatches = list(zip(native_trace[0], native_trace[1], strict=True))
                budget_dispatches = list(zip(budget_trace[0], budget_trace[1], strict=True))
                shared_prefix = min(len(native_dispatches), len(budget_dispatches))
                if (
                    native_dispatches[:shared_prefix]
                    != budget_dispatches[:shared_prefix]
                    or budget.get("action_post_incumbent_hash")
                    != native.get("action_post_incumbent_hash")
                ):
                    raise ValueError(
                        "R/S frozen budget arm changed non-budget dispatch state"
                    )
            for arm in contract.arms:
                row = result_rows[(arm, horizon)]
                if float(row["native_error"]) != native_error:
                    raise ValueError("R/S paired arms use different native references")
                delta = actionability_delta(native_error, float(row["arm_error"]))
                if abs(delta - float(row["delta"])) > 1e-12:
                    raise ValueError("R/S target-action delta does not match raw errors")
                observations.append(
                    ActionCeilingObservation(
                        context_id=context_id,
                        cohort="real_aob",
                        problem_id=str(context["problem_id"]),
                        seed=int(context["seed"]),
                        arm=arm,
                        horizon=horizon,
                        delta=delta,
                        selector_arm="native_eq8",
                    )
                )
    return tuple(observations)


def _average(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(float(value) for value in values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rs_family_target(
    observations: Sequence[ActionCeilingObservation],
    *,
    inferential: bool,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, object]]:
    """Summarize each fixed family action independently at sweep_1."""

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    primary = tuple(row for row in observations if row.horizon == PRIMARY_HORIZON)
    summaries: list[dict[str, object]] = []
    for problem_id in RS_VALIDATION_CASES:
        case_rows = tuple(row for row in primary if row.problem_id == problem_id)
        if not case_rows:
            continue
        target_arm = action_ceiling_capture_contract(
            RS_FAMILY_TARGET_PROFILE,
            problem_id,
        ).arms[1]
        target_rows = tuple(row for row in case_rows if row.arm == target_arm)
        native_rows = tuple(row for row in case_rows if row.arm == "native_eq8")
        if len(target_rows) != len(native_rows):
            raise ValueError("R/S target-action summary is missing paired native rows")
        clusters: dict[int, list[float]] = {}
        for row in target_rows:
            clusters.setdefault(row.seed, []).append(row.delta)
        cluster_keys = tuple(sorted(clusters))
        rng = Random(bootstrap_seed)
        bootstrap_means: list[float] = []
        for _replicate in range(bootstrap_replicates):
            sampled: list[float] = []
            for _cluster in cluster_keys:
                sampled_key = cluster_keys[rng.randrange(len(cluster_keys))]
                sampled.extend(clusters[sampled_key])
            bootstrap_means.append(_average(sampled))
        deltas = [row.delta for row in target_rows]
        mean_delta = _average(deltas)
        delta_lcb = _quantile(bootstrap_means, 0.025)
        delta_ucb = _quantile(bootstrap_means, 0.975)
        positive_count = sum(value > 0.0 for value in deltas)
        material_count = sum(value > MATERIAL_POSITIVE_DELTA for value in deltas)
        catastrophic_count = sum(value <= CATASTROPHIC_DELTA for value in deltas)
        material_rate = material_count / len(deltas)
        if not inferential:
            gate = "mechanical_smoke_only"
        elif len(cluster_keys) < len(RS_VALIDATION_SEEDS):
            gate = "insufficient_seed_clusters"
        elif catastrophic_count:
            gate = "reject_target_action_catastrophic_loss"
        elif mean_delta <= 0.0:
            gate = "redesign_target_action"
        elif delta_lcb <= 0.0:
            gate = "collect_more_target_action_contexts"
        elif material_rate < SPARSE_POSITIVE_THRESHOLD:
            gate = "force_abstain_sparse_headroom"
        else:
            gate = "target_action_validated"
        summaries.append(
            {
                "protocol_version": RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION,
                "profile": RS_FAMILY_TARGET_PROFILE,
                "problem_id": problem_id,
                "target_arm": target_arm,
                "horizon": PRIMARY_HORIZON,
                "context_count": len(target_rows),
                "cluster_count": len(cluster_keys),
                "mean_delta": mean_delta,
                "min_delta": min(deltas),
                "max_delta": max(deltas),
                "delta_lcb": delta_lcb,
                "delta_ucb": delta_ucb,
                "positive_count": positive_count,
                "positive_rate": positive_count / len(deltas),
                "material_positive_count": material_count,
                "material_positive_rate": material_rate,
                "catastrophic_count": catastrophic_count,
                "catastrophic_rate": catastrophic_count / len(deltas),
                "gate": gate,
            }
        )
    return summaries


def build_rs_family_integrity_gate(
    specs: Sequence[TrajectorySpec],
    context_rows: Sequence[Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, str]],
) -> dict[str, int]:
    expected_context_count = 4 * len(specs)
    expected_arm_count = sum(
        4
        * len(
            action_ceiling_capture_contract(
                spec.action_ceiling_profile,
                spec.problem_id,
            ).arms
        )
        * len(ACTION_CEILING_HORIZONS)
        for spec in specs
    )
    expected_identities = {
        (spec.problem_id, spec.seed) for spec in specs
    }
    observed_identities = {
        (str(row.get("problem_id")), int(row.get("seed", "-1")))
        for row in context_rows
    }
    context_counts = {
        identity: sum(
            (str(row.get("problem_id")), int(row.get("seed", "-1")))
            == identity
            for row in context_rows
        )
        for identity in expected_identities
    }
    contexts_by_identity = {
        identity: [
            row
            for row in context_rows
            if (str(row.get("problem_id")), int(row.get("seed", "-1")))
            == identity
        ]
        for identity in expected_identities
    }
    checks = {
        "trajectory_context_coverage": int(observed_identities == expected_identities),
        "four_relations_per_trajectory": int(
            all(count == 4 for count in context_counts.values())
        ),
        "unique_relation_ids_per_trajectory": int(
            all(
                len({str(row.get("relation_id")) for row in rows}) == 4
                for rows in contexts_by_identity.values()
            )
        ),
        "unique_action_set_hashes_per_trajectory": int(
            all(
                len({str(row.get("action_set_hash")) for row in rows}) == 4
                for rows in contexts_by_identity.values()
            )
        ),
        "unique_dispatch_checkpoint_hashes_per_trajectory": int(
            all(
                len(
                    {
                        str(row.get("dispatch_checkpoint_hash"))
                        for row in rows
                    }
                )
                == 4
                for rows in contexts_by_identity.values()
            )
        ),
        "expected_context_count": int(len(context_rows) == expected_context_count),
        "expected_arm_result_count": int(len(arm_rows) == expected_arm_count),
        "two_arms_three_horizons": int(
            len(
                {
                    (row.get("context_id"), row.get("arm"), row.get("horizon"))
                    for row in arm_rows
                }
            )
            == expected_arm_count
        ),
        "native_parity": int(
            all(row.get("native_parity") == "1" for row in context_rows)
        ),
        "truth_fields_complete": int(
            all(
                row.get("status") == "complete"
                and not row.get("invalidation_reason")
                and row.get("runtime_authorized") == "0"
                for row in (*context_rows, *arm_rows)
            )
        ),
    }
    checks["passed"] = int(all(checks.values()))
    if not checks["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"R/S target-action integrity gate failed: {failed}")
    return checks


def summarize_rs_fe_accounting(
    specs: Sequence[TrajectorySpec],
    context_rows: Sequence[Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    contexts = {str(row["context_id"]): row for row in context_rows}
    unique_arms: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in arm_rows:
        key = (str(row["context_id"]), str(row["arm"]))
        existing = unique_arms.setdefault(key, row)
        if (
            existing["action_budget_fes"] != row["action_budget_fes"]
            or existing["action_actual_fes"] != row["action_actual_fes"]
            or existing["natural_endpoint_fe"] != row["natural_endpoint_fe"]
        ):
            raise ValueError("R/S action FE accounting differs across horizons")
    arms = tuple(sorted({arm for _context_id, arm in unique_arms}))
    action_fes = {
        arm: sum(
            int(row["action_actual_fes"])
            for (_context_id, candidate), row in unique_arms.items()
            if candidate == arm
        )
        for arm in arms
    }
    return {
        "nominal_trajectory_fe_total": sum(spec.max_fes for spec in specs),
        "phase_boundary_fe_total": sum(
            int(row["phase_boundary_fe"]) for row in context_rows
        ),
        "sweep_horizon_fe_total": sum(
            int(row["horizon_fe"]) for row in context_rows
        ),
        "branch_action_fe_total": sum(action_fes.values()),
        "branch_action_fe_by_arm": action_fes,
        "branch_evaluated_fe_total": sum(
            int(row["natural_endpoint_fe"])
            - int(contexts[context_id]["dispatch_fe"])
            for (context_id, _arm), row in unique_arms.items()
        ),
    }


def build_integrity_gate(
    specs: Sequence[TrajectorySpec],
    context_rows: Sequence[Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, str]],
) -> dict[str, int]:
    expected_contexts = sum(0 if spec.problem_id == "E1" else 4 for spec in specs)
    expected_arm_rows = (
        expected_contexts * len(ACTION_CEILING_ARMS) * len(ACTION_CEILING_HORIZONS)
    )
    observed_identities = {
        (str(row.get("cohort")), str(row.get("problem_id")), int(row.get("seed", -1)))
        for row in context_rows
    }
    expected_identities = {
        (spec.cohort, spec.problem_id, spec.seed)
        for spec in specs
        if spec.problem_id != "E1"
    }
    contexts_per_identity = {
        identity: sum(
            (
                str(row.get("cohort")),
                str(row.get("problem_id")),
                int(row.get("seed", -1)),
            )
            == identity
            for row in context_rows
        )
        for identity in expected_identities
    }
    checks = {
        "trajectory_context_coverage": int(observed_identities == expected_identities),
        "four_relations_per_overlap_trajectory": int(
            all(count == 4 for count in contexts_per_identity.values())
        ),
        "expected_context_count": int(len(context_rows) == expected_contexts),
        "expected_arm_result_count": int(len(arm_rows) == expected_arm_rows),
        "native_parity": int(
            all(row.get("native_parity") == "1" for row in context_rows)
        ),
        "all_contexts_complete": int(
            all(
                row.get("status") == "complete"
                and not row.get("invalidation_reason")
                for row in context_rows
            )
        ),
        "all_arm_results_complete": int(
            all(
                row.get("status") == "complete"
                and not row.get("invalidation_reason")
                for row in arm_rows
            )
        ),
        "runtime_unauthorized": int(
            all(row.get("runtime_authorized") == "0" for row in context_rows)
            and all(row.get("runtime_authorized") == "0" for row in arm_rows)
        ),
        "branch_matrix_complete": int(
            len(
                {
                    (
                        row.get("context_id"),
                        row.get("arm"),
                        row.get("horizon"),
                    )
                    for row in arm_rows
                }
            )
            == expected_arm_rows
        ),
        "counterfactual_flags_valid": int(
            all(row.get("counterfactual_applied") in {"0", "1"} for row in arm_rows)
        ),
    }
    checks["passed"] = int(all(checks.values()))
    if not checks["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"action-ceiling integrity gate failed: {failed}")
    return checks


def summarize_fe_accounting(
    specs: Sequence[TrajectorySpec],
    context_rows: Sequence[Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    contexts = {str(row["context_id"]): row for row in context_rows}
    unique_arm_rows: dict[tuple[str, str], Mapping[str, str]] = {}
    target_max_by_horizon: dict[str, int] = {}
    for row in arm_rows:
        key = (str(row["context_id"]), str(row["arm"]))
        existing = unique_arm_rows.setdefault(key, row)
        if (
            existing["action_budget_fes"] != row["action_budget_fes"]
            or existing["action_actual_fes"] != row["action_actual_fes"]
            or existing["natural_endpoint_fe"] != row["natural_endpoint_fe"]
        ):
            raise ValueError("arm FE accounting differs across label horizons")
        horizon = str(row["horizon"])
        target_max_by_horizon[horizon] = max(
            target_max_by_horizon.get(horizon, 0),
            int(row["target_fe"]),
        )
    action_fes_by_arm = {
        arm: sum(
            int(row["action_actual_fes"])
            for (_context_id, candidate_arm), row in unique_arm_rows.items()
            if candidate_arm == arm
        )
        for arm in ACTION_CEILING_ARMS
    }
    dispatch_values = [int(row["dispatch_fe"]) for row in context_rows]
    return {
        "nominal_trajectory_fe_total": sum(spec.max_fes for spec in specs),
        "phase_boundary_fe_total": sum(
            int(row["phase_boundary_fe"]) for row in context_rows
        ),
        "dispatch_fe_min": min(dispatch_values) if dispatch_values else None,
        "dispatch_fe_max": max(dispatch_values) if dispatch_values else None,
        "sweep_horizon_fe_total": sum(
            int(row["horizon_fe"]) for row in context_rows
        ),
        "branch_action_fe_total": sum(action_fes_by_arm.values()),
        "branch_action_fe_by_arm": action_fes_by_arm,
        "branch_evaluated_fe_total": sum(
            int(row["natural_endpoint_fe"])
            - int(contexts[context_id]["dispatch_fe"])
            for (context_id, _arm), row in unique_arm_rows.items()
        ),
        "target_fe_max_by_horizon": target_max_by_horizon,
    }


def aggregate_action_ceiling(
    context_rows: Sequence[Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    observations = validate_raw_rows(context_rows, arm_rows)
    summaries: list[dict[str, object]] = []
    for cohort in ("real_aob", "synthetic_conflict"):
        if not any(row.cohort == cohort for row in observations):
            continue
        for horizon in ACTION_CEILING_HORIZONS:
            summaries.append(
                summarize_action_ceiling(
                    observations,
                    cohort=cohort,
                    horizon=horizon,
                )
            )
    return summaries


def _worker_command(spec: TrajectorySpec, output_root: Path) -> tuple[str, ...]:
    command = (
        sys.executable,
        "-m",
        "experiments.pilots.exp_019_conflict_resolution_pilot._diagnostic_worker",
        "--cohort",
        spec.cohort,
        "--case",
        spec.problem_id,
        "--seed",
        str(spec.seed),
        "--max-fes",
        str(spec.max_fes),
        "--output-root",
        str(output_root),
        "--timestamp",
        spec.trajectory_id,
    )
    if spec.action_ceiling_profile != ACTION_CEILING_FULL_MATRIX_PROFILE:
        command += ("--profile", spec.action_ceiling_profile)
    return command


def _run_worker(spec: TrajectorySpec, output_root: Path) -> None:
    completed = subprocess.run(
        _worker_command(spec, output_root),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"action-ceiling worker failed for {spec.trajectory_id}: "
            f"{completed.stderr.strip()}"
        )


def _trajectory_artifacts(spec: TrajectorySpec, output_root: Path) -> tuple[Path, Path]:
    base = output_root / spec.trajectory_id / spec.function_name
    return (
        base / f"{spec.problem_id}_action_ceiling_contexts.csv",
        base / f"{spec.problem_id}_action_ceiling_arm_results.csv",
    )


def _trajectory_aob_input_manifest(
    spec: TrajectorySpec,
    output_root: Path,
) -> Path:
    base = output_root / spec.trajectory_id / spec.function_name
    return base / f"{spec.problem_id}_aob_input_manifest.csv"


def _validate_trajectory_aob_input_manifest(
    spec: TrajectorySpec,
    path: Path,
) -> None:
    rows = _read_csv(path, AOB_INPUT_MANIFEST_FIELDS)
    if not rows:
        raise ValueError(f"AOB input manifest must contain rows: {path}")
    expected_paths = {
        candidate.name: candidate.resolve()
        for candidate in required_aob_data_files(
            VENDOR_DATA_DIR,
            int(spec.problem_id[1:]),
        )
    }
    recorded_files = [str(row.get("file", "")) for row in rows]
    if len(recorded_files) != len(set(recorded_files)) or set(recorded_files) != set(
        expected_paths
    ):
        raise ValueError(f"AOB input manifest coverage mismatch: {path}")
    for row in rows:
        filename = str(row["file"])
        expected_path = expected_paths[filename]
        before_hash = str(row.get("sha256_before", ""))
        after_hash = str(row.get("sha256_after", ""))
        if (
            row.get("problem_id") != spec.problem_id
            or Path(str(row.get("path", ""))).resolve() != expected_path
            or not _is_sha256(before_hash)
            or before_hash != after_hash
            or before_hash != _sha256(expected_path)
            or row.get("unchanged") != "1"
        ):
            raise ValueError(f"AOB input manifest truth mismatch: {path}")


def aggregate_stage_artifacts(
    stage: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cohort: str = "all",
    worker_count: int | None = None,
) -> Path:
    specs = build_specs(stage, cohort=cohort)
    context_rows: list[dict[str, str]] = []
    arm_rows: list[dict[str, str]] = []
    input_artifacts: dict[str, dict[str, str]] = {}
    for spec in specs:
        context_path, arm_path = _trajectory_artifacts(spec, output_root)
        context_rows.extend(_read_csv(context_path, CONTEXT_FIELDS))
        arm_rows.extend(_read_csv(arm_path, ARM_RESULT_FIELDS))
        input_artifacts[spec.trajectory_id] = {
            "contexts_sha256": _sha256(context_path),
            "arm_results_sha256": _sha256(arm_path),
        }
        if stage in {"rs_smoke", "rs_family_validation"}:
            aob_input_path = _trajectory_aob_input_manifest(spec, output_root)
            _validate_trajectory_aob_input_manifest(spec, aob_input_path)
            input_artifacts[spec.trajectory_id]["aob_input_manifest_sha256"] = (
                _sha256(aob_input_path)
            )
    if stage in {"rs_smoke", "rs_family_validation"}:
        observations = validate_rs_family_target_rows(context_rows, arm_rows)
        integrity_gate = build_rs_family_integrity_gate(
            specs,
            context_rows,
            arm_rows,
        )
        summaries = summarize_rs_family_target(
            observations,
            inferential=stage == "rs_family_validation",
        )
        fe_summary = summarize_rs_fe_accounting(specs, context_rows, arm_rows)
        stage_root = output_root / stage
        context_output = stage_root / "action_ceiling_contexts.csv"
        arm_output = stage_root / "action_ceiling_arm_results.csv"
        summary_output = stage_root / "action_ceiling_summary.csv"
        manifest_output = stage_root / "manifest.json"
        _write_csv(context_output, context_rows, CONTEXT_FIELDS)
        _write_csv(arm_output, arm_rows, ARM_RESULT_FIELDS)
        _write_csv(summary_output, summaries, RS_SUMMARY_FIELDS)
        cases = tuple(dict.fromkeys(spec.problem_id for spec in specs))
        arms_by_case = {
            case: list(
                action_ceiling_capture_contract(
                    RS_FAMILY_TARGET_PROFILE,
                    case,
                ).arms
            )
            for case in cases
        }
        target_mapping = {case: arms[1] for case, arms in arms_by_case.items()}
        arm_contract_hashes = {
            case: _canonical_payload_hash(
                {
                    "profile": RS_FAMILY_TARGET_PROFILE,
                    "protocol_version": RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION,
                    "problem_id": case,
                    "arms": arms,
                    "horizons": ACTION_CEILING_HORIZONS,
                }
            )
            for case, arms in arms_by_case.items()
        }
        case_gates = {
            str(row["problem_id"]): str(row["gate"]) for row in summaries
        }
        action_gate_passed = bool(case_gates) and all(
            gate == "target_action_validated" for gate in case_gates.values()
        )
        manifest = {
            "protocol_version": RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION,
            "schema_version": "exp019-rs-family-gcb-action-validation-manifest-v2",
            "profile": RS_FAMILY_TARGET_PROFILE,
            "stage": stage,
            "cohort_filter": "real_aob",
            "worker_count": worker_count,
            "runtime_authorized": 0,
            "runtime_consumed": 0,
            "selector_authorized": 0,
            "trajectory_count": len(specs),
            "context_count": len(context_rows),
            "arm_result_count": len(arm_rows),
            "case_target_mapping": target_mapping,
            "arms_by_case": arms_by_case,
            "arm_contract_hashes": arm_contract_hashes,
            "cutoff_tie_policy": "structural_key",
            "no_relation_context": {
                case: {
                    "status": "not_run",
                    "reason": "no_relation_context",
                }
                for case in RS_NO_RELATION_CONTEXT_CASES
            },
            "statistics": {
                "primary_horizon": PRIMARY_HORIZON,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "material_positive_delta": MATERIAL_POSITIVE_DELTA,
                "catastrophic_delta": CATASTROPHIC_DELTA,
                "sparse_positive_threshold": SPARSE_POSITIVE_THRESHOLD,
                "cluster_unit": "case_seed",
            },
            "case_gates": case_gates,
            "action_gate_passed_all_cases": int(action_gate_passed),
            "primary_recommendation": (
                "all_target_actions_validated"
                if action_gate_passed
                else (
                    "mechanical_smoke_only"
                    if stage == "rs_smoke"
                    else "one_or_more_target_actions_not_validated"
                )
            ),
            "integrity_gate": integrity_gate,
            "fe_summary": fe_summary,
            "inputs": {
                "config_sha256": _sha256(CONFIG_PATH),
                "trajectory_artifacts": input_artifacts,
            },
            "artifacts": {
                path.name: _sha256(path)
                for path in (context_output, arm_output, summary_output)
            },
        }
        _write_json(manifest_output, manifest)
        return manifest_output
    integrity_gate = build_integrity_gate(specs, context_rows, arm_rows)
    summaries = aggregate_action_ceiling(context_rows, arm_rows)
    fe_summary = summarize_fe_accounting(specs, context_rows, arm_rows)

    stage_root = output_root / (
        stage if cohort == "all" else f"{stage}-{cohort}"
    )
    context_output = stage_root / "action_ceiling_contexts.csv"
    arm_output = stage_root / "action_ceiling_arm_results.csv"
    summary_output = stage_root / "action_ceiling_summary.csv"
    manifest_output = stage_root / "manifest.json"
    _write_csv(context_output, context_rows, CONTEXT_FIELDS)
    _write_csv(arm_output, arm_rows, ARM_RESULT_FIELDS)
    _write_csv(summary_output, summaries, SUMMARY_FIELDS)
    unique_branches = {
        (str(row["context_id"]), str(row["arm"])) for row in arm_rows
    }
    manifest = {
        "protocol_version": ACTION_CEILING_PROTOCOL_VERSION,
        "schema_version": CONFIG_SCHEMA_VERSION,
        "stage": stage,
        "cohort_filter": cohort,
        "runtime_authorized": 0,
        "runtime_consumed": 0,
        "counterfactual_applied_result_rows": sum(
            row.get("counterfactual_applied") == "1" for row in arm_rows
        ),
        "counterfactual_branch_count": len(unique_branches),
        "trajectory_count": len(specs),
        "context_count": len(context_rows),
        "arm_result_count": len(arm_rows),
        "cohorts_pooled": 0,
        "decision_authorized": int(stage == "pilot"),
        "integrity_gate": integrity_gate,
        "fe_summary": fe_summary,
        "artifacts": {
            path.name: _sha256(path)
            for path in (context_output, arm_output, summary_output)
        },
        "primary_recommendation": next(
            (
                row["recommendation"]
                for row in summaries
                if row["cohort"] == "real_aob" and row["horizon"] == PRIMARY_HORIZON
            ),
            "no_real_aob_contexts",
        )
        if stage == "pilot"
        else "mechanical_smoke_only",
    }
    _write_json(manifest_output, manifest)
    return manifest_output


def run_diagnostic(
    stage: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    jobs: int | None = None,
    cohort: str = "all",
) -> Path:
    config = load_config()
    specs = build_specs(stage, cohort=cohort)
    if any(spec.cohort == "synthetic_conflict" for spec in specs):
        validate_synthetic_bundle()
    if stage in {"rs_smoke", "rs_family_validation"}:
        configured_jobs = config["rs_family_target_validation"][stage]["jobs"]
    else:
        configured_jobs = (
            config["smoke"]["jobs"] if stage == "smoke" else config["pilot"]["jobs"]
        )
    worker_count = int(configured_jobs if jobs is None else jobs)
    if worker_count <= 0:
        raise ValueError("jobs must be positive")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_worker, spec, output_root): spec for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            future.result()
            print(f"completed {spec.trajectory_id}", flush=True)
    return aggregate_stage_artifacts(
        stage,
        output_root=output_root,
        cohort=cohort,
        worker_count=worker_count,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp019 G1 action ceiling.")
    parser.add_argument(
        "--stage",
        choices=("smoke", "pilot", "rs_smoke", "rs_family_validation"),
        default="smoke",
    )
    parser.add_argument(
        "--cohort",
        choices=("all", "real_aob", "synthetic_conflict"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--jobs", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run_diagnostic(
        args.stage,
        output_root=args.output_root,
        jobs=args.jobs,
        cohort=args.cohort,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
