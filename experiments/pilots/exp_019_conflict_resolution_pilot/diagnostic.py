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
from typing import Any, Mapping, Sequence

from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    ACTION_CEILING_ARM_RESULT_FIELDS,
    ACTION_CEILING_CONTEXT_FIELDS,
    ACTION_CEILING_HORIZONS,
    ACTION_CEILING_PROTOCOL_VERSION,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BUDGET_MAX_UNIFORM_MULTIPLIER,
    CATASTROPHIC_DELTA,
    EFFICIENCY_EWMA_ALPHA,
    MATERIAL_POSITIVE_DELTA,
    PRIMARY_HORIZON,
    SPARSE_POSITIVE_THRESHOLD,
    STAGNATION_EPSILON,
    STAGNATION_TRIGGER_STREAK,
    WARM_START_COOLDOWN_SWEEPS,
    ActionCeilingObservation,
    actionability_delta,
    summarize_action_ceiling,
)
from arac.policy.evidence_overlay import UTILITY_EPSILON

from .benchmark import REPO_ROOT, validate_synthetic_bundle


EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIR / "diagnostic_config.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "exp_019_conflict_resolution_pilot"
CONFIG_SCHEMA_VERSION = "exp019-action-ceiling-config-v4"
SMOKE_CASES = ("E3", "S5")
SMOKE_SEEDS = (117, 118, 119)
SMOKE_JOBS = 6
PILOT_SEEDS = (117, 118, 119, 120, 121)
PILOT_MAX_FES = 3_000_000
SMOKE_MAX_FES = 300_000
REAL_CASES = ("E1", "E3", "A4", "R4", "S5")
SYNTHETIC_CASES = ("E3", "A4", "S5")
CASE_FUNCTIONS = {
    "E1": "elliptic",
    "E3": "elliptic",
    "A4": "ackley",
    "R4": "rastrigin",
    "S5": "schwefel",
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


@dataclass(frozen=True)
class TrajectorySpec:
    stage: str
    cohort: str
    problem_id: str
    seed: int
    max_fes: int

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
    }
    if config.get("continuation_actions") != expected_continuation_actions:
        raise ValueError("action-ceiling continuation action contract drifted")
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
    return config


def build_specs(stage: str) -> tuple[TrajectorySpec, ...]:
    load_config()
    if stage == "smoke":
        return tuple(
            TrajectorySpec("smoke", "real_aob", case, seed, SMOKE_MAX_FES)
            for case in SMOKE_CASES
            for seed in SMOKE_SEEDS
        )
    if stage != "pilot":
        raise ValueError("stage must be smoke or pilot")
    specs = [
        TrajectorySpec("pilot", "real_aob", case, seed, PILOT_MAX_FES)
        for case in REAL_CASES
        for seed in PILOT_SEEDS
    ]
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
        contexts[context_id] = row

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
        sweep_trace = json.loads(str(row.get("execution_sweep_trace", "")))
        order_trace = json.loads(str(row.get("execution_order_trace", "")))
        budget_trace = json.loads(str(row.get("group_budget_trace", "")))
        if (
            not isinstance(sweep_trace, list)
            or not isinstance(order_trace, list)
            or not order_trace
            or not isinstance(budget_trace, list)
            or len(sweep_trace) != len(order_trace)
            or len(order_trace) != len(budget_trace)
            or any(isinstance(value, bool) or int(value) < 0 for value in sweep_trace)
            or any(isinstance(value, bool) or int(value) < 0 for value in order_trace)
            or any(isinstance(value, bool) or int(value) <= 0 for value in budget_trace)
        ):
            raise ValueError("action-ceiling continuation trace is invalid")
        expected_applied = str(
            int(
                mutation_norm > 0.0
                or mean_mutation_norm > 0.0
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
        arm = str(row.get("arm", ""))
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
            raise ValueError("context does not contain all eight arms and horizons")
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
            existing["extra_fes"] != row["extra_fes"]
            or existing["natural_endpoint_fe"] != row["natural_endpoint_fe"]
        ):
            raise ValueError("arm FE accounting differs across label horizons")
        horizon = str(row["horizon"])
        target_max_by_horizon[horizon] = max(
            target_max_by_horizon.get(horizon, 0),
            int(row["target_fe"]),
        )
    extra_by_arm = {
        arm: sum(
            int(row["extra_fes"])
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
        "branch_extra_fe_total": sum(extra_by_arm.values()),
        "branch_extra_fe_by_arm": extra_by_arm,
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
    return (
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


def aggregate_stage_artifacts(
    stage: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    specs = build_specs(stage)
    context_rows: list[dict[str, str]] = []
    arm_rows: list[dict[str, str]] = []
    for spec in specs:
        context_path, arm_path = _trajectory_artifacts(spec, output_root)
        context_rows.extend(_read_csv(context_path, CONTEXT_FIELDS))
        arm_rows.extend(_read_csv(arm_path, ARM_RESULT_FIELDS))
    integrity_gate = build_integrity_gate(specs, context_rows, arm_rows)
    summaries = aggregate_action_ceiling(context_rows, arm_rows)
    fe_summary = summarize_fe_accounting(specs, context_rows, arm_rows)

    stage_root = output_root / stage
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
) -> Path:
    config = load_config()
    specs = build_specs(stage)
    if any(spec.cohort == "synthetic_conflict" for spec in specs):
        validate_synthetic_bundle()
    worker_count = int(
        jobs or (config["smoke"]["jobs"] if stage == "smoke" else config["pilot"]["jobs"])
    )
    if worker_count <= 0:
        raise ValueError("jobs must be positive")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_worker, spec, output_root): spec for spec in specs
        }
        for future in as_completed(futures):
            future.result()
    return aggregate_stage_artifacts(stage, output_root=output_root)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp019 G1 action ceiling.")
    parser.add_argument("--stage", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--jobs", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run_diagnostic(args.stage, output_root=args.output_root, jobs=args.jobs)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
