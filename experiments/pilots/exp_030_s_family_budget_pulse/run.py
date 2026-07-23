"""Run and fail-closed validate the S5 budget-pulse mechanical smoke."""

from __future__ import annotations

# The standalone experiment entry point must register ``src`` before local imports.
# ruff: noqa: E402

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arac.actions.budget_reallocation import (
    FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
    BudgetAllocationAction,
    BudgetAllocationExecutionState,
)
from arac.actions.shrunk_budget_pulse import (
    SHRUNK_BUDGET_PULSE_SCHEMA,
    SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
    ShrunkBudgetPulseExecutionState,
    ShrunkEfficiencyBudgetPulseAction,
    allocate_shrunk_efficiency_budgets,
)
from arac.backends.hcc import required_aob_data_files
from arac.backends.hcc_action_ceiling import (
    freeze_efficiency_budget_action,
    freeze_shrunk_efficiency_budget_pulse_action,
)
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    ACTION_CEILING_ARM_RESULT_FIELDS,
    ACTION_CEILING_CONTEXT_FIELDS,
    ACTION_CEILING_HORIZONS,
    S_FAMILY_BUDGET_PULSE_ARMS,
    S_FAMILY_BUDGET_PULSE_PROFILE,
    S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION,
    actionability_delta,
)
from arac.policy.evidence_overlay import RelationKey


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "exp_030_s_family_budget_pulse"

PROTOCOL_VERSION = S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION
EXPERIMENT_ID = "exp_030_s_family_budget_pulse"
WORKER_MODULE = "experiments.pilots.exp_019_conflict_resolution_pilot._diagnostic_worker"
TRAJECTORY_ID = f"{EXPERIMENT_ID}-s5-seed117"
COHORT = "real_aob"
CASE = "S5"
SEED = 117
CONFIGURED_MAX_FES = 300_000
TERMINAL_FE_POLICY = "native_population_aligned"
DEFAULT_JOBS = 1
EXPECTED_CONTEXTS = 4
EXPECTED_GROUPS = 20
EXPECTED_ARM_ROWS = (
    EXPECTED_CONTEXTS * len(S_FAMILY_BUDGET_PULSE_ARMS) * len(ACTION_CEILING_HORIZONS)
)
AOB_DATA_ROOT = REPOSITORY_ROOT / "vendor" / "hcc" / "AOB" / "AOBG" / "datafile"
AOB_INPUT_MANIFEST_FIELDS = (
    "problem_id",
    "file",
    "path",
    "sha256_before",
    "sha256_after",
    "unchanged",
)
GCB_CONTEXT_FIELDS = (
    "gcb_action_hash",
    "gcb_action_payload",
    "gcb_initial_mean_hash",
    "gcb_parameter_hash",
    "gcb_optimizer_seed",
    "gcb_population_size",
    "gcb_budget_fes",
    "gcb_acceptance_fitness",
)
BUDGET_ARMS = (
    FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
    SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
)
SOURCE_FILES = (
    Path(__file__).resolve(),
    REPOSITORY_ROOT / "src" / "arac" / "policy" / "action_ceiling.py",
    REPOSITORY_ROOT / "src" / "arac" / "actions" / "budget_reallocation.py",
    REPOSITORY_ROOT / "src" / "arac" / "actions" / "shrunk_budget_pulse.py",
    REPOSITORY_ROOT / "src" / "arac" / "backends" / "hcc_action_ceiling.py",
    REPOSITORY_ROOT / "src" / "arac" / "backends" / "hcc_action_ceiling_runtime.py",
    REPOSITORY_ROOT / "scripts" / "hcc_smoke_runner.py",
    REPOSITORY_ROOT
    / "experiments"
    / "pilots"
    / "exp_019_conflict_resolution_pilot"
    / "_diagnostic_worker.py",
)


@dataclass(frozen=True)
class ValidatedArtifacts:
    artifact_dir: Path
    context_rows: tuple[dict[str, str], ...]
    arm_rows: tuple[dict[str, str], ...]
    aob_input_rows: tuple[dict[str, str], ...]
    run_summary: dict[str, object]
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ContextState:
    row: Mapping[str, str]
    populations: tuple[int, ...]
    uniform_budgets: tuple[int, ...]
    horizon_fe: int
    raw_action: BudgetAllocationAction
    shrunk_action: ShrunkEfficiencyBudgetPulseAction


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    _require(isinstance(payload, dict), f"JSON artifact must be an object: {path}")
    return payload


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(
                tuple(reader.fieldnames or ()) == tuple(fields),
                f"CSV schema mismatch: {path}",
            )
            return list(reader)
    except OSError as error:
        raise ValueError(f"missing or unreadable CSV artifact: {path}") from error


def _int(value: object, field: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    _require(str(parsed) == str(value), f"{field} must use canonical integer text")
    _require(parsed >= minimum, f"{field} must be >= {minimum}")
    return parsed


def _float(value: object, field: str, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    _require(math.isfinite(parsed) and parsed >= minimum, f"{field} is invalid")
    return parsed


def _json_list(row: Mapping[str, str], field: str) -> list[object]:
    try:
        value = json.loads(str(row.get(field, "")))
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON") from error
    _require(isinstance(value, list), f"{field} must be a JSON list")
    return value


def _integer_vector(
    row: Mapping[str, str],
    field: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    values = _json_list(row, field)
    _require(
        all(
            not isinstance(value, bool) and isinstance(value, int) and value >= minimum
            for value in values
        ),
        f"{field} contains invalid integers",
    )
    return tuple(int(value) for value in values)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    config = _read_json(path)
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "protocol changed")
    _require(config.get("experiment_id") == EXPERIMENT_ID, "experiment id changed")
    _require(
        config.get("stage") == "action_validation_mechanical_smoke",
        "stage changed",
    )
    execution = config.get("execution")
    _require(isinstance(execution, dict), "execution config is missing")
    expected_execution = {
        "profile": S_FAMILY_BUDGET_PULSE_PROFILE,
        "cohort": COHORT,
        "case": CASE,
        "seed": SEED,
        "max_fes": CONFIGURED_MAX_FES,
        "terminal_fe_policy": TERMINAL_FE_POLICY,
        "jobs": DEFAULT_JOBS,
        "expected_contexts": EXPECTED_CONTEXTS,
        "expected_arm_rows": EXPECTED_ARM_ROWS,
        "arms": list(S_FAMILY_BUDGET_PULSE_ARMS),
        "horizons": list(ACTION_CEILING_HORIZONS),
        "worker_module": WORKER_MODULE,
        "aob_data_root": "vendor/hcc/AOB/AOBG/datafile",
    }
    _require(execution == expected_execution, "execution matrix changed")
    authorization = config.get("authorization")
    _require(
        authorization
        == {
            "runtime_authorized": 0,
            "selector_authorized": 0,
            "inference_authorized": 0,
            "action_gate_authorized": 0,
            "primary_recommendation": "mechanical_smoke_only",
        },
        "authorization gate changed",
    )
    return config


def trajectory_artifact_dir(output_root: Path) -> Path:
    return output_root.resolve() / TRAJECTORY_ID / "schwefel"


def build_worker_command(output_root: Path, python_executable: str) -> tuple[str, ...]:
    return (
        python_executable,
        "-m",
        WORKER_MODULE,
        "--cohort",
        COHORT,
        "--case",
        CASE,
        "--seed",
        str(SEED),
        "--max-fes",
        str(CONFIGURED_MAX_FES),
        "--output-root",
        str(output_root.resolve()),
        "--timestamp",
        TRAJECTORY_ID,
        "--profile",
        S_FAMILY_BUDGET_PULSE_PROFILE,
    )


def run_worker(output_root: Path, python_executable: str) -> None:
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )
    subprocess.run(
        build_worker_command(output_root, python_executable),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def _context_state(row: Mapping[str, str]) -> ContextState:
    populations = _integer_vector(row, "population_sizes", minimum=1)
    uniform = _integer_vector(row, "uniform_group_budgets", minimum=1)
    streaks = _integer_vector(row, "stagnation_streaks", minimum=0)
    efficiency_values = _json_list(row, "efficiency_ewma")
    _require(
        len(populations)
        == len(uniform)
        == len(streaks)
        == len(efficiency_values)
        == EXPECTED_GROUPS,
        "S5 context must contain exactly 20 aligned groups",
    )
    _require(
        all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in efficiency_values
        ),
        "efficiency_ewma contains invalid values",
    )
    efficiencies = tuple(float(value) for value in efficiency_values)
    _require(
        all(budget >= population for budget, population in zip(uniform, populations, strict=True)),
        "uniform budget does not cover one population",
    )
    shared_vectors = tuple(
        tuple(float(value) for value in _json_list(row, field))
        for field in ("anchor_values", "left_values", "right_values", "bridge_values")
    )
    _require(
        bool(shared_vectors[0])
        and all(len(vector) == len(shared_vectors[0]) for vector in shared_vectors)
        and all(math.isfinite(value) for vector in shared_vectors for value in vector),
        "shared-value context is invalid",
    )
    try:
        bridge_weights = json.loads(str(row.get("bridge_weights", "")))
    except json.JSONDecodeError as error:
        raise ValueError("bridge_weights is invalid JSON") from error
    _require(
        isinstance(bridge_weights, dict)
        and set(bridge_weights) == {"left_owner", "right_owner"}
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
            for value in bridge_weights.values()
        )
        and math.isclose(
            math.fsum(float(value) for value in bridge_weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "bridge weights are invalid",
    )

    phase_boundary_fe = _int(row.get("phase_boundary_fe"), "phase_boundary_fe")
    dispatch_fe = _int(row.get("dispatch_fe"), "dispatch_fe")
    issued_sweep = _int(row.get("issued_sweep"), "issued_sweep")
    target_sweep = _int(row.get("target_sweep"), "target_sweep")
    group_index = _int(row.get("group_index"), "group_index")
    horizon_fe = _int(row.get("horizon_fe"), "horizon_fe", minimum=1)
    _require(dispatch_fe >= phase_boundary_fe, "dispatch precedes the Phase1 boundary")
    _require(target_sweep == issued_sweep + 1, "context target sweep is not next")
    _require(group_index < EXPECTED_GROUPS, "context group index is invalid")
    _int(row.get("completed_efficiency_sweeps"), "completed_efficiency_sweeps")

    raw_action = freeze_efficiency_budget_action(
        problem_id=CASE,
        run_seed=SEED,
        checkpoint_fe=dispatch_fe,
        dispatch_checkpoint_hash=str(row["dispatch_checkpoint_hash"]),
        source_efficiency_ewma=efficiencies,
        population_sizes=populations,
        uniform_group_budgets=uniform,
        issued_sweep=target_sweep,
        target_sweep=target_sweep + 1,
    )
    shrunk_action = freeze_shrunk_efficiency_budget_pulse_action(
        problem_id=CASE,
        run_seed=SEED,
        checkpoint_fe=dispatch_fe,
        dispatch_checkpoint_hash=str(row["dispatch_checkpoint_hash"]),
        raw_group_budgets=raw_action.group_budgets,
        population_sizes=populations,
        uniform_group_budgets=uniform,
        issued_sweep=target_sweep,
        target_sweep=target_sweep + 1,
    )
    _require(
        shrunk_action.raw_group_budgets == raw_action.group_budgets
        and shrunk_action.group_budgets
        == allocate_shrunk_efficiency_budgets(
            raw_action.group_budgets,
            uniform,
            populations,
        ),
        "shrunk action is not the frozen 50/50 transform of the raw action",
    )
    return ContextState(row, populations, uniform, horizon_fe, raw_action, shrunk_action)


def _relation_key(relation_id: str) -> RelationKey:
    match = re.fullmatch(r"g(\d+)-(\d+):v(\d+(?:-\d+)*)", relation_id)
    _require(match is not None, "relation_id does not use the runtime format")
    assert match is not None
    owners = (int(match.group(1)), int(match.group(2)))
    shared = tuple(int(value) for value in match.group(3).split("-"))
    _require(
        owners[0] < owners[1]
        and owners[1] < EXPECTED_GROUPS
        and len(shared) == len(set(shared))
        and all(0 <= value < 1000 for value in shared),
        "relation_id contains invalid S5 owners or shared variables",
    )
    return RelationKey(owners, shared)


def _validate_context_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, ContextState]:
    _require(len(rows) == EXPECTED_CONTEXTS, "expected exactly four S5 contexts")
    contexts: dict[str, ContextState] = {}
    relation_ids: set[str] = set()
    action_set_hashes: set[str] = set()
    dispatch_hashes: set[str] = set()
    dispatch_anchor_hashes: set[str] = set()
    for row in rows:
        context_id = str(row.get("context_id", ""))
        relation_id = str(row.get("relation_id", ""))
        _require(
            row.get("protocol_version") == PROTOCOL_VERSION
            and row.get("cohort") == COHORT
            and row.get("problem_id") == CASE
            and row.get("seed") == str(SEED)
            and row.get("native_parity") == "1"
            and row.get("runtime_authorized") == "0"
            and row.get("status") == "complete"
            and not row.get("invalidation_reason")
            and context_id
            and relation_id
            and row.get("selector_arm") in ACTION_CEILING_ARMS
            and bool(row.get("selector_reason")),
            "S5 context truth contract is invalid",
        )
        _require(
            all(
                _is_sha256(row.get(field))
                for field in (
                    "action_set_hash",
                    "checkpoint_hash",
                    "dispatch_checkpoint_hash",
                    "dispatch_anchor_hash",
                )
            ),
            "S5 context hash is invalid",
        )
        _require(
            not any(row.get(field) for field in GCB_CONTEXT_FIELDS),
            "S budget-pulse context contains forbidden GCB state",
        )
        relation = _relation_key(relation_id)
        expected_context_id = (
            f"{COHORT}:{CASE}:seed{SEED}:s{row['target_sweep']}:"
            f"g{relation.owner_group_indices[0]}-{relation.owner_group_indices[1]}:"
            f"{row['dispatch_checkpoint_hash'][:12]}"
        )
        _require(
            _int(row.get("group_index"), "group_index") == relation.owner_group_indices[1]
            and context_id == expected_context_id,
            "context relation/identity chain is invalid",
        )
        _require(context_id not in contexts, "duplicate context_id")
        _require(relation_id not in relation_ids, "duplicate relation_id")
        _require(row["action_set_hash"] not in action_set_hashes, "duplicate action_set_hash")
        _require(
            row["dispatch_checkpoint_hash"] not in dispatch_hashes,
            "duplicate dispatch checkpoint hash",
        )
        _require(
            row["dispatch_anchor_hash"] not in dispatch_anchor_hashes,
            "duplicate dispatch anchor hash",
        )
        contexts[context_id] = _context_state(row)
        relation_ids.add(relation_id)
        action_set_hashes.add(row["action_set_hash"])
        dispatch_hashes.add(row["dispatch_checkpoint_hash"])
        dispatch_anchor_hashes.add(row["dispatch_anchor_hash"])
    return contexts


def _traces(
    row: Mapping[str, str],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    names = (
        "execution_sweep_trace",
        "execution_order_trace",
        "group_budget_trace",
        "execution_start_fe_trace",
    )
    vectors = tuple(
        _integer_vector(
            row,
            name,
            minimum=0 if name != "group_budget_trace" and name != "execution_start_fe_trace" else 1,
        )
        for name in names
    )
    _require(len({len(vector) for vector in vectors}) == 1, "continuation traces are misaligned")
    _require(bool(vectors[0]), "S continuation trace must not be empty")
    sweeps, order, budgets, starts = vectors
    _require(starts == tuple(sorted(set(starts))), "continuation start FE trace is invalid")
    _require(starts[0] == 1, "S continuation must begin at relative FE 1")
    _require(
        all(
            2 <= next_start - start <= budget + 1
            for start, next_start, budget in zip(
                starts[:-1],
                starts[1:],
                budgets[:-1],
                strict=True,
            )
        ),
        "continuation start FE interval exceeds its requested group budget",
    )
    _require(
        all(next_sweep in {sweep, sweep + 1} for sweep, next_sweep in zip(sweeps, sweeps[1:])),
        "continuation sweep trace is discontinuous",
    )
    return sweeps, order, budgets, starts


def _validate_schedule(
    traces: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    state: ContextState,
    action: BudgetAllocationAction | ShrunkEfficiencyBudgetPulseAction | None,
) -> None:
    sweeps, order, budgets, _starts = traces
    for index, (sweep, group, budget) in enumerate(zip(sweeps, order, budgets, strict=True)):
        _require(group < EXPECTED_GROUPS, "continuation group index is invalid")
        expected = state.uniform_budgets[group]
        if action is not None and sweep == action.target_sweep:
            expected = action.group_budgets[group]
        _require(budget == expected, "continuation budget trace changed its frozen allocation")
        if index and sweep == sweeps[index - 1]:
            _require(group > order[index - 1], "continuation changed native group order")
    for sweep in set(sweeps):
        positions = tuple(index for index, value in enumerate(sweeps) if value == sweep)
        groups = tuple(order[index] for index in positions)
        _require(len(groups) == len(set(groups)), "continuation repeats a group within one sweep")
        if len(groups) == EXPECTED_GROUPS:
            _require(set(groups) == set(range(EXPECTED_GROUPS)), "continuation lost a group")
            _require(
                sum(budgets[index] for index in positions) == sum(state.uniform_budgets),
                "complete sweep changed the frozen FE total",
            )


def _expected_dispatch_pairs(
    state: ContextState,
    count: int,
) -> tuple[tuple[int, int], ...]:
    sweep = _int(state.row["target_sweep"], "target_sweep")
    group = _int(state.row["group_index"], "group_index") + 1
    if group == EXPECTED_GROUPS:
        sweep += 1
        group = 0
    pairs: list[tuple[int, int]] = []
    for _ in range(count):
        pairs.append((sweep, group))
        group += 1
        if group == EXPECTED_GROUPS:
            sweep += 1
            group = 0
    return tuple(pairs)


def _validate_horizon_prefixes(
    rows: Mapping[str, Mapping[str, str]],
    state: ContextState,
) -> None:
    traces = {horizon: _traces(rows[horizon]) for horizon in ACTION_CEILING_HORIZONS}
    full = traces["sweep_3"]
    _require(
        tuple(zip(full[0], full[1], strict=True)) == _expected_dispatch_pairs(state, len(full[0])),
        "sweep_3 dispatch trace is not the frozen three-cycle order",
    )
    targets = {"immediate": 1, "sweep_1": state.horizon_fe, "sweep_3": 3 * state.horizon_fe}
    for horizon, target_relative_fe in targets.items():
        expected_length = sum(start <= target_relative_fe for start in full[3])
        current = traces[horizon]
        _require(
            len(current[0]) == expected_length
            and all(
                current[trace_index] == full[trace_index][:expected_length]
                for trace_index in range(4)
            ),
            "action-ceiling horizons are not exact prefixes of one continuation",
        )
    _require(
        len({rows[horizon].get("natural_endpoint_fe") for horizon in ACTION_CEILING_HORIZONS}) == 1,
        "natural endpoint changed across horizons of one branch",
    )


def _parse_budget_lifecycle(
    row: Mapping[str, str],
    expected: BudgetAllocationAction | ShrunkEfficiencyBudgetPulseAction,
) -> BudgetAllocationExecutionState | ShrunkBudgetPulseExecutionState:
    try:
        payload = json.loads(str(row.get("action_lifecycle_payload", "")))
    except json.JSONDecodeError as error:
        raise ValueError("budget lifecycle payload is invalid JSON") from error
    _require(
        isinstance(payload, dict)
        and set(payload) == {"action", "instance", "instance_hash", "execution", "execution_hash"}
        and payload.get("action") == row.get("arm"),
        "budget lifecycle envelope is invalid",
    )
    instance_payload = payload.get("instance")
    execution_payload = payload.get("execution")
    _require(
        isinstance(instance_payload, dict) and isinstance(execution_payload, dict),
        "budget lifecycle sections must be objects",
    )
    instance_fields = dict(instance_payload)
    execution_fields = dict(execution_payload)
    _require(instance_fields.pop("action", None) == row.get("arm"), "instance action changed")
    _require(execution_fields.pop("action", None) == row.get("arm"), "execution action changed")
    if isinstance(expected, ShrunkEfficiencyBudgetPulseAction):
        _require(
            instance_fields.pop("schema", None) == SHRUNK_BUDGET_PULSE_SCHEMA,
            "shrunk action schema changed",
        )
        recorded_action = ShrunkEfficiencyBudgetPulseAction(**instance_fields)
        recorded_state = ShrunkBudgetPulseExecutionState(**execution_fields)
    else:
        recorded_action = BudgetAllocationAction(**instance_fields)
        recorded_state = BudgetAllocationExecutionState(**execution_fields)
    recorded_state.validate_for(recorded_action)
    _require(recorded_action == expected, "recorded budget action differs from frozen evidence")
    _require(recorded_state.status == "consumed", "budget action was not consumed")
    _require(payload["instance_hash"] == expected.action_hash, "instance hash changed")
    _require(
        payload["execution_hash"] == recorded_state.state_hash(recorded_action),
        "execution hash changed",
    )
    _require(
        _canonical_hash(payload) == row.get("action_lifecycle_hash"),
        "lifecycle envelope hash changed",
    )
    return recorded_state


def _validate_budget_arm(
    rows: Mapping[str, Mapping[str, str]],
    state: ContextState,
    action: BudgetAllocationAction | ShrunkEfficiencyBudgetPulseAction,
) -> None:
    arm = action.audit_payload()["action"]
    assert isinstance(arm, str)
    reference_state: BudgetAllocationExecutionState | ShrunkBudgetPulseExecutionState | None = None
    invariant_fields = (
        "action_instance_hash",
        "action_lifecycle_payload",
        "action_lifecycle_hash",
        "action_accepted",
        "action_candidate_hash",
        "action_candidate_fitness",
        "action_post_incumbent_hash",
        "optimizer_parameter_hash",
    )
    first = rows[ACTION_CEILING_HORIZONS[0]]
    _validate_horizon_prefixes(rows, state)
    for horizon in ACTION_CEILING_HORIZONS:
        row = rows[horizon]
        lifecycle = _parse_budget_lifecycle(row, action)
        if reference_state is None:
            reference_state = lifecycle
        _require(
            all(row.get(field) == first.get(field) for field in invariant_fields),
            "budget action outcome differs across horizons",
        )
        traces = _traces(row)
        _validate_schedule(traces, state, action)
        target_positions = tuple(
            index for index, sweep in enumerate(traces[0]) if sweep == action.target_sweep
        )
        expected_candidate_hash = _canonical_hash({"group_budgets": action.group_budgets})
        _require(
            _int(row.get("action_budget_fes"), "action_budget_fes") == 0
            and _int(row.get("action_actual_fes"), "action_actual_fes") == 0
            and row.get("action_instance_hash") == action.action_hash
            and row.get("action_accepted") == "1"
            and row.get("action_candidate_hash") == expected_candidate_hash
            and not row.get("action_candidate_fitness")
            and _is_sha256(row.get("action_post_incumbent_hash"))
            and row.get("optimizer_scope") == "decomposed_groups"
            and row.get("optimizer_parameter_hash") == action.parameter_hash
            and not row.get("optimizer_initial_state_hash")
            and not row.get("optimizer_final_state_hash")
            and _int(row.get("optimizer_population_size"), "optimizer_population_size") == 0
            and _int(row.get("optimizer_generation_count"), "optimizer_generation_count") == 0
            and row.get("selected_candidate") == arm
            and row.get("continuation_policy_applied") == str(int(bool(target_positions))),
            "budget arm execution contract is invalid",
        )
    assert reference_state is not None
    full_traces = _traces(rows["sweep_3"])
    target_positions = tuple(
        index for index, sweep in enumerate(full_traces[0]) if sweep == action.target_sweep
    )
    _require(
        len(target_positions) == EXPECTED_GROUPS
        and {full_traces[1][index] for index in target_positions} == set(range(EXPECTED_GROUPS)),
        "budget action target sweep was not consumed exactly once",
    )
    restored_sweep = action.target_sweep + 1
    restored_positions = tuple(
        index for index, sweep in enumerate(full_traces[0]) if sweep == restored_sweep
    )
    _require(
        len(restored_positions) == EXPECTED_GROUPS
        and {full_traces[1][index] for index in restored_positions} == set(range(EXPECTED_GROUPS))
        and all(
            full_traces[2][index] == state.uniform_budgets[full_traces[1][index]]
            for index in restored_positions
        ),
        "budget pulse did not restore one complete uniform sweep",
    )
    relative_application_fe = min(full_traces[3][index] for index in target_positions)
    _require(
        reference_state.application_fe
        == _int(state.row["dispatch_fe"], "dispatch_fe") + relative_application_fe,
        "budget lifecycle application_fe is not absolute",
    )


def _validate_arm_rows(
    rows: Sequence[Mapping[str, str]],
    contexts: Mapping[str, ContextState],
) -> None:
    _require(len(rows) == EXPECTED_ARM_ROWS, "expected exactly 48 S5 arm rows")
    grouped: dict[str, dict[str, dict[str, Mapping[str, str]]]] = {}
    relative_targets = {"immediate": 1, "sweep_1": 1, "sweep_3": 3}
    for row in rows:
        context_id = str(row.get("context_id", ""))
        _require(context_id in contexts, "arm row references an unknown context")
        state = contexts[context_id]
        arm = str(row.get("arm", ""))
        horizon = str(row.get("horizon", ""))
        _require(
            row.get("protocol_version") == PROTOCOL_VERSION
            and row.get("cohort") == COHORT
            and row.get("problem_id") == CASE
            and row.get("seed") == str(SEED)
            and arm in S_FAMILY_BUDGET_PULSE_ARMS
            and horizon in ACTION_CEILING_HORIZONS
            and row.get("runtime_authorized") == "0"
            and row.get("status") == "complete"
            and not row.get("invalidation_reason")
            and row.get("counterfactual_applied") in {"0", "1"}
            and row.get("continuation_policy_applied") in {"0", "1"}
            and row.get("action_accepted") in {"0", "1"},
            "S5 arm truth contract is invalid",
        )
        target_relative_fe = (
            1 if horizon == "immediate" else relative_targets[horizon] * state.horizon_fe
        )
        traces = _traces(row)
        _require(
            _int(row.get("target_fe"), "target_fe")
            == _int(state.row["dispatch_fe"], "dispatch_fe") + target_relative_fe
            and _int(row.get("natural_endpoint_fe"), "natural_endpoint_fe")
            >= _int(state.row["dispatch_fe"], "dispatch_fe") + 3 * state.horizon_fe
            and all(start <= target_relative_fe for start in traces[3]),
            "S5 arm horizon FE contract is invalid",
        )
        mutation = _float(row.get("mutation_norm"), "mutation_norm")
        mean_mutation = _float(
            row.get("optimizer_mean_mutation_norm"),
            "optimizer_mean_mutation_norm",
        )
        action_actual = _int(row.get("action_actual_fes"), "action_actual_fes")
        expected_applied = str(
            int(
                mutation > 0.0
                or mean_mutation > 0.0
                or action_actual > 0
                or row.get("continuation_policy_applied") == "1"
            )
        )
        _require(
            row.get("counterfactual_applied") == expected_applied
            and _int(row.get("warm_start_trigger_count"), "warm_start_trigger_count") == 0
            and _float(
                row.get("warm_start_mean_shift_norm"),
                "warm_start_mean_shift_norm",
            )
            == 0.0,
            "S5 arm counterfactual flags are invalid",
        )
        bucket = grouped.setdefault(context_id, {}).setdefault(arm, {})
        _require(horizon not in bucket, "duplicate context/arm/horizon row")
        bucket[horizon] = row

    expected_keys = set(ACTION_CEILING_HORIZONS)
    for context_id, state in contexts.items():
        arms = grouped.get(context_id, {})
        _require(set(arms) == set(S_FAMILY_BUDGET_PULSE_ARMS), "S5 context arm set changed")
        _require(
            all(set(arms[arm]) == expected_keys for arm in S_FAMILY_BUDGET_PULSE_ARMS),
            "S5 context horizon set changed",
        )
        raw_rows = arms[FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION]
        shrunk_rows = arms[SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION]
        _validate_budget_arm(raw_rows, state, state.raw_action)
        _validate_budget_arm(shrunk_rows, state, state.shrunk_action)
        _validate_horizon_prefixes(arms["native_eq8"], state)
        _validate_horizon_prefixes(arms["true_no_writeback"], state)
        native_post_hashes = {
            arms["native_eq8"][horizon].get("action_post_incumbent_hash")
            for horizon in ACTION_CEILING_HORIZONS
        }
        _require(
            len(native_post_hashes) == 1 and _is_sha256(next(iter(native_post_hashes))),
            "native post-incumbent hash changed across horizons",
        )
        for arm in S_FAMILY_BUDGET_PULSE_ARMS:
            native_errors = tuple(
                _float(arms[arm][horizon].get("native_error"), "native_error")
                for horizon in ACTION_CEILING_HORIZONS
            )
            arm_errors = tuple(
                _float(arms[arm][horizon].get("arm_error"), "arm_error")
                for horizon in ACTION_CEILING_HORIZONS
            )
            _require(
                all(
                    later <= earlier
                    for earlier, later in zip(
                        native_errors,
                        native_errors[1:],
                        strict=False,
                    )
                )
                and all(
                    later <= earlier
                    for earlier, later in zip(
                        arm_errors,
                        arm_errors[1:],
                        strict=False,
                    )
                ),
                "horizon errors violate best-so-far monotonicity",
            )

        for horizon in ACTION_CEILING_HORIZONS:
            native = arms["native_eq8"][horizon]
            no_writeback = arms["true_no_writeback"][horizon]
            native_traces = _traces(native)
            no_writeback_traces = _traces(no_writeback)
            _validate_schedule(native_traces, state, None)
            _validate_schedule(no_writeback_traces, state, None)
            _require(
                native_traces == no_writeback_traces,
                "native and true-no-writeback continuation schedules differ",
            )
            for row, selected, post_hash_required in (
                (native, "native_eq8", True),
                (no_writeback, "current", False),
            ):
                _require(
                    _int(row.get("action_budget_fes"), "action_budget_fes") == 0
                    and _int(row.get("action_actual_fes"), "action_actual_fes") == 0
                    and not row.get("action_instance_hash")
                    and not row.get("action_lifecycle_payload")
                    and not row.get("action_lifecycle_hash")
                    and row.get("action_accepted") == "0"
                    and not row.get("action_candidate_hash")
                    and not row.get("action_candidate_fitness")
                    and bool(_is_sha256(row.get("action_post_incumbent_hash")))
                    == post_hash_required
                    and row.get("optimizer_scope") == "relation_writeback"
                    and not row.get("optimizer_parameter_hash")
                    and not row.get("optimizer_initial_state_hash")
                    and not row.get("optimizer_final_state_hash")
                    and _int(row.get("optimizer_population_size"), "optimizer_population_size") == 0
                    and _int(row.get("optimizer_generation_count"), "optimizer_generation_count")
                    == 0
                    and row.get("continuation_policy_applied") == "0"
                    and row.get("selected_candidate") == selected,
                    "native/no-writeback arm contains forbidden action state",
                )
            native_error = _float(native.get("native_error"), "native_error")
            _require(
                _float(native.get("arm_error"), "arm_error") == native_error
                and float(str(native.get("delta"))) == 0.0,
                "native is not its own zero-delta reference",
            )
            native_pairs = tuple(zip(native_traces[0], native_traces[1], strict=True))
            for arm in S_FAMILY_BUDGET_PULSE_ARMS:
                row = arms[arm][horizon]
                arm_error = _float(row.get("arm_error"), "arm_error")
                _require(
                    _float(row.get("native_error"), "native_error") == native_error,
                    "paired arm changed native_error",
                )
                recorded_delta = float(str(row.get("delta")))
                _require(
                    math.isfinite(recorded_delta)
                    and abs(actionability_delta(native_error, arm_error) - recorded_delta) <= 1e-12,
                    "paired arm delta does not match raw errors",
                )
                arm_traces = _traces(row)
                arm_pairs = tuple(zip(arm_traces[0], arm_traces[1], strict=True))
                shared = min(len(native_pairs), len(arm_pairs))
                _require(
                    native_pairs[:shared] == arm_pairs[:shared],
                    "budget arm changed native dispatch order",
                )
                if arm in BUDGET_ARMS:
                    _require(
                        row.get("action_post_incumbent_hash")
                        == native.get("action_post_incumbent_hash"),
                        "budget arm changed the target relation writeback",
                    )


def _validate_aob_manifest(path: Path, aob_data_root: Path) -> tuple[dict[str, str], ...]:
    rows = _read_csv(path, AOB_INPUT_MANIFEST_FIELDS)
    expected_paths = tuple(required_aob_data_files(aob_data_root.resolve(), 5))
    expected = {item.name: item.resolve() for item in expected_paths}
    _require(len(rows) == len(expected), "AOB input manifest is incomplete")
    observed: set[str] = set()
    for row in rows:
        filename = str(row.get("file", ""))
        _require(filename in expected and filename not in observed, "AOB input file set changed")
        disk_path = Path(str(row.get("path", ""))).resolve()
        digest = _sha256_file(expected[filename])
        _require(
            row.get("problem_id") == CASE
            and disk_path == expected[filename]
            and row.get("unchanged") == "1"
            and row.get("sha256_before") == digest
            and row.get("sha256_after") == digest,
            "AOB input manifest does not match the current immutable input",
        )
        observed.add(filename)
    _require(observed == set(expected), "AOB input manifest file set changed")
    return tuple(rows)


def _validate_run_summary(
    path: Path,
    contexts: Mapping[str, ContextState],
) -> dict[str, object]:
    summary = _read_json(path)
    _require(
        summary.get("protocol_version") == "hcc-run-summary-v2"
        and summary.get("problem_id") == CASE
        and summary.get("seed") == SEED
        and summary.get("configured_max_fes") == CONFIGURED_MAX_FES,
        "worker run summary identity or configured FE budget is invalid",
    )
    population_vectors = {context.populations for context in contexts.values()}
    _require(
        len(population_vectors) == 1,
        "S5 context population schedule changed within the trajectory",
    )
    terminal_population_ceiling = max(next(iter(population_vectors)))
    expected_comparison_fe = max(1, CONFIGURED_MAX_FES - terminal_population_ceiling)
    terminal_fe = summary.get("fitness_evaluations")
    comparison_fe = summary.get("comparison_fe")
    _require(
        type(terminal_fe) is int
        and expected_comparison_fe <= terminal_fe <= CONFIGURED_MAX_FES,
        "worker terminal FE is outside the native population-aligned budget window",
    )
    _require(
        type(comparison_fe) is int and comparison_fe == expected_comparison_fe,
        "worker comparison FE does not match the frozen population ceiling",
    )
    _require(
        isinstance(summary.get("final_error"), (int, float))
        and not isinstance(summary.get("final_error"), bool)
        and math.isfinite(float(summary["final_error"]))
        and float(summary["final_error"]) >= 0.0
        and isinstance(summary.get("comparison_error"), (int, float))
        and not isinstance(summary.get("comparison_error"), bool)
        and math.isfinite(float(summary["comparison_error"]))
        and float(summary["comparison_error"]) >= 0.0
        and float(summary["final_error"]) <= float(summary["comparison_error"]),
        "worker run summary metrics are invalid",
    )
    return summary


def validate_artifacts(
    artifact_dir: Path,
    *,
    aob_data_root: Path = AOB_DATA_ROOT,
) -> ValidatedArtifacts:
    artifact_dir = artifact_dir.resolve()
    context_path = artifact_dir / f"{CASE}_action_ceiling_contexts.csv"
    arm_path = artifact_dir / f"{CASE}_action_ceiling_arm_results.csv"
    aob_path = artifact_dir / f"{CASE}_aob_input_manifest.csv"
    summary_path = artifact_dir / "run_summary.json"
    context_rows = _read_csv(context_path, ACTION_CEILING_CONTEXT_FIELDS)
    arm_rows = _read_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS)
    contexts = _validate_context_rows(context_rows)
    _validate_arm_rows(arm_rows, contexts)
    aob_rows = _validate_aob_manifest(aob_path, aob_data_root)
    summary = _validate_run_summary(summary_path, contexts)
    return ValidatedArtifacts(
        artifact_dir=artifact_dir,
        context_rows=tuple(context_rows),
        arm_rows=tuple(arm_rows),
        aob_input_rows=aob_rows,
        run_summary=summary,
        artifact_paths=(context_path, arm_path, aob_path, summary_path),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _git_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def build_manifest(
    validated: ValidatedArtifacts,
    *,
    config_path: Path,
    output_root: Path,
    execution_mode: str,
) -> dict[str, object]:
    aggregate_context = output_root / "action_ceiling_contexts.csv"
    aggregate_arms = output_root / "action_ceiling_arm_results.csv"
    artifact_hashes = {
        str(path.relative_to(output_root)).replace("\\", "/"): _sha256_file(path)
        for path in (*validated.artifact_paths, aggregate_context, aggregate_arms)
    }
    source_hashes = {
        str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): _sha256_file(path)
        for path in SOURCE_FILES
    }
    integrity_checks = {
        "csv_schema": 1,
        "context_cardinality": 1,
        "paired_arm_cardinality": 1,
        "native_parity": 1,
        "typed_action_reconstruction": 1,
        "lifecycle_hash": 1,
        "absolute_application_fe": 1,
        "one_shot_target_sweep": 1,
        "uniform_budget_restore": 1,
        "horizon_trace_prefix": 1,
        "aob_inputs_unchanged": 1,
        "native_population_aligned_terminal_fe": 1,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "stage": "action_validation_mechanical_smoke",
        "status": "mechanical_smoke_pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "executor": "Codex",
        "git_head": _git_head(),
        "execution_mode": execution_mode,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "trajectory_id": TRAJECTORY_ID,
        "cohort": COHORT,
        "case": CASE,
        "seed": SEED,
        "profile": S_FAMILY_BUDGET_PULSE_PROFILE,
        "max_fes": CONFIGURED_MAX_FES,
        "fitness_evaluations": validated.run_summary["fitness_evaluations"],
        "fe_summary": {
            "terminal_fe_policy": TERMINAL_FE_POLICY,
            "configured_max_fes": CONFIGURED_MAX_FES,
            "observed_fitness_evaluations": validated.run_summary["fitness_evaluations"],
            "comparison_fe": validated.run_summary["comparison_fe"],
            "terminal_shortfall_fes": (
                CONFIGURED_MAX_FES - int(validated.run_summary["fitness_evaluations"])
            ),
            "population_alignment_window_fes": (
                CONFIGURED_MAX_FES - int(validated.run_summary["comparison_fe"])
            ),
        },
        "context_count": len(validated.context_rows),
        "arm_row_count": len(validated.arm_rows),
        "arms": list(S_FAMILY_BUDGET_PULSE_ARMS),
        "horizons": list(ACTION_CEILING_HORIZONS),
        "aob_inputs": list(validated.aob_input_rows),
        "artifact_sha256": artifact_hashes,
        "source_sha256": source_hashes,
        "integrity_checks": integrity_checks,
        "integrity_gate_passed": 1,
        "native_parity_gate_passed": 1,
        "fe_budget_gate_passed": 1,
        "runtime_authorized": 0,
        "selector_authorized": 0,
        "inference_authorized": 0,
        "action_gate_authorized": 0,
        "primary_recommendation": "mechanical_smoke_only",
    }


def _validate_existing_aggregates(
    validated: ValidatedArtifacts,
    output_root: Path,
) -> None:
    aggregate_context = output_root / "action_ceiling_contexts.csv"
    aggregate_arms = output_root / "action_ceiling_arm_results.csv"
    _require(
        _read_csv(aggregate_context, ACTION_CEILING_CONTEXT_FIELDS) == list(validated.context_rows)
        and _read_csv(aggregate_arms, ACTION_CEILING_ARM_RESULT_FIELDS) == list(validated.arm_rows)
        and _sha256_file(aggregate_context) == _sha256_file(validated.artifact_paths[0])
        and _sha256_file(aggregate_arms) == _sha256_file(validated.artifact_paths[1]),
        "existing aggregate CSVs differ from validated worker artifacts",
    )


def _validate_existing_manifest(
    validated: ValidatedArtifacts,
    *,
    config_path: Path,
    output_root: Path,
) -> dict[str, object]:
    manifest = _read_json(output_root / "manifest.json")
    expected = build_manifest(
        validated,
        config_path=config_path,
        output_root=output_root,
        execution_mode="fresh",
    )
    stable_fields = (
        "protocol_version",
        "experiment_id",
        "stage",
        "status",
        "executor",
        "execution_mode",
        "config_path",
        "config_sha256",
        "trajectory_id",
        "cohort",
        "case",
        "seed",
        "profile",
        "max_fes",
        "fitness_evaluations",
        "fe_summary",
        "context_count",
        "arm_row_count",
        "arms",
        "horizons",
        "aob_inputs",
        "artifact_sha256",
        "source_sha256",
        "integrity_checks",
        "integrity_gate_passed",
        "native_parity_gate_passed",
        "fe_budget_gate_passed",
        "runtime_authorized",
        "selector_authorized",
        "inference_authorized",
        "action_gate_authorized",
        "primary_recommendation",
    )
    _require(
        set(manifest) == set(expected)
        and all(manifest.get(field) == expected[field] for field in stable_fields),
        "existing manifest contract or artifact hash changed",
    )
    _require(
        isinstance(manifest.get("generated_at_utc"), str)
        and bool(manifest["generated_at_utc"])
        and _is_git_hash(manifest.get("git_head")),
        "existing manifest provenance is invalid",
    )
    return manifest


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python_executable: str = sys.executable,
    jobs: int = DEFAULT_JOBS,
    reuse_existing: bool = False,
) -> dict[str, object]:
    load_config(config_path)
    _require(jobs == DEFAULT_JOBS, "exp030 is frozen to one worker")
    output_root = output_root.resolve()
    if reuse_existing:
        validated = validate_artifacts(trajectory_artifact_dir(output_root))
        _validate_existing_aggregates(validated, output_root)
        return _validate_existing_manifest(
            validated,
            config_path=config_path,
            output_root=output_root,
        )

    run_worker(output_root, python_executable)
    validated = validate_artifacts(trajectory_artifact_dir(output_root))
    aggregate_context = output_root / "action_ceiling_contexts.csv"
    aggregate_arms = output_root / "action_ceiling_arm_results.csv"
    _copy_atomic(validated.artifact_paths[0], aggregate_context)
    _copy_atomic(validated.artifact_paths[1], aggregate_arms)
    manifest = build_manifest(
        validated,
        config_path=config_path,
        output_root=output_root,
        execution_mode="fresh",
    )
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jobs", type=int, choices=(DEFAULT_JOBS,), default=DEFAULT_JOBS)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_experiment(
        config_path=args.config,
        output_root=args.output_root,
        python_executable=args.python_executable,
        jobs=args.jobs,
        reuse_existing=args.reuse_existing,
    )
    print(
        f"[{manifest['status']}] contexts={manifest['context_count']} "
        f"arm_rows={manifest['arm_row_count']} FE={manifest['fitness_evaluations']}",
        flush=True,
    )
    print(f"Manifest: {args.output_root / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
