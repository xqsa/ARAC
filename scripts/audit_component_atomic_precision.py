"""Audit the frozen two-arm component precision action-validity protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "component_precision_action_validity_v1.json"
SPEC_PATH = (
    ROOT
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-07-16-component-precision-action-validity-v1.md"
)
PROTOCOL_VERSION = "component-precision-action-validity-v1"
ARMS = ("a0_v37", "a1_precision_component_once")
STAGES = ("screen", "confirm")

BRANCH_COLUMNS = (
    "protocol_version",
    "stage",
    "pair_id",
    "problem_id",
    "seed",
    "arm",
    "fresh_optimizer_execution",
    "status",
    "result_source",
    "action_applied",
    "decision_status",
    "not_applicable_reason",
    "decision_fe",
    "component_id",
    "component_group_indices",
    "component_group_count",
    "component_shared_var_count",
    "component_horizon_requested_fe",
    "component_horizon_actual_fe",
    "terminal_target_fe",
    "terminal_observed_fe",
    "horizon_error",
    "terminal_error",
    "prefix_record_sha256",
    "checkpoint_candidate_sha256",
    "crn_descriptor_sha256",
    "component_plan_sha256",
    "normal_sigma",
    "precision_sigma",
    "public_trace_sha256",
    "terminal_record_sha256",
    "optimizer_fe_used",
    "configured_max_fes",
    "same_budget_violation",
    "component_plan_frozen",
    "mid_horizon_redispatch_count",
    "unique_h_endpoint",
    "component_horizon_complete",
    "config_sha256",
    "preregistration_sha256",
    "source_git_commit",
)
COMPONENT_COLUMNS = (
    "protocol_version",
    "stage",
    "pair_id",
    "problem_id",
    "seed",
    "applicable",
    "component_closed",
    "endpoint_sequence_match",
    "a0_horizon_error",
    "a1_horizon_error",
    "tau_H",
    "component_catastrophic",
)
SURVIVAL_COLUMNS = (
    "protocol_version",
    "stage",
    "pair_id",
    "problem_id",
    "seed",
    "applicable",
    "component_closed",
    "delayed_closed",
    "a0_s_h",
    "a1_s_h",
    "delta_s_h",
    "a0_strict_survival",
    "a1_strict_survival",
    "a0_s_d",
    "a1_s_d",
    "delta_s_d",
)
PAIR_COLUMNS = (
    "protocol_version",
    "stage",
    "pair_id",
    "problem_id",
    "seed",
    "pair_integrity",
    "applicable",
    "not_applicable_reason",
    "prefix_match",
    "checkpoint_match",
    "plan_match",
    "action_applied",
    "abstain_parity",
    "a0_terminal_error",
    "a1_terminal_error",
    "tau_T",
    "terminal_catastrophic",
)
BUDGET_COLUMNS = (
    "protocol_version",
    "stage",
    "pair_id",
    "problem_id",
    "seed",
    "arm",
    "fresh_optimizer_execution",
    "group_indices",
    "population_sizes",
    "requested_group_fes",
    "actual_group_fes",
    "applied_group_sigmas",
    "normal_sigma",
    "precision_sigma",
    "component_horizon_actual_fe",
    "component_precision_fe",
    "optimizer_fe_used",
    "configured_max_fes",
    "same_budget_violation",
    "strict_terminal_reached",
    "aob_unchanged",
    "anti_leakage_pass",
    "component_endpoint_closed",
    "delayed_endpoint_closed",
)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return tuple(reader.fieldnames), list(reader)


def _require_columns(
    header: Sequence[str], required: Sequence[str], *, artifact: str
) -> None:
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{artifact} missing required columns: {', '.join(missing)}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value: object, length: int) -> bool:
    text = str(value)
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


def _flag(row: Mapping[str, str], field: str) -> bool:
    value = row.get(field, "")
    if value not in {"0", "1"}:
        raise ValueError(f"{field} must be 0 or 1")
    return value == "1"


def _integer(row: Mapping[str, str], field: str, *, minimum: int = 0) -> int:
    try:
        value = int(row.get(field, ""))
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _number(
    row: Mapping[str, str], field: str, *, minimum: float | None = None
) -> float:
    try:
        value = float(row.get(field, ""))
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _integer_vector(row: Mapping[str, str], field: str) -> tuple[int, ...]:
    text = row.get(field, "")
    if not text:
        return ()
    try:
        values = tuple(int(value) for value in text.split(";"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a semicolon-separated integer vector") from exc
    if any(value < 0 for value in values):
        raise ValueError(f"{field} values must be non-negative")
    return values


def _number_vector(row: Mapping[str, str], field: str) -> tuple[float, ...]:
    text = row.get(field, "")
    if not text:
        return ()
    try:
        values = tuple(float(value) for value in text.split(";"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a semicolon-separated numeric vector") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field} values must be finite")
    return values


def _close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _log_ratio(comparator: float, treatment: float, *, floor: float) -> float:
    if not all(math.isfinite(value) and value >= 0.0 for value in (comparator, treatment)):
        raise ValueError("errors must be finite and non-negative")
    return math.log(max(comparator, floor) / max(treatment, floor))


def _catastrophic(comparator: float, treatment: float, *, multiplier: float) -> int:
    return int(treatment >= multiplier * max(comparator, 1e-300))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires values")
    index = max(0, math.ceil(float(probability) * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _effect_summary(
    rows: Sequence[Mapping[str, object]],
    field: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int | None]:
    values = [
        (str(row["problem_id"]), int(row["seed"]), float(row[field]))
        for row in rows
    ]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "lcb_95": None,
            "wins": 0,
            "losses": 0,
            "worst_ten_percent_cvar": None,
        }
    raw = [value for _, _, value in values]
    cases = sorted({case for case, _, _ in values})
    seeds = sorted({seed_value for _, seed_value, _ in values})
    lcb: float | None = None
    if len(cases) >= 2 and len(seeds) >= 2 and int(resamples) > 0:
        rng = random.Random(int(seed))
        samples: list[float] = []
        for _ in range(int(resamples)):
            case_weights = Counter(rng.choices(cases, k=len(cases)))
            seed_weights = Counter(rng.choices(seeds, k=len(seeds)))
            weighted_sum = 0.0
            total_weight = 0
            for case, seed_value, value in values:
                weight = case_weights[case] * seed_weights[seed_value]
                weighted_sum += weight * value
                total_weight += weight
            if total_weight:
                samples.append(weighted_sum / total_weight)
        if samples:
            lcb = _quantile(samples, 0.05)
    tail_count = max(1, math.ceil(0.1 * len(raw)))
    return {
        "n": len(raw),
        "mean": statistics.fmean(raw),
        "median": statistics.median(raw),
        "lcb_95": lcb,
        "wins": sum(value > 0.0 for value in raw),
        "losses": sum(value < 0.0 for value in raw),
        "worst_ten_percent_cvar": statistics.fmean(sorted(raw)[:tail_count]),
    }


def clopper_pearson_upper(
    event_count: int,
    sample_count: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Return the one-sided exact binomial upper confidence bound."""
    events = int(event_count)
    total = int(sample_count)
    if total < 0 or events < 0 or events > total:
        raise ValueError("invalid binomial counts")
    if total == 0 or events == total:
        return 1.0
    alpha = 1.0 - float(confidence)
    if events == 0:
        return 1.0 - alpha ** (1.0 / total)

    def cdf(probability: float) -> float:
        return sum(
            math.comb(total, index)
            * probability**index
            * (1.0 - probability) ** (total - index)
            for index in range(events + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _load_config() -> dict[str, object]:
    config = _read_json(CONFIG_PATH)
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("config protocol mismatch")
    if config.get("offline_only") is not True:
        raise ValueError("config must remain offline-only")
    if config.get("runtime_scheduler_authorized") is not False:
        raise ValueError("config cannot authorize runtime")
    if tuple(config.get("arms", ())) != ARMS:
        raise ValueError("config arms mismatch")
    action = config.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("action config missing")
    forbidden_fields = ("probe", "response_gate", "group_mask", "response_arms")
    if any(field in action for field in forbidden_fields):
        raise ValueError("phase one cannot declare a probe, gate, or mask")
    if action.get("dose") != "all_component_groups_once":
        raise ValueError("phase one dose must cover every component group once")
    if action.get("execution") != "trajectory_branch_local_sequential_component_horizon":
        raise ValueError("component horizon execution contract mismatch")
    if action.get("dose_budget_crn_frozen_at_component_start") is not True:
        raise ValueError("component dose, budget, and CRN must be frozen")
    if action.get("mid_horizon_precision_redispatch_allowed") is not False:
        raise ValueError("mid-horizon precision redispatch must be disabled")
    return config


def _artifact_names(config: Mapping[str, object]) -> dict[str, str]:
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts config missing")
    required = {"branches", "component_outcomes", "survival", "pairs", "budget", "gate"}
    if set(artifacts) != required:
        raise ValueError("artifact names mismatch")
    return {name: str(artifacts[name]) for name in required}


def _unique_by_pair(
    rows: Sequence[dict[str, str]], *, artifact: str
) -> tuple[dict[str, dict[str, str]], list[str]]:
    indexed: dict[str, dict[str, str]] = {}
    blockers: list[str] = []
    for row in rows:
        pair_id = row.get("pair_id", "")
        if not pair_id or pair_id in indexed:
            blockers.append(f"{artifact}_duplicate_or_blank_pair:{pair_id}")
        else:
            indexed[pair_id] = row
    return indexed, blockers


def _validate_matrix(
    pairs: Sequence[dict[str, str]], *, stage: str, config: Mapping[str, object]
) -> list[str]:
    stage_config = config[stage]
    assert isinstance(stage_config, Mapping)
    expected = {
        (str(problem_id), int(seed))
        for problem_id in stage_config["cases"]
        for seed in stage_config["seeds"]
    }
    observed: list[tuple[str, int]] = []
    blockers: list[str] = []
    for row in pairs:
        try:
            observed.append((row.get("problem_id", ""), _integer(row, "seed")))
        except ValueError as exc:
            blockers.append(f"invalid_matrix_identity:{exc}")
    if len(observed) != len(set(observed)):
        blockers.append("duplicate_case_seed_pair")
    if set(observed) != expected:
        blockers.append("frozen_matrix_mismatch")
    return blockers


def _validate_inputs(
    *,
    branches: Sequence[dict[str, str]],
    components: Sequence[dict[str, str]],
    survival: Sequence[dict[str, str]],
    pairs: Sequence[dict[str, str]],
    budgets: Sequence[dict[str, str]],
    stage: str,
    config: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    blockers = _validate_matrix(pairs, stage=stage, config=config)
    component_by_pair, duplicate = _unique_by_pair(components, artifact="component")
    blockers.extend(duplicate)
    survival_by_pair, duplicate = _unique_by_pair(survival, artifact="survival")
    blockers.extend(duplicate)
    pair_by_id, duplicate = _unique_by_pair(pairs, artifact="pairs")
    blockers.extend(duplicate)
    branches_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    budgets_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in branches:
        branches_by_pair[row.get("pair_id", "")].append(row)
    for row in budgets:
        budgets_by_pair[row.get("pair_id", "")].append(row)
    known = set(pair_by_id)
    unknown = (
        set(component_by_pair) | set(survival_by_pair) | set(branches_by_pair) | set(budgets_by_pair)
    ) - known
    blockers.extend(f"unknown_pair:{pair_id}" for pair_id in sorted(unknown))

    audit = config["audit"]
    assert isinstance(audit, Mapping)
    floor = float(audit["error_floor"])
    catastrophic_multiplier = float(audit["catastrophic_multiplier"])
    expected_config_hash = _file_sha256(CONFIG_PATH)
    expected_spec_hash = _file_sha256(SPEC_PATH)
    normalized: list[dict[str, object]] = []

    for pair_id, pair in pair_by_id.items():
        pair_blockers: list[str] = []
        try:
            problem_id = pair.get("problem_id", "")
            seed = _integer(pair, "seed")
            applicable = _flag(pair, "applicable")
            if pair.get("protocol_version") != PROTOCOL_VERSION or pair.get("stage") != stage:
                pair_blockers.append("pair_protocol_or_stage_mismatch")
            if not _flag(pair, "pair_integrity"):
                pair_blockers.append("source_pair_integrity_failed")
            if not applicable and not pair.get("not_applicable_reason", ""):
                pair_blockers.append("not_applicable_reason_missing")
            if applicable and pair.get("not_applicable_reason", ""):
                pair_blockers.append("applicable_has_not_applicable_reason")
            for field in ("prefix_match", "checkpoint_match", "plan_match"):
                if not _flag(pair, field):
                    pair_blockers.append(f"{field}_failed")

            arm_rows: dict[str, dict[str, str]] = {}
            for branch in branches_by_pair.get(pair_id, []):
                arm = branch.get("arm", "")
                if arm not in ARMS or arm in arm_rows:
                    pair_blockers.append("branch_arm_duplicate_or_invalid")
                else:
                    arm_rows[arm] = branch
            if set(arm_rows) != set(ARMS):
                pair_blockers.append("incomplete_branch_pair")
                raise ValueError("incomplete branch pair")
            a0, a1 = (arm_rows[arm] for arm in ARMS)
            for arm, branch in arm_rows.items():
                if (
                    branch.get("protocol_version") != PROTOCOL_VERSION
                    or branch.get("stage") != stage
                    or branch.get("problem_id") != problem_id
                    or _integer(branch, "seed") != seed
                ):
                    pair_blockers.append(f"branch_identity_mismatch:{arm}")
                if not _flag(branch, "fresh_optimizer_execution"):
                    pair_blockers.append(f"branch_not_fresh:{arm}")
                if branch.get("status") != "complete" or not branch.get("result_source", ""):
                    pair_blockers.append(f"branch_not_complete:{arm}")
                if _flag(branch, "same_budget_violation"):
                    pair_blockers.append(f"same_budget_violation:{arm}")
                if applicable and not _flag(branch, "component_plan_frozen"):
                    pair_blockers.append(f"component_plan_not_frozen:{arm}")
                if _integer(branch, "mid_horizon_redispatch_count") != 0:
                    pair_blockers.append(f"mid_horizon_redispatch_detected:{arm}")
                if applicable and not _flag(branch, "unique_h_endpoint"):
                    pair_blockers.append(f"unique_h_endpoint_missing:{arm}")
                if applicable and not _flag(branch, "component_horizon_complete"):
                    pair_blockers.append(f"component_horizon_incomplete:{arm}")
                if branch.get("config_sha256") != expected_config_hash:
                    pair_blockers.append(f"config_hash_mismatch:{arm}")
                if branch.get("preregistration_sha256") != expected_spec_hash:
                    pair_blockers.append(f"preregistration_hash_mismatch:{arm}")
                if not _is_hex(branch.get("source_git_commit", ""), 40):
                    pair_blockers.append(f"invalid_source_git_commit:{arm}")
                for field in (
                    "public_trace_sha256",
                    "terminal_record_sha256",
                ):
                    if not _is_hex(branch.get(field, ""), 64):
                        pair_blockers.append(f"invalid_{field}:{arm}")
                component_requested_fe = _integer(
                    branch, "component_horizon_requested_fe"
                )
                component_actual_fe = _integer(branch, "component_horizon_actual_fe")
                if applicable and (
                    component_requested_fe <= 0
                    or component_actual_fe <= 0
                    or component_actual_fe > component_requested_fe
                ):
                    pair_blockers.append(f"component_horizon_fe_invalid:{arm}")
                if _integer(branch, "terminal_observed_fe") != _integer(
                    branch, "terminal_target_fe"
                ):
                    pair_blockers.append(f"terminal_endpoint_not_reached:{arm}")
                if _integer(branch, "optimizer_fe_used") > _integer(
                    branch, "configured_max_fes", minimum=1
                ):
                    pair_blockers.append(f"optimizer_fe_overspend:{arm}")
                _number(branch, "horizon_error", minimum=0.0)
                _number(branch, "terminal_error", minimum=0.0)
            identity_fields = (
                "prefix_record_sha256",
                "checkpoint_candidate_sha256",
                "component_horizon_requested_fe",
                "terminal_target_fe",
            )
            for field in identity_fields:
                if a0.get(field) != a1.get(field):
                    pair_blockers.append(f"paired_{field}_mismatch")
            if applicable:
                if a0.get("decision_status") != "applicable" or a1.get("decision_status") != "applicable":
                    pair_blockers.append("applicable_branch_status_mismatch")
                if _flag(a0, "action_applied") or not _flag(a1, "action_applied"):
                    pair_blockers.append("applicable_action_application_mismatch")
                if not a0.get("component_id", "") or a0.get("component_id") != a1.get("component_id"):
                    pair_blockers.append("component_identity_mismatch")
                for field in (
                    "prefix_record_sha256",
                    "crn_descriptor_sha256",
                    "checkpoint_candidate_sha256",
                ):
                    if not _is_hex(a0.get(field, ""), 64) or a0.get(field) != a1.get(field):
                        pair_blockers.append(f"paired_{field}_mismatch")
                if not _is_hex(a0.get("component_plan_sha256", ""), 64) or a0.get(
                    "component_plan_sha256"
                ) != a1.get("component_plan_sha256"):
                    pair_blockers.append("component_plan_mismatch")
                for field in ("component_group_count", "component_shared_var_count"):
                    if _integer(a0, field, minimum=1) != _integer(a1, field, minimum=1):
                        pair_blockers.append(f"paired_{field}_mismatch")
                if a0.get("component_group_indices") != a1.get("component_group_indices"):
                    pair_blockers.append("paired_component_group_indices_mismatch")
                normal_sigma = _number(a0, "normal_sigma", minimum=0.0)
                precision_sigma = _number(a0, "precision_sigma", minimum=0.0)
                if normal_sigma <= 0.0 or not _close(precision_sigma, 0.5 * normal_sigma):
                    pair_blockers.append("component_sigma_ratio_mismatch")
                if not _close(_number(a1, "normal_sigma"), normal_sigma) or not _close(
                    _number(a1, "precision_sigma"), precision_sigma
                ):
                    pair_blockers.append("paired_component_sigma_mismatch")
            else:
                if _flag(a0, "action_applied") or _flag(a1, "action_applied"):
                    pair_blockers.append("non_applicable_action_applied")
                if a0.get("decision_status") != "not_applicable" or a1.get(
                    "decision_status"
                ) != "not_applicable":
                    pair_blockers.append("non_applicable_branch_status_mismatch")
                if (
                    a0.get("horizon_error") != a1.get("horizon_error")
                    or a0.get("terminal_error") != a1.get("terminal_error")
                    or a0.get("public_trace_sha256") != a1.get("public_trace_sha256")
                ):
                    pair_blockers.append("non_applicable_not_bit_equivalent")

            budget_rows: dict[str, dict[str, str]] = {}
            for budget in budgets_by_pair.get(pair_id, []):
                arm = budget.get("arm", "")
                if arm not in ARMS or arm in budget_rows:
                    pair_blockers.append("budget_arm_duplicate_or_invalid")
                else:
                    budget_rows[arm] = budget
            if set(budget_rows) != set(ARMS):
                pair_blockers.append("incomplete_budget_pair")
                raise ValueError("incomplete budget pair")
            for arm, budget in budget_rows.items():
                branch = arm_rows[arm]
                if (
                    budget.get("protocol_version") != PROTOCOL_VERSION
                    or budget.get("stage") != stage
                    or budget.get("problem_id") != problem_id
                    or _integer(budget, "seed") != seed
                ):
                    pair_blockers.append(f"budget_identity_mismatch:{arm}")
                for field in (
                    "fresh_optimizer_execution",
                    "strict_terminal_reached",
                    "aob_unchanged",
                    "anti_leakage_pass",
                ):
                    if not _flag(budget, field):
                        pair_blockers.append(f"budget_{field}_failed:{arm}")
                if _flag(budget, "same_budget_violation"):
                    pair_blockers.append(f"budget_violation:{arm}")
                if _integer(budget, "optimizer_fe_used") != _integer(
                    branch, "optimizer_fe_used"
                ) or _integer(budget, "configured_max_fes") != _integer(
                    branch, "configured_max_fes"
                ):
                    pair_blockers.append(f"budget_branch_mismatch:{arm}")
                if applicable:
                    group_indices = _integer_vector(budget, "group_indices")
                    populations = _integer_vector(budget, "population_sizes")
                    requested = _integer_vector(budget, "requested_group_fes")
                    actual = _integer_vector(budget, "actual_group_fes")
                    applied_sigmas = _number_vector(budget, "applied_group_sigmas")
                    expected_count = _integer(branch, "component_group_count", minimum=1)
                    if not (
                        len(group_indices)
                        == len(populations)
                        == len(requested)
                        == len(actual)
                        == len(applied_sigmas)
                        == expected_count
                    ):
                        pair_blockers.append(f"budget_group_vector_length_mismatch:{arm}")
                    if group_indices != _integer_vector(branch, "component_group_indices"):
                        pair_blockers.append(f"budget_group_indices_mismatch:{arm}")
                    if any(population <= 0 for population in populations) or any(
                        requested_fe <= 0 or requested_fe % population != 0
                        for requested_fe, population in zip(
                            requested, populations, strict=True
                        )
                    ):
                        pair_blockers.append(f"budget_not_complete_population:{arm}")
                    if any(
                        actual_fe <= 0
                        or actual_fe > requested_fe
                        or actual_fe % population != 0
                        for actual_fe, requested_fe, population in zip(
                            actual, requested, populations, strict=True
                        )
                    ):
                        pair_blockers.append(
                            f"budget_actual_fe_not_generation_complete:{arm}"
                        )
                    normal_sigma = _number(budget, "normal_sigma", minimum=0.0)
                    precision_sigma = _number(budget, "precision_sigma", minimum=0.0)
                    if not _close(precision_sigma, 0.5 * normal_sigma):
                        pair_blockers.append(f"budget_sigma_ratio_mismatch:{arm}")
                    expected_sigma = precision_sigma if arm == ARMS[1] else normal_sigma
                    if any(not _close(value, expected_sigma) for value in applied_sigmas):
                        pair_blockers.append(f"budget_component_dose_mismatch:{arm}")
                    horizon_fe = _integer(
                        budget, "component_horizon_actual_fe", minimum=1
                    )
                    if horizon_fe != sum(actual) or horizon_fe != _integer(
                        branch, "component_horizon_actual_fe", minimum=1
                    ):
                        pair_blockers.append(
                            f"budget_component_horizon_actual_fe_mismatch:{arm}"
                        )
                    if sum(requested) != _integer(
                        branch, "component_horizon_requested_fe", minimum=1
                    ):
                        pair_blockers.append(
                            f"budget_component_horizon_requested_fe_mismatch:{arm}"
                        )
                    precision_fe = _integer(budget, "component_precision_fe")
                    expected_precision_fe = horizon_fe if arm == ARMS[1] else 0
                    if precision_fe != expected_precision_fe:
                        pair_blockers.append(f"budget_component_precision_fe_mismatch:{arm}")
                    if not _flag(budget, "component_endpoint_closed") or not _flag(
                        budget, "delayed_endpoint_closed"
                    ):
                        pair_blockers.append(f"budget_closure_incomplete:{arm}")
                else:
                    for field in (
                        "group_indices",
                        "population_sizes",
                        "requested_group_fes",
                        "actual_group_fes",
                        "applied_group_sigmas",
                        "normal_sigma",
                        "precision_sigma",
                    ):
                        if budget.get(field, ""):
                            pair_blockers.append(f"non_applicable_budget_vector_present:{arm}")
                    if _integer(budget, "component_horizon_actual_fe") != 0 or _integer(
                        budget, "component_precision_fe"
                    ) != 0:
                        pair_blockers.append(f"non_applicable_component_fe_nonzero:{arm}")
            if applicable:
                for field in (
                    "group_indices",
                    "population_sizes",
                    "requested_group_fes",
                    "normal_sigma",
                    "precision_sigma",
                ):
                    if budget_rows[ARMS[0]].get(field) != budget_rows[ARMS[1]].get(field):
                        pair_blockers.append(f"paired_budget_{field}_mismatch")

            component = component_by_pair.get(pair_id)
            shared = survival_by_pair.get(pair_id)
            if component is None or shared is None:
                pair_blockers.append("component_or_survival_row_missing")
                raise ValueError("component or survival row missing")
            for artifact_name, row in (("component", component), ("survival", shared)):
                if (
                    row.get("protocol_version") != PROTOCOL_VERSION
                    or row.get("stage") != stage
                    or row.get("problem_id") != problem_id
                    or _integer(row, "seed") != seed
                    or _flag(row, "applicable") != applicable
                ):
                    pair_blockers.append(f"{artifact_name}_identity_mismatch")

            a0_h = _number(a0, "horizon_error", minimum=0.0)
            a1_h = _number(a1, "horizon_error", minimum=0.0)
            a0_t = _number(a0, "terminal_error", minimum=0.0)
            a1_t = _number(a1, "terminal_error", minimum=0.0)
            tau_h = _log_ratio(a0_h, a1_h, floor=floor)
            tau_t = _log_ratio(a0_t, a1_t, floor=floor)
            cat_h = _catastrophic(a0_h, a1_h, multiplier=catastrophic_multiplier)
            cat_t = _catastrophic(a0_t, a1_t, multiplier=catastrophic_multiplier)
            if not applicable and (tau_h != 0.0 or tau_t != 0.0 or cat_h or cat_t):
                pair_blockers.append("non_applicable_effect_not_zero")

            if applicable and (
                not _flag(component, "component_closed")
                or not _flag(component, "endpoint_sequence_match")
            ):
                pair_blockers.append("component_endpoint_not_closed")
            if not applicable and _flag(component, "component_closed"):
                pair_blockers.append("non_applicable_component_marked_closed")
            component_expected = {
                "a0_horizon_error": a0_h,
                "a1_horizon_error": a1_h,
                "tau_H": tau_h,
            }
            for field, expected in component_expected.items():
                if not _close(_number(component, field), expected):
                    pair_blockers.append(f"component_recompute_mismatch:{field}")
            if int(_flag(component, "component_catastrophic")) != cat_h:
                pair_blockers.append("component_recompute_mismatch:catastrophic")

            if applicable and (
                not _flag(shared, "component_closed")
                or not _flag(shared, "delayed_closed")
            ):
                pair_blockers.append("shared_survival_closure_incomplete")
            if not applicable and (
                _flag(shared, "component_closed") or _flag(shared, "delayed_closed")
            ):
                pair_blockers.append("non_applicable_survival_marked_closed")
            delta_s_h: float | None = None
            delta_s_d: float | None = None
            if applicable:
                a0_s_h = _number(shared, "a0_s_h", minimum=0.0)
                a1_s_h = _number(shared, "a1_s_h", minimum=0.0)
                a0_strict_survival = _flag(shared, "a0_strict_survival")
                a1_strict_survival = _flag(shared, "a1_strict_survival")
                a0_s_d = _number(shared, "a0_s_d", minimum=0.0)
                a1_s_d = _number(shared, "a1_s_d", minimum=0.0)
                if any(value > 1.0 for value in (a0_s_h, a1_s_h, a0_s_d, a1_s_d)):
                    pair_blockers.append("survival_value_out_of_range")
                if a0_strict_survival != (a0_s_h > 0.0) or a1_strict_survival != (
                    a1_s_h > 0.0
                ):
                    pair_blockers.append("strict_survival_mismatch")
                delta_s_h = a1_s_h - a0_s_h
                delta_s_d = a1_s_d - a0_s_d
                if not _close(_number(shared, "delta_s_h"), delta_s_h):
                    pair_blockers.append("survival_recompute_mismatch:delta_s_h")
                if not _close(_number(shared, "delta_s_d"), delta_s_d):
                    pair_blockers.append("survival_recompute_mismatch:delta_s_d")
            elif any(
                shared.get(field, "")
                for field in (
                    "a0_s_h",
                    "a1_s_h",
                    "delta_s_h",
                    "a0_strict_survival",
                    "a1_strict_survival",
                    "a0_s_d",
                    "a1_s_d",
                    "delta_s_d",
                )
            ):
                pair_blockers.append("non_applicable_has_survival_values")

            pair_expected = {
                "a0_terminal_error": a0_t,
                "a1_terminal_error": a1_t,
                "tau_T": tau_t,
            }
            for field, expected in pair_expected.items():
                if not _close(_number(pair, field), expected):
                    pair_blockers.append(f"pair_recompute_mismatch:{field}")
            if int(_flag(pair, "terminal_catastrophic")) != cat_t:
                pair_blockers.append("pair_recompute_mismatch:catastrophic")
            if _flag(pair, "action_applied") != applicable:
                pair_blockers.append("pair_action_applied_mismatch")
            if not applicable:
                if not _flag(pair, "abstain_parity"):
                    pair_blockers.append("non_applicable_abstain_parity_failed")
            elif _flag(pair, "abstain_parity"):
                pair_blockers.append("applicable_marked_abstain_parity")

            if pair_blockers:
                blockers.extend(f"{pair_id}:{blocker}" for blocker in pair_blockers)
                continue
            normalized.append(
                {
                    "pair_id": pair_id,
                    "problem_id": problem_id,
                    "seed": seed,
                    "applicable": applicable,
                    "tau_H": tau_h,
                    "tau_T": tau_t,
                    "component_catastrophic": cat_h,
                    "terminal_catastrophic": cat_t,
                    "delta_s_h": delta_s_h,
                    "delta_s_d": delta_s_d,
                    "a1_strict_survival": a1_strict_survival
                    if applicable
                    else None,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            pair_blockers.append(f"invalid_value:{exc}")
            blockers.extend(f"{pair_id}:{blocker}" for blocker in pair_blockers)
    return normalized, sorted(set(blockers))


def _strata(
    rows: Sequence[Mapping[str, object]], key: str, field: str
) -> dict[object, float]:
    grouped: dict[object, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(float(row[field]))
    return {name: statistics.fmean(values) for name, values in grouped.items()}


def _single_case_absolute_share(rows: Sequence[Mapping[str, object]]) -> float | None:
    contribution: dict[str, float] = defaultdict(float)
    for row in rows:
        contribution[str(row["problem_id"])] += abs(float(row["tau_T"]))
    total = sum(contribution.values())
    return max(contribution.values()) / total if total > 0.0 else None


def _build_gate(
    *,
    stage: str,
    rows: Sequence[Mapping[str, object]],
    integrity_blockers: Sequence[str],
    config: Mapping[str, object],
    resamples: int,
    input_root: Path,
) -> dict[str, object]:
    audit = config["audit"]
    stage_config = config[stage]
    assert isinstance(audit, Mapping)
    assert isinstance(stage_config, Mapping)
    itt = list(rows)
    att = [row for row in rows if bool(row["applicable"])]
    base_seed = int(audit["bootstrap_seed"])
    effects = {
        "itt": {
            "tau_H": _effect_summary(itt, "tau_H", resamples=resamples, seed=base_seed),
            "tau_T": _effect_summary(itt, "tau_T", resamples=resamples, seed=base_seed + 1),
        },
        "att": {
            "tau_H": _effect_summary(att, "tau_H", resamples=resamples, seed=base_seed + 2),
            "tau_T": _effect_summary(att, "tau_T", resamples=resamples, seed=base_seed + 3),
        },
    }
    survival = {
        "delta_s_h": _effect_summary(
            att, "delta_s_h", resamples=resamples, seed=base_seed + 4
        ),
        "delta_s_d": _effect_summary(
            att, "delta_s_d", resamples=resamples, seed=base_seed + 5
        ),
    }
    component_catastrophic = sum(int(row["component_catastrophic"]) for row in itt)
    terminal_catastrophic = sum(int(row["terminal_catastrophic"]) for row in itt)
    att_component_catastrophic = sum(
        int(row["component_catastrophic"]) for row in att
    )
    att_terminal_catastrophic = sum(int(row["terminal_catastrophic"]) for row in att)
    catastrophic = {
        "component": {
            "itt_count": component_catastrophic,
            "att_count": att_component_catastrophic,
            "att_cp_ucb_95": clopper_pearson_upper(
                att_component_catastrophic, len(att)
            ),
        },
        "terminal": {
            "itt_count": terminal_catastrophic,
            "att_count": att_terminal_catastrophic,
            "att_cp_ucb_95": clopper_pearson_upper(
                att_terminal_catastrophic, len(att)
            ),
        },
    }
    seed_means = _strata(itt, "seed", "tau_T")
    case_means = _strata(itt, "problem_id", "tau_T")
    material_floor = float(audit["terminal_material_log_effect"])
    material = [row for row in att if float(row["tau_T"]) >= material_floor]
    maximum_case_share = _single_case_absolute_share(att)
    screen_cases = set(config["screen"]["cases"])
    non_screen = [row for row in itt if row["problem_id"] not in screen_cases]

    def positive(summary: Mapping[str, object], field: str = "mean") -> bool:
        value = summary.get(field)
        return value is not None and float(value) > 0.0

    def nonnegative(summary: Mapping[str, object], field: str = "median") -> bool:
        value = summary.get(field)
        return value is not None and float(value) >= 0.0

    checks: dict[str, bool] = {
        "integrity_100_percent": not integrity_blockers,
        "bootstrap_count_2000": int(resamples) == int(audit["bootstrap_count"]),
        "registered_matrix_complete": len(itt)
        == len(stage_config["cases"]) * len(stage_config["seeds"]),
        "applicable_minimum": len(att) >= int(stage_config["minimum_applicable"]),
        "applicable_case_minimum": len({row["problem_id"] for row in att})
        >= int(stage_config["minimum_applicable_cases"]),
        "all_registered_seeds_present": {int(row["seed"]) for row in itt}
        == set(int(seed) for seed in stage_config["seeds"]),
        "zero_component_catastrophic": component_catastrophic == 0,
        "zero_terminal_catastrophic": terminal_catastrophic == 0,
        "component_and_delayed_closure_100_percent": not integrity_blockers,
    }
    if stage == "screen":
        checks.update(
            {
                "itt_tau_T_mean_positive": positive(effects["itt"]["tau_T"]),
                "att_tau_T_mean_positive": positive(effects["att"]["tau_T"]),
                "itt_tau_H_mean_positive": positive(effects["itt"]["tau_H"]),
                "att_tau_H_mean_positive": positive(effects["att"]["tau_H"]),
                "itt_tau_T_median_nonnegative": nonnegative(effects["itt"]["tau_T"]),
                "att_tau_T_median_nonnegative": nonnegative(effects["att"]["tau_T"]),
                "itt_tau_H_median_nonnegative": nonnegative(effects["itt"]["tau_H"]),
                "att_tau_H_median_nonnegative": nonnegative(effects["att"]["tau_H"]),
                "positive_seed_means_minimum": sum(value > 0.0 for value in seed_means.values())
                >= int(stage_config["minimum_positive_seed_means"]),
                "material_pair_minimum": len(material)
                >= int(stage_config["minimum_material_pairs"]),
                "delta_s_h_mean_nonnegative": nonnegative(
                    survival["delta_s_h"], "mean"
                ),
                "delta_s_h_median_nonnegative": nonnegative(survival["delta_s_h"]),
                "delta_s_d_mean_nonnegative": nonnegative(
                    survival["delta_s_d"], "mean"
                ),
                "delta_s_d_median_nonnegative": nonnegative(survival["delta_s_d"]),
            }
        )
    else:
        cp_limit = float(stage_config["catastrophic_cp_ucb_max"])
        material_fraction = len(material) / len(att) if att else 0.0
        strict_positive_survival = (
            sum(bool(row["a1_strict_survival"]) for row in att) / len(att)
            if att
            else 0.0
        )
        checks.update(
            {
                "terminal_itt_lcb_positive": positive(
                    effects["itt"]["tau_T"], "lcb_95"
                ),
                "terminal_att_lcb_positive": positive(
                    effects["att"]["tau_T"], "lcb_95"
                ),
                "component_att_lcb_positive": positive(
                    effects["att"]["tau_H"], "lcb_95"
                ),
                "terminal_itt_median_nonnegative": nonnegative(
                    effects["itt"]["tau_T"]
                ),
                "terminal_att_median_nonnegative": nonnegative(
                    effects["att"]["tau_T"]
                ),
                "component_att_median_nonnegative": nonnegative(
                    effects["att"]["tau_H"]
                ),
                "catastrophic_cp_ucb_within_0_05": max(
                    catastrophic["component"]["att_cp_ucb_95"],
                    catastrophic["terminal"]["att_cp_ucb_95"],
                )
                <= cp_limit,
                "case_mean_wins_minimum": sum(value > 0.0 for value in case_means.values())
                >= int(stage_config["minimum_case_mean_wins"]),
                "all_seed_means_nonnegative": len(seed_means)
                == len(stage_config["seeds"])
                and all(value >= 0.0 for value in seed_means.values()),
                "strict_positive_seed_means_minimum": sum(
                    value > 0.0 for value in seed_means.values()
                )
                >= int(stage_config["minimum_strict_positive_seed_means"]),
                "worst_ten_percent_cvar_nonnegative": nonnegative(
                    effects["att"]["tau_T"], "worst_ten_percent_cvar"
                ),
                "material_fraction_minimum": material_fraction
                >= float(stage_config["minimum_material_fraction"]),
                "material_case_coverage_minimum": len(
                    {row["problem_id"] for row in material}
                )
                >= int(stage_config["minimum_material_cases"]),
                "material_seed_coverage_minimum": len({row["seed"] for row in material})
                >= int(stage_config["minimum_material_seeds"]),
                "delta_s_h_lcb_nonnegative": nonnegative(
                    survival["delta_s_h"], "lcb_95"
                ),
                "delta_s_h_median_nonnegative": nonnegative(survival["delta_s_h"]),
                "delta_s_d_lcb_nonnegative": nonnegative(
                    survival["delta_s_d"], "lcb_95"
                ),
                "delta_s_d_median_nonnegative": nonnegative(survival["delta_s_d"]),
                "strict_positive_survival_fraction_minimum": strict_positive_survival
                >= float(stage_config["minimum_strict_positive_survival_fraction"]),
                "single_case_absolute_effect_share_within_limit": (
                    maximum_case_share is not None
                    and maximum_case_share
                    <= float(stage_config["maximum_single_case_absolute_effect_share"])
                ),
                "non_screen_16_mean_direction_positive": len(
                    {row["problem_id"] for row in non_screen}
                )
                == 16
                and bool(non_screen)
                and statistics.fmean(float(row["tau_T"]) for row in non_screen) > 0.0,
            }
        )

    failed = sorted(name for name, passed in checks.items() if not passed)
    passed = not failed
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "status": f"{stage}_pass" if passed else f"{stage}_no_go",
        "action_validity_supported": bool(stage == "confirm" and passed),
        "runtime_scheduler_authorized": False,
        "full_24_authorized": False,
        "source_root": str(input_root.resolve()),
        "population": {
            "itt_definition": "all_registered_pairs_with_no_opportunity_as_zero_effect",
            "att_definition": "pre_action_applicable_pairs",
            "itt_count": len(itt),
            "att_count": len(att),
            "att_cases": sorted({str(row["problem_id"]) for row in att}),
            "att_seeds": sorted({int(row["seed"]) for row in att}),
        },
        "bootstrap": {
            "method": "case_by_seed_two_way_cluster",
            "resamples": int(resamples),
            "seed": base_seed,
            "one_sided_confidence": float(audit["confidence_level"]),
        },
        "effects": effects,
        "catastrophic": catastrophic,
        "survival": {
            **survival,
            "strict_positive_survival_fraction": (
                sum(bool(row["a1_strict_survival"]) for row in att) / len(att)
                if att
                else 0.0
            ),
        },
        "strata": {
            "case_tau_T_means": dict(sorted(case_means.items())),
            "seed_tau_T_means": {str(key): value for key, value in sorted(seed_means.items())},
            "maximum_single_case_absolute_effect_share": maximum_case_share,
            "material_pair_count": len(material),
            "material_case_count": len({row["problem_id"] for row in material}),
            "material_seed_count": len({row["seed"] for row in material}),
        },
        "integrity": {
            "status": "pass" if not integrity_blockers else "blocked",
            "blockers": list(integrity_blockers),
        },
        "checks": checks,
        "blockers": failed,
        "hard_stop": "action_validity_only_no_runtime_or_full24_authorization",
    }


def _blocked_gate(
    *, input_root: Path, stage: str, resamples: int, blocker: str
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "status": f"{stage}_no_go",
        "action_validity_supported": False,
        "runtime_scheduler_authorized": False,
        "full_24_authorized": False,
        "source_root": str(input_root.resolve()),
        "bootstrap": {"method": "case_by_seed_two_way_cluster", "resamples": resamples},
        "integrity": {"status": "blocked", "blockers": [blocker]},
        "checks": {"integrity_100_percent": False},
        "blockers": ["integrity_100_percent"],
        "hard_stop": "action_validity_only_no_runtime_or_full24_authorization",
    }


def audit_component_atomic_precision(
    input_root: Path, *, stage: str, resamples: int = 2000
) -> dict[str, object]:
    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    source = input_root.resolve()
    try:
        config = _load_config()
        names = _artifact_names(config)
        branch_header, branches = _read_csv(source / names["branches"])
        component_header, components = _read_csv(source / names["component_outcomes"])
        survival_header, survival = _read_csv(source / names["survival"])
        pair_header, pairs = _read_csv(source / names["pairs"])
        budget_header, budgets = _read_csv(source / names["budget"])
        _require_columns(branch_header, BRANCH_COLUMNS, artifact=names["branches"])
        _require_columns(
            component_header, COMPONENT_COLUMNS, artifact=names["component_outcomes"]
        )
        _require_columns(survival_header, SURVIVAL_COLUMNS, artifact=names["survival"])
        _require_columns(pair_header, PAIR_COLUMNS, artifact=names["pairs"])
        _require_columns(budget_header, BUDGET_COLUMNS, artifact=names["budget"])
        rows, blockers = _validate_inputs(
            branches=branches,
            components=components,
            survival=survival,
            pairs=pairs,
            budgets=budgets,
            stage=stage,
            config=config,
        )
        return _build_gate(
            stage=stage,
            rows=rows,
            integrity_blockers=blockers,
            config=config,
            resamples=resamples,
            input_root=source,
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _blocked_gate(
            input_root=source,
            stage=stage,
            resamples=resamples,
            blocker=f"input_contract_error:{exc}",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = (args.output_dir or args.input_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gate = audit_component_atomic_precision(
        args.input_dir, stage=args.stage, resamples=args.bootstrap_resamples
    )
    output_path = output_root / "component_action_gate.json"
    output_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=False))
    return 0 if gate["status"] == f"{args.stage}_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
