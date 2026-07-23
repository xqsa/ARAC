"""Offline grouped-CV diagnostic for Phase1 evidence and action-ceiling labels.

This module never runs HCC and never authorizes a runtime action.  Its primary
gate routes the frozen beneficial actions for R4 and S5 on held-out seeds.  The
complete 13-arm value regression remains a secondary diagnostic for E3/A4/R4/S5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    ACTION_CEILING_ARM_RESULT_FIELDS,
    ACTION_CEILING_CONTEXT_FIELDS,
    ACTION_CEILING_PROTOCOL_VERSION,
    ACTION_CEILING_TIE_TOLERANCE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CATASTROPHIC_DELTA,
    MATERIAL_POSITIVE_DELTA,
    PRIMARY_HORIZON,
    best_action_ceiling_arm,
)

from .diagnostic import CONFIG_PATH, load_config, validate_raw_rows


EVIDENCE_PREDICTABILITY_SCHEMA_VERSION = "exp019-evidence-predictability-v2"
ANALYSIS_STATUS = "post_hoc_exploratory"
GENERALIZATION_SCOPE = "held_out_seeds_within_fixed_R4_S5"
RUNTIME_AUTHORIZED = 0
PRIMARY_SCOPE = "R4_S5_beneficial_action_routing"
PRIMARY_PREDICTOR = "shared_count_stump"
SECONDARY_13_ARM_PREDICTOR = "ridge_value:combined"
R4_S5_TARGET_ACTIONS = {
    "R4": "gcb",
    "S5": "efficiency_budget_reallocation",
}
PRIMARY_CASES = tuple(R4_S5_TARGET_ACTIONS)
OUT_OF_SCOPE_POLICY = "not_authorized"
SAFE_REFERENCE_ARM = "efficiency_budget_reallocation"
MINIMUM_CASE_MATERIAL_POSITIVE_RATE = 0.90
FEATURE_SET_GROUPS = {
    "topology": ("topology",),
    "geometry": ("geometry",),
    "boundary_state": ("boundary_state",),
    "combined": ("topology", "geometry", "boundary_state"),
}
CONTEXT_ONLY_FEATURE_SETS = (
    "topology",
    "geometry",
    "boundary_state",
    "combined",
)
BASELINE_PREDICTORS = (
    "native_always",
    "train_sbs",
    "train_safe_sbs",
    "train_majority_winner",
    "current_g0_selector",
)
OOF_PREDICTORS = BASELINE_PREDICTORS + tuple(
    f"ridge_value:{feature_set}" for feature_set in FEATURE_SET_GROUPS
)
R4_S5_PAIRWISE_PREDICTORS = (
    "shared_count_stump",
    "l2_logistic:combined",
)
FORBIDDEN_MODEL_FIELDS = (
    "problem_id",
    "seed",
    "context_id",
    "relation_id_raw",
    "shared_variable_ids",
    "action_set_hash",
    "checkpoint_hash",
    "dispatch_checkpoint_hash",
    "dispatch_anchor_hash",
    "dispatch_fe",
    "stagnation_streaks",
    "horizon_fe",
    "gcb_action_payload",
    "gcb_optimizer_seed",
    "gcb_budget_fes",
    "gcb_acceptance_fitness",
    "selector_arm",
    "selector_reason",
    "arm_error",
    "delta",
    "action_accepted",
)
_RELATION_PATTERN = re.compile(r"^g(?P<left>\d+)-(?P<right>\d+):v(?P<shared>\d+(?:-\d+)*)$")
_EPSILON = 1e-12


@dataclass(frozen=True)
class DiagnosticDataset:
    contexts: pd.DataFrame
    deltas: pd.DataFrame
    oracle_arms: tuple[str, ...]
    oracle_deltas: np.ndarray


def _read_contract_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"CSV schema mismatch: {path}")
        return list(reader)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite_float(value: object, field: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


def _json_float_tuple(value: object, field: str) -> tuple[float, ...]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} must be a JSON array") from error
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} must be a non-empty JSON array")
    return tuple(_finite_float(item, field) for item in raw)


def _json_int_tuple(value: object, field: str) -> tuple[int, ...]:
    values = _json_float_tuple(value, field)
    if any(item < 0 or int(item) != item for item in values):
        raise ValueError(f"{field} must contain non-negative integers")
    return tuple(int(item) for item in values)


def _relation_parts(relation_id: str) -> tuple[int, int, tuple[int, ...]]:
    match = _RELATION_PATTERN.fullmatch(relation_id)
    if match is None:
        raise ValueError(f"invalid relation id: {relation_id}")
    shared = tuple(int(value) for value in match.group("shared").split("-"))
    return int(match.group("left")), int(match.group("right")), shared


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= _EPSILON else float(np.dot(left, right) / denominator)


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return 0.0 if abs(mean) <= _EPSILON else float(np.std(values) / abs(mean))


def _normalized_rank(values: np.ndarray, index: int) -> float:
    if values.size <= 1:
        return 1.0
    order = np.argsort(-values, kind="stable")
    rank = int(np.flatnonzero(order == index)[0])
    return 1.0 - rank / (values.size - 1)


def load_diagnostic_dataset(artifact_dir: Path) -> tuple[DiagnosticDataset, dict[str, Any]]:
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_version") != ACTION_CEILING_PROTOCOL_VERSION
        or manifest.get("integrity_gate", {}).get("passed") != 1
        or manifest.get("runtime_authorized") != 0
    ):
        raise ValueError("source action-ceiling manifest did not pass the v6 offline gate")

    context_path = artifact_dir / "action_ceiling_contexts.csv"
    arm_path = artifact_dir / "action_ceiling_arm_results.csv"
    expected_hashes = manifest.get("artifacts", {})
    for path in (context_path, arm_path):
        if expected_hashes.get(path.name) != _sha256_file(path):
            raise ValueError(f"source artifact hash mismatch: {path}")

    context_rows = _read_contract_csv(context_path, ACTION_CEILING_CONTEXT_FIELDS)
    arm_rows = _read_contract_csv(arm_path, ACTION_CEILING_ARM_RESULT_FIELDS)
    observations = validate_raw_rows(context_rows, arm_rows)
    primary = tuple(
        row
        for row in observations
        if row.cohort == "real_aob" and row.horizon == PRIMARY_HORIZON
    )
    if not primary:
        raise ValueError("no real-AOB sweep_1 observations")

    context_lookup = {
        row["context_id"]: row for row in context_rows if row["cohort"] == "real_aob"
    }
    context_ids = tuple(
        sorted(
            {row.context_id for row in primary},
            key=lambda item: (
                context_lookup[item]["problem_id"],
                int(context_lookup[item]["seed"]),
                item,
            ),
        )
    )
    if len(context_ids) != len(context_lookup):
        raise ValueError("context/label coverage mismatch")

    delta_lookup = {(row.context_id, row.arm): row.delta for row in primary}
    delta_values: list[list[float]] = []
    oracle_arms: list[str] = []
    oracle_deltas: list[float] = []
    for context_id in context_ids:
        deltas = {arm: delta_lookup[(context_id, arm)] for arm in ACTION_CEILING_ARMS}
        if not math.isclose(
            deltas["native_eq8"],
            0.0,
            rel_tol=0.0,
            abs_tol=ACTION_CEILING_TIE_TOLERANCE,
        ):
            raise ValueError("native arm is not the zero-Delta reference")
        oracle_arm, oracle_delta = best_action_ceiling_arm(deltas)
        delta_values.append([deltas[arm] for arm in ACTION_CEILING_ARMS])
        oracle_arms.append(oracle_arm)
        oracle_deltas.append(oracle_delta)

    contexts = pd.DataFrame([context_lookup[item] for item in context_ids])
    contexts.index = pd.Index(context_ids, name="context_id_index")
    deltas = pd.DataFrame(delta_values, index=contexts.index, columns=ACTION_CEILING_ARMS)

    cluster_counts = contexts.groupby(["problem_id", "seed"], sort=True).size()
    if cluster_counts.nunique() != 1:
        raise ValueError("case-seed context clusters are unbalanced")
    seed_cases = contexts.groupby("seed", sort=True)["problem_id"].nunique()
    if seed_cases.nunique() != 1 or len(seed_cases) < 3:
        raise ValueError("leave-one-seed-out folds do not share a complete case matrix")

    # EWMA is usable only if it is the frozen Phase1 boundary vector.  A change
    # inside a case-seed cluster would prove that dispatch-time state leaked in.
    if contexts.groupby(["problem_id", "seed"])["efficiency_ewma"].nunique().max() != 1:
        raise ValueError("efficiency EWMA changed after the Phase1 boundary")

    return (
        DiagnosticDataset(
            contexts=contexts,
            deltas=deltas,
            oracle_arms=tuple(oracle_arms),
            oracle_deltas=np.asarray(oracle_deltas, dtype=float),
        ),
        manifest,
    )


def _feature_groups(row: Mapping[str, object]) -> dict[str, dict[str, float]]:
    left_group, right_group, shared_variables = _relation_parts(str(row["relation_id"]))
    anchor = np.asarray(_json_float_tuple(row["anchor_values"], "anchor_values"))
    left = np.asarray(_json_float_tuple(row["left_values"], "left_values"))
    right = np.asarray(_json_float_tuple(row["right_values"], "right_values"))
    bridge = np.asarray(_json_float_tuple(row["bridge_values"], "bridge_values"))
    if not (len(shared_variables) == anchor.size == left.size == right.size == bridge.size):
        raise ValueError("relation and frozen Phase1 candidate dimensions differ")

    populations = np.asarray(_json_int_tuple(row["population_sizes"], "population_sizes"))
    budgets = np.asarray(
        _json_int_tuple(row["uniform_group_budgets"], "uniform_group_budgets")
    )
    efficiency = np.asarray(_json_float_tuple(row["efficiency_ewma"], "efficiency_ewma"))
    if not (populations.size == budgets.size == efficiency.size):
        raise ValueError("group-level Phase1 vectors differ in length")
    group_count = populations.size
    group_denominator = max(group_count - 1, 1)
    if not (0 <= left_group < group_count and 0 <= right_group < group_count):
        raise ValueError("relation owner is outside the grouping")

    population_mean = float(np.mean(populations))
    topology = {
        "shared_count": float(anchor.size),
        "group_count": float(group_count),
        "left_group_position": left_group / group_denominator,
        "right_group_position": right_group / group_denominator,
        "owner_group_gap": (right_group - left_group) / group_denominator,
        "target_group_position": int(row["group_index"]) / group_denominator,
        "left_population_ratio": float(populations[left_group] / population_mean),
        "right_population_ratio": float(populations[right_group] / population_mean),
        "owner_population_difference": float(
            (populations[left_group] - populations[right_group]) / population_mean
        ),
        "population_cv": _coefficient_of_variation(populations.astype(float)),
    }

    coordinate_scale = 1.0 + np.abs(anchor)
    left_step = (left - anchor) / coordinate_scale
    right_step = (right - anchor) / coordinate_scale
    bridge_step = (bridge - anchor) / coordinate_scale
    owner_difference = (left - right) / coordinate_scale
    weights = json.loads(str(row["bridge_weights"]))
    if set(weights) != {"left_owner", "right_owner"}:
        raise ValueError("bridge weights are invalid")
    left_weight = _finite_float(weights["left_owner"], "left bridge weight")
    right_weight = _finite_float(weights["right_owner"], "right bridge weight")
    if not math.isclose(left_weight + right_weight, 1.0, abs_tol=1e-12):
        raise ValueError("bridge weights do not sum to one")
    weighted_bridge = left_weight * left + right_weight * right
    midpoint = 0.5 * (left + right)
    geometry = {
        "left_step_rms": _rms(left_step),
        "right_step_rms": _rms(right_step),
        "bridge_step_rms": _rms(bridge_step),
        "left_step_mean": float(np.mean(left_step)),
        "right_step_mean": float(np.mean(right_step)),
        "bridge_step_mean": float(np.mean(bridge_step)),
        "owner_disagreement_rms": _rms(owner_difference),
        "owner_direction_cosine": _cosine(left_step, right_step),
        "owner_step_norm_balance": float(
            (_rms(left_step) - _rms(right_step))
            / (_rms(left_step) + _rms(right_step) + _EPSILON)
        ),
        "bridge_weight_imbalance": left_weight - right_weight,
        "bridge_weighted_residual_rms": _rms(
            (bridge - weighted_bridge) / coordinate_scale
        ),
        "bridge_midpoint_residual_rms": _rms((bridge - midpoint) / coordinate_scale),
    }

    efficiency_total = float(np.sum(efficiency))
    efficiency_shares = (
        efficiency / efficiency_total
        if efficiency_total > _EPSILON
        else np.full(efficiency.size, 1.0 / efficiency.size)
    )
    positive_shares = efficiency_shares[efficiency_shares > 0.0]
    normalized_entropy = -float(np.sum(positive_shares * np.log(positive_shares))) / math.log(
        efficiency.size
    )
    budget_shares = budgets / float(np.sum(budgets))
    boundary_state = {
        "efficiency_entropy": normalized_entropy,
        "efficiency_share_cv": _coefficient_of_variation(efficiency_shares),
        "efficiency_max_share": float(np.max(efficiency_shares)),
        "left_efficiency_share": float(efficiency_shares[left_group]),
        "right_efficiency_share": float(efficiency_shares[right_group]),
        "owner_efficiency_share_gap": float(
            efficiency_shares[left_group] - efficiency_shares[right_group]
        ),
        "left_efficiency_rank": _normalized_rank(efficiency, left_group),
        "right_efficiency_rank": _normalized_rank(efficiency, right_group),
        "budget_share_cv": _coefficient_of_variation(budget_shares),
        "left_budget_share": float(budget_shares[left_group]),
        "right_budget_share": float(budget_shares[right_group]),
    }

    return {
        "topology": topology,
        "geometry": geometry,
        "boundary_state": boundary_state,
    }


def build_feature_frames(dataset: DiagnosticDataset) -> dict[str, pd.DataFrame]:
    grouped_rows = [_feature_groups(row) for _, row in dataset.contexts.iterrows()]

    frames: dict[str, pd.DataFrame] = {}
    for feature_set, groups in FEATURE_SET_GROUPS.items():
        rows: list[dict[str, float]] = []
        for features in grouped_rows:
            merged: dict[str, float] = {}
            for group in groups:
                for name, value in features[group].items():
                    merged[f"{group}.{name}"] = value
            rows.append(merged)
        frame = pd.DataFrame(rows, index=dataset.contexts.index, dtype=float)
        if frame.empty or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"feature set is empty or non-finite: {feature_set}")
        if any(forbidden in name for name in frame.columns for forbidden in FORBIDDEN_MODEL_FIELDS):
            raise ValueError(f"forbidden model feature entered {feature_set}")
        frames[feature_set] = frame
    return frames


def leave_one_seed_out_splits(contexts: pd.DataFrame) -> list[tuple[int, np.ndarray, np.ndarray]]:
    seeds = sorted(int(value) for value in contexts["seed"].unique())
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    for seed in seeds:
        test = contexts["seed"].astype(int).to_numpy() == seed
        train = ~test
        if not train.any() or not test.any():
            raise ValueError("empty leave-one-seed-out fold")
        train_clusters = set(
            zip(
                contexts.loc[train, "problem_id"],
                contexts.loc[train, "seed"].astype(int),
                strict=True,
            )
        )
        test_clusters = set(
            zip(
                contexts.loc[test, "problem_id"],
                contexts.loc[test, "seed"].astype(int),
                strict=True,
            )
        )
        if train_clusters & test_clusters:
            raise ValueError("case-seed cluster crossed a fold")
        splits.append((seed, np.flatnonzero(train), np.flatnonzero(test)))
    return splits


def _case_macro_mean(values: np.ndarray, cases: np.ndarray) -> float:
    return float(np.mean([np.mean(values[cases == case]) for case in sorted(set(cases))]))


def _ordered_prediction_indices(scores: Sequence[float]) -> list[int]:
    adjusted = np.asarray(scores, dtype=float).copy()
    if adjusted.shape != (len(ACTION_CEILING_ARMS),) or not np.isfinite(adjusted).all():
        raise ValueError("predicted action values must be one finite value per arm")
    adjusted[ACTION_CEILING_ARMS.index("native_eq8")] = 0.0
    remaining = list(range(len(ACTION_CEILING_ARMS)))
    ordered: list[int] = []
    while remaining:
        best_value = max(float(adjusted[index]) for index in remaining)
        selected = next(
            index
            for index in remaining
            if math.isclose(
                float(adjusted[index]),
                best_value,
                rel_tol=0.0,
                abs_tol=ACTION_CEILING_TIE_TOLERANCE,
            )
        )
        ordered.append(selected)
        remaining.remove(selected)
    return ordered


def _predicted_arm_indices(predictions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_ordered_prediction_indices(row)[0] for row in predictions],
        dtype=int,
    )


def _true_oracle_indices(y: np.ndarray) -> np.ndarray:
    indices = []
    for row in y:
        arm, _ = best_action_ceiling_arm(dict(zip(ACTION_CEILING_ARMS, row, strict=True)))
        indices.append(ACTION_CEILING_ARMS.index(arm))
    return np.asarray(indices, dtype=int)


def _value_model(kind: str, parameter: float | int) -> Pipeline:
    if kind != "ridge_value":
        raise ValueError(f"unsupported value model: {kind}")
    estimator = Ridge(alpha=float(parameter))
    return Pipeline((("scale", StandardScaler()), ("model", estimator)))


def _choose_value_parameter(
    kind: str,
    candidates: Sequence[float | int],
    x: np.ndarray,
    y: np.ndarray,
    cases: np.ndarray,
    seeds: np.ndarray,
) -> float | int:
    scored: list[tuple[float, float, float, float | int]] = []
    for parameter in candidates:
        selected_values: list[float] = []
        selected_cases: list[str] = []
        exact: list[float] = []
        catastrophic: list[float] = []
        for held_seed in sorted(set(int(item) for item in seeds)):
            validation = seeds == held_seed
            training = ~validation
            model = _value_model(kind, parameter)
            model.fit(x[training], y[training])
            predicted = model.predict(x[validation])
            selected = _predicted_arm_indices(predicted)
            actual = y[validation, selected]
            selected_values.extend(actual.tolist())
            selected_cases.extend(cases[validation].tolist())
            exact.extend((selected == _true_oracle_indices(y[validation])).astype(float).tolist())
            catastrophic.extend((actual <= CATASTROPHIC_DELTA).astype(float).tolist())
        score = _case_macro_mean(
            np.asarray(selected_values), np.asarray(selected_cases, dtype=object)
        )
        scored.append(
            (
                score,
                -float(np.mean(catastrophic)),
                float(np.mean(exact)),
                parameter,
            )
        )
    # Prefer the smoother/larger regularization parameter on an exact score tie.
    return max(scored, key=lambda item: (item[0], item[1], item[2], float(item[3])))[3]


def _prediction_row(
    dataset: DiagnosticDataset,
    row_index: int,
    *,
    predictor: str,
    feature_set: str,
    fold_seed: int,
    predicted_arm: str,
    predicted_scores: Sequence[float] | None,
    chosen_parameter: object,
) -> dict[str, object]:
    context = dataset.contexts.iloc[row_index]
    selected_delta = float(dataset.deltas.iloc[row_index][predicted_arm])
    oracle_arm = dataset.oracle_arms[row_index]
    oracle_delta = float(dataset.oracle_deltas[row_index])
    top_two = [predicted_arm]
    if predicted_scores is not None:
        top_two = [
            ACTION_CEILING_ARMS[index]
            for index in _ordered_prediction_indices(predicted_scores)[:2]
        ]
    material_available = oracle_delta > MATERIAL_POSITIVE_DELTA
    return {
        "schema_version": EVIDENCE_PREDICTABILITY_SCHEMA_VERSION,
        "predictor": predictor,
        "feature_set": feature_set,
        "fold_seed": fold_seed,
        "problem_id": context["problem_id"],
        "seed": int(context["seed"]),
        "context_id": context["context_id"],
        "relation_id": context["relation_id"],
        "oracle_arm": oracle_arm,
        "oracle_delta": oracle_delta,
        "material_oracle_arm": oracle_arm if material_available else "native_eq8",
        "predicted_arm": predicted_arm,
        "selected_delta": selected_delta,
        "vbs_regret": oracle_delta - selected_delta,
        "exact_correct": int(predicted_arm == oracle_arm),
        "top2_correct": int(oracle_arm in top_two),
        "material_available": int(material_available),
        "material_captured": int(material_available and selected_delta > MATERIAL_POSITIVE_DELTA),
        "no_headroom_abstained": int(not material_available and predicted_arm == "native_eq8"),
        "catastrophic": int(selected_delta <= CATASTROPHIC_DELTA),
        "chosen_parameter": str(chosen_parameter),
        "predicted_deltas": (
            ""
            if predicted_scores is None
            else json.dumps(
                {
                    arm: float(value)
                    for arm, value in zip(ACTION_CEILING_ARMS, predicted_scores, strict=True)
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "runtime_authorized": RUNTIME_AUTHORIZED,
    }


def crossfit_value_model(
    dataset: DiagnosticDataset,
    features: pd.DataFrame,
    *,
    kind: str,
    candidates: Sequence[float | int],
    feature_set: str,
) -> pd.DataFrame:
    x = features.to_numpy(dtype=float)
    y = dataset.deltas.to_numpy(dtype=float)
    cases = dataset.contexts["problem_id"].to_numpy(dtype=object)
    seeds = dataset.contexts["seed"].astype(int).to_numpy()
    rows: list[dict[str, object]] = []
    for fold_seed, train, test in leave_one_seed_out_splits(dataset.contexts):
        parameter = _choose_value_parameter(
            kind,
            candidates,
            x[train],
            y[train],
            cases[train],
            seeds[train],
        )
        model = _value_model(kind, parameter)
        model.fit(x[train], y[train])
        predictions = np.asarray(model.predict(x[test]), dtype=float)
        selected = _predicted_arm_indices(predictions)
        for local_index, row_index in enumerate(test):
            rows.append(
                _prediction_row(
                    dataset,
                    int(row_index),
                    predictor=f"{kind}:{feature_set}",
                    feature_set=feature_set,
                    fold_seed=fold_seed,
                    predicted_arm=ACTION_CEILING_ARMS[int(selected[local_index])],
                    predicted_scores=predictions[local_index],
                    chosen_parameter=parameter,
                )
            )
    return pd.DataFrame(rows)


def _macro_arm_means(y: np.ndarray, cases: np.ndarray) -> dict[str, float]:
    return {
        arm: float(
            np.mean(
                [
                    np.mean(y[cases == case, arm_index])
                    for case in sorted(set(cases))
                ]
            )
        )
        for arm_index, arm in enumerate(ACTION_CEILING_ARMS)
    }


def _best_available_arm(values: Mapping[str, float]) -> str:
    if not values:
        raise ValueError("at least one available arm is required")
    best_value = max(float(value) for value in values.values())
    for arm in ACTION_CEILING_ARMS:
        if arm in values and math.isclose(
            float(values[arm]),
            best_value,
            rel_tol=0.0,
            abs_tol=ACTION_CEILING_TIE_TOLERANCE,
        ):
            return arm
    raise RuntimeError("available-arm tie-break failed")


def crossfit_baselines(dataset: DiagnosticDataset) -> pd.DataFrame:
    y = dataset.deltas.to_numpy(dtype=float)
    cases = dataset.contexts["problem_id"].to_numpy(dtype=object)
    rows: list[dict[str, object]] = []
    for fold_seed, train, test in leave_one_seed_out_splits(dataset.contexts):
        train_means = _macro_arm_means(y[train], cases[train])
        train_sbs, _ = best_action_ceiling_arm(train_means)
        safe_means = {
            arm: value
            for arm, value in train_means.items()
            if not np.any(y[train, ACTION_CEILING_ARMS.index(arm)] <= CATASTROPHIC_DELTA)
        }
        safe_sbs = _best_available_arm(safe_means)
        train_oracle = [dataset.oracle_arms[index] for index in train]
        counts = Counter(train_oracle)
        majority = max(
            ACTION_CEILING_ARMS,
            key=lambda arm: (counts[arm], -ACTION_CEILING_ARMS.index(arm)),
        )
        for row_index in test:
            current = str(dataset.contexts.iloc[row_index]["selector_arm"])
            for predictor, arm in (
                ("native_always", "native_eq8"),
                ("train_sbs", train_sbs),
                ("train_safe_sbs", safe_sbs),
                ("train_majority_winner", majority),
                ("current_g0_selector", current),
            ):
                rows.append(
                    _prediction_row(
                        dataset,
                        int(row_index),
                        predictor=predictor,
                        feature_set="baseline",
                        fold_seed=fold_seed,
                        predicted_arm=arm,
                        predicted_scores=None,
                        chosen_parameter="train_fold_only" if predictor.startswith("train_") else "fixed",
                    )
                )
    return pd.DataFrame(rows)


def _case_stratified_bootstrap(
    rows: pd.DataFrame,
    field: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    cases = sorted(rows["problem_id"].unique())
    clusters = {
        case: {
            int(cluster_seed): group[field].to_numpy(dtype=float)
            for cluster_seed, group in rows[rows["problem_id"] == case].groupby("seed")
        }
        for case in cases
    }
    samples = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        case_values = []
        for case in cases:
            seeds = sorted(clusters[case])
            selected = rng.choice(seeds, size=len(seeds), replace=True)
            case_values.append(
                float(np.mean(np.concatenate([clusters[case][int(item)] for item in selected])))
            )
        samples[replicate] = float(np.mean(case_values))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_predictions(
    oof: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    safe_baseline = oof[oof["predictor"] == "train_safe_sbs"].set_index("context_id")
    if len(safe_baseline) != oof["context_id"].nunique():
        raise ValueError("train-safe-SBS OOF baseline is incomplete")
    summaries: list[dict[str, object]] = []
    for predictor, predictor_rows in oof.groupby("predictor", sort=False):
        predictor_rows = predictor_rows.copy()
        predictor_rows["gain_over_safe_sbs"] = predictor_rows.apply(
            lambda row: float(row["selected_delta"])
            - float(safe_baseline.loc[row["context_id"], "selected_delta"]),
            axis=1,
        )
        for scope, rows in [("all", predictor_rows), *predictor_rows.groupby("problem_id")]:
            selected_mean = (
                _case_macro_mean(
                    rows["selected_delta"].to_numpy(dtype=float),
                    rows["problem_id"].to_numpy(dtype=object),
                )
                if scope == "all"
                else float(rows["selected_delta"].mean())
            )
            oracle_mean = (
                _case_macro_mean(
                    rows["oracle_delta"].to_numpy(dtype=float),
                    rows["problem_id"].to_numpy(dtype=object),
                )
                if scope == "all"
                else float(rows["oracle_delta"].mean())
            )
            if scope == "all":
                delta_lcb, delta_ucb = _case_stratified_bootstrap(
                    rows,
                    "selected_delta",
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                )
                regret_lcb, regret_ucb = _case_stratified_bootstrap(
                    rows,
                    "vbs_regret",
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + 1,
                )
                gain_lcb, gain_ucb = _case_stratified_bootstrap(
                    rows,
                    "gain_over_safe_sbs",
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + 2,
                )
            else:
                delta_lcb = delta_ucb = regret_lcb = regret_ucb = math.nan
                gain_lcb = gain_ucb = math.nan
            material_rows = rows[rows["material_available"] == 1]
            no_headroom_rows = rows[rows["material_available"] == 0]
            summaries.append(
                {
                    "schema_version": EVIDENCE_PREDICTABILITY_SCHEMA_VERSION,
                    "analysis_status": ANALYSIS_STATUS,
                    "predictor": predictor,
                    "feature_set": rows["feature_set"].iloc[0],
                    "scope": scope,
                    "context_count": len(rows),
                    "cluster_count": rows[["problem_id", "seed"]].drop_duplicates().shape[0],
                    "exact_accuracy": float(rows["exact_correct"].mean()),
                    "top2_accuracy": float(rows["top2_correct"].mean()),
                    "mean_selected_delta": selected_mean,
                    "selected_delta_lcb": delta_lcb,
                    "selected_delta_ucb": delta_ucb,
                    "mean_vbs_delta": oracle_mean,
                    "mean_vbs_regret": float(rows["vbs_regret"].mean()),
                    "vbs_regret_lcb": regret_lcb,
                    "vbs_regret_ucb": regret_ucb,
                    "mean_gain_over_train_safe_sbs": float(
                        rows["gain_over_safe_sbs"].mean()
                    ),
                    "gain_over_train_safe_sbs_lcb": gain_lcb,
                    "gain_over_train_safe_sbs_ucb": gain_ucb,
                    "vbs_utility_capture": (
                        selected_mean / oracle_mean if oracle_mean > 0.0 else math.nan
                    ),
                    "material_context_count": len(material_rows),
                    "material_recall": (
                        float(material_rows["material_captured"].mean())
                        if len(material_rows)
                        else math.nan
                    ),
                    "no_headroom_context_count": len(no_headroom_rows),
                    "no_headroom_abstain_rate": (
                        float(no_headroom_rows["no_headroom_abstained"].mean())
                        if len(no_headroom_rows)
                        else math.nan
                    ),
                    "catastrophic_count": int(rows["catastrophic"].sum()),
                    "catastrophic_rate": float(rows["catastrophic"].mean()),
                    "runtime_authorized": RUNTIME_AUTHORIZED,
                }
            )
    return pd.DataFrame(summaries)


def _choose_logistic_c(
    x: np.ndarray,
    y: np.ndarray,
    seeds: np.ndarray,
    candidates: Sequence[float],
) -> float:
    scores = []
    for candidate in candidates:
        predictions: list[int] = []
        labels: list[int] = []
        for held_seed in sorted(set(int(item) for item in seeds)):
            validation = seeds == held_seed
            training = ~validation
            model = Pipeline(
                (
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(C=float(candidate), max_iter=2000, random_state=0),
                    ),
                )
            )
            model.fit(x[training], y[training])
            predictions.extend(model.predict(x[validation]).astype(int).tolist())
            labels.extend(y[validation].astype(int).tolist())
        scores.append((balanced_accuracy_score(labels, predictions), -candidate, candidate))
    return float(max(scores)[2])


def crossfit_r4_s5_pairwise(
    dataset: DiagnosticDataset,
    features: Mapping[str, pd.DataFrame],
    *,
    logistic_cs: Sequence[float],
) -> pd.DataFrame:
    subset = dataset.contexts["problem_id"].isin(PRIMARY_CASES).to_numpy()
    row_indices = np.flatnonzero(subset)
    contexts = dataset.contexts.iloc[row_indices].reset_index(drop=True)
    deltas = dataset.deltas.iloc[row_indices].reset_index(drop=True)
    sep = deltas["gcb"].to_numpy(dtype=float)
    budget = deltas["efficiency_budget_reallocation"].to_numpy(dtype=float)
    target_actions = contexts["problem_id"].map(R4_S5_TARGET_ACTIONS).to_numpy()
    labels = (target_actions == "gcb").astype(int)
    preferred_labels = (sep > budget + ACTION_CEILING_TIE_TOLERANCE).astype(int)
    if set(labels) != {0, 1} or set(preferred_labels) != {0, 1}:
        raise ValueError("R4/S5 routing labels do not contain both classes")

    predictors: list[tuple[str, str, np.ndarray, Callable[[np.ndarray, np.ndarray, np.ndarray], Any]]] = []
    shared_count = features["topology"].iloc[row_indices][
        ["topology.shared_count"]
    ].to_numpy(dtype=float)
    predictors.append(
        (
            "shared_count_stump",
            "topology.shared_count",
            shared_count,
            lambda x, y, _seeds: DecisionTreeClassifier(
                max_depth=1,
                class_weight="balanced",
                random_state=0,
            ).fit(x, y),
        )
    )
    feature_set = "combined"
    x = features[feature_set].iloc[row_indices].to_numpy(dtype=float)

    def logistic_factory(
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_seeds: np.ndarray,
    ) -> Pipeline:
        chosen = _choose_logistic_c(train_x, train_y, train_seeds, logistic_cs)
        return Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(C=chosen, max_iter=2000, random_state=0),
                ),
            )
        ).fit(train_x, train_y)

    predictors.append((f"l2_logistic:{feature_set}", feature_set, x, logistic_factory))

    rows: list[dict[str, object]] = []
    for predictor, feature_set, x, factory in predictors:
        seeds = contexts["seed"].astype(int).to_numpy()
        for fold_seed in sorted(set(seeds)):
            test = seeds == fold_seed
            train = ~test
            model = factory(x[train], labels[train], seeds[train])
            prediction = model.predict(x[test]).astype(int)
            probability = (
                model.predict_proba(x[test])[:, 1]
                if hasattr(model, "predict_proba")
                else prediction.astype(float)
            )
            for local, row_index in enumerate(np.flatnonzero(test)):
                selected_arm = (
                    "gcb"
                    if prediction[local] == 1
                    else "efficiency_budget_reallocation"
                )
                selected_delta = sep[row_index] if prediction[local] == 1 else budget[row_index]
                pairwise_best = max(sep[row_index], budget[row_index])
                safe_reference_delta = float(budget[row_index])
                rows.append(
                    {
                        "schema_version": EVIDENCE_PREDICTABILITY_SCHEMA_VERSION,
                        "predictor": predictor,
                        "feature_set": feature_set,
                        "fold_seed": fold_seed,
                        "problem_id": contexts.iloc[row_index]["problem_id"],
                        "seed": int(contexts.iloc[row_index]["seed"]),
                        "context_id": contexts.iloc[row_index]["context_id"],
                        "target_action": target_actions[row_index],
                        "preferred_arm": (
                            "gcb"
                            if preferred_labels[row_index] == 1
                            else "efficiency_budget_reallocation"
                        ),
                        "predicted_arm": selected_arm,
                        "sep_action_probability": float(probability[local]),
                        "correct": int(prediction[local] == labels[row_index]),
                        "preference_correct": int(
                            prediction[local] == preferred_labels[row_index]
                        ),
                        "selected_delta": float(selected_delta),
                        "safe_reference_arm": SAFE_REFERENCE_ARM,
                        "safe_reference_delta": safe_reference_delta,
                        "gain_over_safe_reference": float(
                            selected_delta - safe_reference_delta
                        ),
                        "pairwise_regret": float(pairwise_best - selected_delta),
                        "catastrophic": int(selected_delta <= CATASTROPHIC_DELTA),
                        "runtime_authorized": RUNTIME_AUTHORIZED,
                    }
                )
    return pd.DataFrame(rows)


def _crossfit_stump_accuracy(
    shared_count: np.ndarray,
    labels: np.ndarray,
    seeds: np.ndarray,
) -> float:
    predictions = np.empty_like(labels)
    for fold_seed in sorted(set(int(item) for item in seeds)):
        test = seeds == fold_seed
        train = ~test
        model = DecisionTreeClassifier(
            max_depth=1,
            class_weight="balanced",
            random_state=0,
        ).fit(shared_count[train], labels[train])
        predictions[test] = model.predict(shared_count[test]).astype(int)
    return float(balanced_accuracy_score(labels, predictions))


def r4_s5_cluster_permutation_test(
    dataset: DiagnosticDataset,
    features: pd.DataFrame,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    subset = dataset.contexts["problem_id"].isin(PRIMARY_CASES).to_numpy()
    contexts = dataset.contexts.loc[subset].reset_index(drop=True)
    x = features.loc[subset, ["topology.shared_count"]].to_numpy(dtype=float)
    labels = (
        contexts["problem_id"].map(R4_S5_TARGET_ACTIONS).to_numpy()
        == "gcb"
    ).astype(int)
    seeds = contexts["seed"].astype(int).to_numpy()
    cluster_keys = list(
        dict.fromkeys(zip(contexts["problem_id"], seeds, strict=True))
    )
    cluster_labels = []
    for key in cluster_keys:
        mask = (contexts["problem_id"] == key[0]).to_numpy() & (seeds == key[1])
        values = np.unique(labels[mask])
        if values.size != 1:
            raise ValueError("target action varies inside a case-seed cluster")
        cluster_labels.append(int(values[0]))
    observed = _crossfit_stump_accuracy(x, labels, seeds)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        shuffled = rng.permutation(cluster_labels)
        permuted = labels.copy()
        for key, label in zip(cluster_keys, shuffled, strict=True):
            mask = (contexts["problem_id"] == key[0]).to_numpy() & (seeds == key[1])
            permuted[mask] = label
        score = _crossfit_stump_accuracy(x, permuted, seeds)
        exceedances += int(score >= observed - 1e-15)
    return {
        "observed_balanced_accuracy": observed,
        "permutations": permutations,
        "exceedances": exceedances,
        "p_value": (exceedances + 1) / (permutations + 1),
    }


def summarize_pairwise(
    rows: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    summaries = []
    for predictor, predictor_rows in rows.groupby("predictor", sort=False):
        for scope, group in [("all", predictor_rows), *predictor_rows.groupby("problem_id")]:
            labels = (group["target_action"] == "gcb").astype(int)
            predictions = (group["predicted_arm"] == "gcb").astype(int)
            delta_lcb, delta_ucb = _case_stratified_bootstrap(
                group,
                "selected_delta",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            regret_lcb, regret_ucb = _case_stratified_bootstrap(
                group,
                "pairwise_regret",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 1,
            )
            gain_lcb, gain_ucb = _case_stratified_bootstrap(
                group,
                "gain_over_safe_reference",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 2,
            )
            both_classes = labels.nunique() == 2
            summaries.append(
                {
                    "schema_version": EVIDENCE_PREDICTABILITY_SCHEMA_VERSION,
                    "analysis_status": ANALYSIS_STATUS,
                    "predictor": predictor,
                    "feature_set": group["feature_set"].iloc[0],
                    "scope": scope,
                    "target_action": (
                        group["target_action"].iloc[0]
                        if group["target_action"].nunique() == 1
                        else "R4_S5_mixed"
                    ),
                    "context_count": len(group),
                    "cluster_count": group[["problem_id", "seed"]]
                    .drop_duplicates()
                    .shape[0],
                    "routing_accuracy": float(group["correct"].mean()),
                    "pairwise_preference_accuracy": float(
                        group["preference_correct"].mean()
                    ),
                    "target_preference_consistency": float(
                        (group["target_action"] == group["preferred_arm"]).mean()
                    ),
                    "balanced_accuracy": (
                        balanced_accuracy_score(labels, predictions)
                        if both_classes
                        else math.nan
                    ),
                    "roc_auc": (
                        roc_auc_score(
                            labels,
                            group["sep_action_probability"].to_numpy(dtype=float),
                        )
                        if both_classes
                        else math.nan
                    ),
                    "mean_selected_delta": float(group["selected_delta"].mean()),
                    "min_selected_delta": float(group["selected_delta"].min()),
                    "selected_delta_lcb": delta_lcb,
                    "selected_delta_ucb": delta_ucb,
                    "positive_count": int((group["selected_delta"] > 0.0).sum()),
                    "positive_rate": float((group["selected_delta"] > 0.0).mean()),
                    "material_positive_count": int(
                        (group["selected_delta"] > MATERIAL_POSITIVE_DELTA).sum()
                    ),
                    "material_positive_rate": float(
                        (group["selected_delta"] > MATERIAL_POSITIVE_DELTA).mean()
                    ),
                    "mean_pairwise_regret": float(group["pairwise_regret"].mean()),
                    "pairwise_regret_lcb": regret_lcb,
                    "pairwise_regret_ucb": regret_ucb,
                    "safe_reference_arm": SAFE_REFERENCE_ARM,
                    "mean_gain_over_safe_reference": float(
                        group["gain_over_safe_reference"].mean()
                    ),
                    "gain_over_safe_reference_lcb": gain_lcb,
                    "gain_over_safe_reference_ucb": gain_ucb,
                    "catastrophic_count": int(group["catastrophic"].sum()),
                    "catastrophic_rate": float(group["catastrophic"].mean()),
                    "runtime_authorized": RUNTIME_AUTHORIZED,
                }
            )
    return pd.DataFrame(summaries)


def _validate_oof_coverage(
    rows: pd.DataFrame,
    *,
    context_ids: Sequence[str],
    expected_predictors: Sequence[str],
) -> dict[str, int]:
    predictors = tuple(expected_predictors)
    if set(rows["predictor"].unique()) != set(predictors):
        raise RuntimeError("OOF predictor set drifted")
    expected = {
        (predictor, context_id)
        for predictor in predictors
        for context_id in context_ids
    }
    actual_pairs = list(zip(rows["predictor"], rows["context_id"], strict=True))
    actual = set(actual_pairs)
    if len(actual_pairs) != len(actual) or actual != expected:
        raise RuntimeError("OOF predictor/context matrix is incomplete or duplicated")
    if not (rows["fold_seed"].astype(int) == rows["seed"].astype(int)).all():
        raise RuntimeError("OOF prediction was assigned to the wrong held-out seed")
    return {
        "predictor_count": len(predictors),
        "expected_row_count": len(expected),
        "actual_row_count": len(rows),
        "unique_predictor_context_pairs": len(actual),
        "fold_seed_matches": 1,
        "passed": 1,
    }


def run_evidence_predictability(artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == artifact_dir or output_dir in artifact_dir.parents:
        raise ValueError("evidence output directory cannot overwrite source artifacts")
    config = load_config(CONFIG_PATH)
    protocol = config.get("evidence_predictability", {})
    if protocol.get("schema_version") != EVIDENCE_PREDICTABILITY_SCHEMA_VERSION:
        raise ValueError("evidence predictability config schema mismatch")
    if (
        protocol.get("analysis_status") != ANALYSIS_STATUS
        or protocol.get("generalization_scope") != GENERALIZATION_SCOPE
    ):
        raise ValueError("evidence predictability interpretation scope drifted")
    if protocol.get("primary_horizon") != PRIMARY_HORIZON:
        raise ValueError("evidence predictability must use sweep_1")
    if (
        protocol.get("primary_scope") != PRIMARY_SCOPE
        or protocol.get("primary_predictor") != PRIMARY_PREDICTOR
        or protocol.get("secondary_13_arm_predictor") != SECONDARY_13_ARM_PREDICTOR
        or protocol.get("target_actions") != R4_S5_TARGET_ACTIONS
        or protocol.get("validation_scope")
        != {"cases": list(PRIMARY_CASES), "out_of_scope": OUT_OF_SCOPE_POLICY}
        or protocol.get("safe_reference_arm") != SAFE_REFERENCE_ARM
        or protocol.get("minimum_case_material_positive_rate")
        != MINIMUM_CASE_MATERIAL_POSITIVE_RATE
    ):
        raise ValueError("evidence predictability target/predictor contract drifted")
    if (
        protocol.get("outer_split") != "leave_one_seed_out"
        or protocol.get("inner_split") != "leave_one_training_seed_out"
        or protocol.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or protocol.get("bootstrap_seed") != BOOTSTRAP_SEED
        or protocol.get("pairwise_permutations") != BOOTSTRAP_REPLICATES
        or protocol.get("pairwise_permutation_seed") != BOOTSTRAP_SEED
    ):
        raise ValueError("evidence predictability split/statistics contract drifted")

    dataset, source_manifest = load_diagnostic_dataset(artifact_dir)
    feature_frames = build_feature_frames(dataset)
    ridge_alphas = tuple(float(value) for value in protocol["ridge_alphas"])
    if not ridge_alphas:
        raise ValueError("model tuning grid cannot be empty")
    if any(value <= 0.0 for value in ridge_alphas):
        raise ValueError("model tuning parameters must be positive")

    oof_frames = [crossfit_baselines(dataset)]
    for feature_set in FEATURE_SET_GROUPS:
        oof_frames.append(
            crossfit_value_model(
                dataset,
                feature_frames[feature_set],
                kind="ridge_value",
                candidates=ridge_alphas,
                feature_set=feature_set,
            )
        )
    oof = pd.concat(oof_frames, ignore_index=True)
    oof_integrity = _validate_oof_coverage(
        oof,
        context_ids=dataset.contexts["context_id"].tolist(),
        expected_predictors=OOF_PREDICTORS,
    )

    summary = summarize_predictions(
        oof,
        bootstrap_replicates=int(protocol["bootstrap_replicates"]),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
    )
    pairwise = crossfit_r4_s5_pairwise(
        dataset,
        feature_frames,
        logistic_cs=ridge_alphas,
    )
    pairwise_context_ids = dataset.contexts[
        dataset.contexts["problem_id"].isin(PRIMARY_CASES)
    ]["context_id"].tolist()
    pairwise_integrity = _validate_oof_coverage(
        pairwise,
        context_ids=pairwise_context_ids,
        expected_predictors=R4_S5_PAIRWISE_PREDICTORS,
    )
    pairwise_summary = summarize_pairwise(
        pairwise,
        bootstrap_replicates=int(protocol["bootstrap_replicates"]),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
    )
    permutation = r4_s5_cluster_permutation_test(
        dataset,
        feature_frames["topology"],
        permutations=int(protocol["pairwise_permutations"]),
        seed=int(protocol["pairwise_permutation_seed"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "evidence_predictability_oof.csv": output_dir / "evidence_predictability_oof.csv",
        "evidence_predictability_summary.csv": output_dir / "evidence_predictability_summary.csv",
        "r4_s5_pairwise_oof.csv": output_dir / "r4_s5_pairwise_oof.csv",
        "r4_s5_pairwise_summary.csv": output_dir / "r4_s5_pairwise_summary.csv",
    }
    _write_csv(paths["evidence_predictability_oof.csv"], oof)
    _write_csv(paths["evidence_predictability_summary.csv"], summary)
    _write_csv(paths["r4_s5_pairwise_oof.csv"], pairwise)
    _write_csv(paths["r4_s5_pairwise_summary.csv"], pairwise_summary)

    feature_manifest = {
        feature_set: list(frame.columns) for feature_set, frame in feature_frames.items()
    }
    primary = pairwise_summary[
        (pairwise_summary["predictor"] == PRIMARY_PREDICTOR)
        & (pairwise_summary["scope"] == "all")
    ]
    if len(primary) != 1:
        raise RuntimeError("primary predictor summary is missing")
    primary_row = primary.iloc[0]
    primary_case_rows = pairwise_summary[
        (pairwise_summary["predictor"] == PRIMARY_PREDICTOR)
        & (pairwise_summary["scope"].isin(PRIMARY_CASES))
    ].set_index("scope")
    if set(primary_case_rows.index) != set(PRIMARY_CASES):
        raise RuntimeError("per-case primary summaries are incomplete")
    secondary = summary[
        (summary["predictor"] == SECONDARY_13_ARM_PREDICTOR)
        & (summary["scope"] == "all")
    ]
    if len(secondary) != 1:
        raise RuntimeError("secondary 13-arm predictor summary is missing")
    secondary_row = secondary.iloc[0]
    case_gate = {
        case: {
            "routing_accuracy": float(primary_case_rows.loc[case, "routing_accuracy"]),
            "selected_delta_lcb": float(
                primary_case_rows.loc[case, "selected_delta_lcb"]
            ),
            "material_positive_rate": float(
                primary_case_rows.loc[case, "material_positive_rate"]
            ),
            "catastrophic_count": int(
                primary_case_rows.loc[case, "catastrophic_count"]
            ),
            "gain_over_safe_reference_lcb": float(
                primary_case_rows.loc[case, "gain_over_safe_reference_lcb"]
            ),
        }
        for case in PRIMARY_CASES
    }
    if any(item["routing_accuracy"] < 1.0 for item in case_gate.values()):
        evidence_gate = "routing_errors"
    elif any(item["catastrophic_count"] > 0 for item in case_gate.values()):
        evidence_gate = "catastrophic_oof_losses"
    elif any(item["selected_delta_lcb"] <= 0.0 for item in case_gate.values()):
        evidence_gate = "nonpositive_case_selected_delta_lcb"
    elif any(
        item["material_positive_rate"] < MINIMUM_CASE_MATERIAL_POSITIVE_RATE
        for item in case_gate.values()
    ):
        evidence_gate = "insufficient_case_material_positive_rate"
    elif any(
        item["gain_over_safe_reference_lcb"] < -ACTION_CEILING_TIE_TOLERANCE
        for item in case_gate.values()
    ):
        evidence_gate = "worse_than_safe_reference"
    else:
        evidence_gate = "small_runtime_validation_R4_S5_only"
    manifest = {
        "schema_version": EVIDENCE_PREDICTABILITY_SCHEMA_VERSION,
        "analysis_status": ANALYSIS_STATUS,
        "generalization_scope": GENERALIZATION_SCOPE,
        "source_protocol_version": ACTION_CEILING_PROTOCOL_VERSION,
        "source_artifact_dir": str(artifact_dir.resolve()),
        "source_manifest_sha256": _sha256_file(artifact_dir / "manifest.json"),
        "analysis_config_sha256": _sha256_file(CONFIG_PATH),
        "source_artifacts": source_manifest["artifacts"],
        "source_integrity_gate_passed": 1,
        "context_count": len(dataset.contexts),
        "case_seed_cluster_count": dataset.contexts[
            ["problem_id", "seed"]
        ].drop_duplicates().shape[0],
        "outer_fold_seeds": sorted(int(value) for value in dataset.contexts["seed"].unique()),
        "outer_split": "leave_one_seed_out",
        "inner_split": "leave_one_training_seed_out",
        "feature_sets": feature_manifest,
        "context_only_feature_sets": list(CONTEXT_ONLY_FEATURE_SETS),
        "evidence_input_scope": "aggregated_action_ceiling_contexts_csv_only",
        "phase1_fields_not_available_in_context_csv": [
            "cohen_d",
            "proposal_disagreement",
            "probe_fitness",
            "probe_utility",
        ],
        "forbidden_model_fields": list(FORBIDDEN_MODEL_FIELDS),
        "identity_features_used": 0,
        "future_features_used": 0,
        "primary_scope": PRIMARY_SCOPE,
        "primary_predictor": PRIMARY_PREDICTOR,
        "primary_target_actions": R4_S5_TARGET_ACTIONS,
        "primary_case_gate": case_gate,
        "minimum_case_material_positive_rate": MINIMUM_CASE_MATERIAL_POSITIVE_RATE,
        "safe_reference_arm": SAFE_REFERENCE_ARM,
        "secondary_13_arm_predictor": SECONDARY_13_ARM_PREDICTOR,
        "oof_integrity_gate": oof_integrity,
        "r4_s5_pairwise_integrity_gate": pairwise_integrity,
        "primary_oof_mean_selected_delta": float(primary_row["mean_selected_delta"]),
        "primary_oof_selected_delta_lcb": float(primary_row["selected_delta_lcb"]),
        "primary_oof_routing_accuracy": float(primary_row["routing_accuracy"]),
        "primary_oof_pairwise_preference_accuracy": float(
            primary_row["pairwise_preference_accuracy"]
        ),
        "primary_oof_positive_rate": float(primary_row["positive_rate"]),
        "primary_oof_material_positive_rate": float(
            primary_row["material_positive_rate"]
        ),
        "primary_oof_catastrophic_count": int(primary_row["catastrophic_count"]),
        "secondary_13_arm_mean_selected_delta": float(
            secondary_row["mean_selected_delta"]
        ),
        "secondary_13_arm_exact_accuracy": float(secondary_row["exact_accuracy"]),
        "secondary_13_arm_catastrophic_count": int(
            secondary_row["catastrophic_count"]
        ),
        "runtime_selector_authorized": 0,
        "evidence_gate": evidence_gate,
        "small_runtime_validation_authorization": {
            "authorized": int(
                evidence_gate == "small_runtime_validation_R4_S5_only"
            ),
            "cases": list(PRIMARY_CASES),
            "target_actions": R4_S5_TARGET_ACTIONS,
            "out_of_scope": OUT_OF_SCOPE_POLICY,
        },
        "global_selector_authorized": 0,
        "runtime_blocker": "offline_counterfactual_not_runtime_validated",
        "r4_s5_pairwise_permutation": permutation,
        "r4_s5_interpretation_limit": (
            "current_R4_and_S5_cases_only; shared_count_is_confounded_with_case"
        ),
        "runtime_authorized": RUNTIME_AUTHORIZED,
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {name: _sha256_file(path) for name, path in paths.items()},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing the aggregated exp019 v6 context/arm CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ARTIFACT_DIR/evidence_predictability).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact_dir = args.artifact_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else artifact_dir / "evidence_predictability"
    )
    manifest = run_evidence_predictability(artifact_dir, output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
