"""Outcome-derived selector for the four independent ARAC actions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Protocol

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from arac.evidence.phase1 import PHASE1_FEATURE_NAMES, PHASE1_PROTOCOL
from arac.runtime.contracts import ACTION_NAMES


SELECTOR_SCHEMA = "arac-outcome-selector-v7"
EVALUATION_SCHEMA = "arac-outcome-selector-evaluation-v7"
OUTCOME_RECORD_SCHEMA = "arac-counterfactual-outcome-v3"
MODEL_FILENAME = "outcome_selector.joblib"
METADATA_FILENAME = "outcome_selector.json"
EVALUATION_FILENAME = "outcome_selector_evaluation.json"
RANDOM_STATE = 20260730
MODEL_PARAMETERS = {
    "n_estimators": 1000,
    "min_samples_leaf": 3,
    "max_features": 0.5,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}
HOLDOUT_GATE_PARAMETERS = {
    "minimum_accuracy": 0.5,
    "minimum_balanced_accuracy": 0.5,
    "maximum_mean_log10_regret": 0.05,
    "maximum_worst_log10_regret": 0.25,
}
TRAINING_CV_GATE_PARAMETERS = dict(HOLDOUT_GATE_PARAMETERS)
# Training must preserve severe-action ordering beyond the publication gate.
# The v5 cap of 0.25 collapsed every larger regret into one target value.
REGRET_TARGET_CAP = 1.0
UNCERTAINTY_PENALTY_CANDIDATES = (0.0, 0.25, 0.5, 1.0)
TAIL_RISK_FRACTION = 0.05


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    temporary.replace(destination)


def _read_json_mapping(path: Path, artifact_name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"selector {artifact_name} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"selector {artifact_name} must be a JSON object")
    return payload


def _publish_selector_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"selector destination is not empty: {destination}")
        destination.rmdir()
    staging.replace(destination)


@dataclass(frozen=True)
class ActionOutcome:
    action_name: str
    final_error: float
    result_hash: str

    def __post_init__(self) -> None:
        if self.action_name not in ACTION_NAMES:
            raise ValueError("unknown action outcome")
        if not math.isfinite(float(self.final_error)) or self.final_error < 0.0:
            raise ValueError("action outcome must be finite and non-negative")
        if len(self.result_hash) != 64:
            raise ValueError("action result hash must be SHA-256")


@dataclass(frozen=True)
class OutcomeRecord:
    """Offline audit row; case metadata is excluded from ``model_row``."""

    case_id: str
    run_seed: int
    checkpoint_hash: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    outcomes: tuple[ActionOutcome, ...]

    def __post_init__(self) -> None:
        if not self.case_id or isinstance(self.run_seed, bool) or self.run_seed < 0:
            raise ValueError("offline outcome provenance is invalid")
        if len(self.checkpoint_hash) != 64:
            raise ValueError("checkpoint_hash must be SHA-256")
        if self.feature_names != PHASE1_FEATURE_NAMES:
            raise ValueError("outcome feature schema drifted")
        if len(self.feature_values) != len(self.feature_names) or not all(
            math.isfinite(value) for value in self.feature_values
        ):
            raise ValueError("outcome features are invalid")
        names = tuple(outcome.action_name for outcome in self.outcomes)
        if names != ACTION_NAMES:
            raise ValueError("outcome record must contain all four actions in fixed order")

    @property
    def action_label(self) -> str:
        best_index = min(
            range(len(self.outcomes)),
            key=lambda index: (self.outcomes[index].final_error, index),
        )
        return self.outcomes[best_index].action_name

    def model_row(self) -> tuple[float, ...]:
        return self.feature_values

    def payload(self) -> dict[str, object]:
        payload = {
            "schema_version": OUTCOME_RECORD_SCHEMA,
            "phase1_protocol": PHASE1_PROTOCOL,
            "case_id": self.case_id,
            "run_seed": self.run_seed,
            "checkpoint_hash": self.checkpoint_hash,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "outcomes": [
                {
                    "action_name": outcome.action_name,
                    "final_error": outcome.final_error,
                    "result_hash": outcome.result_hash,
                }
                for outcome in self.outcomes
            ],
            "action_label": self.action_label,
            "label_source": "common_checkpoint_argmin_terminal_error",
        }
        return payload | {"record_hash": _canonical_sha256(payload)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> OutcomeRecord:
        values = dict(payload)
        claimed_hash = values.pop("record_hash", None)
        if claimed_hash != _canonical_sha256(values):
            raise ValueError("outcome record hash drifted")
        if values.get("schema_version") != OUTCOME_RECORD_SCHEMA:
            raise ValueError("outcome record schema drifted")
        if values.get("phase1_protocol") != PHASE1_PROTOCOL:
            raise ValueError("outcome Phase-I protocol drifted")
        if values.get("label_source") != "common_checkpoint_argmin_terminal_error":
            raise ValueError("outcome label provenance drifted")
        outcomes = tuple(
            ActionOutcome(
                action_name=str(item["action_name"]),
                final_error=float(item["final_error"]),
                result_hash=str(item["result_hash"]),
            )
            for item in values["outcomes"]  # type: ignore[union-attr]
        )
        record = cls(
            case_id=str(values["case_id"]),
            run_seed=int(values["run_seed"]),
            checkpoint_hash=str(values["checkpoint_hash"]),
            feature_names=tuple(str(name) for name in values["feature_names"]),
            feature_values=tuple(float(value) for value in values["feature_values"]),
            outcomes=outcomes,
        )
        if values.get("action_label") != record.action_label:
            raise ValueError("stored outcome label disagrees with terminal argmin")
        return record


class Selector(Protocol):
    def select(self, feature_names: Sequence[str], feature_values: Sequence[float]) -> str: ...


@dataclass(frozen=True)
class OutcomeSelector:
    model: Any
    metadata: Mapping[str, object]

    @classmethod
    def load(cls, directory: Path) -> OutcomeSelector:
        root = Path(directory).resolve()
        metadata_path = root / METADATA_FILENAME
        model_path = root / MODEL_FILENAME
        evaluation_path = root / EVALUATION_FILENAME
        missing = [
            path.name
            for path in (model_path, metadata_path, evaluation_path)
            if not path.is_file()
        ]
        if missing:
            raise ValueError(
                "selector artifact set is incomplete: " + ", ".join(sorted(missing))
            )
        metadata = _read_json_mapping(metadata_path, "metadata")
        if metadata.get("schema_version") != SELECTOR_SCHEMA:
            raise ValueError("selector schema drifted")
        if metadata.get("phase1_protocol") != PHASE1_PROTOCOL:
            raise ValueError("selector Phase-I protocol drifted")
        if metadata.get("feature_names") != list(PHASE1_FEATURE_NAMES):
            raise ValueError("selector feature schema drifted")
        if metadata.get("actions") != list(ACTION_NAMES):
            raise ValueError("selector action order drifted")
        if metadata.get("labels_derived_from_action_outcomes") is not True:
            raise ValueError("selector labels lack outcome provenance")
        if metadata.get("predicts_action_log10_regret") is not True:
            raise ValueError("selector does not predict action regret")
        if metadata.get("regret_target_cap") != REGRET_TARGET_CAP:
            raise ValueError("selector regret target cap drifted")
        uncertainty_penalty = metadata.get("uncertainty_penalty")
        if uncertainty_penalty not in UNCERTAINTY_PENALTY_CANDIDATES:
            raise ValueError("selector uncertainty penalty drifted")
        if metadata.get("model_sha256") != file_sha256(model_path):
            raise ValueError("selector model hash drifted")
        if metadata.get("evaluation_sha256") != file_sha256(evaluation_path):
            raise ValueError("selector evaluation hash drifted")
        evaluation = _read_json_mapping(evaluation_path, "evaluation")
        if evaluation.get("schema_version") != EVALUATION_SCHEMA:
            raise ValueError("selector evaluation schema drifted")
        if evaluation.get("feature_names") != list(PHASE1_FEATURE_NAMES):
            raise ValueError("selector evaluation feature schema drifted")
        if evaluation.get("actions") != list(ACTION_NAMES):
            raise ValueError("selector evaluation action order drifted")
        model_selection = evaluation.get("model_selection")
        if not isinstance(model_selection, Mapping) or (
            model_selection.get("regret_target_cap") != REGRET_TARGET_CAP
        ):
            raise ValueError("selector evaluation regret target cap drifted")
        if model_selection.get("uncertainty_penalty") != uncertainty_penalty:
            raise ValueError("selector evaluation uncertainty penalty drifted")
        training_cv = evaluation.get("training_seed_cv")
        if not isinstance(training_cv, Mapping):
            raise ValueError("selector training CV evaluation is invalid")
        if (
            metadata.get("training_seed_cv_passed") is not True
            or training_cv.get("passed") is not True
        ):
            raise ValueError("selector training CV gate did not pass")
        if training_cv.get("gate_parameters") != TRAINING_CV_GATE_PARAMETERS:
            raise ValueError("selector training CV gate parameters drifted")
        try:
            training_gate_passed = _metrics_passed(
                training_cv,
                TRAINING_CV_GATE_PARAMETERS,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("selector training CV metrics are invalid") from exc
        if not training_gate_passed:
            raise ValueError("selector training CV metrics do not pass the frozen gate")
        holdout = evaluation.get("holdout")
        if not isinstance(holdout, Mapping):
            raise ValueError("selector holdout evaluation is invalid")
        if metadata.get("holdout_passed") is not True or holdout.get("passed") is not True:
            raise ValueError("selector holdout gate did not pass")
        if holdout.get("gate_parameters") != HOLDOUT_GATE_PARAMETERS:
            raise ValueError("selector holdout gate parameters drifted")
        try:
            gate_passed = _holdout_passed(holdout)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("selector holdout metrics are invalid") from exc
        if not gate_passed:
            raise ValueError("selector holdout metrics do not pass the frozen gate")
        model = joblib.load(model_path)
        if int(getattr(model, "n_features_in_", -1)) != len(PHASE1_FEATURE_NAMES):
            raise ValueError("selector model width drifted")
        if int(getattr(model, "n_outputs_", -1)) != len(ACTION_NAMES):
            raise ValueError("selector model output width drifted")
        return cls(model=model, metadata=metadata)

    def select(
        self,
        feature_names: Sequence[str],
        feature_values: Sequence[float],
    ) -> str:
        names = tuple(str(name) for name in feature_names)
        values = tuple(float(value) for value in feature_values)
        if names != PHASE1_FEATURE_NAMES:
            raise ValueError("selector input feature schema drifted")
        if len(values) != len(names) or not all(math.isfinite(value) for value in values):
            raise ValueError("selector input values are invalid")
        scores = _risk_adjusted_scores(
            self.model,
            np.asarray([values], dtype=float),
            uncertainty_penalty=float(self.metadata["uncertainty_penalty"]),
        )
        if scores.shape != (1, len(ACTION_NAMES)) or not np.all(np.isfinite(scores)):
            raise RuntimeError("selector produced invalid action regret scores")
        action = ACTION_NAMES[int(np.argmin(scores[0]))]
        if action not in ACTION_NAMES:
            raise RuntimeError("selector predicted an unsupported action")
        return action


def _regret_metrics(
    records: Sequence[OutcomeRecord],
    predictions: np.ndarray,
) -> dict[str, object]:
    if len(records) != len(predictions):
        raise ValueError("regret records and predictions must be aligned")
    ratios = []
    for record, prediction in zip(records, predictions, strict=True):
        errors = {outcome.action_name: outcome.final_error for outcome in record.outcomes}
        selected = errors[str(prediction)]
        oracle = min(errors.values())
        ratios.append((selected + 1.0) / (oracle + 1.0))
    values = np.asarray(ratios, dtype=float)
    log_values = np.log10(values)
    tail_count = max(1, math.ceil(len(log_values) * TAIL_RISK_FRACTION))
    tail_values = np.sort(log_values)[-tail_count:]
    return {
        "mean_log10_regret": float(np.mean(log_values)),
        "p95_log10_regret": float(np.quantile(log_values, 0.95)),
        "cvar95_log10_regret": float(np.mean(tail_values)),
        "worst_log10_regret": float(np.max(log_values)),
        "mean_selected_to_oracle_ratio": float(np.mean(values)),
        "worst_selected_to_oracle_ratio": float(np.max(values)),
        "within_1_percent_of_oracle": float(np.mean(values <= 1.01)),
        "within_5_percent_of_oracle": float(np.mean(values <= 1.05)),
        "within_10_percent_of_oracle": float(np.mean(values <= 1.10)),
    }


def _metrics(
    records: Sequence[OutcomeRecord],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    confusion = confusion_matrix(targets, predictions, labels=ACTION_NAMES)
    return {
        "record_count": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "confusion_matrix": confusion.tolist(),
        "per_action_recall": {
            action: (
                float(confusion[index, index] / np.sum(confusion[index]))
                if np.sum(confusion[index])
                else None
            )
            for index, action in enumerate(ACTION_NAMES)
        },
        "prediction_counts": dict(sorted(Counter(str(value) for value in predictions).items())),
        "terminal_regret": _regret_metrics(records, predictions),
    }


def _regression_targets(records: Sequence[OutcomeRecord]) -> np.ndarray:
    rows = []
    for record in records:
        errors = np.asarray([outcome.final_error for outcome in record.outcomes], dtype=float)
        oracle = float(np.min(errors))
        log_regret = np.log10((errors + 1.0) / (oracle + 1.0))
        rows.append(np.minimum(log_regret, REGRET_TARGET_CAP))
    return np.asarray(rows, dtype=float)


def _actions_from_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(ACTION_NAMES):
        raise ValueError("selector score matrix has the wrong shape")
    return np.asarray(ACTION_NAMES, dtype=object)[np.argmin(values, axis=1)]


def _score_components(model: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=float)
    mean_scores = np.asarray(model.predict(values), dtype=float)
    expected_shape = (len(values), len(ACTION_NAMES))
    if mean_scores.shape != expected_shape or not np.all(np.isfinite(mean_scores)):
        raise RuntimeError("selector produced invalid action regret scores")
    estimators = tuple(getattr(model, "estimators_", ()))
    if not estimators:
        raise RuntimeError("selector model does not expose ensemble uncertainty")
    tree_scores = np.asarray(
        [np.asarray(estimator.predict(values), dtype=float) for estimator in estimators],
        dtype=float,
    )
    if tree_scores.shape != (len(estimators), *expected_shape) or not np.all(
        np.isfinite(tree_scores)
    ):
        raise RuntimeError("selector produced invalid tree regret scores")
    return mean_scores, np.std(tree_scores, axis=0)


def _risk_adjusted_scores(
    model: Any,
    features: np.ndarray,
    *,
    uncertainty_penalty: float,
) -> np.ndarray:
    if uncertainty_penalty not in UNCERTAINTY_PENALTY_CANDIDATES:
        raise ValueError("selector uncertainty penalty drifted")
    mean_scores, uncertainty = _score_components(model, features)
    return mean_scores + uncertainty_penalty * uncertainty


def _new_model() -> RandomForestRegressor:
    return RandomForestRegressor(**MODEL_PARAMETERS)


def _training_seed_cv(
    records: Sequence[OutcomeRecord],
    features: np.ndarray,
    action_targets: np.ndarray,
    regression_targets: np.ndarray,
) -> dict[str, object]:
    seeds = sorted({record.run_seed for record in records})
    if len(seeds) < 2:
        raise ValueError("selector training requires at least two distinct run seeds")
    means = np.empty((len(records), len(ACTION_NAMES)), dtype=float)
    uncertainties = np.empty_like(means)
    for seed in seeds:
        validation = np.asarray([record.run_seed == seed for record in records], dtype=bool)
        training = ~validation
        model = _new_model()
        model.fit(features[training], regression_targets[training])
        fold_means, fold_uncertainties = _score_components(model, features[validation])
        means[validation] = fold_means
        uncertainties[validation] = fold_uncertainties
    candidates = {}
    for penalty in UNCERTAINTY_PENALTY_CANDIDATES:
        predictions = _actions_from_scores(means + penalty * uncertainties)
        candidates[penalty] = _metrics(records, action_targets, predictions)

    def selection_key(penalty: float) -> tuple[float, float, float, float, float]:
        candidate = candidates[penalty]
        regret = candidate["terminal_regret"]
        if not isinstance(regret, Mapping):
            raise TypeError("selector regret metrics are invalid")
        return (
            float(regret["cvar95_log10_regret"]),
            float(regret["worst_log10_regret"]),
            float(regret["mean_log10_regret"]),
            -float(candidate["balanced_accuracy"]),
            penalty,
        )

    selected_penalty = min(UNCERTAINTY_PENALTY_CANDIDATES, key=selection_key)
    metrics = candidates[selected_penalty]
    metrics["uncertainty_penalty"] = selected_penalty
    metrics["risk_model_selection"] = {
        "criterion": "minimum cvar95, worst, then mean leave-one-seed-out regret",
        "candidates": {
            str(penalty): {
                "accuracy": candidates[penalty]["accuracy"],
                "balanced_accuracy": candidates[penalty]["balanced_accuracy"],
                "terminal_regret": candidates[penalty]["terminal_regret"],
            }
            for penalty in UNCERTAINTY_PENALTY_CANDIDATES
        },
    }
    metrics["held_out_run_seeds"] = seeds
    return metrics


def _metrics_passed(
    metrics: Mapping[str, object],
    parameters: Mapping[str, float],
) -> bool:
    regret = metrics["terminal_regret"]
    if not isinstance(regret, Mapping):
        raise TypeError("selector regret metrics are invalid")
    return bool(
        float(metrics["accuracy"]) >= parameters["minimum_accuracy"]
        and float(metrics["balanced_accuracy"])
        >= parameters["minimum_balanced_accuracy"]
        and float(regret["mean_log10_regret"])
        <= parameters["maximum_mean_log10_regret"]
        and float(regret["worst_log10_regret"])
        <= parameters["maximum_worst_log10_regret"]
    )


def _holdout_passed(metrics: Mapping[str, object]) -> bool:
    return _metrics_passed(metrics, HOLDOUT_GATE_PARAMETERS)


def evaluate_training_selector(
    train_records: Sequence[OutcomeRecord],
) -> dict[str, object]:
    train = tuple(train_records)
    if not train:
        raise ValueError("selector training records must be non-empty")
    train_x = np.asarray([row.model_row() for row in train], dtype=float)
    train_y = np.asarray([row.action_label for row in train], dtype=object)
    train_regrets = _regression_targets(train)
    metrics = _training_seed_cv(train, train_x, train_y, train_regrets)
    metrics["gate_parameters"] = dict(TRAINING_CV_GATE_PARAMETERS)
    metrics["passed"] = _metrics_passed(metrics, TRAINING_CV_GATE_PARAMETERS)
    return metrics


def fit_outcome_selector(
    train_records: Sequence[OutcomeRecord],
    holdout_records: Sequence[OutcomeRecord],
    *,
    output_directory: Path,
) -> dict[str, object]:
    train = tuple(train_records)
    holdout = tuple(holdout_records)
    if not train or not holdout:
        raise ValueError("selector training and holdout records must be non-empty")
    train_keys = {(row.case_id, row.run_seed) for row in train}
    holdout_keys = {(row.case_id, row.run_seed) for row in holdout}
    if train_keys & holdout_keys:
        raise ValueError("selector training and holdout records overlap")
    destination = Path(output_directory).resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"selector destination is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    training_seed_cv = evaluate_training_selector(train)
    if training_seed_cv["passed"] is not True:
        regret = training_seed_cv["terminal_regret"]
        raise RuntimeError(
            "selector training CV gate failed; holdout must remain sealed "
            f"(accuracy={training_seed_cv['accuracy']:.6f}, "
            f"balanced_accuracy={training_seed_cv['balanced_accuracy']:.6f}, "
            f"mean_log10_regret={regret['mean_log10_regret']:.6f}, "
            f"worst_log10_regret={regret['worst_log10_regret']:.6f})"
        )
    train_x = np.asarray([row.model_row() for row in train], dtype=float)
    train_y = np.asarray([row.action_label for row in train], dtype=object)
    train_regrets = _regression_targets(train)
    uncertainty_penalty = float(training_seed_cv["uncertainty_penalty"])
    model = _new_model()
    model.fit(train_x, train_regrets)
    train_predictions = _actions_from_scores(
        _risk_adjusted_scores(
            model,
            train_x,
            uncertainty_penalty=uncertainty_penalty,
        )
    )
    holdout_x = np.asarray([row.model_row() for row in holdout], dtype=float)
    holdout_y = np.asarray([row.action_label for row in holdout], dtype=object)
    holdout_predictions = _actions_from_scores(
        _risk_adjusted_scores(
            model,
            holdout_x,
            uncertainty_penalty=uncertainty_penalty,
        )
    )

    evaluation = {
        "schema_version": EVALUATION_SCHEMA,
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "actions": list(ACTION_NAMES),
        "model_selection": {
            "criterion": "leave-one-training-seed-out tail-risk-adjusted capped regret",
            "regret_target_cap": REGRET_TARGET_CAP,
            "uncertainty_penalty": uncertainty_penalty,
            "holdout_used_for_model_selection": False,
        },
        "training": _metrics(train, train_y, train_predictions),
        "training_seed_cv": training_seed_cv,
        "holdout": _metrics(holdout, holdout_y, holdout_predictions),
        "holdout_errors": [
            {
                "case_id": record.case_id,
                "run_seed": record.run_seed,
                "actual_action": record.action_label,
                "predicted_action": str(holdout_predictions[index]),
            }
            for index, record in enumerate(holdout)
            if record.action_label != holdout_predictions[index]
        ],
    }
    evaluation["holdout"]["gate_parameters"] = dict(HOLDOUT_GATE_PARAMETERS)
    evaluation["holdout"]["passed"] = _holdout_passed(evaluation["holdout"])
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        staged_model_path = staging / MODEL_FILENAME
        staged_evaluation_path = staging / EVALUATION_FILENAME
        staged_metadata_path = staging / METADATA_FILENAME
        joblib.dump(model, staged_model_path)
        _write_json(staged_evaluation_path, evaluation)
        if evaluation["holdout"]["passed"] is not True:
            holdout_metrics = evaluation["holdout"]
            regret = holdout_metrics["terminal_regret"]
            raise RuntimeError(
                "selector holdout gate failed; selector was not published "
                f"(accuracy={holdout_metrics['accuracy']:.6f}, "
                f"balanced_accuracy={holdout_metrics['balanced_accuracy']:.6f}, "
                f"mean_log10_regret={regret['mean_log10_regret']:.6f}, "
                f"worst_log10_regret={regret['worst_log10_regret']:.6f})"
            )
        metadata = {
            "schema_version": SELECTOR_SCHEMA,
            "phase1_protocol": PHASE1_PROTOCOL,
            "feature_names": list(PHASE1_FEATURE_NAMES),
            "actions": list(ACTION_NAMES),
            "model_type": "RandomForestRegressor",
            "model_parameters": MODEL_PARAMETERS,
            "regret_target_cap": REGRET_TARGET_CAP,
            "uncertainty_penalty": uncertainty_penalty,
            "model_sha256": file_sha256(staged_model_path),
            "evaluation_sha256": file_sha256(staged_evaluation_path),
            "training_record_hashes": [row.payload()["record_hash"] for row in train],
            "holdout_record_hashes": [row.payload()["record_hash"] for row in holdout],
            "training_label_counts": dict(sorted(Counter(train_y).items())),
            "holdout_label_counts": dict(sorted(Counter(holdout_y).items())),
            "labels_derived_from_action_outcomes": True,
            "predicts_action_log10_regret": True,
            "inference_contains_case_or_family_identity": False,
            "holdout_used_for_model_selection": False,
            "training_seed_cv_accuracy": training_seed_cv["accuracy"],
            "training_seed_cv_passed": True,
            "holdout_passed": True,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(staged_metadata_path, metadata)
        OutcomeSelector.load(staging)
        _publish_selector_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    model_path = destination / MODEL_FILENAME
    metadata_path = destination / METADATA_FILENAME
    evaluation_path = destination / EVALUATION_FILENAME
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "evaluation_path": str(evaluation_path),
        "model_sha256": metadata["model_sha256"],
        "training_accuracy": evaluation["training"]["accuracy"],
        "training_seed_cv_accuracy": training_seed_cv["accuracy"],
        "holdout_accuracy": evaluation["holdout"]["accuracy"],
        "holdout_passed": evaluation["holdout"]["passed"],
    }


__all__ = [
    "EVALUATION_FILENAME",
    "EVALUATION_SCHEMA",
    "METADATA_FILENAME",
    "MODEL_FILENAME",
    "MODEL_PARAMETERS",
    "REGRET_TARGET_CAP",
    "UNCERTAINTY_PENALTY_CANDIDATES",
    "HOLDOUT_GATE_PARAMETERS",
    "TRAINING_CV_GATE_PARAMETERS",
    "OUTCOME_RECORD_SCHEMA",
    "SELECTOR_SCHEMA",
    "ActionOutcome",
    "OutcomeRecord",
    "OutcomeSelector",
    "Selector",
    "file_sha256",
    "evaluate_training_selector",
    "fit_outcome_selector",
]
