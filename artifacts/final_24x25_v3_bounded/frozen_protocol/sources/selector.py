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
from typing import Any, Protocol

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from arac.evidence.phase1 import PHASE1_FEATURE_NAMES, PHASE1_PROTOCOL
from arac.runtime.contracts import ACTION_NAMES


SELECTOR_SCHEMA = "arac-outcome-selector-v3"
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
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
        if metadata.get("model_sha256") != file_sha256(model_path):
            raise ValueError("selector model hash drifted")
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
        scores = np.asarray(self.model.predict(np.asarray([values], dtype=float)), dtype=float)
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
    return {
        "mean_log10_regret": float(np.mean(log_values)),
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
        rows.append(np.log10((errors + 1.0) / (oracle + 1.0)))
    return np.asarray(rows, dtype=float)


def _actions_from_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(ACTION_NAMES):
        raise ValueError("selector score matrix has the wrong shape")
    return np.asarray(ACTION_NAMES, dtype=object)[np.argmin(values, axis=1)]


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
    predictions = np.empty(len(records), dtype=object)
    for seed in seeds:
        validation = np.asarray([record.run_seed == seed for record in records], dtype=bool)
        training = ~validation
        model = _new_model()
        model.fit(features[training], regression_targets[training])
        predictions[validation] = _actions_from_scores(model.predict(features[validation]))
    metrics = _metrics(records, action_targets, predictions)
    metrics["held_out_run_seeds"] = seeds
    return metrics


def _holdout_passed(metrics: Mapping[str, object]) -> bool:
    regret = metrics["terminal_regret"]
    if not isinstance(regret, Mapping):
        raise TypeError("holdout regret metrics are invalid")
    return bool(
        float(metrics["accuracy"]) >= HOLDOUT_GATE_PARAMETERS["minimum_accuracy"]
        and float(metrics["balanced_accuracy"])
        >= HOLDOUT_GATE_PARAMETERS["minimum_balanced_accuracy"]
        and float(regret["mean_log10_regret"])
        <= HOLDOUT_GATE_PARAMETERS["maximum_mean_log10_regret"]
        and float(regret["worst_log10_regret"])
        <= HOLDOUT_GATE_PARAMETERS["maximum_worst_log10_regret"]
    )


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
    train_x = np.asarray([row.model_row() for row in train], dtype=float)
    train_y = np.asarray([row.action_label for row in train], dtype=object)
    train_regrets = _regression_targets(train)
    training_seed_cv = _training_seed_cv(train, train_x, train_y, train_regrets)
    model = _new_model()
    model.fit(train_x, train_regrets)
    train_predictions = _actions_from_scores(model.predict(train_x))
    holdout_x = np.asarray([row.model_row() for row in holdout], dtype=float)
    holdout_y = np.asarray([row.action_label for row in holdout], dtype=object)
    holdout_predictions = _actions_from_scores(model.predict(holdout_x))

    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / MODEL_FILENAME
    joblib.dump(model, model_path)
    evaluation = {
        "schema_version": "arac-outcome-selector-evaluation-v3",
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "actions": list(ACTION_NAMES),
        "model_selection": {
            "criterion": "leave-one-training-seed-out terminal regret",
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
    evaluation["holdout"]["gate_parameters"] = HOLDOUT_GATE_PARAMETERS
    evaluation["holdout"]["passed"] = _holdout_passed(evaluation["holdout"])
    evaluation_path = destination / EVALUATION_FILENAME
    _write_json(evaluation_path, evaluation)
    metadata = {
        "schema_version": SELECTOR_SCHEMA,
        "phase1_protocol": PHASE1_PROTOCOL,
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "actions": list(ACTION_NAMES),
        "model_type": "RandomForestRegressor",
        "model_parameters": MODEL_PARAMETERS,
        "model_sha256": file_sha256(model_path),
        "evaluation_sha256": file_sha256(evaluation_path),
        "training_record_hashes": [row.payload()["record_hash"] for row in train],
        "holdout_record_hashes": [row.payload()["record_hash"] for row in holdout],
        "training_label_counts": dict(sorted(Counter(train_y).items())),
        "holdout_label_counts": dict(sorted(Counter(holdout_y).items())),
        "labels_derived_from_action_outcomes": True,
        "predicts_action_log10_regret": True,
        "inference_contains_case_or_family_identity": False,
        "holdout_used_for_model_selection": False,
        "training_seed_cv_accuracy": training_seed_cv["accuracy"],
        "holdout_passed": evaluation["holdout"]["passed"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = destination / METADATA_FILENAME
    _write_json(metadata_path, metadata)
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
    "METADATA_FILENAME",
    "MODEL_FILENAME",
    "MODEL_PARAMETERS",
    "HOLDOUT_GATE_PARAMETERS",
    "OUTCOME_RECORD_SCHEMA",
    "SELECTOR_SCHEMA",
    "ActionOutcome",
    "OutcomeRecord",
    "OutcomeSelector",
    "Selector",
    "file_sha256",
    "fit_outcome_selector",
]
