"""Identity-blind selector validation with AOB case-grouped holdouts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from arac.analysis.outcome_selector import (
    MODEL_PARAMETERS,
    REGRET_TARGET_CAP,
    ActionOutcome,
    OutcomeRecord,
)
from arac.evidence.phase1 import PHASE1_FEATURE_NAMES, PHASE1_PROTOCOL
from arac.runtime.contracts import ACTION_NAMES, canonical_sha256


GROUPED_EVALUATION_SCHEMA = "arac-grouped-outcome-selector-evaluation-v1"
GROUPED_SELECTOR_SCHEMA = "arac-grouped-outcome-selector-v1"
GROUPED_MODEL_FILENAME = "grouped_outcome_selector.joblib"
GROUPED_METADATA_FILENAME = "grouped_outcome_selector.json"
GROUPED_EVALUATION_FILENAME = "grouped_evaluation.json"
FROZEN_UNCERTAINTY_PENALTY = 0.0
TAIL_RISK_FRACTION = 0.05
AOB_CASE_PATTERN = re.compile(r"^([AERS])([1-6])$")
EXPECTED_AOB_CASES = tuple(
    f"{family}{index}" for family in "AERS" for index in range(1, 7)
)
GROUPED_GATE_PARAMETERS = {
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


def load_outcome_records(path: Path) -> tuple[OutcomeRecord, ...]:
    source = Path(path)
    records = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid outcome JSON on line {line_number}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"outcome line {line_number} must be a JSON object")
        records.append(OutcomeRecord.from_payload(payload))
    if not records:
        raise ValueError("outcome matrix is empty")
    keys = [(record.case_id, record.run_seed) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("outcome matrix contains duplicate case/seed contexts")
    return tuple(records)


def _case_parts(case_id: str) -> tuple[str, int]:
    match = AOB_CASE_PATTERN.fullmatch(case_id)
    if match is None:
        raise ValueError(f"unsupported AOB case identity: {case_id}")
    return match.group(1), int(match.group(2))


def build_grouped_folds(
    records: Sequence[OutcomeRecord],
    scheme: str,
) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]:
    rows = tuple(records)
    if scheme == "leave_one_case_out":
        def group_key(record: OutcomeRecord) -> str:
            return record.case_id
    elif scheme == "leave_one_variant_index_out":
        def group_key(record: OutcomeRecord) -> str:
            return str(_case_parts(record.case_id)[1])
    else:
        raise ValueError("unknown grouped selector validation scheme")
    groups = sorted({group_key(record) for record in rows}, key=lambda value: (len(value), value))
    folds = []
    for group in groups:
        test = tuple(index for index, record in enumerate(rows) if group_key(record) == group)
        train = tuple(index for index in range(len(rows)) if index not in set(test))
        if not train or not test:
            raise ValueError(f"grouped fold {group} has an empty train or test partition")
        folds.append((group, train, test))
    return tuple(folds)


def _regression_targets(records: Sequence[OutcomeRecord]) -> np.ndarray:
    rows = []
    for record in records:
        errors = np.asarray([outcome.final_error for outcome in record.outcomes], dtype=float)
        oracle = float(np.min(errors))
        log_regret = np.log10(errors + 1.0) - math.log10(oracle + 1.0)
        rows.append(np.minimum(log_regret, REGRET_TARGET_CAP))
    return np.asarray(rows, dtype=float)


def _score_components(model: RandomForestRegressor, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(model.predict(features), dtype=float)
    expected_shape = (len(features), len(ACTION_NAMES))
    if means.shape != expected_shape or not np.all(np.isfinite(means)):
        raise RuntimeError("grouped selector produced invalid mean scores")
    tree_scores = np.asarray(
        [np.asarray(tree.predict(features), dtype=float) for tree in model.estimators_],
        dtype=float,
    )
    if tree_scores.shape != (len(model.estimators_), *expected_shape):
        raise RuntimeError("grouped selector produced invalid tree scores")
    return means, np.std(tree_scores, axis=0)


def _predict(model: RandomForestRegressor, features: np.ndarray) -> np.ndarray:
    means, uncertainty = _score_components(model, features)
    scores = means + FROZEN_UNCERTAINTY_PENALTY * uncertainty
    return np.asarray(ACTION_NAMES, dtype=object)[np.argmin(scores, axis=1)]


def _terminal_regret(
    records: Sequence[OutcomeRecord], predictions: Sequence[str]
) -> dict[str, float]:
    log_regrets = []
    for record, prediction in zip(records, predictions, strict=True):
        errors = {outcome.action_name: outcome.final_error for outcome in record.outcomes}
        selected = errors[str(prediction)]
        oracle = min(errors.values())
        log_regrets.append(math.log10(selected + 1.0) - math.log10(oracle + 1.0))
    values = np.asarray(log_regrets, dtype=float)
    ratios = np.power(10.0, values)
    tail_count = max(1, math.ceil(len(values) * TAIL_RISK_FRACTION))
    return {
        "mean_log10_regret": float(np.mean(values)),
        "p95_log10_regret": float(np.quantile(values, 0.95)),
        "cvar95_log10_regret": float(np.mean(np.sort(values)[-tail_count:])),
        "worst_log10_regret": float(np.max(values)),
        "mean_selected_to_oracle_ratio": float(np.mean(ratios)),
        "worst_selected_to_oracle_ratio": float(np.max(ratios)),
        "within_1_percent_of_oracle": float(np.mean(ratios <= 1.01)),
        "within_5_percent_of_oracle": float(np.mean(ratios <= 1.05)),
        "within_10_percent_of_oracle": float(np.mean(ratios <= 1.10)),
    }


def selector_metrics(
    records: Sequence[OutcomeRecord], predictions: Sequence[str]
) -> dict[str, object]:
    rows = tuple(records)
    predicted = np.asarray(tuple(str(value) for value in predictions), dtype=object)
    targets = np.asarray([record.action_label for record in rows], dtype=object)
    if len(rows) != len(predicted):
        raise ValueError("selector records and predictions are not aligned")
    confusion = confusion_matrix(targets, predicted, labels=ACTION_NAMES)
    return {
        "record_count": len(rows),
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "confusion_matrix": confusion.tolist(),
        "per_action_recall": {
            action: (
                float(confusion[index, index] / np.sum(confusion[index]))
                if np.sum(confusion[index])
                else None
            )
            for index, action in enumerate(ACTION_NAMES)
        },
        "prediction_counts": dict(sorted(Counter(str(value) for value in predicted).items())),
        "terminal_regret": _terminal_regret(rows, predicted),
    }


def metrics_pass_gate(metrics: Mapping[str, object]) -> bool:
    regret = metrics.get("terminal_regret")
    if not isinstance(regret, Mapping):
        raise ValueError("selector terminal regret metrics are missing")
    return bool(
        float(metrics["accuracy"]) >= GROUPED_GATE_PARAMETERS["minimum_accuracy"]
        and float(metrics["balanced_accuracy"])
        >= GROUPED_GATE_PARAMETERS["minimum_balanced_accuracy"]
        and float(regret["mean_log10_regret"])
        <= GROUPED_GATE_PARAMETERS["maximum_mean_log10_regret"]
        and float(regret["worst_log10_regret"])
        <= GROUPED_GATE_PARAMETERS["maximum_worst_log10_regret"]
    )


def evaluate_grouped_scheme(
    records: Sequence[OutcomeRecord], scheme: str
) -> dict[str, object]:
    rows = tuple(records)
    features = np.asarray([record.model_row() for record in rows], dtype=float)
    regrets = _regression_targets(rows)
    predictions = np.empty(len(rows), dtype=object)
    fold_rows = []
    for group, train_indices, test_indices in build_grouped_folds(rows, scheme):
        model = RandomForestRegressor(**MODEL_PARAMETERS)
        model.fit(features[list(train_indices)], regrets[list(train_indices)])
        fold_predictions = _predict(model, features[list(test_indices)])
        predictions[list(test_indices)] = fold_predictions
        test_records = tuple(rows[index] for index in test_indices)
        fold_metrics = selector_metrics(test_records, fold_predictions)
        fold_rows.append(
            {
                "held_out_group": group,
                "held_out_cases": sorted({record.case_id for record in test_records}),
                "training_record_count": len(train_indices),
                "test_record_count": len(test_indices),
                "metrics": fold_metrics,
            }
        )
    aggregate = selector_metrics(rows, predictions)
    aggregate["gate_parameters"] = dict(GROUPED_GATE_PARAMETERS)
    aggregate["passed"] = metrics_pass_gate(aggregate)
    return {
        "scheme": scheme,
        "fold_count": len(fold_rows),
        "folds": fold_rows,
        "aggregate": aggregate,
        "errors": [
            {
                "case_id": record.case_id,
                "run_seed": record.run_seed,
                "actual_action": record.action_label,
                "predicted_action": str(predictions[index]),
                "log10_regret": _terminal_regret((record,), (str(predictions[index]),))[
                    "mean_log10_regret"
                ],
            }
            for index, record in enumerate(rows)
            if record.action_label != predictions[index]
        ],
    }


def fixed_action_baselines(records: Sequence[OutcomeRecord]) -> dict[str, object]:
    rows = tuple(records)
    return {
        action: selector_metrics(rows, (action,) * len(rows)) for action in ACTION_NAMES
    }


def build_grouped_evaluation(records: Sequence[OutcomeRecord]) -> dict[str, object]:
    rows = tuple(records)
    cases = {record.case_id for record in rows}
    seeds_by_case = {
        case: {record.run_seed for record in rows if record.case_id == case}
        for case in cases
    }
    seed_sets = {tuple(sorted(seeds)) for seeds in seeds_by_case.values()}
    if (
        cases != set(EXPECTED_AOB_CASES)
        or len(seed_sets) != 1
        or len(next(iter(seed_sets), ())) != 25
        or len(rows) != 600
    ):
        raise ValueError("grouped selector requires one complete AOB-24 by 25-seed matrix")
    primary = evaluate_grouped_scheme(rows, "leave_one_case_out")
    secondary = evaluate_grouped_scheme(rows, "leave_one_variant_index_out")
    primary_metrics = primary["aggregate"]
    secondary_metrics = secondary["aggregate"]
    if not isinstance(primary_metrics, Mapping) or not isinstance(secondary_metrics, Mapping):
        raise RuntimeError("grouped selector aggregate metrics are invalid")
    body = {
        "schema_version": GROUPED_EVALUATION_SCHEMA,
        "phase1_protocol": PHASE1_PROTOCOL,
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "actions": list(ACTION_NAMES),
        "record_count": len(rows),
        "case_count": len(cases),
        "seed_count": len(next(iter(seed_sets))),
        "model_type": "RandomForestRegressor",
        "model_parameters": dict(MODEL_PARAMETERS),
        "regret_target_cap": REGRET_TARGET_CAP,
        "uncertainty_penalty": FROZEN_UNCERTAINTY_PENALTY,
        "identity_features_used": False,
        "primary": primary,
        "secondary": secondary,
        "fixed_action_baselines": fixed_action_baselines(rows),
        "oracle_label_counts": dict(sorted(Counter(record.action_label for record in rows).items())),
        "selector_freeze_authorized": bool(
            primary_metrics.get("passed") is True and secondary_metrics.get("passed") is True
        ),
    }
    return {**body, "evaluation_hash": canonical_sha256(body)}


@dataclass(frozen=True)
class GroupedOutcomeSelector:
    model: RandomForestRegressor
    metadata: Mapping[str, object]

    @classmethod
    def load(cls, directory: Path) -> GroupedOutcomeSelector:
        root = Path(directory).resolve()
        metadata_path = root / GROUPED_METADATA_FILENAME
        model_path = root / GROUPED_MODEL_FILENAME
        evaluation_path = root / GROUPED_EVALUATION_FILENAME
        if not evaluation_path.is_file():
            raise ValueError("grouped selector evaluation artifact is missing")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("grouped selector metadata must be a JSON object")
        metadata = dict(payload)
        claimed_hash = metadata.pop("metadata_hash", None)
        if claimed_hash != canonical_sha256(metadata):
            raise ValueError("grouped selector metadata hash drifted")
        expected = {
            "schema_version": GROUPED_SELECTOR_SCHEMA,
            "phase1_protocol": PHASE1_PROTOCOL,
            "feature_names": list(PHASE1_FEATURE_NAMES),
            "actions": list(ACTION_NAMES),
            "model_type": "RandomForestRegressor",
            "model_parameters": dict(MODEL_PARAMETERS),
            "regret_target_cap": REGRET_TARGET_CAP,
            "uncertainty_penalty": FROZEN_UNCERTAINTY_PENALTY,
            "identity_features_used": False,
            "grouped_gate_passed": True,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("grouped selector metadata contract drifted")
        if metadata.get("model_sha256") != file_sha256(model_path):
            raise ValueError("grouped selector model hash drifted")
        if metadata.get("evaluation_sha256") != file_sha256(evaluation_path):
            raise ValueError("grouped selector evaluation hash drifted")
        model = joblib.load(model_path)
        if int(getattr(model, "n_features_in_", -1)) != len(PHASE1_FEATURE_NAMES):
            raise ValueError("grouped selector model width drifted")
        if int(getattr(model, "n_outputs_", -1)) != len(ACTION_NAMES):
            raise ValueError("grouped selector output width drifted")
        return cls(model=model, metadata=payload)

    def select(
        self, feature_names: Sequence[str], feature_values: Sequence[float]
    ) -> str:
        names = tuple(str(name) for name in feature_names)
        values = tuple(float(value) for value in feature_values)
        if names != PHASE1_FEATURE_NAMES:
            raise ValueError("grouped selector feature schema drifted")
        if len(values) != len(names) or not all(math.isfinite(value) for value in values):
            raise ValueError("grouped selector feature values are invalid")
        return str(_predict(self.model, np.asarray([values], dtype=float))[0])


def fit_frozen_grouped_selector(
    records: Sequence[OutcomeRecord],
) -> RandomForestRegressor:
    rows = tuple(records)
    if not rows:
        raise ValueError("selector training records must be non-empty")
    model = RandomForestRegressor(**MODEL_PARAMETERS)
    model.fit(
        np.asarray([record.model_row() for record in rows], dtype=float),
        _regression_targets(rows),
    )
    return model


__all__ = [
    "AOB_CASE_PATTERN",
    "ActionOutcome",
    "FROZEN_UNCERTAINTY_PENALTY",
    "EXPECTED_AOB_CASES",
    "GROUPED_EVALUATION_SCHEMA",
    "GROUPED_GATE_PARAMETERS",
    "GROUPED_EVALUATION_FILENAME",
    "GROUPED_METADATA_FILENAME",
    "GROUPED_MODEL_FILENAME",
    "GROUPED_SELECTOR_SCHEMA",
    "GroupedOutcomeSelector",
    "OutcomeRecord",
    "build_grouped_evaluation",
    "build_grouped_folds",
    "file_sha256",
    "fit_frozen_grouped_selector",
    "load_outcome_records",
    "metrics_pass_gate",
    "selector_metrics",
]
