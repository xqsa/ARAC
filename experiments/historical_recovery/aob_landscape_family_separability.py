"""Read-only audit of AOB family separability in frozen Phase-I checkpoints."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arac.evidence.phase1 import PHASE1_FEATURE_NAMES, PHASE1_PROTOCOL
from arac.runtime.contracts import canonical_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "experiments"
    / "historical_recovery"
    / "aob_landscape_family_separability_protocol.json"
)


@dataclass(frozen=True)
class FrozenDataset:
    features: np.ndarray
    feature_names: tuple[str, ...]
    case_ids: tuple[str, ...]
    families: tuple[str, ...]
    variant_indices: tuple[int, ...]
    seeds: tuple[int, ...]
    checkpoint_hashes: tuple[str, ...]
    current_receipt_hashes: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def _without(payload: Mapping[str, Any], key: str) -> tuple[dict[str, Any], Any]:
    body = dict(payload)
    claimed = body.pop(key, None)
    return body, claimed


def _validate_checkpoint_receipt(
    payload: Mapping[str, Any],
    *,
    case_id: str,
    seed: int,
    protocol: Mapping[str, Any],
) -> tuple[tuple[float, ...], str]:
    body, claimed_receipt_hash = _without(payload, "receipt_hash")
    if claimed_receipt_hash != canonical_sha256(body):
        raise ValueError(f"{case_id}/seed-{seed} checkpoint receipt hash drifted")
    if payload.get("schema_version") != "arac-independent-phase1-checkpoint-v1":
        raise ValueError(f"{case_id}/seed-{seed} checkpoint receipt schema drifted")
    if payload.get("case_id") != case_id or payload.get("run_seed") != seed:
        raise ValueError(f"{case_id}/seed-{seed} checkpoint identity drifted")
    if payload.get("max_fes") != protocol["total_budget_fes"]:
        raise ValueError(f"{case_id}/seed-{seed} total budget drifted")

    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{case_id}/seed-{seed} checkpoint payload is invalid")
    checkpoint_hash = canonical_sha256(checkpoint)
    if payload.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError(f"{case_id}/seed-{seed} checkpoint hash drifted")
    expected = {
        "schema_version": "arac-phase-checkpoint-v1",
        "protocol": protocol["phase1_protocol"],
        "run_seed": seed,
        "total_budget_fes": protocol["total_budget_fes"],
        "phase1_fes": protocol["phase1_fes"],
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"{case_id}/seed-{seed} checkpoint field drifted: {field}")

    names = tuple(checkpoint.get("feature_names", ()))
    values = tuple(float(value) for value in checkpoint.get("feature_values", ()))
    expected_names = tuple(protocol["expected_feature_names"])
    if names != expected_names or names != PHASE1_FEATURE_NAMES:
        raise ValueError(f"{case_id}/seed-{seed} feature schema drifted")
    if len(values) != len(names) or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{case_id}/seed-{seed} feature values are invalid")
    return values, checkpoint_hash


def _validate_current_receipt(
    payload: Mapping[str, Any],
    *,
    case_id: str,
    seed: int,
    checkpoint_hash: str,
    protocol: Mapping[str, Any],
) -> str:
    body, claimed_receipt_hash = _without(payload, "receipt_sha256")
    if claimed_receipt_hash != canonical_sha256(body):
        raise ValueError(f"{case_id}/seed-{seed} current E2E receipt hash drifted")
    expected = {
        "schema_version": "arac-current-arac-aob24-recovery-receipt-v1",
        "case_id": case_id,
        "run_seed": seed,
        "phase1_fes": protocol["phase1_fes"],
        "phase2_consumed_fes": protocol["total_budget_fes"] - protocol["phase1_fes"],
        "terminal_fes": protocol["total_budget_fes"],
        "terminal_state_finite": True,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{case_id}/seed-{seed} current receipt field drifted: {field}")
    if payload.get("phase1_checkpoint_hash") != checkpoint_hash:
        raise ValueError(f"{case_id}/seed-{seed} is not bound to the frozen checkpoint")
    if payload.get("action_checkpoint_hash") != checkpoint_hash:
        raise ValueError(f"{case_id}/seed-{seed} action checkpoint binding drifted")
    return str(claimed_receipt_hash)


def _expected_contexts(protocol: Mapping[str, Any]) -> list[tuple[str, str, int, int]]:
    return [
        (f"{family}{variant}", family, variant, int(seed))
        for variant in protocol["variant_indices"]
        for family in protocol["families"]
        for seed in protocol["seeds"]
    ]


def load_dataset(protocol_path: Path = DEFAULT_PROTOCOL) -> FrozenDataset:
    protocol = _load_json(protocol_path)
    if protocol["phase1_protocol"] != PHASE1_PROTOCOL:
        raise ValueError("protocol does not describe the current Phase-I implementation")
    contexts = _expected_contexts(protocol)
    if len(contexts) != protocol["expected_context_count"]:
        raise ValueError("protocol context count is inconsistent")

    checkpoint_root = REPOSITORY_ROOT / protocol["checkpoint_root"]
    current_root = REPOSITORY_ROOT / protocol["current_receipt_root"]
    expected_checkpoint_paths = {
        checkpoint_root / case_id / f"seed_{seed}" / "checkpoint.json"
        for case_id, _, _, seed in contexts
    }
    actual_checkpoint_paths = set(checkpoint_root.glob("*/seed_*/checkpoint.json"))
    if actual_checkpoint_paths != expected_checkpoint_paths:
        raise ValueError("checkpoint file coverage does not match the frozen protocol")
    expected_current_paths = {
        current_root / case_id / f"seed_{seed}" / "receipt.json"
        for case_id, _, _, seed in contexts
    }
    actual_current_paths = set(current_root.glob("*/seed_*/receipt.json"))
    if actual_current_paths != expected_current_paths:
        raise ValueError("current E2E receipt coverage does not match the frozen protocol")

    rows: list[tuple[float, ...]] = []
    case_ids: list[str] = []
    families: list[str] = []
    variants: list[int] = []
    seeds: list[int] = []
    checkpoint_hashes: list[str] = []
    current_receipt_hashes: list[str] = []
    for case_id, family, variant, seed in contexts:
        checkpoint_payload = _load_json(
            checkpoint_root / case_id / f"seed_{seed}" / "checkpoint.json"
        )
        feature_values, checkpoint_hash = _validate_checkpoint_receipt(
            checkpoint_payload,
            case_id=case_id,
            seed=seed,
            protocol=protocol,
        )
        current_payload = _load_json(
            current_root / case_id / f"seed_{seed}" / "receipt.json"
        )
        current_receipt_hash = _validate_current_receipt(
            current_payload,
            case_id=case_id,
            seed=seed,
            checkpoint_hash=checkpoint_hash,
            protocol=protocol,
        )
        rows.append(feature_values)
        case_ids.append(case_id)
        families.append(family)
        variants.append(variant)
        seeds.append(seed)
        checkpoint_hashes.append(checkpoint_hash)
        current_receipt_hashes.append(current_receipt_hash)

    features = np.asarray(rows, dtype=float)
    if features.shape != (protocol["expected_context_count"], len(PHASE1_FEATURE_NAMES)):
        raise ValueError("frozen feature matrix has the wrong shape")
    return FrozenDataset(
        features=features,
        feature_names=PHASE1_FEATURE_NAMES,
        case_ids=tuple(case_ids),
        families=tuple(families),
        variant_indices=tuple(variants),
        seeds=tuple(seeds),
        checkpoint_hashes=tuple(checkpoint_hashes),
        current_receipt_hashes=tuple(current_receipt_hashes),
    )


def variant_folds(
    variant_indices: Sequence[int],
    held_out_variants: Sequence[int],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    variants = np.asarray(variant_indices, dtype=int)
    return tuple(
        (np.flatnonzero(variants != held_out), np.flatnonzero(variants == held_out))
        for held_out in held_out_variants
    )


def _metric(value: float) -> float:
    return round(float(value), 12)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    recall = {
        label: _metric(matrix[index, index] / matrix[index].sum())
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": _metric(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _metric(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "family_recall": recall,
    }


def _pairwise_correlations(rows: np.ndarray) -> dict[str, float]:
    values = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            left_row = rows[left]
            right_row = rows[right]
            if np.std(left_row) == 0.0 or np.std(right_row) == 0.0:
                correlation = 1.0 if np.array_equal(left_row, right_row) else 0.0
            else:
                correlation = float(np.corrcoef(left_row, right_row)[0, 1])
            values.append(correlation)
    return {
        "mean_pairwise_importance_correlation": _metric(np.mean(values)),
        "minimum_pairwise_importance_correlation": _metric(np.min(values)),
    }


def _evaluate_group(
    dataset: FrozenDataset,
    *,
    feature_names: tuple[str, ...],
    feature_indices: np.ndarray,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = tuple(str(label) for label in protocol["families"])
    y = np.asarray(dataset.families)
    x = dataset.features[:, feature_indices]
    folds = variant_folds(dataset.variant_indices, protocol["variant_indices"])
    predictions = np.empty(len(y), dtype="<U1")
    fold_results = []
    importance_rows = []
    model_config = protocol["model"]

    for held_out, (train_indices, test_indices) in zip(protocol["variant_indices"], folds):
        expected_train = protocol["cross_validation"]["training_contexts_per_fold"]
        expected_test = protocol["cross_validation"]["test_contexts_per_fold"]
        if len(train_indices) != expected_train or len(test_indices) != expected_test:
            raise ValueError(f"fold {held_out} has the wrong train/test size")
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classify",
                    LogisticRegression(
                        solver=model_config["solver"],
                        l1_ratio=float(model_config["l1_ratio"]),
                        C=float(model_config["C"]),
                        tol=float(model_config["tol"]),
                        max_iter=int(model_config["max_iter"]),
                        random_state=int(model_config["random_state"]),
                    ),
                ),
            ]
        )
        model.fit(x[train_indices], y[train_indices])
        fold_predictions = model.predict(x[test_indices])
        predictions[test_indices] = fold_predictions
        classifier = model.named_steps["classify"]
        if tuple(classifier.classes_) != tuple(sorted(labels)):
            raise ValueError("classifier label order drifted")
        importance_rows.append(np.mean(np.abs(classifier.coef_), axis=0))
        fold_metrics = _metrics(y[test_indices], fold_predictions, labels)
        fold_metrics.update(
            {
                "held_out_variant_index": int(held_out),
                "training_context_count": len(train_indices),
                "test_context_count": len(test_indices),
                "test_cases": sorted(set(np.asarray(dataset.case_ids)[test_indices])),
            }
        )
        fold_results.append(fold_metrics)

    metrics = _metrics(y, predictions, labels)
    per_case = []
    for case_id in sorted(set(dataset.case_ids), key=lambda value: (value[0], int(value[1:]))):
        case_mask = np.asarray(dataset.case_ids) == case_id
        per_case.append(
            {
                "case_id": case_id,
                "family": case_id[0],
                "accuracy": _metric(accuracy_score(y[case_mask], predictions[case_mask])),
                "correct_count": int(np.sum(y[case_mask] == predictions[case_mask])),
                "context_count": int(np.sum(case_mask)),
            }
        )

    importance = np.asarray(importance_rows)
    feature_importance = [
        {
            "feature": name,
            "mean_abs_standardized_coefficient": _metric(np.mean(importance[:, index])),
            "std_abs_standardized_coefficient": _metric(np.std(importance[:, index])),
        }
        for index, name in enumerate(feature_names)
    ]
    feature_importance.sort(
        key=lambda row: (-row["mean_abs_standardized_coefficient"], row["feature"])
    )
    result = {
        **metrics,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "folds": fold_results,
        "per_case": per_case,
        "coefficient_stability": _pairwise_correlations(importance),
        "top_features": feature_importance[: min(10, len(feature_importance))],
        "feature_importance": feature_importance,
    }
    rows = [
        {
            "case_id": dataset.case_ids[index],
            "run_seed": dataset.seeds[index],
            "held_out_variant_index": dataset.variant_indices[index],
            "true_family": dataset.families[index],
            "predicted_family": str(predictions[index]),
            "correct": dataset.families[index] == predictions[index],
        }
        for index in range(len(y))
    ]
    return result, rows


def run_audit(protocol_path: Path = DEFAULT_PROTOCOL) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = _load_json(protocol_path)
    dataset = load_dataset(protocol_path)
    forbidden = set(protocol["identity_fields_forbidden_as_model_inputs"])
    if forbidden.intersection(dataset.feature_names):
        raise ValueError("an identity or outcome field appears in the model input schema")

    group_results = {}
    prediction_rows = []
    for group_name, bounds in protocol["feature_groups"].items():
        start = int(bounds["start"])
        stop = int(bounds["stop"])
        if not 0 <= start < stop <= len(dataset.feature_names):
            raise ValueError(f"invalid feature group bounds: {group_name}")
        excluded = set(bounds.get("exclude", ()))
        unknown_exclusions = excluded.difference(dataset.feature_names[start:stop])
        if unknown_exclusions:
            raise ValueError(f"unknown exclusions in feature group {group_name}")
        indices = np.asarray(
            [
                index
                for index in range(start, stop)
                if dataset.feature_names[index] not in excluded
            ],
            dtype=int,
        )
        names = tuple(dataset.feature_names[index] for index in indices)
        result, rows = _evaluate_group(
            dataset,
            feature_names=names,
            feature_indices=indices,
            protocol=protocol,
        )
        result["analysis_role"] = bounds["role"]
        result["excluded_features"] = sorted(excluded)
        group_results[group_name] = result
        prediction_rows.extend({"feature_group": group_name, **row} for row in rows)

    primary_name = protocol["primary_feature_group"]
    primary = group_results[primary_name]
    thresholds = protocol["separability_gate"]
    checks = {
        "balanced_accuracy": (
            primary["balanced_accuracy"] >= thresholds["minimum_balanced_accuracy"]
        ),
        "all_family_recalls": (
            min(primary["family_recall"].values()) >= thresholds["minimum_family_recall"]
        ),
        "all_fold_balanced_accuracies": (
            min(row["balanced_accuracy"] for row in primary["folds"])
            >= thresholds["minimum_fold_balanced_accuracy"]
        ),
    }
    dataset_binding = [
        {
            "case_id": dataset.case_ids[index],
            "run_seed": dataset.seeds[index],
            "checkpoint_hash": dataset.checkpoint_hashes[index],
            "current_receipt_hash": dataset.current_receipt_hashes[index],
        }
        for index in range(len(dataset.case_ids))
    ]
    action_mapping = protocol["family_action_mapping"]
    correct_context_count = sum(
        row["correct"]
        for row in prediction_rows
        if row["feature_group"] == primary_name
    )
    case_majority_correct_count = sum(
        row["correct_count"] > row["context_count"] / 2 for row in primary["per_case"]
    )
    audit_body = {
        "schema_version": "arac-aob-landscape-family-separability-audit-v1",
        "protocol_sha256": canonical_sha256(protocol),
        "input_audit": {
            "context_count": len(dataset.case_ids),
            "case_count": len(set(dataset.case_ids)),
            "seed_count_per_case": len(set(dataset.seeds)),
            "feature_count": len(dataset.feature_names),
            "phase1_fes": protocol["phase1_fes"],
            "checkpoint_hash_valid_count": len(dataset.checkpoint_hashes),
            "current_e2e_binding_count": len(dataset.current_receipt_hashes),
            "all_feature_values_finite": bool(np.isfinite(dataset.features).all()),
            "optimizer_or_objective_evaluations_executed": False,
            "dataset_binding_sha256": canonical_sha256(dataset_binding),
        },
        "cross_validation": protocol["cross_validation"],
        "model": protocol["model"],
        "model_input_fields": list(dataset.feature_names),
        "excluded_identity_and_outcome_fields": sorted(forbidden),
        "chance_balanced_accuracy": 1.0 / len(protocol["families"]),
        "feature_groups": group_results,
        "primary_feature_group": primary_name,
        "mapped_action_label_selection": {
            "family_action_mapping": action_mapping,
            "accuracy": primary["accuracy"],
            "correct_context_count": correct_context_count,
            "context_count": len(dataset.case_ids),
            "case_majority_correct_count": case_majority_correct_count,
            "case_count": len(set(dataset.case_ids)),
            "label_semantics": (
                "Agreement with the predefined family-to-action mapping, not terminal oracle "
                "action optimality."
            ),
        },
        "separability_gate": {
            "passed": all(checks.values()),
            "thresholds": thresholds,
            "checks": checks,
        },
        "scientific_boundaries": protocol["scientific_boundaries"],
    }
    audit = {**audit_body, "audit_sha256": canonical_sha256(audit_body)}
    return audit, prediction_rows


def render_report(audit: Mapping[str, Any]) -> str:
    input_audit = audit["input_audit"]
    gate = audit["separability_gate"]
    primary = audit["feature_groups"][audit["primary_feature_group"]]
    mapped = audit["mapped_action_label_selection"]
    lines = [
        "# AOB Phase-I landscape-family separability audit",
        "",
        f"- Frozen contexts: **{input_audit['context_count']}/600**",
        f"- Current E2E checkpoint bindings: **{input_audit['current_e2e_binding_count']}/600**",
        f"- Phase-I budget per context: **{input_audit['phase1_fes']:,} FE**",
        "- New optimizer/objective evaluations: **0**",
        f"- Primary separability gate: **{str(gate['passed']).lower()}**",
        f"- Predefined mapped-action labels correct: **{mapped['correct_context_count']}/"
        f"{mapped['context_count']}**",
        f"- Correct by per-case majority: **{mapped['case_majority_correct_count']}/"
        f"{mapped['case_count']}**",
        "",
        "## Feature-group ablation",
        "",
        "| Feature group | Role | Features | Accuracy | Balanced accuracy | Minimum family recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, result in audit["feature_groups"].items():
        lines.append(
            f"| {name} | {result['analysis_role']} | {result['feature_count']} | "
            f"{result['accuracy']:.3f} | "
            f"{result['balanced_accuracy']:.3f} | {min(result['family_recall'].values()):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Primary all-40 result",
            "",
            f"- Accuracy: **{primary['accuracy']:.3f}**",
            f"- Balanced accuracy: **{primary['balanced_accuracy']:.3f}**",
            "- Label order: `A, E, R, S`",
            "",
            "| True / predicted | A | E | R | S | Recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = ("A", "E", "R", "S")
    for index, label in enumerate(labels):
        row = primary["confusion_matrix"][index]
        lines.append(
            f"| {label} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | "
            f"{primary['family_recall'][label]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Held-out variant folds",
            "",
            "| Variant | Test cases | Accuracy | Balanced accuracy |",
            "|---:|---|---:|---:|",
        ]
    )
    for fold in primary["folds"]:
        lines.append(
            f"| {fold['held_out_variant_index']} | {', '.join(fold['test_cases'])} | "
            f"{fold['accuracy']:.3f} | {fold['balanced_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Main standardized coefficients",
            "",
            "| Feature | Mean absolute coefficient | Fold std |",
            "|---|---:|---:|",
        ]
    )
    for row in primary["top_features"]:
        lines.append(
            f"| {row['feature']} | {row['mean_abs_standardized_coefficient']:.4f} | "
            f"{row['std_abs_standardized_coefficient']:.4f} |"
        )
    lines.extend(["", "## Scientific boundary", ""])
    lines.extend(f"- {boundary}" for boundary in audit["scientific_boundaries"])
    lines.append("")
    return "\n".join(lines)


def _prediction_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "feature_group",
        "case_id",
        "run_seed",
        "held_out_variant_index",
        "true_family",
        "predicted_family",
        "correct",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _outputs(
    audit: Mapping[str, Any], prediction_rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str, str]:
    return (
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        render_report(audit),
        _prediction_csv(prediction_rows),
    )


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args(argv)

    protocol_path = args.protocol.resolve()
    protocol = _load_json(protocol_path)
    output_root = REPOSITORY_ROOT / protocol["output_root"]
    output_paths = (
        output_root / "audit.json",
        output_root / "report.md",
        output_root / "predictions.csv",
    )
    audit, predictions = run_audit(protocol_path)
    expected = _outputs(audit, predictions)
    if args.command == "verify":
        for path, content in zip(output_paths, expected):
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"stale or missing separability output: {path}")
    else:
        for path, content in zip(output_paths, expected):
            _write_atomic(path, content)

    print(
        json.dumps(
            {
                "context_count": audit["input_audit"]["context_count"],
                "balanced_accuracy": audit["feature_groups"]["all_40"][
                    "balanced_accuracy"
                ],
                "separability_gate_passed": audit["separability_gate"]["passed"],
            },
            sort_keys=True,
        )
    )
    return int(args.require_passed and not audit["separability_gate"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
