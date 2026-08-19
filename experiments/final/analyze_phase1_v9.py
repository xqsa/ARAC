"""Read-only Phase-I feature ablation for the frozen action calibration data.

This module deliberately does not import the experiment coordinator or execute
an action.  It reads outcome records and their immutable Phase-I checkpoints,
fits diagnostic models in leave-one-run-seed-out folds, and prints the result
to stdout.  No artifact or report file is written by this script.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

from arac.evidence.mechanism_features import (
    CTP_COVER_FEATURE_NAMES,
    DISAGREEMENT_FEATURE_NAMES,
    TOPOLOGY_FEATURE_NAMES,
    summarize_ctp_cover,
    summarize_relation_disagreement,
    summarize_relation_topology,
)
from arac.runtime.contracts import RelationEvidence


ACTIONS = ("ctp", "smp", "gcb", "aor")
DEFAULT_RANDOM_STATE = 20260730
DEFAULT_TARGET_CAP: float | None = 0.25
DEFAULT_ESTIMATORS = 300
TAIL_THRESHOLD = 0.25


def _parse_target_cap(value: str) -> float | None:
    if value.strip().lower() in {"none", "uncapped", "null"}:
        return None
    try:
        cap = float(value)
    except ValueError as exc:
        raise ValueError("target cap must be a positive number or 'none'") from exc
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("target cap must be a positive finite number")
    return cap


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"outcome row at {path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"outcome file is empty: {path}")
    return rows


def _checkpoint_path(root: Path, row: dict[str, Any]) -> Path:
    return (
        root
        / "checkpoints"
        / str(row["case_id"])
        / f"seed_{int(row['run_seed'])}"
        / "checkpoint.json"
    )


def _load_checkpoint(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = _checkpoint_path(root, row)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint is unreadable: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("checkpoint"), dict):
        raise ValueError(f"checkpoint payload is invalid: {path}")
    checkpoint = payload["checkpoint"]
    if payload.get("case_id") != row.get("case_id"):
        raise ValueError(f"checkpoint case mismatch: {path}")
    if int(payload.get("run_seed", -1)) != int(row.get("run_seed", -1)):
        raise ValueError(f"checkpoint seed mismatch: {path}")
    if payload.get("checkpoint_hash") != row.get("checkpoint_hash"):
        raise ValueError(f"checkpoint hash mismatch: {path}")
    if checkpoint.get("feature_names") != row.get("feature_names"):
        raise ValueError(f"checkpoint feature schema mismatch: {path}")
    return checkpoint


def _checkpoint_relations(checkpoint: dict[str, Any]) -> tuple[RelationEvidence, ...]:
    raw_relations = checkpoint.get("relations", [])
    if not isinstance(raw_relations, list):
        raise ValueError("checkpoint relations must be a list")
    try:
        return tuple(
            RelationEvidence(
                left_block=int(item["left_block"]),
                right_block=int(item["right_block"]),
                strength=float(item["strength"]),
                disagreement=float(item["disagreement"]),
            )
            for item in raw_relations
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint relation evidence is invalid") from exc


def _checkpoint_blocks(checkpoint: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    raw_blocks = checkpoint.get("blocks", [])
    if not isinstance(raw_blocks, list):
        raise ValueError("checkpoint blocks must be a list")
    try:
        blocks = tuple(tuple(int(value) for value in block) for block in raw_blocks)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint blocks are invalid") from exc
    if not blocks or any(not block for block in blocks):
        raise ValueError("checkpoint blocks are invalid")
    return blocks


def _checkpoint_disagreement_features(checkpoint: dict[str, Any]) -> tuple[float, ...]:
    values = summarize_relation_disagreement(_checkpoint_relations(checkpoint))
    if len(values) != len(DISAGREEMENT_FEATURE_NAMES):
        raise ValueError("disagreement feature schema drifted")
    return values


def _checkpoint_topology_features(checkpoint: dict[str, Any]) -> tuple[float, ...]:
    values = summarize_relation_topology(
        _checkpoint_blocks(checkpoint), _checkpoint_relations(checkpoint)
    )
    if len(values) != len(TOPOLOGY_FEATURE_NAMES):
        raise ValueError("topology feature schema drifted")
    return values


def _checkpoint_cover_features(checkpoint: dict[str, Any]) -> tuple[float, ...]:
    values = summarize_ctp_cover(
        _checkpoint_blocks(checkpoint), _checkpoint_relations(checkpoint)
    )
    if len(values) != len(CTP_COVER_FEATURE_NAMES):
        raise ValueError("cover feature schema drifted")
    return values


def _load_dataset(outcome_path: Path, checkpoint_root: Path) -> dict[str, Any]:
    rows = _read_jsonl(outcome_path)
    feature_names = tuple(str(name) for name in rows[0].get("feature_names", ()))
    if not feature_names:
        raise ValueError("outcome rows have no feature schema")
    cases: list[str] = []
    seeds: list[int] = []
    base_rows: list[tuple[float, ...]] = []
    errors: list[tuple[float, ...]] = []
    disagreement_rows: list[tuple[float, ...]] = []
    topology_rows: list[tuple[float, ...]] = []
    cover_rows: list[tuple[float, ...]] = []
    for row in rows:
        if tuple(str(name) for name in row.get("feature_names", ())) != feature_names:
            raise ValueError("outcome feature schema is not constant")
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, list) or tuple(item.get("action_name") for item in outcomes) != ACTIONS:
            raise ValueError("outcome row does not contain the fixed four-action matrix")
        checkpoint = _load_checkpoint(checkpoint_root, row)
        cases.append(str(row["case_id"]))
        seeds.append(int(row["run_seed"]))
        base_rows.append(tuple(float(value) for value in row["feature_values"]))
        errors.append(tuple(float(item["final_error"]) for item in outcomes))
        disagreement_rows.append(_checkpoint_disagreement_features(checkpoint))
        topology_rows.append(_checkpoint_topology_features(checkpoint))
        cover_rows.append(_checkpoint_cover_features(checkpoint))
    error_array = np.asarray(errors, dtype=float)
    if np.any(~np.isfinite(error_array)) or np.any(error_array < 0.0):
        raise ValueError("outcome errors are invalid")
    return {
        "feature_names": feature_names,
        "cases": np.asarray(cases, dtype=object),
        "seeds": np.asarray(seeds, dtype=int),
        "base": np.asarray(base_rows, dtype=float),
        "disagreement": np.asarray(disagreement_rows, dtype=float),
        "topology": np.asarray(topology_rows, dtype=float),
        "cover": np.asarray(cover_rows, dtype=float),
        "errors": error_array,
    }


def _metrics(data: dict[str, Any], predictions: np.ndarray) -> dict[str, Any]:
    errors = data["errors"]
    oracle = np.min(errors, axis=1)
    regrets = np.log10((errors[np.arange(len(errors)), predictions] + 1.0) / (oracle + 1.0))
    labels = np.argmin(errors, axis=1)
    present = sorted(set(int(value) for value in labels))
    recalls = [
        float(np.mean(predictions[labels == action] == action))
        for action in present
    ]
    return {
        "record_count": int(len(predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "mean_log10_regret": float(np.mean(regrets)),
        "worst_log10_regret": float(np.max(regrets)),
        "p95_log10_regret": float(np.quantile(regrets, 0.95)),
        "tail_count_regret_gt_0_25": int(np.sum(regrets > TAIL_THRESHOLD)),
        "tail_fraction_regret_gt_0_25": float(np.mean(regrets > TAIL_THRESHOLD)),
        "within_10_percent_of_oracle": float(np.mean(regrets <= math.log10(1.10))),
    }


def _fit_cv(
    data: dict[str, Any],
    features: np.ndarray,
    *,
    estimators: int,
    target_cap: float | None,
) -> tuple[dict[str, Any], np.ndarray]:
    errors = data["errors"]
    targets = np.log10((errors + 1.0) / (np.min(errors, axis=1, keepdims=True) + 1.0))
    if target_cap is not None:
        targets = np.minimum(targets, float(target_cap))
    predictions = np.empty(len(errors), dtype=int)
    for seed in sorted(set(int(value) for value in data["seeds"])):
        validation = data["seeds"] == seed
        training = ~validation
        model = RandomForestRegressor(
            n_estimators=int(estimators),
            min_samples_leaf=3,
            max_features=0.5,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(features[training], targets[training])
        predictions[validation] = np.argmin(model.predict(features[validation]), axis=1)
    return _metrics(data, predictions), predictions


def _group_metrics(data: dict[str, Any], predictions: np.ndarray, key_values: Iterable[str]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(key_values):
        grouped[str(key)].append(index)
    result: dict[str, Any] = {}
    for key, indices in sorted(grouped.items()):
        subset = {name: value[indices] if isinstance(value, np.ndarray) and value.shape[0] == len(predictions) else value for name, value in data.items()}
        result[key] = _metrics(subset, predictions[indices])
    return result


def _format_table(results: dict[str, Any]) -> str:
    lines = ["feature_set                 n    acc    bacc   mean    p95     worst   tail"]
    for name, payload in results["feature_sets"].items():
        if payload.get("status") != "ok":
            lines.append(f"{name:28s} skipped: {payload.get('reason', 'unknown')}")
            continue
        m = payload["metrics"]
        lines.append(
            f"{name:28s} {m['record_count']:4d} {m['accuracy']:.3f} {m['balanced_accuracy']:.3f} "
            f"{m['mean_log10_regret']:.4f} {m['p95_log10_regret']:.4f} "
            f"{m['worst_log10_regret']:.4f} {m['tail_count_regret_gt_0_25']:4d}"
        )
    return "\n".join(lines)


def analyze(
    *,
    outcome_path: Path,
    checkpoint_root: Path,
    estimators: int = DEFAULT_ESTIMATORS,
    target_cap: float | None = DEFAULT_TARGET_CAP,
) -> dict[str, Any]:
    if target_cap is not None:
        target_cap = _parse_target_cap(str(target_cap))
    data = _load_dataset(outcome_path, checkpoint_root)
    feature_sets: dict[str, np.ndarray] = {
        "base36": data["base"],
        "disagreement_only": data["disagreement"],
        "topology_only": data["topology"],
        "cover_only": data["cover"],
        "base36_plus_disagreement": np.column_stack((data["base"], data["disagreement"])),
        "base36_plus_topology": np.column_stack((data["base"], data["topology"])),
        "base36_plus_cover": np.column_stack((data["base"], data["cover"])),
        "base36_plus_all": np.column_stack(
            (data["base"], data["disagreement"], data["topology"], data["cover"])
        ),
    }
    output: dict[str, Any] = {
        "schema_version": "arac-phase1-v9-offline-analysis-v1",
        "outcome_path": str(outcome_path.resolve()),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "record_count": int(len(data["errors"])),
        "run_seeds": sorted(int(value) for value in set(data["seeds"])),
        "target_cap": target_cap,
        "feature_widths": {name: int(values.shape[1]) for name, values in feature_sets.items()},
        "feature_names": {
            "base36": list(data["feature_names"]),
            "disagreement": list(DISAGREEMENT_FEATURE_NAMES),
            "topology": list(TOPOLOGY_FEATURE_NAMES),
            "cover": list(CTP_COVER_FEATURE_NAMES),
        },
        "feature_sets": {},
        "by_regime": {},
        "by_case": {},
        "progress": {
            "status": "skipped",
            "reason": "legacy receipts do not preserve per-evaluation trajectory or population state",
        },
    }
    for name, values in feature_sets.items():
        metrics, predictions = _fit_cv(
            data,
            values,
            estimators=estimators,
            target_cap=target_cap,
        )
        output["feature_sets"][name] = {
            "status": "ok",
            "metrics": metrics,
            "by_regime": _group_metrics(
                data,
                predictions,
                (str(case)[0] for case in data["cases"]),
            ),
            "by_case": _group_metrics(data, predictions, (str(case) for case in data["cases"])),
        }
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=Path("artifacts/outcome_calibration_v5/train/outcomes.jsonl"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("artifacts/outcome_calibration_v5/train"),
    )
    parser.add_argument("--estimators", type=int, default=DEFAULT_ESTIMATORS)
    parser.add_argument(
        "--target-cap",
        default=str(DEFAULT_TARGET_CAP),
        help="regret target cap; use 'none' for uncapped log-regret targets",
    )
    parser.add_argument("--format", choices=("json", "table", "both"), default="both")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        target_cap = _parse_target_cap(args.target_cap)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    result = analyze(
        outcome_path=args.outcomes,
        checkpoint_root=args.checkpoint_root,
        estimators=args.estimators,
        target_cap=target_cap,
    )
    if args.format in {"table", "both"}:
        print(_format_table(result))
    if args.format == "both":
        print("\njson:")
    if args.format in {"json", "both"}:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
