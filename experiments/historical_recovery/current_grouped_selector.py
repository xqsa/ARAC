"""Evaluate and freeze the current recovered-action selector."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import joblib

from arac.analysis.grouped_outcome_selector import (
    GROUPED_EVALUATION_SCHEMA,
    GROUPED_EVALUATION_FILENAME,
    GROUPED_METADATA_FILENAME,
    GROUPED_MODEL_FILENAME,
    GROUPED_SELECTOR_SCHEMA,
    FROZEN_UNCERTAINTY_PENALTY,
    GroupedOutcomeSelector,
    build_grouped_evaluation,
    file_sha256,
    fit_frozen_grouped_selector,
    load_outcome_records,
)
from arac.analysis.outcome_selector import MODEL_PARAMETERS, REGRET_TARGET_CAP
from arac.evidence.phase1 import PHASE1_FEATURE_NAMES, PHASE1_PROTOCOL
from arac.runtime.contracts import ACTION_NAMES, canonical_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_ROOT = REPOSITORY_ROOT / "artifacts/current_recovered_four_arm_matrix_v2"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts/current_grouped_selector_v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def evaluate(
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    source_root = Path(matrix_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"grouped selector output is not empty: {destination}")
    records = load_outcome_records(source_root / "outcomes.jsonl")
    evaluation = build_grouped_evaluation(records)
    body = {
        **evaluation,
        "outcome_matrix_path": str((source_root / "outcomes.jsonl").resolve()),
        "outcome_matrix_sha256": file_sha256(source_root / "outcomes.jsonl"),
    }
    body.pop("evaluation_hash")
    result = {**body, "evaluation_hash": canonical_sha256(body)}
    _write_json(destination / GROUPED_EVALUATION_FILENAME, result)
    return result


def verify_evaluation(
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    source_root = Path(matrix_root).resolve()
    destination = Path(output_root).resolve()
    observed = _load_json(destination / GROUPED_EVALUATION_FILENAME)
    records = load_outcome_records(source_root / "outcomes.jsonl")
    expected = build_grouped_evaluation(records)
    body = {
        **expected,
        "outcome_matrix_path": str((source_root / "outcomes.jsonl").resolve()),
        "outcome_matrix_sha256": file_sha256(source_root / "outcomes.jsonl"),
    }
    body.pop("evaluation_hash")
    frozen = {**body, "evaluation_hash": canonical_sha256(body)}
    if observed != frozen:
        raise ValueError("grouped selector evaluation drifted")
    return frozen


def freeze(
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    source_root = Path(matrix_root).resolve()
    destination = Path(output_root).resolve()
    evaluation = verify_evaluation(source_root, destination)
    if evaluation.get("selector_freeze_authorized") is not True:
        raise RuntimeError("grouped selector gate failed; selector was not frozen")
    model_path = destination / GROUPED_MODEL_FILENAME
    metadata_path = destination / GROUPED_METADATA_FILENAME
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError("grouped selector artifact already exists")
    records = load_outcome_records(source_root / "outcomes.jsonl")
    staging = Path(tempfile.mkdtemp(prefix=".selector-freeze-", dir=destination))
    try:
        staged_model_path = staging / GROUPED_MODEL_FILENAME
        staged_metadata_path = staging / GROUPED_METADATA_FILENAME
        staged_evaluation_path = staging / GROUPED_EVALUATION_FILENAME
        shutil.copy2(destination / GROUPED_EVALUATION_FILENAME, staged_evaluation_path)
        joblib.dump(fit_frozen_grouped_selector(records), staged_model_path)
        body = {
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
            "evaluation_schema": GROUPED_EVALUATION_SCHEMA,
            "evaluation_sha256": file_sha256(staged_evaluation_path),
            "outcome_matrix_sha256": evaluation["outcome_matrix_sha256"],
            "training_record_hashes": [record.payload()["record_hash"] for record in records],
            "model_sha256": file_sha256(staged_model_path),
        }
        _write_json(
            staged_metadata_path,
            {**body, "metadata_hash": canonical_sha256(body)},
        )
        GroupedOutcomeSelector.load(staging)
        staged_model_path.replace(model_path)
        staged_metadata_path.replace(metadata_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    selector = GroupedOutcomeSelector.load(destination)
    return {
        "selector_directory": str(destination),
        "model_sha256": selector.metadata["model_sha256"],
        "metadata_hash": selector.metadata["metadata_hash"],
        "grouped_gate_passed": True,
    }


def verify_frozen(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    destination = Path(output_root).resolve()
    selector = GroupedOutcomeSelector.load(destination)
    return {
        "selector_directory": str(destination),
        "model_sha256": selector.metadata["model_sha256"],
        "metadata_hash": selector.metadata["metadata_hash"],
        "grouped_gate_passed": selector.metadata["grouped_gate_passed"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "verify-evaluation", "freeze", "verify"))
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        result = evaluate(args.matrix_root, args.output_root)
    elif args.command == "verify-evaluation":
        result = verify_evaluation(args.matrix_root, args.output_root)
    elif args.command == "freeze":
        result = freeze(args.matrix_root, args.output_root)
    else:
        result = verify_frozen(args.output_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
