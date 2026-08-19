# ruff: noqa: E402
"""Calibrate and run the independent two-phase ARAC experiment."""

from __future__ import annotations

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
from typing import Any
import warnings

from arac.actions.registry import ActionRegistry
from arac.analysis.outcome_selector import (
    EVALUATION_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    ActionOutcome,
    OutcomeRecord,
    OutcomeSelector,
    Selector,
    evaluate_training_selector,
    file_sha256,
    fit_outcome_selector,
)
from arac.benchmarks.aob import AobBenchmark, OptimizationProblem
from arac.evidence.phase1 import (
    PHASE1_FEATURE_NAMES,
    PHASE1_FES,
    PHASE1_MIN_FES,
    PHASE1_PROTOCOL,
    phase1_budget,
    run_phase1,
)
from arac.runtime.contracts import (
    ACTION_NAMES,
    ActionContext,
    ActionResult,
    PhaseCheckpoint,
    RelationEvidence,
)
from arac.runtime.ledger import EvaluationLedger


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_CONFIG_PATH = HERE / "config.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts" / "final_24x25_v4"
DEFAULT_CALIBRATION_ROOT = REPOSITORY_ROOT / "artifacts" / "outcome_calibration_v4"
CONFIG_SCHEMA = "arac-independent-final-config-v1"
RECEIPT_SCHEMA = "arac-independent-e2e-receipt-v2"
ARM_SCHEMA = "arac-independent-counterfactual-arm-v2"
CHECKPOINT_RECEIPT_SCHEMA = "arac-independent-phase1-checkpoint-v1"
SUMMARY_SCHEMA = "arac-independent-summary-v1"
CAMPAIGN_MANIFEST_SCHEMA = "arac-independent-campaign-manifest-v1"
CAMPAIGN_MANIFEST_FILENAME = "campaign_manifest.json"
FROZEN_PROTOCOL_SCHEMA = "arac-independent-frozen-protocol-v2"
SOURCE_PATHS = {
    "experiment": Path(__file__).resolve(),
    "benchmark": REPOSITORY_ROOT / "src" / "arac" / "benchmarks" / "aob.py",
    "phase1": REPOSITORY_ROOT / "src" / "arac" / "evidence" / "phase1.py",
    "mechanism_features": REPOSITORY_ROOT
    / "src"
    / "arac"
    / "evidence"
    / "mechanism_features.py",
    "structural_evidence": REPOSITORY_ROOT / "src" / "arac" / "evidence" / "structural.py",
    "contracts": REPOSITORY_ROOT / "src" / "arac" / "runtime" / "contracts.py",
    "ledger": REPOSITORY_ROOT / "src" / "arac" / "runtime" / "ledger.py",
    "optimizers": REPOSITORY_ROOT / "src" / "arac" / "runtime" / "optimizers.py",
    "action_execution": REPOSITORY_ROOT / "src" / "arac" / "actions" / "_execution.py",
    "action_registry": REPOSITORY_ROOT / "src" / "arac" / "actions" / "registry.py",
    "ctp": REPOSITORY_ROOT / "src" / "arac" / "actions" / "ctp.py",
    "smp": REPOSITORY_ROOT / "src" / "arac" / "actions" / "smp.py",
    "gcb": REPOSITORY_ROOT / "src" / "arac" / "actions" / "gcb.py",
    "aor": REPOSITORY_ROOT / "src" / "arac" / "actions" / "aor.py",
    "selector": REPOSITORY_ROOT / "src" / "arac" / "analysis" / "outcome_selector.py",
}
VENDOR_ROOTS = {
    "vendor/aob": REPOSITORY_ROOT / "vendor" / "aob",
}
KNOWN_RUNTIME_WARNINGS = {
    (
        "RuntimeWarning",
        "overflow encountered in scalar multiply",
        ".venv/Lib/site-packages/pypop7/optimizers/es/cmaes.py",
        228,
    ),
    (
        "RuntimeWarning",
        "overflow encountered in exp",
        ".venv/Lib/site-packages/pypop7/optimizers/es/cmaes.py",
        228,
    ),
    (
        "RuntimeWarning",
        "overflow encountered in multiply",
        "src/arac/actions/_execution.py",
        151,
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    temporary.replace(destination)


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object")
    return payload


def _warning_source(filename: str) -> str:
    source = Path(filename).resolve()
    try:
        return source.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return source.name


def _serialize_runtime_warnings(
    caught: Sequence[warnings.WarningMessage],
) -> list[dict[str, object]]:
    counts = Counter(
        (
            item.category.__name__,
            str(item.message),
            _warning_source(item.filename),
            int(item.lineno),
        )
        for item in caught
    )
    return [
        {
            "category": category,
            "message": message,
            "source": source,
            "line": line,
            "count": count,
            "known": (category, message, source, line) in KNOWN_RUNTIME_WARNINGS,
        }
        for (category, message, source, line), count in sorted(counts.items())
    ]


def _call_with_warning_capture(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, list[dict[str, object]]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    return result, _serialize_runtime_warnings(caught)


def _validate_runtime_warnings(payload: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(payload, list):
        raise ValueError(f"{label} runtime warnings are invalid")
    expected = {"category", "message", "source", "line", "count", "known"}
    for item in payload:
        if (
            not isinstance(item, Mapping)
            or set(item) != expected
            or not all(isinstance(item[name], str) and item[name] for name in ("category", "message", "source"))
            or isinstance(item["line"], bool)
            or int(item["line"]) <= 0
            or isinstance(item["count"], bool)
            or int(item["count"]) <= 0
            or not isinstance(item["known"], bool)
        ):
            raise ValueError(f"{label} runtime warning entry is invalid")
        fingerprint = (
            str(item["category"]),
            str(item["message"]),
            str(item["source"]),
            int(item["line"]),
        )
        if bool(item["known"]) != (fingerprint in KNOWN_RUNTIME_WARNINGS):
            raise ValueError(f"{label} runtime warning classification drifted")
    return payload


def _runtime_warning_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = 0
    unexpected = 0
    contexts = 0
    by_action: Counter[str] = Counter()
    for row in rows:
        entries = _validate_runtime_warnings(row.get("runtime_warnings", []), "campaign")
        if entries:
            contexts += 1
        action = str(row.get("action_name", row.get("selected_action", "unknown")))
        for item in entries:
            count = int(item["count"])
            total += count
            by_action[action] += count
            if item["known"] is not True:
                unexpected += count
    return {
        "runtime_warning_count": total,
        "runtime_warning_context_count": contexts,
        "runtime_warning_counts_by_action": dict(sorted(by_action.items())),
        "unexpected_runtime_warning_count": unexpected,
        "all_runtime_warnings_known": unexpected == 0,
    }


def _require_known_runtime_warnings(
    summary: Mapping[str, object],
    *,
    stage: str,
) -> None:
    """Stop stage transitions when a warning is outside the frozen audit set."""

    unexpected = summary.get("unexpected_runtime_warning_count")
    if isinstance(unexpected, bool) or not isinstance(unexpected, int) or unexpected < 0:
        raise ValueError(f"{stage} runtime warning summary is invalid")
    if summary.get("all_runtime_warnings_known") is not (unexpected == 0):
        raise ValueError(f"{stage} runtime warning summary is inconsistent")
    if unexpected:
        raise RuntimeError(
            f"{stage} produced {unexpected} unknown runtime warning(s); "
            "downstream stages are blocked"
        )


def _source_hashes() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in SOURCE_PATHS.items()}


def _directory_tree_sha256(root: Path) -> tuple[int, str]:
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"vendor directory is missing: {directory}")
    entries = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory)
        if (
            not path.is_file()
            or "__pycache__" in relative.parts
            or path.suffix.lower() in {".pyc", ".pyo"}
        ):
            continue
        entries.append((relative.as_posix(), file_sha256(path)))
    return len(entries), _canonical_sha256(entries)


def _vendor_tree_hashes() -> dict[str, dict[str, object]]:
    return {
        name: {
            "file_count": count,
            "tree_sha256": tree_hash,
        }
        for name, root in VENDOR_ROOTS.items()
        for count, tree_hash in (_directory_tree_sha256(root),)
    }


def _hashed_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["manifest_sha256"] = _canonical_sha256(result)
    return result


def _validate_hashed_manifest(
    payload: Mapping[str, object],
    expected_body: Mapping[str, object],
    *,
    label: str,
) -> None:
    body = dict(payload)
    claimed = body.pop("manifest_sha256", None)
    if claimed != _canonical_sha256(body):
        raise ValueError(f"{label} hash drifted")
    if body != dict(expected_body):
        raise ValueError(f"{label} contract drifted")


def _campaign_manifest(
    campaign_kind: str,
    cases: Sequence[str],
    seeds: Sequence[int],
    *,
    max_fes: int,
    config_path: Path | None,
    source_hashes: Mapping[str, str],
    vendor_trees: Mapping[str, Mapping[str, object]],
    selector_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    body = {
        "schema_version": CAMPAIGN_MANIFEST_SCHEMA,
        "campaign_kind": campaign_kind,
        "cases": [str(case) for case in cases],
        "seeds": [int(seed) for seed in seeds],
        "max_fes": int(max_fes),
        "actions": list(ACTION_NAMES),
        "phase1_protocol": PHASE1_PROTOCOL,
        "config_sha256": file_sha256(config_path) if config_path is not None else None,
        "source_hashes": dict(sorted(source_hashes.items())),
        "vendor_trees": {
            name: dict(value) for name, value in sorted(vendor_trees.items())
        },
        "selector_hashes": dict(sorted((selector_hashes or {}).items())),
    }
    return _hashed_manifest(body)


def _prepare_campaign_root(root: Path, manifest: Mapping[str, object], *, resume: bool) -> None:
    destination = Path(root).resolve()
    manifest_path = destination / CAMPAIGN_MANIFEST_FILENAME
    if resume:
        if not destination.is_dir():
            raise FileNotFoundError(f"resume campaign root is missing: {destination}")
        stored = _read_json(manifest_path, "campaign manifest")
        expected = dict(manifest)
        expected_body = dict(expected)
        expected_hash = expected_body.pop("manifest_sha256")
        _validate_hashed_manifest(stored, expected_body, label="campaign manifest")
        if stored.get("manifest_sha256") != expected_hash:
            raise ValueError("campaign manifest hash drifted")
        return
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"fresh output root is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError("fresh output root is not empty; use --resume")
    else:
        destination.mkdir(parents=True)
    _atomic_json(manifest_path, dict(manifest))


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    payload = _read_json(Path(path), "final experiment config")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("final experiment config schema drifted")
    required = {
        "cases",
        "calibration_seeds",
        "holdout_seeds",
        "evaluation_seeds",
        "max_fes",
        "max_workers",
        "selector_directory",
        "selector_files",
    }
    if set(payload) != required | {"schema_version"}:
        raise ValueError("final experiment config keys drifted")
    cases = tuple(str(case).upper() for case in payload["cases"])
    expected_cases = tuple(f"{family}{index}" for family in "AERS" for index in range(1, 7))
    if cases != expected_cases:
        raise ValueError("final experiment must contain the 24 AOB cases in fixed order")
    seed_sets = [
        tuple(int(value) for value in payload[name])
        for name in ("calibration_seeds", "holdout_seeds", "evaluation_seeds")
    ]
    if any(not seeds or len(set(seeds)) != len(seeds) for seeds in seed_sets):
        raise ValueError("seed lists must be non-empty and unique")
    if any(set(left) & set(right) for index, left in enumerate(seed_sets) for right in seed_sets[index + 1 :]):
        raise ValueError("calibration, holdout, and evaluation seeds must be disjoint")
    if int(payload["max_fes"]) <= PHASE1_MIN_FES:
        raise ValueError("max_fes must exceed the minimum Phase-I budget")
    if int(payload["max_workers"]) <= 0:
        raise ValueError("max_workers must be positive")
    selector_files = payload["selector_files"]
    if not isinstance(selector_files, dict):
        raise ValueError("selector_files must be an object")
    allowed_selector_files = {MODEL_FILENAME, METADATA_FILENAME, EVALUATION_FILENAME}
    if set(selector_files) - allowed_selector_files:
        raise ValueError("selector_files contains an unknown artifact")
    return payload


@dataclass(frozen=True)
class MethodExecution:
    checkpoint: PhaseCheckpoint
    selected_action: str
    result: ActionResult


def execute_method(
    problem: OptimizationProblem,
    *,
    run_seed: int,
    max_fes: int,
    selector: Selector,
) -> MethodExecution:
    """Run the complete method without receiving benchmark case identity."""

    ledger = EvaluationLedger(problem, max_fes)
    checkpoint = run_phase1(problem, ledger, run_seed=run_seed).checkpoint
    selected = selector.select(checkpoint.feature_names, checkpoint.feature_values)
    if selected not in ACTION_NAMES:
        raise RuntimeError("selector returned an unsupported action")
    result = ActionRegistry().execute(
        ActionContext(
            action_name=selected,
            checkpoint=checkpoint,
            problem=problem,
            ledger=ledger,
            action_seed=run_seed,
        )
    )
    if result.terminal_fes != max_fes or ledger.count != max_fes:
        raise RuntimeError("method did not terminate at the exact FE budget")
    return MethodExecution(checkpoint=checkpoint, selected_action=selected, result=result)


@dataclass(frozen=True)
class ExperimentContext:
    case_id: str
    run_seed: int
    max_fes: int
    output_root: Path
    config_sha256: str
    selector_directory: Path
    selector_hashes: tuple[tuple[str, str], ...]

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}"

    @property
    def run_directory(self) -> Path:
        return self.output_root / "runs" / self.case_id / f"seed_{self.run_seed}"

    @property
    def receipt_path(self) -> Path:
        return self.run_directory / "receipt.json"


def _validate_selector_files(directory: Path, hashes: Mapping[str, str]) -> None:
    expected_names = {MODEL_FILENAME, METADATA_FILENAME, EVALUATION_FILENAME}
    if set(hashes) != expected_names:
        raise ValueError("all three frozen selector artifacts are required")
    for name, expected in hashes.items():
        if len(expected) != 64 or file_sha256(directory / name) != expected:
            raise ValueError(f"selector artifact hash drifted: {name}")


def _run_context(context: ExperimentContext) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    hashes = dict(context.selector_hashes)
    _validate_selector_files(context.selector_directory, hashes)
    selector = OutcomeSelector.load(context.selector_directory)
    problem = AobBenchmark().load(
        context.case_id,
        output_directory=context.run_directory / "benchmark",
    )
    execution, runtime_warnings = _call_with_warning_capture(
        execute_method,
        problem,
        run_seed=context.run_seed,
        max_fes=context.max_fes,
        selector=selector,
    )
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "config_sha256": context.config_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "max_fes": context.max_fes,
        "phase1_protocol": PHASE1_PROTOCOL,
        "phase1_fes": execution.checkpoint.phase1_fes,
        "checkpoint_hash": execution.checkpoint.checkpoint_hash,
        "feature_names": list(execution.checkpoint.feature_names),
        "feature_values": list(execution.checkpoint.feature_values),
        "selector_hashes": hashes,
        "selector_input_contains_case_or_family_identity": False,
        "selected_action": execution.selected_action,
        "selected_action_only": True,
        "action_result": execution.result.payload(),
        "action_result_hash": execution.result.result_hash,
        "terminal_fes": execution.result.terminal_fes,
        "final_error": execution.result.final_error,
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "runtime_warnings": runtime_warnings,
    }
    payload["receipt_hash"] = _canonical_sha256(payload)
    _atomic_json(context.receipt_path, payload)
    return payload


def validate_receipt(path: Path, context: ExperimentContext) -> dict[str, object]:
    payload = _read_json(path, "end-to-end receipt")
    claimed = payload.pop("receipt_hash", None)
    if claimed != _canonical_sha256(payload):
        raise ValueError(f"{context.key} receipt hash drifted")
    payload["receipt_hash"] = claimed
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "config_sha256": context.config_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "max_fes": context.max_fes,
        "phase1_protocol": PHASE1_PROTOCOL,
        "phase1_fes": phase1_budget(context.max_fes),
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "selector_input_contains_case_or_family_identity": False,
        "selected_action_only": True,
        "terminal_fes": context.max_fes,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"{context.key} receipt {name} drifted")
    if payload.get("selected_action") not in ACTION_NAMES:
        raise ValueError(f"{context.key} selected action is invalid")
    _validate_runtime_warnings(payload.get("runtime_warnings"), context.key)
    result = payload.get("action_result")
    if not isinstance(result, dict) or result.get("action_name") != payload["selected_action"]:
        raise ValueError(f"{context.key} action result disagrees with selection")
    return payload


def _build_contexts(
    cases: Sequence[str],
    seeds: Sequence[int],
    *,
    max_fes: int,
    output_root: Path,
    config_path: Path,
    selector_directory: Path,
    selector_hashes: Mapping[str, str],
) -> tuple[ExperimentContext, ...]:
    return tuple(
        ExperimentContext(
            case_id=str(case),
            run_seed=int(seed),
            max_fes=max_fes,
            output_root=Path(output_root).resolve(),
            config_sha256=file_sha256(config_path),
            selector_directory=Path(selector_directory).resolve(),
            selector_hashes=tuple(sorted(selector_hashes.items())),
        )
        for seed in seeds
        for case in cases
    )


def _freeze_protocol(
    output_root: Path,
    *,
    config_path: Path,
    selector_directory: Path,
    selector_hashes: Mapping[str, str],
    source_hashes: Mapping[str, str],
    phase1_fes: int,
    vendor_trees: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    if vendor_trees is None:
        vendor_trees = _vendor_tree_hashes()
    root = Path(output_root) / "frozen_protocol"
    source_root = root / "sources"
    selector_root = root / "selector"
    config_destination = root / "config.json"
    manifest_path = root / "manifest.json"
    protocol_body = {
        "schema_version": FROZEN_PROTOCOL_SCHEMA,
        "config_sha256": file_sha256(config_path),
        "source_hashes": dict(sorted(source_hashes.items())),
        "vendor_trees": {
            name: dict(value) for name, value in sorted(vendor_trees.items())
        },
        "selector_hashes": dict(sorted(selector_hashes.items())),
        "phase1_protocol": PHASE1_PROTOCOL,
        "phase1_fes": phase1_fes,
        "phase1_max_fes": PHASE1_FES,
        "actions": list(ACTION_NAMES),
        "labels_derived_from_action_outcomes": True,
    }
    if root.exists():
        if not root.is_dir():
            raise ValueError("frozen protocol path is not a directory")
        stored = _read_json(manifest_path, "frozen protocol manifest")
        stored_body = dict(stored)
        stored_body.pop("manifest_sha256", None)
        frozen_at = stored_body.pop("frozen_at_utc", None)
        if not isinstance(frozen_at, str) or not frozen_at:
            raise ValueError("frozen protocol timestamp drifted")
        expected_body = dict(protocol_body)
        expected_body["frozen_at_utc"] = frozen_at
        _validate_hashed_manifest(stored, expected_body, label="frozen protocol manifest")
        if file_sha256(config_destination) != protocol_body["config_sha256"]:
            raise ValueError("frozen config hash drifted")
        for name, expected in source_hashes.items():
            source = SOURCE_PATHS[name]
            destination = source_root / f"{name}{source.suffix}"
            if file_sha256(destination) != expected:
                raise ValueError(f"frozen source hash drifted: {name}")
        if _vendor_tree_hashes() != {
            name: dict(value) for name, value in sorted(vendor_trees.items())
        }:
            raise ValueError("frozen vendor tree hash drifted")
        for name, expected in selector_hashes.items():
            if file_sha256(selector_root / name) != expected:
                raise ValueError(f"frozen selector artifact hash drifted: {name}")
        return
    source_root.mkdir(parents=True, exist_ok=True)
    selector_root.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCE_PATHS.items():
        if file_sha256(source) != source_hashes.get(name):
            raise ValueError(f"source hash drifted while freezing: {name}")
        shutil.copy2(source, source_root / f"{name}{source.suffix}")
    if _vendor_tree_hashes() != {
        name: dict(value) for name, value in sorted(vendor_trees.items())
    }:
        raise ValueError("vendor tree hash drifted while freezing")
    for name, expected in selector_hashes.items():
        source = selector_directory / name
        if file_sha256(source) != expected:
            raise ValueError(f"selector artifact hash drifted while freezing: {name}")
        shutil.copy2(source, selector_root / name)
    shutil.copy2(config_path, config_destination)
    protocol_body["frozen_at_utc"] = _utc_now()
    _atomic_json(manifest_path, _hashed_manifest(protocol_body))


def _run_parallel(
    contexts: Sequence[Any],
    worker: Any,
    *,
    max_workers: int,
    progress_path: Path,
    receipt_path: Any,
    validator: Any,
    resume: bool,
) -> list[dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    pending = []
    for context in contexts:
        path = receipt_path(context)
        if resume and path.is_file():
            rows[context.key] = validator(path, context)
        else:
            pending.append(context)
    failures: dict[str, str] = {}

    def write_progress() -> None:
        _atomic_json(
            progress_path,
            {
                "planned": len(contexts),
                "completed": len(rows),
                "failed": len(failures),
                "pending": len(contexts) - len(rows) - len(failures),
                "max_workers": min(max_workers, len(contexts)),
                "updated_at_utc": _utc_now(),
                "failures": failures,
            },
        )

    write_progress()
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(max_workers, len(pending)),
            max_tasks_per_child=1,
        ) as pool:
            futures = {pool.submit(worker, context): context for context in pending}
            for future in as_completed(futures):
                context = futures[future]
                try:
                    future.result()
                    rows[context.key] = validator(receipt_path(context), context)
                except Exception as error:
                    failures[context.key] = f"{type(error).__name__}: {error}"
                write_progress()
                status = "complete" if context.key in rows else "failed"
                print(f"[{len(rows) + len(failures):04d}/{len(contexts)}] {context.key} {status}", flush=True)
    if failures:
        raise RuntimeError(f"parallel campaign has {len(failures)} failed contexts")
    return [rows[context.key] for context in contexts]


def _summary(rows: Sequence[Mapping[str, object]], max_workers: int) -> dict[str, object]:
    case_summaries = []
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        errors = [float(row["final_error"]) for row in case_rows]
        mean = statistics.fmean(errors)
        standard_deviation = statistics.stdev(errors) if len(errors) > 1 else 0.0
        case_summaries.append(
            {
                "case_id": case_id,
                "seed_count": len(case_rows),
                "mean_final_error": mean,
                "sample_std_final_error": standard_deviation,
                "mean_final_error_sci": f"{mean:.2e}",
                "sample_std_final_error_sci": f"{standard_deviation:.2e}",
                "selection_counts": dict(
                    sorted(Counter(str(row["selected_action"]) for row in case_rows).items())
                ),
            }
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at_utc": _utc_now(),
        "context_count": len(rows),
        "completed_context_count": len(rows),
        "max_workers": max_workers,
        "phase1_fes": next(iter({int(row["phase1_fes"]) for row in rows})),
        "all_terminal_fes_exact": all(row["terminal_fes"] == row["max_fes"] for row in rows),
        "selected_action_only": all(row["selected_action_only"] is True for row in rows),
        "selector_input_contains_case_or_family_identity": False,
        "selection_counts": dict(
            sorted(Counter(str(row["selected_action"]) for row in rows).items())
        ),
        "case_summaries": case_summaries,
    }
    summary.update(_runtime_warning_summary(rows))
    return summary


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cases: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    max_fes: int | None = None,
    max_workers: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    selected_cases = tuple(config["cases"] if cases is None else cases)
    selected_seeds = tuple(config["evaluation_seeds"] if seeds is None else seeds)
    budget = int(config["max_fes"] if max_fes is None else max_fes)
    workers = int(config["max_workers"] if max_workers is None else max_workers)
    active_output = Path(output_root).resolve()
    selector_directory = (REPOSITORY_ROOT / config["selector_directory"]).resolve()
    selector_hashes = dict(config["selector_files"])
    _validate_selector_files(selector_directory, selector_hashes)
    OutcomeSelector.load(selector_directory)
    source_hashes = _source_hashes()
    vendor_trees = _vendor_tree_hashes()
    manifest = _campaign_manifest(
        "end_to_end",
        selected_cases,
        selected_seeds,
        max_fes=budget,
        config_path=config_path,
        source_hashes=source_hashes,
        vendor_trees=vendor_trees,
        selector_hashes=selector_hashes,
    )
    _prepare_campaign_root(active_output, manifest, resume=resume)
    _freeze_protocol(
        active_output,
        config_path=config_path,
        selector_directory=selector_directory,
        selector_hashes=selector_hashes,
        source_hashes=source_hashes,
        vendor_trees=vendor_trees,
        phase1_fes=phase1_budget(budget),
    )
    contexts = _build_contexts(
        selected_cases,
        selected_seeds,
        max_fes=budget,
        output_root=active_output,
        config_path=config_path,
        selector_directory=selector_directory,
        selector_hashes=selector_hashes,
    )
    rows = _run_parallel(
        contexts,
        _run_context,
        max_workers=workers,
        progress_path=active_output / "parallel_progress.json",
        receipt_path=lambda context: context.receipt_path,
        validator=validate_receipt,
        resume=resume,
    )
    summary = _summary(rows, workers)
    _atomic_json(active_output / "summary.json", summary)
    _require_known_runtime_warnings(summary, stage="end-to-end campaign")
    fields = [
        "case_id",
        "run_seed",
        "selected_action",
        "phase1_fes",
        "terminal_fes",
        "final_error",
        "elapsed_seconds",
        "checkpoint_hash",
        "action_result_hash",
        "receipt_hash",
    ]
    with (active_output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})
    return summary


@dataclass(frozen=True)
class ArmContext:
    case_id: str
    run_seed: int
    action_name: str
    max_fes: int
    output_root: Path

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}:{self.action_name}"

    @property
    def receipt_path(self) -> Path:
        return (
            self.output_root
            / "arms"
            / self.case_id
            / f"seed_{self.run_seed}"
            / f"{self.action_name}.json"
        )


@dataclass(frozen=True)
class CheckpointContext:
    case_id: str
    run_seed: int
    max_fes: int
    output_root: Path

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}"

    @property
    def receipt_path(self) -> Path:
        return (
            self.output_root
            / "checkpoints"
            / self.case_id
            / f"seed_{self.run_seed}"
            / "checkpoint.json"
        )


def _checkpoint_from_payload(payload: Mapping[str, object]) -> PhaseCheckpoint:
    relations = tuple(
        RelationEvidence(
            left_block=int(item["left_block"]),
            right_block=int(item["right_block"]),
            strength=float(item["strength"]),
            disagreement=float(item["disagreement"]),
        )
        for item in payload["relations"]  # type: ignore[union-attr]
    )
    return PhaseCheckpoint(
        protocol=str(payload["protocol"]),
        run_seed=int(payload["run_seed"]),
        total_budget_fes=int(payload["total_budget_fes"]),
        phase1_fes=int(payload["phase1_fes"]),
        incumbent=tuple(float(value) for value in payload["incumbent"]),  # type: ignore[union-attr]
        incumbent_error=float(payload["incumbent_error"]),
        feature_names=tuple(str(name) for name in payload["feature_names"]),  # type: ignore[union-attr]
        feature_values=tuple(float(value) for value in payload["feature_values"]),  # type: ignore[union-attr]
        blocks=tuple(
            tuple(int(value) for value in block)
            for block in payload["blocks"]  # type: ignore[union-attr]
        ),
        relations=relations,
    )


def _run_checkpoint(context: CheckpointContext) -> dict[str, object]:
    problem = AobBenchmark().load(context.case_id)
    ledger = EvaluationLedger(problem, context.max_fes)
    checkpoint = run_phase1(problem, ledger, run_seed=context.run_seed).checkpoint
    checkpoint_payload = checkpoint.payload()
    payload = {
        "schema_version": CHECKPOINT_RECEIPT_SCHEMA,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "max_fes": context.max_fes,
        "checkpoint": checkpoint_payload,
        "checkpoint_hash": checkpoint.checkpoint_hash,
    }
    payload["receipt_hash"] = _canonical_sha256(payload)
    _atomic_json(context.receipt_path, payload)
    return payload


def _validate_checkpoint(path: Path, context: CheckpointContext) -> dict[str, object]:
    payload = _read_json(path, "Phase-I checkpoint receipt")
    claimed = payload.pop("receipt_hash", None)
    if claimed != _canonical_sha256(payload):
        raise ValueError(f"{context.key} checkpoint receipt hash drifted")
    payload["receipt_hash"] = claimed
    if payload.get("schema_version") != CHECKPOINT_RECEIPT_SCHEMA:
        raise ValueError(f"{context.key} checkpoint schema drifted")
    if payload.get("case_id") != context.case_id or int(payload.get("run_seed", -1)) != context.run_seed:
        raise ValueError(f"{context.key} checkpoint identity drifted")
    if int(payload.get("max_fes", -1)) != context.max_fes:
        raise ValueError(f"{context.key} checkpoint budget drifted")
    checkpoint_payload = payload.get("checkpoint")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError(f"{context.key} checkpoint payload is invalid")
    checkpoint = _checkpoint_from_payload(checkpoint_payload)
    if checkpoint.checkpoint_hash != payload.get("checkpoint_hash"):
        raise ValueError(f"{context.key} checkpoint hash drifted")
    if checkpoint.run_seed != context.run_seed or checkpoint.total_budget_fes != context.max_fes:
        raise ValueError(f"{context.key} checkpoint contract drifted")
    return payload


def _run_arm(context: ArmContext) -> dict[str, object]:
    problem = AobBenchmark().load(context.case_id)
    checkpoint_path = (
        context.output_root
        / "checkpoints"
        / context.case_id
        / f"seed_{context.run_seed}"
        / "checkpoint.json"
    )
    checkpoint_receipt = _validate_checkpoint(
        checkpoint_path,
        CheckpointContext(
            context.case_id,
            context.run_seed,
            context.max_fes,
            context.output_root,
        ),
    )
    checkpoint_payload = checkpoint_receipt.get("checkpoint")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError(f"{context.key} shared checkpoint is invalid")
    checkpoint = _checkpoint_from_payload(checkpoint_payload)
    if checkpoint.run_seed != context.run_seed or checkpoint.total_budget_fes != context.max_fes:
        raise ValueError(f"{context.key} shared checkpoint contract drifted")
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=context.max_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    result, runtime_warnings = _call_with_warning_capture(
        ActionRegistry().execute,
        ActionContext(
            context.action_name,
            checkpoint,
            problem,
            ledger,
            action_seed=context.run_seed,
        ),
    )
    payload = {
        "schema_version": ARM_SCHEMA,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "action_name": context.action_name,
        "max_fes": context.max_fes,
        "phase1_protocol": PHASE1_PROTOCOL,
        "phase1_fes": checkpoint.phase1_fes,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "feature_names": list(checkpoint.feature_names),
        "feature_values": list(checkpoint.feature_values),
        "action_result": result.payload(),
        "action_result_hash": result.result_hash,
        "final_error": result.final_error,
        "terminal_fes": result.terminal_fes,
        "runtime_warnings": runtime_warnings,
    }
    payload["receipt_hash"] = _canonical_sha256(payload)
    _atomic_json(context.receipt_path, payload)
    return payload


def _validate_arm(path: Path, context: ArmContext) -> dict[str, object]:
    payload = _read_json(path, "counterfactual arm receipt")
    claimed = payload.pop("receipt_hash", None)
    if claimed != _canonical_sha256(payload):
        raise ValueError(f"{context.key} arm receipt hash drifted")
    payload["receipt_hash"] = claimed
    expected = {
        "schema_version": ARM_SCHEMA,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "action_name": context.action_name,
        "max_fes": context.max_fes,
        "phase1_protocol": PHASE1_PROTOCOL,
        "phase1_fes": phase1_budget(context.max_fes),
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "terminal_fes": context.max_fes,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"{context.key} arm field {name} drifted")
    _validate_runtime_warnings(payload.get("runtime_warnings"), context.key)
    checkpoint_receipt = _validate_checkpoint(
        context.output_root
        / "checkpoints"
        / context.case_id
        / f"seed_{context.run_seed}"
        / "checkpoint.json",
        CheckpointContext(
            context.case_id,
            context.run_seed,
            context.max_fes,
            context.output_root,
        ),
    )
    if payload.get("checkpoint_hash") != checkpoint_receipt.get("checkpoint_hash"):
        raise ValueError(f"{context.key} arm does not use the shared checkpoint")
    return payload


def _records_from_arms(rows: Sequence[Mapping[str, object]]) -> tuple[OutcomeRecord, ...]:
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), int(row["run_seed"]))].append(row)
    records = []
    for (case_id, seed), arms in sorted(grouped.items()):
        by_action = {str(arm["action_name"]): arm for arm in arms}
        if set(by_action) != set(ACTION_NAMES):
            raise ValueError(f"{case_id}:seed-{seed} lacks the full action matrix")
        checkpoints = {str(arm["checkpoint_hash"]) for arm in arms}
        features = {tuple(float(value) for value in arm["feature_values"]) for arm in arms}
        if len(checkpoints) != 1 or len(features) != 1:
            raise ValueError(f"{case_id}:seed-{seed} arms do not share one checkpoint")
        records.append(
            OutcomeRecord(
                case_id=case_id,
                run_seed=seed,
                checkpoint_hash=checkpoints.pop(),
                feature_names=PHASE1_FEATURE_NAMES,
                feature_values=features.pop(),
                outcomes=tuple(
                    ActionOutcome(
                        action_name=action,
                        final_error=float(by_action[action]["final_error"]),
                        result_hash=str(by_action[action]["action_result_hash"]),
                    )
                    for action in ACTION_NAMES
                ),
            )
        )
    return tuple(records)


def _write_records(path: Path, records: Sequence[OutcomeRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record.payload(), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)


def run_outcome_campaign(
    cases: Sequence[str],
    seeds: Sequence[int],
    *,
    max_fes: int,
    max_workers: int,
    output_root: Path,
    resume: bool,
    config_path: Path | None = None,
) -> tuple[OutcomeRecord, ...]:
    root = Path(output_root).resolve()
    source_hashes = _source_hashes()
    vendor_trees = _vendor_tree_hashes()
    manifest = _campaign_manifest(
        "outcome_matrix",
        cases,
        seeds,
        max_fes=max_fes,
        config_path=Path(config_path).resolve() if config_path is not None else None,
        source_hashes=source_hashes,
        vendor_trees=vendor_trees,
    )
    _prepare_campaign_root(root, manifest, resume=resume)
    checkpoint_contexts = tuple(
        CheckpointContext(case, int(seed), max_fes, root)
        for seed in seeds
        for case in cases
    )
    _run_parallel(
        checkpoint_contexts,
        _run_checkpoint,
        max_workers=max_workers,
        progress_path=root / "checkpoint_progress.json",
        receipt_path=lambda context: context.receipt_path,
        validator=_validate_checkpoint,
        resume=resume,
    )
    contexts = tuple(
        ArmContext(case, int(seed), action, max_fes, root)
        for seed in seeds
        for case in cases
        for action in ACTION_NAMES
    )
    rows = _run_parallel(
        contexts,
        _run_arm,
        max_workers=max_workers,
        progress_path=root / "parallel_progress.json",
        receipt_path=lambda context: context.receipt_path,
        validator=_validate_arm,
        resume=resume,
    )
    records = _records_from_arms(rows)
    _write_records(root / "outcomes.jsonl", records)
    summary = {
        "schema_version": "arac-outcome-campaign-summary-v2",
        "record_count": len(records),
        "arm_count": len(rows),
        "max_fes": max_fes,
        "labels_derived_from_action_outcomes": True,
        "label_counts": dict(sorted(Counter(row.action_label for row in records).items())),
    }
    summary.update(_runtime_warning_summary(rows))
    _atomic_json(root / "summary.json", summary)
    _require_known_runtime_warnings(summary, stage="outcome campaign")
    return records


def calibrate_selector(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_CALIBRATION_ROOT,
    max_fes: int | None = None,
    max_workers: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    config = load_config(config_path)
    budget = int(config["max_fes"] if max_fes is None else max_fes)
    workers = int(config["max_workers"] if max_workers is None else max_workers)
    root = Path(output_root).resolve()
    train = run_outcome_campaign(
        config["cases"],
        config["calibration_seeds"],
        max_fes=budget,
        max_workers=workers,
        output_root=root / "train",
        resume=resume,
        config_path=config_path,
    )
    training_preflight = evaluate_training_selector(train)
    _atomic_json(
        root / "training_preflight.json",
        {
            "schema_version": "arac-selector-training-preflight-v1",
            "generated_at_utc": _utc_now(),
            "record_count": len(train),
            "metrics": training_preflight,
        },
    )
    if training_preflight["passed"] is not True:
        regret = training_preflight["terminal_regret"]
        raise RuntimeError(
            "selector training CV gate failed; holdout campaign was not started "
            f"(accuracy={training_preflight['accuracy']:.6f}, "
            f"balanced_accuracy={training_preflight['balanced_accuracy']:.6f}, "
            f"mean_log10_regret={regret['mean_log10_regret']:.6f}, "
            f"worst_log10_regret={regret['worst_log10_regret']:.6f})"
        )
    holdout = run_outcome_campaign(
        config["cases"],
        config["holdout_seeds"],
        max_fes=budget,
        max_workers=workers,
        output_root=root / "holdout",
        resume=resume,
        config_path=config_path,
    )
    selector_directory = (REPOSITORY_ROOT / config["selector_directory"]).resolve()
    result = fit_outcome_selector(train, holdout, output_directory=selector_directory)
    summary = {
        "schema_version": "arac-selector-calibration-summary-v1",
        "generated_at_utc": _utc_now(),
        "max_fes": budget,
        "training_records": len(train),
        "holdout_records": len(holdout),
        "training_label_counts": dict(sorted(Counter(row.action_label for row in train).items())),
        "holdout_label_counts": dict(sorted(Counter(row.action_label for row in holdout).items())),
        "training_preflight": training_preflight,
        "selector": result,
    }
    _atomic_json(root / "calibration_summary.json", summary)
    return summary


def _parse_values(raw: str | None, default: Sequence[Any], converter: Any) -> tuple[Any, ...]:
    if raw is None:
        return tuple(default)
    return tuple(converter(value.strip()) for value in raw.split(",") if value.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run the frozen one-action experiment")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--cases")
    run_parser.add_argument("--seeds")
    run_parser.add_argument("--max-fes", type=int)
    run_parser.add_argument("--workers", type=int)
    run_parser.add_argument("--resume", action="store_true")
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="measure all four actions and train the outcome selector"
    )
    calibrate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    calibrate_parser.add_argument("--output-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    calibrate_parser.add_argument("--max-fes", type=int)
    calibrate_parser.add_argument("--workers", type=int)
    calibrate_parser.add_argument("--resume", action="store_true")
    outcomes_parser = subparsers.add_parser(
        "outcomes", help="run a common-checkpoint action outcome matrix without fitting"
    )
    outcomes_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    outcomes_parser.add_argument("--output-root", type=Path, required=True)
    outcomes_parser.add_argument("--cases")
    outcomes_parser.add_argument("--seeds")
    outcomes_parser.add_argument("--max-fes", type=int)
    outcomes_parser.add_argument("--workers", type=int)
    outcomes_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        summary = calibrate_selector(
            config_path=args.config,
            output_root=args.output_root,
            max_fes=args.max_fes,
            max_workers=args.workers,
            resume=args.resume,
        )
    elif args.command == "outcomes":
        config = load_config(args.config)
        records = run_outcome_campaign(
            _parse_values(args.cases, config["cases"], str),
            _parse_values(args.seeds, config["calibration_seeds"], int),
            max_fes=int(config["max_fes"] if args.max_fes is None else args.max_fes),
            max_workers=int(
                config["max_workers"] if args.workers is None else args.workers
            ),
            output_root=args.output_root,
            resume=args.resume,
            config_path=args.config,
        )
        summary = {
            "record_count": len(records),
            "label_counts": dict(
                sorted(Counter(record.action_label for record in records).items())
            ),
        }
    else:
        config = load_config(args.config)
        summary = run_experiment(
            config_path=args.config,
            output_root=args.output_root,
            cases=_parse_values(args.cases, config["cases"], str),
            seeds=_parse_values(args.seeds, config["evaluation_seeds"], int),
            max_fes=args.max_fes,
            max_workers=args.workers,
            resume=args.resume,
        )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG_PATH",
    "MethodExecution",
    "calibrate_selector",
    "execute_method",
    "load_config",
    "run_experiment",
    "run_outcome_campaign",
    "validate_receipt",
]
