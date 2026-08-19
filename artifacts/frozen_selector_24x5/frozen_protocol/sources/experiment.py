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

from arac.actions.registry import ActionRegistry
from arac.analysis.outcome_selector import (
    EVALUATION_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    ActionOutcome,
    OutcomeRecord,
    OutcomeSelector,
    Selector,
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
from arac.runtime.contracts import ACTION_NAMES, ActionContext, ActionResult, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_CONFIG_PATH = HERE / "config.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts" / "final_24x25"
DEFAULT_CALIBRATION_ROOT = REPOSITORY_ROOT / "artifacts" / "outcome_calibration_v1"
CONFIG_SCHEMA = "arac-independent-final-config-v1"
RECEIPT_SCHEMA = "arac-independent-e2e-receipt-v1"
ARM_SCHEMA = "arac-independent-counterfactual-arm-v1"
SUMMARY_SCHEMA = "arac-independent-summary-v1"
SOURCE_PATHS = {
    "experiment": Path(__file__).resolve(),
    "benchmark": REPOSITORY_ROOT / "src" / "arac" / "benchmarks" / "aob.py",
    "phase1": REPOSITORY_ROOT / "src" / "arac" / "evidence" / "phase1.py",
    "structural_evidence": REPOSITORY_ROOT / "src" / "arac" / "evidence" / "structural.py",
    "contracts": REPOSITORY_ROOT / "src" / "arac" / "runtime" / "contracts.py",
    "ledger": REPOSITORY_ROOT / "src" / "arac" / "runtime" / "ledger.py",
    "optimizers": REPOSITORY_ROOT / "src" / "arac" / "runtime" / "optimizers.py",
    "action_execution": REPOSITORY_ROOT / "src" / "arac" / "actions" / "_execution.py",
    "ctp": REPOSITORY_ROOT / "src" / "arac" / "actions" / "ctp.py",
    "smp": REPOSITORY_ROOT / "src" / "arac" / "actions" / "smp.py",
    "gcb": REPOSITORY_ROOT / "src" / "arac" / "actions" / "gcb.py",
    "aor": REPOSITORY_ROOT / "src" / "arac" / "actions" / "aor.py",
    "selector": REPOSITORY_ROOT / "src" / "arac" / "analysis" / "outcome_selector.py",
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
    execution = execute_method(
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
    phase1_fes: int,
) -> None:
    root = Path(output_root) / "frozen_protocol"
    source_root = root / "sources"
    selector_root = root / "selector"
    source_root.mkdir(parents=True, exist_ok=True)
    selector_root.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for name, source in SOURCE_PATHS.items():
        source_hashes[name] = file_sha256(source)
        shutil.copy2(source, source_root / f"{name}{source.suffix}")
    for name, expected in selector_hashes.items():
        source = selector_directory / name
        if file_sha256(source) != expected:
            raise ValueError(f"selector artifact hash drifted while freezing: {name}")
        shutil.copy2(source, selector_root / name)
    _atomic_json(
        root / "manifest.json",
        {
            "schema_version": "arac-independent-frozen-protocol-v1",
            "frozen_at_utc": _utc_now(),
            "config_sha256": file_sha256(config_path),
            "source_hashes": source_hashes,
            "selector_hashes": dict(selector_hashes),
            "phase1_protocol": PHASE1_PROTOCOL,
            "phase1_fes": phase1_fes,
            "phase1_max_fes": PHASE1_FES,
            "actions": list(ACTION_NAMES),
            "labels_derived_from_action_outcomes": True,
        },
    )


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
        with ProcessPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
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
    return {
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
    if not resume and active_output.exists() and any(active_output.iterdir()):
        raise ValueError("fresh output root is not empty; use --resume")
    active_output.mkdir(parents=True, exist_ok=True)
    selector_directory = (REPOSITORY_ROOT / config["selector_directory"]).resolve()
    selector_hashes = dict(config["selector_files"])
    _validate_selector_files(selector_directory, selector_hashes)
    _freeze_protocol(
        active_output,
        config_path=config_path,
        selector_directory=selector_directory,
        selector_hashes=selector_hashes,
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


def _run_arm(context: ArmContext) -> dict[str, object]:
    problem = AobBenchmark().load(context.case_id)
    ledger = EvaluationLedger(problem, context.max_fes)
    checkpoint = run_phase1(problem, ledger, run_seed=context.run_seed).checkpoint
    result = ActionRegistry().execute(
        ActionContext(
            context.action_name,
            checkpoint,
            problem,
            ledger,
            action_seed=context.run_seed,
        )
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
) -> tuple[OutcomeRecord, ...]:
    root = Path(output_root).resolve()
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
    _atomic_json(
        root / "summary.json",
        {
            "schema_version": "arac-outcome-campaign-summary-v1",
            "record_count": len(records),
            "arm_count": len(rows),
            "max_fes": max_fes,
            "labels_derived_from_action_outcomes": True,
            "label_counts": dict(sorted(Counter(row.action_label for row in records).items())),
        },
    )
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
    )
    holdout = run_outcome_campaign(
        config["cases"],
        config["holdout_seeds"],
        max_fes=budget,
        max_workers=workers,
        output_root=root / "holdout",
        resume=resume,
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
