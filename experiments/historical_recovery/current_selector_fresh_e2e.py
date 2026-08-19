"""Run fresh-seed Phase-I to grouped-selector to recovered-Phase-II E2E."""

# Thread caps must be set before numerical imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import traceback
from typing import Any
import warnings

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

from threadpoolctl import threadpool_info, threadpool_limits

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.analysis.grouped_outcome_selector import GroupedOutcomeSelector, file_sha256
from arac.benchmarks.aob import AobBenchmark
from arac.evidence.phase1 import PHASE1_FEATURE_NAMES, PHASE1_PROTOCOL, run_phase1
from arac.runtime.contracts import ACTION_NAMES, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("current_selector_fresh_e2e_protocol.json")
PROTOCOL_SCHEMA = "arac-current-selector-fresh-e2e-protocol-v1"
RECEIPT_SCHEMA = "arac-current-selector-fresh-e2e-receipt-v1"
SUMMARY_SCHEMA = "arac-current-selector-fresh-e2e-summary-v1"
REFERENCE_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*"
    r"(?:\([^)]*\)\s*)?\+/-\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$"
)
SOURCE_PATHS = (
    "experiments/historical_recovery/current_selector_fresh_e2e.py",
    "src/arac/analysis/grouped_outcome_selector.py",
    "src/arac/evidence/phase1.py",
    "src/arac/evidence/structural.py",
    "src/arac/evidence/mechanism_features.py",
    "src/arac/actions/recovered_registry.py",
    "src/arac/actions/recovered.py",
    "src/arac/actions/ctp.py",
    "src/arac/actions/gcb.py",
    "src/arac/actions/_execution.py",
    "src/arac/runtime/contracts.py",
    "src/arac/runtime/ledger.py",
    "src/arac/runtime/optimizers.py",
    "src/arac/runtime/phase2.py",
    "src/arac/benchmarks/aob.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> tuple[int, str]:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if path.is_file() and "__pycache__" not in relative.parts and path.suffix != ".pyc":
            entries.append((relative.as_posix(), _sha256(path)))
    return len(entries), canonical_sha256(entries)


def _parse_reference(value: str) -> tuple[float, float]:
    match = REFERENCE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid historical reference cell: {value!r}")
    return float(match.group(1)), float(match.group(2))


def _load_references(path: Path, column: str, cases: Sequence[str]) -> dict[str, tuple[float, float, str]]:
    references = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            case = str(row["case"]).strip()
            if case in cases:
                raw = str(row[column])
                mean, sample_std = _parse_reference(raw)
                references[case] = (mean, sample_std, raw)
    if set(references) != set(cases):
        raise ValueError("historical reference table does not contain all protocol cases")
    return references


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    cases = tuple(str(value) for value in protocol.get("cases", ()))
    fresh_seeds = tuple(int(value) for value in protocol.get("fresh_seeds", ()))
    training_seeds = tuple(int(value) for value in protocol.get("training_seeds", ()))
    expected_cases = tuple(f"{family}{index}" for family in "AERS" for index in range(1, 7))
    expected = {
        "schema_version": PROTOCOL_SCHEMA,
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "registry": "RecoveredActionRegistry",
        "allow_out_of_bounds": True,
        "boundary_profile": "official_hcc_no_offspring_clipping_reproduction",
        "native_threads": 1,
        "selector_fallback_allowed": False,
        "probe_execution_allowed": False,
        "racing_execution_allowed": False,
    }
    if any(protocol.get(name) != value for name, value in expected.items()):
        raise ValueError("fresh E2E protocol anchor drifted")
    if set(cases) != set(expected_cases) or len(cases) != 24:
        raise ValueError("fresh E2E protocol must contain AOB-24 exactly once")
    if len(fresh_seeds) != 25 or len(set(fresh_seeds)) != 25:
        raise ValueError("fresh E2E protocol must contain 25 unique seeds")
    if set(fresh_seeds) & set(training_seeds):
        raise ValueError("fresh E2E seeds overlap selector matrix seeds")
    if int(protocol.get("max_workers", 0)) <= 0:
        raise ValueError("fresh E2E worker count must be positive")
    selector_root = REPOSITORY_ROOT / str(protocol["selector_directory"])
    GroupedOutcomeSelector.load(selector_root)
    reference_path = REPOSITORY_ROOT / str(protocol["historical_reference_table"])
    _load_references(reference_path, str(protocol["historical_reference_column"]), cases)
    return protocol


@dataclass(frozen=True)
class Context:
    case_id: str
    run_seed: int
    output_root: Path
    selector_root: Path
    manifest_sha256: str

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "runs" / self.case_id / f"seed_{self.run_seed}" / "receipt.json"

    @property
    def failure_path(self) -> Path:
        return self.output_root / "failures" / self.case_id / f"seed_{self.run_seed}.json"


def _warning_rows(caught: Sequence[warnings.WarningMessage]) -> list[dict[str, Any]]:
    counts = Counter((item.category.__name__, str(item.message)) for item in caught)
    return [
        {"category": category, "message": message, "count": count}
        for (category, message), count in sorted(counts.items())
    ]


def _run_context(context: Context) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            pools = [
                {
                    "internal_api": item.get("internal_api"),
                    "num_threads": item.get("num_threads"),
                    "prefix": item.get("prefix"),
                }
                for item in threadpool_info()
            ]
            if any(item["num_threads"] != 1 for item in pools):
                raise RuntimeError(f"native thread limit is not one: {pools}")
            selector = GroupedOutcomeSelector.load(context.selector_root)
            registry = RecoveredActionRegistry()
            problem = AobBenchmark().load(context.case_id)
            ledger = EvaluationLedger(
                problem,
                3_000_000,
                allow_out_of_bounds=registry.allow_out_of_bounds,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                phase1 = run_phase1(problem, ledger, run_seed=context.run_seed)
                checkpoint = phase1.checkpoint
                selected = selector.select(checkpoint.feature_names, checkpoint.feature_values)
                result = execute_phase2_action(
                    selected,
                    checkpoint,
                    problem,
                    ledger,
                    action_seed=context.run_seed,
                    registry=registry,
                )
            if (
                checkpoint.phase1_fes != 180_000
                or result.consumed_fes != 2_820_000
                or result.terminal_fes != 3_000_000
                or result.checkpoint_hash != checkpoint.checkpoint_hash
                or result.action_name != selected
                or result.final_error != ledger.best_error
                or not math.isfinite(result.final_error)
            ):
                raise RuntimeError(f"{context.key} fresh E2E terminal contract failed")
            body = {
                "schema_version": RECEIPT_SCHEMA,
                "manifest_sha256": context.manifest_sha256,
                "case_id": context.case_id,
                "run_seed": context.run_seed,
                "action_seed": result.action_seed,
                "phase1_protocol": checkpoint.protocol,
                "phase1_fes": checkpoint.phase1_fes,
                "phase2_fes": result.consumed_fes,
                "terminal_fes": result.terminal_fes,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "feature_names": list(checkpoint.feature_names),
                "feature_values": list(checkpoint.feature_values),
                "selected_action": selected,
                "selector_model_sha256": selector.metadata["model_sha256"],
                "selector_metadata_hash": selector.metadata["metadata_hash"],
                "selector_fallback_used": False,
                "action_result": result.payload(),
                "action_result_hash": result.result_hash,
                "final_error": result.final_error,
                "allow_out_of_bounds": ledger.allow_out_of_bounds,
                "boundary_profile": "official_hcc_no_offspring_clipping_reproduction",
                "runtime_warnings": _warning_rows(caught),
                "threadpools": pools,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            receipt = {**body, "receipt_hash": canonical_sha256(body)}
            _write_json(context.receipt_path, receipt)
            return receipt
    except BaseException as exc:
        _write_json(
            context.failure_path,
            {
                "schema_version": "arac-current-selector-fresh-e2e-failure-v1",
                "manifest_sha256": context.manifest_sha256,
                "key": context.key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def _validate_receipt(path: Path, context: Context) -> dict[str, Any]:
    receipt = _load_json(path)
    claimed = receipt.pop("receipt_hash", None)
    if claimed != canonical_sha256(receipt):
        raise ValueError(f"{context.key} receipt hash drifted")
    selector = GroupedOutcomeSelector.load(context.selector_root)
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "action_seed": context.run_seed,
        "phase1_protocol": PHASE1_PROTOCOL,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "terminal_fes": 3_000_000,
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "selector_model_sha256": selector.metadata["model_sha256"],
        "selector_metadata_hash": selector.metadata["metadata_hash"],
        "selector_fallback_used": False,
        "allow_out_of_bounds": True,
        "boundary_profile": "official_hcc_no_offspring_clipping_reproduction",
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"{context.key} receipt field drifted: {name}")
    result = receipt.get("action_result")
    if not isinstance(result, Mapping) or (
        receipt.get("selected_action") not in ACTION_NAMES
        or receipt.get("checkpoint_hash") != result.get("checkpoint_hash")
        or receipt.get("selected_action") != result.get("action_name")
        or receipt.get("action_result_hash") != canonical_sha256(result)
        or receipt.get("final_error") != result.get("final_error")
        or not math.isfinite(float(receipt.get("final_error", math.nan)))
    ):
        raise ValueError(f"{context.key} action result drifted")
    receipt["receipt_hash"] = claimed
    return receipt


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    selector_root = REPOSITORY_ROOT / str(protocol["selector_directory"])
    reference_path = REPOSITORY_ROOT / str(protocol["historical_reference_table"])
    vendor_count, vendor_hash = _tree_sha256(REPOSITORY_ROOT / "vendor/aob")
    body = {
        "schema_version": "arac-current-selector-fresh-e2e-manifest-v1",
        "protocol_sha256": _sha256(protocol_path),
        "source_sha256": {relative: _sha256(REPOSITORY_ROOT / relative) for relative in SOURCE_PATHS},
        "selector_model_sha256": file_sha256(selector_root / "grouped_outcome_selector.joblib"),
        "selector_metadata_sha256": file_sha256(selector_root / "grouped_outcome_selector.json"),
        "selector_evaluation_sha256": file_sha256(selector_root / "grouped_evaluation.json"),
        "historical_reference_sha256": _sha256(reference_path),
        "vendor_aob_file_count": vendor_count,
        "vendor_aob_tree_sha256": vendor_hash,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _contexts(protocol: Mapping[str, Any], manifest_sha256: str) -> tuple[Context, ...]:
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    selector_root = REPOSITORY_ROOT / str(protocol["selector_directory"])
    return tuple(
        Context(case, int(seed), output_root, selector_root, manifest_sha256)
        for seed in protocol["fresh_seeds"]
        for case in protocol["cases"]
    )


def _summary(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    references = _load_references(
        REPOSITORY_ROOT / str(protocol["historical_reference_table"]),
        str(protocol["historical_reference_column"]),
        tuple(str(value) for value in protocol["cases"]),
    )
    cases = []
    for case in protocol["cases"]:
        case_rows = [row for row in rows if row["case_id"] == case]
        errors = [float(row["final_error"]) for row in case_rows]
        reference_mean, reference_std, reference_raw = references[str(case)]
        mean = statistics.fmean(errors)
        sample_std = statistics.stdev(errors)
        cases.append(
            {
                "case_id": case,
                "seed_count": len(case_rows),
                "mean_final_error": mean,
                "sample_std_final_error": sample_std,
                "reference_mean": reference_mean,
                "reference_sample_std": reference_std,
                "reference_raw": reference_raw,
                "mean_minus_reference": mean - reference_mean,
                "mean_ratio_to_reference": mean / reference_mean if reference_mean else math.inf,
                "recovered_or_exceeded": mean <= reference_mean,
                "selection_counts": dict(
                    sorted(Counter(str(row["selected_action"]) for row in case_rows).items())
                ),
            }
        )
    recovered = sum(bool(row["recovered_or_exceeded"]) for row in cases)
    warning_count = sum(
        int(warning["count"]) for row in rows for warning in row["runtime_warnings"]
    )
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "context_count": len(rows),
        "case_count": len(cases),
        "seed_count_per_case": len(protocol["fresh_seeds"]),
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "terminal_fes": 3_000_000,
        "all_terminal_fes_exact": all(row["terminal_fes"] == 3_000_000 for row in rows),
        "all_checkpoint_bindings_exact": all(
            row["checkpoint_hash"] == row["action_result"]["checkpoint_hash"] for row in rows
        ),
        "selector_fallback_count": sum(bool(row["selector_fallback_used"]) for row in rows),
        "runtime_warning_count": warning_count,
        "selection_counts": dict(
            sorted(Counter(str(row["selected_action"]) for row in rows).items())
        ),
        "recovered_case_count": recovered,
        "failed_case_count": len(cases) - recovered,
        "gate_passed": recovered == len(cases),
        "case_summaries": cases,
    }
    return {**body, "summary_hash": canonical_sha256(body)}


def run_campaign(
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    resume: bool,
    max_workers: int | None = None,
) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    expected_manifest = _manifest(resolved, protocol)
    if resume:
        if _load_json(output_root / "manifest.json") != expected_manifest:
            raise ValueError("fresh E2E manifest drifted")
    else:
        if output_root.exists():
            raise FileExistsError(f"fresh E2E output already exists: {output_root}")
        output_root.mkdir(parents=True)
        _write_json(output_root / "protocol.json", protocol)
        _write_json(output_root / "manifest.json", expected_manifest)
    contexts = _contexts(protocol, str(expected_manifest["manifest_sha256"]))
    rows = []
    pending = []
    for context in contexts:
        if resume and context.receipt_path.is_file():
            rows.append(_validate_receipt(context.receipt_path, context))
        else:
            pending.append(context)
    failures = []

    def write_progress() -> None:
        _write_json(
            output_root / "progress.json",
            {
                "schema_version": "arac-current-selector-fresh-e2e-progress-v1",
                "total": len(contexts),
                "completed": len(rows),
                "failed": len(failures),
                "pending": len(contexts) - len(rows) - len(failures),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    write_progress()
    workers = int(protocol["max_workers"] if max_workers is None else max_workers)
    if workers <= 0:
        raise ValueError("max_workers must be positive")
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        futures = {executor.submit(_run_context, context): context for context in pending}
        for future in as_completed(futures):
            context = futures[future]
            try:
                future.result()
                rows.append(_validate_receipt(context.receipt_path, context))
            except BaseException as exc:
                failures.append({"key": context.key, "error": f"{type(exc).__name__}: {exc}"})
            write_progress()
    if failures:
        _write_json(output_root / "failure_summary.json", {"failures": failures})
        raise RuntimeError(f"fresh E2E campaign has {len(failures)} failed contexts")
    summary = _summary(rows, protocol)
    _write_json(output_root / "summary.json", summary)
    return summary


def verify_campaign(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    manifest = _manifest(resolved, protocol)
    if _load_json(output_root / "manifest.json") != manifest:
        raise ValueError("fresh E2E manifest drifted")
    contexts = _contexts(protocol, str(manifest["manifest_sha256"]))
    rows = [_validate_receipt(context.receipt_path, context) for context in contexts]
    summary = _summary(rows, protocol)
    if _load_json(output_root / "summary.json") != summary:
        raise ValueError("fresh E2E summary drifted")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    result = (
        run_campaign(args.protocol, resume=args.resume, max_workers=args.workers)
        if args.command == "run"
        else verify_campaign(args.protocol)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
