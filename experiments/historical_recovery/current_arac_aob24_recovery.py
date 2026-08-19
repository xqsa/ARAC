"""Run and verify current ARAC-Core against all 24 displayed AOB means."""

# ruff: noqa: E402

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

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
from pathlib import Path
import statistics
import traceback

from threadpoolctl import threadpool_info, threadpool_limits

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "current_arac_aob24_recovery_protocol.json"
CASES = tuple(f"{family}{index}" for family in "ESRA" for index in range(1, 7))
SEEDS = tuple(range(117, 142))
SOURCE_PATHS = (
    "experiments/historical_recovery/current_arac_aob24_recovery.py",
    "src/arac/core.py",
    "src/arac/evidence/phase1.py",
    "src/arac/evidence/structural.py",
    "src/arac/evidence/mechanism_features.py",
    "src/arac/runtime/contracts.py",
    "src/arac/runtime/ledger.py",
    "src/arac/runtime/optimizers.py",
    "src/arac/actions/registry.py",
    "src/arac/actions/_execution.py",
    "src/arac/actions/aor.py",
    "src/arac/actions/ctp.py",
    "src/arac/actions/smp.py",
    "src/arac/actions/gcb.py",
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _production_hcc_imports() -> list[str]:
    matches = []
    for path in (REPOSITORY_ROOT / "src" / "arac").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        if any(token in source for token in ("from hcc", "import hcc", "vendor.hcc", "vendor/hcc")):
            matches.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    return matches


def _parse_reference(raw: str) -> tuple[float, float]:
    parts = raw.replace("±", "+/-").split("+/-")
    if len(parts) != 2:
        raise ValueError(f"invalid reference cell: {raw}")
    mean, sample_std = (float(part.strip()) for part in parts)
    if not all(math.isfinite(value) and value >= 0.0 for value in (mean, sample_std)):
        raise ValueError(f"invalid reference values: {raw}")
    return mean, sample_std


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = _read_json(protocol_path)
    expected = {
        "schema_version": "arac-current-arac-aob24-recovery-protocol-v1",
        "cases": list(CASES),
        "seeds": list(SEEDS),
        "total_budget_fes": 3_000_000,
        "expected_phase1_fes": 180_000,
        "reference_column": "ARAC Mean +/- Std",
        "selector_execution_allowed": False,
        "probe_execution_allowed": False,
        "racing_execution_allowed": False,
        "hcc_runtime_imports_allowed": False,
        "comparison_rule": "per_case_mean_final_error <= displayed_reference_mean",
        "terminal_run_count": 1,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol drifted: {key}")
    reference_path = REPOSITORY_ROOT / str(protocol["reference_table"])
    if _sha256(reference_path) != protocol.get("reference_table_sha256"):
        raise ValueError("reference table hash drifted")
    return protocol


def load_references(protocol: Mapping[str, object]) -> dict[str, tuple[float, float, str]]:
    path = REPOSITORY_ROOT / str(protocol["reference_table"])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    references = {}
    for row in rows:
        case = str(row["case"])
        if case not in CASES:
            continue
        raw = str(row[str(protocol["reference_column"])])
        mean, sample_std = _parse_reference(raw)
        references[case] = (mean, sample_std, raw)
    if set(references) != set(CASES):
        raise ValueError("reference table does not contain exactly 24 AOB cases")
    return references


def _vendor_tree_hash() -> tuple[int, str]:
    root = REPOSITORY_ROOT / "vendor" / "aob"
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if path.is_file() and "__pycache__" not in relative.parts and path.suffix.lower() not in {".pyc", ".pyo"}:
            entries.append((relative.as_posix(), _sha256(path)))
    return len(entries), _canonical(entries)


def preflight(path: Path = PROTOCOL_PATH, *, resume: bool = False) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    load_references(protocol)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    if output_root.exists() and not resume:
        raise ValueError(f"fresh output already exists: {output_root}")
    missing = [relative for relative in SOURCE_PATHS if not (REPOSITORY_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source inputs: {missing}")
    imports = _production_hcc_imports()
    if imports:
        raise ValueError(f"production HCC imports remain: {imports}")
    vendor_count, vendor_hash = _vendor_tree_hash()
    return {
        "schema_version": "arac-current-arac-aob24-recovery-preflight-v1",
        "protocol_sha256": _sha256(protocol_path),
        "reference_table_sha256": protocol["reference_table_sha256"],
        "output_root": str(output_root.resolve()),
        "source_sha256": {relative: _sha256(REPOSITORY_ROOT / relative) for relative in SOURCE_PATHS},
        "vendor_aob_file_count": vendor_count,
        "vendor_aob_tree_sha256": vendor_hash,
        "production_hcc_runtime_imports": imports,
        "selector_execution_allowed": False,
        "probe_execution_allowed": False,
        "racing_execution_allowed": False,
        "context_count": len(CASES) * len(SEEDS),
        "passed": True,
    }


@dataclass(frozen=True)
class Context:
    case_id: str
    seed: int
    output_root: Path
    manifest_sha256: str

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.seed}"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "runs" / self.case_id / f"seed_{self.seed}" / "receipt.json"

    @property
    def failure_path(self) -> Path:
        return self.output_root / "failures" / self.case_id / f"seed_{self.seed}.json"


def _run_context(context: Context) -> dict[str, object]:
    from arac.benchmarks.aob import AobBenchmark
    from arac.core import run_arac

    started = datetime.now(UTC)
    problem = AobBenchmark().load(
        context.case_id,
        output_directory=context.receipt_path.parent / "benchmark",
    )
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        result = run_arac(
            problem,
            total_budget_fes=3_000_000,
            run_seed=context.seed,
            action_seed=context.seed,
        )
    action = result.core.action_result
    payload = {
        "schema_version": "arac-current-arac-aob24-recovery-receipt-v1",
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.seed,
        "action_seed": context.seed,
        "phase1_fes": result.phase1.checkpoint.phase1_fes,
        "phase1_checkpoint_hash": result.phase1.checkpoint.checkpoint_hash,
        "action_checkpoint_hash": action.checkpoint_hash,
        "selected_action": result.core.decision.action_name,
        "selection_reason": result.core.decision.reason,
        "selection_scores": list(result.core.decision.scores),
        "relation_count": result.core.decision.relation_count,
        "largest_component_fraction": result.core.decision.largest_component_fraction,
        "phase2_consumed_fes": action.consumed_fes,
        "terminal_fes": action.terminal_fes,
        "final_error": action.final_error,
        "terminal_state_finite": math.isfinite(action.final_error),
        "production_hcc_runtime_imports": _production_hcc_imports(),
        "selector_execution_allowed": False,
        "probe_execution_allowed": False,
        "racing_execution_allowed": False,
        "threadpools": pools,
        "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
    }
    receipt = {**payload, "receipt_sha256": _canonical(payload)}
    _write(context.receipt_path, receipt)
    return receipt


def _validate_receipt(path: Path, context: Context) -> dict[str, object]:
    receipt = _read_json(path)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != _canonical(receipt):
        raise ValueError(f"{context.key} receipt hash drifted")
    expected = {
        "schema_version": "arac-current-arac-aob24-recovery-receipt-v1",
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.seed,
        "action_seed": context.seed,
        "phase1_fes": 180_000,
        "phase2_consumed_fes": 2_820_000,
        "terminal_fes": 3_000_000,
        "terminal_state_finite": True,
        "production_hcc_runtime_imports": [],
        "selector_execution_allowed": False,
        "probe_execution_allowed": False,
        "racing_execution_allowed": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{context.key} receipt field drifted: {key}")
    if receipt.get("phase1_checkpoint_hash") != receipt.get("action_checkpoint_hash"):
        raise ValueError(f"{context.key} checkpoint binding failed")
    if receipt.get("selected_action") not in {"aor", "ctp", "smp", "gcb"}:
        raise ValueError(f"{context.key} selected action is invalid")
    if not math.isfinite(float(receipt.get("final_error", math.nan))):
        raise ValueError(f"{context.key} final error is not finite")
    receipt["receipt_sha256"] = claimed
    return receipt


def _manifest(protocol_path: Path, gate: Mapping[str, object]) -> dict[str, object]:
    body = {
        "schema_version": "arac-current-arac-aob24-recovery-manifest-v1",
        "protocol_sha256": _sha256(protocol_path),
        "reference_table_sha256": gate["reference_table_sha256"],
        "source_sha256": gate["source_sha256"],
        "vendor_aob_file_count": gate["vendor_aob_file_count"],
        "vendor_aob_tree_sha256": gate["vendor_aob_tree_sha256"],
        "terminal_run_count": 1,
    }
    return {**body, "manifest_sha256": _canonical(body)}


def _load_manifest(output_root: Path, protocol_path: Path) -> dict[str, object]:
    manifest = _read_json(output_root / "manifest.json")
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != _canonical(manifest):
        raise ValueError("manifest hash drifted")
    if manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("protocol hash drifted")
    for relative, expected in dict(manifest["source_sha256"]).items():
        if _sha256(REPOSITORY_ROOT / relative) != expected:
            raise ValueError(f"campaign source drifted: {relative}")
    count, tree_hash = _vendor_tree_hash()
    if count != manifest.get("vendor_aob_file_count") or tree_hash != manifest.get("vendor_aob_tree_sha256"):
        raise ValueError("vendor AOB tree drifted")
    manifest["manifest_sha256"] = claimed
    return manifest


def _contexts(protocol: Mapping[str, object], manifest_sha256: str) -> tuple[Context, ...]:
    root = REPOSITORY_ROOT / str(protocol["output_root"])
    return tuple(
        Context(case, seed, root, manifest_sha256)
        for seed in SEEDS
        for case in CASES
    )


def _summarize(
    rows: Sequence[Mapping[str, object]],
    references: Mapping[str, tuple[float, float, str]],
    *,
    max_workers: int,
) -> dict[str, object]:
    cases = []
    for case in CASES:
        case_rows = [row for row in rows if row["case_id"] == case]
        errors = [float(row["final_error"]) for row in case_rows]
        mean = statistics.fmean(errors)
        sample_std = statistics.stdev(errors)
        reference_mean, reference_std, raw = references[case]
        cases.append({
            "case_id": case,
            "seed_count": len(case_rows),
            "mean_final_error": mean,
            "sample_std_final_error": sample_std,
            "reference_mean": reference_mean,
            "reference_sample_std": reference_std,
            "reference_raw": raw,
            "mean_minus_reference": mean - reference_mean,
            "mean_ratio_to_reference": mean / reference_mean if reference_mean else (0.0 if mean == 0 else math.inf),
            "recovered_or_exceeded": mean <= reference_mean,
            "selection_counts": dict(sorted(Counter(str(row["selected_action"]) for row in case_rows).items())),
        })
    passing = sum(bool(row["recovered_or_exceeded"]) for row in cases)
    body = {
        "schema_version": "arac-current-arac-aob24-recovery-summary-v1",
        "generated_at_utc": _utc_now(),
        "context_count": len(rows),
        "case_count": len(cases),
        "seed_count_per_case": len(SEEDS),
        "max_workers": max_workers,
        "phase1_fes": 180_000,
        "terminal_fes": 3_000_000,
        "all_terminal_fes_exact": all(row["terminal_fes"] == 3_000_000 for row in rows),
        "all_checkpoint_bindings_exact": all(row["phase1_checkpoint_hash"] == row["action_checkpoint_hash"] for row in rows),
        "selection_counts": dict(sorted(Counter(str(row["selected_action"]) for row in rows).items())),
        "recovered_case_count": passing,
        "failed_case_count": len(cases) - passing,
        "gate_passed": passing == len(cases),
        "case_summaries": cases,
    }
    return {**body, "summary_sha256": _canonical(body)}


def _write_results(output_root: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "case_id", "run_seed", "selected_action", "selection_reason", "phase1_fes",
        "phase2_consumed_fes", "terminal_fes", "final_error", "elapsed_seconds",
        "phase1_checkpoint_hash", "receipt_sha256",
    )
    path = output_root / "results.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    temporary.replace(path)


def run(path: Path = PROTOCOL_PATH, *, resume: bool = False, workers: int | None = None) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    gate = preflight(protocol_path, resume=resume)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    max_workers = int(protocol["max_workers"] if workers is None else workers)
    if max_workers <= 0:
        raise ValueError("workers must be positive")
    if resume:
        manifest = _load_manifest(output_root, protocol_path)
    else:
        output_root.mkdir(parents=True)
        _write(output_root / "protocol.json", protocol)
        _write(output_root / "preflight.json", gate)
        manifest = _manifest(protocol_path, gate)
        _write(output_root / "manifest.json", manifest)
    manifest_sha256 = str(manifest["manifest_sha256"])
    contexts = _contexts(protocol, manifest_sha256)
    rows: dict[str, dict[str, object]] = {}
    pending = []
    for context in contexts:
        if resume and context.receipt_path.is_file():
            rows[context.key] = _validate_receipt(context.receipt_path, context)
        else:
            pending.append(context)
    failures: dict[str, str] = {}

    def write_progress() -> None:
        _write(output_root / "parallel_progress.json", {
            "schema_version": "arac-current-arac-aob24-recovery-progress-v1",
            "planned": len(contexts),
            "completed": len(rows),
            "failed": len(failures),
            "pending": len(contexts) - len(rows) - len(failures),
            "max_workers": min(max_workers, len(contexts)),
            "updated_at_utc": _utc_now(),
            "failures": failures,
        })

    write_progress()
    if pending:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(pending)), max_tasks_per_child=1) as pool:
            futures = {pool.submit(_run_context, context): context for context in pending}
            for future in as_completed(futures):
                context = futures[future]
                try:
                    future.result()
                    rows[context.key] = _validate_receipt(context.receipt_path, context)
                except BaseException as error:
                    failures[context.key] = f"{type(error).__name__}: {error}"
                    _write(context.failure_path, {
                        "schema_version": "arac-current-arac-aob24-recovery-failure-v1",
                        "case_id": context.case_id,
                        "run_seed": context.seed,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    })
                write_progress()
                status = "complete" if context.key in rows else "failed"
                print(f"[{len(rows) + len(failures):04d}/{len(contexts)}] {context.key} {status}", flush=True)
    if failures:
        raise RuntimeError(f"campaign has {len(failures)} failed contexts")
    ordered = [rows[context.key] for context in contexts]
    summary = _summarize(ordered, load_references(protocol), max_workers=max_workers)
    _write(output_root / "summary.json", summary)
    _write_results(output_root, ordered)
    return summary


def verify(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    manifest = _load_manifest(output_root, protocol_path)
    contexts = _contexts(protocol, str(manifest["manifest_sha256"]))
    rows = [_validate_receipt(context.receipt_path, context) for context in contexts]
    expected = _summarize(rows, load_references(protocol), max_workers=int(protocol["max_workers"]))
    stored = _read_json(output_root / "summary.json")
    expected_comparable = dict(expected)
    stored_comparable = dict(stored)
    expected_comparable.pop("generated_at_utc", None)
    stored_comparable.pop("generated_at_utc", None)
    expected_comparable.pop("summary_sha256", None)
    stored_claimed = stored_comparable.pop("summary_sha256", None)
    if stored_claimed != _canonical({key: value for key, value in stored.items() if key != "summary_sha256"}):
        raise ValueError("summary hash drifted")
    if stored_comparable != expected_comparable:
        raise ValueError("summary content drifted")
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "verify"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.protocol, resume=args.resume)
    elif args.command == "run":
        result = run(args.protocol, resume=args.resume, workers=args.workers)
    else:
        result = verify(args.protocol)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
