"""Run paired current-vs-historical-compatible SMP/GCB lifecycle diagnostics."""

# Thread caps must be applied before NumPy, PyPop7, or ARAC imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import traceback
import warnings
from typing import Any, Mapping, Sequence

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

from threadpoolctl import threadpool_info, threadpool_limits

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery import current_recovered_four_arm as fixed


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name(
    "recovery_action_lifecycle_diagnostic_protocol_v1.json"
)
SCHEMA = "arac-recovery-action-lifecycle-diagnostic-receipt-v1"
SUMMARY_SCHEMA = "arac-recovery-action-lifecycle-diagnostic-summary-v1"
MANIFEST_SCHEMA = "arac-recovery-action-lifecycle-diagnostic-manifest-v1"
EXPECTED_CASES = ("E2", "E3", "E4", "E5", "E6", "R1", "R2", "R3", "R4", "R5", "R6")
EXPECTED_SEEDS = (117, 123, 129, 135, 141)
EXPECTED_ACTIONS = {
    **{f"E{index}": "smp" for index in range(2, 7)},
    **{f"R{index}": "gcb" for index in range(1, 7)},
}
EXPECTED_VARIANTS = ("current", "historical_compatible")
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
PHASE2_FES = 2_820_000
MAX_WORKERS = 24
SOURCE_PATHS = (
    "experiments/historical_recovery/recovery_action_lifecycle_diagnostic.py",
    "experiments/historical_recovery/recovery_action_lifecycle_diagnostic_protocol_v1.json",
    "src/arac/actions/recovered.py",
    "src/arac/actions/gcb.py",
    "src/arac/actions/_execution.py",
    "src/arac/runtime/ledger.py",
    "src/arac/runtime/contracts.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical(payload: object) -> str:
    return canonical_sha256(payload)


def _resolved(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = _load_json(protocol_path)
    expected = {
        "schema_version": "arac-recovery-action-lifecycle-diagnostic-protocol-v1",
        "status": "frozen_paired_diagnostic",
        "cases": list(EXPECTED_CASES),
        "seeds": list(EXPECTED_SEEDS),
        "variants": list(EXPECTED_VARIANTS),
        "total_budget_fes": TOTAL_BUDGET_FES,
        "phase1_fes": PHASE1_FES,
        "phase2_fes": PHASE2_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "native_threads": 1,
        "max_workers": MAX_WORKERS,
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
        "reference_thresholds_used_for_decision": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"diagnostic protocol drifted: {key}")
    action_by_case = {str(key): str(value) for key, value in protocol.get("action_by_case", {}).items()}
    if action_by_case != EXPECTED_ACTIONS:
        raise ValueError("diagnostic action mapping drifted")
    for key in ("checkpoint_root", "current_e2e_receipt_root", "output_root"):
        if key != "output_root" and not _resolved(str(protocol[key])).is_dir():
            raise FileNotFoundError(f"diagnostic source root is missing: {key}")
    source_map = protocol.get("historical_compatible_sources")
    if not isinstance(source_map, Mapping) or set(source_map) != {"smp", "gcb"}:
        raise ValueError("historical-compatible source map drifted")
    for action, relative in source_map.items():
        if action not in {"smp", "gcb"} or not _resolved(str(relative)).is_file():
            raise FileNotFoundError(f"historical-compatible source is missing: {action}")
    return protocol


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = list(SOURCE_PATHS) + [str(value) for value in protocol["historical_compatible_sources"].values()]
    return {path: _sha256(_resolved(path)) for path in sorted(set(paths))}


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": _sha256(protocol_path),
        "cases": list(protocol["cases"]),
        "seeds": list(protocol["seeds"]),
        "variants": list(protocol["variants"]),
        "source_sha256": _source_hashes(protocol),
        "checkpoint_root": protocol["checkpoint_root"],
        "current_e2e_receipt_root": protocol["current_e2e_receipt_root"],
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
    }
    return {**body, "manifest_sha256": _canonical(body)}


class ArmContext:
    def __init__(self, case_id: str, run_seed: int, variant: str, output_root: Path, manifest_sha256: str):
        self.case_id = case_id
        self.run_seed = run_seed
        self.variant = variant
        self.output_root = Path(output_root)
        self.manifest_sha256 = manifest_sha256

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}:{self.variant}"

    @property
    def receipt_path(self) -> Path:
        action = EXPECTED_ACTIONS[self.case_id]
        return self.output_root / "arms" / self.case_id / f"seed_{self.run_seed}" / f"{action}_{self.variant}.json"

    @property
    def failure_path(self) -> Path:
        return self.output_root / "failures" / self.case_id / f"seed_{self.run_seed}" / f"{self.variant}.json"


_LEGACY_MODULES: dict[str, Any] = {}


def _legacy_module(action: str, protocol: Mapping[str, Any]) -> Any:
    if action not in _LEGACY_MODULES:
        path = _resolved(str(protocol["historical_compatible_sources"][action]))
        module_name = f"arac_recovery_historical_compatible_{action}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load historical-compatible {action} source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LEGACY_MODULES[action] = module
    return _LEGACY_MODULES[action]


def _threadpools() -> list[dict[str, Any]]:
    return [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
        }
        for item in threadpool_info()
    ]


def _warning_rows(caught: Sequence[warnings.WarningMessage]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for item in caught:
        key = (item.category.__name__, str(item.message))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"category": category, "message": message, "count": count}
        for (category, message), count in sorted(counts.items())
    ]


def _checkpoint(context: ArmContext, protocol: Mapping[str, Any]):
    checkpoint_context = fixed.ArmContext(
        case_id=context.case_id,
        run_seed=context.run_seed,
        action_name=EXPECTED_ACTIONS[context.case_id],
        checkpoint_root=_resolved(str(protocol["checkpoint_root"])),
        current_receipt_root=_resolved(str(protocol["current_e2e_receipt_root"])),
        output_root=context.output_root,
        manifest_sha256=context.manifest_sha256,
    )
    return fixed._load_verified_checkpoint(checkpoint_context)


def _execute_variant(
    action: str,
    variant: str,
    checkpoint: Any,
    problem: Any,
    ledger: EvaluationLedger,
    seed: int,
    protocol: Mapping[str, Any],
):
    action_context = ActionContext(action, checkpoint, problem, ledger, action_seed=seed)
    if variant == "current":
        return RecoveredActionRegistry().execute(action_context)
    module = _legacy_module(action, protocol)
    executor = module.SmpExecutor() if action == "smp" else module.GcbExecutor()
    return executor.execute(action_context)


def _run_arm(context: ArmContext, protocol: Mapping[str, Any]) -> dict[str, Any]:
    started = datetime.now(UTC)
    action = EXPECTED_ACTIONS[context.case_id]
    try:
        with threadpool_limits(limits=1):
            pools = _threadpools()
            if not pools or any(item["num_threads"] != 1 for item in pools):
                raise RuntimeError(f"{context.key} native thread limit is not one: {pools}")
            checkpoint = _checkpoint(context, protocol)
            problem = AobBenchmark().load(context.case_id)
            ledger = EvaluationLedger.from_checkpoint(
                problem,
                total_budget=TOTAL_BUDGET_FES,
                phase1_fes=checkpoint.phase1_fes,
                incumbent=checkpoint.incumbent,
                incumbent_error=checkpoint.incumbent_error,
                allow_out_of_bounds=True,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _execute_variant(
                    action,
                    context.variant,
                    checkpoint,
                    problem,
                    ledger,
                    context.run_seed,
                    protocol,
                )
            if (
                result.action_name != action
                or result.action_seed != context.run_seed
                or result.checkpoint_hash != checkpoint.checkpoint_hash
                or result.consumed_fes != PHASE2_FES
                or result.terminal_fes != TOTAL_BUDGET_FES
                or ledger.count != TOTAL_BUDGET_FES
                or not math.isfinite(result.final_error)
                or result.final_error != ledger.best_error
            ):
                raise RuntimeError(f"{context.key} paired action contract failed")
            body = {
                "schema_version": SCHEMA,
                "manifest_sha256": context.manifest_sha256,
                "case_id": context.case_id,
                "run_seed": context.run_seed,
                "action_name": action,
                "variant": context.variant,
                "action_seed": result.action_seed,
                "phase1_protocol": checkpoint.protocol,
                "phase1_fes": checkpoint.phase1_fes,
                "phase2_fes": result.consumed_fes,
                "terminal_fes": result.terminal_fes,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "final_error": result.final_error,
                "route": result.route,
                "action_result": result.payload(),
                "action_result_hash": result.result_hash,
                "runtime_warnings": _warning_rows(caught),
                "threadpools": pools,
                "native_thread_limit_verified": True,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
                "reference_thresholds_used_for_decision": False,
            }
            receipt = {**body, "receipt_sha256": _canonical(body)}
            _write_json(context.receipt_path, receipt)
            return receipt
    except BaseException as exc:
        _write_json(
            context.failure_path,
            {
                "schema_version": "arac-recovery-action-lifecycle-diagnostic-failure-v1",
                "manifest_sha256": context.manifest_sha256,
                "key": context.key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _validate_receipt(path: Path, context: ArmContext) -> dict[str, Any]:
    receipt = _load_json(path)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != _canonical(receipt):
        raise ValueError(f"{context.key} receipt hash drifted")
    action = EXPECTED_ACTIONS[context.case_id]
    expected = {
        "schema_version": SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "action_name": action,
        "variant": context.variant,
        "action_seed": context.run_seed,
        "phase1_fes": PHASE1_FES,
        "phase2_fes": PHASE2_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "native_thread_limit_verified": True,
        "reference_thresholds_used_for_decision": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{context.key} receipt drifted: {key}")
    if not math.isfinite(float(receipt.get("final_error", math.nan))):
        raise ValueError(f"{context.key} final error is not finite")
    if receipt.get("checkpoint_hash") != receipt.get("action_result", {}).get("checkpoint_hash"):
        raise ValueError(f"{context.key} checkpoint binding drifted")
    if receipt.get("action_result_hash") != _canonical(receipt.get("action_result")):
        raise ValueError(f"{context.key} action result hash drifted")
    receipt["receipt_sha256"] = claimed
    return receipt


def _contexts(output_root: Path, manifest_sha256: str, protocol: Mapping[str, Any]) -> tuple[ArmContext, ...]:
    return tuple(
        ArmContext(case_id, int(seed), variant, output_root, manifest_sha256)
        for case_id in protocol["cases"]
        for seed in protocol["seeds"]
        for variant in protocol["variants"]
    )


def _paired_ratio(current: float, historical: float) -> float | None:
    if historical > 0.0:
        return current / historical
    return 1.0 if current == 0.0 else None


def summarize(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {
        (str(row["case_id"]), int(row["run_seed"]), str(row["variant"])): row
        for row in rows
    }
    pairs = []
    for case_id in protocol["cases"]:
        for seed in protocol["seeds"]:
            current = indexed[(str(case_id), int(seed), "current")]
            historical = indexed[(str(case_id), int(seed), "historical_compatible")]
            current_error = float(current["final_error"])
            historical_error = float(historical["final_error"])
            pairs.append(
                {
                    "case_id": case_id,
                    "run_seed": int(seed),
                    "action_name": EXPECTED_ACTIONS[str(case_id)],
                    "same_checkpoint": current["checkpoint_hash"] == historical["checkpoint_hash"],
                    "checkpoint_hash": current["checkpoint_hash"],
                    "current_final_error": current_error,
                    "historical_compatible_final_error": historical_error,
                    "current_to_historical_ratio": _paired_ratio(current_error, historical_error),
                    "historical_compatible_better": historical_error < current_error,
                    "current_route": current["route"],
                    "historical_compatible_route": historical["route"],
                }
            )
    case_summaries = []
    for case_id in protocol["cases"]:
        active = [row for row in pairs if row["case_id"] == case_id]
        current_values = [float(row["current_final_error"]) for row in active]
        historical_values = [float(row["historical_compatible_final_error"]) for row in active]
        ratios = [row["current_to_historical_ratio"] for row in active]
        finite_ratios = [float(value) for value in ratios if value is not None and float(value) > 0.0]
        geometric_ratio = (
            math.exp(sum(math.log(value) for value in finite_ratios) / len(finite_ratios))
            if len(finite_ratios) == len(ratios)
            else None
        )
        case_summaries.append(
            {
                "case_id": case_id,
                "action_name": EXPECTED_ACTIONS[str(case_id)],
                "seed_count": len(active),
                "current_mean": statistics.fmean(current_values),
                "historical_compatible_mean": statistics.fmean(historical_values),
                "current_sample_std": statistics.stdev(current_values),
                "historical_compatible_sample_std": statistics.stdev(historical_values),
                "current_to_historical_geometric_mean_ratio": geometric_ratio,
                "historical_compatible_better_count": sum(bool(row["historical_compatible_better"]) for row in active),
                "current_better_count": sum(not bool(row["historical_compatible_better"]) for row in active),
                "current_mean_higher": statistics.fmean(current_values) > statistics.fmean(historical_values),
            }
        )
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "context_count": len(rows),
        "expected_context_count": len(protocol["cases"]) * len(protocol["seeds"]) * len(protocol["variants"]),
        "pair_count": len(pairs),
        "same_checkpoint_per_pair": all(bool(row["same_checkpoint"]) for row in pairs),
        "exact_terminal_fes": all(int(row["terminal_fes"]) == TOTAL_BUDGET_FES for row in rows),
        "all_final_errors_finite": all(math.isfinite(float(row["final_error"])) for row in rows),
        "reference_thresholds_used_for_decision": False,
        "case_summaries": case_summaries,
        "pairs": pairs,
        "diagnostic_conclusion": {
            "smp_current_mean_higher_case_count": sum(row["current_mean_higher"] and row["action_name"] == "smp" for row in case_summaries),
            "gcb_current_mean_higher_case_count": sum(row["current_mean_higher"] and row["action_name"] == "gcb" for row in case_summaries),
            "historical_compatible_schedule_isolation_only": True,
            "performance_superiority_claim_authorized": False,
        },
    }
    return {**body, "summary_sha256": _canonical(body)}


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != _canonical(manifest):
        raise ValueError("diagnostic manifest hash drifted")
    manifest["manifest_sha256"] = claimed
    return manifest


def run(
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    resume: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    manifest = _manifest(protocol_path, protocol)
    manifest_path = output_root / "manifest.json"
    if output_root.exists() and not resume:
        raise FileExistsError(f"diagnostic output already exists: {output_root}")
    if resume:
        if not manifest_path.is_file() or _load_manifest(manifest_path) != manifest:
            raise ValueError("diagnostic manifest does not match protocol")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, manifest)
    contexts = _contexts(output_root, str(manifest["manifest_sha256"]), protocol)
    worker_count = int(protocol["max_workers"] if workers is None else workers)
    if not 1 <= worker_count <= len(contexts):
        raise ValueError(f"workers must be in 1..{len(contexts)}")
    rows: dict[str, dict[str, Any]] = {}
    pending = []
    for context in contexts:
        if resume and context.receipt_path.is_file():
            rows[context.key] = _validate_receipt(context.receipt_path, context)
        else:
            pending.append(context)
    _write_json(
        output_root / "progress.json",
        {"planned": len(contexts), "completed": len(rows), "pending": len(pending), "failed": 0, "max_workers": worker_count},
    )
    failures: dict[str, str] = {}
    if pending:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(pending))) as pool:
            futures = {pool.submit(_run_arm, context, protocol): context for context in pending}
            for future in as_completed(futures):
                context = futures[future]
                try:
                    future.result()
                    rows[context.key] = _validate_receipt(context.receipt_path, context)
                except BaseException as error:
                    failures[context.key] = f"{type(error).__name__}: {error}"
                _write_json(
                    output_root / "progress.json",
                    {"planned": len(contexts), "completed": len(rows), "pending": len(contexts) - len(rows) - len(failures), "failed": len(failures), "max_workers": worker_count, "failures": failures},
                )
    if failures:
        _write_json(output_root / "failure_summary.json", {"failures": failures})
        raise RuntimeError(f"diagnostic has {len(failures)} failed arms")
    ordered = [rows[context.key] for context in contexts]
    summary = summarize(ordered, protocol)
    _write_json(output_root / "summary.json", summary)
    return summary


def verify(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    manifest = _load_manifest(output_root / "manifest.json")
    contexts = _contexts(output_root, str(manifest["manifest_sha256"]), protocol)
    rows = [_validate_receipt(context.receipt_path, context) for context in contexts]
    expected = summarize(rows, protocol)
    stored = _load_json(output_root / "summary.json")
    claimed = stored.pop("summary_sha256", None)
    if claimed != _canonical(stored):
        raise ValueError("diagnostic summary hash drifted")
    expected.pop("summary_sha256", None)
    expected.pop("generated_at_utc", None)
    stored.pop("generated_at_utc", None)
    if stored != expected:
        raise ValueError("diagnostic summary content drifted")
    stored["summary_sha256"] = claimed
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    result = run(args.protocol, resume=args.resume, workers=args.workers) if args.command == "run" else verify(args.protocol)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
