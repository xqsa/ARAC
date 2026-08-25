"""Run the recovered-vs-frozen SMP lifecycle paired smoke."""

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
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from threadpoolctl import threadpool_info, threadpool_limits

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery import current_recovered_four_arm as fixed


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("recovery_smp_lifecycle_smoke_protocol_v1.json")
SCHEMA = "arac-recovery-smp-lifecycle-smoke-receipt-v1"
SUMMARY_SCHEMA = "arac-recovery-smp-lifecycle-smoke-summary-v1"
MANIFEST_SCHEMA = "arac-recovery-smp-lifecycle-smoke-manifest-v1"
EXPECTED_CASES = ("E2", "E3", "E4", "E5", "E6")
EXPECTED_SEEDS = (117, 123, 129, 135, 141)
EXPECTED_VARIANTS = ("current_recovered", "historical_compatible")
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
PHASE2_FES = 2_820_000
MAX_WORKERS = 24
CURRENT_PROFILE = "historical_compatible_smp_v1_clip_offspring_true"
_LEGACY_MODULE: Any | None = None


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolved(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-recovery-smp-lifecycle-smoke-protocol-v1",
        "status": "frozen_paired_smoke",
        "cases": list(EXPECTED_CASES),
        "seeds": list(EXPECTED_SEEDS),
        "variants": list(EXPECTED_VARIANTS),
        "total_budget_fes": TOTAL_BUDGET_FES,
        "phase1_fes": PHASE1_FES,
        "phase2_fes": PHASE2_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "action_name": "smp",
        "native_threads": 1,
        "max_workers": MAX_WORKERS,
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
        "reference_thresholds_used_for_decision": False,
        "clip_offspring": True,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"SMP smoke protocol drifted: {key}")
    for key in ("checkpoint_root", "current_e2e_receipt_root", "historical_compatible_source"):
        target = _resolved(str(protocol[key]))
        if not target.exists():
            raise FileNotFoundError(f"SMP smoke source is missing: {key}")
    return protocol


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    source_paths = {
        "protocol": protocol_path,
        "campaign": Path(__file__).resolve(),
        "recovered": REPOSITORY_ROOT / "src/arac/actions/recovered.py",
        "recovered_registry": REPOSITORY_ROOT / "src/arac/actions/recovered_registry.py",
        "smp": REPOSITORY_ROOT / "src/arac/actions/smp.py",
        "execution": REPOSITORY_ROOT / "src/arac/actions/_execution.py",
        "contracts": REPOSITORY_ROOT / "src/arac/runtime/contracts.py",
        "ledger": REPOSITORY_ROOT / "src/arac/runtime/ledger.py",
        "historical_smp": _resolved(str(protocol["historical_compatible_source"])),
    }
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_sha256": _sha256(protocol_path),
        "cases": list(protocol["cases"]),
        "seeds": list(protocol["seeds"]),
        "variants": list(protocol["variants"]),
        "action_name": protocol["action_name"],
        "source_sha256": {name: _sha256(path) for name, path in sorted(source_paths.items())},
        "checkpoint_root": protocol["checkpoint_root"],
        "current_e2e_receipt_root": protocol["current_e2e_receipt_root"],
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
        "clip_offspring": True,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


@dataclass(frozen=True)
class ArmContext:
    case_id: str
    run_seed: int
    variant: str
    output_root: Path
    manifest_sha256: str

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}:{self.variant}"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / self.case_id / f"seed_{self.run_seed}" / f"{self.variant}.json"

    @property
    def failure_path(self) -> Path:
        return self.output_root / "failures" / self.case_id / f"seed_{self.run_seed}" / f"{self.variant}.json"


def _checkpoint(context: ArmContext, protocol: Mapping[str, Any]):
    fixed_context = fixed.ArmContext(
        case_id=context.case_id,
        run_seed=context.run_seed,
        action_name="smp",
        checkpoint_root=_resolved(str(protocol["checkpoint_root"])),
        current_receipt_root=_resolved(str(protocol["current_e2e_receipt_root"])),
        output_root=context.output_root,
        manifest_sha256=context.manifest_sha256,
    )
    return fixed._load_verified_checkpoint(fixed_context)


def _historical_module(protocol: Mapping[str, Any]) -> Any:
    global _LEGACY_MODULE
    if _LEGACY_MODULE is None:
        path = _resolved(str(protocol["historical_compatible_source"]))
        spec = importlib.util.spec_from_file_location("arac_recovery_smp_historical_compatible", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load historical-compatible SMP source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LEGACY_MODULE = module
    return _LEGACY_MODULE


def _threadpools() -> list[dict[str, Any]]:
    return [{"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")} for item in threadpool_info()]


def _warning_rows(caught: Sequence[warnings.WarningMessage]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for item in caught:
        key = (item.category.__name__, str(item.message))
        counts[key] = counts.get(key, 0) + 1
    return [{"category": category, "message": message, "count": count} for (category, message), count in sorted(counts.items())]


def _run_arm(context: ArmContext, protocol: Mapping[str, Any]) -> dict[str, Any]:
    started = datetime.now(UTC)
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
            active_context = ActionContext("smp", checkpoint, problem, ledger, action_seed=context.run_seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if context.variant == "current_recovered":
                    result = RecoveredActionRegistry().execute(active_context)
                    lifecycle_profile = CURRENT_PROFILE
                else:
                    result = _historical_module(protocol).SmpExecutor().execute(active_context)
                    lifecycle_profile = "frozen_historical_smp_source_v1_clip_offspring_true"
            if (
                result.action_name != "smp"
                or result.action_seed != context.run_seed
                or result.checkpoint_hash != checkpoint.checkpoint_hash
                or result.consumed_fes != PHASE2_FES
                or result.terminal_fes != TOTAL_BUDGET_FES
                or ledger.count != TOTAL_BUDGET_FES
                or not math.isfinite(result.final_error)
                or result.final_error != ledger.best_error
            ):
                raise RuntimeError(f"{context.key} terminal SMP contract failed")
            body = {
                "schema_version": SCHEMA,
                "manifest_sha256": context.manifest_sha256,
                "case_id": context.case_id,
                "run_seed": context.run_seed,
                "action_name": "smp",
                "variant": context.variant,
                "action_seed": result.action_seed,
                "phase1_fes": PHASE1_FES,
                "phase2_fes": PHASE2_FES,
                "terminal_fes": TOTAL_BUDGET_FES,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "final_error": result.final_error,
                "route": result.route,
                "lifecycle_profile": lifecycle_profile,
                "clip_offspring": True,
                "action_result": result.payload(),
                "action_result_hash": result.result_hash,
                "runtime_warnings": _warning_rows(caught),
                "threadpools": pools,
                "native_thread_limit_verified": True,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
                "reference_thresholds_used_for_decision": False,
            }
            receipt = {**body, "receipt_sha256": canonical_sha256(body)}
            _write_json(context.receipt_path, receipt)
            return receipt
    except BaseException as exc:
        _write_json(context.failure_path, {
            "schema_version": "arac-recovery-smp-lifecycle-smoke-failure-v1",
            "manifest_sha256": context.manifest_sha256,
            "key": context.key,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


def _validate_receipt(path: Path, context: ArmContext, protocol: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_json(path)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != canonical_sha256(receipt):
        raise ValueError(f"{context.key} receipt hash drifted")
    expected_profile = CURRENT_PROFILE if context.variant == "current_recovered" else "frozen_historical_smp_source_v1_clip_offspring_true"
    for key, value in {
        "schema_version": SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "action_name": "smp",
        "variant": context.variant,
        "action_seed": context.run_seed,
        "phase1_fes": PHASE1_FES,
        "phase2_fes": PHASE2_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "lifecycle_profile": expected_profile,
        "clip_offspring": True,
        "native_thread_limit_verified": True,
        "reference_thresholds_used_for_decision": False,
    }.items():
        if receipt.get(key) != value:
            raise ValueError(f"{context.key} receipt drifted: {key}")
    result = receipt.get("action_result")
    if (
        not isinstance(result, Mapping)
        or receipt.get("checkpoint_hash") != result.get("checkpoint_hash")
        or receipt.get("action_result_hash") != canonical_sha256(result)
        or result.get("consumed_fes") != PHASE2_FES
        or result.get("terminal_fes") != TOTAL_BUDGET_FES
        or result.get("final_error") != receipt.get("final_error")
        or not math.isfinite(float(receipt.get("final_error", math.nan)))
    ):
        raise ValueError(f"{context.key} action result drifted")
    receipt["receipt_sha256"] = claimed
    return receipt


def _prepare_output(output_root: Path, manifest: Mapping[str, Any], *, resume: bool) -> None:
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"SMP smoke output already exists: {output_root}")
        if _load_json(manifest_path) != manifest:
            raise ValueError("SMP smoke manifest does not match frozen protocol")
        return
    output_root.mkdir(parents=True)
    _write_json(manifest_path, manifest)


def _ratio(current: float, historical: float) -> float | None:
    if historical > 0.0:
        return current / historical
    return 1.0 if current == 0.0 else None


def summarize(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {(str(row["case_id"]), int(row["run_seed"]), str(row["variant"])): row for row in rows}
    pairs = []
    for case_id in EXPECTED_CASES:
        for seed in EXPECTED_SEEDS:
            current = indexed[(case_id, seed, "current_recovered")]
            historical = indexed[(case_id, seed, "historical_compatible")]
            current_error = float(current["final_error"])
            historical_error = float(historical["final_error"])
            pairs.append({
                "case_id": case_id,
                "run_seed": seed,
                "same_checkpoint": current["checkpoint_hash"] == historical["checkpoint_hash"],
                "current_final_error": current_error,
                "historical_compatible_final_error": historical_error,
                "current_to_historical_ratio": _ratio(current_error, historical_error),
                "current_route": current["route"],
                "historical_compatible_route": historical["route"],
                "exact_final_error": current_error == historical_error,
            })
    case_summaries = []
    for case_id in EXPECTED_CASES:
        active = [row for row in pairs if row["case_id"] == case_id]
        current_values = [float(row["current_final_error"]) for row in active]
        historical_values = [float(row["historical_compatible_final_error"]) for row in active]
        case_summaries.append({
            "case_id": case_id,
            "current_mean": statistics.fmean(current_values),
            "historical_compatible_mean": statistics.fmean(historical_values),
            "current_sample_std": statistics.stdev(current_values),
            "historical_compatible_sample_std": statistics.stdev(historical_values),
            "exact_final_error_pair_count": sum(bool(row["exact_final_error"]) for row in active),
            "current_to_historical_geometric_mean_ratio": math.exp(sum(math.log(float(row["current_to_historical_ratio"])) for row in active) / len(active)),
        })
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "context_count": len(rows),
        "expected_context_count": len(EXPECTED_CASES) * len(EXPECTED_SEEDS) * len(EXPECTED_VARIANTS),
        "pair_count": len(pairs),
        "same_checkpoint_per_pair": all(bool(row["same_checkpoint"]) for row in pairs),
        "exact_terminal_fes": all(int(row["terminal_fes"]) == TOTAL_BUDGET_FES for row in rows),
        "all_final_errors_finite": all(math.isfinite(float(row["final_error"])) for row in rows),
        "all_current_routes_reachable": all("rescue_" in str(row["route"]) and "global_polish_" in str(row["route"]) for row in rows if row["variant"] == "current_recovered"),
        "current_routes_have_no_noop_tail": all("noop_" not in str(row["route"]) for row in rows if row["variant"] == "current_recovered"),
        "all_clip_profiles_explicit": all(bool(row["clip_offspring"]) for row in rows),
        "case_summaries": case_summaries,
        "pairs": pairs,
        "smoke_gate_passed": (
            len(rows) == 50
            and all(bool(row["same_checkpoint"]) for row in pairs)
            and all(int(row["terminal_fes"]) == TOTAL_BUDGET_FES for row in rows)
            and all(math.isfinite(float(row["final_error"])) for row in rows)
            and all("rescue_" in str(row["route"]) and "global_polish_" in str(row["route"]) for row in rows if row["variant"] == "current_recovered")
            and all("noop_" not in str(row["route"]) for row in rows if row["variant"] == "current_recovered")
        ),
        "performance_superiority_claim_authorized": False,
        "reference_thresholds_used_for_decision": False,
    }
    return {**body, "summary_sha256": canonical_sha256(body)}


def run(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False, workers: int | None = None) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    output_root = _resolved(str(protocol["output_root"]))
    manifest = _manifest(resolved, protocol)
    _prepare_output(output_root, manifest, resume=resume)
    contexts = tuple(ArmContext(case_id, seed, variant, output_root, manifest["manifest_sha256"]) for case_id in EXPECTED_CASES for seed in EXPECTED_SEEDS for variant in EXPECTED_VARIANTS)
    rows: list[dict[str, Any]] = []
    pending = []
    for context in contexts:
        if resume and context.receipt_path.is_file():
            rows.append(_validate_receipt(context.receipt_path, context, protocol))
        else:
            pending.append(context)
    worker_count = int(protocol["max_workers"] if workers is None else workers)
    if not 1 <= worker_count <= len(contexts):
        raise ValueError("workers must be in 1..50")
    _write_json(output_root / "progress.json", {"planned": len(contexts), "completed": len(rows), "pending": len(pending), "failed": 0, "max_workers": worker_count})
    failures: dict[str, str] = {}
    if pending:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_run_arm, context, protocol): context for context in pending}
            for future in as_completed(futures):
                context = futures[future]
                try:
                    rows.append(_validate_receipt(context.receipt_path, context, protocol))
                    future.result()
                except BaseException as exc:
                    failures[context.key] = f"{type(exc).__name__}: {exc}"
                _write_json(output_root / "progress.json", {"planned": len(contexts), "completed": len(rows), "pending": len(contexts) - len(rows) - len(failures), "failed": len(failures), "max_workers": worker_count, "failures": failures})
    if failures:
        _write_json(output_root / "failure_summary.json", {"failures": failures})
        raise RuntimeError(f"SMP smoke has {len(failures)} failed arms")
    ordered = [next(row for row in rows if (row["case_id"], int(row["run_seed"]), row["variant"]) == (context.case_id, context.run_seed, context.variant)) for context in contexts]
    summary = summarize(ordered, protocol)
    _write_json(output_root / "summary.json", summary)
    return summary


def verify(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    output_root = _resolved(str(protocol["output_root"]))
    manifest = _manifest(resolved, protocol)
    if _load_json(output_root / "manifest.json") != manifest:
        raise ValueError("SMP smoke manifest drifted")
    contexts = tuple(ArmContext(case_id, seed, variant, output_root, manifest["manifest_sha256"]) for case_id in EXPECTED_CASES for seed in EXPECTED_SEEDS for variant in EXPECTED_VARIANTS)
    rows = [_validate_receipt(context.receipt_path, context, protocol) for context in contexts]
    expected = summarize(rows, protocol)
    stored = _load_json(output_root / "summary.json")
    if stored != expected:
        raise ValueError("SMP smoke summary drifted")
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.protocol, resume=args.resume, workers=args.workers) if args.command == "run" else verify(args.protocol)
    print(json.dumps({"context_count": result["context_count"], "pair_count": result["pair_count"], "smoke_gate_passed": result["smoke_gate_passed"]}, sort_keys=True))
    return 0 if result["smoke_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_PROTOCOL", "EXPECTED_CASES", "EXPECTED_SEEDS", "EXPECTED_VARIANTS", "load_protocol", "run", "summarize", "verify"]
