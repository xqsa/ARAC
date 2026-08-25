"""Verify zero-relation recovered SMP preserves the prior E1 results."""

# Thread caps must be applied before NumPy, PyPop7, or ARAC imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
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
DEFAULT_PROTOCOL = Path(__file__).with_name("recovery_smp_zero_relation_preservation_protocol_v1.json")
SCHEMA = "arac-recovery-smp-zero-relation-preservation-receipt-v1"
SUMMARY_SCHEMA = "arac-recovery-smp-zero-relation-preservation-summary-v1"
MANIFEST_SCHEMA = "arac-recovery-smp-zero-relation-preservation-manifest-v1"
EXPECTED_SEEDS = (117, 123, 129, 135, 141)
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
PHASE2_FES = 2_820_000
ZERO_PROFILE = "zero_relation_recovered_smp_v1_clip_offspring_false"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolved(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-recovery-smp-zero-relation-preservation-protocol-v1",
        "status": "frozen_e1_preservation_gate",
        "case_id": "E1",
        "seeds": list(EXPECTED_SEEDS),
        "action_name": "smp",
        "total_budget_fes": TOTAL_BUDGET_FES,
        "phase1_fes": PHASE1_FES,
        "phase2_fes": PHASE2_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "native_threads": 1,
        "max_workers": 5,
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"E1 preservation protocol drifted: {key}")
    for key in ("checkpoint_root", "current_e2e_receipt_root", "baseline_receipt_root"):
        if not _resolved(str(protocol[key])).exists():
            raise FileNotFoundError(f"E1 preservation source is missing: {key}")
    return protocol


def _baseline_path(protocol: Mapping[str, Any], seed: int) -> Path:
    return _resolved(str(protocol["baseline_receipt_root"])) / f"seed_{seed}" / "smp.json"


def _baseline_receipt(protocol: Mapping[str, Any], seed: int) -> dict[str, Any]:
    payload = _load_json(_baseline_path(protocol, seed))
    claimed = payload.pop("receipt_hash", None)
    if claimed != canonical_sha256(payload):
        raise ValueError(f"E1 seed {seed} baseline receipt hash drifted")
    payload["receipt_hash"] = claimed
    if (
        payload.get("case_id") != "E1"
        or payload.get("run_seed") != seed
        or payload.get("action_name") != "smp"
        or payload.get("terminal_fes") != TOTAL_BUDGET_FES
    ):
        raise ValueError(f"E1 seed {seed} baseline receipt identity drifted")
    return payload


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    baseline_paths = {_seed: _baseline_path(protocol, _seed) for _seed in EXPECTED_SEEDS}
    sources = {
        "protocol": protocol_path,
        "campaign": Path(__file__).resolve(),
        "recovered": REPOSITORY_ROOT / "src/arac/actions/recovered.py",
        "recovered_registry": REPOSITORY_ROOT / "src/arac/actions/recovered_registry.py",
        "execution": REPOSITORY_ROOT / "src/arac/actions/_execution.py",
        "ledger": REPOSITORY_ROOT / "src/arac/runtime/ledger.py",
        "contracts": REPOSITORY_ROOT / "src/arac/runtime/contracts.py",
        "fixed_campaign": Path(fixed.__file__).resolve(),
    }
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_sha256": _sha256(protocol_path),
        "case_id": "E1",
        "seeds": list(EXPECTED_SEEDS),
        "action_name": "smp",
        "source_sha256": {name: _sha256(path) for name, path in sorted(sources.items())},
        "baseline_receipt_sha256": {str(seed): _sha256(path) for seed, path in sorted(baseline_paths.items())},
        "checkpoint_root": protocol["checkpoint_root"],
        "current_e2e_receipt_root": protocol["current_e2e_receipt_root"],
        "patch_enabled": False,
        "soft_routing_enabled": False,
        "selector_enabled": False,
        "baseline_semantics": "frozen_previous_screen_receipt_final_error_only",
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


@dataclass(frozen=True)
class ArmContext:
    run_seed: int
    output_root: Path
    manifest_sha256: str

    @property
    def key(self) -> str:
        return f"E1:seed-{self.run_seed}:smp"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / f"seed_{self.run_seed}" / "smp.json"

    @property
    def failure_path(self) -> Path:
        return self.output_root / "failures" / f"seed_{self.run_seed}.json"


def _checkpoint(context: ArmContext, protocol: Mapping[str, Any]):
    fixed_context = fixed.ArmContext(
        case_id="E1",
        run_seed=context.run_seed,
        action_name="smp",
        checkpoint_root=_resolved(str(protocol["checkpoint_root"])),
        current_receipt_root=_resolved(str(protocol["current_e2e_receipt_root"])),
        output_root=context.output_root,
        manifest_sha256=context.manifest_sha256,
    )
    return fixed._load_verified_checkpoint(fixed_context)


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
            baseline = _baseline_receipt(protocol, context.run_seed)
            problem = AobBenchmark().load("E1")
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
                result = RecoveredActionRegistry().execute(
                    ActionContext("smp", checkpoint, problem, ledger, action_seed=context.run_seed)
                )
            baseline_error = float(baseline["final_error"])
            exact_baseline_match = result.final_error == baseline_error
            if (
                result.action_name != "smp"
                or result.action_seed != context.run_seed
                or result.checkpoint_hash != checkpoint.checkpoint_hash
                or result.consumed_fes != PHASE2_FES
                or result.terminal_fes != TOTAL_BUDGET_FES
                or ledger.count != TOTAL_BUDGET_FES
                or not math.isfinite(result.final_error)
                or not exact_baseline_match
            ):
                raise RuntimeError(f"{context.key} E1 preservation contract failed")
            body = {
                "schema_version": SCHEMA,
                "manifest_sha256": context.manifest_sha256,
                "case_id": "E1",
                "run_seed": context.run_seed,
                "action_name": "smp",
                "action_seed": result.action_seed,
                "phase1_fes": PHASE1_FES,
                "phase2_fes": PHASE2_FES,
                "terminal_fes": TOTAL_BUDGET_FES,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "baseline_receipt_sha256": baseline["receipt_hash"],
                "baseline_final_error": baseline_error,
                "final_error": result.final_error,
                "exact_baseline_match": exact_baseline_match,
                "route": result.route,
                "lifecycle_profile": ZERO_PROFILE,
                "action_result": result.payload(),
                "action_result_hash": result.result_hash,
                "runtime_warnings": _warning_rows(caught),
                "threadpools": pools,
                "native_thread_limit_verified": True,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            receipt = {**body, "receipt_sha256": canonical_sha256(body)}
            _write_json(context.receipt_path, receipt)
            return receipt
    except BaseException as exc:
        _write_json(context.failure_path, {
            "schema_version": "arac-recovery-smp-zero-relation-preservation-failure-v1",
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
    baseline = _baseline_receipt(protocol, context.run_seed)
    for key, value in {
        "schema_version": SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": "E1",
        "run_seed": context.run_seed,
        "action_name": "smp",
        "action_seed": context.run_seed,
        "phase1_fes": PHASE1_FES,
        "phase2_fes": PHASE2_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "baseline_receipt_sha256": baseline["receipt_hash"],
        "baseline_final_error": baseline["final_error"],
        "exact_baseline_match": True,
        "lifecycle_profile": ZERO_PROFILE,
        "native_thread_limit_verified": True,
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
            raise FileExistsError(f"E1 preservation output already exists: {output_root}")
        if _load_json(manifest_path) != manifest:
            raise ValueError("E1 preservation manifest does not match frozen protocol")
        return
    output_root.mkdir(parents=True)
    _write_json(manifest_path, manifest)


def summarize(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "case_id": "E1",
        "context_count": len(rows),
        "expected_context_count": len(EXPECTED_SEEDS),
        "all_terminal_fes_exact": all(int(row["terminal_fes"]) == TOTAL_BUDGET_FES for row in rows),
        "all_checkpoint_bindings_exact": all(row["checkpoint_hash"] == row["action_result"]["checkpoint_hash"] for row in rows),
        "all_receipt_hashes_valid": True,
        "all_exact_baseline_matches": all(bool(row["exact_baseline_match"]) for row in rows),
        "final_errors": [{"run_seed": int(row["run_seed"]), "baseline": float(row["baseline_final_error"]), "current": float(row["final_error"])} for row in rows],
        "preservation_gate_passed": len(rows) == len(EXPECTED_SEEDS) and all(bool(row["exact_baseline_match"]) for row in rows) and all(int(row["terminal_fes"]) == TOTAL_BUDGET_FES for row in rows),
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
    contexts = tuple(ArmContext(seed, output_root, manifest["manifest_sha256"]) for seed in EXPECTED_SEEDS)
    rows: list[dict[str, Any]] = []
    pending = []
    for context in contexts:
        if resume and context.receipt_path.is_file():
            rows.append(_validate_receipt(context.receipt_path, context, protocol))
        else:
            pending.append(context)
    worker_count = int(protocol["max_workers"] if workers is None else workers)
    if not 1 <= worker_count <= len(contexts):
        raise ValueError("workers must be in 1..5")
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
        raise RuntimeError(f"E1 preservation has {len(failures)} failed arms")
    ordered = [next(row for row in rows if int(row["run_seed"]) == context.run_seed) for context in contexts]
    summary = summarize(ordered, protocol)
    _write_json(output_root / "summary.json", summary)
    return summary


def verify(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    output_root = _resolved(str(protocol["output_root"]))
    manifest = _manifest(resolved, protocol)
    if _load_json(output_root / "manifest.json") != manifest:
        raise ValueError("E1 preservation manifest drifted")
    contexts = tuple(ArmContext(seed, output_root, manifest["manifest_sha256"]) for seed in EXPECTED_SEEDS)
    rows = [_validate_receipt(context.receipt_path, context, protocol) for context in contexts]
    expected = summarize(rows, protocol)
    if _load_json(output_root / "summary.json") != expected:
        raise ValueError("E1 preservation summary drifted")
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.protocol, resume=args.resume, workers=args.workers) if args.command == "run" else verify(args.protocol)
    print(json.dumps({"context_count": result["context_count"], "preservation_gate_passed": result["preservation_gate_passed"]}, sort_keys=True))
    return 0 if result["preservation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_PROTOCOL", "EXPECTED_SEEDS", "load_protocol", "run", "summarize", "verify"]
