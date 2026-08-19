"""Run a short, read-only semantic decomposition of the current SMP port."""

# Thread caps must be applied before importing NumPy, pypop7, or ARAC modules.
# ruff: noqa: E402

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import warnings

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import arac.actions._execution as execution
from arac.actions._execution import (
    STATE_RESCUE_FRACTION,
    STATE_RESCUE_MIN_FES,
    run_stateful_block_visits_with_sessions,
    run_full_space,
    run_zero_relation_hybrid_rescue,
    terminal_result,
)
from arac.actions.registry import ActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, ActionResult, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery import independent_smp_schedule_recovery as schedule_recovery
from experiments.historical_recovery.replay import _checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("smp_semantic_parity_diagnostic_protocol.json")
SCHEMA = "arac-smp-semantic-parity-diagnostic-receipt-v1"
SUMMARY_SCHEMA = "arac-smp-semantic-parity-diagnostic-summary-v1"
MANIFEST_SCHEMA = "arac-smp-semantic-parity-diagnostic-manifest-v1"
VARIANTS = (
    "schedule_only",
    "stateful_only",
    "stateful_prefix_no_rescue",
    "stateful_plus_rescue",
    "current_complete_smp",
    "stateful_native_restart_on",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolved(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = _load_json(protocol_path)
    required = {
        "schema_version",
        "status",
        "purpose",
        "production_hcc_dependency_allowed",
        "selector_execution_allowed",
        "case_id",
        "run_seed",
        "phase1_fes",
        "screen_step_fes",
        "native_threads",
        "max_workers",
        "checkpoint",
        "historical_action",
        "output_root",
        "variants",
        "restart_note",
        "acceptance_gates",
    }
    if set(protocol) != required:
        raise ValueError("SMP diagnostic protocol keys drifted")
    if protocol["schema_version"] != "arac-smp-semantic-parity-diagnostic-protocol-v1":
        raise ValueError("SMP diagnostic protocol schema drifted")
    if protocol["status"] != "frozen_short_prefix":
        raise ValueError("SMP diagnostic protocol must remain frozen")
    if protocol["production_hcc_dependency_allowed"] or protocol["selector_execution_allowed"]:
        raise ValueError("SMP diagnostic cannot enable HCC or selector")
    if protocol["case_id"] != "E1" or protocol["run_seed"] != 117:
        raise ValueError("SMP diagnostic anchor drifted")
    if protocol["phase1_fes"] != 180_000 or protocol["screen_step_fes"] != 120_000:
        raise ValueError("SMP diagnostic FE boundary drifted")
    if protocol["native_threads"] != 1 or protocol["max_workers"] != 3:
        raise ValueError("SMP diagnostic thread boundary drifted")
    if tuple(protocol["variants"]) != VARIANTS:
        raise ValueError("SMP diagnostic variants drifted")
    checkpoint_path = _resolved(str(protocol["checkpoint"]))
    historical_path = _resolved(str(protocol["historical_action"]))
    if not checkpoint_path.is_file() or not historical_path.is_file():
        raise ValueError("SMP diagnostic input artifact is missing")
    checkpoint_wrapper = _load_json(checkpoint_path)
    checkpoint = _checkpoint(checkpoint_wrapper["checkpoint"])
    if checkpoint_wrapper.get("checkpoint_hash") != checkpoint.checkpoint_hash:
        raise ValueError("SMP diagnostic checkpoint hash is inconsistent")
    if (
        checkpoint.run_seed != protocol["run_seed"]
        or checkpoint.phase1_fes != protocol["phase1_fes"]
        or checkpoint.total_budget_fes != 3_000_000
    ):
        raise ValueError("SMP diagnostic checkpoint boundary drifted")
    historical = _load_json(historical_path)
    action = historical.get("action")
    if not isinstance(action, dict) or action.get("name") != "smp":
        raise ValueError("EXP-052 SMP action artifact drifted")
    if action.get("stale_window") != 3 or len(action.get("target_groups", [])) != 20:
        raise ValueError("EXP-052 SMP action contract drifted")
    return protocol


def _context(checkpoint, problem, screen_fes: int) -> ActionContext:
    screen_checkpoint = replace(
        checkpoint,
        total_budget_fes=checkpoint.phase1_fes + int(screen_fes),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=screen_checkpoint.total_budget_fes,
        phase1_fes=screen_checkpoint.phase1_fes,
        incumbent=screen_checkpoint.incumbent,
        incumbent_error=screen_checkpoint.incumbent_error,
    )
    return ActionContext("smp", screen_checkpoint, problem, ledger, action_seed=checkpoint.run_seed)


def _consume_noop(context: ActionContext, requested_fes: int) -> int:
    """Account exact FE without adding another optimization mechanism."""

    requested = min(int(requested_fes), context.ledger.remaining)
    if requested <= 0:
        return 0
    incumbent = context.ledger.best_x
    consumed = 0
    while consumed < requested:
        batch = min(1024, requested - consumed)
        context.ledger.evaluate(np.repeat(incumbent[None, :], batch, axis=0))
        consumed += batch
    return consumed


def _result_payload(result: ActionResult, events: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "consumed_fes": result.consumed_fes,
        "terminal_fes": result.terminal_fes,
        "final_error": result.final_error,
        "result_hash": result.result_hash,
        "route": result.route,
        "optimizer_package": result.optimizer_package,
        "optimizer_version": result.optimizer_version,
        "events": dict(events),
    }


def _stateful_arm(
    context: ActionContext,
    *,
    requested_fes: int,
    rescue: bool,
    rescue_budget: int = 0,
) -> tuple[ActionResult, dict[str, Any]]:
    stateful_budget = int(requested_fes) - int(rescue_budget)
    stateful_fes, visit_count, restart_count, sessions = run_stateful_block_visits_with_sessions(
        context,
        requested_fes=stateful_budget,
    )
    rescue_fes = 0
    probe_fes = 0
    coverage_fes = 0
    cold_rescue_visits = 0
    persistent_rescue_visits = 0
    if rescue and context.checkpoint.relations == ():
        (
            rescue_fes,
            probe_fes,
            coverage_fes,
            cold_rescue_visits,
            persistent_rescue_visits,
        ) = run_zero_relation_hybrid_rescue(
            context,
            requested_fes=min(int(rescue_budget), context.ledger.remaining),
            sessions=sessions,
        )
    terminal_polish_fes = 0
    if rescue and context.ledger.remaining:
        terminal_polish_fes = run_full_space(
            context,
            algorithm="sepcmaes",
            namespace="smp-terminal",
        ).consumed_fes
    noop_fes = _consume_noop(context, context.ledger.remaining)
    result = terminal_result(
        context,
        route=(
            f"diagnostic_stateful_{stateful_fes}_visits_{visit_count}_"
            f"restarts_{restart_count}_rescue_{rescue_fes}_noop_{noop_fes}"
        ),
    )
    events = {
        "stateful_fes": stateful_fes,
        "visit_count": visit_count,
        "stagnation_restart_count": restart_count,
        "rescue_fes": rescue_fes,
        "probe_fes": probe_fes,
        "coverage_fes": coverage_fes,
        "cold_rescue_visits": cold_rescue_visits,
        "persistent_rescue_visits": persistent_rescue_visits,
        "terminal_polish_fes": terminal_polish_fes,
        "noop_fes": noop_fes,
        "requested_fes": requested_fes,
        "rescue_budget": rescue_budget,
        "manual_session_loop": True,
    }
    return result, events


@contextmanager
def _native_restart_option(enabled: bool):
    """Set the pypop7 option only inside this diagnostic process."""

    original = execution.CMAES

    def factory(problem: Mapping[str, Any], options: Mapping[str, Any]):
        updated = dict(options)
        updated["is_restart"] = bool(enabled)
        return original(problem, updated)

    execution.CMAES = factory
    try:
        yield
    finally:
        execution.CMAES = original


def _run_variant(variant: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported diagnostic variant: {variant}")
    checkpoint_wrapper = _load_json(_resolved(str(protocol["checkpoint"])))
    source_checkpoint = _checkpoint(checkpoint_wrapper["checkpoint"])
    problem = AobBenchmark().load(str(protocol["case_id"]))
    screen_fes = int(protocol["screen_step_fes"])
    context = _context(source_checkpoint, problem, screen_fes)
    pools = [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
            "user_api": item.get("user_api"),
        }
        for item in threadpool_info()
    ]
    if not pools or any(item["num_threads"] != 1 for item in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    recovery_protocol = schedule_recovery.load_protocol(schedule_recovery.DEFAULT_PROTOCOL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if variant == "schedule_only":
            events = schedule_recovery.execute_schedule(context, screen_fes, protocol=recovery_protocol)
            _consume_noop(context, context.ledger.remaining)
            result = terminal_result(context, route="diagnostic_schedule_only")
        elif variant == "stateful_only":
            result, events = _stateful_arm(context, requested_fes=screen_fes, rescue=False)
        elif variant == "stateful_prefix_no_rescue":
            rescue_budget = int(screen_fes * STATE_RESCUE_FRACTION) if screen_fes >= STATE_RESCUE_MIN_FES else 0
            result, events = _stateful_arm(
                context,
                requested_fes=screen_fes,
                rescue=False,
                rescue_budget=rescue_budget,
            )
            events["stateful_budget_match"] = screen_fes - rescue_budget
            events["matched_no_rescue"] = True
        elif variant == "stateful_plus_rescue":
            rescue_budget = int(screen_fes * STATE_RESCUE_FRACTION) if screen_fes >= STATE_RESCUE_MIN_FES else 0
            result, events = _stateful_arm(
                context,
                requested_fes=screen_fes,
                rescue=True,
                rescue_budget=rescue_budget,
            )
        elif variant == "current_complete_smp":
            result = ActionRegistry().execute(context)
            events = {"route": result.route, "production_complete": True}
        else:
            with _native_restart_option(True):
                result, events = _stateful_arm(context, requested_fes=screen_fes, rescue=False)
            events["native_restart_option"] = True
            events["native_restart_semantics_reached"] = False
    runtime_warnings = [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    ]
    payload = {
        "schema_version": SCHEMA,
        "variant": variant,
        "case_id": protocol["case_id"],
        "run_seed": source_checkpoint.run_seed,
        "source_checkpoint_hash": source_checkpoint.checkpoint_hash,
        "screen_checkpoint_hash": context.checkpoint.checkpoint_hash,
        "source_phase1_fes": source_checkpoint.phase1_fes,
        "screen_step_fes": screen_fes,
        "result": _result_payload(result, events),
        "runtime_warnings": runtime_warnings,
        "native_thread_limit_verified": all(item["num_threads"] == 1 for item in pools),
        "threadpools": pools,
        "production_hcc_runtime_imports": [],
    }
    payload["receipt_hash"] = canonical_sha256(payload)
    return payload


def _manifest(protocol_path: Path, protocol: Mapping[str, Any], receipts: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_paths = [
        "experiments/historical_recovery/smp_semantic_parity_diagnostic.py",
        "experiments/historical_recovery/smp_semantic_parity_diagnostic_protocol.json",
        "src/arac/actions/smp.py",
        "src/arac/actions/_execution.py",
        "src/arac/runtime/ledger.py",
    ]
    source_hashes = {path: _file_sha256(_resolved(path)) for path in source_paths}
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_sha256": _file_sha256(protocol_path),
        "protocol": dict(protocol),
        "source_hashes": source_hashes,
        "checkpoint_sha256": _file_sha256(_resolved(str(protocol["checkpoint"]))),
        "historical_action_sha256": _file_sha256(_resolved(str(protocol["historical_action"]))),
        "receipt_hashes": sorted(str(receipt["receipt_hash"]) for receipt in receipts),
        "production_hcc_runtime_imports": [],
        "selector_execution_allowed": False,
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def _summarize(receipts: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant = {str(receipt["variant"]): receipt for receipt in receipts}
    exact = all(
        receipt["result"]["consumed_fes"] == receipt["screen_step_fes"]
        and receipt["result"]["terminal_fes"] == receipt["source_phase1_fes"] + receipt["screen_step_fes"]
        for receipt in receipts
    )
    warnings_empty = all(not receipt["runtime_warnings"] for receipt in receipts)
    same_source = len({receipt["source_checkpoint_hash"] for receipt in receipts}) == 1
    same_screen = len({receipt["screen_checkpoint_hash"] for receipt in receipts}) == 1
    stateful = by_variant.get("stateful_prefix_no_rescue")
    rescued = by_variant.get("stateful_plus_rescue")
    restart_off = by_variant.get("stateful_only")
    restart_on = by_variant.get("stateful_native_restart_on")
    rescue_delta = None
    if stateful and rescued:
        rescue_delta = float(stateful["result"]["final_error"] - rescued["result"]["final_error"])
    restart_identical = None
    if restart_off and restart_on:
        restart_identical = (
            restart_off["result"]["result_hash"] == restart_on["result"]["result_hash"]
            and restart_off["result"]["final_error"] == restart_on["result"]["final_error"]
        )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "variant_count": len(receipts),
        "all_exact_screen_fes": exact,
        "all_runtime_warnings_empty": warnings_empty,
        "all_same_source_checkpoint": same_source,
        "all_same_screen_checkpoint": same_screen,
        "native_thread_limit_verified": all(receipt["native_thread_limit_verified"] for receipt in receipts),
        "rescue_final_error_delta_no_rescue_minus_rescue": rescue_delta,
        "native_restart_on_off_identical": restart_identical,
        "terminal_parity_evaluated": False,
        "selector_evaluation_authorized": False,
        "production_smp_integration_authorized": False,
        "receipts": [
            {
                "variant": receipt["variant"],
                "final_error": receipt["result"]["final_error"],
                "consumed_fes": receipt["result"]["consumed_fes"],
                "route": receipt["result"]["route"],
                "receipt_hash": receipt["receipt_hash"],
            }
            for receipt in sorted(receipts, key=lambda item: str(item["variant"]))
        ],
    }


def run(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    if output_root.exists():
        raise ValueError(f"diagnostic output root already exists: {output_root}")
    with threadpool_limits(limits=1):
        receipts: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=int(protocol["max_workers"])) as executor:
            futures = {
                executor.submit(_run_variant, variant, protocol): variant
                for variant in protocol["variants"]
            }
            for future in as_completed(futures):
                receipts.append(future.result())
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "receipts").mkdir()
    for receipt in receipts:
        _write_json_atomic(output_root / "receipts" / f"{receipt['variant']}.json", receipt)
    manifest = _manifest(protocol_path, protocol, receipts)
    _write_json_atomic(output_root / "manifest.json", manifest)
    summary_body = _summarize(receipts)
    summary = {**summary_body, "summary_hash": canonical_sha256(summary_body)}
    _write_json_atomic(output_root / "summary.json", summary)
    _write_json_atomic(output_root / "frozen_protocol.json", protocol)
    return summary


def _validate_receipt(path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_json(path)
    body = dict(receipt)
    receipt_hash = body.pop("receipt_hash", None)
    if receipt_hash != canonical_sha256(body):
        raise ValueError(f"receipt hash mismatch: {path}")
    if receipt.get("schema_version") != SCHEMA:
        raise ValueError(f"receipt schema drifted: {path}")
    if receipt.get("source_checkpoint_hash") != _checkpoint(_load_json(_resolved(str(protocol["checkpoint"])))['checkpoint']).checkpoint_hash:
        raise ValueError(f"receipt checkpoint drifted: {path}")
    result = receipt.get("result", {})
    if result.get("consumed_fes") != protocol["screen_step_fes"]:
        raise ValueError(f"receipt FE drifted: {path}")
    if receipt.get("runtime_warnings") != [] or receipt.get("native_thread_limit_verified") is not True:
        raise ValueError(f"receipt runtime gate failed: {path}")
    return receipt


def check(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    if not output_root.is_dir():
        raise ValueError("diagnostic output root is missing")
    receipts = [
        _validate_receipt(output_root / "receipts" / f"{variant}.json", protocol)
        for variant in protocol["variants"]
    ]
    manifest = _load_json(output_root / "manifest.json")
    manifest_hash = manifest.pop("manifest_sha256", None)
    if manifest_hash != canonical_sha256(manifest):
        raise ValueError("diagnostic manifest hash mismatch")
    if manifest.get("protocol_sha256") != _file_sha256(protocol_path):
        raise ValueError("diagnostic protocol hash drifted")
    if manifest.get("source_hashes", {}).get("src/arac/actions/smp.py") != _file_sha256(_resolved("src/arac/actions/smp.py")):
        raise ValueError("production SMP source changed since diagnostic")
    summary = _load_json(output_root / "summary.json")
    summary_hash = summary.pop("summary_hash", None)
    if summary_hash != canonical_sha256(summary):
        raise ValueError("diagnostic summary hash mismatch")
    decision = _summarize(receipts)
    if summary != decision or not summary["all_exact_screen_fes"] or not summary["all_runtime_warnings_empty"]:
        raise ValueError("diagnostic summary gate failed")
    return {
        "integrity_gate_passed": True,
        "all_exact_screen_fes": summary["all_exact_screen_fes"],
        "rescue_final_error_delta_no_rescue_minus_rescue": summary["rescue_final_error_delta_no_rescue_minus_rescue"],
        "native_restart_on_off_identical": summary["native_restart_on_off_identical"],
        "terminal_parity_evaluated": False,
        "selector_evaluation_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "check"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        payload = load_protocol(args.protocol)
        print(json.dumps({"protocol": "valid", "variants": payload["variants"]}, indent=2))
    elif args.command == "run":
        payload = run(args.protocol)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        payload = check(args.protocol)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
