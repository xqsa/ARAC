"""Pre-registered AOB pilot with restored Phase-I and matched-budget control."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import traceback
from typing import Any

from arac.actions.registry import ActionRegistry
from arac.analysis.mechanism_policy import (
    run_mechanism_baseline,
    select_mechanism_action,
)
from arac.benchmarks.aob import AobBenchmark
from arac.evidence.phase1 import phase1_budget, run_phase1
from arac.runtime.contracts import ACTION_NAMES, ActionContext, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.probe_policy import run_probe_commit_policy
from experiments.phase2_v2_pilot import (
    _counted_problem,
    _file_sha256,
    _tree_sha256,
    _write_json_atomic,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "phase2_v2_aob_restored_pilot_config.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "phase2_v2_aob_restored_pilot_v2"
CONFIG_SCHEMA = "arac-phase2-v2-aob-restored-pilot-config-v2"
MANIFEST_SCHEMA = "arac-phase2-v2-aob-restored-pilot-manifest-v2"
RECEIPT_SCHEMA = "arac-phase2-v2-aob-restored-pilot-receipt-v2"
SUMMARY_SCHEMA = "arac-phase2-v2-aob-restored-pilot-summary-v2"
METHODS = (
    "probe_commit_v2",
    "mechanism_score_full_v1",
    "mechanism_score_matched_v1",
)
FORBIDDEN_SEEDS = {117, 129, 141, 142, 20260753}
SOURCE_PATHS = (
    "experiments/phase2_v2_aob_restored_pilot.py",
    "experiments/phase2_v2_pilot.py",
    "experiments/phase2_v2_validation.py",
    "src/arac/actions/phase2_v2.py",
    "src/arac/actions/registry.py",
    "src/arac/analysis/mechanism_policy.py",
    "src/arac/benchmarks/aob.py",
    "src/arac/evidence/phase1.py",
    "src/arac/runtime/branches.py",
    "src/arac/runtime/optimizers.py",
    "src/arac/runtime/probe_policy.py",
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    values = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "global_max_fes",
        "phase1_fes",
        "branch_probe_fes",
        "decision_horizon_fes",
        "exploration_floor_fes",
        "min_relative_margin",
        "min_leader_stability",
        "max_workers",
        "methods",
        "aob_cases",
        "run_seeds",
    }
    if set(values) != required or values["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("restored pilot config schema or keys drifted")
    if tuple(values["methods"]) != METHODS:
        raise ValueError("restored pilot methods drifted")
    for name in (
        "global_max_fes",
        "phase1_fes",
        "branch_probe_fes",
        "decision_horizon_fes",
        "exploration_floor_fes",
        "max_workers",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"restored pilot {name} must be positive integer")
    if phase1_budget(values["global_max_fes"]) != values["phase1_fes"]:
        raise ValueError("restored pilot Phase-I budget drifted")
    if not 0 < values["decision_horizon_fes"] < values["branch_probe_fes"]:
        raise ValueError("restored pilot decision horizon is outside probe")
    if 4 * values["branch_probe_fes"] > (
        values["global_max_fes"] - values["phase1_fes"]
    ):
        raise ValueError("restored pilot branch probes exceed Phase-II budget")
    cases = values["aob_cases"]
    if (
        not isinstance(cases, list)
        or not cases
        or len(cases) != len(set(cases))
        or any(
            not isinstance(case, str)
            or len(case) < 2
            or case[0] not in "AERS"
            or not case[1:].isdigit()
            or not 1 <= int(case[1:]) <= 6
            for case in cases
        )
    ):
        raise ValueError("restored pilot AOB cases are invalid")
    seeds = values["run_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0 for seed in seeds)
        or set(seeds) & FORBIDDEN_SEEDS
    ):
        raise ValueError("restored pilot seeds are invalid or forbidden")
    return values


def _contexts(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"suite": "aob", "case": case, "run_seed": seed}
        for case in config["aob_cases"]
        for seed in config["run_seeds"]
    )


def _context_id(context: dict[str, Any]) -> str:
    return f"aob_{context['case']}_s{context['run_seed']}"


def matched_total_fes(config: dict[str, Any]) -> int:
    return int(config["global_max_fes"]) - 4 * int(config["branch_probe_fes"])


def _phase1_payload(checkpoint) -> dict[str, Any]:
    features = dict(zip(checkpoint.feature_names, checkpoint.feature_values, strict=True))
    return {
        "checkpoint_sha256": checkpoint.checkpoint_hash,
        "protocol": checkpoint.protocol,
        "phase1_fes": checkpoint.phase1_fes,
        "block_count": len(checkpoint.blocks),
        "relation_count": checkpoint.overlap_relation_count,
        "structural_inference_complete": float(features["structural_inference_complete"]),
    }


def _common_payload(method: str, checkpoint) -> dict[str, Any]:
    return {"method": method, "phase1": _phase1_payload(checkpoint)}


def _run_full_mechanism(carrier: ActionContext, method: str) -> tuple[dict[str, Any], int]:
    result = run_mechanism_baseline(carrier)
    payload = {
        **_common_payload(method, carrier.checkpoint),
        "selected_action": result.decision.action_name,
        "archive_source_action": result.decision.action_name,
        "selection_reason": result.decision.reason,
        "final_error": result.action_result.final_error,
        "incumbent": list(result.action_result.incumbent),
        "global_total_fes": result.action_result.terminal_fes,
        "objective_fes": result.action_result.terminal_fes,
        "selected_ledger_fes": carrier.ledger.count,
        "selected_action_fes": carrier.ledger.count - carrier.checkpoint.phase1_fes,
        "reserved_probe_tax_fes": 0,
        "terminal_complete": True,
        "route": result.action_result.route,
        "optimizer_package": result.action_result.optimizer_package,
        "optimizer_version": result.action_result.optimizer_version,
        "action_result_sha256": result.action_result.result_hash,
        "state_sha256": None,
        "mechanism_scores": dict(result.decision.scores),
        "largest_component_fraction": result.decision.largest_component_fraction,
        "numerical_repair_count": result.numerical_repair_count,
    }
    return payload, carrier.ledger.count


def _run_matched_mechanism(carrier: ActionContext, config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    decision = select_mechanism_action(carrier.checkpoint)
    action_context = ActionContext(
        decision.action_name,
        carrier.checkpoint,
        carrier.problem,
        carrier.ledger,
        carrier.action_seed,
    )
    state = ActionRegistry().initialize(action_context)
    target = matched_total_fes(config)
    continuation = target - carrier.ledger.count
    if continuation <= 0:
        raise ValueError("matched continuation budget is not positive")
    state.step(continuation)
    snapshot = state.snapshot()
    payload = {
        **_common_payload("mechanism_score_matched_v1", carrier.checkpoint),
        "selected_action": decision.action_name,
        "archive_source_action": decision.action_name,
        "selection_reason": decision.reason,
        "final_error": carrier.ledger.best_error,
        "incumbent": list(carrier.ledger.best_x),
        "global_total_fes": int(config["global_max_fes"]),
        "objective_fes": carrier.ledger.count,
        "selected_ledger_fes": carrier.ledger.count,
        "selected_action_fes": state.consumed_fes,
        "reserved_probe_tax_fes": int(config["global_max_fes"]) - target,
        "terminal_complete": False,
        "route": state.route,
        "optimizer_package": state.optimizer_package,
        "optimizer_version": state.optimizer_version,
        "action_result_sha256": None,
        "state_sha256": snapshot.snapshot_hash,
        "mechanism_scores": dict(decision.scores),
        "largest_component_fraction": decision.largest_component_fraction,
        "numerical_repair_count": state.numerical_repair_count,
    }
    return payload, carrier.ledger.count


def _run_one(context: dict[str, Any], method: str, config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    raw_problem = AobBenchmark().load(context["case"])
    problem, counter = _counted_problem(raw_problem)
    total = int(config["global_max_fes"])
    ledger = EvaluationLedger(problem, total)
    phase1 = run_phase1(problem, ledger, run_seed=int(context["run_seed"]))
    carrier = ActionContext(
        "aor",
        phase1.checkpoint,
        problem,
        ledger,
        action_seed=int(context["run_seed"]),
    )
    if method == "probe_commit_v2":
        result = run_probe_commit_policy(
            carrier,
            global_total_fes=total,
            branch_probe_fes=int(config["branch_probe_fes"]),
            decision_horizon_fes=int(config["decision_horizon_fes"]),
            exploration_floor_fes=int(config["exploration_floor_fes"]),
            min_relative_margin=float(config["min_relative_margin"]),
            min_leader_stability=float(config["min_leader_stability"]),
        )
        payload = {
            **_common_payload(method, phase1.checkpoint),
            "selected_action": result.selected_action,
            "archive_source_action": result.archive_source_action,
            "selection_reason": result.commit_reason,
            "final_error": result.final_error,
            "incumbent": list(result.incumbent),
            "global_total_fes": result.global_total_fes,
            "objective_fes": result.aggregate_fes,
            "selected_ledger_fes": result.selected_ledger_fes,
            "selected_action_fes": result.selected_action_fes,
            "reserved_probe_tax_fes": 4 * result.branch_probe_fes,
            "terminal_complete": True,
            "continuation_fes": result.continuation_fes,
            "branch_probe_fes": result.branch_probe_fes,
            "route": result.route,
            "optimizer_package": result.optimizer_package,
            "optimizer_version": result.optimizer_version,
            "action_result_sha256": None,
            "state_sha256": result.selected_state_hash,
            "numerical_repair_count": result.numerical_repair_count,
            "probe_final_errors": dict(result.probe_final_errors),
            "decision": {
                "action_name": result.decision.action_name,
                "reason": result.decision.reason,
                "relative_margin": result.decision.relative_margin,
                "leader_stability": result.decision.leader_stability,
            },
        }
        expected = result.aggregate_fes
    elif method == "mechanism_score_full_v1":
        payload, expected = _run_full_mechanism(carrier, method)
    elif method == "mechanism_score_matched_v1":
        payload, expected = _run_matched_mechanism(carrier, config)
    else:
        raise ValueError(f"unknown restored pilot method: {method}")
    if counter.count != expected:
        raise RuntimeError(f"{method} counter {counter.count} != expected {expected}")
    return payload, counter.count


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_print(payload: object) -> None:
    try:
        print(payload, flush=True)
    except (BrokenPipeError, OSError):
        pass


def _limit_native_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _manifest(
    config_file: Path,
    config: dict[str, Any],
    *,
    max_workers: int,
) -> dict[str, Any]:
    source_hashes = {
        relative: _file_sha256(REPOSITORY_ROOT / relative)
        for relative in SOURCE_PATHS
    }
    vendor_count, vendor_hash = _tree_sha256(REPOSITORY_ROOT / "vendor" / "aob")
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "config_sha256": _file_sha256(config_file),
        "context_count": len(_contexts(config)),
        "method_count": len(METHODS),
        "max_workers": max_workers,
        "parallel_unit": "context_triplet",
        "source_hashes": source_hashes,
        "aob_vendor_tree": {"file_count": vendor_count, "tree_sha256": vendor_hash},
        "matched_total_fes": matched_total_fes(config),
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"restored pilot {label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"restored pilot {label} must contain an object")
    return payload


def _prepare_campaign_root(
    root: Path,
    config_file: Path,
    manifest: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    receipt_root = root / "receipts"
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"restored pilot resume root is missing: {root}")
        stored = _read_json(root / "manifest.json", "manifest")
        body = dict(stored)
        claimed_hash = body.pop("manifest_sha256", None)
        if claimed_hash != canonical_sha256(body):
            raise ValueError("restored pilot manifest hash drifted")
        if stored != manifest:
            raise ValueError("restored pilot resume manifest drifted")
        frozen_config = root / "config.json"
        if _file_sha256(frozen_config) != manifest["config_sha256"]:
            raise ValueError("restored pilot frozen config drifted")
        if not receipt_root.is_dir():
            raise FileNotFoundError("restored pilot receipt directory is missing")
        return receipt_root
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"restored pilot output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir()
    shutil.copyfile(config_file, root / "config.json")
    _write_json_atomic(root / "manifest.json", manifest)
    return receipt_root


def _run_index(context_index: int, method: str) -> int:
    return context_index * len(METHODS) + METHODS.index(method)


def _receipt_path(
    receipt_root: Path,
    context_index: int,
    context: dict[str, Any],
    method: str,
) -> Path:
    run_index = _run_index(context_index, method)
    return receipt_root / f"{run_index:03d}_{_context_id(context)}_{method}.json"


def _validate_receipt(
    path: Path,
    *,
    context_index: int,
    context: dict[str, Any],
    method: str,
    config: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    receipt = _read_json(path, "receipt")
    body = dict(receipt)
    claimed_hash = body.pop("receipt_sha256", None)
    if claimed_hash != canonical_sha256(body):
        raise ValueError(f"restored pilot receipt hash drifted: {path.name}")
    expected_common = {
        "schema_version": RECEIPT_SCHEMA,
        "run_index": _run_index(context_index, method),
        "context_id": _context_id(context),
        "benchmark": context,
        "method": method,
        "manifest_sha256": manifest_sha256,
    }
    for name, expected in expected_common.items():
        if receipt.get(name) != expected:
            raise ValueError(f"restored pilot receipt {name} drifted: {path.name}")
    if receipt.get("status") != "completed":
        raise RuntimeError(f"restored pilot failed receipt cannot resume: {path.name}")
    expected_keys = {
        "schema_version",
        "status",
        "run_index",
        "context_id",
        "benchmark",
        "method",
        "manifest_sha256",
        "checkpoint_sha256",
        "objective_fes",
        "result",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError(f"restored pilot receipt keys drifted: {path.name}")
    result = receipt.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("phase1"), dict):
        raise ValueError(f"restored pilot receipt result is invalid: {path.name}")
    phase1 = result["phase1"]
    checkpoint_hash = receipt.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_hash, str)
        or not checkpoint_hash
        or phase1.get("checkpoint_sha256") != checkpoint_hash
        or phase1.get("phase1_fes") != config["phase1_fes"]
        or phase1.get("structural_inference_complete") != 1.0
    ):
        raise ValueError(f"restored pilot checkpoint receipt drifted: {path.name}")
    expected_objective_fes = (
        matched_total_fes(config)
        if method == "mechanism_score_matched_v1"
        else config["global_max_fes"]
    )
    expected_probe_tax = (
        0 if method == "mechanism_score_full_v1" else 4 * config["branch_probe_fes"]
    )
    if (
        receipt.get("objective_fes") != expected_objective_fes
        or result.get("objective_fes") != expected_objective_fes
        or result.get("global_total_fes") != config["global_max_fes"]
        or result.get("reserved_probe_tax_fes") != expected_probe_tax
        or result.get("terminal_complete")
        is (method == "mechanism_score_matched_v1")
        or result.get("selected_action") not in ACTION_NAMES
    ):
        raise ValueError(f"restored pilot budget receipt drifted: {path.name}")
    return receipt


def _write_method_receipt(
    receipt_root: Path,
    *,
    context_index: int,
    context: dict[str, Any],
    method: str,
    config: dict[str, Any],
    manifest_sha256: str,
) -> None:
    run_index = _run_index(context_index, method)
    path = _receipt_path(receipt_root, context_index, context, method)
    try:
        result, objective_fes = _run_one(context, method, config)
        payload = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "completed",
            "run_index": run_index,
            "context_id": _context_id(context),
            "benchmark": context,
            "method": method,
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": result["phase1"]["checkpoint_sha256"],
            "objective_fes": objective_fes,
            "result": result,
        }
    except Exception as exc:
        payload = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "failed",
            "run_index": run_index,
            "context_id": _context_id(context),
            "benchmark": context,
            "method": method,
            "manifest_sha256": manifest_sha256,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    _write_json_atomic(path, receipt)


def _run_context_triplet(
    context_index: int,
    context: dict[str, Any],
    config: dict[str, Any],
    manifest_sha256: str,
    receipt_root: Path,
    resume: bool,
) -> str:
    for method in METHODS:
        path = _receipt_path(receipt_root, context_index, context, method)
        if resume and path.is_file():
            continue
        if path.exists():
            raise ValueError(f"restored pilot fresh receipt already exists: {path}")
        _write_method_receipt(
            receipt_root,
            context_index=context_index,
            context=context,
            method=method,
            config=config,
            manifest_sha256=manifest_sha256,
        )
    return _context_id(context)


def _load_existing_receipts(
    receipt_root: Path,
    contexts: tuple[dict[str, Any], ...],
    config: dict[str, Any],
    manifest_sha256: str,
) -> dict[int, dict[str, Any]]:
    expected_names = {
        _receipt_path(receipt_root, context_index, context, method).name
        for context_index, context in enumerate(contexts)
        for method in METHODS
    }
    unexpected = {path.name for path in receipt_root.glob("*.json")} - expected_names
    if unexpected:
        raise ValueError(f"restored pilot unexpected receipts: {sorted(unexpected)}")
    receipts = {}
    for context_index, context in enumerate(contexts):
        for method in METHODS:
            path = _receipt_path(receipt_root, context_index, context, method)
            if path.is_file():
                receipt = _validate_receipt(
                    path,
                    context_index=context_index,
                    context=context,
                    method=method,
                    config=config,
                    manifest_sha256=manifest_sha256,
                )
                receipts[receipt["run_index"]] = receipt
    return receipts


def _run_parallel(
    receipt_root: Path,
    contexts: tuple[dict[str, Any], ...],
    config: dict[str, Any],
    manifest_sha256: str,
    *,
    max_workers: int,
    resume: bool,
) -> list[dict[str, Any]]:
    receipts = _load_existing_receipts(
        receipt_root,
        contexts,
        config,
        manifest_sha256,
    )
    pending = [
        (context_index, context)
        for context_index, context in enumerate(contexts)
        if any(
            _run_index(context_index, method) not in receipts
            for method in METHODS
        )
    ]
    failures: dict[str, str] = {}
    progress_path = receipt_root.parent / "parallel_progress.json"

    def write_progress() -> None:
        completed_contexts = sum(
            all(_run_index(index, method) in receipts for method in METHODS)
            for index in range(len(contexts))
        )
        _write_json_atomic(
            progress_path,
            {
                "planned_contexts": len(contexts),
                "planned_receipts": len(contexts) * len(METHODS),
                "completed_contexts": completed_contexts,
                "completed_receipts": len(receipts),
                "failed_contexts": len(failures),
                "pending_contexts": len(contexts) - completed_contexts - len(failures),
                "max_workers": min(max_workers, len(contexts)),
                "failures": failures,
                "updated_at_utc": _utc_now(),
            },
        )

    write_progress()
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(max_workers, len(pending)),
            max_tasks_per_child=1,
        ) as pool:
            futures = {
                pool.submit(
                    _run_context_triplet,
                    context_index,
                    context,
                    config,
                    manifest_sha256,
                    receipt_root,
                    resume,
                ): (context_index, context)
                for context_index, context in pending
            }
            for future in as_completed(futures):
                context_index, context = futures[future]
                context_id = _context_id(context)
                try:
                    future.result()
                    for method in METHODS:
                        receipt = _validate_receipt(
                            _receipt_path(receipt_root, context_index, context, method),
                            context_index=context_index,
                            context=context,
                            method=method,
                            config=config,
                            manifest_sha256=manifest_sha256,
                        )
                        receipts[receipt["run_index"]] = receipt
                except Exception as exc:
                    failures[context_id] = f"{type(exc).__name__}: {exc}"
                write_progress()
                status = "complete" if context_id not in failures else "failed"
                _safe_print(
                    f"[{len(receipts):02d}/{len(contexts) * len(METHODS)}] "
                    f"{context_id} {status}"
                )
    if failures:
        raise RuntimeError(f"restored pilot has {len(failures)} failed contexts")
    expected_receipts = len(contexts) * len(METHODS)
    if len(receipts) != expected_receipts:
        raise RuntimeError(
            f"restored pilot receipt count {len(receipts)} != {expected_receipts}"
        )
    return [receipts[index] for index in range(expected_receipts)]


def _comparison_rows(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for receipt in receipts:
        grouped.setdefault(receipt["context_id"], {})[receipt["method"]] = receipt
    rows = []
    for context_id, methods in sorted(grouped.items()):
        if set(methods) != set(METHODS):
            raise ValueError(f"restored pilot method triplet incomplete: {context_id}")
        probe = methods["probe_commit_v2"]
        full = methods["mechanism_score_full_v1"]
        matched = methods["mechanism_score_matched_v1"]
        hashes = {
            receipt["checkpoint_sha256"]
            for receipt in (probe, full, matched)
        }
        if len(hashes) != 1:
            raise ValueError(f"restored pilot checkpoint drifted: {context_id}")
        probe_error = float(probe["result"]["final_error"])
        full_error = float(full["result"]["final_error"])
        matched_error = float(matched["result"]["final_error"])
        rows.append(
            {
                "context_id": context_id,
                "probe_action": probe["result"]["selected_action"],
                "full_action": full["result"]["selected_action"],
                "matched_action": matched["result"]["selected_action"],
                "phase1_fes": probe["result"]["phase1"]["phase1_fes"],
                "structural_inference_complete": probe["result"]["phase1"][
                    "structural_inference_complete"
                ],
                "probe_error": probe_error,
                "full_error": full_error,
                "matched_error": matched_error,
                "probe_vs_matched_shifted_log10": math.log10(
                    (probe_error + 1.0) / (matched_error + 1.0)
                ),
                "matched_vs_full_shifted_log10": math.log10(
                    (matched_error + 1.0) / (full_error + 1.0)
                ),
                "probe_vs_full_shifted_log10": math.log10(
                    (probe_error + 1.0) / (full_error + 1.0)
                ),
                "same_probe_matched_action": probe["result"]["selected_action"]
                == matched["result"]["selected_action"],
                "same_full_matched_action": full["result"]["selected_action"]
                == matched["result"]["selected_action"],
                "probe_tax_fes": probe["result"]["reserved_probe_tax_fes"],
            }
        )
    return rows


def run_pilot(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
    max_workers: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(output_root).resolve()
    config = load_config(config_file)
    workers = int(config["max_workers"] if max_workers is None else max_workers)
    if workers <= 0:
        raise ValueError("restored pilot max_workers must be positive")
    _limit_native_threads()
    contexts = _contexts(config)
    manifest = _manifest(config_file, config, max_workers=workers)
    receipt_root = _prepare_campaign_root(
        root,
        config_file,
        manifest,
        resume=resume,
    )
    completed = _run_parallel(
        receipt_root,
        contexts,
        config,
        manifest["manifest_sha256"],
        max_workers=workers,
        resume=resume,
    )

    comparison_rows: list[dict[str, Any]] = []
    comparison_error = None
    try:
        comparison_rows = _comparison_rows(completed)
        with (root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)
    except ValueError as exc:
        comparison_error = str(exc)

    expected_fes = {
        "probe_commit_v2": config["global_max_fes"],
        "mechanism_score_full_v1": config["global_max_fes"],
        "mechanism_score_matched_v1": matched_total_fes(config),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "context_count": len(contexts),
        "expected_method_runs": len(contexts) * len(METHODS),
        "max_workers": workers,
        "completed": len(completed),
        "failed": 0,
        "all_checkpoint_triplets_equal": comparison_error is None,
        "all_single_commit": all(
            receipt["method"] != "probe_commit_v2"
            or receipt["result"]["selected_action"] in ACTION_NAMES
            for receipt in completed
        ),
        "phase1_fes": sorted(
            {
                receipt["result"]["phase1"]["phase1_fes"]
                for receipt in completed
            }
        ),
        "structural_inference_complete_counts": dict(
            Counter(
                str(receipt["result"]["phase1"]["structural_inference_complete"])
                for receipt in completed
            )
        ),
        "method_action_counts": {
            method: {
                action: sum(
                    receipt["method"] == method
                    and receipt["result"]["selected_action"] == action
                    for receipt in completed
                )
                for action in ACTION_NAMES
            }
            for method in METHODS
        },
        "expected_objective_fes": expected_fes,
        "actual_objective_fes_by_method": {
            method: sorted(
                {
                    receipt["objective_fes"]
                    for receipt in completed
                    if receipt["method"] == method
                }
            )
            for method in METHODS
        },
        "comparison_error": comparison_error,
        "comparison": comparison_rows,
        "quality_claim": "limited_restored_phase1_pilot_not_generalization",
    }
    _write_json_atomic(root / "summary.json", summary)
    if comparison_error is not None:
        raise RuntimeError(f"restored AOB pilot comparison failed: {comparison_error}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--workers", type=int)
    run_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    _safe_print(
        json.dumps(
            run_pilot(
                config_path=args.config,
                output_root=args.output_root,
                max_workers=args.workers,
                resume=args.resume,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
