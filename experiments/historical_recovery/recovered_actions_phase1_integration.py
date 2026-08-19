"""Validate recovered actions from retained Phase-I checkpoints."""

# Thread caps must be set before numerical imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import warnings

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

from threadpoolctl import threadpool_info, threadpool_limits

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.core import run_arac
from arac.runtime.contracts import ACTION_NAMES, ActionContext, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery.replay import _checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("recovered_actions_fixed_action_protocol.json")
DEFAULT_CORE_PROTOCOL = Path(__file__).with_name("recovered_actions_core_e2e_protocol.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-recovered-actions-fixed-action-protocol-v1",
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "run_seed": 117,
        "action_seed": 117,
        "max_workers": 4,
        "native_threads": 1,
        "selector_execution_allowed": False,
        "registry": "RecoveredActionRegistry",
        "allow_out_of_bounds": True,
        "default_registry_replacement_allowed": False,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("fixed-action protocol anchor drifted")
    lanes = protocol.get("lanes", [])
    if len(lanes) != 4 or {lane.get("action") for lane in lanes} != set(ACTION_NAMES):
        raise ValueError("fixed-action lanes drifted")
    for lane in lanes:
        checkpoint_path = REPOSITORY_ROOT / str(lane["checkpoint"])
        if _sha256(checkpoint_path) != lane["checkpoint_file_sha256"]:
            raise ValueError(f"checkpoint file drifted: {lane['action']}")
        wrapper = _load_json(checkpoint_path)
        checkpoint = _checkpoint(wrapper["checkpoint"])
        if checkpoint.checkpoint_hash != lane["checkpoint_hash"]:
            raise ValueError(f"checkpoint payload drifted: {lane['action']}")
    return protocol


def load_core_protocol(path: Path = DEFAULT_CORE_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-recovered-actions-core-e2e-protocol-v1",
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "run_seed": 117,
        "action_seed": 117,
        "max_workers": 4,
        "native_threads": 1,
        "selector_execution_allowed": True,
        "selector_fallback_allowed": False,
        "registry": "RecoveredActionRegistry",
        "allow_out_of_bounds": True,
        "default_registry_replacement_allowed": False,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("core E2E protocol anchor drifted")
    lanes = protocol.get("lanes", [])
    if len(lanes) != 4 or {lane.get("expected_action") for lane in lanes} != set(ACTION_NAMES):
        raise ValueError("core E2E lanes drifted")
    fixed_summary_path = REPOSITORY_ROOT / str(protocol["fixed_action_summary"])
    if _sha256(fixed_summary_path) != protocol["fixed_action_summary_sha256"]:
        raise ValueError("fixed-action summary file drifted")
    fixed_summary = _load_json(fixed_summary_path)
    if (
        fixed_summary.get("fixed_action_gate_passed") is not True
        or fixed_summary.get("selector_execution_authorized") is not True
    ):
        raise ValueError("fixed-action gate does not authorize core E2E")
    return protocol


def _threadpools() -> list[dict[str, Any]]:
    return [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
        }
        for item in threadpool_info()
    ]


def _run_lane(lane: Mapping[str, Any], output_root_text: str) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        pools = _threadpools()
        if not pools or any(pool["num_threads"] != 1 for pool in pools):
            raise RuntimeError(f"native thread limit is not one: {pools}")
        wrapper = _load_json(REPOSITORY_ROOT / str(lane["checkpoint"]))
        checkpoint = _checkpoint(wrapper["checkpoint"])
        problem = AobBenchmark().load(str(lane["case_id_audit_metadata"]))
        registry = RecoveredActionRegistry()
        ledger = EvaluationLedger.from_checkpoint(
            problem,
            total_budget=checkpoint.total_budget_fes,
            phase1_fes=checkpoint.phase1_fes,
            incumbent=checkpoint.incumbent,
            incumbent_error=checkpoint.incumbent_error,
            allow_out_of_bounds=registry.allow_out_of_bounds,
        )
        context = ActionContext(
            str(lane["action"]),
            checkpoint,
            problem,
            ledger,
            action_seed=117,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = registry.execute(context)
        body = {
            "schema_version": "arac-recovered-fixed-action-receipt-v1",
            "action": lane["action"],
            "case_id_audit_metadata": lane["case_id_audit_metadata"],
            "run_seed": checkpoint.run_seed,
            "action_seed": result.action_seed,
            "checkpoint_hash": checkpoint.checkpoint_hash,
            "phase1_fes": checkpoint.phase1_fes,
            "consumed_fes": result.consumed_fes,
            "terminal_fes": result.terminal_fes,
            "final_error": result.final_error,
            "route": result.route,
            "result_hash": result.result_hash,
            "historical_p90": lane["historical_p90"],
            "historical_level_passed": result.final_error <= float(lane["historical_p90"]),
            "allow_out_of_bounds": ledger.allow_out_of_bounds,
            "runtime_warnings": [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            ],
            "threadpools": pools,
            "native_thread_limit_verified": True,
            "selector_execution_authorized": False,
        }
        receipt = {**body, "receipt_hash": canonical_sha256(body)}
        _write_json(Path(output_root_text) / "receipts" / f"{lane['action']}.json", receipt)
        return receipt


def _summary(receipts: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(
        (
            {
                "action": row["action"],
                "case_id_audit_metadata": row["case_id_audit_metadata"],
                "final_error": row["final_error"],
                "historical_p90": row["historical_p90"],
                "historical_level_passed": row["historical_level_passed"],
                "receipt_hash": row["receipt_hash"],
            }
            for row in receipts
        ),
        key=lambda row: ACTION_NAMES.index(str(row["action"])),
    )
    body = {
        "schema_version": "arac-recovered-fixed-action-summary-v1",
        "lane_count": len(rows),
        "historical_level_pass_count": sum(row["historical_level_passed"] for row in rows),
        "fixed_action_gate_passed": len(rows) == 4 and all(
            row["historical_level_passed"] for row in rows
        ),
        "selector_execution_authorized": len(rows) == 4 and all(
            row["historical_level_passed"] for row in rows
        ),
        "rows": rows,
    }
    return {**body, "summary_hash": canonical_sha256(body)}


def run_actions(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    if output_root.exists():
        raise ValueError(f"fixed-action output already exists: {output_root}")
    manifest_body = {
        "schema_version": "arac-recovered-fixed-action-manifest-v1",
        "protocol_sha256": _sha256(protocol_path),
        "source_sha256": {
            relative: _sha256(REPOSITORY_ROOT / relative)
            for relative in (
                "src/arac/actions/_execution.py",
                "src/arac/actions/recovered.py",
                "src/arac/actions/recovered_registry.py",
                "src/arac/actions/ctp.py",
                "src/arac/actions/gcb.py",
                "src/arac/runtime/contracts.py",
                "src/arac/runtime/ledger.py",
            )
        },
        "selector_execution_allowed": False,
    }
    _write_json(
        output_root / "manifest.json",
        {**manifest_body, "manifest_hash": canonical_sha256(manifest_body)},
    )
    receipts = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_lane, lane, str(output_root)): lane["action"]
            for lane in protocol["lanes"]
        }
        for future in as_completed(futures):
            receipts.append(future.result())
    summary = _summary(receipts)
    _write_json(output_root / "summary.json", summary)
    return summary


def check_actions(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(path)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    receipts = []
    for lane in protocol["lanes"]:
        receipt = _load_json(output_root / "receipts" / f"{lane['action']}.json")
        claimed = receipt.pop("receipt_hash")
        if claimed != canonical_sha256(receipt):
            raise ValueError(f"receipt hash drifted: {lane['action']}")
        receipt["receipt_hash"] = claimed
        if (
            receipt["checkpoint_hash"] != lane["checkpoint_hash"]
            or receipt["consumed_fes"] != 2_820_000
            or receipt["terminal_fes"] != 3_000_000
            or receipt["allow_out_of_bounds"] is not True
            or receipt["runtime_warnings"] != []
            or receipt["native_thread_limit_verified"] is not True
        ):
            raise ValueError(f"fixed-action contract failed: {lane['action']}")
        receipts.append(receipt)
    expected = _summary(receipts)
    if _load_json(output_root / "summary.json") != expected:
        raise ValueError("fixed-action summary drifted")
    return expected


def _run_core_lane(lane: Mapping[str, Any], output_root_text: str) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        pools = _threadpools()
        if not pools or any(pool["num_threads"] != 1 for pool in pools):
            raise RuntimeError(f"native thread limit is not one: {pools}")
        problem = AobBenchmark().load(str(lane["case_id_audit_metadata"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_arac(
                problem,
                total_budget_fes=3_000_000,
                run_seed=117,
                action_seed=117,
                registry=RecoveredActionRegistry(),
            )
        checkpoint = result.phase1.checkpoint
        decision = result.core.decision
        action = result.core.action_result
        expected_action = str(lane["expected_action"])
        checkpoint_reproduced = checkpoint.checkpoint_hash == lane["expected_checkpoint_hash"]
        selection_match = decision.action_name == expected_action
        historical_level_passed = action.final_error <= float(lane["historical_p90"])
        body = {
            "schema_version": "arac-recovered-core-e2e-receipt-v1",
            "case_id_audit_metadata": lane["case_id_audit_metadata"],
            "run_seed": checkpoint.run_seed,
            "action_seed": action.action_seed,
            "phase1_fes": checkpoint.phase1_fes,
            "checkpoint_hash": checkpoint.checkpoint_hash,
            "expected_checkpoint_hash": lane["expected_checkpoint_hash"],
            "checkpoint_reproduced": checkpoint_reproduced,
            "action_checkpoint_hash": action.checkpoint_hash,
            "selected_action": decision.action_name,
            "expected_action": expected_action,
            "selection_match": selection_match,
            "selection_reason": decision.reason,
            "selection_scores": list(decision.scores),
            "structural_inference_complete": decision.structural_inference_complete,
            "relation_count": decision.relation_count,
            "largest_component_fraction": decision.largest_component_fraction,
            "consumed_fes": action.consumed_fes,
            "terminal_fes": action.terminal_fes,
            "final_error": action.final_error,
            "historical_p90": lane["historical_p90"],
            "historical_level_passed": historical_level_passed,
            "e2e_lane_passed": checkpoint_reproduced
            and selection_match
            and historical_level_passed,
            "route": action.route,
            "result_hash": action.result_hash,
            "allow_out_of_bounds": True,
            "runtime_warnings": [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            ],
            "threadpools": pools,
            "native_thread_limit_verified": True,
            "selector_fallback_used": False,
        }
        receipt = {**body, "receipt_hash": canonical_sha256(body)}
        _write_json(
            Path(output_root_text)
            / "receipts"
            / f"{lane['case_id_audit_metadata']}.json",
            receipt,
        )
        return receipt


def _core_summary(receipts: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(
        (
            {
                "case_id_audit_metadata": row["case_id_audit_metadata"],
                "expected_action": row["expected_action"],
                "selected_action": row["selected_action"],
                "selection_match": row["selection_match"],
                "checkpoint_reproduced": row["checkpoint_reproduced"],
                "final_error": row["final_error"],
                "historical_p90": row["historical_p90"],
                "historical_level_passed": row["historical_level_passed"],
                "e2e_lane_passed": row["e2e_lane_passed"],
                "receipt_hash": row["receipt_hash"],
            }
            for row in receipts
        ),
        key=lambda row: str(row["case_id_audit_metadata"]),
    )
    complete = len(rows) == 4
    body = {
        "schema_version": "arac-recovered-core-e2e-summary-v1",
        "lane_count": len(rows),
        "checkpoint_reproduction_count": sum(row["checkpoint_reproduced"] for row in rows),
        "selection_match_count": sum(row["selection_match"] for row in rows),
        "historical_level_pass_count": sum(row["historical_level_passed"] for row in rows),
        "e2e_lane_pass_count": sum(row["e2e_lane_passed"] for row in rows),
        "execution_integrity_passed": complete,
        "core_gate_passed": complete and all(row["e2e_lane_passed"] for row in rows),
        "selector_fallback_used": False,
        "rows": rows,
    }
    return {**body, "summary_hash": canonical_sha256(body)}


def run_core(path: Path = DEFAULT_CORE_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = load_core_protocol(protocol_path)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    if output_root.exists():
        raise ValueError(f"core E2E output already exists: {output_root}")
    source_paths = (
        "experiments/historical_recovery/recovered_actions_phase1_integration.py",
        "src/arac/core.py",
        "src/arac/evidence/phase1.py",
        "src/arac/evidence/structural.py",
        "src/arac/evidence/mechanism_features.py",
        "src/arac/actions/_execution.py",
        "src/arac/actions/recovered.py",
        "src/arac/actions/recovered_registry.py",
        "src/arac/actions/ctp.py",
        "src/arac/actions/gcb.py",
        "src/arac/runtime/contracts.py",
        "src/arac/runtime/ledger.py",
    )
    manifest_body = {
        "schema_version": "arac-recovered-core-e2e-manifest-v1",
        "protocol_sha256": _sha256(protocol_path),
        "fixed_action_summary_sha256": protocol["fixed_action_summary_sha256"],
        "source_sha256": {
            relative: _sha256(REPOSITORY_ROOT / relative) for relative in source_paths
        },
        "selector_execution_allowed": True,
        "selector_fallback_allowed": False,
    }
    _write_json(
        output_root / "manifest.json",
        {**manifest_body, "manifest_hash": canonical_sha256(manifest_body)},
    )
    receipts = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_core_lane, lane, str(output_root)): lane["case_id_audit_metadata"]
            for lane in protocol["lanes"]
        }
        for future in as_completed(futures):
            receipts.append(future.result())
    summary = _core_summary(receipts)
    _write_json(output_root / "summary.json", summary)
    return summary


def check_core(path: Path = DEFAULT_CORE_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = load_core_protocol(protocol_path)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = _load_json(output_root / "manifest.json")
    manifest_hash = manifest.pop("manifest_hash")
    if manifest_hash != canonical_sha256(manifest):
        raise ValueError("core E2E manifest hash drifted")
    if manifest["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("core E2E protocol hash drifted")
    for relative, expected_hash in manifest["source_sha256"].items():
        if _sha256(REPOSITORY_ROOT / relative) != expected_hash:
            raise ValueError(f"core E2E source drifted: {relative}")

    receipts = []
    for lane in protocol["lanes"]:
        case_id = str(lane["case_id_audit_metadata"])
        receipt = _load_json(output_root / "receipts" / f"{case_id}.json")
        claimed = receipt.pop("receipt_hash")
        if claimed != canonical_sha256(receipt):
            raise ValueError(f"core E2E receipt hash drifted: {case_id}")
        receipt["receipt_hash"] = claimed
        if (
            receipt["expected_action"] != lane["expected_action"]
            or receipt["expected_checkpoint_hash"] != lane["expected_checkpoint_hash"]
            or receipt["historical_p90"] != lane["historical_p90"]
            or receipt["checkpoint_reproduced"]
            != (receipt["checkpoint_hash"] == lane["expected_checkpoint_hash"])
            or receipt["action_checkpoint_hash"] != receipt["checkpoint_hash"]
            or receipt["selection_match"]
            != (receipt["selected_action"] == lane["expected_action"])
            or receipt["historical_level_passed"]
            != (receipt["final_error"] <= lane["historical_p90"])
            or receipt["e2e_lane_passed"]
            != (
                receipt["checkpoint_reproduced"]
                and receipt["selection_match"]
                and receipt["historical_level_passed"]
            )
            or receipt["run_seed"] != 117
            or receipt["action_seed"] != 117
            or receipt["phase1_fes"] != 180_000
            or receipt["consumed_fes"] != 2_820_000
            or receipt["terminal_fes"] != 3_000_000
            or receipt["allow_out_of_bounds"] is not True
            or receipt["runtime_warnings"] != []
            or receipt["native_thread_limit_verified"] is not True
            or receipt["selector_fallback_used"] is not False
        ):
            raise ValueError(f"core E2E execution contract failed: {case_id}")
        receipts.append(receipt)
    expected = _core_summary(receipts)
    if _load_json(output_root / "summary.json") != expected:
        raise ValueError("core E2E summary drifted")
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("run-actions", "check-actions", "run-core", "check-core"),
    )
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run-actions":
        result = run_actions(DEFAULT_PROTOCOL if args.protocol is None else args.protocol)
    elif args.command == "check-actions":
        result = check_actions(DEFAULT_PROTOCOL if args.protocol is None else args.protocol)
    elif args.command == "run-core":
        result = run_core(DEFAULT_CORE_PROTOCOL if args.protocol is None else args.protocol)
    else:
        result = check_core(DEFAULT_CORE_PROTOCOL if args.protocol is None else args.protocol)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
