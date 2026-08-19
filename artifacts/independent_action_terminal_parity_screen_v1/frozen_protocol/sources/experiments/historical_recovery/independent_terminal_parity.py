"""Run the fixed-total-budget independent action terminal-parity screen."""

# Thread caps must be applied before importing numerical modules.
# ruff: noqa: E402

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import warnings

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

from threadpoolctl import threadpool_info, threadpool_limits

from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ACTION_NAMES, canonical_sha256
from experiments.historical_recovery.independent_semantic_parity_pilot import (
    REPOSITORY_ROOT,
    _candidate_mechanism_passed,
    _context,
    _file_sha256,
    _hcc_runtime_imports,
    _load_json,
    _write_json_atomic,
    execute_historical_semantic_port,
)
from experiments.historical_recovery.replay import _checkpoint


DEFAULT_PROTOCOL = Path(__file__).with_name("independent_terminal_parity_protocol.json")
MANIFEST_SCHEMA = "arac-independent-terminal-parity-manifest-v1"
RECEIPT_SCHEMA = "arac-independent-terminal-parity-receipt-v1"
SUMMARY_SCHEMA = "arac-independent-terminal-parity-summary-v1"
SOURCE_PATHS = (
    "experiments/historical_recovery/independent_terminal_parity.py",
    "experiments/historical_recovery/independent_terminal_parity_protocol.json",
    "experiments/historical_recovery/independent_semantic_parity_pilot.py",
)
PRODUCTION_SOURCES = {
    "action_execution": "src/arac/actions/_execution.py",
    "action_registry": "src/arac/actions/registry.py",
    "aor": "src/arac/actions/aor.py",
    "benchmark": "src/arac/benchmarks/aob.py",
    "contracts": "src/arac/runtime/contracts.py",
    "ctp": "src/arac/actions/ctp.py",
    "gcb": "src/arac/actions/gcb.py",
    "ledger": "src/arac/runtime/ledger.py",
    "optimizers": "src/arac/runtime/optimizers.py",
    "smp": "src/arac/actions/smp.py",
}


def nearest_rank_p90(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("historical errors must be finite non-negative values")
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def _verify_hash(payload: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    body = dict(payload)
    claimed = body.pop(key, None)
    if claimed != canonical_sha256(body):
        raise ValueError(f"{label} hash drifted")
    return payload


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    if protocol.get("schema_version") != "arac-independent-terminal-parity-protocol-v1":
        raise ValueError("terminal parity protocol schema drifted")
    if (
        protocol.get("total_budget_fes") != 3_000_000
        or protocol.get("phase1_fes") != 180_000
        or protocol.get("checkpoint_seed") != 117
        or protocol.get("historical_gate") != "same_case_nearest_rank_p90"
        or protocol.get("max_workers") != 2
        or protocol.get("native_threads") != 1
        or protocol.get("selector_execution_allowed") is not False
    ):
        raise ValueError("terminal parity protocol anchor drifted")
    lanes = protocol.get("lanes", [])
    if len(lanes) != 4 or {lane.get("action") for lane in lanes} != set(ACTION_NAMES):
        raise ValueError("terminal parity lanes drifted")
    if protocol.get("required_candidate_actions") != ["aor", "smp"]:
        raise ValueError("terminal parity candidate set drifted")
    for lane in lanes:
        for key in ("checkpoint", "current_receipt", "reference_contract"):
            if not (REPOSITORY_ROOT / str(lane[key])).is_file():
                raise ValueError(f"terminal parity input is missing: {lane['action']}/{key}")
        if not list(REPOSITORY_ROOT.glob(str(lane["historical_glob"]))):
            raise ValueError(f"historical distribution is missing: {lane['action']}")
    return protocol


def _historical_stats(lane: Mapping[str, Any]) -> dict[str, Any]:
    paths = sorted(REPOSITORY_ROOT.glob(str(lane["historical_glob"])))
    errors = [float(_load_json(path)["final_error"]) for path in paths]
    return {
        "count": len(errors),
        "minimum": min(errors),
        "median": float(sorted(errors)[len(errors) // 2]),
        "p90": nearest_rank_p90(errors),
        "maximum": max(errors),
        "source_hashes": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): _file_sha256(path) for path in paths
        },
    }


def _baseline_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    campaign_path = REPOSITORY_ROOT / str(protocol["current_campaign_manifest"])
    campaign = _verify_hash(_load_json(campaign_path), "manifest_sha256", "current campaign")
    for key, relative in PRODUCTION_SOURCES.items():
        if campaign["source_hashes"].get(key) != _file_sha256(REPOSITORY_ROOT / relative):
            raise ValueError(f"current campaign source drifted: {key}")
    rows = []
    required = set(protocol["required_candidate_actions"])
    for lane in protocol["lanes"]:
        action = str(lane["action"])
        receipt = _verify_hash(
            _load_json(REPOSITORY_ROOT / str(lane["current_receipt"])),
            "receipt_hash",
            f"current receipt {action}",
        )
        if (
            receipt.get("action_name") != action
            or receipt.get("case_id") != lane["case_id"]
            or receipt.get("run_seed") != protocol["checkpoint_seed"]
            or receipt.get("terminal_fes") != protocol["total_budget_fes"]
            or receipt.get("runtime_warnings") != []
        ):
            raise ValueError(f"current receipt contract drifted: {action}")
        stats = _historical_stats(lane)
        current_error = float(receipt["final_error"])
        current_passed = current_error <= stats["p90"]
        if (action in required) == current_passed:
            raise ValueError(f"pre-registered candidate decision drifted: {action}")
        rows.append(
            {
                "action": action,
                "case_id": lane["case_id"],
                "current_error": current_error,
                "current_receipt": lane["current_receipt"],
                "current_receipt_sha256": _file_sha256(
                    REPOSITORY_ROOT / str(lane["current_receipt"])
                ),
                "current_historical_level_passed": current_passed,
                "candidate_required": action in required,
                "historical": stats,
            }
        )
    return rows


def preflight(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if _hcc_runtime_imports():
        raise ValueError("production HCC runtime imports remain")
    output_root = REPOSITORY_ROOT / str(protocol["new_output_root"])
    if output_root.exists():
        raise ValueError(f"terminal parity output already exists: {output_root}")
    rows = _baseline_rows(protocol)
    return {
        "historical_gate": protocol["historical_gate"],
        "current_passed_actions": [
            row["action"] for row in rows if row["current_historical_level_passed"]
        ],
        "required_candidate_actions": protocol["required_candidate_actions"],
        "output_root": str(output_root.resolve()),
        "source_inputs_valid": True,
    }


def _threadpools() -> list[dict[str, Any]]:
    return [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
        }
        for item in threadpool_info()
    ]


def _run_candidate(
    lane: Mapping[str, Any],
    baseline: Mapping[str, Any],
    output_root_text: str,
) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        pools = _threadpools()
        if not pools or any(item["num_threads"] != 1 for item in pools):
            raise RuntimeError(f"native thread limit is not one: {pools}")
        checkpoint_wrapper = _load_json(REPOSITORY_ROOT / str(lane["checkpoint"]))
        checkpoint = _checkpoint(checkpoint_wrapper["checkpoint"])
        if checkpoint.total_budget_fes != 3_000_000 or checkpoint.phase1_fes != 180_000:
            raise ValueError("terminal checkpoint budget drifted")
        problem = AobBenchmark().load(str(lane["case_id"]))
        context = _context(
            str(lane["action"]),
            checkpoint,
            problem,
            action_seed=checkpoint.run_seed,
        )
        contract = _load_json(REPOSITORY_ROOT / str(lane["reference_contract"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result, events = execute_historical_semantic_port(context, contract)
        runtime_warnings = [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ]
        action = str(lane["action"])
        candidate_passed = float(result.final_error) <= float(baseline["historical"]["p90"])
        payload = {
            "schema_version": RECEIPT_SCHEMA,
            "action": action,
            "case_id": lane["case_id"],
            "run_seed": checkpoint.run_seed,
            "checkpoint_hash": checkpoint.checkpoint_hash,
            "total_budget_fes": checkpoint.total_budget_fes,
            "phase1_fes": checkpoint.phase1_fes,
            "consumed_fes": result.consumed_fes,
            "terminal_fes": result.terminal_fes,
            "final_error": result.final_error,
            "route": result.route,
            "result_hash": result.result_hash,
            "events": events,
            "mechanism_trace_passed": _candidate_mechanism_passed(action, events),
            "runtime_warnings": runtime_warnings,
            "threadpools": pools,
            "native_thread_limit_verified": all(item["num_threads"] == 1 for item in pools),
            "historical_p90": baseline["historical"]["p90"],
            "historical_level_passed": candidate_passed,
            "selector_evaluation_authorized": False,
        }
        payload["receipt_hash"] = canonical_sha256(payload)
        _write_json_atomic(Path(output_root_text) / "receipts" / f"{action}.json", payload)
        return payload


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_sha256": _file_sha256(protocol_path),
        "source_hashes": {
            path: _file_sha256(REPOSITORY_ROOT / path) for path in SOURCE_PATHS
        },
        "production_source_hashes": {
            key: _file_sha256(REPOSITORY_ROOT / path)
            for key, path in PRODUCTION_SOURCES.items()
        },
        "production_hcc_runtime_imports": _hcc_runtime_imports(),
        "selector_execution_allowed": False,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _summary(
    baselines: list[dict[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    lanes = []
    for row in baselines:
        candidate = candidates.get(row["action"])
        selected_arm = "historical_semantic_port" if candidate else "current_production_reused"
        selected_error = (
            float(candidate["final_error"]) if candidate else float(row["current_error"])
        )
        lanes.append(
            {
                "action": row["action"],
                "case_id": row["case_id"],
                "selected_arm": selected_arm,
                "selected_error": selected_error,
                "historical_count": row["historical"]["count"],
                "historical_median": row["historical"]["median"],
                "historical_p90": row["historical"]["p90"],
                "historical_level_passed": selected_error <= row["historical"]["p90"],
                "current_error": row["current_error"],
                "candidate_error": None if candidate is None else candidate["final_error"],
            }
        )
    pass_count = sum(row["historical_level_passed"] for row in lanes)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "lane_count": len(lanes),
        "historical_level_pass_count": pass_count,
        "terminal_screen_passed": pass_count == len(lanes) == 4,
        "selector_evaluation_authorized": False,
        "lanes": lanes,
    }


def _prepare_output(protocol_path: Path, protocol: Mapping[str, Any]) -> Path:
    output_root = (REPOSITORY_ROOT / str(protocol["new_output_root"])).resolve()
    if output_root.exists():
        raise ValueError(f"terminal parity output already exists: {output_root}")
    sources = output_root / "frozen_protocol" / "sources"
    sources.mkdir(parents=True)
    shutil.copy2(protocol_path, output_root / "frozen_protocol" / "protocol.json")
    for relative in SOURCE_PATHS:
        destination = sources / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, destination)
    _write_json_atomic(output_root / "frozen_protocol" / "manifest.json", _manifest(protocol_path, protocol))
    return output_root


def run_screen(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    preflight(protocol_file)
    baselines = _baseline_rows(protocol)
    by_action = {row["action"]: row for row in baselines}
    lanes = {
        str(lane["action"]): lane
        for lane in protocol["lanes"]
        if lane["action"] in protocol["required_candidate_actions"]
    }
    output_root = _prepare_output(protocol_file, protocol)
    candidates: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=int(protocol["max_workers"])) as executor:
        futures = {
            executor.submit(_run_candidate, lane, by_action[action], str(output_root)): action
            for action, lane in lanes.items()
        }
        for future in as_completed(futures):
            result = future.result()
            candidates[str(result["action"])] = result
    summary = _summary(baselines, candidates)
    _write_json_atomic(output_root / "summary.json", summary)
    return summary


def check_screen(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    output_root = (REPOSITORY_ROOT / str(protocol["new_output_root"])).resolve()
    manifest = _verify_hash(
        _load_json(output_root / "frozen_protocol" / "manifest.json"),
        "manifest_sha256",
        "terminal manifest",
    )
    if manifest != _manifest(protocol_file, protocol):
        raise ValueError("terminal manifest inputs drifted")
    baselines = _baseline_rows(protocol)
    candidates = {}
    for action in protocol["required_candidate_actions"]:
        receipt = _verify_hash(
            _load_json(output_root / "receipts" / f"{action}.json"),
            "receipt_hash",
            f"terminal receipt {action}",
        )
        if (
            receipt.get("action") != action
            or receipt.get("consumed_fes") != 2_820_000
            or receipt.get("terminal_fes") != 3_000_000
            or receipt.get("runtime_warnings") != []
            or receipt.get("native_thread_limit_verified") is not True
            or receipt.get("mechanism_trace_passed") is not True
            or receipt.get("selector_evaluation_authorized") is not False
        ):
            raise ValueError(f"terminal execution gate failed: {action}")
        candidates[action] = receipt
    expected = _summary(baselines, candidates)
    if _load_json(output_root / "summary.json") != expected:
        raise ValueError("terminal summary drifted")
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "check"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.protocol)
    elif args.command == "run":
        result = run_screen(args.protocol)
    else:
        result = check_screen(args.protocol)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
