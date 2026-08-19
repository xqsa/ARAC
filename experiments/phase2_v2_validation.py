"""Pre-registered AOB + IOH comparison of probe and mechanism policies."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import traceback
from typing import Any

from arac.analysis.mechanism_policy import run_mechanism_baseline
from arac.benchmarks.aob import AobBenchmark
from arac.benchmarks.ioh_bbob import IohBbobBenchmark
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
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "phase2_v2_validation_config.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "phase2_v2_validation_ioh_v2"
CONFIG_SCHEMA = "arac-phase2-v2-validation-config-v1"
MANIFEST_SCHEMA = "arac-phase2-v2-validation-manifest-v1"
RECEIPT_SCHEMA = "arac-phase2-v2-validation-receipt-v1"
SUMMARY_SCHEMA = "arac-phase2-v2-validation-summary-v1"
METHODS = ("probe_commit_v2", "mechanism_score_v1")
FORBIDDEN_SEEDS = {117, 129, 141, 142, 20260753}
SOURCE_PATHS = (
    "experiments/phase2_v2_pilot.py",
    "experiments/phase2_v2_validation.py",
    "src/arac/actions/phase2_v2.py",
    "src/arac/analysis/delayed_commit.py",
    "src/arac/analysis/mechanism_policy.py",
    "src/arac/benchmarks/aob.py",
    "src/arac/benchmarks/ioh_bbob.py",
    "src/arac/evidence/phase1.py",
    "src/arac/evidence/structural.py",
    "src/arac/runtime/branches.py",
    "src/arac/runtime/optimizers.py",
    "src/arac/runtime/probe_policy.py",
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    values = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "ioh_version",
        "methods",
        "global_max_fes",
        "branch_probe_fes",
        "decision_horizon_fes",
        "exploration_floor_fes",
        "min_relative_margin",
        "min_leader_stability",
        "aob_cases",
        "ioh_function_ids",
        "ioh_instances",
        "ioh_dimension",
        "run_seeds",
    }
    if set(values) != required or values["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("validation config schema or keys drifted")
    if tuple(values["methods"]) != METHODS:
        raise ValueError("validation methods drifted")
    for name in (
        "global_max_fes",
        "branch_probe_fes",
        "decision_horizon_fes",
        "exploration_floor_fes",
        "ioh_dimension",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"validation {name} must be a positive integer")
    if not 0 < values["decision_horizon_fes"] < values["branch_probe_fes"]:
        raise ValueError("validation decision horizon is outside the probe")
    phase2_fes = values["global_max_fes"] - phase1_budget(values["global_max_fes"])
    if 4 * values["branch_probe_fes"] > phase2_fes:
        raise ValueError("validation branch probes exceed the Phase-II budget")
    if importlib.metadata.version("ioh") != values["ioh_version"]:
        raise ValueError("validation IOH version drifted")
    if (
        not isinstance(values["aob_cases"], list)
        or not values["aob_cases"]
        or len(values["aob_cases"]) != len(set(values["aob_cases"]))
    ):
        raise ValueError("validation AOB cases must be unique")
    for name in ("ioh_function_ids", "ioh_instances", "run_seeds"):
        entries = values[name]
        if (
            not isinstance(entries, list)
            or not entries
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in entries)
            or len(entries) != len(set(entries))
        ):
            raise ValueError(f"validation {name} must contain unique positive integers")
    if not set(values["ioh_function_ids"]).issubset(range(1, 25)):
        raise ValueError("validation IOH function IDs must be in 1..24")
    if set(values["run_seeds"]) & FORBIDDEN_SEEDS:
        raise ValueError("validation reuses a forbidden development seed")
    return values


def _contexts(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for case in config["aob_cases"]:
        for seed in config["run_seeds"]:
            rows.append(
                {
                    "suite": "aob",
                    "case": str(case),
                    "run_seed": int(seed),
                }
            )
    for function_id in config["ioh_function_ids"]:
        for instance in config["ioh_instances"]:
            for seed in config["run_seeds"]:
                rows.append(
                    {
                        "suite": "ioh_bbob",
                        "function_id": int(function_id),
                        "instance": int(instance),
                        "dimension": int(config["ioh_dimension"]),
                        "run_seed": int(seed),
                    }
                )
    return tuple(rows)


def _context_id(context: dict[str, Any]) -> str:
    if context["suite"] == "aob":
        return f"aob_{context['case']}_s{context['run_seed']}"
    return (
        f"ioh_f{context['function_id']:02d}_i{context['instance']}_"
        f"d{context['dimension']}_s{context['run_seed']}"
    )


def _load_problem(context: dict[str, Any]):
    if context["suite"] == "aob":
        return AobBenchmark().load(context["case"])
    return IohBbobBenchmark().load(
        context["function_id"],
        instance=context["instance"],
        dimension=context["dimension"],
    )


def _phase1_payload(checkpoint) -> dict[str, Any]:
    features = dict(
        zip(checkpoint.feature_names, checkpoint.feature_values, strict=True)
    )
    return {
        "checkpoint_sha256": checkpoint.checkpoint_hash,
        "protocol": checkpoint.protocol,
        "phase1_fes": checkpoint.phase1_fes,
        "block_count": len(checkpoint.blocks),
        "relation_count": checkpoint.overlap_relation_count,
        "structural_inference_complete": float(
            features["structural_inference_complete"]
        ),
    }


def _run_method(
    context_spec: dict[str, Any],
    method: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    raw_problem = _load_problem(context_spec)
    problem, counter = _counted_problem(raw_problem)
    total_fes = int(config["global_max_fes"])
    ledger = EvaluationLedger(problem, total_fes)
    phase1 = run_phase1(
        problem,
        ledger,
        run_seed=int(context_spec["run_seed"]),
    )
    carrier = ActionContext(
        "aor",
        phase1.checkpoint,
        problem,
        ledger,
        action_seed=int(context_spec["run_seed"]),
    )
    common = {
        "method": method,
        "phase1": _phase1_payload(phase1.checkpoint),
    }
    if method == "probe_commit_v2":
        result = run_probe_commit_policy(
            carrier,
            global_total_fes=total_fes,
            branch_probe_fes=int(config["branch_probe_fes"]),
            decision_horizon_fes=int(config["decision_horizon_fes"]),
            exploration_floor_fes=int(config["exploration_floor_fes"]),
            min_relative_margin=float(config["min_relative_margin"]),
            min_leader_stability=float(config["min_leader_stability"]),
        )
        payload = {
            **common,
            "selected_action": result.selected_action,
            "committed_action_count": 1,
            "archive_source_action": result.archive_source_action,
            "selection_reason": result.commit_reason,
            "final_error": result.final_error,
            "incumbent": list(result.incumbent),
            "global_total_fes": result.global_total_fes,
            "selected_ledger_fes": result.selected_ledger_fes,
            "selected_action_fes": result.selected_action_fes,
            "continuation_fes": result.continuation_fes,
            "branch_probe_fes": result.branch_probe_fes,
            "route": result.route,
            "optimizer_package": result.optimizer_package,
            "optimizer_version": result.optimizer_version,
            "selected_state_sha256": result.selected_state_hash,
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
    elif method == "mechanism_score_v1":
        result = run_mechanism_baseline(carrier)
        payload = {
            **common,
            "selected_action": result.decision.action_name,
            "committed_action_count": 1,
            "archive_source_action": result.decision.action_name,
            "selection_reason": result.decision.reason,
            "final_error": result.action_result.final_error,
            "incumbent": list(result.action_result.incumbent),
            "global_total_fes": result.action_result.terminal_fes,
            "route": result.action_result.route,
            "optimizer_package": result.action_result.optimizer_package,
            "optimizer_version": result.action_result.optimizer_version,
            "action_result_sha256": result.action_result.result_hash,
            "mechanism_scores": dict(result.decision.scores),
            "largest_component_fraction": result.decision.largest_component_fraction,
            "numerical_repair_count": result.numerical_repair_count,
        }
        expected = result.action_result.terminal_fes
    else:
        raise ValueError(f"unknown validation method: {method}")
    if counter.count != expected or expected != total_fes:
        raise RuntimeError(
            f"{method} objective counter recorded {counter.count} FE, expected {total_fes}"
        )
    return payload, counter.count


def _comparison_rows(
    receipts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_context: dict[str, dict[str, dict[str, Any]]] = {}
    for receipt in receipts:
        by_context.setdefault(receipt["context_id"], {})[receipt["method"]] = receipt
    rows: list[dict[str, Any]] = []
    for context_id, methods in sorted(by_context.items()):
        if set(methods) != set(METHODS):
            raise ValueError(f"validation context lacks a method pair: {context_id}")
        probe = methods["probe_commit_v2"]
        mechanism = methods["mechanism_score_v1"]
        if probe["checkpoint_sha256"] != mechanism["checkpoint_sha256"]:
            raise ValueError(f"validation checkpoint pair drifted: {context_id}")
        probe_error = float(probe["result"]["final_error"])
        mechanism_error = float(mechanism["result"]["final_error"])
        scale = max(abs(probe_error), abs(mechanism_error), 1.0)
        if abs(probe_error - mechanism_error) <= 1e-12 * scale:
            winner = "tie"
        elif probe_error < mechanism_error:
            winner = "probe_commit_v2"
        else:
            winner = "mechanism_score_v1"
        rows.append(
            {
                "context_id": context_id,
                "suite": probe["benchmark"]["suite"],
                "probe_error": probe_error,
                "mechanism_error": mechanism_error,
                "shifted_log10_probe_over_mechanism": math.log10(
                    (probe_error + 1.0) / (mechanism_error + 1.0)
                ),
                "winner": winner,
                "probe_action": probe["result"]["selected_action"],
                "mechanism_action": mechanism["result"]["selected_action"],
                "probe_reason": probe["result"]["selection_reason"],
                "mechanism_reason": mechanism["result"]["selection_reason"],
            }
        )

    grouped: dict[str, Any] = {}
    for suite in ("aob", "ioh_bbob", "all"):
        subset = rows if suite == "all" else [row for row in rows if row["suite"] == suite]
        winners = Counter(row["winner"] for row in subset)
        ratios = [
            float(row["shifted_log10_probe_over_mechanism"])
            for row in subset
        ]
        grouped[suite] = {
            "contexts": len(subset),
            "probe_wins": winners["probe_commit_v2"],
            "mechanism_wins": winners["mechanism_score_v1"],
            "ties": winners["tie"],
            "mean_shifted_log10_probe_over_mechanism": (
                statistics.fmean(ratios) if ratios else None
            ),
            "median_shifted_log10_probe_over_mechanism": (
                statistics.median(ratios) if ratios else None
            ),
        }
    return rows, grouped


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_validation(*, config_path: Path, output_root: Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"validation output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    receipt_root = root / "receipts"
    receipt_root.mkdir()
    config = load_config(config_file)
    contexts = _contexts(config)

    source_hashes = {
        relative: _file_sha256(REPOSITORY_ROOT / relative)
        for relative in SOURCE_PATHS
    }
    vendor_count, vendor_hash = _tree_sha256(REPOSITORY_ROOT / "vendor" / "aob")
    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA,
        "config_sha256": _file_sha256(config_file),
        "ioh_version": importlib.metadata.version("ioh"),
        "context_count": len(contexts),
        "method_count": len(METHODS),
        "source_hashes": source_hashes,
        "aob_vendor_tree": {
            "file_count": vendor_count,
            "tree_sha256": vendor_hash,
        },
    }
    manifest = {**manifest_payload, "manifest_sha256": canonical_sha256(manifest_payload)}
    _write_json_atomic(root / "manifest.json", manifest)
    shutil.copyfile(config_file, root / "config.json")

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_index = 0
    for context_spec in contexts:
        context_id = _context_id(context_spec)
        for method in METHODS:
            receipt_path = receipt_root / f"{run_index:03d}_{context_id}_{method}.json"
            try:
                result, objective_fes = _run_method(context_spec, method, config)
                payload = {
                    "schema_version": RECEIPT_SCHEMA,
                    "status": "completed",
                    "run_index": run_index,
                    "context_id": context_id,
                    "benchmark": context_spec,
                    "method": method,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "checkpoint_sha256": result["phase1"]["checkpoint_sha256"],
                    "objective_fes": objective_fes,
                    "result": result,
                }
                receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
                completed.append(receipt)
            except Exception as exc:
                payload = {
                    "schema_version": RECEIPT_SCHEMA,
                    "status": "failed",
                    "run_index": run_index,
                    "context_id": context_id,
                    "benchmark": context_spec,
                    "method": method,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
                failures.append(receipt)
            _write_json_atomic(receipt_path, receipt)
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "failed": len(failures),
                        "total": len(contexts) * len(METHODS),
                        "context_id": context_id,
                        "method": method,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            run_index += 1

    comparison_rows: list[dict[str, Any]] = []
    comparison: dict[str, Any] = {}
    comparison_error: str | None = None
    if not failures:
        try:
            comparison_rows, comparison = _comparison_rows(completed)
            _write_csv_atomic(root / "results.csv", comparison_rows)
        except ValueError as exc:
            comparison_error = str(exc)

    method_action_counts = {
        method: {
            action: sum(
                receipt["result"]["selected_action"] == action
                for receipt in completed
                if receipt["method"] == method
            )
            for action in ACTION_NAMES
        }
        for method in METHODS
    }
    probe_receipts = [
        receipt for receipt in completed if receipt["method"] == "probe_commit_v2"
    ]
    numerical_repairs = {
        method: {
            "total": sum(
                int(receipt["result"]["numerical_repair_count"])
                for receipt in completed
                if receipt["method"] == method
            ),
            "runs": sum(
                int(receipt["result"]["numerical_repair_count"]) > 0
                for receipt in completed
                if receipt["method"] == method
            ),
        }
        for method in METHODS
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "context_count": len(contexts),
        "expected_method_runs": len(contexts) * len(METHODS),
        "completed": len(completed),
        "failed": len(failures),
        "all_exact_global_fes": all(
            receipt["objective_fes"] == config["global_max_fes"]
            for receipt in completed
        ),
        "checkpoint_pairs_equal": comparison_error is None and not failures,
        "all_single_commit": all(
            receipt["result"]["committed_action_count"] == 1
            for receipt in completed
        ),
        "comparison_error": comparison_error,
        "method_action_counts": method_action_counts,
        "numerical_sigma_floor_repairs": numerical_repairs,
        "probe_cap_fallback_count": sum(
            receipt["result"]["selection_reason"].startswith("probe_cap_")
            for receipt in probe_receipts
        ),
        "probe_context_count": len(probe_receipts),
        "comparison": comparison,
        "quality_claim": "limited_pre_registered_validation_not_generalization",
    }
    _write_json_atomic(root / "summary.json", summary)
    if failures or comparison_error is not None:
        raise RuntimeError(
            f"Phase-II v2 validation failed: {len(failures)} run failures; "
            f"comparison={comparison_error}"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    summary = run_validation(
        config_path=arguments.config,
        output_root=arguments.output_root,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
