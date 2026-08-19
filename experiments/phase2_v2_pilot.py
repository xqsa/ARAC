"""Small, pre-registered AOB + IOH pilot for the Phase-II v2 policy."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Any

import numpy as np

from arac.benchmarks.aob import AobBenchmark, OptimizationProblem
from arac.benchmarks.ioh_bbob import IohBbobBenchmark
from arac.evidence.phase1 import phase1_budget, run_phase1
from arac.runtime.contracts import ACTION_NAMES, ActionContext, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.probe_policy import ProbePolicyResult, run_probe_commit_policy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "phase2_v2_pilot_config_v2.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "phase2_v2_pilot_ioh_v3"
CONFIG_SCHEMA = "arac-phase2-v2-pilot-config-v2"
MANIFEST_SCHEMA = "arac-phase2-v2-pilot-manifest-v1"
RECEIPT_SCHEMA = "arac-phase2-v2-pilot-receipt-v1"
SUMMARY_SCHEMA = "arac-phase2-v2-pilot-summary-v1"
FORBIDDEN_SEEDS = {117, 129, 141, 142, 20260753}
SOURCE_PATHS = (
    "experiments/phase2_v2_pilot.py",
    "src/arac/actions/phase2_v2.py",
    "src/arac/benchmarks/aob.py",
    "src/arac/benchmarks/ioh_bbob.py",
    "src/arac/evidence/phase1.py",
    "src/arac/evidence/structural.py",
    "src/arac/runtime/branches.py",
    "src/arac/runtime/optimizers.py",
    "src/arac/runtime/probe_policy.py",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> tuple[int, str]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    payload = [
        [path.relative_to(root).as_posix(), _file_sha256(path)]
        for path in files
    ]
    return len(files), canonical_sha256(payload)


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    values = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "ioh_version",
        "global_max_fes",
        "branch_probe_fes",
        "decision_horizon_fes",
        "exploration_floor_fes",
        "min_relative_margin",
        "min_leader_stability",
        "runs",
    }
    if set(values) != required or values["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("pilot config schema or keys drifted")
    for name in (
        "global_max_fes",
        "branch_probe_fes",
        "decision_horizon_fes",
        "exploration_floor_fes",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"pilot {name} must be a positive integer")
    if not 0 < values["decision_horizon_fes"] < values["branch_probe_fes"]:
        raise ValueError("pilot decision horizon is outside the probe")
    if 4 * values["branch_probe_fes"] > (
        values["global_max_fes"] - phase1_budget(values["global_max_fes"])
    ):
        raise ValueError("pilot branch probes exceed the Phase-II budget")
    if importlib.metadata.version("ioh") != values["ioh_version"]:
        raise ValueError("pilot IOH version drifted")
    runs = values["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("pilot runs must be a non-empty list")
    seeds = []
    for run in runs:
        if not isinstance(run, dict) or run.get("suite") not in {"aob", "ioh_bbob"}:
            raise ValueError("pilot run suite is invalid")
        seed = run.get("run_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("pilot run_seed must be a non-negative integer")
        seeds.append(seed)
        if run["suite"] == "aob":
            if set(run) != {"suite", "case", "run_seed"}:
                raise ValueError("pilot AOB run keys drifted")
        elif set(run) != {
            "suite",
            "function_id",
            "instance",
            "dimension",
            "run_seed",
        }:
            raise ValueError("pilot IOH run keys drifted")
    if len(seeds) != len(set(seeds)) or set(seeds) & FORBIDDEN_SEEDS:
        raise ValueError("pilot run seeds overlap or reuse a forbidden seed")
    return values


class _ObjectiveCounter:
    def __init__(self, objective) -> None:
        self.objective = objective
        self.count = 0

    def __call__(self, values):
        candidates = np.asarray(values, dtype=float)
        requested = 1 if candidates.ndim == 1 else len(candidates)
        result = self.objective(values)
        self.count += requested
        return result


def _counted_problem(problem: OptimizationProblem) -> tuple[OptimizationProblem, _ObjectiveCounter]:
    counter = _ObjectiveCounter(problem.objective)
    return (
        OptimizationProblem(
            objective=counter,
            dimension=problem.dimension,
            lower_bounds=problem.lower_bounds,
            upper_bounds=problem.upper_bounds,
            optimum=problem.optimum,
        ),
        counter,
    )


def _load_problem(run: dict[str, Any]) -> OptimizationProblem:
    if run["suite"] == "aob":
        return AobBenchmark().load(str(run["case"]))
    return IohBbobBenchmark().load(
        int(run["function_id"]),
        instance=int(run["instance"]),
        dimension=int(run["dimension"]),
    )


def _run_label(run: dict[str, Any]) -> str:
    if run["suite"] == "aob":
        return f"aob_{run['case']}_{run['run_seed']}"
    return (
        f"ioh_f{int(run['function_id']):02d}_i{run['instance']}_"
        f"d{run['dimension']}_{run['run_seed']}"
    )


def _result_payload(result: ProbePolicyResult) -> dict[str, Any]:
    return {
        "selected_action": result.selected_action,
        "archive_source_action": result.archive_source_action,
        "commit_reason": result.commit_reason,
        "checkpoint_sha256": result.checkpoint_hash,
        "global_total_fes": result.global_total_fes,
        "action_schedule_total_fes": result.action_schedule_total_fes,
        "selected_ledger_fes": result.selected_ledger_fes,
        "phase1_fes": result.phase1_fes,
        "branch_probe_fes": result.branch_probe_fes,
        "continuation_fes": result.continuation_fes,
        "aggregate_fes": result.aggregate_fes,
        "selected_action_fes": result.selected_action_fes,
        "final_error": result.final_error,
        "incumbent": list(result.incumbent),
        "probe_final_errors": dict(result.probe_final_errors),
        "decision": {
            "action_name": result.decision.action_name,
            "reason": result.decision.reason,
            "observed_fes": result.decision.observed_fes,
            "exploration_floor_fes": result.decision.exploration_floor_fes,
            "relative_margin": result.decision.relative_margin,
            "leader_stability": result.decision.leader_stability,
        },
        "route": result.route,
        "optimizer_package": result.optimizer_package,
        "optimizer_version": result.optimizer_version,
        "selected_state_sha256": result.selected_state_hash,
        "numerical_repair_count": result.numerical_repair_count,
    }


def _run_one(run: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    raw_problem = _load_problem(run)
    problem, counter = _counted_problem(raw_problem)
    ledger = EvaluationLedger(problem, int(config["global_max_fes"]))
    phase1 = run_phase1(problem, ledger, run_seed=int(run["run_seed"]))
    context = ActionContext(
        "aor",
        phase1.checkpoint,
        problem,
        ledger,
        action_seed=int(run["run_seed"]),
    )
    result = run_probe_commit_policy(
        context,
        global_total_fes=int(config["global_max_fes"]),
        branch_probe_fes=int(config["branch_probe_fes"]),
        decision_horizon_fes=int(config["decision_horizon_fes"]),
        exploration_floor_fes=int(config["exploration_floor_fes"]),
        min_relative_margin=float(config["min_relative_margin"]),
        min_leader_stability=float(config["min_leader_stability"]),
    )
    if counter.count != result.aggregate_fes:
        raise RuntimeError(
            f"objective counter recorded {counter.count} FE, expected {result.aggregate_fes}"
        )
    return _result_payload(result), counter.count


def run_pilot(*, config_path: Path, output_root: Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"pilot output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    receipts = root / "receipts"
    receipts.mkdir()

    config = load_config(config_file)
    config_sha256 = _file_sha256(config_file)
    source_hashes = {
        relative: _file_sha256(REPOSITORY_ROOT / relative)
        for relative in SOURCE_PATHS
    }
    vendor_count, vendor_hash = _tree_sha256(REPOSITORY_ROOT / "vendor" / "aob")
    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA,
        "config_sha256": config_sha256,
        "ioh_version": importlib.metadata.version("ioh"),
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
    for index, run in enumerate(config["runs"]):
        label = _run_label(run)
        receipt_path = receipts / f"{index:03d}_{label}.json"
        try:
            result, objective_fes = _run_one(run, config)
            payload = {
                "schema_version": RECEIPT_SCHEMA,
                "status": "completed",
                "run_index": index,
                "benchmark": run,
                "manifest_sha256": manifest["manifest_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "objective_fes": objective_fes,
                "result": result,
            }
            receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
            completed.append(receipt)
        except Exception as exc:
            payload = {
                "schema_version": RECEIPT_SCHEMA,
                "status": "failed",
                "run_index": index,
                "benchmark": run,
                "manifest_sha256": manifest["manifest_sha256"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
            failures.append(receipt)
        _write_json_atomic(receipt_path, receipt)

    selected_counts = Counter(
        receipt["result"]["selected_action"] for receipt in completed
    )
    archive_source_counts = Counter(
        receipt["result"]["archive_source_action"] for receipt in completed
    )
    commit_reason_counts = Counter(
        receipt["result"]["commit_reason"] for receipt in completed
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "run_count": len(config["runs"]),
        "completed": len(completed),
        "failed": len(failures),
        "all_exact_global_fes": all(
            receipt["objective_fes"] == config["global_max_fes"]
            for receipt in completed
        ),
        "selected_action_counts": {
            action: selected_counts.get(action, 0) for action in ACTION_NAMES
        },
        "archive_source_action_counts": {
            action: archive_source_counts.get(action, 0) for action in ACTION_NAMES
        },
        "commit_reason_counts": dict(sorted(commit_reason_counts.items())),
        "all_single_commit": all(
            receipt["result"]["selected_action"] in ACTION_NAMES
            and len(receipt["result"]["selected_state_sha256"]) == 64
            for receipt in completed
        ),
        "quality_claim": "none_protocol_pilot_only",
    }
    _write_json_atomic(root / "summary.json", summary)
    if failures:
        raise RuntimeError(f"Phase-II v2 pilot failed {len(failures)} run(s)")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    summary = run_pilot(
        config_path=arguments.config,
        output_root=arguments.output_root,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
