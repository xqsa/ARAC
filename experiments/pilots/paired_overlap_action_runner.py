"""Shared runner for paired non-dispatch overlap-action pilots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.arac.backends.hcc_budget import (
    _parse_hcc_budget_summary,
    _parse_hcc_evaluation_record_with_optimizer_final_fe,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "hcc_smoke_runner.py"
VENDOR_ROOT = REPOSITORY_ROOT / "vendor" / "hcc"
DEFAULT_AOB_DATA_ROOT = VENDOR_ROOT / "AOB" / "AOBG" / "datafile"
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "pilots"
    / "exp_021_shared_variable_repair_pilot"
    / "config.json"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "exp_021_shared_variable_repair_pilot"
CASE_TO_FUNCTION = {
    "E3": ("elliptic", 3),
    "A4": ("ackley", 4),
    "S5": ("schwefel", 5),
}
BASELINE_LANE_ID = "hcc_baseline"
BASELINE_ACTION = "conservative_no_action"
SUPPORTED_TARGET_ACTIONS = frozenset(
    {"allow_beneficial_coordination", "repair_shared_variable_binding"}
)
SUBPROCESS_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    case: str
    seed: int
    max_fes: int
    lane_id: str
    arac_action: str
    output_root: Path

    @property
    def trajectory_id(self) -> str:
        return f"{self.experiment_id}-{self.case.lower()}-seed{self.seed}-{self.lane_id}"


@dataclass(frozen=True)
class RunResult:
    trajectory_id: str
    case: str
    seed: int
    lane_id: str
    arac_action: str
    enable_relation_dispatch: bool
    evidence_overlay_mode: str
    status: str
    final_error: float
    fitness_record_fe: int
    max_fes: int
    same_budget_violation: int
    action_trace_rows: int
    selected_actions: str
    elapsed_seconds: float
    returncode: int
    output_root: Path
    error_detail: str


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment config requires experiment_id")
    execution = payload.get("execution")
    lanes = payload.get("lanes")
    comparison = payload.get("comparison")
    if not isinstance(execution, dict) or not isinstance(lanes, list):
        raise ValueError("experiment config requires execution and lanes")
    if not isinstance(comparison, dict):
        raise ValueError("experiment config requires comparison")
    if execution.get("enable_relation_dispatch") is not False:
        raise ValueError("paired action pilot requires enable_relation_dispatch=false")
    if execution.get("evidence_overlay_mode") != "off":
        raise ValueError("paired action pilot requires evidence_overlay_mode=off")
    if execution.get("budget_accounting") != "strict":
        raise ValueError("paired action pilot requires strict FE accounting")
    if execution.get("search_state_backend") != "phase_i_mmes":
        raise ValueError("paired action pilot requires phase_i_mmes")
    if tuple(execution.get("cases", ())) != tuple(CASE_TO_FUNCTION):
        raise ValueError("paired action pilot cases must be E3, A4, and S5")
    seeds = execution.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise ValueError("execution.seeds must contain non-negative integers")
    max_fes = execution.get("max_fes")
    if isinstance(max_fes, bool) or not isinstance(max_fes, int) or max_fes <= 0:
        raise ValueError("execution.max_fes must be a positive integer")
    lane_actions = {
        str(lane.get("lane_id")): str(lane.get("arac_action"))
        for lane in lanes
        if isinstance(lane, dict)
    }
    if len(lane_actions) != 2 or len(lane_actions) != len(lanes):
        raise ValueError("paired action pilot requires exactly two unique lanes")
    if comparison.get("baseline_lane") != BASELINE_LANE_ID:
        raise ValueError("comparison baseline must be hcc_baseline")
    action_lane = str(comparison.get("action_lane", ""))
    if set(lane_actions) != {BASELINE_LANE_ID, action_lane}:
        raise ValueError("comparison lanes must match configured lanes")
    if lane_actions[BASELINE_LANE_ID] != BASELINE_ACTION:
        raise ValueError("hcc_baseline must use conservative_no_action")
    if lane_actions[action_lane] not in SUPPORTED_TARGET_ACTIONS:
        raise ValueError("comparison action is not a supported non-dispatch target")
    return payload


def build_run_matrix(config: Mapping[str, object], output_root: Path) -> list[RunSpec]:
    execution = config["execution"]
    lanes = config["lanes"]
    assert isinstance(execution, dict) and isinstance(lanes, list)
    return [
        RunSpec(
            experiment_id=str(config["experiment_id"]),
            case=str(case),
            seed=int(seed),
            max_fes=int(execution["max_fes"]),
            lane_id=str(lane["lane_id"]),
            arac_action=str(lane["arac_action"]),
            output_root=(
                output_root
                / "runs"
                / str(case)
                / f"seed_{seed}"
                / str(lane["lane_id"])
            ),
        )
        for case in execution["cases"]
        for seed in execution["seeds"]
        for lane in lanes
    ]


def build_command(
    spec: RunSpec,
    config: Mapping[str, object],
    *,
    python_executable: str,
) -> tuple[str, ...]:
    execution = config["execution"]
    assert isinstance(execution, dict)
    function_name, function_id = CASE_TO_FUNCTION[spec.case]
    data_root = Path(str(execution.get("aob_data_root", DEFAULT_AOB_DATA_ROOT)))
    if not data_root.is_absolute():
        data_root = REPOSITORY_ROOT / data_root
    command = [
        python_executable,
        str(RUNNER_PATH),
        "--functions",
        function_name,
        "--ids",
        str(function_id),
        "--output-root",
        str(spec.output_root),
        "--aob-data-root",
        str(data_root.resolve()),
        "--timestamp",
        spec.trajectory_id,
        "--seed",
        str(spec.seed),
        "--max-fes",
        str(spec.max_fes),
        "--arac-action",
        spec.arac_action,
        "--budget-accounting",
        str(execution["budget_accounting"]),
        "--search-state-backend",
        str(execution["search_state_backend"]),
        "--relation-policy",
        str(execution["relation_policy"]),
        "--evidence-overlay-mode",
        str(execution["evidence_overlay_mode"]),
    ]
    if execution.get("skip_plots") is True:
        command.append("--skip-plots")
    if execution.get("enable_relation_dispatch") is True:
        command.append("--enable-relation-dispatch")
    return tuple(command)


def _read_budget_audit(output_root: Path) -> tuple[int, int]:
    paths = sorted(output_root.rglob("*budget_summary.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one budget summary, found {len(paths)}")
    with paths[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError("budget summary must contain exactly one row")
    return int(rows[0]["fitness_record_fe"]), int(rows[0]["same_budget_violation"])


def _read_action_trace(output_root: Path, case: str) -> tuple[int, str]:
    paths = sorted(output_root.rglob(f"{case}_action_trace.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one case action trace, found {len(paths)}")
    with paths[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actions = sorted({row["selected_action_name"] for row in rows})
    return len(rows), ";".join(actions)


def _tail(value: str, limit: int = 2000) -> str:
    return (value or "")[-limit:].replace("\x00", "")


def execute_one(
    spec: RunSpec,
    config: Mapping[str, object],
    *,
    python_executable: str,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> RunResult:
    spec.output_root.mkdir(parents=True, exist_ok=True)
    command = build_command(spec, config, python_executable=python_executable)
    environment = {**os.environ, **SUBPROCESS_ENVIRONMENT}
    started = time.perf_counter()
    completed = command_runner(
        command,
        cwd=VENDOR_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return RunResult(
            trajectory_id=spec.trajectory_id,
            case=spec.case,
            seed=spec.seed,
            lane_id=spec.lane_id,
            arac_action=spec.arac_action,
            enable_relation_dispatch=False,
            evidence_overlay_mode="off",
            status=f"failed_returncode_{completed.returncode}",
            final_error=float("nan"),
            fitness_record_fe=0,
            max_fes=spec.max_fes,
            same_budget_violation=1,
            action_trace_rows=0,
            selected_actions="",
            elapsed_seconds=elapsed,
            returncode=completed.returncode,
            output_root=spec.output_root,
            error_detail=_tail(completed.stderr or completed.stdout),
        )
    try:
        final_error, _fe_used, _optimizer_fe = (
            _parse_hcc_evaluation_record_with_optimizer_final_fe(
                spec.output_root,
                budget_limit=spec.max_fes,
            )
        )
        budget = _parse_hcc_budget_summary(spec.output_root)
        fitness_record_fe, same_budget_violation = _read_budget_audit(spec.output_root)
        action_trace_rows, selected_actions = _read_action_trace(spec.output_root, spec.case)
        if budget.get("fitness_record_fe") != fitness_record_fe:
            raise RuntimeError("budget parsers disagree on fitness_record_fe")
        if not math.isfinite(final_error) or final_error < 0.0:
            raise RuntimeError("final error must be finite and non-negative")
        if action_trace_rows < 1 or selected_actions != spec.arac_action:
            raise RuntimeError(
                f"action trace mismatch: expected {spec.arac_action}, got {selected_actions}"
            )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        return RunResult(
            trajectory_id=spec.trajectory_id,
            case=spec.case,
            seed=spec.seed,
            lane_id=spec.lane_id,
            arac_action=spec.arac_action,
            enable_relation_dispatch=False,
            evidence_overlay_mode="off",
            status="audit_failed",
            final_error=float("nan"),
            fitness_record_fe=0,
            max_fes=spec.max_fes,
            same_budget_violation=1,
            action_trace_rows=0,
            selected_actions="",
            elapsed_seconds=elapsed,
            returncode=completed.returncode,
            output_root=spec.output_root,
            error_detail=str(error),
        )
    return RunResult(
        trajectory_id=spec.trajectory_id,
        case=spec.case,
        seed=spec.seed,
        lane_id=spec.lane_id,
        arac_action=spec.arac_action,
        enable_relation_dispatch=False,
        evidence_overlay_mode="off",
        status="completed",
        final_error=final_error,
        fitness_record_fe=fitness_record_fe,
        max_fes=spec.max_fes,
        same_budget_violation=same_budget_violation,
        action_trace_rows=action_trace_rows,
        selected_actions=selected_actions,
        elapsed_seconds=elapsed,
        returncode=completed.returncode,
        output_root=spec.output_root,
        error_detail="",
    )


def build_paired_comparison(
    results: Sequence[RunResult],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    execution = config["execution"]
    comparison = config["comparison"]
    assert isinstance(execution, dict) and isinstance(comparison, dict)
    by_key = {(result.case, result.seed, result.lane_id): result for result in results}
    rows: list[dict[str, object]] = []
    for case in execution["cases"]:
        for seed in execution["seeds"]:
            baseline = by_key.get((str(case), int(seed), str(comparison["baseline_lane"])))
            action = by_key.get((str(case), int(seed), str(comparison["action_lane"])))
            complete = (
                baseline is not None
                and action is not None
                and baseline.status == "completed"
                and action.status == "completed"
                and baseline.same_budget_violation == 0
                and action.same_budget_violation == 0
                and baseline.fitness_record_fe
                >= baseline.max_fes - int(comparison["maximum_terminal_fe_shortfall"])
                and action.fitness_record_fe
                >= action.max_fes - int(comparison["maximum_terminal_fe_shortfall"])
            )
            if not complete:
                rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "status": "blocked",
                        "baseline_final_error": "",
                        "action_final_error": "",
                        "baseline_fe": "" if baseline is None else baseline.fitness_record_fe,
                        "action_fe": "" if action is None else action.fitness_record_fe,
                        "equal_fe": 0,
                        "fe_difference": "",
                        "same_budget_gate": 0,
                        "log_gain": "",
                        "relative_gain": "",
                        "action_better": 0,
                        "catastrophic_loss": 0,
                    }
                )
                continue
            assert baseline is not None and action is not None
            epsilon = float(comparison["comparison_epsilon"])
            log_gain = math.log((baseline.final_error + epsilon) / (action.final_error + epsilon))
            relative_gain = (baseline.final_error - action.final_error) / max(
                abs(baseline.final_error), epsilon
            )
            catastrophic = action.final_error > (
                float(comparison["catastrophic_multiplier"])
                * max(baseline.final_error, epsilon)
            )
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "status": "completed",
                    "baseline_final_error": baseline.final_error,
                    "action_final_error": action.final_error,
                    "baseline_fe": baseline.fitness_record_fe,
                    "action_fe": action.fitness_record_fe,
                    "equal_fe": int(
                        baseline.fitness_record_fe == action.fitness_record_fe
                    ),
                    "fe_difference": action.fitness_record_fe - baseline.fitness_record_fe,
                    "same_budget_gate": 1,
                    "log_gain": log_gain,
                    "relative_gain": relative_gain,
                    "action_better": int(action.final_error < baseline.final_error),
                    "catastrophic_loss": int(catastrophic),
                }
            )
    return rows


def build_decision(
    run_results: Sequence[RunResult],
    paired_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, object]:
    comparison = config["comparison"]
    assert isinstance(comparison, dict)
    completed = [row for row in paired_rows if row["status"] == "completed"]
    blockers = [result.trajectory_id for result in run_results if result.status != "completed"]
    blockers.extend(
        f"{row['case']}-seed{row['seed']}:paired_FE_or_completion_gate"
        for row in paired_rows
        if row["status"] != "completed"
    )
    if len(completed) != len(paired_rows):
        status = "pilot_blocked"
        answer = "The action effect could not be evaluated because the paired run gate did not close."
        median_log_gain = None
        positive_pairs = 0
        catastrophic_pairs = 0
    else:
        log_gains = [float(row["log_gain"]) for row in completed]
        median_log_gain = statistics.median(log_gains)
        positive_pairs = sum(int(row["action_better"]) for row in completed)
        catastrophic_pairs = sum(int(row["catastrophic_loss"]) for row in completed)
        positive = (
            positive_pairs >= int(comparison["minimum_positive_pairs"])
            and median_log_gain > 0.0
            and catastrophic_pairs <= int(comparison["maximum_catastrophic_pairs"])
        )
        status = "pilot_positive_effect" if positive else "pilot_no_positive_effect"
        answer = (
            "A positive effect was observed in this single-seed 100k-FE pilot."
            if positive
            else "No positive effect was established in this single-seed 100k-FE pilot."
        )
    return {
        "experiment_id": config["experiment_id"],
        "protocol_version": config["protocol_version"],
        "status": status,
        "answer": answer,
        "completed_pairs": len(completed),
        "expected_pairs": len(paired_rows),
        "positive_pairs": positive_pairs,
        "median_log_gain": median_log_gain,
        "catastrophic_pairs": catastrophic_pairs,
        "blockers": sorted(set(blockers)),
        "scope": "fresh_HCC_single_seed_100k_FE_pilot_not_a_statistical_final_claim",
    }


def _csv_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: _csv_value(row.get(field, "")) for field in fields} for row in rows
        )


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python_executable: str = sys.executable,
    jobs: int | None = None,
) -> tuple[list[RunResult], list[dict[str, object]], dict[str, object]]:
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = load_config(config_path)
    specs = build_run_matrix(config, output_root)
    execution = config["execution"]
    assert isinstance(execution, dict)
    worker_count = max(1, int(execution["jobs"] if jobs is None else jobs))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            executor.map(
                lambda spec: execute_one(
                    spec,
                    config,
                    python_executable=python_executable,
                ),
                specs,
            )
        )
    paired_rows = build_paired_comparison(results, config)
    decision = build_decision(results, paired_rows, config)
    run_rows = [asdict(result) for result in results]
    _write_csv(output_root / "run_results.csv", run_rows)
    _write_csv(output_root / "paired_comparison.csv", paired_rows)
    (output_root / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": config["protocol_version"],
        "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "executor": "Codex",
        "git_commit": _git_commit(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runner_path": str(RUNNER_PATH.resolve()),
        "output_root": str(output_root),
        "trajectory_count": len(results),
        "enable_relation_dispatch": False,
        "evidence_overlay_mode": "off",
        "target_action": next(
            str(lane["arac_action"])
            for lane in config["lanes"]
            if lane["lane_id"] == config["comparison"]["action_lane"]
        ),
        "action_logic_changed_by_experiment": False,
        "fresh_optimizer_execution": all(result.status == "completed" for result in results),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results, paired_rows, decision


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args(argv)
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
) -> int:
    args = parse_args(argv, description=description)
    results, paired_rows, decision = run_experiment(
        config_path=args.config,
        output_root=args.output_root,
        python_executable=args.python_executable,
        jobs=args.jobs,
    )
    for result in results:
        print(
            f"[{result.case}/{result.lane_id}] status={result.status} "
            f"FE={result.fitness_record_fe} error={result.final_error:.6e}",
            flush=True,
        )
    for row in paired_rows:
        print(
            f"[{row['case']}/paired] status={row['status']} "
            f"log_gain={row['log_gain']}",
            flush=True,
        )
    print(json.dumps(decision, sort_keys=True), flush=True)
    return 0 if decision["status"] != "pilot_blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
