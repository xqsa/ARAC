"""Run the 5-seed E3/A4/S5/R4 Cohen's d relation-dispatch pilot."""

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
    _parse_hcc_evaluation_record_with_optimizer_final_fe,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "hcc_smoke_runner.py"
VENDOR_ROOT = REPOSITORY_ROOT / "vendor" / "hcc"
DEFAULT_AOB_DATA_ROOT = VENDOR_ROOT / "AOB" / "AOBG" / "datafile"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "exp_020_cohen_d_dispatch_pilot"
CASE_TO_FUNCTION = {
    "E3": ("elliptic", 3),
    "A4": ("ackley", 4),
    "S5": ("schwefel", 5),
    "R4": ("rastrigin", 4),
}
V37_ACTION = "arac_evidence_action_controller_v37"
RELATION_POLICY = "cohen_d_repair"
REPAIR_ACTION = "repair_shared_variable_binding"
CONSERVATIVE_ACTION = "conservative_no_action"
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
    output_root: Path

    @property
    def trajectory_id(self) -> str:
        return f"{self.experiment_id}-{self.case.lower()}-seed{self.seed}"


@dataclass(frozen=True)
class RunResult:
    trajectory_id: str
    case: str
    seed: int
    status: str
    final_error: float
    fitness_record_fe: int
    max_fes: int
    same_budget_violation: int
    relation_count: int
    repair_count: int
    conservative_count: int
    trigger_mismatch_count: int
    selected_relation_actions: str
    elapsed_seconds: float
    returncode: int
    output_root: Path
    error_detail: str


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")
    execution = payload.get("execution")
    controls = payload.get("controls")
    acceptance = payload.get("acceptance")
    if not isinstance(execution, dict):
        raise ValueError("experiment config requires execution")
    if not isinstance(controls, dict) or not isinstance(acceptance, dict):
        raise ValueError("experiment config requires controls and acceptance")
    if tuple(execution.get("cases", ())) != tuple(CASE_TO_FUNCTION):
        raise ValueError("exp_020 cases must be E3, A4, S5, and R4")
    seeds = execution.get("seeds")
    if seeds != [1, 2, 3, 4, 5]:
        raise ValueError("exp_020 freezes seeds 1 through 5")
    if execution.get("max_fes") != 100_000:
        raise ValueError("exp_020 freezes max_fes=100000")
    if execution.get("arac_action") != V37_ACTION:
        raise ValueError("exp_020 requires frozen v37")
    if execution.get("relation_policy") != RELATION_POLICY:
        raise ValueError("exp_020 requires cohen_d_repair")
    if execution.get("enable_relation_dispatch") is not True:
        raise ValueError("exp_020 requires relation dispatch")
    if execution.get("evidence_overlay_mode") != "off":
        raise ValueError("exp_020 requires evidence overlay off")
    if execution.get("budget_accounting") != "strict":
        raise ValueError("exp_020 requires strict FE accounting")
    if execution.get("search_state_backend") != "phase_i_mmes":
        raise ValueError("exp_020 requires phase_i_mmes")
    threshold = float(execution.get("cohen_d_threshold", math.nan))
    if threshold != 0.8:
        raise ValueError("exp_020 freezes cohen_d_threshold=0.8")
    if controls.get("conforming_overlap_cases") != ["R4"]:
        raise ValueError("R4 must be the conforming-overlap control")
    if int(execution.get("jobs", 0)) < 1:
        raise ValueError("execution.jobs must be positive")
    return payload


def build_run_matrix(config: Mapping[str, object], output_root: Path) -> list[RunSpec]:
    execution = config["execution"]
    assert isinstance(execution, dict)
    return [
        RunSpec(
            experiment_id=str(config["experiment_id"]),
            case=str(case),
            seed=int(seed),
            max_fes=int(execution["max_fes"]),
            output_root=output_root / "runs" / str(case) / f"seed_{seed}",
        )
        for case in execution["cases"]
        for seed in execution["seeds"]
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
        str(execution["arac_action"]),
        "--budget-accounting",
        str(execution["budget_accounting"]),
        "--search-state-backend",
        str(execution["search_state_backend"]),
        "--relation-policy",
        str(execution["relation_policy"]),
        "--evidence-overlay-mode",
        str(execution["evidence_overlay_mode"]),
        "--enable-relation-dispatch",
    ]
    if execution.get("skip_plots") is True:
        command.append("--skip-plots")
    return tuple(command)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _single_artifact(output_root: Path, name: str) -> Path:
    paths = sorted(output_root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name}, found {len(paths)}")
    return paths[0]


def read_budget_audit(spec: RunSpec) -> tuple[int, int]:
    rows = _read_csv(_single_artifact(spec.output_root, f"{spec.case}_budget_summary.csv"))
    if len(rows) != 1:
        raise RuntimeError("budget summary must contain exactly one row")
    return int(rows[0]["fitness_record_fe"]), int(rows[0]["same_budget_violation"])


def build_relation_audit(
    spec: RunSpec,
    *,
    threshold: float,
) -> list[dict[str, object]]:
    relation_rows = _read_csv(
        _single_artifact(spec.output_root, f"{spec.case}_overlap_relations.csv")
    )
    trace_rows = [
        row
        for row in _read_csv(_single_artifact(spec.output_root, f"{spec.case}_action_trace.csv"))
        if row["relation_id"]
    ]
    traces_by_relation = {row["relation_id"]: row for row in trace_rows}
    if len(traces_by_relation) != len(trace_rows):
        raise RuntimeError("relation action trace contains duplicate relation ids")
    if len(relation_rows) != len(trace_rows):
        raise RuntimeError("relation evidence and action trace counts differ")

    audit_rows: list[dict[str, object]] = []
    for relation in relation_rows:
        relation_id = relation["relation_id"]
        trace = traces_by_relation.get(relation_id)
        if trace is None:
            raise RuntimeError(f"missing action trace for {relation_id}")
        cohen_d = float(relation["cohen_d"])
        trace_cohen_d = float(trace["cohen_d"])
        trace_threshold = float(trace["cohen_d_threshold"])
        left_count = int(relation["left_top_k_count"])
        right_count = int(relation["right_top_k_count"])
        expected = REPAIR_ACTION if cohen_d > threshold else CONSERVATIVE_ACTION
        selected = trace["selected_action_name"]
        consistent = (
            selected == expected
            and trace["expected_action_name"] == expected
            and math.isclose(trace_cohen_d, cohen_d, rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(trace_threshold, threshold, rel_tol=0.0, abs_tol=1e-15)
            and int(trace["left_top_k_count"]) == left_count
            and int(trace["right_top_k_count"]) == right_count
            and 0 < left_count <= 5
            and 0 < right_count <= 5
        )
        audit_rows.append(
            {
                "trajectory_id": spec.trajectory_id,
                "case": spec.case,
                "seed": spec.seed,
                "relation_id": relation_id,
                "outer_iter": relation["outer_iter"],
                "group_left": relation["group_left"],
                "group_right": relation["group_right"],
                "shared_vars": relation["shared_vars"],
                "cohen_d": cohen_d,
                "threshold": threshold,
                "above_threshold": int(cohen_d > threshold),
                "left_top_k_count": left_count,
                "right_top_k_count": right_count,
                "left_distribution_centers": relation["left_distribution_centers"],
                "right_distribution_centers": relation["right_distribution_centers"],
                "left_distribution_standard_deviations": relation[
                    "left_distribution_standard_deviations"
                ],
                "right_distribution_standard_deviations": relation[
                    "right_distribution_standard_deviations"
                ],
                "expected_action_name": expected,
                "selected_action_name": selected,
                "state_mutated": trace["state_mutated"],
                "trigger_consistent": int(consistent),
            }
        )
    return audit_rows


def _tail(value: str, limit: int = 2000) -> str:
    return (value or "")[-limit:].replace("\x00", "")


def execute_one(
    spec: RunSpec,
    config: Mapping[str, object],
    *,
    python_executable: str,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    run_subprocess: bool = True,
) -> tuple[RunResult, list[dict[str, object]]]:
    spec.output_root.mkdir(parents=True, exist_ok=True)
    command = build_command(spec, config, python_executable=python_executable)
    if run_subprocess:
        started = time.perf_counter()
        completed = command_runner(
            command,
            cwd=VENDOR_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, **SUBPROCESS_ENVIRONMENT},
        )
        elapsed = time.perf_counter() - started
    else:
        completed = subprocess.CompletedProcess(command, 0, "", "")
        elapsed = 0.0
    if completed.returncode != 0:
        return (
            RunResult(
                spec.trajectory_id,
                spec.case,
                spec.seed,
                f"failed_returncode_{completed.returncode}",
                math.nan,
                0,
                spec.max_fes,
                1,
                0,
                0,
                0,
                0,
                "",
                elapsed,
                completed.returncode,
                spec.output_root,
                _tail(completed.stderr or completed.stdout),
            ),
            [],
        )

    try:
        final_error, _fe_used, _optimizer_fe = _parse_hcc_evaluation_record_with_optimizer_final_fe(
            spec.output_root,
            budget_limit=spec.max_fes,
        )
        audit_rows = build_relation_audit(
            spec,
            threshold=float(config["execution"]["cohen_d_threshold"]),
        )
        if not audit_rows:
            raise RuntimeError("trajectory produced no overlap relations")
        selected = sorted({str(row["selected_action_name"]) for row in audit_rows})
        mismatch_count = sum(1 - int(row["trigger_consistent"]) for row in audit_rows)
        repair_count = sum(int(row["above_threshold"]) for row in audit_rows)
        conservative_count = len(audit_rows) - repair_count
        fitness_record_fe, same_budget_violation = read_budget_audit(spec)
        if not math.isfinite(final_error) or final_error < 0.0:
            raise RuntimeError("final error must be finite and non-negative")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        return (
            RunResult(
                spec.trajectory_id,
                spec.case,
                spec.seed,
                "audit_failed",
                math.nan,
                0,
                spec.max_fes,
                1,
                0,
                0,
                0,
                0,
                "",
                elapsed,
                completed.returncode,
                spec.output_root,
                str(error),
            ),
            [],
        )

    return (
        RunResult(
            spec.trajectory_id,
            spec.case,
            spec.seed,
            "completed",
            final_error,
            fitness_record_fe,
            spec.max_fes,
            same_budget_violation,
            len(audit_rows),
            repair_count,
            conservative_count,
            mismatch_count,
            ";".join(selected),
            elapsed,
            completed.returncode,
            spec.output_root,
            "",
        ),
        audit_rows,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_cohen_d_summary(
    audit_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for case in (*CASE_TO_FUNCTION, "TARGETS", "ALL"):
        rows = [
            row
            for row in audit_rows
            if case == "ALL" or (case == "TARGETS" and row["case"] != "R4") or row["case"] == case
        ]
        if not rows:
            continue
        values = [float(row["cohen_d"]) for row in rows]
        repair_count = sum(int(row["above_threshold"]) for row in rows)
        consistent_count = sum(int(row["trigger_consistent"]) for row in rows)
        summaries.append(
            {
                "case": case,
                "control_role": (
                    "conforming_overlap"
                    if case == "R4"
                    else "target_aggregate"
                    if case == "TARGETS"
                    else "all_aggregate"
                    if case == "ALL"
                    else "target"
                ),
                "trajectory_count": len({row["trajectory_id"] for row in rows}),
                "relation_count": len(rows),
                "cohen_d_min": min(values),
                "cohen_d_p25": _quantile(values, 0.25),
                "cohen_d_median": statistics.median(values),
                "cohen_d_p75": _quantile(values, 0.75),
                "cohen_d_max": max(values),
                "above_threshold_count": repair_count,
                "above_threshold_rate": repair_count / len(rows),
                "repair_count": sum(row["selected_action_name"] == REPAIR_ACTION for row in rows),
                "repair_rate": sum(row["selected_action_name"] == REPAIR_ACTION for row in rows)
                / len(rows),
                "repair_state_mutated_count": sum(
                    row["selected_action_name"] == REPAIR_ACTION
                    and str(row.get("state_mutated", "")) == "1"
                    for row in rows
                ),
                "repair_state_mutated_rate": (
                    sum(
                        row["selected_action_name"] == REPAIR_ACTION
                        and str(row.get("state_mutated", "")) == "1"
                        for row in rows
                    )
                    / repair_count
                    if repair_count
                    else 0.0
                ),
                "trigger_consistent_count": consistent_count,
                "trigger_consistent_rate": consistent_count / len(rows),
            }
        )
    return summaries


def build_decision(
    results: Sequence[RunResult],
    audit_rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, object]:
    acceptance = config["acceptance"]
    assert isinstance(acceptance, dict)
    shortfall = int(acceptance["maximum_terminal_fe_shortfall"])
    blockers = [
        result.trajectory_id
        for result in results
        if result.status != "completed"
        or result.same_budget_violation != 0
        or result.fitness_record_fe < result.max_fes - shortfall
        or result.trigger_mismatch_count != 0
    ]
    expected_trajectories = len(CASE_TO_FUNCTION) * 5
    if len(results) != expected_trajectories:
        blockers.append("trajectory_count_mismatch")
    if not audit_rows:
        blockers.append("no_relation_evidence")
    overall = next((row for row in summaries if row["case"] == "ALL"), None)
    if overall is None or float(overall["trigger_consistent_rate"]) != 1.0:
        blockers.append("relation_trigger_inconsistency")
    by_case = {str(row["case"]): row for row in summaries}
    r4_median = None if "R4" not in by_case else float(by_case["R4"]["cohen_d_median"])
    target_median = (
        None if "TARGETS" not in by_case else float(by_case["TARGETS"]["cohen_d_median"])
    )
    control_separation_observed = (
        None if r4_median is None or target_median is None else r4_median < target_median
    )
    return {
        "experiment_id": config["experiment_id"],
        "status": "mechanism_verified" if not blockers else "pilot_blocked",
        "trajectory_count": len(results),
        "completed_trajectory_count": sum(result.status == "completed" for result in results),
        "relation_count": len(audit_rows),
        "repair_count": sum(int(row["above_threshold"]) for row in audit_rows),
        "conservative_count": sum(1 - int(row["above_threshold"]) for row in audit_rows),
        "trigger_mismatch_count": sum(1 - int(row["trigger_consistent"]) for row in audit_rows),
        "cohen_d_threshold": config["execution"]["cohen_d_threshold"],
        "r4_control_median": r4_median,
        "target_cases_median": target_median,
        "r4_to_target_median_ratio": (
            None if r4_median is None or target_median in {None, 0.0} else r4_median / target_median
        ),
        "control_separation_observed": control_separation_observed,
        "distribution_conclusion": (
            "R4 had lower Cohen's d than the pooled target cases."
            if control_separation_observed is True
            else "R4 did not have lower Cohen's d than the pooled target cases."
            if control_separation_observed is False
            else "Control separation could not be evaluated."
        ),
        "blockers": sorted(set(blockers)),
        "conclusion": (
            "Every runtime relation action matched the preregistered Cohen's d > 0.8 rule."
            if not blockers
            else "The runtime mechanism audit did not pass all preregistered gates."
        ),
    }


def _csv_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
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
    reuse_existing: bool = False,
) -> tuple[
    list[RunResult],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = load_config(config_path)
    specs = build_run_matrix(config, output_root)
    execution = config["execution"]
    assert isinstance(execution, dict)
    worker_count = max(1, int(execution["jobs"] if jobs is None else jobs))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        outcomes = list(
            executor.map(
                lambda spec: execute_one(
                    spec,
                    config,
                    python_executable=python_executable,
                    run_subprocess=not reuse_existing,
                ),
                specs,
            )
        )
    results = [outcome[0] for outcome in outcomes]
    audit_rows = [row for outcome in outcomes for row in outcome[1]]
    summaries = build_cohen_d_summary(audit_rows) if audit_rows else []
    decision = build_decision(results, audit_rows, summaries, config)

    _write_csv(output_root / "run_results.csv", [asdict(result) for result in results])
    if audit_rows:
        _write_csv(output_root / "cohen_d_relations.csv", audit_rows)
        audit_fields = (
            "trajectory_id",
            "case",
            "seed",
            "relation_id",
            "cohen_d",
            "threshold",
            "above_threshold",
            "expected_action_name",
            "selected_action_name",
            "trigger_consistent",
        )
        _write_csv(
            output_root / "repair_trigger_audit.csv",
            [{field: row[field] for field in audit_fields} for row in audit_rows],
        )
        _write_csv(output_root / "cohen_d_summary.csv", summaries)
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
        "enable_relation_dispatch": True,
        "relation_policy": RELATION_POLICY,
        "evidence_overlay_mode": "off",
        "fresh_optimizer_execution": all(result.status == "completed" for result in results),
        "reused_existing_outputs": reuse_existing,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results, audit_rows, summaries, decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results, _audit_rows, summaries, decision = run_experiment(
        config_path=args.config,
        output_root=args.output_root,
        python_executable=args.python_executable,
        jobs=args.jobs,
        reuse_existing=args.reuse_existing,
    )
    for result in results:
        print(
            f"[{result.case}/seed{result.seed}] status={result.status} "
            f"FE={result.fitness_record_fe} relations={result.relation_count} "
            f"repairs={result.repair_count}",
            flush=True,
        )
    for row in summaries:
        print(
            f"[{row['case']}] n={row['relation_count']} "
            f"median={float(row['cohen_d_median']):.6e} "
            f"repair_rate={float(row['repair_rate']):.3f}",
            flush=True,
        )
    print(json.dumps(decision, sort_keys=True), flush=True)
    return 0 if decision["status"] == "mechanism_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
