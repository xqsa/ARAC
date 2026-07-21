"""Run the exp026 persistent Phase2 action validation cohort.

This experiment runs exactly one authorized persistent action per R2-R6/S2-S6
trajectory.  It deliberately does not rerun native HCC or paper baselines.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "hcc_smoke_runner.py"
VENDOR_ROOT = REPOSITORY_ROOT / "vendor" / "hcc"
DEFAULT_AOB_DATA_ROOT = VENDOR_ROOT / "AOB" / "AOBG" / "datafile"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "exp_026_arac_vs_hcc_paired"

PROTOCOL_VERSION = "persistent-phase2-action-validation-v1"
RUN_SUMMARY_PROTOCOL_VERSION = "hcc-run-summary-v3"
PERSISTENT_ACTION_ARTIFACT_SCHEMA = "persistent-phase2-action-v1"
EXACT_MAX_FES = 3_000_000
VALIDATION_SEEDS = (117, 118, 119, 120, 121)
SUPPORTED_CASES = ("R2", "R3", "R4", "R5", "R6", "S2", "S3", "S4", "S5", "S6")
CASE_TO_FUNCTION = {
    "R2": ("rastrigin", 2),
    "R3": ("rastrigin", 3),
    "R4": ("rastrigin", 4),
    "R5": ("rastrigin", 5),
    "R6": ("rastrigin", 6),
    "S2": ("schwefel", 2),
    "S3": ("schwefel", 3),
    "S4": ("schwefel", 4),
    "S5": ("schwefel", 5),
    "S6": ("schwefel", 6),
}
R_ACTION = "full_space_sep_cma"
S_ACTION = "persistent_frozen_efficiency_budget_reallocation"
V37_ACTION = "arac_evidence_action_controller_v37"
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
    action: str
    output_root: Path

    @property
    def trajectory_id(self) -> str:
        return f"{self.experiment_id}-{self.case.lower()}-seed{self.seed}"

    @property
    def run_directory(self) -> Path:
        return self.output_root / "runs" / self.case / f"seed_{self.seed}"

    @property
    def result_directory(self) -> Path:
        function_name, _ = CASE_TO_FUNCTION[self.case]
        return self.run_directory / self.trajectory_id / function_name


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _expected_action(case: str) -> str:
    if case.startswith("R"):
        return R_ACTION
    if case.startswith("S"):
        return S_ACTION
    raise ValueError(f"unsupported persistent Phase2 case: {case!r}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _as_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _final_error(value: object, source: str) -> float:
    try:
        error = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} final_error is not numeric") from exc
    if not math.isfinite(error) or error < 0.0:
        raise ValueError(f"{source} final_error must be finite and non-negative")
    return error


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "config must be a JSON object")
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "unsupported protocol")
    _require(
        payload.get("stage") == "persistent_phase2_action_validation",
        "exp026 is persistent Phase2 action validation",
    )
    execution = payload.get("execution")
    _require(isinstance(execution, dict), "execution config missing")
    _require(tuple(execution.get("cases", ())) == SUPPORTED_CASES, "unsupported AOB cases")
    _require(tuple(execution.get("seeds", ())) == VALIDATION_SEEDS, "seed schedule changed")
    _require(execution.get("max_fes") == EXACT_MAX_FES, "exp026 requires exact 3M FE")
    _require(execution.get("jobs") == 20, "exp026 freezes jobs=20")
    for field, expected in {
        "budget_accounting": "strict",
        "search_state_backend": "phase_i_mmes",
        "cmaes_restart": True,
        "mmes_restart": True,
        "skip_plots": True,
    }.items():
        _require(execution.get(field) == expected, f"execution {field} must be {expected!r}")
    contract = execution.get("runner_contract")
    _require(isinstance(contract, dict), "runner_contract missing")
    for field, expected in {
        "arac_action": V37_ACTION,
        "enable_relation_dispatch": True,
        "relation_policy": "persistent_phase2",
        "runtime_probe_repair_mode": "hard_repair",
        "evidence_overlay_mode": "paired_owner",
        "group_optimizer_mode": "full_cmaes",
    }.items():
        _require(contract.get(field) == expected, f"runner_contract {field} must be {expected!r}")
    case_actions = execution.get("case_actions")
    _require(isinstance(case_actions, dict), "case_actions missing")
    _require(set(case_actions) == set(SUPPORTED_CASES), "case_actions must cover only exp026 cases")
    for case in SUPPORTED_CASES:
        _require(case_actions.get(case) == _expected_action(case), f"unexpected action for {case}")
    analysis = payload.get("analysis")
    _require(isinstance(analysis, dict), "analysis config missing")
    _require(analysis.get("primary_metric") == "exact_3000000_fe_best_so_far_error", "primary metric changed")
    _require(tuple(analysis.get("case_summary", ())) == ("mean", "median", "sample_std", "bootstrap_mean_95_ci"), "case summary changed")
    _require(analysis.get("bootstrap_method") == "within_case_seed_bootstrap", "bootstrap method changed")
    _require(analysis.get("bootstrap_replicates") == 2000, "bootstrap count changed")
    _require(analysis.get("bootstrap_seed") == 2026071901, "bootstrap seed changed")
    _require(analysis.get("paper_comparison") == "descriptive_ratio_only", "paper comparison must stay descriptive")
    paper = payload.get("paper_reference")
    _require(isinstance(paper, dict), "paper reference missing")
    _require(paper.get("max_fes") == EXACT_MAX_FES, "paper reference FE mismatch")
    paper_cases = paper.get("cases")
    _require(isinstance(paper_cases, dict) and set(paper_cases) == set(SUPPORTED_CASES), "paper cases mismatch")
    return payload


def build_run_matrix(config: Mapping[str, object], output_root: Path) -> list[RunSpec]:
    execution = config["execution"]
    assert isinstance(execution, dict)
    case_actions = execution["case_actions"]
    assert isinstance(case_actions, dict)
    specs = [
        RunSpec(str(config["experiment_id"]), case, seed, str(case_actions[case]), output_root)
        for case in SUPPORTED_CASES
        for seed in VALIDATION_SEEDS
    ]
    _require(len(specs) == 50, "exp026 must contain exactly 50 trajectories")
    _require(len({(item.case, item.seed) for item in specs}) == len(specs), "trajectory keys must be unique")
    return specs


def build_command(spec: RunSpec, config: Mapping[str, object], python_executable: str) -> tuple[str, ...]:
    execution = config["execution"]
    assert isinstance(execution, dict)
    contract = execution["runner_contract"]
    assert isinstance(contract, dict)
    function_name, function_id = CASE_TO_FUNCTION[spec.case]
    data_root = Path(str(execution.get("aob_data_root", DEFAULT_AOB_DATA_ROOT)))
    if not data_root.is_absolute():
        data_root = REPOSITORY_ROOT / data_root
    command = [
        python_executable, str(RUNNER_PATH), "--functions", function_name, "--ids", str(function_id),
        "--output-root", str(spec.run_directory), "--aob-data-root", str(data_root.resolve()),
        "--timestamp", spec.trajectory_id, "--seed", str(spec.seed), "--max-fes", str(EXACT_MAX_FES),
        "--arac-action", str(contract["arac_action"]), "--budget-accounting", str(execution["budget_accounting"]),
        "--search-state-backend", str(execution["search_state_backend"]), "--relation-policy", str(contract["relation_policy"]),
        "--persistent-phase2-action", spec.action, "--evidence-overlay-mode", str(contract["evidence_overlay_mode"]),
        "--runtime-probe-repair-mode", str(contract["runtime_probe_repair_mode"]), "--group-optimizer-mode", str(contract["group_optimizer_mode"]),
        "--enable-relation-dispatch",
    ]
    if execution.get("skip_plots") is True:
        command.append("--skip-plots")
    return tuple(command)


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing at exact path: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def read_trajectory_artifacts(spec: RunSpec) -> dict[str, object]:
    summary_path = spec.result_directory / "run_summary.json"
    action_path = spec.result_directory / "persistent_phase2_action.json"
    summary = _read_json(summary_path, "runner summary")
    for field, expected in {
        "protocol_version": RUN_SUMMARY_PROTOCOL_VERSION,
        "problem_id": spec.case,
        "seed": spec.seed,
        "configured_max_fes": EXACT_MAX_FES,
        "fitness_evaluations": EXACT_MAX_FES,
    }.items():
        _require(summary.get(field) == expected, f"runner summary {field} mismatch: {summary_path}")
    error = _final_error(summary.get("final_error"), str(summary_path))

    artifact = _read_json(action_path, "persistent Phase2 action artifact")
    for field, expected in {
        "schema_version": PERSISTENT_ACTION_ARTIFACT_SCHEMA,
        "problem_id": spec.case,
        "run_seed": spec.seed,
        "configured_max_fes": EXACT_MAX_FES,
        "terminal_fe": EXACT_MAX_FES,
        "selected_action": spec.action,
        "selection_count": 1,
        "runtime_authorized": True,
        "runtime_consumed": True,
    }.items():
        _require(artifact.get(field) == expected, f"persistent artifact {field} mismatch: {action_path}")
    action_hash = artifact.get("action_hash")
    _require(_is_sha256(action_hash), f"persistent artifact action_hash invalid: {action_path}")
    lifecycle = artifact.get("lifecycle")
    _require(isinstance(lifecycle, dict), f"persistent artifact lifecycle missing: {action_path}")
    _require(lifecycle.get("action_hash") == action_hash, f"persistent artifact lifecycle hash mismatch: {action_path}")
    _require(lifecycle.get("status") == "completed", f"persistent artifact action not completed: {action_path}")
    _require(_as_positive_int(lifecycle.get("consumed_fes"), "lifecycle consumed_fes") <= EXACT_MAX_FES, f"persistent artifact consumed_fes exceeds FE: {action_path}")
    return {
        "trajectory_id": spec.trajectory_id,
        "case": spec.case,
        "seed": spec.seed,
        "action": spec.action,
        "final_error": error,
        "fitness_evaluations": EXACT_MAX_FES,
        "summary_path": str(summary_path),
        "action_artifact_path": str(action_path),
        "action_hash": action_hash,
        "action_consumed_fes": lifecycle["consumed_fes"],
    }


def _validate_existing_trajectory(
    spec: RunSpec,
    *,
    execution_source: str,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        result = read_trajectory_artifacts(spec)
    except (OSError, TypeError, ValueError) as error:
        return {
            "trajectory_id": spec.trajectory_id,
            "case": spec.case,
            "seed": spec.seed,
            "action": spec.action,
            "ok": False,
            "status": "artifact_gate_failed",
            "execution_source": execution_source,
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(error),
        }
    result.update(
        {
            "ok": True,
            "status": "completed",
            "execution_source": execution_source,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return result


def _read_log_tail(path: Path, max_characters: int = 2000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_characters * 4))
        return handle.read().decode("utf-8", errors="replace")[-max_characters:]


def run_one(spec: RunSpec, config: Mapping[str, object], python_executable: str, *, run_subprocess: bool = True) -> dict[str, object]:
    if not run_subprocess:
        return _validate_existing_trajectory(spec, execution_source="offline_validation")

    spec.run_directory.mkdir(parents=True, exist_ok=True)
    command = build_command(spec, config, python_executable)
    runner_log_path = spec.run_directory / "runner.log"
    started = time.perf_counter()
    with runner_log_path.open("wb") as runner_log:
        completed = subprocess.run(
            command,
            cwd=VENDOR_ROOT,
            stdout=runner_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **SUBPROCESS_ENVIRONMENT},
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            "trajectory_id": spec.trajectory_id,
            "case": spec.case,
            "seed": spec.seed,
            "action": spec.action,
            "ok": False,
            "status": f"runner_failed_{completed.returncode}",
            "execution_source": "fresh_execution",
            "elapsed_seconds": elapsed,
            "runner_log_path": str(runner_log_path),
            "stderr_tail": _read_log_tail(runner_log_path),
        }
    result = _validate_existing_trajectory(spec, execution_source="fresh_execution")
    result["elapsed_seconds"] = elapsed
    result["runner_log_path"] = str(runner_log_path)
    return result


def _run_one_resumable(
    spec: RunSpec,
    config: Mapping[str, object],
    python_executable: str,
) -> dict[str, object]:
    existing = _validate_existing_trajectory(
        spec,
        execution_source="reused_valid_artifact",
    )
    if existing["ok"] is True:
        return existing

    result = run_one(spec, config, python_executable, run_subprocess=True)
    result["execution_source"] = "rerun_after_artifact_gate_failure"
    result["resume_gate_error"] = existing["error"]
    return result


def _print_trajectory_progress(result: Mapping[str, object]) -> None:
    print(
        f"[{result['case']}/seed{result['seed']}] {result['status']} "
        f"source={result['execution_source']}",
        flush=True,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "quantile requires values")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_mean_ci(values: Sequence[float], *, replicates: int, seed: int) -> tuple[float, float]:
    _require(len(values) == len(VALIDATION_SEEDS), "each case requires exactly five seeds")
    rng = random.Random(seed)
    means = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(replicates)]
    return _quantile(means, 0.025), _quantile(means, 0.975)


def build_case_summaries(results: Sequence[Mapping[str, object]], config: Mapping[str, object]) -> list[dict[str, object]]:
    _require(len(results) == 50, "summary requires exactly 50 trajectories")
    expected_keys = {(case, seed) for case in SUPPORTED_CASES for seed in VALIDATION_SEEDS}
    by_key = {(str(row.get("case")), int(row.get("seed", -1))): row for row in results}
    _require(len(by_key) == len(results) and set(by_key) == expected_keys, "summary trajectory set is incomplete or duplicated")
    _require(all(row.get("ok") is True and row.get("status") == "completed" for row in results), "only completed trajectories may be summarized")
    analysis = config["analysis"]
    paper = config["paper_reference"]
    assert isinstance(analysis, dict) and isinstance(paper, dict)
    paper_cases = paper["cases"]
    assert isinstance(paper_cases, dict)
    summaries: list[dict[str, object]] = []
    for index, case in enumerate(SUPPORTED_CASES):
        rows = [by_key[(case, seed)] for seed in VALIDATION_SEEDS]
        values = [_final_error(row.get("final_error"), f"{case}/seed{row.get('seed')}") for row in rows]
        mean = statistics.fmean(values)
        ci_low, ci_high = _bootstrap_mean_ci(values, replicates=int(analysis["bootstrap_replicates"]), seed=int(analysis["bootstrap_seed"]) + index)
        reference = paper_cases[case]
        assert isinstance(reference, dict)
        bold_mean = _final_error(reference.get("reported_bold_mean"), f"paper {case} bold")
        numeric_mean = _final_error(reference.get("numeric_best_mean"), f"paper {case} numeric")
        summaries.append({
            "case": case, "action": _expected_action(case), "seed_count": len(values), "mean_error": mean,
            "median_error": statistics.median(values), "sample_std_error": statistics.stdev(values),
            "bootstrap_mean_95_ci": [ci_low, ci_high],
            "paper_reported_bold_solver": reference["reported_bold_solver"], "paper_reported_bold_mean": bold_mean,
            "observed_to_paper_bold_mean_ratio": mean / bold_mean,
            "paper_numeric_best_solver": reference["numeric_best_solver"], "paper_numeric_best_mean": numeric_mean,
            "observed_to_paper_numeric_best_mean_ratio": mean / numeric_mean,
            "comparison_note": "Five-seed descriptive ratio only; no paper baseline was rerun and no inferential comparison is valid.",
        })
    return summaries


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python_executable: str = sys.executable,
    jobs: int | None = None,
    reuse_existing: bool = False,
    resume: bool = False,
    progress_callback: Callable[[Mapping[str, object]], None] = _print_trajectory_progress,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    _require(not (reuse_existing and resume), "reuse_existing and resume are mutually exclusive")
    config = load_config(config_path)
    specs = build_run_matrix(config, output_root)
    workers = int(config["execution"]["jobs"]) if jobs is None else jobs  # type: ignore[index]
    _require(workers > 0, "jobs must be positive")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if resume:
            futures = [pool.submit(_run_one_resumable, spec, config, python_executable) for spec in specs]
        else:
            futures = [
                pool.submit(
                    run_one,
                    spec,
                    config,
                    python_executable,
                    run_subprocess=not reuse_existing,
                )
                for spec in specs
            ]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress_callback(result)
    results.sort(key=lambda row: (str(row["case"]), int(row["seed"])))
    summaries: list[dict[str, object]] = []
    if all(row.get("ok") is True for row in results):
        summaries = build_case_summaries(results, config)
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "trajectory_count": len(results),
        "completed_trajectory_count": sum(row.get("ok") is True for row in results),
        "integrity_gate_passed": len(results) == 50 and len(summaries) == len(SUPPORTED_CASES),
        "exact_max_fes": EXACT_MAX_FES, "native_rerun": False, "paper_baseline_rerun": False,
        "five_seed_descriptive_only": True, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "execution_mode": "offline_validation" if reuse_existing else "resume" if resume else "fresh",
        "worker_count": workers,
        "reused_trajectory_count": sum(row.get("execution_source") == "reused_valid_artifact" for row in results),
        "executed_trajectory_count": sum(row.get("execution_source") in {"fresh_execution", "rerun_after_artifact_gate_failure"} for row in results),
    }
    _write_json(output_root / "run_summary.json", {**manifest, "results": results, "case_summaries": summaries})
    return results, summaries, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=None)
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--reuse-existing",
        action="store_true",
        help="validate all existing artifacts without launching runner subprocesses",
    )
    execution_mode.add_argument(
        "--resume",
        action="store_true",
        help="reuse strictly valid trajectories and rerun only missing or invalid ones",
    )
    args = parser.parse_args(argv)
    if args.jobs is not None and args.jobs <= 0:
        parser.error("--jobs must be positive")
    _results, summaries, manifest = run_experiment(
        config_path=args.config,
        output_root=args.output_root,
        python_executable=args.python_executable,
        jobs=args.jobs,
        reuse_existing=args.reuse_existing,
        resume=args.resume,
    )
    for summary in summaries:
        print(f"[{summary['case']}] n=5 mean={summary['mean_error']:.6e} median={summary['median_error']:.6e}", flush=True)
    print(f"Summary: {args.output_root / 'run_summary.json'}", flush=True)
    return 0 if manifest["integrity_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
