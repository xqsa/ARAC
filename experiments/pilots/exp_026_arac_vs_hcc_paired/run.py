"""Run exp026 as fixed-action validation against native HCC.

Arm A keeps native Eq.8 overlap handling and full CMA-ES. Arm B changes only
the group optimizer covariance update to the frozen diagonal action. Evidence,
selection, bandits, runtime probes, and relation dispatch remain disabled.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "hcc_smoke_runner.py"
VENDOR_ROOT = REPOSITORY_ROOT / "vendor" / "hcc"
DEFAULT_AOB_DATA_ROOT = VENDOR_ROOT / "AOB" / "AOBG" / "datafile"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "exp_026_arac_vs_hcc_paired"

PROTOCOL_VERSION = "paired-action-validation-v2"
RUN_SUMMARY_PROTOCOL_VERSION = "hcc-run-summary-v1"
SUPPORTED_CASES = ("E1", "E3", "A4", "R4", "S5")
VALIDATION_SEEDS = (117, 118, 119, 120, 121)
ERROR_EPSILON = 1e-300

SUBPROCESS_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

CASE_TO_FUNCTION: dict[str, tuple[str, int]] = {
    "E1": ("elliptic", 1),
    "E3": ("elliptic", 3),
    "A4": ("ackley", 4),
    "R4": ("rastrigin", 4),
    "S5": ("schwefel", 5),
}

_FIXED_ARM_FIELDS = {
    "action_surface": "group_optimizer_type",
    "arac_action": "native_eq8",
    "enable_relation_dispatch": False,
    "relation_policy": "controller_v31",
    "runtime_probe_repair_mode": "hard_repair",
    "evidence_overlay_mode": "off",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_arm(arm: Mapping[str, object], *, expected_mode: str) -> None:
    _require(isinstance(arm.get("label"), str) and bool(arm["label"]), "arm label missing")
    for field, expected in _FIXED_ARM_FIELDS.items():
        _require(arm.get(field) == expected, f"arm {field} must be {expected!r}")
    _require(
        arm.get("group_optimizer_mode") == expected_mode,
        f"arm group_optimizer_mode must be {expected_mode!r}",
    )


def load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "config must be a JSON object")
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "unsupported protocol")
    _require(payload.get("stage") == "action_validation", "exp026 is action validation")

    execution = payload.get("execution")
    _require(isinstance(execution, dict), "execution config missing")
    _require(tuple(execution.get("cases", ())) == SUPPORTED_CASES, "unsupported AOB cases")
    _require(tuple(execution.get("seeds", ())) == VALIDATION_SEEDS, "seed schedule changed")
    _require(int(execution.get("max_fes", 0)) >= 300_000, "exp026 requires at least 300k FE")
    _require(int(execution.get("jobs", 0)) > 0, "jobs must be positive")
    _require(execution.get("budget_accounting") == "strict", "strict FE accounting required")
    _require(execution.get("search_state_backend") == "phase_i_mmes", "backend changed")
    _require(execution.get("cmaes_restart") is True, "CMA-ES restart must remain enabled")
    _require(execution.get("mmes_restart") is True, "MMES restart must remain enabled")

    arm_a = execution.get("arm_a")
    arm_b = execution.get("arm_b")
    _require(isinstance(arm_a, dict) and isinstance(arm_b, dict), "paired arms missing")
    _validate_arm(arm_a, expected_mode="full_cmaes")
    _validate_arm(arm_b, expected_mode="diagonal_covariance")
    _require(arm_a["label"] != arm_b["label"], "arm labels must be unique")

    analysis = payload.get("analysis")
    _require(isinstance(analysis, dict), "analysis config missing")
    _require(int(analysis.get("bootstrap_replicates", 0)) == 2000, "bootstrap count changed")
    _require(int(analysis.get("bootstrap_seed", 0)) == 2026071901, "bootstrap seed changed")
    _require(float(analysis.get("material_positive_multiplier", 0.0)) == 1.01, "material threshold changed")
    _require(float(analysis.get("catastrophic_multiplier", 0.0)) == 1.20, "catastrophic threshold changed")
    return payload


def trajectory_id(config: Mapping[str, object], arm_label: str, case: str, seed: int) -> str:
    return f"{config['experiment_id']}-{arm_label}-{case.lower()}-seed{seed}"


def run_directory(output_root: Path, arm_label: str, case: str, seed: int) -> Path:
    return output_root / "runs" / arm_label / case / f"seed_{seed}"


def expected_summary_path(
    output_root: Path,
    config: Mapping[str, object],
    arm_label: str,
    case: str,
    seed: int,
) -> Path:
    function_name, _ = CASE_TO_FUNCTION[case]
    return (
        run_directory(output_root, arm_label, case, seed)
        / trajectory_id(config, arm_label, case, seed)
        / function_name
        / "run_summary.json"
    )


def build_command(
    arm_label: str,
    arm_cfg: Mapping[str, object],
    case: str,
    seed: int,
    config: Mapping[str, object],
    output_root: Path,
    python_executable: str,
) -> tuple[str, ...]:
    if case not in CASE_TO_FUNCTION:
        raise ValueError(f"unsupported AOB case: {case!r}")
    execution = config["execution"]
    assert isinstance(execution, dict)
    function_name, function_id = CASE_TO_FUNCTION[case]
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
        str(run_directory(output_root, arm_label, case, seed)),
        "--aob-data-root",
        str(data_root.resolve()),
        "--timestamp",
        trajectory_id(config, arm_label, case, seed),
        "--seed",
        str(seed),
        "--max-fes",
        str(execution["max_fes"]),
        "--arac-action",
        str(arm_cfg["arac_action"]),
        "--budget-accounting",
        str(execution["budget_accounting"]),
        "--search-state-backend",
        str(execution["search_state_backend"]),
        "--relation-policy",
        str(arm_cfg["relation_policy"]),
        "--runtime-probe-repair-mode",
        str(arm_cfg["runtime_probe_repair_mode"]),
        "--evidence-overlay-mode",
        str(arm_cfg["evidence_overlay_mode"]),
        "--group-optimizer-mode",
        str(arm_cfg["group_optimizer_mode"]),
    ]
    if arm_cfg.get("enable_relation_dispatch"):
        command.append("--enable-relation-dispatch")
    if execution.get("skip_plots"):
        command.append("--skip-plots")
    return tuple(command)


def _validated_error(value: object, *, source: str) -> float:
    try:
        error = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} final_error is not numeric") from exc
    if not math.isfinite(error) or error < 0.0:
        raise ValueError(f"{source} final_error must be finite and non-negative")
    return error


def read_run_summary(
    path: Path,
    *,
    expected_case: str,
    expected_seed: int,
    expected_max_fes: int,
    expected_optimizer_mode: str,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"runner summary missing at exact path: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"runner summary is invalid JSON: {path}") from exc
    _require(isinstance(payload, dict), f"runner summary must be an object: {path}")
    expected = {
        "protocol_version": RUN_SUMMARY_PROTOCOL_VERSION,
        "problem_id": expected_case,
        "seed": expected_seed,
        "configured_max_fes": expected_max_fes,
        "group_optimizer_mode": expected_optimizer_mode,
    }
    for field, value in expected.items():
        _require(payload.get(field) == value, f"runner summary {field} mismatch: {path}")
    fitness_evaluations = payload.get("fitness_evaluations")
    _require(
        isinstance(fitness_evaluations, int) and fitness_evaluations > 0,
        f"runner summary fitness_evaluations invalid: {path}",
    )
    payload["final_error"] = _validated_error(payload.get("final_error"), source=str(path))
    return payload


def run_one(
    arm_label: str,
    arm_cfg: Mapping[str, object],
    case: str,
    seed: int,
    config: Mapping[str, object],
    output_root: Path,
    python_executable: str,
) -> dict[str, object]:
    command = build_command(
        arm_label, arm_cfg, case, seed, config, output_root, python_executable
    )
    output_directory = run_directory(output_root, arm_label, case, seed)
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=VENDOR_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **SUBPROCESS_ENVIRONMENT},
    )
    elapsed = time.perf_counter() - started
    ok = completed.returncode == 0
    print(f"[{arm_label}/{case}/seed{seed}] {'OK' if ok else 'FAIL'} {elapsed:.0f}s", flush=True)
    if not ok:
        print(f"  stderr tail: {completed.stderr[-500:]}", flush=True)
        return {
            "arm": arm_label,
            "case": case,
            "seed": seed,
            "ok": False,
            "elapsed": elapsed,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-500:],
        }

    execution = config["execution"]
    assert isinstance(execution, dict)
    summary_path = expected_summary_path(output_root, config, arm_label, case, seed)
    summary = read_run_summary(
        summary_path,
        expected_case=case,
        expected_seed=seed,
        expected_max_fes=int(execution["max_fes"]),
        expected_optimizer_mode=str(arm_cfg["group_optimizer_mode"]),
    )
    return {
        "arm": arm_label,
        "case": case,
        "seed": seed,
        "ok": True,
        "elapsed": elapsed,
        "returncode": completed.returncode,
        "summary_path": str(summary_path),
        "fitness_evaluations": summary["fitness_evaluations"],
        "final_error": summary["final_error"],
    }


def paired_delta(native_error: float, action_error: float) -> float:
    native = _validated_error(native_error, source="native")
    action = _validated_error(action_error, source="action")
    return math.log((native + ERROR_EPSILON) / (action + ERROR_EPSILON))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_case_macro(
    pairs_by_case: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    replicates: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    rng = random.Random(seed)
    macro_means: list[float] = []
    material_rates: list[float] = []
    for _ in range(replicates):
        case_means: list[float] = []
        sampled_material: list[float] = []
        for case in SUPPORTED_CASES:
            clusters = pairs_by_case[case]
            sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
            case_means.append(statistics.fmean(float(row["delta"]) for row in sampled))
            sampled_material.extend(float(bool(row["material_positive"])) for row in sampled)
        macro_means.append(statistics.fmean(case_means))
        material_rates.append(statistics.fmean(sampled_material))
    return macro_means, material_rates


def build_paired_analysis(
    results: Sequence[Mapping[str, object]],
    *,
    native_label: str,
    action_label: str,
    expected_cases: Sequence[str],
    expected_seeds: Sequence[int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    material_positive_multiplier: float,
    catastrophic_multiplier: float,
) -> dict[str, object]:
    if tuple(expected_cases) != SUPPORTED_CASES:
        raise ValueError("analysis case set differs from the fixed validation cohort")
    expected_keys = {(case, int(seed)) for case in expected_cases for seed in expected_seeds}
    indexes: dict[str, dict[tuple[str, int], Mapping[str, object]]] = {
        native_label: {},
        action_label: {},
    }
    for result in results:
        arm = str(result.get("arm"))
        if arm not in indexes:
            raise ValueError(f"unexpected arm in result: {arm!r}")
        if result.get("ok") is not True:
            raise ValueError("failed run cannot enter paired analysis")
        key = (str(result.get("case")), int(result.get("seed", -1)))
        if key in indexes[arm]:
            raise ValueError(f"duplicate paired result: {arm}/{key}")
        indexes[arm][key] = result
    for arm, index in indexes.items():
        if set(index) != expected_keys:
            missing = sorted(expected_keys - set(index))
            extra = sorted(set(index) - expected_keys)
            raise ValueError(f"incomplete {arm} result set; missing={missing}, extra={extra}")

    material_delta = math.log(material_positive_multiplier)
    catastrophic_delta = -math.log(catastrophic_multiplier)
    pairs: list[dict[str, object]] = []
    pairs_by_case: dict[str, list[dict[str, object]]] = {
        case: [] for case in expected_cases
    }
    for case in expected_cases:
        function_name, function_id = CASE_TO_FUNCTION[case]
        for seed in expected_seeds:
            key = (case, int(seed))
            native_error = _validated_error(
                indexes[native_label][key].get("final_error"), source=f"{native_label}/{key}"
            )
            action_error = _validated_error(
                indexes[action_label][key].get("final_error"), source=f"{action_label}/{key}"
            )
            delta = paired_delta(native_error, action_error)
            pair = {
                "case": case,
                "seed": int(seed),
                "function_name": function_name,
                "function_id": function_id,
                "native_final_error": native_error,
                "action_final_error": action_error,
                "delta": delta,
                "material_positive": delta > material_delta,
                "catastrophic": (
                    action_error + ERROR_EPSILON
                    >= catastrophic_multiplier * (native_error + ERROR_EPSILON)
                ),
            }
            pairs.append(pair)
            pairs_by_case[case].append(pair)

    case_summaries = {
        case: {
            "pair_count": len(case_pairs),
            "mean_delta": statistics.fmean(float(row["delta"]) for row in case_pairs),
            "material_positive_count": sum(bool(row["material_positive"]) for row in case_pairs),
            "catastrophic_count": sum(bool(row["catastrophic"]) for row in case_pairs),
        }
        for case, case_pairs in pairs_by_case.items()
    }
    macro_mean = statistics.fmean(
        float(summary["mean_delta"]) for summary in case_summaries.values()
    )
    bootstrap_macro, bootstrap_material = _bootstrap_case_macro(
        pairs_by_case,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    material_count = sum(bool(row["material_positive"]) for row in pairs)
    catastrophic_pairs = [
        f"{row['case']}:seed{row['seed']}" for row in pairs if row["catastrophic"]
    ]
    macro_lcb = _quantile(bootstrap_macro, 0.025)
    if catastrophic_pairs:
        decision = "reject_action_catastrophic_loss"
    elif macro_mean <= 0.0:
        decision = "redesign_action"
    elif macro_lcb <= 0.0:
        decision = "collect_more_action_contexts"
    else:
        decision = "candidate_for_broader_action_validation"

    return {
        "protocol_version": PROTOCOL_VERSION,
        "delta_definition": "log((native + 1e-300) / (action + 1e-300))",
        "pair_count": len(pairs),
        "case_count": len(case_summaries),
        "case_seed_cluster_count": len(pairs),
        "paired_mean_delta": statistics.fmean(float(row["delta"]) for row in pairs),
        "case_macro_mean_delta": macro_mean,
        "case_macro_mean_delta_lcb": macro_lcb,
        "case_macro_mean_delta_ucb": _quantile(bootstrap_macro, 0.975),
        "material_positive_delta": material_delta,
        "material_positive_count": material_count,
        "material_positive_rate": material_count / len(pairs),
        "material_positive_rate_lcb": _quantile(bootstrap_material, 0.025),
        "material_positive_rate_ucb": _quantile(bootstrap_material, 0.975),
        "catastrophic_delta": catastrophic_delta,
        "catastrophic_count": len(catastrophic_pairs),
        "catastrophic_rate": len(catastrophic_pairs) / len(pairs),
        "catastrophic_pairs": catastrophic_pairs,
        "bootstrap": {
            "method": "within_case_case_seed_cluster_bootstrap",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "decision": decision,
        "case_summaries": case_summaries,
        "pairs": pairs,
    }


def _write_experiment_summary(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    execution = config["execution"]
    analysis_config = config["analysis"]
    assert isinstance(execution, dict) and isinstance(analysis_config, dict)
    jobs = args.jobs or int(execution["jobs"])
    if jobs <= 0:
        parser.error("--jobs must be positive")
    arm_a = execution["arm_a"]
    arm_b = execution["arm_b"]
    assert isinstance(arm_a, dict) and isinstance(arm_b, dict)
    arm_configs = ((str(arm_a["label"]), arm_a), (str(arm_b["label"]), arm_b))
    tasks = [
        (label, arm, case, int(seed))
        for label, arm in arm_configs
        for case in execution["cases"]
        for seed in execution["seeds"]
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"exp026: {len(tasks)} fixed-action runs, {execution['max_fes']:,} FE each, "
        f"{jobs} parallel",
        flush=True,
    )
    print(f"Output: {args.output_root}", flush=True)

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                run_one,
                label,
                arm,
                case,
                seed,
                config,
                args.output_root,
                args.python_executable,
            )
            for label, arm, case, seed in tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda row: (str(row["arm"]), str(row["case"]), int(row["seed"])))
    failed_runs = sum(result["ok"] is not True for result in results)
    paired_analysis = None
    if failed_runs == 0:
        paired_analysis = build_paired_analysis(
            results,
            native_label=str(arm_a["label"]),
            action_label=str(arm_b["label"]),
            expected_cases=tuple(str(case) for case in execution["cases"]),
            expected_seeds=tuple(int(seed) for seed in execution["seeds"]),
            bootstrap_replicates=int(analysis_config["bootstrap_replicates"]),
            bootstrap_seed=int(analysis_config["bootstrap_seed"]),
            material_positive_multiplier=float(analysis_config["material_positive_multiplier"]),
            catastrophic_multiplier=float(analysis_config["catastrophic_multiplier"]),
        )

    experiment_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": config["experiment_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(results),
        "ok_runs": len(results) - failed_runs,
        "failed_runs": failed_runs,
        "integrity_gate_passed": failed_runs == 0 and paired_analysis is not None,
        "paired_analysis": paired_analysis,
        "results": results,
    }
    summary_path = args.output_root / "run_summary.json"
    _write_experiment_summary(summary_path, experiment_summary)
    print(f"Summary: {summary_path}", flush=True)
    if paired_analysis is not None:
        print(
            "Paired case-macro Delta: "
            f"{paired_analysis['case_macro_mean_delta']:.6g} "
            f"[95% {paired_analysis['case_macro_mean_delta_lcb']:.6g}, "
            f"{paired_analysis['case_macro_mean_delta_ucb']:.6g}]",
            flush=True,
        )
        print(f"Decision: {paired_analysis['decision']}", flush=True)
    return 0 if failed_runs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
