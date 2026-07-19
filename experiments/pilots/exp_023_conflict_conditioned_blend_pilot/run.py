"""Run the conflict-conditioned context blend pilot (exp_023).

Design (change from exp_022):
- Same runtime_probe dispatch infrastructure as exp_022.
- When probe says shadow_action == "repair", apply conflict_conditioned_context_blend
  instead of repair_shared_variable_binding (winner-take-all).
- Blend formula:
    u_excess = max(0, (utility - threshold) / threshold)
    sharpening = tanh(u_excess)   # 0 at threshold, → 1 as utility >> threshold
    w_current = base_w + sharpening * (1 - base_w)   if current is winner
    w_current = base_w * (1 - sharpening)             if previous is winner
  Degrades to HCC Eq.8 weighted blend at utility == threshold.
  Approaches winner-take-all as utility >> threshold.
- Probe utility is threaded through runtime_relation_action_map as tuple[str, float].

Cases: E3 (conflicting), S5 (conflicting), R4 (conforming control).
Seeds: same 5 seeds as exp_022 for direct comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "hcc_smoke_runner.py"
VENDOR_ROOT = REPOSITORY_ROOT / "vendor" / "hcc"
DEFAULT_AOB_DATA_ROOT = VENDOR_ROOT / "AOB" / "AOBG" / "datafile"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "exp_023_conflict_conditioned_blend_pilot"

SUBPROCESS_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

CASE_TO_FUNCTION = {
    "E3": ("elliptic", 3),
    "S5": ("schwefel", 5),
    "R4": ("rastrigin", 4),
}


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload.get("execution", {}).get("relation_policy") == "runtime_probe", \
        "exp_023 requires relation_policy=runtime_probe"
    assert payload.get("execution", {}).get("evidence_overlay_mode") != "off", \
        "exp_023 requires evidence overlay for probe barrier"
    return payload


def build_command(
    case: str,
    seed: int,
    config: dict,
    output_root: Path,
    python_executable: str,
) -> tuple[str, ...]:
    execution = config["execution"]
    function_name, function_id = CASE_TO_FUNCTION[case]
    data_root = Path(str(execution.get("aob_data_root", DEFAULT_AOB_DATA_ROOT)))
    if not data_root.is_absolute():
        data_root = REPOSITORY_ROOT / data_root
    trajectory_id = f"{config['experiment_id']}-{case.lower()}-seed{seed}"
    cmd = [
        python_executable,
        str(RUNNER_PATH),
        "--functions", function_name,
        "--ids", str(function_id),
        "--output-root", str(output_root / "runs" / case / f"seed_{seed}"),
        "--aob-data-root", str(data_root.resolve()),
        "--timestamp", trajectory_id,
        "--seed", str(seed),
        "--max-fes", str(execution["max_fes"]),
        "--arac-action", execution["arac_action"],
        "--budget-accounting", execution["budget_accounting"],
        "--search-state-backend", execution["search_state_backend"],
        "--relation-policy", execution["relation_policy"],
        "--runtime-probe-repair-mode", execution["runtime_probe_repair_mode"],
        "--evidence-overlay-mode", execution["evidence_overlay_mode"],
        "--enable-relation-dispatch",
    ]
    if execution.get("skip_plots"):
        cmd.append("--skip-plots")
    return tuple(cmd)


def run_one(case: str, seed: int, config: dict, output_root: Path, python_executable: str):
    cmd = build_command(case, seed, config, output_root, python_executable)
    env = {**os.environ, **SUBPROCESS_ENVIRONMENT}
    out_dir = output_root / "runs" / case / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=VENDOR_ROOT, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0
    print(f"[{case}/seed{seed}] {'OK' if ok else 'FAIL'} {elapsed:.0f}s", flush=True)
    if not ok:
        print(f"  stderr tail: {result.stderr[-500:]}", flush=True)
    return {"case": case, "seed": seed, "ok": ok, "elapsed": elapsed,
            "returncode": result.returncode, "stderr_tail": result.stderr[-300:]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    execution = config["execution"]
    cases = execution["cases"]
    seeds = execution["seeds"]
    jobs = args.jobs or execution.get("jobs", 3)

    tasks = [(case, seed) for case in cases for seed in seeds]
    print(f"exp_023: {len(tasks)} runs × {execution['max_fes']:,} FE each, {jobs} parallel")
    print(f"Output: {args.output_root}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_one, case, seed, config, args.output_root, args.python_executable): (case, seed)
            for case, seed in tasks
        }
        for fut in futures:
            results.append(fut.result())

    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count
    print(f"\nCompleted: {ok_count} OK, {fail_count} failed")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "run_summary.json"
    summary = {
        "experiment_id": config["experiment_id"],
        "total_runs": len(results),
        "ok_runs": ok_count,
        "failed_runs": fail_count,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
