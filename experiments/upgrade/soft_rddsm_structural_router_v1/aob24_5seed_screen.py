"""Matched AOB 24 x 5-seed screen for the structural-router candidate.

This is an experiment runner, not a production entry point.  It pairs the
candidate with the frozen-selector reference seeds exactly:

* cases: A/E/R/S 1..6;
* run/action seeds: 20260745..20260749;
* Phase-I: 180,000 FE;
* terminal: 3,000,000 FE.

Each completed row is appended to ``results.jsonl`` so an interrupted screen
can be resumed without rerunning completed pairs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from arac.benchmarks.aob import AobBenchmark
from experiments.upgrade.soft_rddsm_structural_router_v1 import (
    run_and_execute_soft_rddsm_structural_route,
)


CASES = tuple(f"{family}{index}" for family in "AERS" for index in range(1, 7))
SEEDS = (20260745, 20260746, 20260747, 20260748, 20260749)
PHASE1_FES = 180_000
TOTAL_BUDGET_FES = 3_000_000
SCHEMA = "arac-soft-rddsm-structural-router-aob24-5seed-v1"
DEFAULT_OUTPUT = Path("artifacts/soft_rddsm_structural_router_aob24_5seed")


@dataclass(frozen=True)
class Task:
    case_id: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.case_id}|{self.seed}"


def _run_one(task: Task) -> dict[str, Any]:
    # Restrict BLAS/OpenMP fan-out inside each worker.  The benchmark is
    # already parallelised at the task level.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    problem = AobBenchmark().load(task.case_id)
    result = run_and_execute_soft_rddsm_structural_route(
        problem,
        run_seed=task.seed,
        action_seed=task.seed,
    )
    phase1 = result.structural_run.phase1
    evidence = phase1.overlap_evidence
    return {
        "schema_version": SCHEMA,
        "case_id": task.case_id,
        "run_seed": task.seed,
        "action_seed": task.seed,
        "action_name": result.action_result.action_name,
        "route": result.action_result.route,
        "final_error": float(result.action_result.final_error),
        "phase1_fes": int(phase1.checkpoint.phase1_fes),
        "terminal_fes": int(result.action_result.terminal_fes),
        "phase1_incumbent_error": float(phase1.checkpoint.incumbent_error),
        "evidence_complete": bool(evidence.complete),
        "evidence_ready": bool(result.structural_run.adaptation.ready),
        "shared_candidate_count": len(phase1.shared_candidates),
        "membership_shared_count": sum(
            len(owners) > 1 for owners in evidence.memberships
        ),
        "evidence_hash": phase1.overlap_evidence_hash,
        "source_checkpoint_hash": result.source_checkpoint_hash,
        "action_checkpoint_hash": result.action_checkpoint_hash,
        "action_result_hash": result.action_result.result_hash,
        "task_key": task.key,
    }


def _read_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("task_key", ""))
        if key:
            completed[key] = row
    return completed


def _write_manifest(root: Path, *, workers: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA,
        "cases": list(CASES),
        "seeds": list(SEEDS),
        "phase1_fes": PHASE1_FES,
        "total_budget_fes": TOTAL_BUDGET_FES,
        "workers": workers,
        "pair_count": len(CASES) * len(SEEDS),
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_screen(*, output: Path = DEFAULT_OUTPUT, workers: int = 8) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    _write_manifest(output, workers=workers)

    completed = _read_completed(results_path)
    tasks = [
        Task(case_id, seed)
        for case_id in CASES
        for seed in SEEDS
        if f"{case_id}|{seed}" not in completed
    ]
    print(
        json.dumps(
            {
                "output": str(output),
                "completed": len(completed),
                "remaining": len(tasks),
                "workers": workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool, results_path.open(
            "a", encoding="utf-8", buffering=1
        ) as handle:
            futures = {pool.submit(_run_one, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # keep failure visible and resumable
                    row = {
                        "schema_version": SCHEMA,
                        "case_id": task.case_id,
                        "run_seed": task.seed,
                        "action_seed": task.seed,
                        "task_key": task.key,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                completed[task.key] = row
                print(
                    json.dumps(
                        {
                            "task": task.key,
                            "status": row.get("status", "completed"),
                            "completed": len(completed),
                            "total": len(CASES) * len(SEEDS),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    summary = summarize(completed)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize(rows_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows_by_key.values())
    successful = [row for row in rows if row.get("status", "completed") == "completed"]
    by_case: list[dict[str, Any]] = []
    for case_id in CASES:
        case_rows = [row for row in successful if row.get("case_id") == case_id]
        errors = [float(row["final_error"]) for row in case_rows]
        by_case.append(
            {
                "case_id": case_id,
                "seed_count": len(case_rows),
                "mean_final_error": _mean(errors),
                "final_errors": errors,
                "actions": {
                    action: sum(row.get("action_name") == action for row in case_rows)
                    for action in ("aor", "smp", "ctp", "gcb")
                },
                "phase1_exact": all(
                    row.get("phase1_fes") == PHASE1_FES for row in case_rows
                ),
                "terminal_exact": all(
                    row.get("terminal_fes") == TOTAL_BUDGET_FES for row in case_rows
                ),
            }
        )
    all_errors = [float(row["final_error"]) for row in successful]
    return {
        "schema_version": SCHEMA,
        "case_count": len(CASES),
        "seed_count": len(SEEDS),
        "expected_pair_count": len(CASES) * len(SEEDS),
        "completed_pair_count": len(rows),
        "successful_pair_count": len(successful),
        "failed_pair_count": sum(row.get("status") == "failed" for row in rows),
        "phase1_fes": PHASE1_FES,
        "total_budget_fes": TOTAL_BUDGET_FES,
        "case_summaries": by_case,
        "overall_mean_final_error": _mean(all_errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    run_screen(output=args.output, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
