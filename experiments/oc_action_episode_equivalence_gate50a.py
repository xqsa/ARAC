"""Gate 50a: realistic-scale episode segmentation equivalence for the four actions.

Absorption prerequisite (Gate 50): a one-shot v2 execution and the same
run segmented at irregular boundaries with snapshot/restore every segment
must agree bit-exactly on terminal count, best error and incumbent.  The
existing unit tests prove the mechanism at 34 FE; this gate proves it at
a scale where the actions' internal phases actually engage (SMP rescue
>= 100k FE, GCB native windows, CTP coverage/polish transitions).
"""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path

import numpy as np

from arac.actions.registry import ActionRegistry
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger

DIMENSION = 1000
BLOCK_SIZE = 50
TOTAL_FES = 600_000
PHASE1_FES = 100_000
ACTION_SEED = 20260850
CHUNK_PATTERN = (37, 9_991, 111, 50_003, 7, 123_457, 999, 13, 61_003, 2_003)
OUTPUT_SCHEMA = "arac-oc-gate50a-equivalence-v1"
OUTPUT_ROOT = Path("artifacts/oc_action_episode_equivalence_gate50a")
ACTIONS = ("ctp", "smp", "gcb", "aor")


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        for start in range(0, DIMENSION, BLOCK_SIZE):
            block = batch[:, start : start + BLOCK_SIZE]
            result += 0.5 * np.sum(block**2, axis=1) ** 2 / BLOCK_SIZE
        for start in range(BLOCK_SIZE, DIMENSION, BLOCK_SIZE):
            left = batch[:, start - 1]
            right = batch[:, start]
            result += 0.25 * left**2 * right**2
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _checkpoint(incumbent_error: float) -> PhaseCheckpoint:
    blocks = tuple(
        tuple(range(start, start + BLOCK_SIZE)) for start in range(0, DIMENSION, BLOCK_SIZE)
    )
    relations = tuple(
        RelationEvidence(left_block=index, right_block=index + 1, strength=0.5, disagreement=0.1)
        for index in range(len(blocks) - 1)
    )
    return PhaseCheckpoint(
        protocol="gate50a-equivalence-v1",
        run_seed=7,
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=(0.1,) * DIMENSION,
        incumbent_error=incumbent_error,
        feature_names=("log10_center_error", "line_high_frequency_fraction_median"),
        feature_values=(math.log10(max(incumbent_error, 1.0)), 0.4),
        blocks=blocks,
        relations=relations,
    )


def _context(action_name: str, problem, checkpoint) -> ActionContext:
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    return ActionContext(action_name, checkpoint, problem, ledger, action_seed=ACTION_SEED)


def run_one_shot(action_name: str) -> dict[str, object]:
    problem = _problem()
    incumbent_error = float(problem.objective(np.asarray([0.1] * DIMENSION)))
    checkpoint = _checkpoint(incumbent_error)
    context = _context(action_name, problem, checkpoint)
    result = ActionRegistry().execute_v2(context)
    return {
        "action": action_name,
        "mode": "one_shot",
        "terminal_fes": context.ledger.count,
        "final_error": context.ledger.best_error,
        "incumbent": [float(v) for v in context.ledger.best_x],
        "consumed_fes": result.consumed_fes,
    }


def run_segmented(action_name: str) -> dict[str, object]:
    problem = _problem()
    incumbent_error = float(problem.objective(np.asarray([0.1] * DIMENSION)))
    checkpoint = _checkpoint(incumbent_error)
    context = _context(action_name, problem, checkpoint)
    registry = ActionRegistry()
    state = registry.initialize(context)
    segments = 0
    restores = 0
    pattern_index = 0
    while not state.complete:
        budget = state.total_fes - state.context.ledger.count
        chunk = min(budget, CHUNK_PATTERN[pattern_index % len(CHUNK_PATTERN)])
        pattern_index += 1
        state.step(chunk)
        segments += 1
        if not state.complete:
            snapshot = state.snapshot()
            restored_ledger = EvaluationLedger.from_phase2_snapshot(problem, snapshot)
            restored_context = ActionContext(
                action_name,
                checkpoint,
                problem,
                restored_ledger,
                action_seed=ACTION_SEED,
            )
            state = registry.resume(restored_context, snapshot)
            restores += 1
    result = state.result()
    return {
        "action": action_name,
        "mode": "segmented",
        "terminal_fes": state.context.ledger.count,
        "final_error": state.context.ledger.best_error,
        "incumbent": [float(v) for v in state.context.ledger.best_x],
        "consumed_fes": result.consumed_fes,
        "segments": segments,
        "snapshot_restores": restores,
    }


def _cached(directory: Path, name: str, runner, action: str) -> dict[str, object]:
    path = directory / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["result"]
    row = runner(action)
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": OUTPUT_SCHEMA, "result": row}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cells_dir = OUTPUT_ROOT / "cells"
    jobs = []
    for action in ACTIONS:
        jobs.append((f"{action}_one_shot", run_one_shot, action))
        jobs.append((f"{action}_segmented", run_segmented, action))
    rows: dict[tuple[str, str], dict[str, object]] = {}
    pending = [
        (name, runner, action)
        for name, runner, action in jobs
        if not (cells_dir / f"{name}.json").exists()
    ]
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(runner, action): (name, action) for name, runner, action in pending
            }
            for future in as_completed(futures):
                name, action = futures[future]
                row = future.result()
                cells_dir.mkdir(parents=True, exist_ok=True)
                (cells_dir / f"{name}.json").write_text(
                    json.dumps(
                        {"schema_version": OUTPUT_SCHEMA, "result": row}, indent=2, sort_keys=True
                    ),
                    encoding="utf-8",
                )
                print(f"completed {name}: final={row['final_error']:.6g}", flush=True)
    for name, _runner, action in jobs:
        payload = json.loads((cells_dir / f"{name}.json").read_text(encoding="utf-8"))["result"]
        mode = "one_shot" if name.endswith("one_shot") else "segmented"
        rows[(action, mode)] = payload

    checks: dict[str, bool] = {}
    per_action: list[dict[str, object]] = []
    for action in ACTIONS:
        one = rows[(action, "one_shot")]
        seg = rows[(action, "segmented")]
        identical = (
            one["terminal_fes"] == seg["terminal_fes"] == TOTAL_FES
            and one["final_error"] == seg["final_error"]
            and one["incumbent"] == seg["incumbent"]
            and one["consumed_fes"] == seg["consumed_fes"]
        )
        checks[f"{action}_segment_equivalence"] = identical
        per_action.append(
            {
                "action": action,
                "one_shot_final_error": one["final_error"],
                "segmented_final_error": seg["final_error"],
                "segments": seg.get("segments"),
                "snapshot_restores": seg.get("snapshot_restores"),
                "bit_identical": identical,
            }
        )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "dimension": DIMENSION,
            "block_size": BLOCK_SIZE,
            "total_fes": TOTAL_FES,
            "phase1_fes": PHASE1_FES,
            "action_seed": ACTION_SEED,
            "chunk_pattern": list(CHUNK_PATTERN),
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
        "actions": per_action,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(checks, indent=1))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
