"""Budget feasibility pilot for sparse overlap discovery at d=1000."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.sparse_overlap_discovery import discover_overlap_sparse
from arac.runtime.ledger import EvaluationLedger


DIMENSION = 1000
GROUPS = ((0, 1, 2), (2, 3, 4), (100, 101, 102), (102, 103, 104))
ANCHOR_COUNT = 5
ROUNDS = 12
BUCKET_SIZE = 16
PHASE1_BUDGET = 180_000
MAX_CANDIDATE_PAIRS = 128
STEP = 0.25
SEED = 20260813


def _problem() -> OptimizationProblem:
    def objective(values: np.ndarray) -> float | np.ndarray:
        converted = np.asarray(values, dtype=float)
        batch = converted[np.newaxis, :] if converted.ndim == 1 else converted
        result = np.sum(batch**2, axis=1)
        for group in GROUPS:
            local = batch[:, group]
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    result += local[:, left] * local[:, right]
        return float(result[0]) if converted.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def run_pilot() -> dict[str, object]:
    problem = _problem()
    anchor_rng = np.random.default_rng(SEED)
    anchors = tuple(
        tuple(float(value) for value in row)
        for row in anchor_rng.uniform(-2.0, 2.0, size=(ANCHOR_COUNT, DIMENSION))
    )
    ledger = EvaluationLedger(problem, PHASE1_BUDGET)
    result = discover_overlap_sparse(
        problem,
        ledger,
        anchors=anchors,
        step=STEP,
        run_seed=SEED,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=MAX_CANDIDATE_PAIRS,
    )
    expected_shared = tuple(
        variable
        for variable in range(DIMENSION)
        if sum(variable in group for group in GROUPS) > 1
    )
    inferred_shared = tuple(
        variable
        for variable, owners in enumerate(result.evidence.memberships)
        if len(owners) > 1
    )
    return {
        "schema_version": "arac-oracle-sparse-overlap-discovery-gate6-budget-v1",
        "dimension": DIMENSION,
        "groups": GROUPS,
        "seed": SEED,
        "anchor_count": ANCHOR_COUNT,
        "rounds": ROUNDS,
        "bucket_size": BUCKET_SIZE,
        "phase1_budget": PHASE1_BUDGET,
        "consumed_fes": result.consumed_fes,
        "expected_fes": result.expected_fes,
        "remaining_fes": ledger.remaining,
        "separated_pair_fraction": result.separated_pair_fraction,
        "candidate_pair_count": result.candidate_pair_count,
        "expected_shared": expected_shared,
        "inferred_shared": inferred_shared,
        "inferred_groups": result.evidence.groups,
        "complete": result.complete,
        "complete_reason": result.complete_reason,
        "budget_passed": (
            result.complete
            and result.separated_pair_fraction == 1.0
            and inferred_shared == expected_shared
            and result.consumed_fes <= PHASE1_BUDGET
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oracle_sparse_overlap_discovery_gate6/d1000_budget_pilot.json"),
    )
    args = parser.parse_args()
    payload = run_pilot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("complete", "consumed_fes", "remaining_fes", "candidate_pair_count", "inferred_shared", "budget_passed")}, indent=2, sort_keys=True))
    return 0 if payload["budget_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
