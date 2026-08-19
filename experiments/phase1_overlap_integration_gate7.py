"""Fresh Gate 7: Phase-I overlap evidence to adapter at d=1000."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence import (
    PHASE1_OVERLAP_PILOT_PROTOCOL,
    run_phase1_overlap_pilot,
)


DIMENSION = 1000
GROUPS = ((0, 1, 2), (2, 3, 4), (100, 101, 102), (102, 103, 104))
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
SEED = 20260813
ANCHOR_COUNT = 5
ROUNDS = 12
BUCKET_SIZE = 16


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


def run_gate() -> dict[str, object]:
    result = run_phase1_overlap_pilot(
        _problem(),
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=SEED,
        anchor_count=ANCHOR_COUNT,
        step=0.25,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=128,
    )
    expected_shared = (2, 102)
    inferred_shared = tuple(
        variable
        for variable, owners in enumerate(result.evidence.memberships)
        if len(owners) > 1
    )
    adapter = result.adaptation
    structure = adapter.structure
    structure_matches = structure is not None and structure.groups == result.evidence.groups
    return {
        "schema_version": "arac-phase1-overlap-integration-gate7-v1",
        "protocol": PHASE1_OVERLAP_PILOT_PROTOCOL,
        "dimension": DIMENSION,
        "total_budget_fes": TOTAL_BUDGET_FES,
        "phase1_budget_fes": result.target_phase1_fes,
        "phase1_consumed_fes": result.consumed_fes,
        "phase1_remaining_to_total": TOTAL_BUDGET_FES - result.consumed_fes,
        "seed": SEED,
        "anchor_count": ANCHOR_COUNT,
        "rounds": ROUNDS,
        "bucket_size": BUCKET_SIZE,
        "discovery_complete": result.discovery.complete,
        "discovery_reason": result.discovery.complete_reason,
        "discovery_consumed_fes": result.discovery.consumed_fes,
        "discovery_expected_fes": result.discovery.expected_fes,
        "separated_pair_fraction": result.discovery.separated_pair_fraction,
        "candidate_pair_count": result.discovery.candidate_pair_count,
        "expected_shared": expected_shared,
        "inferred_shared": inferred_shared,
        "adapter_status": adapter.status,
        "adapter_reason": adapter.reason,
        "checkpoint_hash": result.checkpoint.checkpoint_hash,
        "structure_matches_evidence": structure_matches,
        "gate_checks": {
            "exact_phase1_boundary": result.consumed_fes == PHASE1_FES == result.checkpoint.phase1_fes,
            "discovery_complete": result.discovery.complete,
            "coverage_complete": result.discovery.separated_pair_fraction == 1.0,
            "shared_variables_exact": inferred_shared == expected_shared,
            "adapter_ready": adapter.ready,
            "structure_matches_evidence": structure_matches,
            "phase2_budget_preserved": TOTAL_BUDGET_FES - result.consumed_fes == 2_820_000,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase1_overlap_integration_gate7/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate()
    payload["gate_passed"] = all(payload["gate_checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
