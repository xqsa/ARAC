"""Gate 28: real 3M-FE end-to-end overlap-focused ARAC run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from arac.overlap_core import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    run_overlap_arac,
)
from experiments.large_scale_headroom_gate25 import (
    DIMENSION,
    PHASE1_FES,
    SEED,
    TOTAL_BUDGET_FES,
    _problem,
)


# Historical Gate25 reference only. Its artifact predates the current
# checkpoint accounting and is not a paired performance control.
GATE25_ONE_CYCLE_ERROR = 427.9864170987402


def run_gate() -> dict[str, object]:
    problem, _truth_structure = _problem()
    result = run_overlap_arac(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=SEED,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        phase1_kwargs={
            "anchor_count": 5,
            "step": 0.25,
            "rounds": 12,
            "bucket_size": 16,
            "max_candidate_pairs": 128,
        },
    )
    inferred_shared = tuple(
        variable
        for variable, owners in enumerate(result.phase1.evidence.memberships)
        if len(owners) > 1
    )
    checks = {
        "phase1_exact": result.phase1.consumed_fes == PHASE1_FES,
        "phase2_exact": result.phase2_consumed_fes == TOTAL_BUDGET_FES - PHASE1_FES,
        "terminal_exact": result.terminal_fes == TOTAL_BUDGET_FES,
        "adapter_ready": result.phase1.adaptation.ready,
        "discovery_complete": result.phase1.discovery.complete,
        "shared_variables_present": bool(inferred_shared),
        "evidence_components_used": result.overlap_components
        == tuple(component for component in result.phase1.adaptation.structure.connected_components() if len(component) > 1),
        "refresh_cycles_exact": len(result.cycles) == DEFAULT_REFRESH_CYCLES,
        "proposal_fe_exact": all(
            cycle.proposal_fes
            == len(result.overlap_groups) * result.proposal_budget_fes
            for cycle in result.cycles
        ),
        "strict_best": all(
            cycle.best_error_after <= cycle.best_error_before for cycle in result.cycles
        ),
        "gain_decomposition": all(
            cycle.proposal_gain >= 0.0
            and cycle.coordination_gain >= 0.0
            and abs(
                cycle.best_error_before
                - cycle.best_error_after
                - cycle.proposal_gain
                - cycle.coordination_gain
            )
            <= 1.0e-10
            for cycle in result.cycles
        ),
        "improves_phase1": result.final_error <= result.phase1.checkpoint.incumbent_error,
        "beats_legacy_gate25_reference": result.final_error <= GATE25_ONE_CYCLE_ERROR,
    }
    return {
        "schema_version": "arac-overlap-end-to-end-gate28-v1",
        "protocol": {
            "dimension": DIMENSION,
            "total_budget_fes": TOTAL_BUDGET_FES,
            "phase1_fes": PHASE1_FES,
            "phase2_fes": TOTAL_BUDGET_FES - PHASE1_FES,
            "refresh_cycles": DEFAULT_REFRESH_CYCLES,
            "neighborhood_fes_per_component_cycle": DEFAULT_NEIGHBORHOOD_FES,
            "run_seed": SEED,
        },
        "checkpoint_hash": result.phase1.checkpoint.checkpoint_hash,
        "checkpoint_error": result.phase1.checkpoint.incumbent_error,
        "inferred_shared_variables": inferred_shared,
        "overlap_groups": result.overlap_groups,
        "overlap_components": result.overlap_components,
        "proposal_budget_fes": result.proposal_budget_fes,
        "cycles": [asdict(cycle) for cycle in result.cycles],
        "tail_fes": result.tail_fes,
        "phase2_consumed_fes": result.phase2_consumed_fes,
        "terminal_fes": result.terminal_fes,
        "final_error": result.final_error,
        "total_proposal_gain": sum(cycle.proposal_gain for cycle in result.cycles),
        "total_coordination_gain": sum(cycle.coordination_gain for cycle in result.cycles),
        "legacy_gate25_reference_error": GATE25_ONE_CYCLE_ERROR,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_arac_end_to_end_gate28/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_passed": payload["gate_passed"],
                "gate_checks": payload["gate_checks"],
                "checkpoint_error": payload["checkpoint_error"],
                "final_error": payload["final_error"],
                "proposal_budget_fes": payload["proposal_budget_fes"],
                "tail_fes": payload["tail_fes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
