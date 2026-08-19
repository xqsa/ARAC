"""Gate 9: real local black-box proposals consumed by overlap coordination."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arac.coordination import OverlapCoordinator, produce_local_proposal
from arac.evidence import Phase1OverlapAdapter, run_phase1_overlap_pilot
from arac.runtime.ledger import EvaluationLedger

if __package__:
    from experiments.phase1_overlap_integration_gate7 import (
        BUCKET_SIZE,
        DIMENSION,
        GROUPS,
        ROUNDS,
        SEED,
        TOTAL_BUDGET_FES,
        _problem,
    )
else:
    from phase1_overlap_integration_gate7 import (  # type: ignore[no-redef]
        BUCKET_SIZE,
        DIMENSION,
        GROUPS,
        ROUNDS,
        SEED,
        TOTAL_BUDGET_FES,
        _problem,
    )


PROPOSAL_BUDGET_FES = 64
POPULATION_SIZE = 8
PROPOSAL_ALGORITHM = "sepcmaes"


def _proposal_payload(run) -> dict[str, object]:
    proposal = run.proposal
    return {
        "group": proposal.group,
        "values": proposal.values,
        "improvement": proposal.improvement,
        "uncertainty": proposal.uncertainty,
        "algorithm": run.algorithm,
        "best_error": run.best_error,
        "consumed_fes": run.consumed_fes,
        "global_start_fes": run.global_start_fes,
        "global_end_fes": run.global_end_fes,
    }


def run_gate(
    *,
    proposal_budget_fes: int = PROPOSAL_BUDGET_FES,
    proposal_algorithm: str = PROPOSAL_ALGORITHM,
) -> dict[str, object]:
    problem = _problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=SEED,
        anchor_count=5,
        step=0.25,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=128,
    )
    adaptation = Phase1OverlapAdapter().adapt(pilot.checkpoint, pilot.evidence)
    if not adaptation.ready or adaptation.structure is None:
        raise RuntimeError("Gate 9 requires complete Phase-I overlap adaptation")
    structure = adaptation.structure
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        phase1_fes=pilot.checkpoint.phase1_fes,
        incumbent=pilot.checkpoint.incumbent,
        incumbent_error=pilot.checkpoint.incumbent_error,
    )
    phase1_anchor = tuple(float(value) for value in pilot.checkpoint.incumbent)
    phase1_error = float(pilot.checkpoint.incumbent_error)
    proposal_runs = []
    component_results = []
    for component in structure.connected_components():
        if len(component) <= 1:
            continue
        component_runs = []
        for group in component:
            run = produce_local_proposal(
                structure,
                group,
                problem=problem,
                global_ledger=ledger,
                anchor=phase1_anchor,
                anchor_error=phase1_error,
                budget_fes=proposal_budget_fes,
                seed=SEED ^ (0x9E37 * (group + 1)),
                algorithm=proposal_algorithm,
                population_size=POPULATION_SIZE,
                sigma=0.5,
            )
            proposal_runs.append(run)
            component_runs.append(run.proposal)
        before_coordination = ledger.count
        coordinator = OverlapCoordinator(structure, ledger)
        result = coordinator.coordinate(
            component,
            tuple(component_runs),
            search_base=np.asarray(phase1_anchor, dtype=float),
        )
        component_results.append(
            {
                "component": component,
                "shared_variables": tuple(
                    variable
                    for variable in structure.shared_variables
                    if set(structure.owners(variable)).issubset(set(component))
                ),
                "residual_variables": tuple(item.variable for item in result.residuals),
                "conflict_level": result.conflict_level.value,
                "conflict_streak": result.conflict_streak,
                "candidate_count": len(result.candidates),
                "candidate_names": tuple(item.name for item in result.candidates),
                "candidate_errors": result.candidate_errors,
                "accepted": result.accepted,
                "accepted_candidate": result.accepted_candidate,
                "best_error_before": result.best_error_before,
                "best_error_after": result.best_error_after,
                "coordination_consumed_fes": ledger.count - before_coordination,
                "proposal_groups": tuple(proposal.group for proposal in component_runs),
            }
        )

    proposal_fes = sum(run.consumed_fes for run in proposal_runs)
    coordination_fes = sum(int(item["coordination_consumed_fes"]) for item in component_results)
    phase2_fes = ledger.count - pilot.checkpoint.phase1_fes
    expected_shared = (2, 102)
    residual_variables = tuple(
        sorted(variable for item in component_results for variable in item["residual_variables"])
    )
    gate_checks = {
        "exact_phase1_boundary": pilot.checkpoint.phase1_fes == 180_000,
        "adapter_ready": adaptation.ready,
        "all_overlap_components_have_real_proposals": len(component_results) == 2
        and all(
            tuple(item["proposal_groups"]) == tuple(item["component"])
            for item in component_results
        ),
        "proposal_covers_complete_groups": all(
            len(run.proposal.values) == len(structure.groups[run.proposal.group])
            and {variable for variable, _ in run.proposal.values}
            == set(structure.groups[run.proposal.group])
            for run in proposal_runs
        ),
        "proposal_metrics_finite_nonnegative": all(
            np.isfinite(run.proposal.improvement)
            and run.proposal.improvement >= 0.0
            and all(np.isfinite(sigma) and sigma >= 0.0 for _, sigma in run.proposal.uncertainty)
            for run in proposal_runs
        ),
        "proposal_fe_exact": proposal_fes == len(proposal_runs) * proposal_budget_fes,
        "global_phase2_fe_reconciles": phase2_fes == proposal_fes + coordination_fes,
        "residual_variables_exact": residual_variables == expected_shared,
        "strict_best_preserved": all(
            float(item["best_error_after"]) <= float(item["best_error_before"])
            for item in component_results
        ),
        "phase2_budget_not_exhausted": ledger.count < TOTAL_BUDGET_FES,
    }
    return {
        "schema_version": "arac-phase1-overlap-real-proposals-gate9-v1",
        "dimension": DIMENSION,
        "groups": GROUPS,
        "total_budget_fes": TOTAL_BUDGET_FES,
        "phase1_fes": pilot.checkpoint.phase1_fes,
        "phase2_budget": TOTAL_BUDGET_FES - pilot.checkpoint.phase1_fes,
        "phase1_incumbent_error": phase1_error,
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "adapter_ready": adaptation.ready,
        "proposal_algorithm": proposal_algorithm,
        "proposal_budget_fes_each": proposal_budget_fes,
        "proposal_population_size": POPULATION_SIZE,
        "proposals": tuple(_proposal_payload(run) for run in proposal_runs),
        "components": component_results,
        "proposal_fes": proposal_fes,
        "coordination_fes": coordination_fes,
        "phase2_fes_consumed": phase2_fes,
        "phase2_ledger_count_after_coordination": ledger.count,
        "shared_variables": structure.shared_variables,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase1_overlap_real_proposals_gate9/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
