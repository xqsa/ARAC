"""Gate 22: discovered Phase-I structure through real full-context write-back."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arac.coordination import OverlapCoordinator, produce_local_proposal
from arac.evidence import Phase1OverlapAdapter, run_phase1_overlap_pilot
from arac.runtime.ledger import EvaluationLedger

from experiments.phase1_overlap_integration_gate7 import (
    BUCKET_SIZE,
    DIMENSION,
    ROUNDS,
    SEED,
    TOTAL_BUDGET_FES,
    _problem,
)


PROPOSAL_BUDGET_FES = 64
PROPOSAL_POPULATION_SIZE = 8
PROPOSAL_ALGORITHM = "sepcmaes"
WRITEBACK_ROUNDS = 16


def _proposal_payload(run) -> dict[str, object]:
    proposal = run.proposal
    return {
        "group": proposal.group,
        "values": proposal.values,
        "improvement": proposal.improvement,
        "uncertainty": proposal.uncertainty,
        "algorithm": run.algorithm,
        "consumed_fes": run.consumed_fes,
        "global_start_fes": run.global_start_fes,
        "global_end_fes": run.global_end_fes,
    }


def run_gate() -> dict[str, object]:
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
        raise RuntimeError("Gate 22 requires complete Phase-I overlap adaptation")
    structure = adaptation.structure
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        phase1_fes=pilot.checkpoint.phase1_fes,
        incumbent=pilot.checkpoint.incumbent,
        incumbent_error=pilot.checkpoint.incumbent_error,
    )
    anchor = tuple(float(value) for value in pilot.checkpoint.incumbent)
    anchor_error = float(pilot.checkpoint.incumbent_error)
    all_runs = []
    components = []
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
                anchor=anchor,
                anchor_error=anchor_error,
                budget_fes=PROPOSAL_BUDGET_FES,
                seed=SEED ^ (0x9E37 * (group + 1)),
                algorithm=PROPOSAL_ALGORITHM,
                population_size=PROPOSAL_POPULATION_SIZE,
                sigma=0.5,
            )
            all_runs.append(run)
            component_runs.append(run.proposal)
        coordinator = OverlapCoordinator(structure, ledger)
        arbitration = coordinator.coordinate(component, tuple(component_runs), ctp_budget_fes=0)
        if len(arbitration.candidates) not in (3, 4):
            raise RuntimeError("unexpected production arbitration candidate count")
        writeback = coordinator.full_context_writeback(
            component,
            tuple(component_runs),
            rounds=WRITEBACK_ROUNDS,
        )
        components.append(
            {
                "component": component,
                "groups": tuple(component),
                "shared_variables": tuple(
                    variable
                    for variable in structure.shared_variables
                    if set(structure.owners(variable)).issubset(set(component))
                ),
                "proposal_groups": tuple(proposal.group for proposal in component_runs),
            "arbitration_fes": len(arbitration.candidates),
                "writeback_fes": writeback.consumed_fes,
                "writeback_rounds": len(writeback.rounds),
                "accepted_rounds": tuple(item.round_index for item in writeback.rounds if item.accepted),
                "strict_best": all(item.best_error_after <= item.best_error_before for item in writeback.rounds),
                "best_error_before": writeback.best_error_before,
                "best_error_after": writeback.best_error_after,
            }
        )
    proposal_fes = sum(run.consumed_fes for run in all_runs)
    arbitration_fes = sum(int(item["arbitration_fes"]) for item in components)
    writeback_fes = sum(int(item["writeback_fes"]) for item in components)
    phase2_fes = ledger.count - pilot.checkpoint.phase1_fes
    inferred_shared = tuple(
        variable for variable, owners in enumerate(pilot.evidence.memberships) if len(owners) > 1
    )
    checks = {
        "exact_phase1_boundary": pilot.consumed_fes == pilot.checkpoint.phase1_fes == 180_000,
        "adapter_ready": adaptation.ready,
        "discovery_complete": pilot.discovery.complete,
        "discovered_shared_exact": inferred_shared == (2, 102),
        "two_overlap_components": len(components) == 2,
        "proposal_groups_from_adapter": all(item["proposal_groups"] == item["groups"] for item in components),
        "proposal_coverage": all(
            len(run.proposal.values) == len(structure.groups[run.proposal.group])
            and {variable for variable, _ in run.proposal.values} == set(structure.groups[run.proposal.group])
            for run in all_runs
        ),
        "proposal_fe_exact": proposal_fes == len(all_runs) * PROPOSAL_BUDGET_FES,
        "writeback_trace_exact": all(
            item["writeback_fes"] == 2 * WRITEBACK_ROUNDS and item["writeback_rounds"] == WRITEBACK_ROUNDS
            for item in components
        ),
        "strict_best": all(item["strict_best"] for item in components),
        "phase2_fe_reconciles": phase2_fes == proposal_fes + arbitration_fes + writeback_fes,
        "expected_phase2_fe": phase2_fes == proposal_fes + arbitration_fes + writeback_fes,
        "budget_preserved": ledger.count < TOTAL_BUDGET_FES,
    }
    return {
        "schema_version": "arac-phase1-full-context-integration-gate22-v1",
        "protocol": {
            "dimension": DIMENSION,
            "total_budget_fes": TOTAL_BUDGET_FES,
            "phase1_fes": 180_000,
            "proposal_budget_fes": PROPOSAL_BUDGET_FES,
            "writeback_rounds": WRITEBACK_ROUNDS,
            "writeback_fes_per_component": 2 * WRITEBACK_ROUNDS,
        },
        "phase1_consumed_fes": pilot.consumed_fes,
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "inferred_shared_variables": inferred_shared,
        "components": components,
        "proposals": [_proposal_payload(run) for run in all_runs],
        "proposal_fes": proposal_fes,
        "arbitration_fes": arbitration_fes,
        "writeback_fes": writeback_fes,
        "phase2_fes_consumed": phase2_fes,
        "phase2_ledger_count": ledger.count,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase1_full_context_integration_gate22/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
