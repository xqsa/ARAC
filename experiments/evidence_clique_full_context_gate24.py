"""Gate 24: full-context coordination using Phase-I evidence cliques."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.coordination import GraphCoordinationScheduler, OverlapCoordinator, produce_local_proposal
from arac.evidence import Phase1OverlapAdapter, run_phase1_overlap_pilot
from arac.runtime.ledger import EvaluationLedger

from experiments.interaction_phase1_discovery_gate23 import (
    ANCHOR_COUNT,
    BUCKET_SIZE,
    MAX_CANDIDATE_PAIRS,
    MODE,
    OVERLAP_BUDGET,
    PHASE1_FES,
    ROUNDS,
    SEED,
    STEP,
    TOTAL_BUDGET_FES,
    TOPOLOGY,
    _problem,
)


PROPOSAL_BUDGET_FES = 48
PROPOSAL_POPULATION_SIZE = 8
PROPOSAL_ALGORITHM = "sepcmaes"
WRITEBACK_ROUNDS = 16
CONTINUATION_FES = 32


@dataclass(frozen=True)
class ArmResult:
    arm: str
    selected_component: tuple[int, ...]
    final_error: float
    checkpoint_error: float
    proposal_fes: int
    arbitration_fes: int
    continuation_fes: int
    consumed_phase2_fes: int
    strict_best: bool
    trace_rounds: int
    accepted_rounds: tuple[int, ...] = ()


def _phase1():
    problem, objective = _problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=SEED,
        anchor_count=ANCHOR_COUNT,
        step=STEP,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=MAX_CANDIDATE_PAIRS,
    )
    adaptation = Phase1OverlapAdapter().adapt(pilot.checkpoint, pilot.evidence)
    if not adaptation.ready or adaptation.structure is None:
        raise RuntimeError("Gate 24 requires a ready Phase-I adapter")
    return problem, objective, pilot, adaptation.structure


def _proposal_payload(problem, structure, pilot):
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        phase1_fes=pilot.checkpoint.phase1_fes,
        incumbent=pilot.checkpoint.incumbent,
        incumbent_error=pilot.checkpoint.incumbent_error,
    )
    anchor = tuple(float(value) for value in pilot.checkpoint.incumbent)
    anchor_error = float(pilot.checkpoint.incumbent_error)
    runs = []
    for group in range(len(structure.groups)):
        runs.append(
            produce_local_proposal(
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
        )
    proposals = tuple(run.proposal for run in runs)
    expected = len(structure.groups) * PROPOSAL_BUDGET_FES
    if ledger.count - pilot.checkpoint.phase1_fes != expected:
        raise RuntimeError("evidence-clique proposal FE drifted")
    return ledger, proposals, expected


def _select_component(structure, proposals, ledger):
    scheduler = GraphCoordinationScheduler(
        OverlapCoordinator(structure, ledger, medium_threshold=0.0, high_threshold=0.0)
    )
    priorities = scheduler.prioritize(proposals)
    if not priorities:
        raise RuntimeError("evidence cliques produced no overlap component")
    return priorities[0].component


def _arm(problem, structure, pilot, proposal_ledger, proposals, selected, *, arm: str) -> ArmResult:
    ledger = EvaluationLedger(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        initial_count=proposal_ledger.count,
        initial_incumbent=tuple(float(value) for value in proposal_ledger.best_x),
        initial_error=float(proposal_ledger.best_error),
    )
    coordinator = OverlapCoordinator(structure, ledger, medium_threshold=0.0, high_threshold=0.0)
    selected_proposals = tuple(proposal for proposal in proposals if proposal.group in selected)
    arbitration = coordinator.coordinate(selected, selected_proposals, ctp_budget_fes=0)
    arbitration_fes = len(arbitration.candidates)
    if arm == "full_context":
        writeback = coordinator.full_context_writeback(selected, selected_proposals, rounds=WRITEBACK_ROUNDS)
        continuation = writeback.consumed_fes
        accepted = tuple(item.round_index for item in writeback.rounds if item.accepted)
        trace_rounds = len(writeback.rounds)
        final_error = ledger.best_error
        strict_best = final_error <= proposal_ledger.best_error
    elif arm == "current_ctp":
        continuation = coordinator._repair_shared_core(
            selected,
            selected_proposals,
            budget_fes=CONTINUATION_FES,
            seed=SEED ^ 0x51ED,
            base=ledger.best_x,
        )
        accepted = ()
        trace_rounds = 0
        final_error = ledger.best_error
        strict_best = final_error <= proposal_ledger.best_error
    elif arm == "owner_full":
        rng = np.random.default_rng(SEED ^ 0xA0B0)
        candidates = np.repeat(ledger.best_x[np.newaxis, :], CONTINUATION_FES, axis=0)
        for index in range(CONTINUATION_FES):
            proposal = selected_proposals[index % len(selected_proposals)]
            for variable, value in proposal.values:
                candidates[index, variable] = value + float(
                    rng.normal(0.0, max(np.finfo(float).eps, proposal.sigma(variable)))
                )
        np.clip(candidates, problem.lower_array, problem.upper_array, out=candidates)
        ledger.evaluate(candidates)
        continuation = CONTINUATION_FES
        accepted = ()
        trace_rounds = 0
        final_error = ledger.best_error
        strict_best = final_error <= proposal_ledger.best_error
    else:
        raise ValueError(f"unknown arm {arm}")
    return ArmResult(
        arm=arm,
        selected_component=selected,
        final_error=float(final_error),
        checkpoint_error=float(proposal_ledger.best_error),
        proposal_fes=len(structure.groups) * PROPOSAL_BUDGET_FES,
        arbitration_fes=arbitration_fes,
        continuation_fes=continuation,
        consumed_phase2_fes=(len(structure.groups) * PROPOSAL_BUDGET_FES) + arbitration_fes + continuation,
        strict_best=bool(strict_best),
        trace_rounds=trace_rounds,
        accepted_rounds=accepted,
    )


def run_gate() -> dict[str, object]:
    problem, objective, pilot, structure = _phase1()
    proposal_ledger, proposals, proposal_fes = _proposal_payload(problem, structure, pilot)
    selected = _select_component(structure, proposals, proposal_ledger)
    arms = tuple(
        _arm(problem, structure, pilot, proposal_ledger, proposals, selected, arm=arm)
        for arm in ("current_ctp", "full_context", "owner_full")
    )
    current, full, owner = arms
    inferred_shared = tuple(variable for variable, owners in enumerate(pilot.evidence.memberships) if len(owners) > 1)
    checks = {
        "phase1_boundary": pilot.consumed_fes == PHASE1_FES,
        "adapter_ready": pilot.adaptation.ready,
        "shared_precision_one": set(inferred_shared).issubset(
            set(objective.structure.shared_variables)
        ),
        "shared_recall_one": set(objective.structure.shared_variables).issubset(
            set(inferred_shared)
        ),
        "proposal_coverage": all(
            set(variable for variable, _ in proposal.values) == set(structure.groups[proposal.group])
            for proposal in proposals
        ),
        "proposal_fes_exact": proposal_fes == len(structure.groups) * PROPOSAL_BUDGET_FES,
        "paired_arbitration": current.arbitration_fes == full.arbitration_fes == owner.arbitration_fes,
        "paired_continuation": current.continuation_fes == full.continuation_fes == owner.continuation_fes == CONTINUATION_FES,
        "full_trace_exact": full.trace_rounds == WRITEBACK_ROUNDS and full.continuation_fes == 2 * WRITEBACK_ROUNDS,
        "strict_best": all(arm.strict_best for arm in arms),
        "full_no_worse_current": full.final_error <= current.final_error,
        "full_no_worse_owner": full.final_error <= owner.final_error,
    }
    return {
        "schema_version": "arac-evidence-clique-full-context-gate24-v1",
        "protocol": {
            "mode": MODE,
            "topology": TOPOLOGY,
            "overlap_budget": OVERLAP_BUDGET,
            "seed": SEED,
            "phase1_fes": PHASE1_FES,
            "proposal_budget_fes": PROPOSAL_BUDGET_FES,
            "writeback_rounds": WRITEBACK_ROUNDS,
        },
        "inferred_groups": structure.groups,
        "inferred_shared": inferred_shared,
        "selected_component": selected,
        "proposal_fes": proposal_fes,
        "arms": [asdict(arm) for arm in arms],
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/evidence_clique_full_context_gate24/confirmation_fresh.json"))
    args = parser.parse_args()
    payload = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "selected_component": payload["selected_component"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
