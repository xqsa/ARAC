"""Gate 21: complete local-proposal write-back into a shared context vector.

This is an offline mechanism diagnostic.  It tests the classical overlapping
cooperative-coevolution idea that a group optimizer emits a complete local
proposal, which is written into the current global context vector and then
evaluated as a complete black-box candidate.  The production coordinator is
intentionally untouched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.coordination import GraphCoordinationScheduler, OverlapCoordinator
from arac.runtime.ledger import EvaluationLedger
from experiments.overlap_joint_patch_gate20 import _repair_joint
from experiments.overlap_sequential_shared_patch_gate18 import (
    EVALS_PER_ROUND,
    ROUNDS,
)
from experiments.overlap_value_aware_dispatch_gate15 import (
    ARM_TOTAL_BUDGET_FES,
    CTP_BUDGET_FES,
    FRESH_SEEDS,
    MODES,
    OVERLAP_BUDGETS,
    PROBE_FES_PER_COMPONENT,
    TOPOLOGIES,
    _combined_problem,
    _new_scheduler,
    _owner_control,
    _proposal_payload,
)


@dataclass(frozen=True)
class ArmResult:
    arm: str
    selected_component: tuple[int, ...]
    final_error: float
    checkpoint_error: float
    consumed_fes: int
    probe_fes: int
    arbitration_fes: int
    continuation_fes: int
    strict_best: bool
    accepted_rounds: tuple[int, ...] = ()
    round_count: int = 0


@dataclass(frozen=True)
class ContextResult:
    mode: str
    topology: str
    overlap_budget: int
    seed: int
    component_count: int
    selected_component: tuple[int, ...]
    probes_identical: bool
    proposals_identical: bool
    fe_parity: bool
    strict_best: bool
    trace_complete: bool
    current: ArmResult
    full_context: ArmResult
    joint_patch: ArmResult
    owner_full: ArmResult
    full_vs_current_gain: float
    full_vs_joint_gain: float
    full_vs_owner_gain: float


def _probe_map(probes):
    return tuple((item.component, float(item.estimated_gain)) for item in probes)


def _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, component, *, arm, seed):
    ledger, scheduler = _new_scheduler(problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes)
    scheduler.prime(proposals)
    priming_fes = ledger.count - checkpoint_fes
    selected = tuple(component)
    if arm == "current":
        result = scheduler.dispatch_value_probe(
            proposals,
            total_ctp_budget_fes=CTP_BUDGET_FES,
            forced_component=selected,
            seed=seed ^ 0xC7A5,
        )
        probes = result.value_probes
        final_error = ledger.best_error
        archive_ok = all(event.best_error_after <= event.best_error_before for event in result.events)
        arbitration_fes = 4
        continuation_fes = result.consumed_ctp_fes
        accepted_rounds = ()
        round_count = 0
    else:
        probes = scheduler.value_probe(proposals)
        selected_proposals = tuple(proposal for proposal in proposals if proposal.group in selected)
        if arm == "owner_full":
            _, final_error, archive_ok = _owner_control(
                scheduler, proposals, selected, seed=seed ^ 0x51ED, budget_fes=CTP_BUDGET_FES
            )
            arbitration_fes = 4
            continuation_fes = CTP_BUDGET_FES
            accepted_rounds = ()
            round_count = 0
        else:
            arbitration = scheduler.coordinator.coordinate(selected, selected_proposals, ctp_budget_fes=0)
            if len(arbitration.candidates) != 4:
                raise RuntimeError("expected four arbitration candidates")
            arbitration_fes = 4
        if arm == "full_context":
            writeback = scheduler.coordinator.full_context_writeback(
                selected,
                selected_proposals,
                rounds=ROUNDS,
            )
            continuation_fes = writeback.consumed_fes
            accepted_rounds = tuple(item.round_index for item in writeback.rounds if item.accepted)
            round_count = len(writeback.rounds)
            final_error = ledger.best_error
            archive_ok = final_error <= checkpoint_error
        elif arm == "joint_patch":
            continuation_fes, _, accepted_rounds, _, _ = _repair_joint(scheduler, selected, proposals)
            round_count = ROUNDS
            final_error = ledger.best_error
            archive_ok = final_error <= checkpoint_error
        else:
            if arm != "owner_full":
                raise ValueError(f"unknown arm: {arm}")
    if len(probes) < 2:
        raise RuntimeError("expected at least two overlap components")
    consumed = ledger.count - checkpoint_fes
    expected = priming_fes + len(probes) * PROBE_FES_PER_COMPONENT + 4 + CTP_BUDGET_FES
    if consumed != expected:
        raise RuntimeError(f"{arm} FE mismatch: {consumed} != {expected}")
    return ArmResult(
        arm=arm,
        selected_component=selected,
        final_error=float(final_error),
        checkpoint_error=float(checkpoint_error),
        consumed_fes=consumed,
        probe_fes=len(probes) * PROBE_FES_PER_COMPONENT,
        arbitration_fes=arbitration_fes,
        continuation_fes=continuation_fes,
        strict_best=bool(archive_ok and final_error <= checkpoint_error),
        accepted_rounds=accepted_rounds,
        round_count=round_count,
    ), _probe_map(probes)


def _context(mode, topology, overlap_budget, seed):
    problem, structure, _ = _combined_problem(mode, topology, overlap_budget, seed)
    checkpoint_x, checkpoint_error, proposals, checkpoint_fes = _proposal_payload(problem, structure, seed)
    probe_scheduler = GraphCoordinationScheduler(
        OverlapCoordinator(structure, EvaluationLedger(problem, total_budget=ARM_TOTAL_BUDGET_FES))
    )
    components = probe_scheduler.overlap_components
    if len(components) < 2:
        raise RuntimeError("expected at least two overlap components")
    _, selector = _new_scheduler(problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes)
    selector.prime(proposals)
    probes = selector.value_probe(proposals)
    priorities = {item.component: item.priority_score for item in selector.prioritize(proposals)}
    selected = max(
        probes,
        key=lambda item: (item.estimated_gain, priorities[item.component], tuple(-value for value in item.component)),
    ).component
    arms = {
        name: _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, selected, arm=name, seed=seed)
        for name in ("current", "full_context", "joint_patch", "owner_full")
    }
    current, current_probes = arms["current"]
    full, full_probes = arms["full_context"]
    joint, joint_probes = arms["joint_patch"]
    owner, owner_probes = arms["owner_full"]
    return ContextResult(
        mode,
        topology,
        overlap_budget,
        seed,
        len(components),
        selected,
        current_probes == full_probes == joint_probes == owner_probes,
        True,
        len({item.consumed_fes for item in (current, full, joint, owner)}) == 1,
        all(item.strict_best for item in (current, full, joint, owner)),
        full.round_count == ROUNDS and full.continuation_fes == ROUNDS * EVALS_PER_ROUND,
        current,
        full,
        joint,
        owner,
        current.final_error - full.final_error,
        joint.final_error - full.final_error,
        owner.final_error - full.final_error,
    )


def run_gate(*, workers=1):
    jobs = tuple(
        (mode, topology, budget, seed)
        for topology in TOPOLOGIES
        for budget in OVERLAP_BUDGETS
        for seed in FRESH_SEEDS
        for mode in MODES
    )
    if workers == 1:
        contexts = tuple(_context(*job) for job in jobs)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            contexts = tuple(executor.map(lambda job: _context(*job), jobs))
    contexts = tuple(sorted(contexts, key=lambda row: (row.topology, row.overlap_budget, row.seed, row.mode)))
    current = np.asarray([row.full_vs_current_gain for row in contexts])
    joint = np.asarray([row.full_vs_joint_gain for row in contexts])
    owner = np.asarray([row.full_vs_owner_gain for row in contexts])
    cells = tuple(
        {
            "topology": topology,
            "overlap_budget": budget,
            "context_count": sum(row.topology == topology and row.overlap_budget == budget for row in contexts),
            "complete": sum(row.topology == topology and row.overlap_budget == budget for row in contexts) == 2 * len(FRESH_SEEDS),
        }
        for topology in TOPOLOGIES
        for budget in OVERLAP_BUDGETS
    )
    checks = {
        "context_count_60": len(contexts) == 60,
        "cells_complete": all(row["complete"] for row in cells),
        "components": all(row.component_count >= 2 for row in contexts),
        "probes_identical": all(row.probes_identical for row in contexts),
        "proposals_identical": all(row.proposals_identical for row in contexts),
        "fe_parity": all(row.fe_parity for row in contexts),
        "strict_best": all(row.strict_best for row in contexts),
        "trace_complete": all(row.trace_complete for row in contexts),
        "full_vs_current_ge_0_60": float(np.mean(current >= 0.0)) >= 0.60,
        "full_vs_owner_ge_0_60": float(np.mean(owner >= 0.0)) >= 0.60,
        "full_vs_owner_median_nonnegative": float(np.median(owner)) >= 0.0,
    }
    return {
        "schema_version": "arac-overlap-full-context-writeback-gate21-v1",
        "protocol": {"contexts": 60, "rounds": ROUNDS, "evaluations_per_round": EVALS_PER_ROUND, "repair_fes": CTP_BUDGET_FES},
        "context_count": len(contexts),
        "contexts": [asdict(row) for row in contexts],
        "cell_summary": cells,
        "summary": {
            "full_vs_current_win_or_tie": float(np.mean(current >= 0.0)),
            "full_vs_current_median_gain": float(np.median(current)),
            "full_vs_joint_win_or_tie": float(np.mean(joint >= 0.0)),
            "full_vs_joint_median_gain": float(np.median(joint)),
            "full_vs_owner_win_or_tie": float(np.mean(owner >= 0.0)),
            "full_vs_owner_median_gain": float(np.median(owner)),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/overlap_full_context_writeback_gate21/confirmation_fresh.json"))
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
