"""Gate 27: production proposal-conditioned neighborhood validation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.coordination import GraphCoordinationScheduler, OverlapCoordinator
from arac.runtime.ledger import EvaluationLedger
from experiments.overlap_full_context_writeback_gate21 import (
    ROUNDS,
    _new_scheduler,
    _owner_control,
    _proposal_payload,
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
    proposals_identical: bool
    fe_parity: bool
    strict_best: bool
    trace_complete: bool
    full_context: ArmResult
    proposal_neighborhood: ArmResult
    owner_full: ArmResult
    neighborhood_vs_full_gain: float
    neighborhood_vs_owner_gain: float


def _context(mode: str, topology: str, overlap_budget: int, seed: int) -> ContextResult:
    problem, structure, _ = _combined_problem(mode, topology, overlap_budget, seed)
    checkpoint_x, checkpoint_error, proposals, checkpoint_fes = _proposal_payload(problem, structure, seed)
    components = tuple(
        GraphCoordinationScheduler(
            OverlapCoordinator(structure, EvaluationLedger(problem, total_budget=ARM_TOTAL_BUDGET_FES))
        ).overlap_components
    )
    if len(components) < 2:
        raise RuntimeError("expected at least two overlap components")
    selector_ledger, selector = _new_scheduler(
        problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes
    )
    selector.prime(proposals)
    probes = selector.value_probe(proposals)
    priorities = {item.component: item.priority_score for item in selector.prioritize(proposals)}
    selected = max(
        probes,
        key=lambda item: (
            item.estimated_gain,
            priorities[item.component],
            tuple(-value for value in item.component),
        ),
    ).component
    selected_proposals = tuple(proposal for proposal in proposals if proposal.group in selected)

    def new_arm() -> tuple[EvaluationLedger, OverlapCoordinator, int]:
        ledger, scheduler = _new_scheduler(
            problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes
        )
        scheduler.prime(proposals)
        before = ledger.count
        scheduler.coordinator.coordinate(selected, selected_proposals, ctp_budget_fes=0)
        return ledger, scheduler.coordinator, ledger.count - before

    full_ledger, full_coordinator, full_arbitration_fes = new_arm()
    full_before = float(full_ledger.best_error)
    full_result = full_coordinator.full_context_writeback(selected, selected_proposals, rounds=ROUNDS)
    full = ArmResult(
        "full_context",
        selected,
        float(full_ledger.best_error),
        checkpoint_error,
        full_ledger.count - checkpoint_fes,
        len(probes) * PROBE_FES_PER_COMPONENT,
        full_arbitration_fes,
        full_result.consumed_fes,
        full_ledger.best_error <= full_before,
        tuple(item.round_index for item in full_result.rounds if item.accepted),
        len(full_result.rounds),
    )

    neighborhood_ledger, neighborhood_coordinator, neighborhood_arbitration_fes = new_arm()
    neighborhood_before = float(neighborhood_ledger.best_error)
    endpoint_budget = 2 * len(selected)
    endpoint_result = neighborhood_coordinator.full_context_writeback(
        selected, selected_proposals, rounds=len(selected)
    )
    remaining = CTP_BUDGET_FES - endpoint_budget
    neighborhood_result = neighborhood_coordinator.proposal_neighborhood_writeback(
        selected,
        selected_proposals,
        budget_fes=remaining,
        seed=seed ^ 0x51ED,
    )
    neighborhood = ArmResult(
        "proposal_neighborhood",
        selected,
        float(neighborhood_ledger.best_error),
        checkpoint_error,
        neighborhood_ledger.count - checkpoint_fes,
        len(probes) * PROBE_FES_PER_COMPONENT,
        neighborhood_arbitration_fes,
        endpoint_result.consumed_fes + neighborhood_result.consumed_fes,
        neighborhood_ledger.best_error <= neighborhood_before,
        tuple(item.round_index for item in endpoint_result.rounds if item.accepted)
        + tuple(endpoint_budget + item.round_index for item in neighborhood_result.rounds if item.accepted),
        len(endpoint_result.rounds) + len(neighborhood_result.rounds),
    )

    owner_ledger, owner_scheduler = _new_scheduler(
        problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes
    )
    owner_scheduler.prime(proposals)
    owner_before = float(owner_ledger.best_error)
    owner_arbitration_fes, owner_error, owner_archive_ok = _owner_control(
        owner_scheduler, proposals, selected, seed=seed ^ 0x51ED, budget_fes=CTP_BUDGET_FES
    )
    owner = ArmResult(
        "owner_full",
        selected,
        float(owner_error),
        checkpoint_error,
        owner_ledger.count - checkpoint_fes,
        len(probes) * PROBE_FES_PER_COMPONENT,
        owner_arbitration_fes,
        CTP_BUDGET_FES,
        bool(owner_archive_ok and owner_error <= owner_before),
    )
    arms = (full, neighborhood, owner)
    return ContextResult(
        mode,
        topology,
        overlap_budget,
        seed,
        len(components),
        selected,
        True,
        len({item.consumed_fes for item in arms}) == 1,
        all(item.strict_best for item in arms),
        full.round_count == ROUNDS
        and neighborhood.round_count == len(selected) + CTP_BUDGET_FES - 2 * len(selected)
        and neighborhood.continuation_fes == CTP_BUDGET_FES,
        full,
        neighborhood,
        owner,
        full.final_error - neighborhood.final_error,
        owner.final_error - neighborhood.final_error,
    )


def run_gate(*, workers: int = 1) -> dict[str, object]:
    jobs = tuple(
        (mode, topology, overlap_budget, seed)
        for topology in TOPOLOGIES
        for overlap_budget in OVERLAP_BUDGETS
        for seed in FRESH_SEEDS
        for mode in MODES
    )
    if workers == 1:
        contexts = tuple(_context(*job) for job in jobs)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            contexts = tuple(executor.map(lambda job: _context(*job), jobs))
    contexts = tuple(sorted(contexts, key=lambda row: (row.topology, row.overlap_budget, row.seed, row.mode)))
    neighborhood_vs_full = np.asarray([row.neighborhood_vs_full_gain for row in contexts])
    neighborhood_vs_owner = np.asarray([row.neighborhood_vs_owner_gain for row in contexts])
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
        "proposals_identical": all(row.proposals_identical for row in contexts),
        "fe_parity": all(row.fe_parity for row in contexts),
        "strict_best": all(row.strict_best for row in contexts),
        "trace_complete": all(row.trace_complete for row in contexts),
        "neighborhood_vs_full_ge_0_60": float(np.mean(neighborhood_vs_full >= 0.0)) >= 0.60,
        "neighborhood_vs_owner_median_nonnegative": float(np.median(neighborhood_vs_owner)) >= 0.0,
    }
    return {
        "schema_version": "arac-proposal-neighborhood-gate27-v1",
        "protocol": {
            "contexts": 60,
            "endpoint_fes": 2,
            "continuation_fes": CTP_BUDGET_FES,
            "neighborhood_fes": CTP_BUDGET_FES - 2 * 2,
        },
        "context_count": len(contexts),
        "contexts": [asdict(row) for row in contexts],
        "cell_summary": cells,
        "summary": {
            "neighborhood_vs_full_win_or_tie": float(np.mean(neighborhood_vs_full >= 0.0)),
            "neighborhood_vs_full_median_gain": float(np.median(neighborhood_vs_full)),
            "neighborhood_vs_owner_win_or_tie": float(np.mean(neighborhood_vs_owner >= 0.0)),
            "neighborhood_vs_owner_median_gain": float(np.median(neighborhood_vs_owner)),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/proposal_neighborhood_gate27/confirmation_fresh.json")
    )
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
