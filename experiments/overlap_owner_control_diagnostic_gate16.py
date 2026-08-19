"""Gate 16: diagnose CTP timing versus owner-control search domain."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.coordination import GraphCoordinationScheduler, OverlapCoordinator
from arac.runtime.ledger import EvaluationLedger
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


@dataclass(frozen=True)
class ContextResult:
    mode: str
    topology: str
    overlap_budget: int
    seed: int
    selected_component: tuple[int, ...]
    component_count: int
    probe_gains: tuple[tuple[tuple[int, ...], float], ...]
    proposals_identical: bool
    probe_parity: bool
    fe_parity: bool
    strict_best: bool
    current: ArmResult
    after_arbitration: ArmResult
    owner_shared_core: ArmResult
    owner_full: ArmResult
    after_vs_current_gain: float
    after_vs_shared_owner_gain: float
    after_vs_full_owner_gain: float


def _probe_map(probes):
    return tuple((item.component, float(item.estimated_gain)) for item in probes)


def _repair_after_arbitration(scheduler, component, proposals, *, seed: int):
    selected = tuple(proposal for proposal in proposals if proposal.group in component)
    before = float(scheduler.coordinator.ledger.best_error)
    arbitration = scheduler.coordinator.coordinate(component, selected, ctp_budget_fes=0)
    if len(arbitration.candidates) != 4:
        raise RuntimeError("expected four arbitration candidates")
    used = scheduler.coordinator._repair_shared_core(
        component,
        selected,
        budget_fes=CTP_BUDGET_FES,
        seed=seed,
        base=scheduler.coordinator.ledger.best_x,
    )
    if used != CTP_BUDGET_FES:
        raise RuntimeError("post-arbitration repair did not consume 32 FE")
    return 4, scheduler.coordinator.ledger.best_error, scheduler.coordinator.ledger.best_error <= before


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
        arbitration_fes = 4
        continuation_fes = result.consumed_ctp_fes
        final_error = ledger.best_error
        archive_ok = all(event.best_error_after <= event.best_error_before for event in result.events)
    else:
        probes = scheduler.value_probe(proposals)
    if len(probes) < 2:
        raise RuntimeError("Gate 16 expects at least two overlap components")
    if arm == "after_arbitration":
        arbitration_fes, final_error, archive_ok = _repair_after_arbitration(
            scheduler, selected, proposals, seed=seed ^ 0xC7A5
        )
        continuation_fes = CTP_BUDGET_FES
    elif arm == "owner_shared_core":
        selected_proposals = tuple(proposal for proposal in proposals if proposal.group in selected)
        arbitration_fes = 4
        result = scheduler.coordinator.coordinate(selected, selected_proposals, ctp_budget_fes=0)
        if len(result.candidates) != 4:
            raise RuntimeError("expected four arbitration candidates")
        base = scheduler.coordinator.ledger.best_x
        rng = np.random.default_rng(seed ^ 0x51ED)
        shared = scheduler.coordinator._component_variables(selected)
        by_group = {proposal.group: proposal for proposal in selected_proposals}
        candidates = np.repeat(base[np.newaxis, :], CTP_BUDGET_FES, axis=0)
        for index in range(CTP_BUDGET_FES):
            for variable in shared:
                owners = tuple(owner for owner in structure.owners(variable) if owner in selected)
                proposal = by_group[owners[index % len(owners)]]
                value = proposal.value(variable)
                candidates[index, variable] = value + float(rng.normal(0.0, max(np.finfo(float).eps, proposal.sigma(variable))))
        np.clip(candidates, problem.lower_array, problem.upper_array, out=candidates)
        before = ledger.best_error
        ledger.evaluate(candidates)
        continuation_fes = CTP_BUDGET_FES
        final_error = ledger.best_error
        archive_ok = final_error <= before
    elif arm == "owner_full":
        arbitration_fes, final_error, archive_ok = _owner_control(
            scheduler, proposals, selected, seed=seed ^ 0x51ED, budget_fes=CTP_BUDGET_FES
        )
        continuation_fes = CTP_BUDGET_FES
    elif arm != "current":
        raise ValueError(f"unknown arm: {arm}")
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
    ), _probe_map(probes)


def _context(mode: str, topology: str, overlap_budget: int, seed: int) -> ContextResult:
    problem, structure, _ = _combined_problem(mode, topology, overlap_budget, seed)
    checkpoint_x, checkpoint_error, proposals, checkpoint_fes = _proposal_payload(problem, structure, seed)
    probe_scheduler = GraphCoordinationScheduler(OverlapCoordinator(structure, EvaluationLedger(problem, total_budget=ARM_TOTAL_BUDGET_FES)))
    components = probe_scheduler.overlap_components
    if len(components) < 2:
        raise RuntimeError(f"expected at least two components, got {components}")
    # Select the component using the same online value rule as Gate 15.
    _, scheduler = _new_scheduler(problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes)
    scheduler.prime(proposals)
    probes = scheduler.value_probe(proposals)
    priorities = {item.component: item.priority_score for item in scheduler.prioritize(proposals)}
    selected = max(
        probes,
        key=lambda item: (
            item.estimated_gain,
            priorities[item.component],
            tuple(-value for value in item.component),
        ),
    ).component
    current, current_probes = _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, selected, arm="current", seed=seed)
    after, after_probes = _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, selected, arm="after_arbitration", seed=seed)
    shared, shared_probes = _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, selected, arm="owner_shared_core", seed=seed)
    full, full_probes = _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, selected, arm="owner_full", seed=seed)
    return ContextResult(
        mode=mode,
        topology=topology,
        overlap_budget=overlap_budget,
        seed=seed,
        selected_component=selected,
        component_count=len(components),
        probe_gains=current_probes,
        proposals_identical=True,
        probe_parity=current_probes == after_probes == shared_probes == full_probes,
        fe_parity=len({current.consumed_fes, after.consumed_fes, shared.consumed_fes, full.consumed_fes}) == 1,
        strict_best=all(item.strict_best for item in (current, after, shared, full)),
        current=current,
        after_arbitration=after,
        owner_shared_core=shared,
        owner_full=full,
        after_vs_current_gain=float(current.final_error - after.final_error),
        after_vs_shared_owner_gain=float(shared.final_error - after.final_error),
        after_vs_full_owner_gain=float(full.final_error - after.final_error),
    )


def run_gate(*, workers: int = 1) -> dict[str, object]:
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
    contexts = tuple(sorted(contexts, key=lambda item: (item.topology, item.overlap_budget, item.seed, item.mode)))
    timing = np.asarray([item.after_vs_current_gain for item in contexts], dtype=float)
    shared = np.asarray([item.after_vs_shared_owner_gain for item in contexts], dtype=float)
    full = np.asarray([item.after_vs_full_owner_gain for item in contexts], dtype=float)
    cells = tuple(
        {
            "topology": topology,
            "overlap_budget": budget,
            "context_count": sum(item.topology == topology and item.overlap_budget == budget for item in contexts),
            "complete": sum(item.topology == topology and item.overlap_budget == budget for item in contexts) == 2 * len(FRESH_SEEDS),
        }
        for topology in TOPOLOGIES
        for budget in OVERLAP_BUDGETS
    )
    checks = {
        "context_count_60": len(contexts) == 60,
        "at_least_two_components_all": all(item.component_count >= 2 for item in contexts),
        "cells_complete": all(item["complete"] for item in cells),
        "probe_parity": all(item.probe_parity for item in contexts),
        "proposal_parity": all(item.proposals_identical for item in contexts),
        "fe_parity": all(item.fe_parity for item in contexts),
        "strict_best": all(item.strict_best for item in contexts),
    }
    return {
        "schema_version": "arac-overlap-owner-control-diagnostic-gate16-v1",
        "protocol": {"contexts": 60, "probe_fes_per_component": PROBE_FES_PER_COMPONENT, "continuation_fes": CTP_BUDGET_FES, "arms": ("current", "after_arbitration", "owner_shared_core", "owner_full")},
        "context_count": len(contexts),
        "contexts": [asdict(item) for item in contexts],
        "cell_summary": cells,
        "summary": {
            "after_vs_current_win_or_tie": float(np.mean(timing >= 0.0)),
            "after_vs_current_median_gain": float(np.median(timing)),
            "after_vs_shared_owner_win_or_tie": float(np.mean(shared >= 0.0)),
            "after_vs_shared_owner_median_gain": float(np.median(shared)),
            "after_vs_full_owner_win_or_tie": float(np.mean(full >= 0.0)),
            "after_vs_full_owner_median_gain": float(np.median(full)),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/overlap_owner_control_diagnostic_gate16/confirmation_fresh.json"))
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
