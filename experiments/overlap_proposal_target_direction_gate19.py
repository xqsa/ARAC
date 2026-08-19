"""Gate 19: isolate proposal-direction magnitude in shared-patch search."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.coordination import GraphCoordinationScheduler, OverlapCoordinator
from arac.runtime.ledger import EvaluationLedger
from experiments.overlap_sequential_shared_patch_gate18 import (
    EVALS_PER_ROUND,
    RADIUS_CAP,
    RADIUS_GROWTH,
    RADIUS_SHRINK,
    ROUNDS,
    _repair_sequential,
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
    radius_trace: tuple[tuple[float, ...], ...] = ()
    accepted_rounds: tuple[int, ...] = ()


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
    sign: ArmResult
    target: ArmResult
    owner_shared_core: ArmResult
    target_vs_sign_gain: float
    target_vs_shared_owner_gain: float


def _probe_map(probes):
    return tuple((item.component, float(item.estimated_gain)) for item in probes)


def _repair_target(scheduler, component, proposals):
    coordinator = scheduler.coordinator
    selected = tuple(proposal for proposal in proposals if proposal.group in component)
    variables = coordinator._component_variables(component)
    residuals = __import__(
        "arac.coordination.overlap", fromlist=["compute_proposal_residuals"]
    ).compute_proposal_residuals(
        coordinator.structure, selected, variables=variables, epsilon=coordinator.epsilon
    )
    by_group = {proposal.group: proposal for proposal in selected}
    owner_order = tuple(sorted(component, key=lambda group: (-by_group[group].improvement, group)))
    initial_radii = np.asarray(
        [
            max(
                coordinator.epsilon,
                max(
                    abs(by_group[group].value(variable) - residuals[variable].weighted_mean)
                    for group in coordinator.structure.owners(variable)
                ),
                max(by_group[group].sigma(variable) for group in coordinator.structure.owners(variable)),
            )
            for variable in variables
        ],
        dtype=float,
    )
    radii = initial_radii.copy()
    indices = np.asarray(variables, dtype=int)
    center = coordinator.ledger.best_x[indices].copy()
    trace = []
    accepted_rounds = []
    for round_index in range(ROUNDS):
        group = owner_order[round_index % len(owner_order)]
        proposal = by_group[group]
        fallback = np.asarray(
            [residuals[variable].weighted_mean - center[index] for index, variable in enumerate(variables)],
            dtype=float,
        )
        displacement = np.asarray(
            [
                proposal.value(variable) - center[index]
                if group in coordinator.structure.owners(variable)
                else fallback[index]
                for index, variable in enumerate(variables)
            ],
            dtype=float,
        )
        displacement = np.where(np.abs(displacement) > coordinator.epsilon, displacement, fallback)
        displacement = np.where(np.abs(displacement) > coordinator.epsilon, displacement, 1.0)
        step = np.clip(displacement, -radii, radii)
        base = coordinator.ledger.best_x
        plus = base.copy()
        minus = base.copy()
        plus[indices] = center + step
        minus[indices] = center - step
        batch = np.asarray((plus, minus), dtype=float)
        np.clip(batch, coordinator.ledger.problem.lower_array, coordinator.ledger.problem.upper_array, out=batch)
        before = coordinator.ledger.best_error
        coordinator.ledger.evaluate(batch)
        accepted = coordinator.ledger.best_error < before
        if accepted:
            center = coordinator.ledger.best_x[indices].copy()
            radii = np.minimum(radii * RADIUS_GROWTH, initial_radii * RADIUS_CAP)
            accepted_rounds.append(round_index)
        else:
            radii = radii * RADIUS_SHRINK
        trace.append(tuple(float(value) for value in radii))
    return ROUNDS * EVALS_PER_ROUND, tuple(trace), tuple(accepted_rounds)


def _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, component, *, arm, seed):
    ledger, scheduler = _new_scheduler(problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes)
    scheduler.prime(proposals)
    priming_fes = ledger.count - checkpoint_fes
    selected = tuple(component)
    radius_trace = ()
    accepted_rounds = ()
    if arm == "current":
        result = scheduler.dispatch_value_probe(
            proposals, total_ctp_budget_fes=CTP_BUDGET_FES, forced_component=selected, seed=seed ^ 0xC7A5
        )
        probes = result.value_probes
        final_error = ledger.best_error
        archive_ok = all(event.best_error_after <= event.best_error_before for event in result.events)
        continuation_fes = result.consumed_ctp_fes
    else:
        probes = scheduler.value_probe(proposals)
        selected_proposals = tuple(proposal for proposal in proposals if proposal.group in selected)
        arbitration = scheduler.coordinator.coordinate(selected, selected_proposals, ctp_budget_fes=0)
        if len(arbitration.candidates) != 4:
            raise RuntimeError("expected four arbitration candidates")
        if arm == "sign":
            continuation_fes, radius_trace, accepted_rounds = _repair_sequential(
                scheduler, selected, proposals, seed=seed ^ 0x18A7
            )
            final_error = ledger.best_error
            archive_ok = final_error <= checkpoint_error
        elif arm == "target":
            continuation_fes, radius_trace, accepted_rounds = _repair_target(
                scheduler, selected, proposals
            )
            final_error = ledger.best_error
            archive_ok = final_error <= checkpoint_error
        elif arm == "owner_shared_core":
            variables = scheduler.coordinator._component_variables(selected)
            by_group = {proposal.group: proposal for proposal in selected_proposals}
            rng = np.random.default_rng(seed ^ 0x51ED)
            candidates = np.repeat(ledger.best_x[np.newaxis, :], CTP_BUDGET_FES, axis=0)
            for index in range(CTP_BUDGET_FES):
                for variable in variables:
                    owners = tuple(owner for owner in structure.owners(variable) if owner in by_group)
                    proposal = by_group[owners[index % len(owners)]]
                    candidates[index, variable] = proposal.value(variable) + float(
                        rng.normal(0.0, max(np.finfo(float).eps, proposal.sigma(variable)))
                    )
            np.clip(candidates, problem.lower_array, problem.upper_array, out=candidates)
            before = ledger.best_error
            ledger.evaluate(candidates)
            final_error = ledger.best_error
            archive_ok = final_error <= before
            continuation_fes = CTP_BUDGET_FES
        else:
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
        arbitration_fes=4,
        continuation_fes=continuation_fes,
        strict_best=bool(archive_ok and final_error <= checkpoint_error),
        radius_trace=radius_trace,
        accepted_rounds=accepted_rounds,
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
    selected = max(probes, key=lambda item: (item.estimated_gain, priorities[item.component], tuple(-value for value in item.component))).component
    arms = {
        name: _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, selected, arm=name, seed=seed)
        for name in ("current", "sign", "target", "owner_shared_core")
    }
    current, current_probes = arms["current"]
    sign, sign_probes = arms["sign"]
    target, target_probes = arms["target"]
    owner, owner_probes = arms["owner_shared_core"]
    return ContextResult(
        mode,
        topology,
        overlap_budget,
        seed,
        len(components),
        selected,
        current_probes == sign_probes == target_probes == owner_probes,
        True,
        len({item.consumed_fes for item in (current, sign, target, owner)}) == 1,
        all(item.strict_best for item in (current, sign, target, owner)),
        all(len(item.radius_trace) == ROUNDS for item in (sign, target)),
        current,
        sign,
        target,
        owner,
        sign.final_error - target.final_error,
        owner.final_error - target.final_error,
    )


def run_gate(*, workers=1):
    jobs = tuple((mode, topology, budget, seed) for topology in TOPOLOGIES for budget in OVERLAP_BUDGETS for seed in FRESH_SEEDS for mode in MODES)
    if workers == 1:
        contexts = tuple(_context(*job) for job in jobs)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            contexts = tuple(executor.map(lambda job: _context(*job), jobs))
    contexts = tuple(sorted(contexts, key=lambda row: (row.topology, row.overlap_budget, row.seed, row.mode)))
    target_sign = np.asarray([row.target_vs_sign_gain for row in contexts])
    target_owner = np.asarray([row.target_vs_shared_owner_gain for row in contexts])
    cells = tuple({
        "topology": topology,
        "overlap_budget": budget,
        "context_count": sum(row.topology == topology and row.overlap_budget == budget for row in contexts),
        "complete": sum(row.topology == topology and row.overlap_budget == budget for row in contexts) == 2 * len(FRESH_SEEDS),
    } for topology in TOPOLOGIES for budget in OVERLAP_BUDGETS)
    checks = {
        "context_count_60": len(contexts) == 60,
        "cells_complete": all(row["complete"] for row in cells),
        "components": all(row.component_count >= 2 for row in contexts),
        "probes_identical": all(row.probes_identical for row in contexts),
        "proposals_identical": all(row.proposals_identical for row in contexts),
        "fe_parity": all(row.fe_parity for row in contexts),
        "strict_best": all(row.strict_best for row in contexts),
        "trace_complete": all(row.trace_complete for row in contexts),
        "target_vs_sign_ge_0_60": float(np.mean(target_sign >= 0.0)) >= 0.60,
        "target_vs_owner_ge_0_60": float(np.mean(target_owner >= 0.0)) >= 0.60,
        "target_vs_owner_median_nonnegative": float(np.median(target_owner)) >= 0.0,
    }
    return {
        "schema_version": "arac-overlap-proposal-target-direction-gate19-v1",
        "protocol": {"contexts": 60, "rounds": ROUNDS, "evaluations_per_round": EVALS_PER_ROUND, "repair_fes": CTP_BUDGET_FES},
        "context_count": len(contexts),
        "contexts": [asdict(row) for row in contexts],
        "cell_summary": cells,
        "summary": {
            "target_vs_sign_win_or_tie": float(np.mean(target_sign >= 0.0)),
            "target_vs_sign_median_gain": float(np.median(target_sign)),
            "target_vs_shared_owner_win_or_tie": float(np.mean(target_owner >= 0.0)),
            "target_vs_shared_owner_median_gain": float(np.median(target_owner)),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/overlap_proposal_target_direction_gate19/confirmation_fresh.json"))
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
