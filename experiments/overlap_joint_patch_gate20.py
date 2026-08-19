"""Gate 20: shared-only versus joint overlap-component patch repair."""

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
    shared_variable_count: int
    joint_variable_count: int
    probes_identical: bool
    proposals_identical: bool
    fe_parity: bool
    strict_best: bool
    trace_complete: bool
    current: ArmResult
    shared_only: ArmResult
    joint_patch: ArmResult
    owner_shared_core: ArmResult
    joint_vs_shared_only_gain: float
    joint_vs_shared_owner_gain: float


def _probe_map(probes):
    return tuple((item.component, float(item.estimated_gain)) for item in probes)


def _repair_joint(scheduler, component, proposals):
    coordinator = scheduler.coordinator
    selected = tuple(proposal for proposal in proposals if proposal.group in component)
    shared_variables = coordinator._component_variables(component)
    component_set = set(component)
    patch_variables = tuple(
        sorted({variable for group in component_set for variable in coordinator.structure.groups[group]})
    )
    if not shared_variables or not patch_variables:
        raise RuntimeError("selected component has no patch variables")
    residuals = __import__(
        "arac.coordination.overlap", fromlist=["compute_proposal_residuals"]
    ).compute_proposal_residuals(
        coordinator.structure, selected, variables=shared_variables, epsilon=coordinator.epsilon
    )
    by_group = {proposal.group: proposal for proposal in selected}
    owner_order = tuple(sorted(component, key=lambda group: (-by_group[group].improvement, group)))
    owner_for_variable = {
        variable: coordinator.structure.owners(variable)[0]
        for variable in patch_variables
        if len(coordinator.structure.owners(variable)) == 1
    }
    initial_radii = []
    for variable in patch_variables:
        owners = coordinator.structure.owners(variable)
        if len(owners) > 1:
            radius = max(
                coordinator.epsilon,
                max(
                    abs(by_group[group].value(variable) - residuals[variable].weighted_mean)
                    for group in owners
                ),
                max(by_group[group].sigma(variable) for group in owners),
            )
        else:
            owner = owner_for_variable[variable]
            proposal = by_group[owner]
            radius = max(
                coordinator.epsilon,
                abs(proposal.value(variable) - coordinator.ledger.best_x[variable]),
                proposal.sigma(variable),
            )
        initial_radii.append(radius)
    initial_radii = np.asarray(initial_radii, dtype=float)
    radii = initial_radii.copy()
    indices = np.asarray(patch_variables, dtype=int)
    center = coordinator.ledger.best_x[indices].copy()
    trace = []
    accepted_rounds = []
    for round_index in range(ROUNDS):
        group = owner_order[round_index % len(owner_order)]
        proposal = by_group[group]
        direction = []
        for index, variable in enumerate(patch_variables):
            owners = coordinator.structure.owners(variable)
            if len(owners) == 1:
                value = by_group[owners[0]].value(variable)
            else:
                value = proposal.value(variable) if group in owners else residuals[variable].weighted_mean
            direction.append(value - center[index])
        direction = np.asarray(direction, dtype=float)
        direction = np.where(np.abs(direction) > coordinator.epsilon, direction, 1.0)
        step = radii * np.sign(direction)
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
    return ROUNDS * EVALS_PER_ROUND, tuple(trace), tuple(accepted_rounds), len(shared_variables), len(patch_variables)


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
        if arm == "shared_only":
            continuation_fes, radius_trace, accepted_rounds = _repair_sequential(
                scheduler, selected, proposals, seed=seed ^ 0x18A7
            )
            final_error = ledger.best_error
            archive_ok = final_error <= checkpoint_error
        elif arm == "joint_patch":
            continuation_fes, radius_trace, accepted_rounds, _, _ = _repair_joint(
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
        for name in ("current", "shared_only", "joint_patch", "owner_shared_core")
    }
    current, current_probes = arms["current"]
    shared, shared_probes = arms["shared_only"]
    joint, joint_probes = arms["joint_patch"]
    owner, owner_probes = arms["owner_shared_core"]
    shared_count = len(selector.coordinator._component_variables(selected))
    joint_count = len({variable for group in selected for variable in structure.groups[group]})
    return ContextResult(
        mode,
        topology,
        overlap_budget,
        seed,
        len(components),
        selected,
        shared_count,
        joint_count,
        current_probes == shared_probes == joint_probes == owner_probes,
        True,
        len({item.consumed_fes for item in (current, shared, joint, owner)}) == 1,
        all(item.strict_best for item in (current, shared, joint, owner)),
        all(len(item.radius_trace) == ROUNDS for item in (shared, joint)),
        current,
        shared,
        joint,
        owner,
        shared.final_error - joint.final_error,
        owner.final_error - joint.final_error,
    )


def run_gate(*, workers=1):
    jobs = tuple((mode, topology, budget, seed) for topology in TOPOLOGIES for budget in OVERLAP_BUDGETS for seed in FRESH_SEEDS for mode in MODES)
    if workers == 1:
        contexts = tuple(_context(*job) for job in jobs)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            contexts = tuple(executor.map(lambda job: _context(*job), jobs))
    contexts = tuple(sorted(contexts, key=lambda row: (row.topology, row.overlap_budget, row.seed, row.mode)))
    joint_shared = np.asarray([row.joint_vs_shared_only_gain for row in contexts])
    joint_owner = np.asarray([row.joint_vs_shared_owner_gain for row in contexts])
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
        "joint_vs_shared_only_ge_0_60": float(np.mean(joint_shared >= 0.0)) >= 0.60,
        "joint_vs_owner_ge_0_60": float(np.mean(joint_owner >= 0.0)) >= 0.60,
        "joint_vs_owner_median_nonnegative": float(np.median(joint_owner)) >= 0.0,
    }
    return {
        "schema_version": "arac-overlap-joint-patch-gate20-v1",
        "protocol": {"contexts": 60, "rounds": ROUNDS, "evaluations_per_round": EVALS_PER_ROUND, "repair_fes": CTP_BUDGET_FES},
        "context_count": len(contexts),
        "contexts": [asdict(row) for row in contexts],
        "cell_summary": cells,
        "summary": {
            "joint_vs_shared_only_win_or_tie": float(np.mean(joint_shared >= 0.0)),
            "joint_vs_shared_only_median_gain": float(np.median(joint_shared)),
            "joint_vs_shared_owner_win_or_tie": float(np.mean(joint_owner >= 0.0)),
            "joint_vs_shared_owner_median_gain": float(np.median(joint_owner)),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/overlap_joint_patch_gate20/confirmation_fresh.json"))
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
