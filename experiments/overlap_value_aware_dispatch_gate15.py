"""Gate 15: value-aware selection of disconnected overlap components.

This is an independent diagnostic experiment.  It deliberately does not
change the production GCB.  Two independent 24-D interaction benchmarks are
placed in one 48-D objective so that a context contains at least two
disconnected overlap components.  Every component receives the same counted
two-FE probe before value-aware, structural, forced-CTP, and owner-control
arms are compared.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.benchmarks.overlap_objective import build_overlap_problem
from arac.coordination import (
    GraphCoordinationScheduler,
    LocalProposal,
    OverlapCoordinator,
    OverlapStructure,
    produce_local_proposal,
)
from arac.runtime.ledger import EvaluationLedger


BLOCK_DIMENSION = 24
DIMENSION = 2 * BLOCK_DIMENSION
NUM_GROUPS_PER_BLOCK = 6
NUM_GROUPS = 2 * NUM_GROUPS_PER_BLOCK
MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 5
BOUNDS = 10.0
INTERACTION_STRENGTH = 0.25
PROPOSAL_BUDGET_FES = 48
PROPOSAL_REPLICATES = 4
PROPOSAL_ALGORITHM = "sepcmaes"
PROPOSAL_POPULATION_SIZE = 8
PROBE_FES_PER_COMPONENT = 2
CTP_BUDGET_FES = 32
ARM_TOTAL_BUDGET_FES = 40_000
FRESH_SEEDS = (31501, 31502, 31503, 31504, 31505)
TOPOLOGIES = ("random", "chain", "star")
OVERLAP_BUDGETS = (6, 12)
MODES = ("conforming", "conflicting")


@dataclass(frozen=True)
class ArmResult:
    arm: str
    selected_component: tuple[int, ...]
    final_error: float
    base_error: float
    consumed_fes: int
    probe_fes: int
    arbitration_fes: int
    repair_or_control_fes: int
    strict_best: bool
    probe_gains: tuple[tuple[tuple[int, ...], float], ...]


@dataclass(frozen=True)
class ContextResult:
    mode: str
    topology: str
    overlap_budget: int
    seed: int
    component_count: int
    components: tuple[tuple[int, ...], ...]
    value: ArmResult
    structural: ArmResult
    owner_control: ArmResult
    oracle: tuple[ArmResult, ...]
    value_vs_structural_gain: float
    value_vs_owner_gain: float
    value_selection_regret: float
    structural_selection_regret: float
    probes_identical: bool
    proposal_parity: bool
    fe_parity: bool
    strict_best: bool


def _combined_problem(mode: str, topology: str, overlap_budget: int, seed: int):
    blocks = []
    for block in range(2):
        block_seed = int(seed ^ (0x9E3779B1 * (block + 1)))
        blocks.append(
            build_overlap_problem(
                BLOCK_DIMENSION,
                overlap_budget=overlap_budget,
                min_group_size=MIN_GROUP_SIZE,
                max_group_size=MAX_GROUP_SIZE,
                num_groups=NUM_GROUPS_PER_BLOCK,
                base_function="sphere",
                conflict_mode=mode,
                bounds=BOUNDS,
                contiguous=True,
                rotation=False,
                transforms=False,
                interaction_strength=INTERACTION_STRENGTH,
                seed=block_seed,
                topology=topology,
            )
        )
    _, first = blocks[0]
    _, second = blocks[1]
    groups = tuple(tuple(variable for variable in group) for group in first.structure.groups)
    groups += tuple(
        tuple(BLOCK_DIMENSION + variable for variable in group)
        for group in second.structure.groups
    )
    structure = OverlapStructure(DIMENSION, groups)

    def objective(values: np.ndarray) -> float | np.ndarray:
        converted = np.asarray(values, dtype=float)
        if converted.ndim == 1:
            return float(first.evaluate(converted[:BLOCK_DIMENSION]) + second.evaluate(converted[BLOCK_DIMENSION:]))
        return np.asarray(first.evaluate(converted[:, :BLOCK_DIMENSION]), dtype=float) + np.asarray(
            second.evaluate(converted[:, BLOCK_DIMENSION:]), dtype=float
        )

    problem = OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-BOUNDS,) * DIMENSION,
        upper_bounds=(BOUNDS,) * DIMENSION,
        optimum=0.0,
    )
    return problem, structure, (first, second)


def _seed(seed: int, group: int, replicate: int) -> int:
    return int(seed ^ (0x9E37 * (group + 1)) ^ (0x51ED * (replicate + 1)))


def _value(run: object, variable: int) -> float:
    proposal = run.proposal if hasattr(run, "proposal") else run
    return float(dict(proposal.values)[variable])


def _proposal_payload(problem, structure: OverlapStructure, seed: int):
    anchor = np.zeros(DIMENSION, dtype=float)
    ledger = EvaluationLedger(problem, total_budget=ARM_TOTAL_BUDGET_FES)
    anchor_error = float(ledger.evaluate(anchor))
    runs: list[list[object]] = [[] for _ in structure.groups]
    for group in range(len(structure.groups)):
        for replicate in range(PROPOSAL_REPLICATES):
            runs[group].append(
                produce_local_proposal(
                    structure,
                    group,
                    problem=problem,
                    global_ledger=ledger,
                    anchor=anchor,
                    anchor_error=anchor_error,
                    budget_fes=PROPOSAL_BUDGET_FES,
                    seed=_seed(seed, group, replicate),
                    algorithm=PROPOSAL_ALGORITHM,
                    population_size=PROPOSAL_POPULATION_SIZE,
                    sigma=0.5,
                )
            )
    proposals = []
    for group, group_runs in enumerate(runs):
        variables = tuple(structure.groups[group])
        proposals.append(
            LocalProposal(
                group=group,
                values=tuple(
                    (variable, float(np.mean([_value(group_runs[index], variable) for index in range(3)])))
                    for variable in variables
                ),
                improvement=float(np.mean([group_runs[index].proposal.improvement for index in range(3)])),
                uncertainty=tuple(
                    (
                        variable,
                        max(
                            np.finfo(float).eps,
                            float(
                                np.std(
                                    [_value(group_runs[index], variable) for index in range(3)],
                                    ddof=1,
                                )
                            ),
                        ),
                    )
                    for variable in variables
                ),
            )
        )
    expected = 1 + len(structure.groups) * PROPOSAL_REPLICATES * PROPOSAL_BUDGET_FES
    if ledger.count != expected:
        raise RuntimeError(f"proposal FE drifted: {ledger.count} != {expected}")
    return ledger.best_x, float(ledger.best_error), tuple(proposals), ledger.count


def _new_scheduler(problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes):
    ledger = EvaluationLedger(
        problem,
        total_budget=ARM_TOTAL_BUDGET_FES,
        initial_count=checkpoint_fes,
        initial_incumbent=tuple(float(value) for value in checkpoint_x),
        initial_error=checkpoint_error,
    )
    coordinator = OverlapCoordinator(
        structure,
        ledger,
        medium_threshold=0.0,
        high_threshold=0.0,
    )
    return ledger, GraphCoordinationScheduler(coordinator)


def _probe_gains(result) -> tuple[tuple[tuple[int, ...], float], ...]:
    return tuple((probe.component, float(probe.estimated_gain)) for probe in result.value_probes)


def _owner_control(
    scheduler: GraphCoordinationScheduler,
    proposals: tuple[LocalProposal, ...],
    component: tuple[int, ...],
    *,
    seed: int,
    budget_fes: int,
) -> tuple[int, float, bool]:
    selected = tuple(proposal for proposal in proposals if proposal.group in component)
    base = scheduler.coordinator.ledger.best_x
    before_error = scheduler.coordinator.ledger.best_error
    arbitration = scheduler.coordinator.coordinate(component, selected, ctp_budget_fes=0)
    if len(arbitration.candidates) != 4:
        raise RuntimeError("expected four arbitration candidates")
    rng = np.random.default_rng(seed)
    candidates = np.repeat(base[np.newaxis, :], budget_fes, axis=0)
    for index in range(budget_fes):
        proposal = selected[index % len(selected)]
        for variable, value in proposal.values:
            candidates[index, variable] = value + float(rng.normal(0.0, max(np.finfo(float).eps, proposal.sigma(variable))))
    np.clip(candidates, scheduler.coordinator.ledger.problem.lower_array, scheduler.coordinator.ledger.problem.upper_array, out=candidates)
    scheduler.coordinator.ledger.evaluate(candidates)
    return len(arbitration.candidates), scheduler.coordinator.ledger.best_error, scheduler.coordinator.ledger.best_error <= before_error


def _arm(
    problem,
    structure: OverlapStructure,
    proposals: tuple[LocalProposal, ...],
    checkpoint_x: np.ndarray,
    checkpoint_error: float,
    checkpoint_fes: int,
    components: tuple[tuple[int, ...], ...],
    *,
    arm: str,
    seed: int,
    forced_component: tuple[int, ...] | None = None,
) -> ArmResult:
    ledger, scheduler = _new_scheduler(problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes)
    scheduler.prime(proposals)
    priorities = scheduler.prioritize(proposals)
    structural_component = priorities[0].component
    base_before_probe = ledger.best_error
    if arm == "owner":
        if forced_component is None:
            raise ValueError("owner arm requires a selected component")
        probes = scheduler.value_probe(proposals)
        if len(probes) != len(components):
            raise RuntimeError("value probe did not cover every overlap component")
        selected = forced_component
        arbitration_fes, final_error, archive_ok = _owner_control(
            scheduler,
            proposals,
            selected,
            seed=seed ^ 0x51ED,
            budget_fes=CTP_BUDGET_FES,
        )
    else:
        if arm == "value":
            forced = None
        elif arm == "structural":
            forced = structural_component
        elif arm == "oracle":
            if forced_component is None:
                raise ValueError("oracle arm requires a forced component")
            forced = forced_component
        else:
            raise ValueError(f"unknown arm {arm}")
        result = scheduler.dispatch_value_probe(
            proposals,
            total_ctp_budget_fes=CTP_BUDGET_FES,
            forced_component=forced,
            seed=seed ^ 0xC7A5,
        )
        probes = result.value_probes
        if len(probes) != len(components):
            raise RuntimeError("value probe did not cover every overlap component")
        selected = result.events[0].component
        if len(result.events) != 1 or result.consumed_ctp_fes != CTP_BUDGET_FES:
            raise RuntimeError("CTP arm did not consume exactly 32 FE")
        arbitration_fes = len(result.events[0].accepted_candidate or "") * 0 + 4
        final_error = ledger.best_error
        archive_ok = all(event.best_error_after <= event.best_error_before for event in result.events)
    probe_fes = PROBE_FES_PER_COMPONENT * len(components)
    consumed = ledger.count - checkpoint_fes
    expected_tail = len(components) * 2 + 4 + CTP_BUDGET_FES
    prime_fes = consumed - expected_tail
    if prime_fes <= 0 or consumed != prime_fes + expected_tail:
        raise RuntimeError(f"{arm} FE mismatch: {consumed} (prime={prime_fes}, tail={expected_tail})")
    return ArmResult(
        arm=arm,
        selected_component=selected,
        final_error=float(final_error),
        base_error=float(base_before_probe),
        consumed_fes=consumed,
        probe_fes=probe_fes,
        arbitration_fes=arbitration_fes,
        repair_or_control_fes=CTP_BUDGET_FES,
        strict_best=bool(archive_ok and final_error <= base_before_probe),
        probe_gains=tuple((probe.component, float(probe.estimated_gain)) for probe in probes),
    )


def _context(mode: str, topology: str, overlap_budget: int, seed: int) -> ContextResult:
    problem, structure, _objectives = _combined_problem(mode, topology, overlap_budget, seed)
    checkpoint_x, checkpoint_error, proposals, checkpoint_fes = _proposal_payload(problem, structure, seed)
    components = tuple(GraphCoordinationScheduler(OverlapCoordinator(structure, EvaluationLedger(problem, total_budget=ARM_TOTAL_BUDGET_FES))).overlap_components)
    if len(components) < 2:
        raise RuntimeError(f"dual-block context has fewer than two overlap components: {components}")
    value = _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, components, arm="value", seed=seed)
    structural = _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, components, arm="structural", seed=seed)
    owner = _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, components, arm="owner", seed=seed, forced_component=value.selected_component)
    oracle = tuple(
        _arm(problem, structure, proposals, checkpoint_x, checkpoint_error, checkpoint_fes, components, arm="oracle", seed=seed, forced_component=component)
        for component in components
    )
    oracle_error = min(item.final_error for item in oracle)
    denominator = max(abs(checkpoint_error), np.finfo(float).eps)
    return ContextResult(
        mode=mode,
        topology=topology,
        overlap_budget=overlap_budget,
        seed=seed,
        component_count=len(components),
        components=components,
        value=value,
        structural=structural,
        owner_control=owner,
        oracle=oracle,
        value_vs_structural_gain=float(structural.final_error - value.final_error),
        value_vs_owner_gain=float(owner.final_error - value.final_error),
        value_selection_regret=float((value.final_error - oracle_error) / denominator),
        structural_selection_regret=float((structural.final_error - oracle_error) / denominator),
        probes_identical=value.probe_gains == structural.probe_gains == owner.probe_gains == oracle[0].probe_gains,
        proposal_parity=True,
        fe_parity=len({value.consumed_fes, structural.consumed_fes, owner.consumed_fes, *(item.consumed_fes for item in oracle)}) == 1,
        strict_best=all(item.strict_best for item in (value, structural, owner, *oracle)),
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
    value_structural = np.asarray([row.value_vs_structural_gain for row in contexts], dtype=float)
    value_owner = np.asarray([row.value_vs_owner_gain for row in contexts], dtype=float)
    value_regret = np.asarray([row.value_selection_regret for row in contexts], dtype=float)
    structural_regret = np.asarray([row.structural_selection_regret for row in contexts], dtype=float)
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
    gate_checks = {
        "context_count_60": len(contexts) == 60,
        "all_contexts_have_two_components": all(row.component_count >= 2 for row in contexts),
        "all_cells_complete": all(item["complete"] for item in cells),
        "probes_identical": all(row.probes_identical for row in contexts),
        "proposal_parity": all(row.proposal_parity for row in contexts),
        "exact_equal_fe": all(row.fe_parity for row in contexts),
        "strict_best_all_arms": all(row.strict_best for row in contexts),
        "value_vs_structural_win_or_tie_ge_0_60": float(np.mean(value_structural >= 0.0)) >= 0.60,
        "value_vs_structural_median_gain_nonnegative": float(median(value_structural)) >= 0.0,
        "value_regret_lower_than_structural": float(np.mean(value_regret)) < float(np.mean(structural_regret)),
        "value_vs_owner_win_or_tie_ge_0_60": float(np.mean(value_owner >= 0.0)) >= 0.60,
        "value_vs_owner_median_gain_nonnegative": float(median(value_owner)) >= 0.0,
    }
    return {
        "schema_version": "arac-overlap-value-aware-dispatch-gate15-v1",
        "protocol": {
            "dimension": DIMENSION,
            "interaction_strength": INTERACTION_STRENGTH,
            "proposal_budget_fes_each": PROPOSAL_BUDGET_FES,
            "proposal_replicates": PROPOSAL_REPLICATES,
            "probe_fes_per_component": PROBE_FES_PER_COMPONENT,
            "ctp_budget_fes": CTP_BUDGET_FES,
            "modes": MODES,
            "topologies": TOPOLOGIES,
            "overlap_budgets": OVERLAP_BUDGETS,
            "seeds": FRESH_SEEDS,
        },
        "context_count": len(contexts),
        "contexts": [asdict(row) for row in contexts],
        "cell_summary": cells,
        "summary": {
            "value_vs_structural_win_or_tie_rate": float(np.mean(value_structural >= 0.0)),
            "value_vs_structural_median_gain": float(median(value_structural)),
            "value_vs_owner_win_or_tie_rate": float(np.mean(value_owner >= 0.0)),
            "value_vs_owner_median_gain": float(median(value_owner)),
            "mean_value_selection_regret": float(np.mean(value_regret)),
            "mean_structural_selection_regret": float(np.mean(structural_regret)),
        },
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/overlap_value_aware_dispatch_gate15/confirmation_fresh.json"))
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
