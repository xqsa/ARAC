"""Paired oracle diagnostic for graph-conditioned overlap dispatch."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import GraphCoordinationScheduler, LocalProposal, OverlapCoordinator, OverlapStructure
from arac.runtime.ledger import EvaluationLedger


DEFAULT_SEEDS = tuple(range(2026081701, 2026081726))
DISPATCH_BUDGET_FES = 32


@dataclass(frozen=True)
class TopologySpec:
    name: str
    groups: tuple[tuple[int, ...], ...]
    shared_variables: tuple[int, ...]


TOPOLOGIES = (
    TopologySpec(
        "pair_pair",
        ((0, 1), (1, 2), (3, 4), (4, 5)),
        (1, 4),
    ),
    TopologySpec(
        "pair_hub",
        ((0, 1), (1, 2), (1, 3), (4, 5), (5, 6)),
        (1, 5),
    ),
)


@dataclass(frozen=True)
class GcbTrialResult:
    topology: str
    seed: int
    selected_component: tuple[int, ...]
    canonical_component: tuple[int, ...]
    selected_priority: float
    canonical_priority: float
    gcb_error: float
    canonical_error: float
    gcb_gain: float
    gcb_consumed_fes: int
    canonical_consumed_fes: int
    gcb_ledger_fes: int
    canonical_ledger_fes: int
    gcb_archive_nonworsening: bool
    conflict_levels: tuple[str, ...]


def _problem(dimension: int) -> OptimizationProblem:
    weights = np.ones(dimension, dtype=float)
    if dimension == 7:
        # The hub's shared coordinate carries a larger objective loss. This
        # makes topology-aware dispatch test an urgency that is represented in
        # the local proposal evidence, rather than an unrelated outer weight.
        weights[1] = 8.0

    def objective(x: np.ndarray) -> float | np.ndarray:
        values = np.asarray(x, dtype=float)
        batch = values[np.newaxis, :] if values.ndim == 1 else values
        # Each component has a distinct target; shared coordinates create the
        # conflicting signal while all non-shared coordinates remain benign.
        result = np.sum(weights * (batch - 0.25) ** 2, axis=1)
        return float(result[0]) if values.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=dimension,
        lower_bounds=(-4.0,) * dimension,
        upper_bounds=(4.0,) * dimension,
    )


def _proposal(spec: TopologySpec, group: int, *, seed: int, case: str) -> LocalProposal:
    rng = np.random.default_rng(seed + group * 17)
    variables = spec.groups[group]
    shared = tuple(variable for variable in variables if variable in spec.shared_variables)
    values = []
    sigmas = []
    for variable in variables:
        if variable not in shared:
            value = 0.0
        elif case == "high":
            owners = tuple(
                owner
                for owner, owner_variables in enumerate(spec.groups)
                if variable in owner_variables
            )
            owner_index = owners.index(group)
            sign = -1.0 if owner_index % 2 == 0 else 1.0
            value = sign * (1.25 + 0.10 * rng.random())
        else:
            value = 0.10 + 0.03 * rng.normal()
        values.append((variable, float(value)))
        sigmas.append((variable, 0.08 if case == "high" else 0.20))
    contribution = 8.0 if spec.name == "pair_hub" and group < 3 else 1.0
    return LocalProposal(
        group=group,
        values=tuple(values),
        improvement=contribution,
        uncertainty=tuple(sigmas),
    )


def _ledger(problem: OptimizationProblem, dimension: int, budget: int) -> EvaluationLedger:
    incumbent = (2.0,) * dimension
    return EvaluationLedger(
        problem,
        total_budget=budget,
        initial_incumbent=incumbent,
        initial_error=float(problem.objective(np.asarray(incumbent))),
    )


def _build(spec: TopologySpec, seed: int) -> tuple[OptimizationProblem, tuple[LocalProposal, ...]]:
    problem = _problem(max(max(group) for group in spec.groups) + 1)
    proposals = tuple(
        _proposal(spec, group, seed=seed, case="high")
        for group in range(len(spec.groups))
    )
    return problem, proposals


def _repair_seed(seed: int, replicate: int = 0) -> int:
    """Derive a CTP stream independent of every proposal-generation stream."""

    state = np.random.SeedSequence((seed, 0x435450, replicate)).generate_state(1)
    return int(state[0])


def _run_scheduler(
    spec: TopologySpec,
    proposals: tuple[LocalProposal, ...],
    *,
    seed: int,
    selected: str,
) -> tuple[tuple[int, ...], float, int, int, bool, tuple[str, ...]]:
    problem = _problem(max(max(group) for group in spec.groups) + 1)
    coordinator = OverlapCoordinator(
        OverlapStructure(dimension=problem.dimension, groups=spec.groups),
        _ledger(problem, problem.dimension, budget=100),
    )
    scheduler = GraphCoordinationScheduler(coordinator)
    scheduler.prime(proposals)
    priorities = scheduler.prioritize(proposals)
    if selected == "gcb":
        selected_priority = priorities[0]
    else:
        selected_priority = next(
            priority for priority in priorities if priority.component == min(priority.component for priority in priorities)
        )
    result = scheduler.dispatch(
        proposals,
        total_ctp_budget_fes=DISPATCH_BUDGET_FES,
        max_components=1,
        seed=_repair_seed(seed),
    ) if selected == "gcb" else _canonical_dispatch(
        scheduler,
        proposals,
        selected_priority,
        _repair_seed(seed),
    )
    return (
        selected_priority.component,
        float(coordinator.ledger.best_error),
        result.consumed_ctp_fes,
        result.ledger_fes,
        all(event.best_error_after <= event.best_error_before for event in result.events),
        tuple(priority.conflict_level.value for priority in priorities),
    )


def _canonical_dispatch(
    scheduler: GraphCoordinationScheduler,
    proposals: tuple[LocalProposal, ...],
    priority,
    seed: int,
):
    component_proposals = scheduler._component_proposals(priority.component, proposals)
    result = scheduler.coordinator.coordinate(
        priority.component,
        component_proposals,
        ctp_budget_fes=DISPATCH_BUDGET_FES,
        ctp_seed=seed,
    )
    from arac.coordination.gcb import DispatchEvent, GcbDispatchResult

    return GcbDispatchResult(
        priorities=(priority,),
        priming_results=scheduler._priming_results,
        events=(
            DispatchEvent(
                component=priority.component,
                priority_score=priority.priority_score,
                requested_ctp_fes=DISPATCH_BUDGET_FES,
                consumed_ctp_fes=result.ctp_consumed_fes,
                ledger_fes=len(result.candidates) + result.ctp_consumed_fes,
                best_error_before=result.best_error_before,
                best_error_after=result.best_error_after,
                accepted_candidate=result.accepted_candidate,
            ),
        ),
        total_ctp_budget_fes=DISPATCH_BUDGET_FES,
        consumed_ctp_fes=result.ctp_consumed_fes,
        unspent_ctp_fes=DISPATCH_BUDGET_FES - result.ctp_consumed_fes,
        ledger_fes=scheduler.coordinator.ledger.count,
    )


def run_trial(topology: TopologySpec, seed: int) -> GcbTrialResult:
    problem, proposals = _build(topology, seed)
    gcb_component, gcb_error, gcb_fes, gcb_ledger_fes, gcb_nonworsening, levels = _run_scheduler(
        topology, proposals, seed=seed, selected="gcb"
    )
    canonical_component, canonical_error, canonical_fes, canonical_ledger_fes, _, _ = _run_scheduler(
        topology, proposals, seed=seed, selected="canonical"
    )
    # Recompute priorities on the same frozen proposal evidence for the audit.
    audit_coordinator = OverlapCoordinator(
        OverlapStructure(problem.dimension, topology.groups),
        _ledger(problem, problem.dimension, budget=100),
    )
    audit_scheduler = GraphCoordinationScheduler(audit_coordinator)
    audit_scheduler.prime(proposals)
    priorities = audit_scheduler.prioritize(proposals)
    return GcbTrialResult(
        topology=topology.name,
        seed=seed,
        selected_component=gcb_component,
        canonical_component=canonical_component,
        selected_priority=priorities[0].priority_score,
        canonical_priority=next(
            item.priority_score for item in priorities if item.component == canonical_component
        ),
        gcb_error=gcb_error,
        canonical_error=canonical_error,
        gcb_gain=canonical_error - gcb_error,
        gcb_consumed_fes=gcb_fes,
        canonical_consumed_fes=canonical_fes,
        gcb_ledger_fes=gcb_ledger_fes,
        canonical_ledger_fes=canonical_ledger_fes,
        gcb_archive_nonworsening=gcb_nonworsening,
        conflict_levels=levels,
    )


def _task(task: tuple[str, int]) -> GcbTrialResult:
    name, seed = task
    topology = next(item for item in TOPOLOGIES if item.name == name)
    return run_trial(topology, seed)


def _summary(trials: list[GcbTrialResult]) -> dict[str, object]:
    result: dict[str, object] = {}
    for topology in (item.name for item in TOPOLOGIES):
        values = [trial for trial in trials if trial.topology == topology]
        gains = np.asarray([trial.gcb_gain for trial in values], dtype=float)
        result[topology] = {
            "runs": len(values),
            "selection_accuracy": sum(
                trial.selected_component != trial.canonical_component
                and trial.selected_priority > trial.canonical_priority
                for trial in values
            )
            / len(values),
            "gcb_wins": int(np.sum(gains > 1e-12)),
            "gcb_losses": int(np.sum(gains < -1e-12)),
            "gcb_ties": int(np.sum(np.abs(gains) <= 1e-12)),
            "median_gain": float(np.median(gains)),
            "exact_budget": all(
                trial.gcb_consumed_fes == DISPATCH_BUDGET_FES
                and trial.canonical_consumed_fes == DISPATCH_BUDGET_FES
                and trial.gcb_ledger_fes == trial.canonical_ledger_fes
                for trial in values
            ),
            "archive_nonworsening": all(trial.gcb_archive_nonworsening for trial in values),
        }
    checks = {
        "all_topologies_have_90pct_gcb_wins_or_ties": all(
            (item["gcb_wins"] + item["gcb_ties"]) / item["runs"] >= 0.9
            for item in result.values()
        ),
        "all_budgets_exact": all(item["exact_budget"] for item in result.values()),
        "all_archives_nonworsening": all(item["archive_nonworsening"] for item in result.values()),
    }
    result["gate_checks"] = checks
    result["gate_passed"] = all(checks.values())
    return result


def run_diagnostic(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    *,
    workers: int = 1,
) -> dict[str, object]:
    tasks = [(topology.name, seed) for seed in seeds for topology in TOPOLOGIES]
    if workers == 1:
        trials = [_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trials = list(executor.map(_task, tasks))
    return {
        "schema_version": "arac-oracle-gcb-gate3-v2",
        "seeds": list(seeds),
        "dispatch_budget_fes": DISPATCH_BUDGET_FES,
        "trials": [asdict(trial) for trial in trials],
        "summary": _summary(trials),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--seed-count", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=Path("artifacts/oracle_gcb_gate3/result.json"))
    args = parser.parse_args()
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    payload = run_diagnostic(seeds, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
