"""Measure CTP repair variance for isomorphic pair-pair components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import GraphCoordinationScheduler, LocalProposal, OverlapCoordinator, OverlapStructure
from arac.runtime.ledger import EvaluationLedger
from experiments.oracle_gcb_gate3 import (
    DISPATCH_BUDGET_FES,
    TOPOLOGIES,
    _build,
    _repair_seed,
    _run_scheduler,
)


CONFIRMATION_SEEDS = tuple(range(2026081801, 2026081826))


def _problem(spec) -> OptimizationProblem:
    return _build(spec, 0)[0]


def _run_component(spec, proposals, component, repair_seed: int) -> float:
    problem = _problem(spec)
    incumbent = (2.0,) * problem.dimension
    ledger = EvaluationLedger(
        problem,
        total_budget=100,
        initial_incumbent=incumbent,
        initial_error=float(problem.objective(np.asarray(incumbent))),
    )
    coordinator = OverlapCoordinator(
        OverlapStructure(problem.dimension, spec.groups),
        ledger,
    )
    scheduler = GraphCoordinationScheduler(coordinator)
    scheduler.prime(proposals)
    result = coordinator.coordinate(
        component,
        scheduler._component_proposals(component, tuple(proposals)),
        ctp_budget_fes=DISPATCH_BUDGET_FES,
        ctp_seed=repair_seed,
    )
    if not result.ctp_triggered or result.ctp_consumed_fes != DISPATCH_BUDGET_FES:
        raise AssertionError("variance diagnostic did not trigger the frozen CTP repair")
    return float(coordinator.ledger.best_error)


def diagnose(*, proposal_seed: int, replicates: int) -> dict[str, object]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    spec = next(item for item in TOPOLOGIES if item.name == "pair_pair")
    _, proposals = _build(spec, proposal_seed)
    components = ((0, 1), (2, 3))
    selected_component = _run_scheduler(
        spec,
        proposals,
        seed=proposal_seed,
        selected="gcb",
    )[0]
    values = {
        str(component): [
            _run_component(spec, proposals, component, _repair_seed(proposal_seed, index))
            for index in range(replicates)
        ]
        for component in components
    }
    summary = {
        component: {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
        }
        for component, errors in values.items()
    }
    mean_gap = summary[str(components[1])]["mean_error"] - summary[str(components[0])]["mean_error"]
    lower_expected_error_component = components[1] if mean_gap < 0.0 else components[0]
    return {
        "proposal_seed": proposal_seed,
        "replicates": replicates,
        "values": values,
        "summary": summary,
        "component_23_minus_01_mean_error": mean_gap,
        "gcb_selected_component": list(selected_component),
        "lower_expected_error_component": list(lower_expected_error_component),
        "gcb_selected_lower_expected_error": selected_component == lower_expected_error_component,
    }


def _swap_pair_pair_proposals(proposals: tuple[LocalProposal, ...]) -> tuple[LocalProposal, ...]:
    variable_map = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
    group_map = {0: 2, 1: 3, 2: 0, 3: 1}
    swapped = []
    for proposal in proposals:
        swapped.append(
            LocalProposal(
                group=group_map[proposal.group],
                values=tuple((variable_map[variable], value) for variable, value in proposal.values),
                improvement=proposal.improvement,
                uncertainty=tuple(
                    (variable_map[variable], value) for variable, value in proposal.uncertainty
                ),
            )
        )
    return tuple(sorted(swapped, key=lambda item: item.group))


def check_component_exchange(seed: int) -> dict[str, object]:
    spec = next(item for item in TOPOLOGIES if item.name == "pair_pair")
    _, proposals = _build(spec, seed)
    original = _run_scheduler(spec, proposals, seed=seed, selected="gcb")
    swapped = _run_scheduler(
        spec,
        _swap_pair_pair_proposals(proposals),
        seed=seed,
        selected="gcb",
    )
    component_map = {(0, 1): (2, 3), (2, 3): (0, 1)}
    return {
        "seed": seed,
        "original_component": list(original[0]),
        "swapped_component": list(swapped[0]),
        "component_equivariant": swapped[0] == component_map[original[0]],
        "original_error": original[1],
        "swapped_error": swapped[1],
        "objective_invariant": bool(np.isclose(original[1], swapped[1], rtol=0.0, atol=1e-12)),
        "budget_invariant": original[2:4] == swapped[2:4],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-seeds", type=int, nargs="+", default=CONFIRMATION_SEEDS)
    parser.add_argument("--replicates", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("artifacts/oracle_gcb_gate3/component_variance.json"))
    args = parser.parse_args()
    trials = [
        diagnose(proposal_seed=proposal_seed, replicates=args.replicates)
        for proposal_seed in args.proposal_seeds
    ]
    payload = {
        "schema_version": "arac-gcb-component-variance-v1",
        "replicates": args.replicates,
        "trials": trials,
        "exchange_checks": [check_component_exchange(seed) for seed in args.proposal_seeds],
        "summary": {
            "gcb_selected_lower_expected_error": sum(
                trial["gcb_selected_lower_expected_error"] for trial in trials
            ),
            "runs": len(trials),
            "all_exchange_checks_pass": all(
                check["component_equivariant"]
                and check["objective_invariant"]
                and check["budget_invariant"]
                for check in (check_component_exchange(seed) for seed in args.proposal_seeds)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
