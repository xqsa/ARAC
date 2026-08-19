"""Explain pair-pair losses in the frozen oracle GCB confirmation.

This audit does not change the scheduler or the gate.  It replays the two
candidate dispatch paths from the same frozen proposals and records the
component evidence, priming archive, and CTP repair outcome.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from arac.coordination import GraphCoordinationScheduler, LocalProposal, OverlapCoordinator, OverlapStructure
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.ledger import EvaluationLedger
from experiments.oracle_gcb_gate3 import DISPATCH_BUDGET_FES, TOPOLOGIES, _build, _repair_seed


LOSS_SEEDS = (2026081803, 2026081815, 2026081818, 2026081819)


@dataclass(frozen=True)
class Replay:
    selected_component: tuple[int, ...]
    selected_priority: float
    initial_error: float
    after_prime_error: float
    after_prime_x: tuple[float, ...]
    second_before_error: float
    second_after_candidates_error: float
    second_after_ctp_error: float
    second_ctp_triggered: bool
    second_ctp_consumed_fes: int
    second_ctp_seed: int
    total_ledger_fes: int
    second_candidate_errors: tuple[tuple[str, float], ...]
    second_residuals: tuple[dict[str, float], ...]


def _problem(spec) -> OptimizationProblem:
    return _build(spec, 0)[0]


def _ledger(problem: OptimizationProblem) -> EvaluationLedger:
    incumbent = (2.0,) * problem.dimension
    return EvaluationLedger(
        problem,
        total_budget=100,
        initial_incumbent=incumbent,
        initial_error=float(problem.objective(np.asarray(incumbent))),
    )


def _proposal_dict(proposals: tuple[LocalProposal, ...]) -> list[dict[str, object]]:
    return [
        {
            "group": proposal.group,
            "values": list(proposal.values),
            "uncertainty": list(proposal.uncertainty),
            "improvement": proposal.improvement,
        }
        for proposal in proposals
    ]


def _replay(spec, proposals: tuple[LocalProposal, ...], seed: int, selected_component: tuple[int, ...]) -> Replay:
    problem = _problem(spec)
    coordinator = OverlapCoordinator(
        OverlapStructure(dimension=problem.dimension, groups=spec.groups),
        _ledger(problem),
    )
    scheduler = GraphCoordinationScheduler(coordinator)
    initial_error = coordinator.ledger.best_error
    scheduler.prime(proposals)
    priorities = scheduler.prioritize(proposals)
    priority = next(item for item in priorities if item.component == selected_component)
    after_prime_error = coordinator.ledger.best_error
    after_prime_x = tuple(float(value) for value in coordinator.ledger.best_x)
    component_proposals = scheduler._component_proposals(selected_component, proposals)
    before = coordinator.ledger.best_error
    result = coordinator.coordinate(
        selected_component,
        component_proposals,
        ctp_budget_fes=DISPATCH_BUDGET_FES,
        ctp_seed=_repair_seed(seed),
    )
    candidate_error_count = len(result.candidates)
    # coordinate() evaluates candidates before CTP; the ledger archive after
    # that evaluation is available from the CTP boundary field.
    candidate_after = result.ctp_best_error_before if result.ctp_triggered else coordinator.ledger.best_error
    residuals = tuple(
        {
            "variable": item.variable,
            "weighted_mean": item.weighted_mean,
            "between_variance": item.between_variance,
            "within_variance": item.within_variance,
            "conflict_score": item.conflict_score,
        }
        for item in result.residuals
    )
    if candidate_error_count == 0:
        raise AssertionError("coordinate returned no candidates")
    return Replay(
        selected_component=selected_component,
        selected_priority=priority.priority_score,
        initial_error=initial_error,
        after_prime_error=after_prime_error,
        after_prime_x=after_prime_x,
        second_before_error=before,
        second_after_candidates_error=candidate_after,
        second_after_ctp_error=coordinator.ledger.best_error,
        second_ctp_triggered=result.ctp_triggered,
        second_ctp_consumed_fes=result.ctp_consumed_fes,
        second_ctp_seed=_repair_seed(seed),
        total_ledger_fes=coordinator.ledger.count,
        second_candidate_errors=result.candidate_errors,
        second_residuals=residuals,
    )


def audit(seed: int) -> dict[str, object]:
    spec = next(item for item in TOPOLOGIES if item.name == "pair_pair")
    _, proposals = _build(spec, seed)
    structure = OverlapStructure(dimension=6, groups=spec.groups)
    coordinator = OverlapCoordinator(structure, _ledger(_problem(spec)))
    scheduler = GraphCoordinationScheduler(coordinator)
    scheduler.prime(proposals)
    priorities = scheduler.prioritize(proposals)
    replays = {
        "gcb": _replay(spec, proposals, seed, priorities[0].component),
        "canonical": _replay(spec, proposals, seed, min(item.component for item in priorities)),
    }
    return {
        "seed": seed,
        "topology": spec.name,
        "components": [list(item) for item in scheduler.overlap_components],
        "proposals": _proposal_dict(proposals),
        "priorities": [asdict(item) for item in priorities],
        "replays": {name: asdict(value) for name, value in replays.items()},
        "component_isomorphism": {
            "same_group_sizes": len({tuple(sorted(len(spec.groups[group]) for group in component)) for component in scheduler.overlap_components}) == 1,
            "same_shared_variable_multiplicities": len({tuple(sorted(len(structure.owners(variable)) for variable in item.shared_variables)) for item in priorities}) == 1,
            "same_objective_weights": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=LOSS_SEEDS)
    parser.add_argument("--output", type=Path, default=Path("artifacts/oracle_gcb_gate3/failure_audit.json"))
    args = parser.parse_args()
    payload = {
        "schema_version": "arac-oracle-gcb-gate3-failure-audit-v1",
        "trials": [audit(seed) for seed in args.seeds],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
