"""Gate 10: paired real-proposal overlap coordination effectiveness.

This gate isolates Phase-II coordination on the repaired continuous overlap
benchmark.  It deliberately uses the benchmark's oracle groups: discovery is
not part of this gate, so a failure is attributable to proposal/residual/CTP
coordination rather than Phase-I structure recovery.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from arac.benchmarks.overlap_objective import build_overlap_problem
from arac.coordination import (
    ConflictLevel,
    GraphCoordinationScheduler,
    LocalProposal,
    OverlapCoordinator,
    OverlapStructure as CoordinationStructure,
    produce_local_proposal,
)
from arac.runtime.ledger import EvaluationLedger


DIMENSION = 24
NUM_GROUPS = 6
MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 5
BASE_FUNCTION = "sphere"
BOUNDS = 10.0
PROPOSAL_BUDGET_FES = 48
PROPOSAL_ALGORITHM = "sepcmaes"
PROPOSAL_POPULATION_SIZE = 8
CTP_BUDGET_FES = 32
ARM_TOTAL_BUDGET_FES = 512
FRESH_SEEDS = (31001, 31002, 31003, 31004, 31005)
TOPOLOGIES = ("random", "chain", "star")
OVERLAP_BUDGETS = (6, 12)
MODES = ("conforming", "conflicting")


def _problem_and_structure(mode: str, topology: str, overlap_budget: int, seed: int):
    problem, objective = build_overlap_problem(
        DIMENSION,
        overlap_budget=overlap_budget,
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        num_groups=NUM_GROUPS,
        base_function=BASE_FUNCTION,
        conflict_mode=mode,
        bounds=BOUNDS,
        contiguous=True,
        rotation=False,
        transforms=False,
        seed=seed,
        topology=topology,
    )
    structure = CoordinationStructure(
        dimension=DIMENSION,
        groups=tuple(tuple(group) for group in objective.structure.groups),
    )
    return problem, objective, structure


def _seed_for_group(seed: int, group: int) -> int:
    return int(seed ^ (0x9E37 * (group + 1)))


def _new_ledger(problem, total_budget: int = ARM_TOTAL_BUDGET_FES) -> tuple[EvaluationLedger, np.ndarray, float]:
    zero = np.zeros(problem.dimension, dtype=float)
    ledger = EvaluationLedger(problem, total_budget=total_budget)
    zero_error = float(ledger.evaluate(zero))
    if ledger.count != 1:
        raise RuntimeError("zero anchor did not consume exactly one FE")
    return ledger, zero, zero_error


def _proposals(
    structure: CoordinationStructure,
    problem,
    ledger: EvaluationLedger,
    anchor: np.ndarray,
    anchor_error: float,
    seed: int,
) -> tuple[object, ...]:
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
                seed=_seed_for_group(seed, group),
                algorithm=PROPOSAL_ALGORITHM,
                population_size=PROPOSAL_POPULATION_SIZE,
                sigma=0.5,
            )
        )
    return tuple(runs)


def _proposal_payload(runs: Iterable[object]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "group": run.proposal.group,
            "values": tuple((int(variable), float(value)) for variable, value in run.proposal.values),
            "improvement": float(run.proposal.improvement),
            "uncertainty": tuple(
                (int(variable), float(value)) for variable, value in run.proposal.uncertainty
            ),
            "best_error": float(run.best_error),
            "consumed_fes": int(run.consumed_fes),
        }
        for run in runs
    )


def _strict_best(results: Iterable[object], initial_error: float, final_error: float) -> bool:
    result_list = tuple(results)
    return (
        final_error <= initial_error
        and all(result.best_error_after <= result.best_error_before for result in result_list)
    )


def _owner_continuation(
    ledger: EvaluationLedger,
    proposals: tuple[LocalProposal, ...],
    component: tuple[int, ...],
    *,
    base: np.ndarray,
    seed: int,
    budget_fes: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, object]:
    """Spend equal FE using one owner proposal per candidate, never consensus."""

    if budget_fes <= 0:
        return {"consumed_fes": 0, "policy": "owner_round_robin_uncertainty_noise"}
    by_group = {proposal.group: proposal for proposal in proposals}
    rng = np.random.default_rng(seed)
    candidates = np.repeat(np.asarray(base, dtype=float)[np.newaxis, :], budget_fes, axis=0)
    for index in range(budget_fes):
        group = component[index % len(component)]
        proposal = by_group[group]
        for variable, value in proposal.values:
            sigma = max(np.finfo(float).eps, proposal.sigma(variable))
            candidates[index, variable] = value + float(rng.normal(0.0, sigma))
    np.clip(candidates, lower, upper, out=candidates)
    before = ledger.count
    ledger.evaluate(candidates)
    consumed = ledger.count - before
    if consumed != budget_fes:
        raise RuntimeError("owner control drifted from its exact FE budget")
    return {
        "consumed_fes": consumed,
        "policy": "owner_round_robin_uncertainty_noise",
        "groups": tuple(component[index % len(component)] for index in range(budget_fes)),
    }


def _arm(
    mode: str,
    topology: str,
    overlap_budget: int,
    seed: int,
    *,
    coordination: bool,
) -> dict[str, object]:
    problem, objective, structure = _problem_and_structure(mode, topology, overlap_budget, seed)
    ledger, anchor, anchor_error = _new_ledger(problem)
    proposal_runs = _proposals(structure, problem, ledger, anchor, anchor_error, seed)
    proposals = tuple(run.proposal for run in proposal_runs)
    coordinator = OverlapCoordinator(structure, ledger)
    scheduler = GraphCoordinationScheduler(coordinator)
    prime_results = scheduler.prime(proposals)
    priorities = scheduler.prioritize(proposals)
    eligible = tuple(
        item
        for item in priorities
        if item.conflict_level is ConflictLevel.HIGH and item.conflict_streak >= 1
    )
    selected = eligible[0].component if eligible else None
    second_result = None
    dispatch = None
    control = None
    if selected is not None:
        if coordination:
            dispatch = scheduler.dispatch(
                proposals,
                total_ctp_budget_fes=CTP_BUDGET_FES,
                max_components=1,
                seed=seed ^ 0xC7A5,
            )
            if not dispatch.events or dispatch.events[0].component != selected:
                raise RuntimeError("GCB dispatch did not select the pre-registered component")
            ctp_triggered = dispatch.events[0].consumed_ctp_fes > 0
        else:
            # The control arm repeats exactly the selected candidate arbitration,
            # then spends the same FE through independent owner continuations.
            second_result = coordinator.coordinate(
                selected,
                tuple(proposal for proposal in proposals if proposal.group in selected),
                ctp_budget_fes=0,
                ctp_seed=seed ^ 0xC7A5,
            )
            control = _owner_continuation(
                ledger,
                proposals,
                selected,
                base=ledger.best_x,
                seed=seed ^ 0x51ED,
                budget_fes=CTP_BUDGET_FES,
                lower=problem.lower_array,
                upper=problem.upper_array,
            )
            ctp_triggered = False
    else:
        ctp_triggered = False

    final_error = float(ledger.best_error)
    prime_fes = sum(len(result.candidates) for result in prime_results)
    second_fes = 0 if second_result is None else len(second_result.candidates)
    ctp_fes = 0 if dispatch is None else dispatch.consumed_ctp_fes
    continuation_fes = 0 if control is None else int(control["consumed_fes"])
    return {
        "mode": mode,
        "topology": topology,
        "overlap_budget": overlap_budget,
        "seed": seed,
        "groups": structure.groups,
        "shared_variables": structure.shared_variables,
        "proposal_runs": _proposal_payload(proposal_runs),
        "proposal_fes": sum(run.consumed_fes for run in proposal_runs),
        "anchor_error": anchor_error,
        "prime_fes": prime_fes,
        "prime_results": tuple(
            {
                "component": result.component,
                "conflict_level": result.conflict_level.value,
                "max_residual_score": max(
                    (residual.conflict_score for residual in result.residuals), default=0.0
                ),
                "residuals": tuple(
                    {
                        "variable": residual.variable,
                        "conflict_score": residual.conflict_score,
                        "between_variance": residual.between_variance,
                        "within_variance": residual.within_variance,
                    }
                    for residual in result.residuals
                ),
                "conflict_streak": result.conflict_streak,
                "best_error_before": result.best_error_before,
                "best_error_after": result.best_error_after,
            }
            for result in prime_results
        ),
        "eligible_components": tuple(item.component for item in eligible),
        "selected_component": selected,
        "ctp_triggered": ctp_triggered,
        "ctp_fes": ctp_fes,
        "continuation_fes": continuation_fes,
        "second_arbitration_fes": second_fes,
        "control_policy": None if control is None else control["policy"],
        "final_error": final_error,
        "consumed_fes": ledger.count,
        "strict_best": _strict_best(prime_results, anchor_error, final_error)
        and (second_result is None or second_result.best_error_after <= second_result.best_error_before),
        "objective_optimum_is_attainable": objective.optimum_is_attainable,
    }


def _context(mode: str, topology: str, overlap_budget: int, seed: int) -> dict[str, object]:
    coordination = _arm(mode, topology, overlap_budget, seed, coordination=True)
    control = _arm(mode, topology, overlap_budget, seed, coordination=False)
    if coordination["groups"] != control["groups"]:
        raise RuntimeError("paired arms did not share the same overlap structure")
    if coordination["proposal_runs"] != control["proposal_runs"]:
        raise RuntimeError("paired real local proposals are not identical")
    if coordination["consumed_fes"] != control["consumed_fes"]:
        raise RuntimeError(
            f"paired arms FE mismatch: {coordination['consumed_fes']} != {control['consumed_fes']}"
        )
    gain = float(control["final_error"]) - float(coordination["final_error"])
    return {
        "mode": mode,
        "topology": topology,
        "overlap_budget": overlap_budget,
        "seed": seed,
        "groups": coordination["groups"],
        "shared_variables": coordination["shared_variables"],
        "coordination": coordination,
        "control": control,
        "proposal_parity": True,
        "fe_parity": coordination["consumed_fes"] == control["consumed_fes"],
        "strict_best": bool(coordination["strict_best"] and control["strict_best"]),
        "selected_component_parity": coordination["selected_component"]
        == control["selected_component"],
        "max_residual_score": float(
            max(
                (
                    item["max_residual_score"]
                    for item in coordination["prime_results"]
                ),
                default=0.0,
            )
        ),
        "ctp_triggered": bool(coordination["ctp_triggered"]),
        "coordination_final_error": coordination["final_error"],
        "control_final_error": control["final_error"],
        "gain": gain,
        "win_or_tie": float(coordination["final_error"]) <= float(control["final_error"]),
    }


def _auc(conflicting: Iterable[float], conforming: Iterable[float]) -> float:
    left = tuple(float(value) for value in conflicting)
    right = tuple(float(value) for value in conforming)
    if not left or not right:
        return float("nan")
    wins = sum(
        1.0 if conflict > conform else 0.5 if conflict == conform else 0.0
        for conflict in left
        for conform in right
    )
    return wins / (len(left) * len(right))


def _cell_summary(contexts: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    result = []
    for topology in TOPOLOGIES:
        for overlap_budget in OVERLAP_BUDGETS:
            rows = tuple(
                item
                for item in contexts
                if item["topology"] == topology and item["overlap_budget"] == overlap_budget
            )
            result.append(
                {
                    "topology": topology,
                    "overlap_budget": overlap_budget,
                    "context_count": len(rows),
                    "conforming_count": sum(item["mode"] == "conforming" for item in rows),
                    "conflicting_count": sum(item["mode"] == "conflicting" for item in rows),
                    "complete": len(rows) == 2 * len(FRESH_SEEDS),
                }
            )
    return tuple(result)


def run_gate(*, workers: int = 1) -> dict[str, object]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
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
    contexts = tuple(sorted(contexts, key=lambda item: (item["topology"], item["overlap_budget"], item["seed"], item["mode"])))
    conforming_scores = tuple(item["max_residual_score"] for item in contexts if item["mode"] == "conforming")
    conflicting_scores = tuple(item["max_residual_score"] for item in contexts if item["mode"] == "conflicting")
    conflicting = tuple(item for item in contexts if item["mode"] == "conflicting")
    conforming = tuple(item for item in contexts if item["mode"] == "conforming")
    trigger_rate_conforming = sum(item["ctp_triggered"] for item in conforming) / len(conforming)
    trigger_rate_conflicting = sum(item["ctp_triggered"] for item in conflicting) / len(conflicting)
    gains = tuple(float(item["gain"]) for item in conflicting)
    win_rate = sum(item["win_or_tie"] for item in conflicting) / len(conflicting)
    cells = _cell_summary(contexts)
    gate_checks = {
        "context_count": len(contexts) == 60,
        "paired_proposals_identical": all(item["proposal_parity"] for item in contexts),
        "paired_fe_exact": all(item["fe_parity"] for item in contexts),
        "strict_best_all_arms": all(item["strict_best"] for item in contexts),
        "residual_auc_ge_0_65": _auc(conflicting_scores, conforming_scores) >= 0.65,
        "conforming_trigger_rate_le_0_20": trigger_rate_conforming <= 0.20,
        "conflicting_trigger_rate_ge_0_80": trigger_rate_conflicting >= 0.80,
        "conflicting_win_or_tie_rate_ge_0_60": win_rate >= 0.60,
        "conflicting_median_gain_ge_0": float(np.median(gains)) >= 0.0,
        "all_topology_overlap_cells_complete": all(item["complete"] for item in cells),
    }
    return {
        "schema_version": "arac-overlap-coordination-effectiveness-gate10-v1",
        "protocol": {
            "dimension": DIMENSION,
            "num_groups": NUM_GROUPS,
            "base_group_size": (MIN_GROUP_SIZE, MAX_GROUP_SIZE),
            "base_function": BASE_FUNCTION,
            "rotation": False,
            "transforms": False,
            "proposal_algorithm": PROPOSAL_ALGORITHM,
            "proposal_budget_fes_each": PROPOSAL_BUDGET_FES,
            "proposal_population_size": PROPOSAL_POPULATION_SIZE,
            "ctp_budget_fes": CTP_BUDGET_FES,
            "seeds": FRESH_SEEDS,
            "topologies": TOPOLOGIES,
            "overlap_budgets": OVERLAP_BUDGETS,
            "control_policy": "owner_round_robin_uncertainty_noise",
        },
        "context_count": len(contexts),
        "contexts": contexts,
        "cell_summary": cells,
        "summary": {
            "residual_auc_conflicting_over_conforming": _auc(conflicting_scores, conforming_scores),
            "conforming_ctp_trigger_rate": trigger_rate_conforming,
            "conflicting_ctp_trigger_rate": trigger_rate_conflicting,
            "conflicting_win_or_tie_rate": win_rate,
            "conflicting_median_gain": float(np.median(gains)),
            "conflicting_mean_gain": float(np.mean(gains)),
        },
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_coordination_effectiveness_gate10/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
