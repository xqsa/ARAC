"""Gate 13: calibrated proposals paired with existing overlap coordination."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import median

import numpy as np

from arac.benchmarks.overlap_objective import build_overlap_problem
from arac.coordination import (
    ConflictLevel,
    GraphCoordinationScheduler,
    LocalProposal,
    OverlapCoordinator,
    OverlapStructure,
    produce_local_proposal,
)
from arac.runtime.ledger import EvaluationLedger


DIMENSION = 24
NUM_GROUPS = 6
MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 5
BASE_FUNCTION = "sphere"
BOUNDS = 10.0
INTERACTION_STRENGTH = 0.25
PROPOSAL_BUDGET_FES = 48
PROPOSAL_REPLICATES = 4
PROPOSAL_ALGORITHM = "sepcmaes"
PROPOSAL_POPULATION_SIZE = 8
CTP_BUDGET_FES = 32
ARM_TOTAL_BUDGET_FES = 2_000
FRESH_SEEDS = (31001, 31002, 31003, 31004, 31005)
TOPOLOGIES = ("random", "chain", "star")
OVERLAP_BUDGETS = (6, 12)
MODES = ("conforming", "conflicting")


def _build(mode: str, topology: str, overlap_budget: int, seed: int):
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
        interaction_strength=INTERACTION_STRENGTH,
        seed=seed,
        topology=topology,
    )
    structure = OverlapStructure(
        DIMENSION,
        tuple(tuple(group) for group in objective.structure.groups),
    )
    return problem, objective, structure


def _replicate_seed(seed: int, group: int, replicate: int) -> int:
    return int(seed ^ (0x9E37 * (group + 1)) ^ (0x51ED * (replicate + 1)))


def _proposal_value(run: object, variable: int) -> float:
    values = run.proposal.values if hasattr(run, "proposal") else run.values
    return float(dict(values)[variable])


def _proposal_sigma(run: object, variable: int) -> float:
    uncertainty = run.proposal.uncertainty if hasattr(run, "proposal") else run.uncertainty
    return float(dict(uncertainty)[variable])


def _calibrated_proposals(
    structure: OverlapStructure,
    runs: list[list[object]],
) -> tuple[tuple[LocalProposal, ...], tuple[dict[str, object], ...]]:
    proposals: list[LocalProposal] = []
    heldout_records: list[dict[str, object]] = []
    for group, group_runs in enumerate(runs):
        variables = tuple(structure.groups[group])
        values = tuple(
            (
                variable,
                float(np.mean([_proposal_value(group_runs[replicate], variable) for replicate in range(3)])),
            )
            for variable in variables
        )
        uncertainty = tuple(
            (
                variable,
                max(
                    np.finfo(float).eps,
                    float(
                        np.std(
                            [_proposal_value(group_runs[replicate], variable) for replicate in range(3)],
                            ddof=1,
                        )
                    ),
                ),
            )
            for variable in variables
        )
        improvement = float(np.mean([group_runs[replicate].proposal.improvement for replicate in range(3)]))
        proposals.append(
            LocalProposal(
                group=group,
                values=values,
                improvement=improvement,
                uncertainty=uncertainty,
            )
        )
    for variable in structure.shared_variables:
        owners = structure.owners(variable)
        heldout_values = {
            group: _proposal_value(runs[group][3], variable)
            for group in owners
        }
        calibration_values = {
            group: _proposal_value(proposals[group], variable)
            for group in owners
        }
        pooled_variance = float(
            np.mean(
                [proposals[group].sigma(variable) ** 2 for group in owners]
            )
        )
        heldout_records.append(
            {
                "variable": variable,
                "owners": owners,
                "heldout_disagreement": float(max(heldout_values.values()) - min(heldout_values.values())),
                "calibration_disagreement": float(max(calibration_values.values()) - min(calibration_values.values())),
                "heldout_standardized_score": float(
                    (max(heldout_values.values()) - min(heldout_values.values()))
                    / (np.sqrt(pooled_variance) + 1.0e-12)
                ),
            }
        )
    return tuple(proposals), tuple(heldout_records)


def _proposal_payload(proposals: tuple[LocalProposal, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "group": proposal.group,
            "values": proposal.values,
            "uncertainty": proposal.uncertainty,
            "improvement": proposal.improvement,
        }
        for proposal in proposals
    )


def _run_proposals(problem, structure: OverlapStructure, seed: int):
    zero = np.zeros(DIMENSION, dtype=float)
    ledger = EvaluationLedger(problem, total_budget=ARM_TOTAL_BUDGET_FES)
    anchor_error = float(ledger.evaluate(zero))
    runs: list[list[object]] = [[] for _ in structure.groups]
    for group in range(len(structure.groups)):
        for replicate in range(PROPOSAL_REPLICATES):
            runs[group].append(
                produce_local_proposal(
                    structure,
                    group,
                    problem=problem,
                    global_ledger=ledger,
                    anchor=zero,
                    anchor_error=anchor_error,
                    budget_fes=PROPOSAL_BUDGET_FES,
                    seed=_replicate_seed(seed, group, replicate),
                    algorithm=PROPOSAL_ALGORITHM,
                    population_size=PROPOSAL_POPULATION_SIZE,
                    sigma=0.5,
                )
            )
    expected = 1 + NUM_GROUPS * PROPOSAL_REPLICATES * PROPOSAL_BUDGET_FES
    if ledger.count != expected:
        raise RuntimeError(f"proposal replicate FE drifted: {ledger.count} != {expected}")
    proposals, heldout = _calibrated_proposals(structure, runs)
    return ledger, zero, anchor_error, runs, proposals, heldout


def _owner_continuation(
    ledger: EvaluationLedger,
    proposals: tuple[LocalProposal, ...],
    component: tuple[int, ...],
    *,
    base: np.ndarray,
    seed: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> int:
    rng = np.random.default_rng(seed)
    by_group = {proposal.group: proposal for proposal in proposals}
    candidates = np.repeat(base[np.newaxis, :], CTP_BUDGET_FES, axis=0)
    for index in range(CTP_BUDGET_FES):
        group = component[index % len(component)]
        proposal = by_group[group]
        for variable, value in proposal.values:
            candidates[index, variable] = value + float(
                rng.normal(0.0, max(np.finfo(float).eps, proposal.sigma(variable)))
            )
    np.clip(candidates, lower, upper, out=candidates)
    before = ledger.count
    ledger.evaluate(candidates)
    return ledger.count - before


def _arm(
    mode: str,
    topology: str,
    overlap_budget: int,
    seed: int,
    *,
    coordination: bool,
) -> dict[str, object]:
    problem, objective, structure = _build(mode, topology, overlap_budget, seed)
    ledger, anchor, anchor_error, runs, proposals, heldout = _run_proposals(problem, structure, seed)
    coordinator = OverlapCoordinator(structure, ledger)
    scheduler = GraphCoordinationScheduler(coordinator)
    prime = scheduler.prime(proposals)
    priorities = scheduler.prioritize(proposals)
    eligible = tuple(
        item
        for item in priorities
        if item.conflict_level is ConflictLevel.HIGH and item.conflict_streak >= 1
    )
    selected = eligible[0].component if eligible else None
    second = None
    dispatch = None
    control_fes = 0
    if selected is not None:
        if coordination:
            dispatch = scheduler.dispatch(
                proposals,
                total_ctp_budget_fes=CTP_BUDGET_FES,
                max_components=1,
                seed=seed ^ 0xC7A5,
            )
            if not dispatch.events or dispatch.events[0].component != selected:
                raise RuntimeError("GCB selected a component different from frozen priority")
        else:
            second = coordinator.coordinate(
                selected,
                tuple(proposal for proposal in proposals if proposal.group in selected),
                ctp_budget_fes=0,
                ctp_seed=seed ^ 0xC7A5,
            )
            control_fes = _owner_continuation(
                ledger,
                proposals,
                selected,
                base=ledger.best_x,
                seed=seed ^ 0x51ED,
                lower=problem.lower_array,
                upper=problem.upper_array,
            )
    final_error = float(ledger.best_error)
    ctp_fes = 0 if dispatch is None else int(dispatch.consumed_ctp_fes)
    consumed = ledger.count
    return {
        "mode": mode,
        "topology": topology,
        "overlap_budget": overlap_budget,
        "seed": seed,
        "groups": structure.groups,
        "shared_variables": structure.shared_variables,
        "proposal_payload": _proposal_payload(proposals),
        "raw_replicate_count_per_group": tuple(len(group_runs) for group_runs in runs),
        "proposal_fes": 1 + NUM_GROUPS * PROPOSAL_REPLICATES * PROPOSAL_BUDGET_FES,
        "heldout_residuals": heldout,
        "prime_results": tuple(
            {
                "component": result.component,
                "conflict_level": result.conflict_level.value,
                "conflict_streak": result.conflict_streak,
                "max_residual_score": max(
                    (residual.conflict_score for residual in result.residuals), default=0.0
                ),
                "best_error_before": result.best_error_before,
                "best_error_after": result.best_error_after,
            }
            for result in prime
        ),
        "eligible_components": tuple(item.component for item in eligible),
        "selected_component": selected,
        "ctp_triggered": dispatch is not None and ctp_fes > 0,
        "ctp_fes": ctp_fes,
        "control_fes": control_fes,
        "second_arbitration_fes": 0 if second is None else len(second.candidates),
        "final_error": final_error,
        "consumed_fes": consumed,
        "strict_best": final_error <= anchor_error
        and all(result.best_error_after <= result.best_error_before for result in prime)
        and (second is None or second.best_error_after <= second.best_error_before),
        "objective_optimum_is_attainable": objective.optimum_is_attainable,
    }


def _context(mode: str, topology: str, overlap_budget: int, seed: int) -> dict[str, object]:
    coordination = _arm(mode, topology, overlap_budget, seed, coordination=True)
    control = _arm(mode, topology, overlap_budget, seed, coordination=False)
    for key in ("groups", "shared_variables", "proposal_payload", "proposal_fes", "consumed_fes"):
        if coordination[key] != control[key]:
            raise RuntimeError(f"paired {key} mismatch")
    gain = float(control["final_error"]) - float(coordination["final_error"])
    heldout_scores = [item["heldout_standardized_score"] for item in coordination["heldout_residuals"]]
    return {
        "mode": mode,
        "topology": topology,
        "overlap_budget": overlap_budget,
        "seed": seed,
        "groups": coordination["groups"],
        "shared_variables": coordination["shared_variables"],
        "coordination": coordination,
        "control": control,
        "proposal_parity": coordination["proposal_payload"] == control["proposal_payload"],
        "fe_parity": coordination["consumed_fes"] == control["consumed_fes"],
        "strict_best": bool(coordination["strict_best"] and control["strict_best"]),
        "max_heldout_standardized_score": float(max(heldout_scores, default=0.0)),
        "max_calibrated_residual_score": float(
            max(
                (
                    result["max_residual_score"]
                    for result in coordination["prime_results"]
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


def _auc(conflicting: list[float], conforming: list[float]) -> float:
    return float(
        sum(
            1.0 if left > right else 0.5 if left == right else 0.0
            for left in conflicting
            for right in conforming
        )
        / (len(conflicting) * len(conforming))
    )


def _cells(contexts: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    cells = []
    for topology in TOPOLOGIES:
        for overlap_budget in OVERLAP_BUDGETS:
            rows = tuple(
                item
                for item in contexts
                if item["topology"] == topology and item["overlap_budget"] == overlap_budget
            )
            cells.append(
                {
                    "topology": topology,
                    "overlap_budget": overlap_budget,
                    "context_count": len(rows),
                    "complete": len(rows) == 2 * len(FRESH_SEEDS),
                }
            )
    return tuple(cells)


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
    contexts = tuple(sorted(contexts, key=lambda row: (row["topology"], row["overlap_budget"], row["seed"], row["mode"])))
    conforming = [row for row in contexts if row["mode"] == "conforming"]
    conflicting = [row for row in contexts if row["mode"] == "conflicting"]
    calibrated_conforming = [row["max_calibrated_residual_score"] for row in conforming]
    calibrated_conflicting = [row["max_calibrated_residual_score"] for row in conflicting]
    heldout_conforming = [row["max_heldout_standardized_score"] for row in conforming]
    heldout_conflicting = [row["max_heldout_standardized_score"] for row in conflicting]
    gains = [float(row["gain"]) for row in conflicting]
    trigger_conforming = sum(bool(row["ctp_triggered"]) for row in conforming) / len(conforming)
    trigger_conflicting = sum(bool(row["ctp_triggered"]) for row in conflicting) / len(conflicting)
    win_rate = sum(bool(row["win_or_tie"]) for row in conflicting) / len(conflicting)
    gate_checks = {
        "context_count_60": len(contexts) == 60,
        "paired_proposals_identical": all(row["proposal_parity"] for row in contexts),
        "paired_fe_exact": all(row["fe_parity"] for row in contexts),
        "strict_best_all_arms": all(row["strict_best"] for row in contexts),
        "heldout_metrics_finite_both_modes": all(
            np.isfinite(row["max_heldout_standardized_score"])
            for row in contexts
        ),
        "calibrated_residual_auc_ge_0_65": _auc(calibrated_conflicting, calibrated_conforming) >= 0.65,
        "conforming_trigger_rate_le_0_20": trigger_conforming <= 0.20,
        "conflicting_trigger_rate_ge_0_80": trigger_conflicting >= 0.80,
        "conflicting_win_or_tie_rate_ge_0_60": win_rate >= 0.60,
        "conflicting_median_gain_ge_0": float(median(gains)) >= 0.0,
        "all_cells_complete": all(item["complete"] for item in _cells(contexts)),
    }
    return {
        "schema_version": "arac-overlap-calibrated-coordination-gate13-v1",
        "protocol": {
            "dimension": DIMENSION,
            "num_groups": NUM_GROUPS,
            "interaction_strength": INTERACTION_STRENGTH,
            "proposal_algorithm": PROPOSAL_ALGORITHM,
            "proposal_budget_fes_each": PROPOSAL_BUDGET_FES,
            "proposal_replicates": PROPOSAL_REPLICATES,
            "calibration_replicates": (0, 1, 2),
            "heldout_replicate": 3,
            "ctp_budget_fes": CTP_BUDGET_FES,
            "equal_fe_control": "owner_round_robin_uncertainty_noise",
            "seeds": FRESH_SEEDS,
            "topologies": TOPOLOGIES,
            "overlap_budgets": OVERLAP_BUDGETS,
        },
        "context_count": len(contexts),
        "contexts": contexts,
        "cell_summary": _cells(contexts),
        "summary": {
            "calibrated_residual_auc_conflicting_over_conforming": _auc(
                calibrated_conflicting, calibrated_conforming
            ),
            "heldout_residual_auc_conflicting_over_conforming": _auc(
                heldout_conflicting, heldout_conforming
            ),
            "conforming_ctp_trigger_rate": trigger_conforming,
            "conflicting_ctp_trigger_rate": trigger_conflicting,
            "conflicting_win_or_tie_rate": win_rate,
            "conflicting_median_gain": float(median(gains)),
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
        default=Path("artifacts/overlap_calibrated_coordination_gate13/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
