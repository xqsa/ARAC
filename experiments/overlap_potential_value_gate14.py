"""Gate 14: forced CTP versus equal-FE owner control potential value."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import median

import numpy as np

from arac.benchmarks.overlap_objective import build_overlap_problem
from arac.coordination import (
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
        base_function="sphere",
        conflict_mode=mode,
        bounds=BOUNDS,
        contiguous=True,
        rotation=False,
        transforms=False,
        interaction_strength=INTERACTION_STRENGTH,
        seed=seed,
        topology=topology,
    )
    structure = OverlapStructure(DIMENSION, tuple(tuple(group) for group in objective.structure.groups))
    return problem, objective, structure


def _seed(seed: int, group: int, replicate: int) -> int:
    return int(seed ^ (0x9E37 * (group + 1)) ^ (0x51ED * (replicate + 1)))


def _value(run: object, variable: int) -> float:
    proposal = run.proposal if hasattr(run, "proposal") else run
    return float(dict(proposal.values)[variable])


def _sigma(run: object, variable: int) -> float:
    proposal = run.proposal if hasattr(run, "proposal") else run
    return float(dict(proposal.uncertainty)[variable])


def _proposals(problem, structure: OverlapStructure, seed: int):
    zero = np.zeros(DIMENSION, dtype=float)
    ledger = EvaluationLedger(problem, total_budget=ARM_TOTAL_BUDGET_FES)
    anchor_error = float(ledger.evaluate(zero))
    runs: list[list[object]] = [[] for _ in structure.groups]
    for group in range(NUM_GROUPS):
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
                values=tuple((variable, float(np.mean([_value(group_runs[r], variable) for r in range(3)]))) for variable in variables),
                improvement=float(np.mean([group_runs[r].proposal.improvement for r in range(3)])),
                uncertainty=tuple((variable, max(np.finfo(float).eps, float(np.std([_value(group_runs[r], variable) for r in range(3)], ddof=1)))) for variable in variables),
            )
        )
    if ledger.count != 1 + NUM_GROUPS * PROPOSAL_REPLICATES * PROPOSAL_BUDGET_FES:
        raise RuntimeError("proposal FE drifted")
    return ledger, zero, anchor_error, tuple(proposals), runs


def _owner_control(ledger: EvaluationLedger, proposals: tuple[LocalProposal, ...], component: tuple[int, ...], base: np.ndarray, seed: int, lower: np.ndarray, upper: np.ndarray) -> int:
    rng = np.random.default_rng(seed)
    by_group = {proposal.group: proposal for proposal in proposals}
    candidates = np.repeat(base[np.newaxis, :], CTP_BUDGET_FES, axis=0)
    for index in range(CTP_BUDGET_FES):
        proposal = by_group[component[index % len(component)]]
        for variable, value in proposal.values:
            candidates[index, variable] = value + float(rng.normal(0.0, max(np.finfo(float).eps, proposal.sigma(variable))))
    np.clip(candidates, lower, upper, out=candidates)
    before = ledger.count
    ledger.evaluate(candidates)
    return ledger.count - before


def _arm(mode: str, topology: str, overlap_budget: int, seed: int, *, ctp: bool) -> dict[str, object]:
    problem, objective, structure = _build(mode, topology, overlap_budget, seed)
    ledger, anchor, anchor_error, proposals, runs = _proposals(problem, structure, seed)
    coordinator = OverlapCoordinator(structure, ledger)
    scheduler = GraphCoordinationScheduler(coordinator)
    prime = scheduler.prime(proposals)
    priorities = scheduler.prioritize(proposals)
    if not priorities:
        raise RuntimeError("no overlap component available for forced potential-value trial")
    selected = priorities[0].component
    if ctp:
        # Force the shared-core trial even when residual level is not high.
        before = ledger.count
        result = coordinator.coordinate(
            selected,
            tuple(proposal for proposal in proposals if proposal.group in selected),
            ctp_budget_fes=CTP_BUDGET_FES,
            ctp_seed=seed ^ 0xC7A5,
        )
        # If the frozen coordinator refuses CTP for non-persistent residuals,
        # spend the same shared-core budget through its explicit repair kernel.
        arbitration_fes = len(result.candidates)
        if ledger.count - before < arbitration_fes + CTP_BUDGET_FES:
            coordinator._repair_shared_core(
                selected,
                tuple(proposal for proposal in proposals if proposal.group in selected),
                budget_fes=CTP_BUDGET_FES - max(0, ledger.count - before - arbitration_fes),
                seed=seed ^ 0xC7A5,
                base=ledger.best_x,
            )
        else:
            pass
        total_extra_fes = ledger.count - before
        event = {
            "conflict_level": result.conflict_level.value,
            "ctp_triggered_by_policy": result.ctp_triggered,
            "consumed_ctp_fes": int(total_extra_fes - arbitration_fes),
        }
        extra_fes = total_extra_fes
    else:
        second = coordinator.coordinate(
            selected,
            tuple(proposal for proposal in proposals if proposal.group in selected),
            ctp_budget_fes=0,
            ctp_seed=seed ^ 0xC7A5,
        )
        extra_fes = _owner_control(
            ledger, proposals, selected, ledger.best_x, seed ^ 0x51ED, problem.lower_array, problem.upper_array
        )
        event = {
            "conflict_level": second.conflict_level.value,
            "ctp_triggered_by_policy": False,
            "consumed_ctp_fes": extra_fes,
        }
    return {
        "mode": mode,
        "topology": topology,
        "overlap_budget": overlap_budget,
        "seed": seed,
        "groups": structure.groups,
        "shared_variables": structure.shared_variables,
        "proposal_payload": tuple((proposal.group, proposal.values, proposal.uncertainty, proposal.improvement) for proposal in proposals),
        "proposal_fes": 1 + NUM_GROUPS * PROPOSAL_REPLICATES * PROPOSAL_BUDGET_FES,
        "selected_component": selected,
        "prime_results": tuple({"component": result.component, "conflict_level": result.conflict_level.value, "max_residual_score": max((residual.conflict_score for residual in result.residuals), default=0.0), "best_error_before": result.best_error_before, "best_error_after": result.best_error_after} for result in prime),
        "event": event,
        "extra_fes": extra_fes,
        "consumed_fes": ledger.count,
        "final_error": float(ledger.best_error),
        "strict_best": float(ledger.best_error) <= anchor_error and all(result.best_error_after <= result.best_error_before for result in prime),
        "objective_optimum_is_attainable": objective.optimum_is_attainable,
    }


def _context(mode: str, topology: str, overlap_budget: int, seed: int) -> dict[str, object]:
    ctp = _arm(mode, topology, overlap_budget, seed, ctp=True)
    control = _arm(mode, topology, overlap_budget, seed, ctp=False)
    for key in ("groups", "shared_variables", "proposal_payload", "proposal_fes", "selected_component", "consumed_fes"):
        if ctp[key] != control[key]:
            raise RuntimeError(f"paired {key} mismatch")
    gain = float(control["final_error"]) - float(ctp["final_error"])
    return {
        "mode": mode,
        "topology": topology,
        "overlap_budget": overlap_budget,
        "seed": seed,
        "ctp": ctp,
        "control": control,
        "proposal_parity": ctp["proposal_payload"] == control["proposal_payload"],
        "fe_parity": ctp["consumed_fes"] == control["consumed_fes"],
        "strict_best": bool(ctp["strict_best"] and control["strict_best"]),
        "gain": gain,
        "win_or_tie": float(ctp["final_error"]) <= float(control["final_error"]),
        "max_residual_score": max((item["max_residual_score"] for item in ctp["prime_results"]), default=0.0),
    }


def _rank_correlation(left: list[float], right: list[float], *, spearman: bool) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if spearman:
        x = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
        y = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_gate(*, workers: int = 1) -> dict[str, object]:
    jobs = tuple((mode, topology, overlap_budget, seed) for topology in TOPOLOGIES for overlap_budget in OVERLAP_BUDGETS for seed in FRESH_SEEDS for mode in MODES)
    if workers == 1:
        contexts = tuple(_context(*job) for job in jobs)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            contexts = tuple(executor.map(lambda job: _context(*job), jobs))
    contexts = tuple(sorted(contexts, key=lambda row: (row["topology"], row["overlap_budget"], row["seed"], row["mode"])))
    gains = [float(row["gain"]) for row in contexts]
    residuals = [float(row["max_residual_score"]) for row in contexts]
    cells = tuple({"topology": topology, "overlap_budget": budget, "context_count": sum(row["topology"] == topology and row["overlap_budget"] == budget for row in contexts), "complete": sum(row["topology"] == topology and row["overlap_budget"] == budget for row in contexts) == 2 * len(FRESH_SEEDS)} for topology in TOPOLOGIES for budget in OVERLAP_BUDGETS)
    gate_checks = {
        "context_count_60": len(contexts) == 60,
        "paired_proposals_identical": all(row["proposal_parity"] for row in contexts),
        "paired_fe_exact": all(row["fe_parity"] for row in contexts),
        "forced_ctp_exact_32_fes": all(row["ctp"]["event"]["consumed_ctp_fes"] == CTP_BUDGET_FES for row in contexts),
        "strict_best_all_arms": all(row["strict_best"] for row in contexts),
        "overall_win_or_tie_ge_0_60": sum(row["win_or_tie"] for row in contexts) / len(contexts) >= 0.60,
        "median_gain_ge_0": float(median(gains)) >= 0.0,
        "correlations_finite": bool(
            np.isfinite(_rank_correlation(residuals, gains, spearman=False))
            and np.isfinite(_rank_correlation(residuals, gains, spearman=True))
        ),
        "all_cells_complete": all(item["complete"] for item in cells),
    }
    return {
        "schema_version": "arac-overlap-potential-value-gate14-v1",
        "protocol": {"interaction_strength": INTERACTION_STRENGTH, "proposal_budget_fes_each": PROPOSAL_BUDGET_FES, "proposal_replicates": PROPOSAL_REPLICATES, "forced_ctp_fes": CTP_BUDGET_FES, "control_policy": "owner_round_robin_uncertainty_noise", "seeds": FRESH_SEEDS, "topologies": TOPOLOGIES, "overlap_budgets": OVERLAP_BUDGETS},
        "context_count": len(contexts),
        "contexts": contexts,
        "cell_summary": cells,
        "summary": {"overall_win_or_tie_rate": sum(row["win_or_tie"] for row in contexts) / len(contexts), "median_gain": float(median(gains)), "mean_gain": float(np.mean(gains)), "pearson_residual_gain": _rank_correlation(residuals, gains, spearman=False), "spearman_residual_gain": _rank_correlation(residuals, gains, spearman=True)},
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/overlap_potential_value_gate14/confirmation_fresh.json"))
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
