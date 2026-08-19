"""Gate 11: replicate-calibrated residual and sphere identifiability audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import median

import numpy as np

from arac.benchmarks.overlap_objective import build_overlap_problem
from arac.coordination import OverlapStructure, produce_local_proposal
from arac.runtime.ledger import EvaluationLedger


DIMENSION = 24
NUM_GROUPS = 6
MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 5
BASE_FUNCTION = "sphere"
BOUNDS = 10.0
PROPOSAL_BUDGET_FES = 48
PROPOSAL_REPLICATES = 4
PROPOSAL_ALGORITHM = "sepcmaes"
PROPOSAL_POPULATION_SIZE = 8
TOTAL_BUDGET_FES = 2_000
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


def _run_replicates(problem, structure: OverlapStructure, seed: int):
    zero = np.zeros(problem.dimension, dtype=float)
    ledger = EvaluationLedger(problem, total_budget=TOTAL_BUDGET_FES)
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
    expected = 1 + len(structure.groups) * PROPOSAL_REPLICATES * PROPOSAL_BUDGET_FES
    if ledger.count != expected:
        raise RuntimeError(f"replicate FE drifted: {ledger.count} != {expected}")
    return runs, ledger.count, anchor_error


def _value_map(run: object) -> dict[int, float]:
    return {int(variable): float(value) for variable, value in run.proposal.values}


def _sigma_map(run: object) -> dict[int, float]:
    return {int(variable): float(value) for variable, value in run.proposal.uncertainty}


def _calibrated_residuals(structure: OverlapStructure, runs: list[list[object]]) -> tuple[dict[str, object], ...]:
    records = []
    for variable in structure.shared_variables:
        owners = structure.owners(variable)
        calibration_values = {
            group: [_value_map(runs[group][replicate])[variable] for replicate in range(3)]
            for group in owners
        }
        heldout_values = {group: _value_map(runs[group][3])[variable] for group in owners}
        means = {group: float(np.mean(values)) for group, values in calibration_values.items()}
        between = float(np.var(np.asarray(tuple(means.values())), ddof=0))
        within_numerator = sum(
            (len(values) - 1) * float(np.var(np.asarray(values), ddof=1))
            for values in calibration_values.values()
            if len(values) > 1
        )
        within_denominator = sum(max(0, len(values) - 1) for values in calibration_values.values())
        within = within_numerator / within_denominator if within_denominator else 0.0
        heldout_disagreement = float(max(heldout_values.values()) - min(heldout_values.values()))
        calibration_disagreement = float(max(means.values()) - min(means.values()))
        score = between / (within + 1.0e-12)
        heldout_standardized = heldout_disagreement / (float(np.sqrt(within)) + 1.0e-12)
        records.append(
            {
                "variable": variable,
                "owners": owners,
                "owner_means": means,
                "heldout_values": heldout_values,
                "calibration_disagreement": calibration_disagreement,
                "heldout_disagreement": heldout_disagreement,
                "between_variance": between,
                "within_variance": within,
                "calibrated_score": score,
                "heldout_standardized_score": heldout_standardized,
                "owner_sigma_medians": {
                    group: float(
                        median(
                            _sigma_map(runs[group][replicate])[variable]
                            for replicate in range(PROPOSAL_REPLICATES)
                        )
                    )
                    for group in owners
                },
            }
        )
    return tuple(records)


def _sphere_equivalence(objective, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed ^ 0xA0B0)
    probes = rng.uniform(-BOUNDS, BOUNDS, size=(32, DIMENSION))
    weights = np.asarray(objective._weights, dtype=float)
    effective_a = np.zeros(DIMENSION, dtype=float)
    effective_b = np.zeros(DIMENSION, dtype=float)
    constant = 0.0
    for group_index, group in enumerate(objective.structure.groups):
        weight = float(weights[group_index])
        optimum = np.asarray(objective._optima[group_index], dtype=float)
        for variable, local_optimum in zip(group, optimum, strict=True):
            effective_a[variable] += weight
            effective_b[variable] += weight * float(local_optimum)
    means = effective_b / effective_a
    for group_index, group in enumerate(objective.structure.groups):
        weight = float(weights[group_index])
        optimum = np.asarray(objective._optima[group_index], dtype=float)
        for variable, local_optimum in zip(group, optimum, strict=True):
            constant += weight * (float(local_optimum) - means[variable]) ** 2
    original = np.asarray(objective.evaluate(probes), dtype=float)
    surrogate = np.sum(effective_a[np.newaxis, :] * (probes - means[np.newaxis, :]) ** 2, axis=1) + constant
    order_original = np.argsort(original, kind="stable")
    order_surrogate = np.argsort(surrogate, kind="stable")
    return {
        "max_absolute_error": float(np.max(np.abs(original - surrogate))),
        "mean_absolute_error": float(np.mean(np.abs(original - surrogate))),
        "ranking_identical": bool(np.array_equal(order_original, order_surrogate)),
        "effective_optimum": tuple(float(value) for value in means),
        "irreducible_constant": float(constant),
        "probe_count": len(probes),
    }


def _context(mode: str, topology: str, overlap_budget: int, seed: int) -> dict[str, object]:
    problem, objective, structure = _build(mode, topology, overlap_budget, seed)
    runs, consumed_fes, anchor_error = _run_replicates(problem, structure, seed)
    residuals = _calibrated_residuals(structure, runs)
    equivalence = _sphere_equivalence(objective, seed)
    return {
        "mode": mode,
        "topology": topology,
        "overlap_budget": overlap_budget,
        "seed": seed,
        "groups": structure.groups,
        "shared_variables": structure.shared_variables,
        "anchor_error": anchor_error,
        "consumed_fes": consumed_fes,
        "expected_fes": 1 + NUM_GROUPS * PROPOSAL_REPLICATES * PROPOSAL_BUDGET_FES,
        "replicate_count_per_group": tuple(len(group_runs) for group_runs in runs),
        "proposal_runs": tuple(
            tuple(
                {
                    "group": run.proposal.group,
                    "replicate": replicate,
                    "values": run.proposal.values,
                    "uncertainty": run.proposal.uncertainty,
                    "improvement": run.proposal.improvement,
                    "consumed_fes": run.consumed_fes,
                }
                for replicate, run in enumerate(group_runs)
            )
            for group_runs in runs
        ),
        "residuals": residuals,
        "equivalence": equivalence,
    }


def _summary(contexts: tuple[dict[str, object], ...], mode: str) -> dict[str, object]:
    rows = [item for item in contexts if item["mode"] == mode]
    residuals = [residual for row in rows for residual in row["residuals"]]
    return {
        "context_count": len(rows),
        "calibration_disagreement_median": float(median(item["calibration_disagreement"] for item in residuals)),
        "heldout_disagreement_median": float(median(item["heldout_disagreement"] for item in residuals)),
        "calibrated_score_median": float(median(item["calibrated_score"] for item in residuals)),
        "heldout_standardized_score_median": float(median(item["heldout_standardized_score"] for item in residuals)),
        "within_variance_median": float(median(item["within_variance"] for item in residuals)),
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
    contexts = tuple(sorted(contexts, key=lambda item: (item["topology"], item["overlap_budget"], item["seed"], item["mode"])))
    conforming = [item for item in contexts if item["mode"] == "conforming"]
    conflicting = [item for item in contexts if item["mode"] == "conflicting"]
    conflicting_scores = [float(residual["calibrated_score"]) for item in conflicting for residual in item["residuals"]]
    conforming_scores = [float(residual["calibrated_score"]) for item in conforming for residual in item["residuals"]]
    calibrated_auc = _auc(conflicting_scores, conforming_scores)
    gate_checks = {
        "context_count_60": len(contexts) == 60,
        "exact_replicates_and_fe": all(
            item["consumed_fes"] == item["expected_fes"] == 1 + NUM_GROUPS * PROPOSAL_REPLICATES * PROPOSAL_BUDGET_FES
            and item["replicate_count_per_group"] == (4, 4, 4, 4, 4, 4)
            for item in contexts
        ),
        "heldout_metrics_finite": all(
            np.isfinite(residual["heldout_standardized_score"])
            for item in contexts
            for residual in item["residuals"]
        ),
        "sphere_equivalence_all_contexts": all(
            item["equivalence"]["max_absolute_error"] <= 1.0e-8
            and item["equivalence"]["ranking_identical"]
            for item in contexts
        ),
        "calibrated_residual_reported_both_modes": bool(conforming and conflicting),
    }
    return {
        "schema_version": "arac-overlap-residual-calibration-gate11-v1",
        "protocol": {
            "dimension": DIMENSION,
            "num_groups": NUM_GROUPS,
            "base_function": BASE_FUNCTION,
            "rotation": False,
            "transforms": False,
            "proposal_algorithm": PROPOSAL_ALGORITHM,
            "proposal_budget_fes_each": PROPOSAL_BUDGET_FES,
            "proposal_replicates": PROPOSAL_REPLICATES,
            "calibration_replicates": (0, 1, 2),
            "heldout_replicate": 3,
            "seeds": FRESH_SEEDS,
            "topologies": TOPOLOGIES,
            "overlap_budgets": OVERLAP_BUDGETS,
        },
        "context_count": len(contexts),
        "contexts": contexts,
        "summary": {
            "conforming": _summary(contexts, "conforming"),
            "conflicting": _summary(contexts, "conflicting"),
            "calibrated_residual_auc_conflicting_over_conforming": calibrated_auc,
        },
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "scientific_findings": {
            "protocol_integrity_passed": all(gate_checks.values()),
            "residual_separation_supported": calibrated_auc >= 0.65,
            "hidden_conflict_identifiable_from_sphere_rankings": False,
        },
        "registered_limitation": (
            "The sphere equivalence gate is a required limitation audit. Even if calibrated residuals "
            "separate the generated modes, the black-box objective surface is an effective quadratic plus "
            "a constant, so hidden subgroup conflict is not identifiable from rankings alone."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_residual_calibration_gate11/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
