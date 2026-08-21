"""v6.0-a oracle gate for frozen counterfactual coupling receipts.

The gate uses deterministic quadratic overlap objectives so the expected sign
of ``G_coupled`` is known before running the coordinator.  It is deliberately
an offline diagnostic: production scheduling is not modified by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import (
    LocalProposal,
    OverlapCoordinator,
    OverlapStructure,
    evaluate_frozen_private_counterfactual,
)
from arac.runtime.ledger import EvaluationLedger


OUTPUT_SCHEMA = "arac-oc-coupling-diagnostic-gate-v1"
TOPOLOGIES = ("chain", "star")
REGIMES = ("none", "synergy", "conflict", "neutral")
SEEDS = (2026082001, 2026082002)
PATCH_BUDGET_FES = 8


@dataclass(frozen=True)
class Cell:
    topology: str
    regime: str
    seed: int


@dataclass(frozen=True)
class ControlledOverlapObjective:
    topology: str
    regime: str
    target: float
    interaction_strength: float = 0.5

    def __call__(self, values: np.ndarray) -> float | np.ndarray:
        converted = np.asarray(values, dtype=float)
        single = converted.ndim == 1
        batch = converted[np.newaxis, :] if single else converted
        if batch.ndim != 2 or batch.shape[1] != 4:
            raise ValueError("controlled objective expects dimension four")
        if self.topology == "chain":
            shared = batch[:, 1]
            private = batch[:, (0, 2)]
        elif self.topology == "star":
            shared = batch[:, 0]
            private = batch[:, (1, 2, 3)]
        else:
            raise ValueError("unknown topology")
        private_mean = np.mean(private, axis=1)
        private_loss = 5.0 * np.mean((private - self.target) ** 2, axis=1)
        if self.regime in {"none", "neutral"}:
            interaction = np.zeros(batch.shape[0], dtype=float)
        elif self.regime == "synergy":
            interaction = self.interaction_strength * (private_mean - shared) ** 2
        elif self.regime == "conflict":
            interaction = self.interaction_strength * (private_mean + shared) ** 2
        else:
            raise ValueError("unknown regime")
        return float((private_loss + interaction)[0]) if single else private_loss + interaction


def _structure(topology: str) -> tuple[OverlapStructure, tuple[int, ...], tuple[int, ...]]:
    if topology == "chain":
        structure = OverlapStructure(dimension=4, groups=((0, 1), (1, 2), (3,)))
        return structure, (0, 1), (1,)
    if topology == "star":
        structure = OverlapStructure(dimension=4, groups=((0, 1), (0, 2), (0, 3)))
        return structure, (0, 1, 2), (0,)
    raise ValueError("unknown topology")


def _target(seed: int) -> float:
    return 1.0 + 0.05 * float(seed % 5)


def _candidate_target(topology: str, regime: str, target: float) -> np.ndarray:
    candidate = np.full(4, target, dtype=float)
    if regime == "neutral":
        candidate[1 if topology == "chain" else 0] = 0.0
    return candidate


def _proposals(structure: OverlapStructure, candidate: np.ndarray) -> tuple[LocalProposal, ...]:
    result = []
    for group, variables in enumerate(structure.groups):
        result.append(
            LocalProposal(
                group=group,
                values=tuple((variable, float(candidate[variable])) for variable in variables),
                improvement=1.0,
                uncertainty=tuple((variable, 0.01) for variable in variables),
            )
        )
    return tuple(result)


def _selected_candidate(result):
    candidates = {candidate.name: candidate for candidate in result.candidates}
    candidates.pop("incumbent", None)
    errors = dict(result.candidate_errors)
    if result.accepted_candidate in candidates:
        name = result.accepted_candidate
    else:
        name = min(candidates, key=lambda item: (errors[item], item))
    return name, candidates[name], float(errors[name])


def _run_cell(cell: Cell) -> dict[str, object]:
    structure, component, scope = _structure(cell.topology)
    target = _target(cell.seed)
    candidate = _candidate_target(cell.topology, cell.regime, target)
    objective = ControlledOverlapObjective(cell.topology, cell.regime, target)
    problem = OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-2.0,) * 4,
        upper_bounds=(2.0,) * 4,
    )
    incumbent = np.zeros(4, dtype=float)
    initial_error = float(objective(incumbent))
    proposals = _proposals(structure, candidate)
    component_proposals = tuple(proposal for proposal in proposals if proposal.group in component)

    ledger = EvaluationLedger(
        problem,
        total_budget=16,
        initial_count=1,
        initial_incumbent=tuple(incumbent),
        initial_error=initial_error,
    )
    coordinator = OverlapCoordinator(structure, ledger)
    arbitration = coordinator.coordinate(
        component,
        component_proposals,
        reuse_incumbent=True,
    )
    candidate_name, selected, selected_error = _selected_candidate(arbitration)
    receipt = evaluate_frozen_private_counterfactual(
        ledger,
        component=component,
        scope=scope,
        incumbent=incumbent,
        best_error_before=arbitration.best_error_before,
        candidate_name=candidate_name,
        candidate=selected.vector,
        full_candidate_error=selected_error,
    )

    repair_ledger = EvaluationLedger(
        problem,
        total_budget=1 + PATCH_BUDGET_FES,
        initial_count=1,
        initial_incumbent=tuple(incumbent),
        initial_error=initial_error,
    )
    repair_coordinator = OverlapCoordinator(structure, repair_ledger)
    repair_before = float(repair_ledger.best_error)
    repair_consumed = repair_coordinator.dispatch_repair(
        component,
        component_proposals,
        budget_fes=PATCH_BUDGET_FES,
        seed=cell.seed,
        strategy="sequential_joint_patch",
    )
    repair_after = float(repair_ledger.best_error)

    return {
        "cell": asdict(cell),
        "target": target,
        "component": list(component),
        "scope": list(scope),
        "candidate": [float(value) for value in candidate],
        "arbitration": {
            "candidate_name": candidate_name,
            "best_error_before": arbitration.best_error_before,
            "best_error_after": arbitration.best_error_after,
            "gain": arbitration.best_error_before - arbitration.best_error_after,
            "consumed_fes": ledger.count - 1,
            "candidate_errors": dict(arbitration.candidate_errors),
        },
        "counterfactual": receipt.payload(),
        "repair": {
            "consumed_fes": repair_consumed,
            "best_error_before": repair_before,
            "best_error_after": repair_after,
            "gain": repair_before - repair_after,
            "strict_best": repair_after <= repair_before,
        },
        "strict_best": arbitration.best_error_after <= arbitration.best_error_before,
        "archive_preserved": receipt.archive_preserved,
        "terminal_fes": ledger.count,
    }


def _correlation(left: list[float], right: list[float], *, rank: bool) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if rank:
        x = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
        y = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_gate() -> dict[str, object]:
    cells = tuple(Cell(topology, regime, seed) for topology in TOPOLOGIES for regime in REGIMES for seed in SEEDS)
    results = [_run_cell(cell) for cell in cells]
    by_regime = {
        regime: [row for row in results if row["cell"]["regime"] == regime]
        for regime in REGIMES
    }
    coupled = [float(row["counterfactual"]["coupled_gain"]) for row in results]
    repair = [float(row["repair"]["gain"]) for row in results]
    median_coupled = {
        regime: float(np.median([row["counterfactual"]["coupled_gain"] for row in rows]))
        for regime, rows in by_regime.items()
    }
    checks = {
        "cell_count_16": len(results) == 16,
        "counterfactual_one_fe": all(row["counterfactual"]["consumed_fes"] == 1 for row in results),
        "archive_preserved": all(row["archive_preserved"] for row in results),
        "arbitration_strict_best": all(row["strict_best"] for row in results),
        "repair_strict_best": all(row["repair"]["strict_best"] for row in results),
        "repair_exact_budget": all(row["repair"]["consumed_fes"] == PATCH_BUDGET_FES for row in results),
        "none_near_zero": all(abs(row["counterfactual"]["coupled_gain"]) <= 1e-10 for row in by_regime["none"]),
        "neutral_near_zero": all(abs(row["counterfactual"]["coupled_gain"]) <= 1e-10 for row in by_regime["neutral"]),
        "synergy_positive": median_coupled["synergy"] > 0.05,
        "conflict_negative": median_coupled["conflict"] < -0.05,
        "coupling_repair_correlations_finite": all(
            np.isfinite(_correlation(coupled, repair, rank=rank)) for rank in (False, True)
        ),
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "topologies": TOPOLOGIES,
            "regimes": REGIMES,
            "seeds": SEEDS,
            "patch_budget_fes": PATCH_BUDGET_FES,
            "note": "offline diagnostic only; does not alter production scheduling",
        },
        "results": results,
        "summary": {
            "median_coupled_gain_by_regime": median_coupled,
            "coupled_gain_repair_gain_pearson": _correlation(coupled, repair, rank=False),
            "coupled_gain_repair_gain_spearman": _correlation(coupled, repair, rank=True),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oc_coupling_diagnostic_gate/confirmation.json"),
    )
    args = parser.parse_args()
    payload = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
