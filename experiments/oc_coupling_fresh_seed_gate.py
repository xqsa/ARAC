"""Fresh-seed replication gate for the v6.0-a coupling receipt.

This module deliberately reuses the controlled oracle setup from the small
diagnostic gate but varies target and interaction strength over a preregistered
seed set.  It reports bootstrap confidence intervals and never changes the
production coordinator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

try:
    from experiments.oc_coupling_diagnostic_gate import (
        PATCH_BUDGET_FES,
        ControlledOverlapObjective,
        _proposals,
        _structure,
    )
except ModuleNotFoundError:  # direct ``python experiments/script.py`` entry
    from oc_coupling_diagnostic_gate import (  # type: ignore[no-redef]
        PATCH_BUDGET_FES,
        ControlledOverlapObjective,
        _proposals,
        _structure,
    )
from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import (
    OverlapCoordinator,
    evaluate_frozen_private_counterfactual,
)
from arac.runtime.ledger import EvaluationLedger


OUTPUT_SCHEMA = "arac-oc-coupling-fresh-seed-gate-v1"
TOPOLOGIES = ("chain", "star")
REGIMES = ("none", "synergy", "conflict", "neutral")
SEEDS = tuple(range(2026082101, 2026082126))
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 2026082099
AUTHORITY_THRESHOLD = 0.30


@dataclass(frozen=True)
class FreshCell:
    topology: str
    regime: str
    seed: int


def _parameters(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    target = float(rng.uniform(0.75, 1.25))
    strength = float(rng.uniform(0.25, 0.85))
    return target, strength


def _candidate(topology: str, regime: str, target: float) -> np.ndarray:
    candidate = np.full(4, target, dtype=float)
    if regime == "neutral":
        candidate[1 if topology == "chain" else 0] = 0.0
    return candidate


def run_cell(cell: FreshCell) -> dict[str, object]:
    structure, component, scope = _structure(cell.topology)
    target, strength = _parameters(cell.seed)
    candidate = _candidate(cell.topology, cell.regime, target)
    objective = ControlledOverlapObjective(
        cell.topology,
        cell.regime,
        target,
        interaction_strength=strength,
    )
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
    arbitration = coordinator.coordinate(component, component_proposals, reuse_incumbent=True)
    candidates = {item.name: item for item in arbitration.candidates if item.name != "incumbent"}
    errors = dict(arbitration.candidate_errors)
    candidate_name = arbitration.accepted_candidate if arbitration.accepted_candidate in candidates else min(candidates, key=lambda name: (errors[name], name))
    selected = candidates[candidate_name]
    receipt = evaluate_frozen_private_counterfactual(
        ledger,
        component=component,
        scope=scope,
        incumbent=incumbent,
        best_error_before=arbitration.best_error_before,
        candidate_name=candidate_name,
        candidate=selected.vector,
        full_candidate_error=float(errors[candidate_name]),
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
        "interaction_strength": strength,
        "counterfactual": receipt.payload(),
        "repair_gain": repair_before - repair_after,
        "repair_consumed_fes": repair_consumed,
        "strict_best": bool(
            arbitration.best_error_after <= arbitration.best_error_before
            and repair_after <= repair_before
        ),
        "archive_preserved": receipt.archive_preserved,
        "counterfactual_one_fe": receipt.consumed_fes == 1,
    }


def _quantile_interval(values: list[float], *, seed: int) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(BOOTSTRAP_REPLICATES, data.size), replace=True)
    medians = np.median(samples, axis=1)
    return (
        float(np.median(data)),
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def _correlation_interval(left: list[float], right: list[float], *, rank: bool) -> tuple[float, float, float]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(rank))
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_REPLICATES, len(x)))
    values = []
    for sample in indices:
        sx = x[sample]
        sy = y[sample]
        if rank:
            sx = np.argsort(np.argsort(sx, kind="stable"), kind="stable").astype(float)
            sy = np.argsort(np.argsort(sy, kind="stable"), kind="stable").astype(float)
        if np.std(sx) == 0.0 or np.std(sy) == 0.0:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(sx, sy)[0, 1]))
    return (
        float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0,
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def run_gate(*, workers: int = 1) -> dict[str, object]:
    cells = tuple(FreshCell(topology, regime, seed) for topology in TOPOLOGIES for regime in REGIMES for seed in SEEDS)
    if workers == 1:
        results = [run_cell(cell) for cell in cells]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(cells))) as executor:
            results = list(executor.map(run_cell, cells))
    results.sort(key=lambda row: tuple(row["cell"].values()))
    by_regime = {
        regime: [row for row in results if row["cell"]["regime"] == regime]
        for regime in REGIMES
    }
    by_topology_regime = {
        f"{topology}:{regime}": [
            row for row in results
            if row["cell"]["topology"] == topology and row["cell"]["regime"] == regime
        ]
        for topology in TOPOLOGIES for regime in REGIMES
    }
    regime_intervals = {
        regime: _quantile_interval(
            [float(row["counterfactual"]["coupled_gain"]) for row in rows],
            seed=BOOTSTRAP_SEED + index,
        )
        for index, (regime, rows) in enumerate(by_regime.items())
    }
    topology_regime_intervals = {
        key: _quantile_interval(
            [float(row["counterfactual"]["coupled_gain"]) for row in rows],
            seed=BOOTSTRAP_SEED + 100 + index,
        )
        for index, (key, rows) in enumerate(by_topology_regime.items())
    }
    coupled = [float(row["counterfactual"]["coupled_gain"]) for row in results]
    repair = [float(row["repair_gain"]) for row in results]
    pearson = _correlation_interval(coupled, repair, rank=False)
    spearman = _correlation_interval(coupled, repair, rank=True)
    checks = {
        "cell_count_200": len(results) == 200,
        "cell_contracts": all(
            row["counterfactual_one_fe"]
            and row["archive_preserved"]
            and row["strict_best"]
            and row["repair_consumed_fes"] == PATCH_BUDGET_FES
            for row in results
        ),
        "none_ci_contains_zero": regime_intervals["none"][1] <= 0.0 <= regime_intervals["none"][2],
        "neutral_ci_contains_zero": regime_intervals["neutral"][1] <= 0.0 <= regime_intervals["neutral"][2],
        "synergy_ci_positive": regime_intervals["synergy"][1] > 0.0,
        "conflict_ci_negative": regime_intervals["conflict"][2] < 0.0,
        "topology_sign_consistent": all(
            (topology_regime_intervals[f"{topology}:synergy"][1] > 0.0)
            and (topology_regime_intervals[f"{topology}:conflict"][2] < 0.0)
            for topology in TOPOLOGIES
        ),
        "correlations_finite": all(np.isfinite(interval[0]) for interval in (pearson, spearman)),
    }
    promotion = {
        "authority_threshold": AUTHORITY_THRESHOLD,
        "spearman_ci_lower": spearman[1],
        "promotion_recommended": spearman[1] >= AUTHORITY_THRESHOLD,
        "reason": "promote only when the lower 95% Spearman CI clears the preregistered threshold",
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "topologies": TOPOLOGIES,
            "regimes": REGIMES,
            "seeds": SEEDS,
            "cell_count": len(cells),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "authority_threshold": AUTHORITY_THRESHOLD,
            "production_scheduler_modified": False,
        },
        "summary": {
            "regime_coupled_gain_ci": regime_intervals,
            "topology_regime_coupled_gain_ci": topology_regime_intervals,
            "pearson_ci": pearson,
            "spearman_ci": spearman,
            "promotion": promotion,
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oc_coupling_fresh_seed_gate/confirmation.json"),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
