"""Paired fresh gate comparing one- and two-baseline coupling signals."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

try:
    from experiments.oc_coupling_fresh_seed_gate import (
        AUTHORITY_THRESHOLD,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        PATCH_BUDGET_FES,
        REGIMES,
        SEEDS,
        TOPOLOGIES,
        FreshCell,
        _candidate,
        _parameters,
        _correlation_interval,
        _quantile_interval,
        _structure,
    )
except ModuleNotFoundError:  # direct ``python experiments/script.py`` entry
    from oc_coupling_fresh_seed_gate import (  # type: ignore[no-redef]
        AUTHORITY_THRESHOLD,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        PATCH_BUDGET_FES,
        REGIMES,
        SEEDS,
        TOPOLOGIES,
        FreshCell,
        _candidate,
        _parameters,
        _correlation_interval,
        _quantile_interval,
        _structure,
    )

try:
    from experiments.oc_coupling_diagnostic_gate import (
        ControlledOverlapObjective,
        _proposals,
    )
except ModuleNotFoundError:  # direct script entry after local fallback imports
    from oc_coupling_diagnostic_gate import (  # type: ignore[no-redef]
        ControlledOverlapObjective,
        _proposals,
    )
from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import (
    OverlapCoordinator,
    evaluate_two_baseline_counterfactual,
)
from arac.runtime.ledger import EvaluationLedger


OUTPUT_SCHEMA = "arac-oc-coupling-two-baseline-gate-v1"


def _select_candidate(arbitration):
    candidates = {item.name: item for item in arbitration.candidates if item.name != "incumbent"}
    errors = dict(arbitration.candidate_errors)
    name = (
        arbitration.accepted_candidate
        if arbitration.accepted_candidate in candidates
        else min(candidates, key=lambda item: (errors[item], item))
    )
    return name, candidates[name], float(errors[name])


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
        total_budget=20,
        initial_count=1,
        initial_incumbent=tuple(incumbent),
        initial_error=initial_error,
    )
    coordinator = OverlapCoordinator(structure, ledger)
    arbitration = coordinator.coordinate(component, component_proposals, reuse_incumbent=True)
    candidate_name, selected, selected_error = _select_candidate(arbitration)
    receipt = evaluate_two_baseline_counterfactual(
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
    conditional_shared_gain = receipt.full_gain - receipt.private_gain
    return {
        "cell": asdict(cell),
        "target": target,
        "interaction_strength": strength,
        "receipt": receipt.payload(),
        "signals": {
            "conditional_shared_gain": conditional_shared_gain,
            "interaction_gain": receipt.interaction_gain,
            "interaction_abs_gain": abs(receipt.interaction_gain),
        },
        "repair_gain": repair_before - repair_after,
        "repair_consumed_fes": repair_consumed,
        "counterfactual_two_fe": receipt.consumed_fes == 2,
        "archive_preserved": receipt.archive_preserved,
        "strict_best": bool(
            arbitration.best_error_after <= arbitration.best_error_before
            and repair_after <= repair_before
        ),
    }


def _regime_intervals(results: list[dict[str, object]], field: str) -> dict[str, tuple[float, float, float]]:
    output = {}
    for index, regime in enumerate(REGIMES):
        values = [
            float(row["signals"][field])
            for row in results
            if row["cell"]["regime"] == regime
        ]
        output[regime] = _quantile_interval(values, seed=BOOTSTRAP_SEED + 500 + index)
    return output


def run_gate(*, workers: int = 1) -> dict[str, object]:
    cells = tuple(FreshCell(topology, regime, seed) for topology in TOPOLOGIES for regime in REGIMES for seed in SEEDS)
    if workers == 1:
        results = [run_cell(cell) for cell in cells]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(cells))) as executor:
            results = list(executor.map(run_cell, cells))
    results.sort(key=lambda row: tuple(row["cell"].values()))
    repair = [float(row["repair_gain"]) for row in results]
    signals = {
        field: [float(row["signals"][field]) for row in results]
        for field in ("conditional_shared_gain", "interaction_gain", "interaction_abs_gain")
    }
    correlation_intervals = {
        field: {
            "pearson": _correlation_interval(values, repair, rank=False),
            "spearman": _correlation_interval(values, repair, rank=True),
        }
        for field, values in signals.items()
    }
    interaction_intervals = _regime_intervals(results, "interaction_gain")
    checks = {
        "cell_count_200": len(results) == 200,
        "cell_contracts": all(
            row["counterfactual_two_fe"]
            and row["archive_preserved"]
            and row["strict_best"]
            and row["repair_consumed_fes"] == PATCH_BUDGET_FES
            for row in results
        ),
        "interaction_none_ci_contains_zero": interaction_intervals["none"][1] <= 0.0 <= interaction_intervals["none"][2],
        "interaction_neutral_ci_contains_zero": interaction_intervals["neutral"][1] <= 0.0 <= interaction_intervals["neutral"][2],
        "interaction_synergy_ci_positive": interaction_intervals["synergy"][1] > 0.0,
        "interaction_conflict_ci_negative": interaction_intervals["conflict"][2] < 0.0,
        "correlations_finite": all(
            np.isfinite(interval["spearman"][0])
            for interval in correlation_intervals.values()
        ),
    }
    absolute_spearman = correlation_intervals["interaction_abs_gain"]["spearman"]
    promotion = {
        "authority_threshold": AUTHORITY_THRESHOLD,
        "absolute_interaction_spearman_ci_lower": absolute_spearman[1],
        "promotion_recommended": absolute_spearman[1] >= AUTHORITY_THRESHOLD,
        "reason": "two-baseline interaction is promoted only when absolute interaction Spearman CI clears threshold",
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
            "interaction_gain_regime_ci": interaction_intervals,
            "signal_repair_correlation_ci": correlation_intervals,
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
        default=Path("artifacts/oc_coupling_two_baseline_gate/confirmation.json"),
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
