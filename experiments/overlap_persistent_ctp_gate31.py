"""Gate 31: verify persistent CTP lifecycle and exact-budget integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import LocalProposal, OverlapCoordinator, OverlapStructure
from arac.evidence import run_phase1_overlap_pilot
from arac.overlap_core import PERSISTENT_CTP_MODE, run_overlap_from_pilot
from arac.runtime.ledger import EvaluationLedger


OUTPUT = Path("artifacts/overlap_persistent_ctp_gate31/confirmation_fresh.json")


def _small_problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        result += 0.25 * batch[:, 0] ** 2 * batch[:, 1] ** 2
        result += 0.25 * batch[:, 1] ** 2 * batch[:, 2] ** 2
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )


def _proposal(group: int, value: float, sigma: float = 0.05) -> LocalProposal:
    variables = ((0, 0.0), (1, value)) if group == 0 else ((1, value), (2, 0.0))
    return LocalProposal(
        group=group,
        values=variables,
        improvement=1.0,
        uncertainty=tuple((variable, sigma) for variable, _ in variables),
    )


def _direct_ctp_probe() -> dict[str, object]:
    problem = OptimizationProblem(
        objective=lambda values: np.sum(np.asarray(values, dtype=float) ** 2, axis=-1),
        dimension=3,
        lower_bounds=(-5.0,) * 3,
        upper_bounds=(5.0,) * 3,
    )
    structure = OverlapStructure(
        dimension=3,
        groups=((0, 1), (1, 2)),
        member_confidences=((1, 0, 0.9), (1, 1, 0.8)),
    )
    ledger = EvaluationLedger(
        problem,
        total_budget=32,
        initial_count=1,
        initial_incumbent=(2.0, 2.0, 2.0),
        initial_error=12.0,
    )
    coordinator = OverlapCoordinator(structure, ledger)
    proposals = (_proposal(0, -2.0), _proposal(1, 2.0))
    first = coordinator.coordinate((0, 1), proposals, ctp_budget_fes=8, ctp_seed=11)
    second = coordinator.coordinate((0, 1), proposals, ctp_budget_fes=8, ctp_seed=11)
    return {
        "first_conflict_streak": first.conflict_streak,
        "first_ctp_triggered": first.ctp_triggered,
        "second_conflict_streak": second.conflict_streak,
        "second_ctp_triggered": second.ctp_triggered,
        "second_ctp_fes": second.ctp_consumed_fes,
        "ledger_fes": ledger.count,
        "strict_best": second.best_error_after <= second.best_error_before,
    }


def _integration_probe() -> dict[str, object]:
    problem = _small_problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=2_000,
        run_seed=89,
        anchors=((-1.0,) * 4, (1.0,) * 4),
        step=0.25,
        rounds=8,
        bucket_size=2,
        max_candidate_pairs=16,
    )
    result = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=PERSISTENT_CTP_MODE,
        refresh_cycles=3,
        neighborhood_fes=8,
    )
    return {
        "checkpoint_hash": result.phase1.checkpoint.checkpoint_hash,
        "phase1_fes": result.phase1.consumed_fes,
        "terminal_fes": result.terminal_fes,
        "phase2_consumed_fes": result.phase2_consumed_fes,
        "strict_best": all(item.best_error_after <= item.best_error_before for item in result.cycles),
        "ctp_fes": [item.ctp_fes for item in result.cycles],
        "ctp_triggered_components": [item.ctp_triggered_components for item in result.cycles],
        "max_conflict_streak": [item.max_conflict_streak for item in result.cycles],
        "final_error": result.final_error,
    }


def main() -> int:
    payload = {
        "schema_version": "arac-overlap-persistent-ctp-gate31-v1",
        "direct_ctp": _direct_ctp_probe(),
        "integration": _integration_probe(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
