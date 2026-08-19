from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery.independent_semantic_parity_pilot import (
    _candidate_mechanism_passed,
    _summarize,
    execute_historical_semantic_port,
    load_protocol,
)


def _problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )


def _checkpoint() -> PhaseCheckpoint:
    return PhaseCheckpoint(
        protocol="semantic-parity-test",
        run_seed=117,
        total_budget_fes=4_900,
        phase1_fes=100,
        incumbent=(0.0,) * 40,
        incumbent_error=0.0,
        feature_names=("dummy",),
        feature_values=(0.0,),
        blocks=tuple(tuple(range(start, start + 10)) for start in range(0, 40, 10)),
    )


def _contract(action: str) -> dict[str, object]:
    if action == "aor":
        return {
            "optimizer_route": "full_space_sep_cmaes",
            "initial_mean": 0.0,
            "sigma": 0.5,
            "population_size": 24,
            "restart": False,
        }
    if action == "ctp":
        return {
            "action": {
                "coverage_sweeps": 4,
                "group_polish_mode": "unbounded_group_polish",
                "restart_policy": "none",
            }
        }
    if action == "smp":
        return {
            "action": {
                "name": "smp",
                "stale_window": 3,
                "state_fields": [
                    "covariance",
                    "path_sigma",
                    "sigma",
                    "mean",
                    "rng_state",
                ],
            }
        }
    return {
        "execution_mode": "one_native_sweep_burst_then_native",
        "native_resumed": True,
        "candidate_action_budget_fes": 154_731,
        "terminal_fe": 3_000_000,
        "source_group_actual_fes": [10_000, 12_000, 8_000, 11_000],
    }


def test_protocol_binds_four_historical_semantic_contracts() -> None:
    protocol = load_protocol()

    assert protocol["arms"] == ["current_production", "historical_semantic_port"]
    assert {lane["action"] for lane in protocol["lanes"]} == {
        "aor",
        "ctp",
        "smp",
        "gcb",
    }
    assert all("reference_contract" in lane for lane in protocol["lanes"])


@pytest.mark.parametrize("action", ("aor", "ctp", "smp", "gcb"))
def test_historical_semantic_port_consumes_exact_screen_budget(action: str) -> None:
    problem = _problem()
    checkpoint = _checkpoint()
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    context = ActionContext(action, checkpoint, problem, ledger, action_seed=117)

    result, events = execute_historical_semantic_port(context, _contract(action))

    assert result.terminal_fes == checkpoint.total_budget_fes
    assert result.consumed_fes == 4_800
    assert result.final_error == 0.0
    assert _candidate_mechanism_passed(action, events) is True


def test_summary_does_not_authorize_selector_or_terminal_parity() -> None:
    receipts = []
    for action in ("ctp", "smp", "gcb", "aor"):
        receipts.append(
            {
                "action": action,
                "case_id_audit_metadata": action.upper(),
                "exact_screen_fes": True,
                "same_screen_checkpoint": True,
                "candidate_mechanism_passed": True,
                "native_thread_limit_verified": True,
                "historical_terminal_parity_evaluated": False,
                "receipt_hash": action * 16,
                "arms": {
                    "current_production": {
                        "reference_mechanism_evaluated": False,
                        "reference_mechanism_match": None,
                        "final_error": 2.0,
                    },
                    "historical_semantic_port": {"final_error": 1.0},
                },
            }
        )

    summary = _summarize(receipts)

    assert summary["mechanism_screen_passed"] is True
    assert summary["current_reference_evaluated_count"] == 0
    assert summary["all_native_thread_limits_verified"] is True
    assert summary["historical_terminal_parity_evaluated"] is False
    assert summary["selector_evaluation_authorized"] is False


def test_contract_only_events_cannot_pass_live_mechanism_gate() -> None:
    assert (
        _candidate_mechanism_passed(
            "smp",
            {
                "state_persistence": True,
                "stale_window": 3,
                "state_fields": ["covariance", "path_sigma", "sigma", "mean", "rng_state"],
            },
        )
        is False
    )
    assert (
        _candidate_mechanism_passed(
            "ctp",
            {
                "coverage_sweeps": 4,
                "polish_mode": "unbounded_group_polish",
                "polish_fes": 100,
            },
        )
        is False
    )
