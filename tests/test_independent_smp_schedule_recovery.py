from __future__ import annotations

from copy import deepcopy

import numpy as np

import experiments.historical_recovery.independent_smp_schedule_recovery as recovery
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger
from experiments.historical_recovery.independent_smp_schedule_recovery import (
    _historical_facts,
    _mechanism_gate,
    execute_schedule,
    load_protocol,
)


def _problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )


def _checkpoint(total_budget_fes: int = 100_100) -> PhaseCheckpoint:
    return PhaseCheckpoint(
        protocol="smp-long-visit-test",
        run_seed=117,
        total_budget_fes=total_budget_fes,
        phase1_fes=100,
        incumbent=(0.0,) * 40,
        incumbent_error=0.0,
        feature_names=("dummy",),
        feature_values=(0.0,),
        blocks=tuple(tuple(range(start, start + 10)) for start in range(0, 40, 10)),
    )


def _context(total_budget_fes: int = 100_100) -> ActionContext:
    problem = _problem()
    checkpoint = _checkpoint(total_budget_fes)
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    return ActionContext("smp", checkpoint, problem, ledger, action_seed=117)


def test_protocol_binds_exp052_artifact_and_source_limits() -> None:
    protocol = load_protocol()
    facts = _historical_facts(protocol)

    assert facts["record_count"] == 165
    assert facts["restore_count"] == 122
    assert facts["reset_count"] == 36
    assert facts["visits_per_group"] == [9, 9, 9, 9, 9] + [8] * 15
    assert facts["exact_runner_source_available"] is False
    assert facts["historical_p90"] == 1.8255606813339802


def test_schedule_uses_long_visits_and_group_level_resets() -> None:
    context = _context()
    protocol = load_protocol()

    schedule = execute_schedule(context, context.ledger.remaining)

    assert schedule["actual_fes"] == 100_000
    assert schedule["end_fes"] == 100_100
    assert schedule["visit_count"] < 100
    assert schedule["median_actual_population_batches"] >= 10
    assert schedule["restart_count"] > 0
    assert schedule["stagnation_reset_count"] == schedule["restart_count"]
    assert schedule["terminal_group_states_finite"] is True
    assert schedule["restart_identity_change_count"] == schedule["restart_count"]
    assert schedule["cross_visit_state_digest_matches"] == schedule[
        "cross_visit_boundary_checks"
    ]
    long_requests = sum(
        event["requested_fes"] > event["population_size"]
        for event in schedule["events"]
    )
    assert long_requests >= schedule["visit_count"] - 1
    assert _mechanism_gate(schedule, protocol, terminal=True) is True


def test_mechanism_gate_rejects_generation_sized_visits() -> None:
    context = _context()
    protocol = load_protocol()
    schedule = execute_schedule(context, context.ledger.remaining)
    invalid = deepcopy(schedule)
    invalid["median_actual_population_batches"] = 1

    assert _mechanism_gate(invalid, protocol, terminal=True) is False


def test_numerical_restart_keeps_consumed_fes_inside_the_same_visit(
    monkeypatch,
) -> None:
    protocol = load_protocol()
    baseline = execute_schedule(_context(4_100), 4_000, protocol=protocol)
    original_advance = recovery._PersistentBlockSession.advance
    injected = False

    def advance_with_one_consumed_generation_failure(self, **kwargs) -> None:
        nonlocal injected
        original_advance(self, **kwargs)
        if not injected and self.block_index == 0:
            injected = True
            self.optimizer.sigma = np.inf
            raise FloatingPointError("injected post-evaluation overflow")

    monkeypatch.setattr(
        recovery._PersistentBlockSession,
        "advance",
        advance_with_one_consumed_generation_failure,
    )
    context = _context(4_100)
    schedule = execute_schedule(context, 4_000, protocol=protocol)

    assert injected is True
    assert schedule["actual_fes"] == 4_000
    assert context.ledger.count == 4_100
    assert schedule["visit_count"] == baseline["visit_count"]
    assert schedule["stagnation_reset_count"] == baseline["stagnation_reset_count"]
    assert schedule["numerical_restart_count"] == 1
    assert schedule["numerical_restart_affected_visits"] == 1
    assert schedule["optimizer_identity_preserved_visits"] == schedule["visit_count"] - 1
    assert schedule["terminal_group_states_finite"] is True
    restart = schedule["numerical_restart_events"][0]
    assert restart["generation_consumed_fes"] == restart["population_size"]
    assert restart["optimizer_identity_changed"] is True
    assert restart["state_finite_after_restart"] is True
    assert _mechanism_gate(schedule, protocol, terminal=True) is True
