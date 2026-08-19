from __future__ import annotations

import json

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import (
    Phase2StateError,
    ResumablePhase2State,
    validate_snapshot_context,
)


def _problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=2,
        lower_bounds=(-5.0, -5.0),
        upper_bounds=(5.0, 5.0),
    )


def _context(*, action_name: str = "aor", count: int = 2) -> ActionContext:
    problem = _problem()
    checkpoint = PhaseCheckpoint(
        protocol="phase2-test-v1",
        run_seed=3,
        total_budget_fes=20,
        phase1_fes=2,
        incumbent=(1.0, 1.0),
        incumbent_error=2.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=((0,), (1,)),
    )
    if count == checkpoint.phase1_fes:
        ledger = EvaluationLedger.from_checkpoint(
            problem,
            total_budget=checkpoint.total_budget_fes,
            phase1_fes=checkpoint.phase1_fes,
            incumbent=checkpoint.incumbent,
            incumbent_error=checkpoint.incumbent_error,
        )
    else:
        raise ValueError("test context only supports the frozen checkpoint boundary")
    return ActionContext(action_name, checkpoint, problem, ledger, action_seed=17)


class _CounterState(ResumablePhase2State):
    def __init__(self, context: ActionContext, *, ticks: int = 0) -> None:
        super().__init__(context)
        self.ticks = ticks
        self.trace: list[int] = []

    def _advance(self, budget_fes: int) -> None:
        for _ in range(budget_fes):
            self.ticks += 1
            candidate = np.asarray((1.0 / (self.ticks + 1),) * 2, dtype=float)
            self.context.ledger.evaluate(candidate)
            self.trace.append(self.ticks)

    def _snapshot_payload(self) -> bytes:
        return json.dumps(
            {"ticks": self.ticks, "trace": self.trace},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def restore(cls, context: ActionContext, snapshot):
        validate_snapshot_context(context, snapshot)
        payload = json.loads(snapshot.state_payload.decode("utf-8"))
        state = cls(context, ticks=int(payload["ticks"]))
        state.trace = [int(value) for value in payload["trace"]]
        if state.snapshot().state_hash != snapshot.state_hash:
            raise Phase2StateError("counter state payload does not match snapshot")
        return state


def test_step_uses_exact_requested_budget_and_reports_prefix_state() -> None:
    state = _CounterState(_context())

    result = state.step(5)

    assert result.step_fes == 5
    assert result.consumed_fes == 5
    assert result.total_fes == 18
    assert result.complete is False
    assert state.context.ledger.count == 7
    assert state.trace == [1, 2, 3, 4, 5]
    assert result.state_hash == state.snapshot().state_hash


def test_snapshot_restore_preserves_the_same_prefix_and_no_duplicate_fe() -> None:
    resumed = _CounterState(_context())
    resumed.step(5)
    snapshot = resumed.snapshot()

    restored_context = ActionContext(
        "aor",
        resumed.context.checkpoint,
        resumed.context.problem,
        EvaluationLedger.from_phase2_snapshot(
            resumed.context.problem,
            snapshot,
        ),
        action_seed=17,
    )
    restored = _CounterState.restore(restored_context, snapshot)
    restored.step(5)

    uninterrupted = _CounterState(_context())
    uninterrupted.step(10)

    assert restored.trace == uninterrupted.trace
    assert restored.ticks == uninterrupted.ticks == 10
    assert restored.context.ledger.count == uninterrupted.context.ledger.count == 12
    assert restored.context.ledger.best_error == uninterrupted.context.ledger.best_error
    assert restored.context.ledger.best_x.tolist() == uninterrupted.context.ledger.best_x.tolist()


def test_step_rejects_over_budget_and_completed_state() -> None:
    state = _CounterState(_context())

    with pytest.raises(ValueError, match="exceeds the remaining"):
        state.step(19)
    assert state.context.ledger.count == 2

    state.step(18)
    assert state.complete is True
    with pytest.raises(Phase2StateError, match="completed"):
        state.step(1)


def test_snapshot_context_bindings_are_hard_gates() -> None:
    state = _CounterState(_context())
    state.step(3)
    snapshot = state.snapshot()

    wrong_action = ActionContext(
        "ctp",
        state.context.checkpoint,
        state.context.problem,
        EvaluationLedger.from_phase2_snapshot(state.context.problem, snapshot),
        action_seed=17,
    )
    with pytest.raises(Phase2StateError, match="action"):
        validate_snapshot_context(wrong_action, snapshot)

    wrong_seed = ActionContext(
        "aor",
        state.context.checkpoint,
        state.context.problem,
        EvaluationLedger.from_phase2_snapshot(state.context.problem, snapshot),
        action_seed=18,
    )
    with pytest.raises(Phase2StateError, match="seed"):
        validate_snapshot_context(wrong_seed, snapshot)
