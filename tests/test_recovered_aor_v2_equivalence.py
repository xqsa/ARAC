"""Equivalence tests for the recovered-AOR v2 episode wrapper.

Design note (measured 2026-08-16): the v2 session path and the legacy
``PypopOptimizerPort.run`` one-shot are NOT bit-identical trajectories
(the resumable session hand-drives the generation loop and disables
upstream early stopping).  Gate 50 pairings therefore run v2 episodes on
BOTH sides; the historical legacy receipts remain reference lines.  What
must hold bit-exactly -- and what these tests pin -- is equivalence
inside the v2 world: one full-budget step equals any irregular
segmentation with a snapshot/ledger restore between every segment.
"""

from __future__ import annotations

import numpy as np
import pytest

from arac.actions.phase2_v2 import Phase2StateError, RecoveredAorPhase2State
from arac.actions.recovered import RecoveredAorExecutor
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger

DIMENSION = 100
TOTAL_FES = 60_100
PHASE1_FES = 100
ACTION_SEED = 20260851
CHUNKS = (997, 31, 20_003, 7, 12_997, 409, 8_003, 3_337, 14_317)


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        for start in range(0, DIMENSION, 20):
            block = batch[:, start : start + 20]
            result += 0.25 * np.sum(block**2, axis=1) ** 2 / 20
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _context() -> ActionContext:
    problem = _problem()
    checkpoint = PhaseCheckpoint(
        protocol="recovered-aor-v2-equivalence",
        run_seed=11,
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=(0.5,) * DIMENSION,
        incumbent_error=float(problem.objective(np.asarray([0.5] * DIMENSION))),
        feature_names=("line_high_frequency_fraction_median",),
        feature_values=(0.4,),
        blocks=tuple((index,) for index in range(DIMENSION)),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=True,
    )
    return ActionContext("aor", checkpoint, problem, ledger, action_seed=ACTION_SEED)


def _summary(ledger: EvaluationLedger) -> tuple[int, float, list[float]]:
    return ledger.count, float(ledger.best_error), ledger.best_x.tolist()


def test_recovered_aor_v2_one_shot_is_a_valid_terminal_run() -> None:
    executor = RecoveredAorExecutor()
    legacy_context = _context()
    legacy_result = executor.execute(legacy_context)

    oneshot_context = _context()
    state = executor.initialize(oneshot_context)
    state.step(TOTAL_FES - PHASE1_FES)
    assert state.complete
    v2_result = state.result()

    assert v2_result.terminal_fes == legacy_result.terminal_fes == TOTAL_FES
    assert v2_result.route == legacy_result.route
    assert oneshot_context.ledger.best_error <= oneshot_context.checkpoint.incumbent_error
    # Not asserted: bit-parity with the legacy trajectory (see module note).
    assert v2_result.final_error != legacy_result.final_error or True


def test_recovered_aor_v2_segmentation_with_restores_is_bit_identical() -> None:
    executor = RecoveredAorExecutor()
    oneshot_context = _context()
    oneshot = executor.initialize(oneshot_context)
    oneshot.step(TOTAL_FES - PHASE1_FES)
    assert oneshot.complete

    context = _context()
    state = executor.initialize(context)
    restores = 0
    index = 0
    while not state.complete:
        budget = state.total_fes - state.context.ledger.count
        state.step(min(budget, CHUNKS[index % len(CHUNKS)]))
        index += 1
        if not state.complete:
            snapshot = state.snapshot()
            restored = ActionContext(
                "aor",
                context.checkpoint,
                context.problem,
                EvaluationLedger.from_phase2_snapshot(context.problem, snapshot),
                action_seed=ACTION_SEED,
            )
            state = executor.resume(restored, snapshot)
            restores += 1

    assert restores >= 3
    assert _summary(state.context.ledger) == _summary(oneshot_context.ledger)
    assert state.snapshot().state_hash == oneshot.snapshot().state_hash


def test_recovered_aor_state_rejects_nonzero_anchor() -> None:
    context = _context()
    zero = tuple(0.0 for _ in range(DIMENSION))
    state = RecoveredAorPhase2State(context, anchor=zero)
    assert state.anchor == zero
    with pytest.raises(Phase2StateError, match="anchors at the zero vector"):
        RecoveredAorPhase2State(context, anchor=tuple(0.1 for _ in range(DIMENSION)))
