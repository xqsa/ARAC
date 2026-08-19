"""Equivalence tests for the recovered-SMP v2 episode wrapper.

The recovered SMP loop is the one place where the v2 wrapper can be
bit-identical to the legacy executor: both drive the same hand-written
generation loop, and the state machine never splits a generation (vendor
objectives are batch-shape sensitive at the last bit).  Pinned here:

1. one full-budget v2 step reproduces the legacy ``execute`` bit-exactly,
   including the route string;
2. arbitrary segmentation with snapshot/ledger restores between every
   step is bit-identical to the one-shot run and reports aligned
   consumption honestly (``step_fes`` may be smaller than requested).
"""

from __future__ import annotations

import numpy as np

from arac.actions.recovered import RecoveredSmpExecutor
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger

DIMENSION = 96
BLOCKS = tuple(tuple(range(start, start + 16)) for start in range(0, DIMENSION, 16))
TOTAL_FES = 40_100
PHASE1_FES = 100
ACTION_SEED = 20260852
CHUNKS = (997, 31, 2_003, 7, 12_997, 409, 8_003, 3, 14_317, 61)


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        for block in BLOCKS:
            inner = batch[:, list(block)]
            result += 0.25 * np.sum(inner**2, axis=1) ** 2 / len(block)
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
        protocol="recovered-smp-v2-equivalence",
        run_seed=13,
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=(0.5,) * DIMENSION,
        incumbent_error=float(problem.objective(np.asarray([0.5] * DIMENSION))),
        feature_names=("line_high_frequency_fraction_median",),
        feature_values=(0.4,),
        blocks=BLOCKS,
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=True,
    )
    return ActionContext("smp", checkpoint, problem, ledger, action_seed=ACTION_SEED)


def _summary(ledger: EvaluationLedger) -> tuple[int, float, list[float]]:
    return ledger.count, float(ledger.best_error), ledger.best_x.tolist()


def test_recovered_smp_v2_one_shot_matches_legacy_bitexactly() -> None:
    executor = RecoveredSmpExecutor()
    legacy_context = _context()
    legacy_result = executor.execute(legacy_context)

    oneshot_context = _context()
    state = executor.initialize(oneshot_context)
    step = state.step(TOTAL_FES - PHASE1_FES)
    assert state.complete
    v2_result = state.result()

    assert step.step_fes == TOTAL_FES - PHASE1_FES
    assert _summary(oneshot_context.ledger) == _summary(legacy_context.ledger)
    assert v2_result.final_error == legacy_result.final_error
    assert v2_result.route == legacy_result.route


def test_recovered_smp_v2_arbitrary_segmentation_is_bit_identical() -> None:
    executor = RecoveredSmpExecutor()
    oneshot_context = _context()
    oneshot = executor.initialize(oneshot_context)
    oneshot.step(TOTAL_FES - PHASE1_FES)

    context = _context()
    state = executor.initialize(context)
    restores = 0
    partial_steps = 0
    index = 0
    from arac.runtime.phase2 import Phase2StateError

    while not state.complete:
        remaining = state.total_fes - state.context.ledger.count
        request = min(remaining, CHUNKS[index % len(CHUNKS)])
        try:
            step = state.step(request)
        except Phase2StateError:
            # Sub-unit budgets are refused by contract; a scheduler retries
            # with a larger aligned request.
            request = min(remaining, max(request * 4, 64))
            step = state.step(request)
        index += 1
        if step.step_fes < request:
            partial_steps += 1
        if not state.complete:
            snapshot = state.snapshot()
            restored = ActionContext(
                "smp",
                context.checkpoint,
                context.problem,
                EvaluationLedger.from_phase2_snapshot(
                context.problem, snapshot, allow_out_of_bounds=True
            ),
                action_seed=ACTION_SEED,
            )
            state = executor.resume(restored, snapshot)
            restores += 1

    assert restores >= 3
    assert partial_steps >= 1  # aligned consumption honestly reported
    assert _summary(state.context.ledger) == _summary(oneshot_context.ledger)
    assert state.route == oneshot.route
    assert state.snapshot().state_hash == oneshot.snapshot().state_hash
