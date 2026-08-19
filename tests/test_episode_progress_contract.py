"""Progress-contract targeted tests (v4 upgrade plan section 3).

The four schedulable episode states expose ``progress()`` as their single
source of phase semantics; the scheduler must never keep a second
name-keyed phase map (CTP's coverage boundary, GSS's warmup sweep, SMP's
visit alignment, AOR's correction window).
"""

from __future__ import annotations

import numpy as np
import pytest

from arac.actions.ctp import CtpExecutor
from arac.actions.gcb import GcbExecutor
from arac.actions.recovered import RecoveredAorExecutor, RecoveredSmpExecutor
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint
from arac.runtime.phase2 import EpisodeProgress, Phase2StateError

DIMENSION = 24
BLOCKS = tuple(tuple(range(start, start + 6)) for start in range(0, DIMENSION, 6))
WINDOW = 400


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


def _context(total: int = 8_000, phase1: int = 200) -> tuple[OptimizationProblem, ActionContext]:
    problem = _problem()
    incumbent = tuple(0.5 for _ in range(DIMENSION))
    checkpoint = PhaseCheckpoint(
        protocol="progress-unit",
        run_seed=3,
        total_budget_fes=total,
        phase1_fes=phase1,
        incumbent=incumbent,
        incumbent_error=float(problem.objective(np.asarray(incumbent))),
        feature_names=("log10_center_error", "line_high_frequency_fraction_median"),
        feature_values=(1.0, 0.4),
        blocks=BLOCKS,
        relations=(),
    )
    from arac.runtime.ledger import EvaluationLedger

    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=total,
        phase1_fes=phase1,
        incumbent=incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=True,
    )
    context = ActionContext("ctp", checkpoint, problem, ledger, action_seed=20260853)
    return problem, context


def _state(action: str):
    problem, context = _context()
    context = ActionContext(
        action,
        context.checkpoint,
        problem,
        context.ledger,
        action_seed=context.action_seed,
    )
    executors = {
        "ctp": CtpExecutor,
        "gcb": GcbExecutor,
        "smp": RecoveredSmpExecutor,
        "aor": RecoveredAorExecutor,
    }
    return executors[action]().initialize(context)


REQUIRED_FIELDS = (
    "episode",
    "phase",
    "consumed_fes",
    "next_boundary_fes",
    "min_step_fes",
    "maturity_target_fes",
    "protocol_mature",
    "contract",
)


def test_all_four_states_expose_the_contract() -> None:
    for action in ("ctp", "gcb", "smp", "aor"):
        state = _state(action)
        progress = state.progress(maturity_window_fes=WINDOW)
        assert isinstance(progress, EpisodeProgress)
        for name in REQUIRED_FIELDS:
            assert hasattr(progress, name), f"{action}.{name} missing"
        assert progress.episode == action
        assert progress.consumed_fes == 0
        assert progress.protocol_mature is False
        assert progress.contract.endswith("-v1")
    with pytest.raises(ValueError):
        _state("ctp").progress(maturity_window_fes=0)


def test_ctp_phase_transitions_and_coverage_boundary() -> None:
    state = _state("ctp")
    start = state.progress(maturity_window_fes=WINDOW)
    assert start.phase == "coverage"
    assert start.min_step_fes == 1
    # Step exactly to the coverage boundary: the regime is complete but
    # still NOT protocol-mature (the polish window must run -- the v4
    # probe-ending-at-coverage bug this contract exists to kill).
    coverage = start.maturity_target_fes - WINDOW
    state.step(coverage)
    boundary = state.progress(maturity_window_fes=WINDOW)
    assert boundary.phase in ("coverage", "polish")
    assert boundary.protocol_mature is False
    assert boundary.maturity_target_fes == start.maturity_target_fes
    state.step(WINDOW)
    mature = state.progress(maturity_window_fes=WINDOW)
    assert mature.phase == "polish"
    assert mature.protocol_mature is True


def test_gss_warmup_sweep_before_maturity() -> None:
    state = _state("gcb")
    start = state.progress(maturity_window_fes=WINDOW)
    assert start.phase == "warmup"
    warmup = start.maturity_target_fes - WINDOW
    state.step(warmup)
    mid = state.progress(maturity_window_fes=WINDOW)
    assert mid.phase != "warmup"
    assert mid.protocol_mature is False
    state.step(WINDOW)
    mature = state.progress(maturity_window_fes=WINDOW)
    assert mature.protocol_mature is True


def test_smp_generation_alignment_and_visit_maturity() -> None:
    state = _state("smp")
    start = state.progress(maturity_window_fes=WINDOW)
    assert start.min_step_fes > 1  # generation-aligned, never split a visit
    assert start.phase in ("block_sweep", "visit")
    step = state.step(start.min_step_fes)
    assert step.step_fes > 0
    after = state.progress(maturity_window_fes=WINDOW)
    assert after.phase in ("block_sweep", "visit", "drain")
    # A below-unit request is refused loudly, not silently split.
    with pytest.raises((Phase2StateError, ValueError)):
        state.step(1)


def test_aor_single_regime_window_maturity() -> None:
    state = _state("aor")
    start = state.progress(maturity_window_fes=WINDOW)
    assert start.phase == "global_correction"
    assert start.min_step_fes == 1
    assert start.maturity_target_fes == WINDOW
    state.step(WINDOW)
    mature = state.progress(maturity_window_fes=WINDOW)
    assert mature.protocol_mature is True


def test_maturity_window_clamps_to_the_action_budget() -> None:
    state = _state("aor")
    progress = state.progress(maturity_window_fes=10_000)
    assert progress.maturity_target_fes == 8_000 - 200
