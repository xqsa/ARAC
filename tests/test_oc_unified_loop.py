"""Integration tests for CoordinatorState and the unified ARAC-OC loop."""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.contract import OcCoordinatorConfig, OperatorReceipt
from arac.coordination.loop import OC_UNIFIED_MODE, OperatorFailure, run_oc_unified
from arac.coordination.episodes import PhaseAwareSchedulerConfig
from arac.coordination.operators import AorOperator
from arac.coordination.overlap import OverlapStructure
from arac.coordination.state import CoordinatorState
from arac.evidence import run_phase1_overlap_pilot
from arac import run_arac_oc
from arac.runtime.ledger import EvaluationLedger


def _structure() -> OverlapStructure:
    return OverlapStructure(
        dimension=4,
        groups=((0, 1), (1, 2), (2, 3)),
        member_confidences=((1, 0, 0.8), (1, 1, 0.6), (2, 1, 0.7), (2, 2, 0.9)),
    )


def _state(*, checkpoint_hash: str = "", **config_overrides) -> CoordinatorState:
    return CoordinatorState(
        _structure(),
        [(0, 1, 2)],
        config=OcCoordinatorConfig(tau_enter=0.5, tau_exit=0.2, k_enter=2, k_exit=2, **config_overrides),
        checkpoint_hash=checkpoint_hash,
    )


def test_state_ema_initialises_with_first_observation() -> None:
    state = _state()
    state.observe_probes((0, 1, 2), (1,), {1: 0.9})
    assert state.ema_c[1] == pytest.approx(0.9)
    state.observe_probes((0, 1, 2), (1,), {1: 0.1})
    assert state.ema_c[1] == pytest.approx(0.7 * 0.9 + 0.3 * 0.1)


def test_probe_amplitude_does_not_open_dispatch_level() -> None:
    state = _state()
    component = (0, 1, 2)
    state.observe_probes(component, (1,), {1: 1_000.0})
    assert state.level[component] == "low"
    assert state.conflict_streak[component] == 0
    assert state.ema_c[1] == pytest.approx(1_000.0)


def test_proposal_residual_streak_opens_and_closes_dispatch_level() -> None:
    state = _state()
    component = (0, 1, 2)
    state.observe_proposal_conflict(component, high_conflict=True)
    assert state.level[component] == "low"
    assert state.conflict_streak[component] == 1
    state.observe_proposal_conflict(component, high_conflict=True)
    assert state.level[component] == "medium"
    assert state.conflict_streak[component] == 2
    state.observe_proposal_conflict(component, high_conflict=False)
    assert state.level[component] == "low"
    assert state.conflict_streak[component] == 0


def test_state_feedback_updates_credit_pulse_stall_and_deactivation() -> None:
    state = _state(stall_cap=2, pulse_min_fes=8, pulse_max_fes=64)
    component = (0, 1, 2)
    state.update_dispatch(
        component, cycle_index=0, action="ctp_shared_core", gained=True,
        scope=(1, 2), realized_gain=1.0, predicted_gain=2.0,
    )
    assert state.qhat[(1, 0)] == pytest.approx(0.7 * 0.8 + 0.3 * 0.5)
    assert state.pulse_fes[component] == 12          # 8 * 1.5
    assert state.stall[component] == 0
    assert component in state.active_components()
    state.update_dispatch(
        component, cycle_index=5, action="ctp_shared_core", gained=False,
        scope=(1,), realized_gain=0.0, predicted_gain=1.0,
    )
    state.update_dispatch(
        component, cycle_index=9, action="ctp_shared_core", gained=False,
        scope=(1,), realized_gain=0.0, predicted_gain=1.0,
    )
    assert state.stall[component] == 2
    assert component not in state.active_components()
    assert state.pulse_fes[component] == 8           # decayed to the floor


def test_arbitration_only_does_not_update_dispatch_feedback() -> None:
    state = _state()
    component = (0, 1, 2)
    before = state.snapshot()
    state.update_dispatch(
        component,
        cycle_index=0,
        action="arbitration_only",
        gained=False,
        scope=(1,),
        realized_gain=0.0,
        predicted_gain=0.0,
    )
    assert state.snapshot().state_hash == before.state_hash
    assert state.stall[component] == 0
    assert state.cooldown_until[component] == -1
    assert state.pulse_fes[component] == state.config.pulse_min_fes
    assert state.qhat[(1, 0)] == pytest.approx(0.8)


def test_stall_guard_closes_failed_operator_path_but_keeps_sensing() -> None:
    state = _state(stall_cap=2)
    component = (0, 1, 2)
    state.update_dispatch(
        component,
        cycle_index=0,
        action="ctp_shared_core",
        gained=False,
        scope=(1,),
        realized_gain=0.0,
        predicted_gain=1.0,
    )
    state.record_stall_guard(component, cycle_index=2)
    assert state.stall[component] == 2
    assert state.conflict_streak[component] == 0
    assert state.level[component] == "low"
    assert component not in state.active_components()
    assert state.sensing_components() == (component,)


def test_dispatch_deactivation_does_not_disable_sensing() -> None:
    state = _state(stall_cap=2)
    component = (0, 1, 2)
    for cycle_index in (0, 2):
        state.update_dispatch(
            component,
            cycle_index=cycle_index,
            action="ctp_shared_core",
            gained=False,
            scope=(1,),
            realized_gain=0.0,
            predicted_gain=1.0,
        )
    assert component not in state.active_components()
    assert state.sensing_components() == (component,)


def test_state_snapshot_restore_roundtrip() -> None:
    state = _state()
    state.observe_probes((0, 1, 2), (1, 2), {1: 0.9, 2: 0.4})
    snapshot = state.snapshot()
    clone = _state()
    clone.restore(snapshot)
    assert clone.snapshot().state_hash == snapshot.state_hash
    clone.update_dispatch(
        (0, 1, 2), cycle_index=0, action="aor", gained=True,
        scope=(1,), realized_gain=1.0, predicted_gain=1.0,
    )
    assert clone.snapshot().state_hash != snapshot.state_hash
    assert clone.escalation_used[(0, 1, 2)]


def test_state_restore_rejects_checkpoint_config_and_structure_mismatch() -> None:
    state = _state(checkpoint_hash="a" * 64)
    snapshot = state.snapshot()

    wrong_checkpoint = _state(checkpoint_hash="b" * 64)
    with pytest.raises(ValueError, match="checkpoint hash"):
        wrong_checkpoint.restore(snapshot)

    wrong_config = _state(checkpoint_hash="a" * 64, ema_alpha=0.5)
    with pytest.raises(ValueError, match="config hash"):
        wrong_config.restore(snapshot)

    wrong_structure = OverlapStructure(
        dimension=4,
        groups=((0, 1), (1, 2), (2, 3)),
        member_confidences=((1, 0, 0.9), (1, 1, 0.6), (2, 1, 0.7), (2, 2, 0.9)),
    )
    wrong_structure_state = CoordinatorState(
        wrong_structure,
        [(0, 1, 2)],
        config=OcCoordinatorConfig(tau_enter=0.5, tau_exit=0.2, k_enter=2, k_exit=2),
        checkpoint_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="structure hash"):
        wrong_structure_state.restore(snapshot)


def _overlap_problem() -> OptimizationProblem:
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


_PHASE1_KWARGS = {
    "anchors": ((-1.0,) * 4, (1.0,) * 4),
    "step": 0.25,
    "rounds": 8,
    "bucket_size": 2,
    "max_candidate_pairs": 16,
}


def _run(problem, config=None):
    pilot = run_phase1_overlap_pilot(
        problem, total_budget_fes=2_000, run_seed=101, **_PHASE1_KWARGS
    )
    return run_oc_unified(
        problem,
        pilot,
        refresh_cycles=3,
        sense_budget_fes=8,
        config=config,
    )


def test_unified_loop_exact_budget_reproducible_and_receipted() -> None:
    problem = _overlap_problem()
    first = _run(problem)
    second = _run(problem)
    assert first == second
    assert first.coordination_mode == OC_UNIFIED_MODE
    assert first.terminal_fes == 2_000
    assert 0 < first.phase2_consumed_fes <= 2_000
    assert first.final_error > 0.0
    for receipt in first.receipts:
        assert isinstance(receipt, OperatorReceipt)
        assert receipt.actual_fes == receipt.reserved_fes
        assert len(receipt.state_hash) == 64
    for trace in first.cycles:
        assert trace.best_error_after <= trace.best_error_before
        assert len(trace.state_hash) == 64
        assert trace.smp_fes >= 0
    # every cycle's FE accounting reconciles with the ledger progression
    assert sum(
        trace.sense_fes
        + trace.smp_fes
        + trace.probe_fes
        + trace.arbitration_fes
        + trace.operator_fes
        for trace in first.cycles
    ) <= first.phase2_consumed_fes
    assert all(trace.probe_fes % 2 == 0 for trace in first.cycles)


def test_unified_loop_fail_closed_on_operator_exception(monkeypatch) -> None:
    problem = _overlap_problem()

    def explode(self, plan, *, coordinator, proposals=None):
        coordinator.ledger.evaluate(np.full((2, problem.dimension), 0.05))
        raise RuntimeError("primitive exploded")

    monkeypatch.setattr(AorOperator, "execute_plan", explode)
    original_observe = CoordinatorState.observe_proposal_conflict

    def force_aor(self, component, *, high_conflict):
        original_observe(self, component, high_conflict=True)
        self.conflict_streak[tuple(component)] = 6
        self.level[tuple(component)] = "medium"

    monkeypatch.setattr(CoordinatorState, "observe_proposal_conflict", force_aor)
    with pytest.raises(OperatorFailure) as failure:
        _run(problem)
    receipt = failure.value.receipt
    assert receipt.status == "operator_failed"
    assert receipt.exception_name == "RuntimeError"
    assert receipt.actual_fes == 2
    assert receipt.remaining_fes > 0
    assert len(receipt.state_hash) == 64


def test_archive_value_gate_consumes_operator_fes_without_accepting_tiny_move() -> None:
    problem = _overlap_problem()
    ledger = EvaluationLedger(
        problem,
        total_budget=16,
        initial_count=1,
        initial_incumbent=(1.0,) * 4,
        initial_error=4.0,
    )
    snapshot = ledger.archive_snapshot()
    ledger.evaluate((0.1,) * 4)
    assert ledger.best_error < snapshot.error
    ledger.restore_archive(snapshot)
    assert ledger.best_error == snapshot.error
    assert np.array_equal(ledger.best_x, snapshot.incumbent)
    assert ledger.count == 2  # the consumed FE is not refunded


def test_canonical_arac_oc_entrypoint_uses_unified_loop() -> None:
    result = run_arac_oc(
        _overlap_problem(),
        total_budget_fes=2_000,
        run_seed=101,
        refresh_cycles=1,
        sense_budget_fes=8,
        phase1_kwargs=_PHASE1_KWARGS,
    )
    assert result.coordination_mode == OC_UNIFIED_MODE
    assert result.phase1.adaptation.ready
    assert result.terminal_fes == 2_000


def test_canonical_arac_oc_entrypoint_has_explicit_v5_1_mode() -> None:
    config = PhaseAwareSchedulerConfig(
        maturity_window_fes=200,
        revelation_horizon_fes=800,
        exploration_and_development_cap=0.8,
        exploitation_reserve_ratio=0.05,
        cold_start_probe_cap=0.25,
        probe_min_fes=100,
        segment_fes=400,
        calibration_ref="unit",
    )
    result = run_arac_oc(
        _overlap_problem(),
        total_budget_fes=2_000,
        run_seed=101,
        scheduler_mode="v5_1",
        scheduler_config=config,
        phase1_kwargs=_PHASE1_KWARGS,
    )
    assert result.scheduler_version == "v5.1"
    assert result.coordinator_name == "GCB coordinator"
    assert result.episode_names["gcb"] == "gss"
    assert result.terminal_fes == 2_000


def test_unified_loop_enters_tail_when_current_sense_window_is_unaffordable() -> None:
    problem = _overlap_problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=1_000,
        run_seed=101,
        **_PHASE1_KWARGS,
    )
    result = run_oc_unified(
        problem,
        pilot,
        refresh_cycles=3,
        sense_budget_fes=512,
    )
    assert result.cycles == ()
    assert result.tail_fes == 760
    assert result.terminal_fes == 1_000
