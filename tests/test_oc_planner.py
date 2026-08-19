"""Unit tests for the OC dispatch planner (complete OperatorPlans)."""

from __future__ import annotations

import pytest

from arac.coordination.contract import (
    OC_ACTION_AOR,
    OC_ACTION_ARBITRATION,
    OC_ACTION_CTP_RESTRICTED,
    OC_ACTION_CTP_SHARED_CORE,
    OC_ACTION_SMP,
    OcCoordinatorConfig,
)
from arac.coordination.overlap import OverlapStructure
from arac.coordination.planner import ComponentSignal, OcDispatchPlanner

CHAIN = (0, 1, 2, 3)   # hub 2, relative 2/3 -> not complex in relative mode
STAR = (4, 5, 6, 7)    # hub 3, relative 1.0 -> complex topology


def _structure() -> OverlapStructure:
    return OverlapStructure(
        dimension=14,
        groups=(
            (0, 1, 2),      # g0
            (2, 3, 4),      # g1
            (4, 5, 6),      # g2
            (6, 7),         # g3
            (8, 9, 10),     # g4 star centre
            (10, 11),       # g5
            (9, 12),        # g6
            (8, 13),        # g7
        ),
    )


def _planner(**config_overrides) -> OcDispatchPlanner:
    return OcDispatchPlanner(
        _structure(),
        [CHAIN, STAR],
        config=OcCoordinatorConfig(hub_mode="relative", **config_overrides),
        base_seed=20260850,
    )


def _signal(component=CHAIN, **overrides) -> ComponentSignal:
    base = dict(component=component, level="high", pulse_fes=16, qhat_mean=1.0)
    base.update(overrides)
    return ComponentSignal(**base)


def test_hub_topology_matches_gate38_definitions() -> None:
    planner = _planner()
    assert planner.hub_degree(CHAIN) == 2
    assert planner.relative_hub(CHAIN) == pytest.approx(2 / 3)
    assert planner.hub_degree(STAR) == 3
    assert planner.relative_hub(STAR) == pytest.approx(1.0)
    assert planner.shared_scope_variables(CHAIN) == (2, 4, 6)
    assert planner.shared_scope_variables(STAR) == (8, 9, 10)


def test_select_scope_ranks_by_ema_and_shrinks_to_budget() -> None:
    planner = _planner()
    ema_c = {2: 0.9, 4: 0.1, 6: 0.5}
    assert planner.select_scope(CHAIN, ema_c, probe_budget_fes=6) == (2, 6, 4)
    assert planner.select_scope(CHAIN, ema_c, probe_budget_fes=4) == (2, 6)
    assert planner.select_scope(CHAIN, ema_c, probe_budget_fes=2) == (2,)
    assert planner.select_scope(CHAIN, ema_c, probe_budget_fes=0) == ()
    tie = {2: 0.5, 4: 0.5, 6: 0.5}
    assert planner.select_scope(CHAIN, tie, probe_budget_fes=2) == (2,)


def test_prioritize_filters_and_orders() -> None:
    planner = _planner()
    signals = [
        _signal(component=CHAIN, max_c=0.4),
        _signal(component=STAR, max_c=0.9),
        _signal(component=CHAIN, level="low", max_c=99.0, in_cooldown=True),
        _signal(component=STAR, level="low", max_c=99.0, active=False),
    ]
    ranked = planner.prioritize(signals)
    assert [signal.component for signal in ranked] == [STAR, CHAIN]
    persistence = _signal(component=CHAIN, max_c=0.8, conflict_streak=4)
    assert planner.prioritize([persistence, _signal(component=STAR, max_c=0.9)])[0].component == CHAIN


def _make(planner, signal, **kwargs):
    defaults = dict(cycle_index=5, scope=(2, 4), probe_widths={2: 1.0, 4: 3.0}, available_fes=64)
    defaults.update(kwargs)
    return planner.make_plan(signal, **defaults)


def test_low_level_plans_arbitration_only() -> None:
    planner = _planner()
    plan = _make(planner, _signal(level="low"))
    assert plan.action == OC_ACTION_ARBITRATION
    assert plan.reserved_fes == 0
    assert plan.predicted_gain == 0.0
    assert plan.conflict_level == "low"


def test_medium_level_plans_restricted_ctp_with_pulse_budget() -> None:
    planner = _planner()
    plan = _make(planner, _signal(level="medium", pulse_fes=12), available_fes=8)
    assert plan.action == OC_ACTION_CTP_RESTRICTED
    assert plan.reserved_fes == 8
    assert plan.predicted_gain == pytest.approx(2.0)
    assert _make(planner, _signal(level="medium", pulse_fes=12), available_fes=64).reserved_fes == 12


def test_restricted_ctp_requires_a_real_episode_window() -> None:
    planner = _planner(operator_episode_min_fes=32)
    plan = _make(
        planner,
        _signal(level="medium", pulse_fes=8),
        available_fes=64,
    )
    assert plan.action == OC_ACTION_CTP_RESTRICTED
    assert plan.reserved_fes == 32


def test_restricted_ctp_shortfall_is_explicit_arbitration() -> None:
    planner = _planner(operator_episode_min_fes=32)
    plan = _make(
        planner,
        _signal(level="medium", pulse_fes=8),
        available_fes=16,
    )
    assert plan.action == OC_ACTION_ARBITRATION
    assert plan.reason == "operator_episode_budget_unavailable"
    assert plan.reserved_fes == 0


def test_high_level_shared_core_and_smp_trust_rebuild() -> None:
    planner = _planner()
    plan = _make(
        planner,
        _signal(component=STAR, level="medium", conflict_streak=2),
        scope=(8, 9),
        probe_widths={8: 1.0, 9: 1.0},
    )
    assert plan.action == OC_ACTION_CTP_SHARED_CORE
    assert plan.conflict_level == "high"
    rebuild = _make(planner, _signal(level="medium", conflict_streak=2, qhat_mean=0.3))
    assert rebuild.action == OC_ACTION_SMP
    assert rebuild.reason == "high_smp_trust_rebuild"


def test_shared_core_requires_a_two_fe_window_per_shared_variable() -> None:
    planner = _planner()
    plan = _make(
        planner,
        _signal(component=STAR, level="medium", conflict_streak=2, pulse_fes=4),
        scope=(8, 9, 10),
        probe_widths={8: 1.0, 9: 1.0, 10: 1.0},
    )
    assert plan.action == OC_ACTION_ARBITRATION
    assert plan.reason == "shared_core_budget_unavailable"
    assert plan.reserved_fes == 0


def test_persistent_chain_conflict_uses_restricted_ctp() -> None:
    planner = _planner()
    plan = _make(planner, _signal(level="medium", conflict_streak=2))
    assert plan.action == OC_ACTION_CTP_RESTRICTED
    assert plan.conflict_level == "medium"


def test_stalled_dispatch_keeps_arbitration_alive_without_repeating_repair() -> None:
    planner = _planner()
    plan = _make(
        planner,
        _signal(level="medium", conflict_streak=2, stall=1),
    )
    assert plan.action == OC_ACTION_ARBITRATION
    assert plan.reason == "stall_guard_arbitration"


def test_high_probe_amplitude_without_residual_persistence_stays_arbitration() -> None:
    planner = _planner()
    plan = _make(planner, _signal(level="low", max_c=100.0, mean_c=50.0))
    assert plan.action == OC_ACTION_ARBITRATION


def test_material_arbitration_gain_suppresses_same_cycle_operator_pulse() -> None:
    planner = _planner(arbitration_value_ratio=0.01)
    plan = _make(
        planner,
        _signal(level="medium", conflict_streak=2),
        arbitration_gain=2.0,
        arbitration_reference_error=100.0,
    )
    assert plan.action == OC_ACTION_ARBITRATION
    assert plan.reason == "arbitration_value_gate"
    assert plan.reserved_fes == 0


def test_small_arbitration_gain_does_not_suppress_operator_pulse() -> None:
    planner = _planner(arbitration_value_ratio=0.01)
    plan = _make(
        planner,
        _signal(level="medium", conflict_streak=2),
        arbitration_gain=0.5,
        arbitration_reference_error=100.0,
    )
    assert plan.action == OC_ACTION_CTP_RESTRICTED


def test_smp_budget_shortfall_is_explicitly_rerouted() -> None:
    planner = _planner()
    plan = _make(
        planner,
        _signal(level="high", qhat_mean=0.3),
        available_fes=2,
    )
    assert plan.action == OC_ACTION_CTP_SHARED_CORE
    assert plan.reason == "smp_budget_unavailable_ctp"


def test_smp_reservation_covers_each_owner_session_minimum() -> None:
    planner = _planner()
    plan = _make(
        planner,
        _signal(level="high", qhat_mean=0.3),
        available_fes=24,
    )
    assert plan.action == OC_ACTION_SMP
    assert plan.reserved_fes >= 24


def test_aor_requires_two_fes_for_mmes_population() -> None:
    planner = _planner()
    with pytest.raises(ValueError, match="aor requires at least 2 FE"):
        _make(
            planner,
            _signal(component=STAR, level="high"),
            scope=(8,),
            probe_widths={8: 1.0},
            available_fes=1,
        )


def test_complex_topology_and_escalation_route_to_aor() -> None:
    planner = _planner()
    star = _make(planner, _signal(component=STAR, level="high"), scope=(8, 9), probe_widths={8: 1.0, 9: 1.0})
    assert star.action == OC_ACTION_AOR
    assert star.conflict_level == "complex"
    assert star.reason == "complex_topology_aor"
    escalated = _make(planner, _signal(component=CHAIN, level="high", conflict_streak=6))
    assert escalated.action == OC_ACTION_AOR
    assert escalated.reason == "persistent_escalation_aor"
    spent = _make(planner, _signal(component=CHAIN, level="high", conflict_streak=6, escalation_used=True))
    assert spent.action == OC_ACTION_CTP_SHARED_CORE


def test_make_plan_validates_scope_and_budget_inputs() -> None:
    planner = _planner()
    with pytest.raises(ValueError, match="subset of the component"):
        _make(planner, _signal(), scope=(0, 13))
    with pytest.raises(ValueError, match="non-empty scope"):
        _make(planner, _signal(), scope=())
    with pytest.raises(ValueError, match="dispatch pool exhausted"):
        _make(planner, _signal(), available_fes=0)
    with pytest.raises(ValueError, match="unknown component"):
        _make(planner, _signal(component=(9, 10)))


def test_plan_hash_is_deterministic_and_cycle_sensitive() -> None:
    planner = _planner()
    first = _make(planner, _signal(level="medium"))
    same = _make(planner, _signal(level="medium"))
    assert first.plan_hash == same.plan_hash
    assert _make(planner, _signal(level="medium"), cycle_index=6).plan_hash != first.plan_hash
