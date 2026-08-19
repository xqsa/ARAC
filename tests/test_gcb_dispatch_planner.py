"""Unit tests for the pre-registered GCB dispatch planner."""

from __future__ import annotations

import pytest

from arac.coordination import (
    DISPATCH_COORDINATE_CTP,
    DISPATCH_JOINT_CMAES,
    DISPATCH_JOINT_CTP,
    DISPATCH_NEIGHBORHOOD,
    GcbDispatchConfig,
    GcbDispatchPlanner,
    OverlapStructure,
)


def _chain_structure() -> OverlapStructure:
    return OverlapStructure(dimension=5, groups=((0, 1), (1, 2), (2, 3), (3, 4)))


def _star_structure() -> OverlapStructure:
    return OverlapStructure(dimension=5, groups=((0, 1), (0, 2), (0, 3), (0, 4)))


def _planner(structure: OverlapStructure, component: tuple[int, ...], **config) -> GcbDispatchPlanner:
    return GcbDispatchPlanner(
        structure,
        (component,),
        envelope_fes=32,
        config=GcbDispatchConfig(**config),
    )


def test_hub_degree_separates_star_from_chain() -> None:
    chain = _planner(_chain_structure(), (0, 1, 2, 3))
    star = _planner(_star_structure(), (0, 1, 2, 3))

    assert chain.hub_degree((0, 1, 2, 3)) == 2
    assert star.hub_degree((0, 1, 2, 3)) == 3


def test_not_persistent_keeps_neighborhood_envelope() -> None:
    planner = _planner(_chain_structure(), (0, 1, 2, 3))

    plan = planner.plan((0, 1, 2, 3), cycle_index=0, conflict_streak=1)

    assert plan.action == DISPATCH_NEIGHBORHOOD
    assert plan.reason == "not_persistent"
    assert plan.reserved_fes == 0


def test_persistent_pairwise_topology_dispatches_coordinate_ctp() -> None:
    planner = _planner(_chain_structure(), (0, 1, 2, 3))

    plan = planner.plan((0, 1, 2, 3), cycle_index=0, conflict_streak=2)

    assert plan.action == DISPATCH_COORDINATE_CTP
    assert plan.reason == "pairwise_topology"
    assert plan.reserved_fes == 32


def test_persistent_complex_topology_dispatches_joint_ctp() -> None:
    planner = _planner(_star_structure(), (0, 1, 2, 3))

    plan = planner.plan((0, 1, 2, 3), cycle_index=0, conflict_streak=2)

    assert plan.action == DISPATCH_JOINT_CTP
    assert plan.reason == "complex_topology"


def test_escalation_streak_dispatches_joint_cmaes_once() -> None:
    planner = _planner(_chain_structure(), (0, 1, 2, 3))

    plan = planner.plan((0, 1, 2, 3), cycle_index=0, conflict_streak=6)

    assert plan.action == DISPATCH_JOINT_CMAES
    assert plan.reason == "persistent_escalation"
    planner.record_outcome((0, 1, 2, 3), cycle_index=0, action=DISPATCH_JOINT_CMAES, gained=False)

    later = planner.plan((0, 1, 2, 3), cycle_index=5, conflict_streak=8)

    assert later.action == DISPATCH_COORDINATE_CTP
    assert later.escalation_used is True


def test_cooldown_blocks_the_cycle_after_a_dispatch() -> None:
    planner = _planner(_chain_structure(), (0, 1, 2, 3))

    planner.record_outcome(
        (0, 1, 2, 3), cycle_index=2, action=DISPATCH_COORDINATE_CTP, gained=True
    )

    blocked = planner.plan((0, 1, 2, 3), cycle_index=3, conflict_streak=5)
    released = planner.plan((0, 1, 2, 3), cycle_index=4, conflict_streak=5)

    assert blocked.action == DISPATCH_NEIGHBORHOOD
    assert blocked.reason == "cooldown"
    assert released.action == DISPATCH_COORDINATE_CTP


def test_gain_resets_stall_and_keeps_dispatch_enabled() -> None:
    planner = _planner(_chain_structure(), (0, 1, 2, 3))

    planner.record_outcome(
        (0, 1, 2, 3), cycle_index=0, action=DISPATCH_COORDINATE_CTP, gained=False
    )
    planner.record_outcome(
        (0, 1, 2, 3), cycle_index=2, action=DISPATCH_COORDINATE_CTP, gained=True
    )

    plan = planner.plan((0, 1, 2, 3), cycle_index=4, conflict_streak=5)

    assert plan.stall_count == 0
    assert plan.action == DISPATCH_COORDINATE_CTP


def test_stall_cap_disables_the_component_permanently() -> None:
    planner = _planner(_chain_structure(), (0, 1, 2, 3))

    planner.record_outcome(
        (0, 1, 2, 3), cycle_index=0, action=DISPATCH_COORDINATE_CTP, gained=False
    )
    planner.record_outcome(
        (0, 1, 2, 3), cycle_index=2, action=DISPATCH_COORDINATE_CTP, gained=False
    )

    for cycle_index in (4, 6, 8):
        plan = planner.plan((0, 1, 2, 3), cycle_index=cycle_index, conflict_streak=9)
        assert plan.action == DISPATCH_NEIGHBORHOOD
        assert plan.reason == "stalled_out"


def test_record_outcome_rejects_neighborhood_action() -> None:
    planner = _planner(_chain_structure(), (0, 1, 2, 3))

    with pytest.raises(ValueError, match="dispatch action"):
        planner.record_outcome(
            (0, 1, 2, 3), cycle_index=0, action=DISPATCH_NEIGHBORHOOD, gained=False
        )


def test_duplicate_components_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        GcbDispatchPlanner(
            _chain_structure(),
            ((0, 1, 2, 3), (0, 1, 2, 3)),
            envelope_fes=32,
        )


def test_relative_hub_mode_separates_star_from_chain() -> None:
    chain = GcbDispatchPlanner(
        _chain_structure(),
        ((0, 1, 2, 3),),
        envelope_fes=32,
        config=GcbDispatchConfig(hub_mode="relative", complex_hub_ratio=0.9),
    )
    star = GcbDispatchPlanner(
        _star_structure(),
        ((0, 1, 2, 3),),
        envelope_fes=32,
        config=GcbDispatchConfig(hub_mode="relative", complex_hub_ratio=0.9),
    )

    assert chain.relative_hub((0, 1, 2, 3)) == pytest.approx(2 / 3)
    assert star.relative_hub((0, 1, 2, 3)) == 1.0
    assert (
        chain.plan((0, 1, 2, 3), cycle_index=0, conflict_streak=2).action
        == DISPATCH_COORDINATE_CTP
    )
    assert (
        star.plan((0, 1, 2, 3), cycle_index=0, conflict_streak=2).action
        == DISPATCH_JOINT_CTP
    )


def test_invalid_hub_mode_and_ratio_fail_closed() -> None:
    with pytest.raises(ValueError, match="hub_mode"):
        GcbDispatchConfig(hub_mode="density")
    with pytest.raises(ValueError, match="complex_hub_ratio"):
        GcbDispatchConfig(complex_hub_ratio=1.5)
    with pytest.raises(ValueError, match="complex_hub_ratio"):
        GcbDispatchConfig(complex_hub_ratio=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"persistent_streak": 0},
        {"escalation_streak": 1},
        {"stall_cap": 0},
        {"cooldown_cycles": 0},
        {"complex_hub_degree": 1},
    ],
)
def test_invalid_configs_fail_closed(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        GcbDispatchConfig(**kwargs)
