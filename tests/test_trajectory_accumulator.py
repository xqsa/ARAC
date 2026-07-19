from __future__ import annotations

import pytest

from arac.evidence.overlap_relation_builder import OverlapRelation
from arac.evidence.trajectory_accumulator import TrajectoryAccumulator


def _relation(**overrides: object) -> OverlapRelation:
    values = {
        "relation_id": "O0_0_1",
        "problem_id": "E3",
        "outer_iter": 0,
        "group_left": 0,
        "group_right": 1,
        "shared_vars": (2,),
        "overlap_strength": 1.0,
        "delta_signal": 1.0,
        "rank_signal": 1.0,
        "budget_remaining_ratio": 1.0,
        "previous_delta": 2.0,
        "current_delta": 1.0,
        "delta_signed_gap": -1.0,
        "delta_ratio_gap": 0.5,
    }
    values.update(overrides)
    return OverlapRelation(**values)


def test_trajectory_accumulator_tracks_momentum_and_conflict_trend() -> None:
    accumulator = TrajectoryAccumulator(window=4)

    first = accumulator.update(_relation(delta_signed_gap=-1.0, delta_ratio_gap=0.2))
    second = accumulator.update(_relation(delta_signed_gap=-2.0, delta_ratio_gap=0.4))
    reversed_direction = accumulator.update(
        _relation(delta_signed_gap=1.0, delta_ratio_gap=0.1)
    )

    assert first.delta_momentum == 0.0
    assert second.delta_momentum == 1.0
    assert second.conflict_trend > 0.0
    assert reversed_direction.delta_momentum == -1.0
    assert reversed_direction.conflict_trend < 0.0


def test_trajectory_accumulator_uses_consecutive_zero_gain_tail() -> None:
    accumulator = TrajectoryAccumulator(window=4)
    accumulator.update(_relation())
    accumulator.update(
        _relation(previous_delta=0.0, current_delta=0.0, delta_signed_gap=0.0)
    )
    features = accumulator.update(
        _relation(previous_delta=0.0, current_delta=0.0, delta_signed_gap=0.0)
    )

    assert features.stagnation_score == 0.5


def test_trajectory_accumulator_separates_structural_relations() -> None:
    accumulator = TrajectoryAccumulator(window=4)
    accumulator.update(_relation(delta_signed_gap=-1.0))

    other = accumulator.update(
        _relation(
            relation_id="O0_1_2",
            group_left=1,
            group_right=2,
            shared_vars=(3,),
            delta_signed_gap=-1.0,
        )
    )

    assert other.delta_momentum == 0.0


def test_trajectory_accumulator_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TrajectoryAccumulator(window=0)
