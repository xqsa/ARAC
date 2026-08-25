from __future__ import annotations

import pytest

from arac.coordination.soft_routing import compute_activity, rank_normalize


def test_rank_normalization_is_bounded_and_deterministic_with_ties() -> None:
    assert rank_normalize((3.0, 1.0, 1.0, 5.0)) == (2 / 3, 1 / 6, 1 / 6, 1.0)


def test_activity_is_continuous_and_orders_patch_scope_only() -> None:
    decision = compute_activity((0.0, 10.0, 5.0), (5.0, 0.0, 2.0))
    assert all(0.0 <= value <= 1.0 for value in decision.activity)
    assert decision.scope_order[0] == 1
    assert decision.candidate_strength == decision.activity
    assert decision.radius_upper_bound == decision.activity


@pytest.mark.parametrize("values", ((), (float("nan"),), (float("inf"),)))
def test_rank_normalization_rejects_invalid_signal(values) -> None:
    with pytest.raises(ValueError):
        rank_normalize(values)


def test_activity_rejects_mismatched_width() -> None:
    with pytest.raises(ValueError, match="widths"):
        compute_activity((1.0, 2.0), (1.0,))
