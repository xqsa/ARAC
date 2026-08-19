from __future__ import annotations

import pytest

from arac.overlap_core import capped_proposal_budget


def test_capped_proposal_budget_respects_total_sense_share() -> None:
    components = ((0, 1, 2, 3), (4, 5))
    phase2_fes = 100_000
    cycles = 5
    budget = capped_proposal_budget(
        phase2_fes,
        components,
        refresh_cycles=cycles,
        neighborhood_fes=32,
        sense_budget_share=0.4,
    )

    total_sense = cycles * sum(map(len, components)) * budget
    assert total_sense <= int(phase2_fes * 0.4)
    assert budget >= 8


def test_full_share_matches_frozen_budget_when_affordable() -> None:
    components = ((0, 1, 2),)
    expected = capped_proposal_budget(
        200_000,
        components,
        refresh_cycles=4,
        neighborhood_fes=16,
        sense_budget_share=1.0,
    )
    assert expected == 16_658


@pytest.mark.parametrize(
    "kwargs",
    (
        {"sense_budget_share": 0.0},
        {"sense_budget_share": 1.1},
        {"sense_budget_share": True},
    ),
)
def test_capped_proposal_budget_rejects_invalid_share(kwargs) -> None:
    with pytest.raises(ValueError, match="sense_budget_share"):
        capped_proposal_budget(
            100_000,
            ((0, 1),),
            refresh_cycles=4,
            neighborhood_fes=16,
            **kwargs,
        )


def test_capped_proposal_budget_rejects_cap_that_cannot_pay_minimum() -> None:
    with pytest.raises(ValueError, match="minimum proposal"):
        capped_proposal_budget(
            100_000,
            tuple((index,) for index in range(20)),
            refresh_cycles=16,
            neighborhood_fes=32,
            sense_budget_share=0.01,
        )
