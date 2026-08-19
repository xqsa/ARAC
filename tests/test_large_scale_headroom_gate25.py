from __future__ import annotations

from experiments.large_scale_headroom_gate25 import (
    CONTINUATION_FES,
    DIMENSION,
    PHASE1_FES,
    PROPOSAL_BUDGET_FES,
    WRITEBACK_ROUNDS,
)


def test_gate25_protocol_is_frozen() -> None:
    assert DIMENSION == 1000
    assert PHASE1_FES == 180_000
    assert PROPOSAL_BUDGET_FES == 64
    assert WRITEBACK_ROUNDS == 16
    assert CONTINUATION_FES == 32
