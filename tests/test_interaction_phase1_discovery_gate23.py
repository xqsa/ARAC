from __future__ import annotations

from experiments.interaction_phase1_discovery_gate23 import (
    ANCHOR_COUNT,
    BUCKET_SIZE,
    DIMENSION,
    MAX_CANDIDATE_PAIRS,
    PHASE1_FES,
    ROUNDS,
)


def test_gate23_protocol_is_frozen() -> None:
    assert DIMENSION == 24
    assert PHASE1_FES == 180_000
    assert ANCHOR_COUNT == 5
    assert ROUNDS == 12
    assert BUCKET_SIZE == 4
    assert MAX_CANDIDATE_PAIRS == 128
