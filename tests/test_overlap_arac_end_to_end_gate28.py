from __future__ import annotations

from experiments.overlap_arac_end_to_end_gate28 import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    DIMENSION,
    PHASE1_FES,
    TOTAL_BUDGET_FES,
)


def test_gate28_protocol_is_frozen() -> None:
    assert DIMENSION == 1_000
    assert TOTAL_BUDGET_FES == 3_000_000
    assert PHASE1_FES == 180_000
    assert DEFAULT_REFRESH_CYCLES == 16
    assert DEFAULT_NEIGHBORHOOD_FES == 32
