from __future__ import annotations

from experiments.phase1_full_context_integration_gate22 import (
    PROPOSAL_BUDGET_FES,
    WRITEBACK_ROUNDS,
)


def test_gate22_protocol_budget_is_frozen() -> None:
    assert PROPOSAL_BUDGET_FES == 64
    assert WRITEBACK_ROUNDS == 16
    assert 4 * PROPOSAL_BUDGET_FES + 2 * (3 + 2 * WRITEBACK_ROUNDS) == 326
