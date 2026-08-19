from __future__ import annotations

from experiments.evidence_clique_full_context_gate24 import (
    CONTINUATION_FES,
    PROPOSAL_BUDGET_FES,
    WRITEBACK_ROUNDS,
)


def test_gate24_protocol_budget_is_frozen() -> None:
    assert PROPOSAL_BUDGET_FES == 48
    assert WRITEBACK_ROUNDS == 16
    assert CONTINUATION_FES == 32
