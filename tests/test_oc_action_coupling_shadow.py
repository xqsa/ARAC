from __future__ import annotations

import numpy as np

from experiments.oc_action_semantic_gate import _context


def test_action_conditioned_coupling_receipts_are_contract_safe() -> None:
    context = _context("conforming", "random", 6, 31501)

    assert context.coupling_receipt_parity is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert context.promotion_applied is False
    assert all(
        np.isfinite(float(getattr(arm, "coupled_gain")))
        for arm in (
            context.owner_control,
            context.shared_core,
            context.expanded_shared_private,
            context.duplicated_shared_competition,
            context.duplicated_shared_local_competition,
        )
    )
