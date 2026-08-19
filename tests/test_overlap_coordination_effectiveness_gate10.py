from __future__ import annotations

import numpy as np

from experiments.overlap_coordination_effectiveness_gate10 import (
    _auc,
    _context,
    _cell_summary,
)


def test_gate10_context_has_real_proposal_and_exact_fe_pairing() -> None:
    context = _context("conforming", "chain", 6, 31001)

    assert context["proposal_parity"] is True
    assert context["fe_parity"] is True
    assert context["strict_best"] is True
    assert context["coordination"]["proposal_fes"] == 6 * 48
    assert context["coordination"]["consumed_fes"] == context["control"]["consumed_fes"]
    assert context["coordination"]["ctp_fes"] in {0, 32}
    assert context["control"]["continuation_fes"] in {0, 32}


def test_gate10_conforming_and_conflicting_use_same_structure_but_different_proposals() -> None:
    conforming = _context("conforming", "star", 12, 31002)
    conflicting = _context("conflicting", "star", 12, 31002)

    assert conforming["groups"] == conflicting["groups"]
    assert conforming["shared_variables"] == conflicting["shared_variables"]
    assert conforming["coordination"]["proposal_runs"] != conflicting["coordination"]["proposal_runs"]
    assert conforming["coordination"]["proposal_fes"] == conflicting["coordination"]["proposal_fes"] == 288


def test_gate10_auc_uses_pairwise_probability_with_half_ties() -> None:
    assert _auc((3.0, 4.0), (1.0, 2.0)) == 1.0
    assert _auc((1.0,), (1.0, 2.0)) == 0.25
    assert np.isnan(_auc((), (1.0,)))


def test_gate10_cell_summary_requires_both_modes_and_all_five_seeds() -> None:
    contexts = tuple(
        {
            "topology": "random",
            "overlap_budget": 6,
            "mode": mode,
        }
        for mode in ("conforming", "conflicting")
        for _ in range(5)
    )
    summary = _cell_summary(contexts)
    random_cell = next(item for item in summary if item["topology"] == "random" and item["overlap_budget"] == 6)
    assert random_cell["context_count"] == 10
    assert random_cell["conforming_count"] == 5
    assert random_cell["conflicting_count"] == 5
    assert random_cell["complete"] is True
