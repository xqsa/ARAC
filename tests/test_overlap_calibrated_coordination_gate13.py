from __future__ import annotations

import numpy as np

from experiments.overlap_calibrated_coordination_gate13 import (
    CTP_BUDGET_FES,
    PROPOSAL_BUDGET_FES,
    PROPOSAL_REPLICATES,
    _build,
    _context,
    _run_proposals,
)


def test_gate13_calibrates_from_first_three_and_keeps_heldout_separate() -> None:
    problem, _objective, structure = _build("conflicting", "random", 6, 31001)
    _ledger, _anchor, _anchor_error, runs, proposals, heldout = _run_proposals(problem, structure, 31001)

    assert all(len(group_runs) == PROPOSAL_REPLICATES for group_runs in runs)
    assert len(proposals) == len(structure.groups) == 6
    assert len(heldout) == len(structure.shared_variables)
    assert all(np.isfinite(item["heldout_standardized_score"]) for item in heldout)
    for group, proposal in enumerate(proposals):
        for variable, value in proposal.values:
            expected = np.mean(
                [dict(runs[group][replicate].proposal.values)[variable] for replicate in range(3)]
            )
            assert value == expected


def test_gate13_paired_arms_have_exact_fe_and_strict_best() -> None:
    context = _context("conforming", "chain", 12, 31002)

    assert context["proposal_parity"] is True
    assert context["fe_parity"] is True
    assert context["strict_best"] is True
    assert np.isfinite(context["max_calibrated_residual_score"])
    assert np.isfinite(context["max_heldout_standardized_score"])
    assert context["coordination"]["proposal_fes"] == 1 + 6 * 4 * PROPOSAL_BUDGET_FES
    assert context["coordination"]["consumed_fes"] == context["control"]["consumed_fes"]
    assert context["control"]["ctp_fes"] == 0
    assert context["control"]["control_fes"] in {0, CTP_BUDGET_FES}


def test_gate13_same_seed_reproduces_calibrated_proposals() -> None:
    first = _context("conflicting", "star", 6, 31003)
    second = _context("conflicting", "star", 6, 31003)

    assert first["coordination"]["proposal_payload"] == second["coordination"]["proposal_payload"]
    assert first["coordination"]["heldout_residuals"] == second["coordination"]["heldout_residuals"]
    assert first["coordination"]["final_error"] == second["coordination"]["final_error"]


def test_gate13_different_modes_keep_structure_but_change_calibrated_proposals() -> None:
    conforming = _context("conforming", "random", 6, 31004)
    conflicting = _context("conflicting", "random", 6, 31004)

    assert conforming["groups"] == conflicting["groups"]
    assert conforming["shared_variables"] == conflicting["shared_variables"]
    assert conforming["coordination"]["proposal_payload"] != conflicting["coordination"]["proposal_payload"]
