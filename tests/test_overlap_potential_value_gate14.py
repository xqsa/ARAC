from __future__ import annotations

import numpy as np

from experiments.overlap_potential_value_gate14 import (
    CTP_BUDGET_FES,
    _context,
    _rank_correlation,
)


def test_gate14_forced_ctp_and_control_have_exact_paired_budget() -> None:
    context = _context("conflicting", "random", 6, 31001)

    assert context["proposal_parity"] is True
    assert context["fe_parity"] is True
    assert context["strict_best"] is True
    assert context["ctp"]["event"]["consumed_ctp_fes"] == CTP_BUDGET_FES
    assert context["control"]["extra_fes"] == CTP_BUDGET_FES


def test_gate14_forces_ctp_even_when_policy_level_is_not_high() -> None:
    context = _context("conforming", "chain", 12, 31002)

    assert context["ctp"]["event"]["consumed_ctp_fes"] == CTP_BUDGET_FES
    assert context["ctp"]["event"]["ctp_triggered_by_policy"] in {True, False}


def test_gate14_same_seed_is_reproducible() -> None:
    first = _context("conflicting", "star", 6, 31003)
    second = _context("conflicting", "star", 6, 31003)

    assert first["gain"] == second["gain"]
    assert first["ctp"]["final_error"] == second["ctp"]["final_error"]
    assert first["control"]["final_error"] == second["control"]["final_error"]


def test_gate14_rank_correlations_are_finite_and_zero_for_constant_input() -> None:
    assert np.isfinite(_rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], spearman=False))
    assert np.isfinite(_rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], spearman=True))
    assert _rank_correlation([1.0, 1.0], [2.0, 3.0], spearman=False) == 0.0
