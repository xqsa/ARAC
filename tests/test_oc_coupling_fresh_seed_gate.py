from __future__ import annotations

import numpy as np

from experiments.oc_coupling_fresh_seed_gate import (
    AUTHORITY_THRESHOLD,
    FreshCell,
    _correlation_interval,
    _parameters,
    _quantile_interval,
    run_cell,
)


def test_fresh_parameters_are_deterministic_and_non_degenerate() -> None:
    assert _parameters(2026082101) == _parameters(2026082101)
    assert _parameters(2026082101) != _parameters(2026082102)


def test_fresh_cell_preserves_contracts_and_exact_patch_budget() -> None:
    result = run_cell(FreshCell("chain", "synergy", 2026082101))
    assert result["counterfactual_one_fe"] is True
    assert result["archive_preserved"] is True
    assert result["strict_best"] is True
    assert result["repair_consumed_fes"] == 8


def test_bootstrap_intervals_are_finite_and_authority_threshold_is_explicit() -> None:
    median, lower, upper = _quantile_interval([0.0, 0.1, 0.2], seed=17)
    assert lower <= median <= upper
    interval = _correlation_interval([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0], rank=True)
    assert all(np.isfinite(value) for value in interval)
    assert AUTHORITY_THRESHOLD == 0.30
