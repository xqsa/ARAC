from __future__ import annotations

import numpy as np

from experiments.oc_coupling_diagnostic_gate import (
    Cell,
    PATCH_BUDGET_FES,
    _correlation,
    _run_cell,
    run_gate,
)


def test_coupling_gate_separates_controlled_regimes() -> None:
    payload = run_gate()
    assert payload["gate_passed"] is True
    checks = payload["gate_checks"]
    assert all(checks.values())
    medians = payload["summary"]["median_coupled_gain_by_regime"]
    assert medians["none"] == 0.0
    assert medians["neutral"] == 0.0
    assert medians["synergy"] > 0.05
    assert medians["conflict"] < -0.05


def test_coupling_cell_is_reproducible_and_budget_exact() -> None:
    cell = Cell("chain", "synergy", 2026082001)
    first = _run_cell(cell)
    second = _run_cell(cell)
    assert first == second
    assert first["counterfactual"]["consumed_fes"] == 1
    assert first["archive_preserved"] is True
    assert first["repair"]["consumed_fes"] == PATCH_BUDGET_FES


def test_correlation_handles_constant_inputs_without_nan() -> None:
    assert _correlation([1.0, 1.0], [2.0, 3.0], rank=False) == 0.0
    assert np.isfinite(_correlation([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0], rank=True))
