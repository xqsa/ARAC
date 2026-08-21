from __future__ import annotations

from experiments.oc_coupling_fresh_seed_gate import FreshCell
from experiments.oc_coupling_two_baseline_gate import run_cell


def test_two_baseline_fresh_cell_is_deterministic_and_contract_safe() -> None:
    cell = FreshCell("star", "conflict", 2026082101)
    first = run_cell(cell)
    second = run_cell(cell)
    assert first == second
    assert first["counterfactual_two_fe"] is True
    assert first["archive_preserved"] is True
    assert first["strict_best"] is True
    assert first["repair_consumed_fes"] == 8
