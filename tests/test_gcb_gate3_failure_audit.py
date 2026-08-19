from __future__ import annotations

from experiments.diagnose_gcb_component_variance import check_component_exchange, diagnose
from experiments.oracle_gcb_gate3 import TOPOLOGIES, _repair_seed, run_trial


def test_gate3_pairing_uses_equal_total_ledger_budget() -> None:
    trial = run_trial(next(item for item in TOPOLOGIES if item.name == "pair_pair"), 2026081803)

    assert trial.gcb_consumed_fes == trial.canonical_consumed_fes == 32
    assert trial.gcb_ledger_fes == trial.canonical_ledger_fes == 44


def test_repair_seed_is_independent_and_reproducible() -> None:
    assert _repair_seed(17) == _repair_seed(17)
    assert _repair_seed(17) != 17
    assert _repair_seed(17, 1) != _repair_seed(17, 2)


def test_isomorphic_pair_components_have_measurable_repair_variance() -> None:
    result = diagnose(proposal_seed=2026081803, replicates=8)

    assert all(item["std_error"] > 0.0 for item in result["summary"].values())


def test_pair_pair_component_exchange_is_equivariant() -> None:
    result = check_component_exchange(2026081803)

    assert result["component_equivariant"]
    assert result["objective_invariant"]
    assert result["budget_invariant"]
