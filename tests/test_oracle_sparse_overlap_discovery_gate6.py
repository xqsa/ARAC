from __future__ import annotations

from experiments.oracle_sparse_overlap_discovery_gate6 import _pilot_trial, _summary, _trial, run_diagnostic


def test_gate6_trial_recovers_pair_topology_in_both_modes() -> None:
    for mode in ("conforming", "conflicting"):
        trial = _trial((2026081301, "pair", "ackley", mode))
        assert trial.group_exact
        assert trial.expected_shared == trial.inferred_shared == (2,)
        assert trial.shared_precision == trial.shared_recall == 1.0
        assert trial.separated_pair_fraction == 1.0
        assert trial.consumed_fes == trial.expected_fes
        assert trial.adapter_ready
        assert trial.deterministic


def test_gate6_disjoint_control_has_no_shared_false_positive() -> None:
    trial = _trial((2026081301, "disjoint", "rastrigin", "conflicting"))

    assert trial.group_exact
    assert trial.expected_shared == trial.inferred_shared == ()
    assert trial.consumed_fes == trial.expected_fes


def test_gate6_small_diagnostic_has_all_hard_checks() -> None:
    payload = run_diagnostic(seeds=(2026081301,), workers=1)

    assert payload["summary"]["runs"] == 50
    assert payload["summary"]["pilot_runs"] == 1
    assert payload["summary"]["gate_passed"]
    assert all(payload["summary"]["gate_checks"].values())


def test_gate6_summary_fails_when_fe_is_not_sparse() -> None:
    trial = _pilot_trial(2026081301)
    failed = trial.__class__(**{**trial.__dict__, "consumed_fes": trial.full_pair_fes})

    summary = _summary([failed])

    assert summary["gate_passed"] is False
