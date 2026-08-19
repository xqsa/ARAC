from __future__ import annotations

from experiments.oracle_overlap_discovery_gate5 import _summary, _trial, run_diagnostic


def test_gate5_trial_recovers_pair_topology_in_both_modes() -> None:
    for mode in ("conforming", "conflicting"):
        trial = _trial((2026081301, "pair", "ackley", mode))
        assert trial.group_exact
        assert trial.expected_shared == trial.inferred_shared == (2,)
        assert trial.shared_precision == trial.shared_recall == 1.0
        assert trial.consumed_fes == trial.expected_fes == 80
        assert trial.adapter_ready
        assert trial.deterministic


def test_gate5_disjoint_control_has_no_shared_false_positive() -> None:
    trial = _trial((2026081301, "disjoint", "rastrigin", "conflicting"))

    assert trial.group_exact
    assert trial.expected_shared == trial.inferred_shared == ()
    assert trial.consumed_fes == trial.expected_fes == 110


def test_gate5_small_diagnostic_has_all_hard_checks() -> None:
    payload = run_diagnostic(seeds=(2026081301,), workers=1)

    assert payload["summary"]["runs"] == 50
    assert payload["summary"]["gate_passed"]
    assert all(payload["summary"]["gate_checks"].values())


def test_gate5_summary_fails_when_a_trial_is_not_exact() -> None:
    trial = _trial((2026081301, "pair", "sphere", "conforming"))
    failed = trial.__class__(
        **{
            **trial.__dict__,
            "group_exact": False,
        }
    )

    summary = _summary([failed])

    assert summary["gate_passed"] is False
