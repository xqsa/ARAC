from __future__ import annotations

from experiments.oracle_aor_escalation_gate4 import run_diagnostic, run_trial


def test_gate4_trial_uses_equal_bounded_escalation_budget() -> None:
    trial = run_trial("conflicting", 2026082002)

    assert trial.escalation_triggered
    assert trial.aor_consumed_fes == trial.control_consumed_fes == 64
    assert trial.aor_total_ledger_fes == trial.control_total_ledger_fes
    assert trial.aor_archive_nonworsening
    assert trial.control_archive_nonworsening


def test_gate4_conforming_does_not_escalate_for_smoke_seeds() -> None:
    result = run_diagnostic((2026082001, 2026082002), workers=1)

    conforming = result["summary"]["conforming"]
    assert conforming["escalation_trigger_rate"] <= 0.5
