from __future__ import annotations

from experiments.oracle_ctp_gate2 import CTP_BUDGET_FES, run_diagnostic, run_trial


def test_persistent_conflict_ctp_consumes_exact_budget() -> None:
    trial = run_trial("conflicting", 17, sample_fes=256)

    assert trial.ctp_triggered
    assert trial.ctp_consumed_fes == CTP_BUDGET_FES
    assert trial.baseline_fes == 8 + CTP_BUDGET_FES
    assert trial.ctp_fes == 8 + CTP_BUDGET_FES
    assert trial.ctp_error >= 0.0
    assert trial.budget_matched is True
    assert trial.ctp_archive_nonworsening


def test_ctp_diagnostic_keeps_conforming_path_untriggered() -> None:
    result = run_diagnostic((17, 23, 31), sample_fes=256, workers=1)

    assert result["summary"]["conforming"]["ctp_trigger_rate"] == 0.0
    assert result["summary"]["conflicting"]["ctp_trigger_rate"] == 1.0
    assert result["summary"]["conforming"]["ctp_archive_nonworsening"]
