from __future__ import annotations

from pathlib import Path

import pytest

from experiments.historical_recovery.audit_exp052_environment import build_report


@pytest.mark.skipif(
    not Path(".venv/Scripts/python.exe").is_file(),
    reason="project virtual environment is unavailable",
)
def test_exp052_environment_has_session_binding_but_is_not_receipt_bound() -> None:
    report = build_report()

    assert report["all_expected_packages_match"] is True
    assert report["all_pinned_packages_match"] is True
    assert report["environment"]["python_version"] == "3.12.7"
    assert report["session_evidence_complete"] is True
    assert report["formal_session_dependency_mutation_scan"]["events"] == []
    assert report["session_observed_environment_binding"] is True
    assert report["receipt_environment_binding"] is False
    assert report["environment_binding_complete"] is False
    assert report["replay_authorized"] is False
