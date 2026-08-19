from __future__ import annotations

import json
from pathlib import Path

from experiments.historical_recovery.audit_independent_action_semantic_parity import (
    build_report,
    write_report,
)


def test_semantic_parity_audit_separates_reference_layers() -> None:
    report = build_report()
    lanes = {lane["lane"]: lane for lane in report["lanes"]}

    assert report["production_hcc_runtime_clean"] is True
    assert report["production_hcc_runtime_imports"] == []
    assert report["selector_evaluation_authorized"] is False
    assert set(lanes) == {"AOR", "CTP", "SMP", "GCB"}
    assert lanes["AOR"]["independent_v3_parity"] == "compatible"
    assert lanes["AOR"]["historical_hcc_parity"] == "not_equivalent"
    assert lanes["CTP"]["independent_v3_parity"] == "different"
    assert lanes["SMP"]["historical_hcc_parity"] == "unresolved"
    assert lanes["GCB"]["independent_v3_parity"] == "different"
    assert not any(lane["ready_for_selector_evaluation"] for lane in lanes.values())


def test_semantic_parity_sources_are_auditable() -> None:
    report = build_report()

    for lane in report["lanes"]:
        assert lane["current_source"]["auditable"] is True
        assert lane["frozen_v3_source"]["auditable"] is True
        assert lane["historical_protocol_verdict"] == "partial"
        assert lane["historical_replay_authorized"] is False


def test_semantic_parity_report_writer(tmp_path) -> None:
    report = build_report()
    json_path = tmp_path / "parity.json"
    markdown_path = tmp_path / "parity.md"

    write_report(report, json_path, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["schema_version"] == "arac-independent-action-semantic-parity-v1"
    assert "Historical HCC-backed results are numerical golden references only" in markdown
    assert "Do not restore an HCC production runner" in markdown


def test_frozen_pilot_protocol_is_design_only_and_selector_free() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            root
            / "experiments"
            / "historical_recovery"
            / "independent_semantic_parity_protocol.json"
        ).read_text(encoding="utf-8")
    )

    assert protocol["status"] == "frozen_design_not_run"
    assert protocol["selector_execution_allowed"] is False
    assert protocol["common_anchor"]["screen_step_fes"] == 120000
    assert protocol["common_anchor"]["native_threads"] == 1
    assert len(protocol["lanes"]) == 4
    assert {lane["action"] for lane in protocol["lanes"]} == {
        "aor",
        "ctp",
        "smp",
        "gcb",
    }
