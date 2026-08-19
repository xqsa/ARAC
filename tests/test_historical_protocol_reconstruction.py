from __future__ import annotations

import json

from experiments.historical_recovery.reconstruct_historical_protocols import (
    build_report,
    write_report,
)


def test_historical_lanes_are_not_replay_authorized_without_exact_sources() -> None:
    report = build_report()

    assert report["replay_authorized_lanes"] == []
    assert report["recovery_interpretation"] == (
        "golden_reference_only_independent_semantic_parity"
    )
    assert {lane["lane"] for lane in report["lanes"]} == {"AOR", "CTP", "GCB", "SMP"}
    assert all(lane["verdict"] == "partial" for lane in report["lanes"])
    assert "exact_exp057_worker_source_missing" in report["lanes"][0]["blockers"]
    assert "exact_exp058_runner_sha_unavailable" in report["lanes"][1]["blockers"]
    assert "exact_exp059_runner_sha_unavailable" in report["lanes"][2]["blockers"]
    assert "historical_25_seed_smp_lane_absent" in report["lanes"][3]["blockers"]


def test_report_writer_emits_machine_readable_and_markdown_outputs(tmp_path) -> None:
    report = build_report()
    json_path = tmp_path / "reconstruction.json"
    markdown_path = tmp_path / "reconstruction.md"

    write_report(report, json_path, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "arac-historical-protocol-reconstruction-v1"
    assert payload["replay_authorized_lanes"] == []
    assert "Authorized lanes: **none**." in markdown_path.read_text(encoding="utf-8")
    assert "Production ARAC must not restore an HCC runtime dependency." in (
        markdown_path.read_text(encoding="utf-8")
    )
