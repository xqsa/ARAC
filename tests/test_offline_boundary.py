from __future__ import annotations

from pathlib import Path

from scripts.audit_project_structure import audit_project_structure


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_source_has_no_offline_path_references() -> None:
    report = audit_project_structure(ROOT)

    runtime_errors = [
        finding
        for finding in report.errors
        if finding.path.startswith("src/arac/")
        and any(token in finding.rule for token in ("paper", "historical", "results"))
    ]

    assert runtime_errors == []


def test_documents_and_failed_experiment_record_have_explicit_boundaries() -> None:
    assert (ROOT / "docs" / "design" / "core-method.md").is_file()
    assert (ROOT / "docs" / "design" / "boundaries.md").is_file()
    assert (ROOT / "docs" / "protocols" / "aob-final-evaluation-protocol.md").is_file()

    archive_readme = (
        ROOT
        / "archive"
        / "failed-experiments"
        / "v33-late-stagnation-nda-takeover"
        / "README.md"
    )
    content = archive_readme.read_text(encoding="utf-8")
    assert "v3.3" in content
    assert "stable" in content
    assert "runtime" in content
    assert "3.28e5" in content


def test_offline_evidence_is_readable_outside_runtime_package() -> None:
    paper_values = ROOT / "references" / "paper_reported_table2_hcc_es.csv"
    historical_values = ROOT / "references" / "hcc_main_historical_result_inventory.csv"

    assert paper_values.read_text(encoding="utf-8").strip()
    assert historical_values.read_text(encoding="utf-8").strip()
