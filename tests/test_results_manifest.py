from __future__ import annotations

import csv
from pathlib import Path


def _write_run(run_dir: Path, *, complete: bool, case: str, seed: str) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.md").write_text(
        "# exp_test Run Manifest\n"
        "Protocol: pilot\n"
        "Config: configs/test.yaml\n"
        "- git commit: abc123\n"
        "Claim level: pilot_evidence\n"
        + ("Budget: 3000000 FE per lane/case\n" if complete else ""),
        encoding="utf-8",
    )
    (run_dir / "same_budget_ledger.csv").write_text(
        "problem_id,seed,total_fe,same_budget_violation\n"
        f"{case},{seed},3000000,0\n",
        encoding="utf-8",
    )


def test_manifest_emits_stable_schema_and_marks_missing_metadata_partial(tmp_path: Path) -> None:
    from scripts.build_results_manifest import FIELDNAMES, build_manifest

    results = tmp_path / "results"
    _write_run(results / "exp_b", complete=False, case="R2", seed="2")
    _write_run(results / "exp_a", complete=True, case="E1", seed="1")

    output = tmp_path / "manifest.csv"
    build_manifest(results, output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == list(FIELDNAMES)
    assert [row["experiment_id"] for row in rows] == ["exp_a", "exp_b"]
    assert rows[0]["status"] == "complete"
    assert rows[1]["status"] == "partial"
    assert rows[0]["case_id"] == "E1"
    assert rows[0]["seed"] == "1"
    assert rows[0]["total_fe"] == "3000000"
    assert "paper" not in " ".join(rows[0]).lower()


def test_manifest_is_deterministic_and_does_not_modify_results(tmp_path: Path) -> None:
    from scripts.build_results_manifest import build_manifest

    results = tmp_path / "results"
    _write_run(results / "exp_a", complete=True, case="E1", seed="1")
    before = sorted(
        (path.relative_to(results), path.read_bytes())
        for path in results.rglob("*")
        if path.is_file()
    )

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    build_manifest(results, first)
    build_manifest(results, second)

    assert first.read_bytes() == second.read_bytes()
    after = sorted(
        (path.relative_to(results), path.read_bytes())
        for path in results.rglob("*")
        if path.is_file()
    )
    assert before == after
