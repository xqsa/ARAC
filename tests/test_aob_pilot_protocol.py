from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = [f"{family}{idx}" for family in "ESRA" for idx in range(1, 7)]
GAMMA = {1: 0, 2: 1, 3: 3, 4: 5, 5: 7, 6: 10}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_paper_reported_hcc_es_anchor_covers_aob_24_cases() -> None:
    rows = read_csv(ROOT / "references" / "paper_reported_table2_hcc_es.csv")

    assert len(rows) == 24
    assert [row["problem_id"] for row in rows] == PROBLEMS
    assert {row["source"] for row in rows} == {"Two-Phase CC Table 2"}
    assert {row["comparison_role"] for row in rows} == {"paper_reported_evaluation_only"}
    assert {row["runtime_dispatch_allowed"] for row in rows} == {"0"}

    by_problem = {row["problem_id"]: row for row in rows}
    for problem in PROBLEMS:
        row = by_problem[problem]
        idx = int(problem[1])
        assert int(row["dimension"]) == 1000
        assert int(row["overlap_gamma"]) == GAMMA[idx]
        assert float(row["reported_mean"]) >= 0
        assert float(row["reported_time"]) > 0


def test_aob_pilot_config_is_one_run_against_final_25_run_protocol() -> None:
    config = (ROOT / "configs" / "aob_pilot.yaml").read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "aob-final-evaluation-protocol.md").read_text(
        encoding="utf-8"
    )

    assert "runs: 1" in config
    assert "final_target_runs: 25" in config
    assert "total_fe: 3000000" in config
    assert "E1" in config and "A6" in config

    assert "Current pilot: 1 independent run" in protocol
    assert "Final protocol: 25 independent runs" in protocol
    assert "paper-reported evaluation-only baselines" in protocol
    assert "must not enter runtime dispatch" in protocol
