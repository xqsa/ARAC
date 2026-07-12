from __future__ import annotations

import csv
from pathlib import Path

from experiments.final.exp_005_hcc_final_protocol_pilot import run as final_protocol


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references" / "paper_reported_table2_best_by_case.csv"


def test_complete_paper_best_reference_covers_all_24_aob_cases() -> None:
    with REFERENCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected_cases = {
        f"{family}{function_id}"
        for family in "ESRA"
        for function_id in range(1, 7)
    }
    by_case = {row["case"]: row for row in rows}

    assert set(by_case) == expected_cases
    assert len(rows) == 24
    assert all(row["runtime_dispatch_allowed"] == "0" for row in rows)
    assert float(by_case["S1"]["paper_best"]) == 1.92e-3
    assert float(by_case["R5"]["paper_best"]) == 2.48e5
    assert float(by_case["A6"]["paper_best"]) == 7.80e4


def test_final_protocol_uses_complete_offline_paper_best_reference() -> None:
    assert final_protocol.DEFAULT_PAPER_BEST_MATRIX == REFERENCE
