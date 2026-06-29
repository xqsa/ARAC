from __future__ import annotations

from pathlib import Path

import pytest

from arac.backends.hcc import (
    _parse_hcc_evaluation_record,
    _parse_hcc_evaluation_record_with_optimizer_final_fe,
)


def test_parse_hcc_evaluation_record_reads_final_line(tmp_path: Path) -> None:
    record = tmp_path / "evaluation_record.txt"
    record.write_text("...\nFin:   2000 1.234e-05\n", encoding="utf-8")

    final_error, fe = _parse_hcc_evaluation_record(tmp_path)

    assert fe == 2000
    assert final_error == pytest.approx(1.234e-05)


def test_parse_hcc_evaluation_record_prefers_budget_checkpoint(tmp_path: Path) -> None:
    record = tmp_path / "evaluation_record.txt"
    record.write_text(
        "\n".join(
            [
                "Algorithm: elliptic_2",
                "                    2.000e+03                9.000000                   9.000000e+00",
                "                    Fin:2.128e+03            8.000000                   8.000000e+00",
            ]
        ),
        encoding="utf-8",
    )

    final_error, fe = _parse_hcc_evaluation_record(tmp_path, budget_limit=2000)

    assert fe == 2000
    assert final_error == pytest.approx(9.0)


def test_parse_hcc_evaluation_record_exposes_optimizer_final_fe(tmp_path: Path) -> None:
    record = tmp_path / "evaluation_record.txt"
    record.write_text(
        "\n".join(
            [
                "Algorithm: elliptic_2",
                "                    2.000e+03                9.000000                   9.000000e+00",
                "                    Fin:2.128e+03            8.000000                   8.000000e+00",
            ]
        ),
        encoding="utf-8",
    )

    final_error, fe, optimizer_final_fe = (
        _parse_hcc_evaluation_record_with_optimizer_final_fe(
            tmp_path,
            budget_limit=2000,
        )
    )

    assert fe == 2000
    assert optimizer_final_fe == 2128
    assert final_error == pytest.approx(9.0)


def test_parse_hcc_evaluation_record_uses_budget_summary_for_rounded_final_fe(
    tmp_path: Path,
) -> None:
    record = tmp_path / "evaluation_record.txt"
    record.write_text(
        "\n".join(
            [
                "Algorithm: schwefel_4",
                "                    3.000e+06                10.000000                  1.000000e+01",
                "                    Fin:3.000e+06            9.000000                   9.000000e+00",
            ]
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "S4_budget_summary.csv"
    summary.write_text(
        "\n".join(
            [
                "problem_id,budget_accounting,max_fes,optimizer_reported_fe,fitness_record_fe,budget_aligned_fe,same_budget_violation",
                "S4,source,3000000,3000178,3000178,3000000,1",
            ]
        ),
        encoding="utf-8",
    )

    final_error, fe, optimizer_final_fe = (
        _parse_hcc_evaluation_record_with_optimizer_final_fe(
            tmp_path,
            budget_limit=3_000_000,
        )
    )

    assert fe == 3_000_000
    assert optimizer_final_fe == 3_000_178
    assert final_error == pytest.approx(10.0)
