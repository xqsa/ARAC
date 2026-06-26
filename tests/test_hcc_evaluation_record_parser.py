from __future__ import annotations

from pathlib import Path

import pytest

from arac.backends.hcc import _parse_hcc_evaluation_record


def test_parse_hcc_evaluation_record_reads_final_line(tmp_path: Path) -> None:
    record = tmp_path / "evaluation_record.txt"
    record.write_text("...\nFin:   2000 1.234e-05\n", encoding="utf-8")

    final_error, fe = _parse_hcc_evaluation_record(tmp_path)

    assert fe == 2000
    assert final_error == pytest.approx(1.234e-05)
