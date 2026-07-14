from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_paper_best_offline.py"
    spec = importlib.util.spec_from_file_location("compare_paper_best_offline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_comparison_rows_reports_seed_mean_worst_and_catastrophic() -> None:
    module = _load_module()
    rows = module.build_comparison_rows(
        [
            {"problem_id": "E1", "seed": "1", "hcc_smoke_final_error": "90"},
            {"problem_id": "E1", "seed": "2", "hcc_smoke_final_error": "100"},
            {"problem_id": "E1", "seed": "3", "hcc_smoke_final_error": "130"},
        ],
        [{"case": "E1", "paper_best": "100"}],
        paper_source="reference.csv",
    )

    row = rows[0]
    assert row["best_of_three_win"] == 1
    assert row["mean_win"] == 0
    assert row["worst_win"] == 0
    assert row["seed_win_count"] == 1
    assert row["catastrophic_seed_count"] == 1
    assert row["runtime_dispatch_used"] == 0


def test_build_comparison_rows_rejects_missing_seed() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="missing seeds"):
        module.build_comparison_rows(
            [{"problem_id": "E1", "seed": "1", "hcc_smoke_final_error": "90"}],
            [{"case": "E1", "paper_best": "100"}],
            paper_source="reference.csv",
        )


@pytest.mark.parametrize(
    ("result_rows", "paper_rows", "message"),
    [
        (
            [
                {"problem_id": "E1", "seed": "1", "hcc_smoke_final_error": "90"},
                {"problem_id": "E1", "seed": "1", "hcc_smoke_final_error": "91"},
            ],
            [{"case": "E1", "paper_best": "100"}],
            "duplicate result seed",
        ),
        (
            [
                {"problem_id": "E1", "seed": "1", "hcc_smoke_final_error": "90"},
            ],
            [
                {"case": "E1", "paper_best": "100"},
                {"case": "E1", "paper_best": "101"},
            ],
            "duplicate paper-best case",
        ),
        (
            [
                {"problem_id": "E1", "seed": "1", "hcc_smoke_final_error": "90"},
            ],
            [{"case": "E1", "paper_best": "0"}],
            "finite and positive",
        ),
    ],
)
def test_build_comparison_rows_rejects_ambiguous_or_invalid_inputs(
    result_rows: list[dict[str, str]],
    paper_rows: list[dict[str, str]],
    message: str,
) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match=message):
        module.build_comparison_rows(
            result_rows,
            paper_rows,
            paper_source="reference.csv",
        )
