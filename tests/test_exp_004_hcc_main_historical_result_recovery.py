from __future__ import annotations

import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_record(path: Path, final_fe: str, final_error: str, runtime: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Algorithm           Record Point             Fitness Value",
                f"                    Fin:{final_fe}            {final_error}            {float(final_error):.6e}",
                f"                    Run Time:                {runtime}                 {float(runtime):.6e}",
            ]
        ),
        encoding="utf-8",
    )


def test_exp_004_recovers_hcc_main_historical_results_and_compares_paper(
    tmp_path: Path,
) -> None:
    from experiments.exp_004_hcc_main_historical_result_recovery.run import (
        run_hcc_main_historical_result_recovery,
    )

    hcc_result_root = tmp_path / "HCC-main" / "HCC_SRC" / "result"
    _write_record(
        hcc_result_root
        / "mi_arac_action_clean_hcc_anchor_5seed"
        / "seed-1"
        / "E2"
        / "elliptic"
        / "evaluation_record.txt",
        final_fe="3.000e+06",
        final_error="1000.0",
        runtime="12.5",
    )
    _write_record(
        hcc_result_root
        / "mi_arac_action_clean_hcc_anchor_5seed"
        / "seed-2"
        / "A1"
        / "ackley"
        / "evaluation_record.txt",
        final_fe="3.001e+06",
        final_error="90000.0",
        runtime="13.5",
    )

    output = run_hcc_main_historical_result_recovery(
        output_dir=tmp_path / "exp004",
        hcc_result_root=hcc_result_root,
    )

    assert (output / "hcc_main_historical_result_inventory.csv").exists()
    assert (output / "hcc_main_vs_paper_reported_comparison.csv").exists()
    assert (output / "hcc_main_historical_results_audit.md").exists()

    inventory = _read_csv(output / "hcc_main_historical_result_inventory.csv")
    assert len(inventory) == 2
    e2 = next(row for row in inventory if row["problem_id"] == "E2")
    assert e2["seed"] == "1"
    assert e2["final_error"] == "1.000000e+03"
    assert e2["fe_used"] == "3000000"
    assert e2["runtime_dispatch_allowed"] == "0"
    assert e2["usable_for_offline_evaluation"] == "1"

    comparison = _read_csv(output / "hcc_main_vs_paper_reported_comparison.csv")
    e2_comparison = next(row for row in comparison if row["problem_id"] == "E2")
    assert e2_comparison["paper_reported_mean"] == "6.87E6"
    assert e2_comparison["better_than_paper_reported"] == "1"
    assert e2_comparison["runtime_dispatch_allowed"] == "0"

    a1_comparison = next(row for row in comparison if row["problem_id"] == "A1")
    assert a1_comparison["better_than_paper_reported"] == "0"
