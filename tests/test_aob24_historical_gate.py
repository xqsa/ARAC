from __future__ import annotations

from experiments.historical_recovery import run_aob24_historical_gate as gate


def test_reference_table_and_case_mapping_are_complete() -> None:
    references = gate._reference_rows()

    assert len(references) == 24
    assert references["E1"]["displayed_mean"] == "5.69E+05"
    assert references["S1"]["displayed_mean"] == "1.04E+00"
    assert references["R1"]["displayed_mean"] == "1.56E+05"
    assert references["A1"]["displayed_mean"] == "7.80E+04"
    assert {case for cases in gate.CASE_MAPPING.values() for case in cases} == set(references)


def test_completed_non_smp_lanes_pass_at_displayed_precision() -> None:
    summaries = {}
    for action in ("ctp", "gcb", "aor"):
        path = gate.LANES[action]["summary"]
        summaries[action] = gate._read_json(path)
    summaries["smp"] = {
        "case_summaries": [
            {
                "case": case,
                "trajectory_count": 25,
                "mean": reference["numeric_mean"],
                "sample_std": 0.0,
            }
            for case, reference in gate._reference_rows().items()
            if case.startswith("E")
        ]
    }

    evaluations = gate._evaluate_summaries(summaries)
    non_smp = [row for row in evaluations if row["action"] != "smp"]
    assert len(non_smp) == 18
    assert all(row["displayed_mean_no_higher"] for row in non_smp)
    assert any(not row["raw_mean_no_higher"] for row in non_smp)
