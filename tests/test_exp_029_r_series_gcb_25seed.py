from __future__ import annotations

import math
from pathlib import Path

from experiments.pilots.exp_026_arac_vs_hcc_paired import run as relation_run
from experiments.pilots.exp_027_r1_gcb import run as boundary_run
from experiments.pilots.exp_029_r_series_gcb_25seed import run as exp029


def test_config_freezes_25_seed_r_series_matrix() -> None:
    config = exp029.load_config()
    execution = config["execution"]
    assert isinstance(execution, dict)
    assert execution["cases"] == list(exp029.CASES)
    assert execution["seeds"] == list(range(117, 142))
    assert execution["max_fes"] == 3_000_000
    assert execution["action"] == "gcb"


def test_matrix_is_round_robin_and_contains_150_unique_trajectories(
    tmp_path: Path,
) -> None:
    specs = exp029.build_specs(tmp_path)
    assert len(specs) == 150
    assert tuple(spec.case for spec in specs[:6]) == exp029.CASES
    assert len({(spec.case, spec.seed) for spec in specs}) == 150


def test_backend_specs_use_current_gcb_adapters(tmp_path: Path) -> None:
    r1 = exp029._backend_spec(exp029.RunSpec("R1", 117, tmp_path))
    r4 = exp029._backend_spec(exp029.RunSpec("R4", 117, tmp_path))
    assert isinstance(r1, boundary_run.RunSpec)
    assert r1.experiment_id == exp029.EXPERIMENT_ID
    assert isinstance(r4, relation_run.RunSpec)
    assert r4.action == relation_run.R_ACTION == "gcb"
    assert r4.experiment_id == exp029.EXPERIMENT_ID


def test_case_summary_uses_sample_standard_deviation() -> None:
    results = []
    for case_index, case in enumerate(exp029.CASES, start=1):
        for seed_index, seed in enumerate(exp029.SEEDS):
            results.append(
                {
                    "case": case,
                    "seed": seed,
                    "ok": True,
                    "final_error": float(case_index * 100 + seed_index),
                }
            )
    summaries = exp029.build_case_summaries(results)
    assert len(summaries) == 6
    assert all(summary["n"] == 25 for summary in summaries)
    assert math.isclose(summaries[0]["mean_error"], 112.0)
    assert math.isclose(
        summaries[0]["sample_std_error"],
        math.sqrt(1300.0 / 24.0),
    )
