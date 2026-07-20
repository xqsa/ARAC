from __future__ import annotations

import math
from pathlib import Path

import pytest

from arac.actions.group_optimizer_type import (
    DIAGONAL_COVARIANCE_MODE,
    FULL_CMAES_MODE,
)
from scripts import hcc_smoke_runner


@pytest.mark.parametrize(
    "group_optimizer_mode",
    [FULL_CMAES_MODE, DIAGONAL_COVARIANCE_MODE],
)
def test_real_e1_run_problem_entrypoint_completes(
    group_optimizer_mode: str,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / group_optimizer_mode
    output_path.mkdir()
    config = hcc_smoke_runner.SmokeConfig(
        max_fes=200,
        seed=117,
        run_id=f"entrypoint-{group_optimizer_mode}",
        verbose=0,
        arac_action=hcc_smoke_runner.NATIVE_EQ8_ACTION,
        enable_relation_dispatch=False,
        relation_policy_mode="controller_v31",
        budget_accounting="strict",
        skip_plots=True,
        aob_data_root=hcc_smoke_runner.DATA_DIR.resolve(),
        evidence_overlay_mode="off",
        runtime_probe_repair_mode="hard_repair",
        group_optimizer_mode=group_optimizer_mode,
    )

    fitness_record, elapsed, trace_rows = hcc_smoke_runner.run_problem(
        "elliptic",
        1,
        output_path,
        config,
    )

    assert fitness_record
    assert all(math.isfinite(float(value)) for value in fitness_record)
    assert elapsed >= 0.0
    assert isinstance(trace_rows, list)
