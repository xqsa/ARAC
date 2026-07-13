from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_STAGES = {
    "pilots",
    "infrastructure",
    "recovery",
    "ablations",
    "final",
}
CANONICAL_EXPERIMENTS = {
    "exp_001_schema_smoke": "pilots",
    "exp_002_aob_1run_pilot": "pilots",
    "exp_003_hcc_runtime_consumer_smoke": "pilots",
    "exp_004_hcc_main_historical_result_recovery": "recovery",
    "exp_005_hcc_final_protocol_pilot": "final",
}


def test_experiments_are_grouped_by_research_stage() -> None:
    stages = {
        child.name
        for child in (ROOT / "experiments").iterdir()
        if child.is_dir() and not child.name.startswith("__")
    }

    assert EXPECTED_STAGES <= stages
    for experiment_id, stage in CANONICAL_EXPERIMENTS.items():
        experiment_dir = ROOT / "experiments" / stage / experiment_id
        assert experiment_dir.is_dir()
        assert (experiment_dir / "README.md").is_file()
        assert (experiment_dir / "run.py").is_file()
        assert (experiment_dir / "expected_outputs.md").is_file()
        assert f"Stage: `{stage}`" in (experiment_dir / "README.md").read_text(
            encoding="utf-8"
        )


def test_experiment_packages_are_importable_from_a_non_repository_cwd() -> None:
    import subprocess
    import sys

    command = [
        sys.executable,
        "-c",
        "from experiments.pilots.exp_001_schema_smoke import run as exp001; "
        "from experiments.pilots.exp_002_aob_1run_pilot import run as exp002; "
        "from experiments.pilots.exp_003_hcc_runtime_consumer_smoke import run as exp003; "
        "from experiments.recovery.exp_004_hcc_main_historical_result_recovery "
        "import run as exp004; "
        "from experiments.final.exp_005_hcc_final_protocol_pilot import run as exp005; "
        "print(exp001.RUN_ID, exp002.RUN_ID, exp003.RUN_ID, exp004.RUN_ID, exp005.RUN_ID)",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    completed = subprocess.run(
        command,
        cwd=Path.home(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_experiment_outputs_default_to_repository_results_root() -> None:
    from experiments.paths import repository_root, results_root

    assert repository_root() == ROOT
    assert results_root() == ROOT / "results"
