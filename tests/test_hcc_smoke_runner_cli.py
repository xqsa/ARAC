from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_runner_module():
    hcc_src = Path(__file__).resolve().parents[1] / "HCC_SRC"
    sys.path.insert(0, str(hcc_src))
    runner_path = hcc_src / "arac_hcc_smoke_runner.py"
    spec = importlib.util.spec_from_file_location("arac_hcc_smoke_runner_for_test", runner_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hcc_smoke_runner_parses_arac_action_argument() -> None:
    runner = _load_runner_module()

    args = runner.parse_args(
        [
            "--functions",
            "elliptic",
            "--ids",
            "2",
            "--output-root",
            "out",
            "--seed",
            "1",
            "--max-fes",
            "2000",
            "--arac-action",
            "repair_shared_variable_binding",
        ]
    )

    assert args.arac_action == "repair_shared_variable_binding"
