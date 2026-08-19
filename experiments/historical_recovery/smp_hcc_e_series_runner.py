"""Expose EXP-052 SMP to all six elliptic IDs without changing its runner."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TARGET_RUNNER = (
    REPOSITORY_ROOT
    / ".codex-tasks"
    / "historical-level-recovery"
    / "raw"
    / "replay-tree-candidate-v1"
    / "scripts"
    / "hcc_smoke_runner.py"
)
TARGET_RUNNER_SHA256 = "b17021e8ffe1de76fea48b52ed3c00a62b4cc93bf4c2c759604064d14ebc68ac"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_target() -> ModuleType:
    if _sha256(TARGET_RUNNER) != TARGET_RUNNER_SHA256:
        raise ValueError("exact EXP-052 runner hash drifted")
    spec = importlib.util.spec_from_file_location("_arac_exp052_smp_e_series_target", TARGET_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load exact EXP-052 runner: {TARGET_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    expected_pairs = frozenset(
        {
            ("elliptic", 1),
            ("elliptic", 3),
            *(
                (function, fun_id)
                for function in ("ackley", "rastrigin", "schwefel")
                for fun_id in range(1, 7)
            ),
        }
    )
    if module.ACTIVE_FUNCTION_ID_PAIRS != expected_pairs:
        raise ValueError("EXP-052 active function pairs drifted")
    module.ACTIVE_FUNCTION_ID_PAIRS = frozenset(
        {
            *expected_pairs,
            *(("elliptic", fun_id) for fun_id in range(1, 7)),
        }
    )
    return module


def main() -> int:
    _load_target().main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
