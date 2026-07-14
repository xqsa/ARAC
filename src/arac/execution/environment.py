"""Pinned numerical environment contract for auditable HCC execution."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable


PINNED_HCC_RUNTIME_ENVIRONMENT = {
    "python": "3.12.13",
    "numpy": "2.3.5",
    "matplotlib": "3.11.0",
    "PyYAML": "6.0.3",
    "scipy": "1.18.0",
    "torch": "2.12.1",
    "cma": "4.4.4",
    "blas_name": "scipy-openblas",
    "blas_version": "0.3.30",
    "pythonhashseed": "0",
    "omp_num_threads": "1",
    "openblas_num_threads": "1",
    "mkl_num_threads": "1",
    "numexpr_num_threads": "1",
}
EnvironmentProbe = Callable[[str], dict[str, str]]

_ENVIRONMENT_PROBE_SOURCE = """
import importlib.metadata as metadata
import json
import os
import platform

import numpy as np

blas = getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas", {})
print(json.dumps({
    "python": platform.python_version(),
    "numpy": metadata.version("numpy"),
    "matplotlib": metadata.version("matplotlib"),
    "PyYAML": metadata.version("PyYAML"),
    "scipy": metadata.version("scipy"),
    "torch": metadata.version("torch"),
    "cma": metadata.version("cma"),
    "blas_name": str(blas.get("name", "missing")),
    "blas_version": str(blas.get("version", "missing")),
    "pythonhashseed": os.environ.get("PYTHONHASHSEED", "missing"),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "missing"),
    "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "missing"),
    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "missing"),
    "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS", "missing"),
}, sort_keys=True))
"""


def probe_python_environment(python_executable: str) -> dict[str, str]:
    try:
        completed = subprocess.run(
            [python_executable, "-c", _ENVIRONMENT_PROBE_SOURCE],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        observed = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"unable to audit HCC runtime environment via {python_executable}: {exc}"
        ) from exc
    if not isinstance(observed, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in observed.items()
    ):
        raise RuntimeError("HCC runtime environment probe returned an invalid payload")
    return observed


def hcc_runtime_environment_failures(observed: dict[str, str]) -> list[str]:
    return [
        f"{name}:expected={expected},observed={observed.get(name, 'missing')}"
        for name, expected in PINNED_HCC_RUNTIME_ENVIRONMENT.items()
        if observed.get(name) != expected
    ]


def require_pinned_hcc_runtime_environment(
    python_executable: str,
    *,
    environment_probe: EnvironmentProbe | None = None,
) -> dict[str, str]:
    observed = (environment_probe or probe_python_environment)(python_executable)
    failures = hcc_runtime_environment_failures(observed)
    if failures:
        raise RuntimeError(
            "pinned HCC runtime environment gate failed: " + ";".join(failures)
        )
    return observed
