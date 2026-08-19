"""Narrow adapter for the third-party AOB benchmark suite."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys

import numpy as np


_FAMILY_TO_OBJECTIVE = {
    "A": "ackley",
    "E": "elliptic",
    "R": "rastrigin",
    "S": "schwefel",
}


@dataclass(frozen=True)
class OptimizationProblem:
    """Identity-free numeric problem passed from a benchmark into ARAC."""

    objective: Callable[[np.ndarray], float | np.ndarray]
    dimension: int
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    optimum: float = 0.0

    def __post_init__(self) -> None:
        if not callable(self.objective):
            raise TypeError("objective must be callable")
        if isinstance(self.dimension, bool) or self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if len(self.lower_bounds) != self.dimension or len(self.upper_bounds) != self.dimension:
            raise ValueError("bounds must match the problem dimension")
        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("bounds must be finite")
        if np.any(lower >= upper):
            raise ValueError("every lower bound must be smaller than its upper bound")
        if not math.isfinite(float(self.optimum)):
            raise ValueError("optimum must be finite")

    @property
    def lower_array(self) -> np.ndarray:
        return np.asarray(self.lower_bounds, dtype=float)

    @property
    def upper_array(self) -> np.ndarray:
        return np.asarray(self.upper_bounds, dtype=float)


class AobBenchmark:
    """Load AOB cases while exposing only their public numeric optimization surface."""

    def __init__(
        self,
        *,
        vendor_root: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        self._vendor_root = (
            repository_root / "vendor" / "aob"
            if vendor_root is None
            else Path(vendor_root)
        ).resolve()
        self._data_root = (
            self._vendor_root / "AOB" / "AOBG" / "datafile"
            if data_root is None
            else Path(data_root)
        ).resolve()
        if not (self._vendor_root / "AOB" / "AOB.py").is_file():
            raise FileNotFoundError(f"AOB implementation is missing: {self._vendor_root}")
        if not self._data_root.is_dir():
            raise FileNotFoundError(f"AOB data root is missing: {self._data_root}")

    @property
    def data_root(self) -> Path:
        """Return the data location for provenance and input-integrity audits only."""

        return self._data_root

    def load(self, case_id: str, *, output_directory: Path | None = None) -> OptimizationProblem:
        normalized = str(case_id).strip().upper()
        if len(normalized) < 2 or normalized[0] not in _FAMILY_TO_OBJECTIVE:
            raise ValueError("case_id must be one AOB ID such as A1")
        if not normalized[1:].isdigit() or int(normalized[1:]) not in range(1, 7):
            raise ValueError("AOB function ID must be in 1..6")

        vendor_path = str(self._vendor_root)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        benchmark_type = importlib.import_module("AOB.AOB").Benchmark
        output = None if output_directory is None else str(Path(output_directory).resolve())
        benchmark = benchmark_type(output, data_dir=self._data_root)
        objective = benchmark.get_function(
            _FAMILY_TO_OBJECTIVE[normalized[0]],
            int(normalized[1:]),
        )
        info = objective.info()
        dimension = int(info["dimension"])
        lower = float(info["lower"])
        upper = float(info["upper"])
        optimum = float(info.get("best", 0.0))
        return OptimizationProblem(
            objective=objective,
            dimension=dimension,
            lower_bounds=(lower,) * dimension,
            upper_bounds=(upper,) * dimension,
            optimum=optimum,
        )


__all__ = ["AobBenchmark", "OptimizationProblem"]
