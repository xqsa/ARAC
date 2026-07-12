from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def test_optimizer_evaluate_fitness_clips_batch_to_remaining_budget() -> None:
    vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "hcc"
    sys.path.insert(0, str(vendor_root))
    from HCC.OPT.CMAES.optimizer import Optimizer

    seen_shapes: list[tuple[int, ...]] = []

    def fitness(x_batch):
        seen_shapes.append(np.asarray(x_batch).shape)
        return np.arange(len(x_batch), dtype=float)

    optimizer = Optimizer(
        {"fitness_function": fitness, "ndim_problem": 2},
        {"max_function_evaluations": 3, "n_function_evaluations": 2},
    )

    y = optimizer._evaluate_fitness(np.zeros((5, 2)))

    assert seen_shapes == [(1, 2)]
    assert y.tolist() == [0.0]
    assert optimizer.n_function_evaluations == 3
