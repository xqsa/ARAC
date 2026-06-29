from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def test_optimizer_evaluate_fitness_clips_batch_to_remaining_budget() -> None:
    hcc_src = Path(__file__).resolve().parents[1] / "HCC_SRC"
    sys.path.insert(0, str(hcc_src))
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
