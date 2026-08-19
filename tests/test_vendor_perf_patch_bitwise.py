"""Permanent bitwise guard for the AOB vendor performance patch.

The 2026-08-17 patch (vendor/aob/AOB) precomputes hot-loop constants and
gates fitness_record behind a flag.  It must never change results: this
test replays a fixed-seed batch stream through every gate case and every
batch shape and asserts bitwise equality with the frozen golden output
(artifacts/vendor_perf_patch_20260817/golden_reference.npy).

The replay must mirror the golden generation exactly: ONE rng stream
shared across the cases in fixed order (R2, A3, S5, R6), each case
drawing the (1, 8, 24)-row batches and the extra single-row call.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

# ruff: noqa: E402
# Thread caps must be set before NumPy imports.

from pathlib import Path

import numpy as np

from arac.benchmarks.aob import AobBenchmark

GOLDEN_PATH = Path("artifacts/vendor_perf_patch_20260817/golden_reference.npy")
CASES = ("R2", "A3", "S5", "R6")
GOLDEN_SEED = 20260817


def _replay_all() -> dict[str, list[np.ndarray]]:
    rng = np.random.default_rng(GOLDEN_SEED)
    bench = AobBenchmark()
    outputs: dict[str, list[np.ndarray]] = {}
    for case in CASES:
        problem = bench.load(case)
        case_outputs = []
        for rows in (1, 8, 24):
            batch = rng.uniform(
                problem.lower_bounds[0], problem.upper_bounds[0], size=(rows, problem.dimension)
            )
            case_outputs.append(np.asarray(problem.objective(batch), dtype=float).copy())
            if rows == 1:
                case_outputs.append(np.asarray(problem.objective(batch[0]), dtype=float).copy())
        outputs[case] = case_outputs
    return outputs


def test_golden_reference_exists() -> None:
    assert GOLDEN_PATH.exists(), "golden reference is part of the patch evidence"


def test_vendor_objective_is_bitwise_identical_to_golden() -> None:
    golden = np.load(GOLDEN_PATH, allow_pickle=True).item()
    outputs = _replay_all()
    assert set(outputs) == set(golden)
    for case in CASES:
        actual_list = outputs[case]
        expected_list = golden[case]
        assert len(actual_list) == len(expected_list)
        for index, (actual, expected) in enumerate(zip(actual_list, expected_list)):
            assert np.array_equal(actual, expected), (
                f"{case} output {index} drifted from the pre-patch golden "
                f"(max diff {np.max(np.abs(actual - expected)) if actual.shape == expected.shape else 'shape'})"
            )


def test_fitness_record_disabled_by_default() -> None:
    problem = AobBenchmark().load("A3")
    rng = np.random.default_rng(0)
    batch = rng.uniform(-100, 100, size=(4, problem.dimension))
    problem.objective(batch)
    # The adapter exposes the vendor function instance as the objective.
    assert problem.objective.fitness_record == [], (
        "fitness_record must stay empty unless record_fitness is explicitly enabled"
    )
