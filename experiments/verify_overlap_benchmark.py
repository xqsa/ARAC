"""Smoke test for the synthetic overlapping-variable benchmark.

Verifies the three properties that must hold before the suite can be trusted:

1. The generated decomposition actually overlaps: at least one variable
   belongs to more than one group, and the overlap budget is honoured.
2. ``conforming`` instances have a reachable global optimum (f(x*) = 0),
   confirming the per-group optima agree on shared variables.
3. ``conflicting`` instances have *no* reachable zero, and the shared
   variables genuinely disagree across their owning groups.

Run::

    .venv/Scripts/python.exe experiments/verify_overlap_benchmark.py
"""

from __future__ import annotations

import numpy as np

from arac.benchmarks.overlap_groups import generate_overlap_groups
from arac.benchmarks.overlap_objective import BASE_FUNCTIONS, build_overlap_problem


def _line(label: str, value: str | float) -> None:
    if isinstance(value, float):
        value = f"{value:.6e}"
    print(f"  {label:<36s} {value}")


def _check_overlap_structure() -> None:
    print("[1] overlap structure")
    for budget in (5, 10, 20):
        structure = generate_overlap_groups(
            dimension=100,
            overlap_budget=budget,
            min_group_size=2,
            max_group_size=8,
            contiguous=False,
            seed=42,
        )
        shared = structure.shared_variables
        total_slots = sum(structure.group_sizes)
        extra_slots = total_slots - structure.grouping.dimension
        _line(f"budget={budget} groups", len(structure.groups))
        _line(f"budget={budget} total slots", total_slots)
        _line(f"budget={budget} shared vars", len(shared))
        _line(f"budget={budget} extra slots", extra_slots)
        assert len(shared) > 0, "no shared variables generated"
        assert extra_slots == budget, "overlap budget is not exact"
        for variable in shared:
            assert len(structure.membership[variable]) >= 2
    print("  OK\n")


def _check_topologies_and_batch_objective() -> None:
    print("[5] topology controls and batch objective")
    for topology in ("random", "chain", "star"):
        problem, objective = build_overlap_problem(
            dimension=100,
            overlap_budget=12,
            min_group_size=3,
            max_group_size=8,
            base_function="sphere",
            conflict_mode="conforming",
            contiguous=False,
            topology=topology,
            rotation=True,
            transforms=True,
            seed=13,
        )
        points = np.random.default_rng(4).uniform(-10.0, 10.0, size=(3, 100))
        batch = np.asarray(problem.objective(points), dtype=float)
        scalar = np.asarray([problem.objective(point) for point in points], dtype=float)
        _line(f"{topology} shared vars", len(objective.structure.shared_variables))
        assert sum(objective.structure.group_sizes) - 100 == 12
        np.testing.assert_allclose(batch, scalar, rtol=1.0e-12, atol=1.0e-8)

        edges = {
            (left, right)
            for owners in objective.structure.membership
            for left in owners
            for right in owners
            if left < right
        }
        if topology == "chain":
            assert all(right == left + 1 for left, right in edges)
        elif topology == "star":
            assert all(left == 0 for left, _ in edges)
    print("  OK\n")


def _check_conforming_optimum() -> None:
    print("[2] conforming optimum reachable (f(x*) = 0)")
    for base in BASE_FUNCTIONS:
        problem, objective = build_overlap_problem(
            dimension=120,
            overlap_budget=12,
            min_group_size=3,
            max_group_size=9,
            base_function=base,
            conflict_mode="conforming",
            bounds=100.0,
            seed=7,
        )
        optimum_point = objective.optimum_point()
        value = problem.objective(optimum_point)
        _line(base, value)
        assert value < 1.0e-6, f"conforming {base} optimum not zero: {value}"
    print("  OK\n")


def _check_conflicting_disagreement() -> None:
    print("[3] conflicting optimum unreachable + shared vars disagree")
    for base in BASE_FUNCTIONS:
        problem, objective = build_overlap_problem(
            dimension=120,
            overlap_budget=12,
            min_group_size=3,
            max_group_size=9,
            base_function=base,
            conflict_mode="conflicting",
            bounds=100.0,
            seed=7,
        )
        heaviest_point = objective.optimum_point()
        residual = problem.objective(heaviest_point)
        if not np.isfinite(residual):
            residual = float("inf")
        _line(f"{base} residual at heaviest-group optimum", residual)
        assert residual > 1.0e-3, f"conflicting {base} unexpectedly solved: {residual}"

        shared = objective.structure.shared_variables
        disagreements = 0
        for variable in shared:
            owners = objective.structure.membership[variable]
            values = []
            for group in owners:
                group_members = list(objective.structure.groups[group])
                local_index = group_members.index(variable)
                values.append(float(objective._optima[group][local_index]))
            if max(values) - min(values) > 1.0e-9:
                disagreements += 1
        _line(f"{base} shared vars with disagreeing optima", f"{disagreements}/{len(shared)}")
        assert disagreements > 0, f"conflicting {base} has no disagreeing shared variable"
    print("  OK\n")


def _check_group_contribution_audit() -> None:
    print("[4] per-group contribution decomposes the objective")
    problem, objective = build_overlap_problem(
        dimension=100,
        overlap_budget=10,
        min_group_size=3,
        max_group_size=8,
        base_function="rastrigin",
        conflict_mode="conflicting",
        seed=11,
    )
    rng = np.random.default_rng(3)
    point = rng.uniform(-50.0, 50.0, size=problem.dimension)
    total = problem.objective(point)
    contributions = objective.per_group_contribution(point)
    _line("f(x)", total)
    _line("Σ contributions", float(contributions.sum()))
    _line("max group contribution", float(contributions.max()))
    scale = max(1.0, abs(total))
    assert abs(total - float(contributions.sum())) < 1.0e-6 * scale
    print("  OK\n")


def main() -> None:
    _check_overlap_structure()
    _check_conforming_optimum()
    _check_conflicting_disagreement()
    _check_group_contribution_audit()
    _check_topologies_and_batch_objective()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
