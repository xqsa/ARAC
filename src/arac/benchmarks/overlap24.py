"""A matched 24-case benchmark for cross-suite overlap generalisation.

The suite deliberately keeps the optimisation surface identity-blind.  A
caller that runs ARAC receives only :class:`OptimizationProblem`; groups,
shared variables and the realised overlap graph are available only through
the explicit ``load_with_truth`` audit path.

The 24 cases are a balanced factorial design:

``4 base-function families × 6 structural recipes``

The six recipes contain both conforming and conflicting overlap, and cover
random, chain and star group topologies.  The same instance seed is reused by
the four base families for a given recipe.  This is intentional: it makes the
families matched controls rather than confounding a landscape change with a
different grouping or rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.benchmarks.overlap_objective import (
    ConflictMode,
    OverlapObjective,
    build_overlap_problem,
)


__all__ = [
    "OVERLAP24_BASE_FUNCTIONS",
    "OVERLAP24_CASE_SCHEMA",
    "OVERLAP24_DEFAULT_SEED",
    "OVERLAP24_DIMENSION",
    "OVERLAP24_NUM_GROUPS",
    "OVERLAP24_RECIPES",
    "Overlap24Benchmark",
    "Overlap24CaseSpec",
    "Overlap24Recipe",
    "build_overlap24_case",
    "build_overlap24_case_manifest",
    "overlap24_cases",
]


OVERLAP24_CASE_SCHEMA = "arac-overlap24-cross-suite-v1"
OVERLAP24_DEFAULT_SEED = 20260825
OVERLAP24_DIMENSION = 1000
OVERLAP24_NUM_GROUPS = 20
OVERLAP24_MIN_GROUP_SIZE = 45
OVERLAP24_MAX_GROUP_SIZE = 55
OVERLAP24_BOUNDS = 100.0
OVERLAP24_INTERACTION_STRENGTH = 0.10
OVERLAP24_BASE_FUNCTIONS: tuple[str, ...] = (
    "ackley",
    "elliptic",
    "rastrigin",
    "schwefel",
)

# The structural budgets are extra membership slots, not a count of distinct
# shared variables.  A budget of 40 is enough to realise every edge in a
# 20-group chain/star graph.  Random/6 is deliberately sparse, so its realised
# graph supplies a non-connected CTP regime while chain/star supply GCB cases.
OVERLAP24_RECIPES: tuple["Overlap24Recipe", ...]


@dataclass(frozen=True)
class Overlap24Recipe:
    """One structural row of the 4 × 6 suite design."""

    name: str
    conflict_mode: ConflictMode
    topology: str
    overlap_budget: int

    def __post_init__(self) -> None:
        if self.conflict_mode not in ("conforming", "conflicting"):
            raise ValueError("conflict_mode must be 'conforming' or 'conflicting'")
        if self.topology not in ("random", "chain", "star"):
            raise ValueError("topology must be 'random', 'chain' or 'star'")
        if isinstance(self.overlap_budget, bool) or self.overlap_budget <= 0:
            raise ValueError("overlap_budget must be a positive integer")


OVERLAP24_RECIPES = (
    Overlap24Recipe("conforming_chain", "conforming", "chain", 40),
    Overlap24Recipe("conforming_star", "conforming", "star", 40),
    Overlap24Recipe("conforming_random", "conforming", "random", 6),
    Overlap24Recipe("conflicting_chain", "conflicting", "chain", 40),
    Overlap24Recipe("conflicting_star", "conflicting", "star", 40),
    Overlap24Recipe("conflicting_random", "conflicting", "random", 6),
)


@dataclass(frozen=True)
class Overlap24CaseSpec:
    """Public, non-secret description of one fixed benchmark instance."""

    case_id: str
    base_function: str
    recipe: str
    conflict_mode: ConflictMode
    topology: str
    overlap_budget: int
    instance_seed: int

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if self.base_function not in OVERLAP24_BASE_FUNCTIONS:
            raise ValueError("base_function is not part of the overlap24 suite")
        if self.conflict_mode not in ("conforming", "conflicting"):
            raise ValueError("conflict_mode must be 'conforming' or 'conflicting'")
        if self.topology not in ("random", "chain", "star"):
            raise ValueError("topology must be 'random', 'chain' or 'star'")
        if isinstance(self.instance_seed, bool) or self.instance_seed < 0:
            raise ValueError("instance_seed must be a non-negative integer")


def _recipe_instance_seed(suite_seed: int, recipe: Overlap24Recipe) -> int:
    """Derive a stable seed shared by matched mode/family cases.

    Conforming and conflicting counterparts with the same topology and budget
    therefore share the exact grouping, while their local optima still differ.
    """

    topology_code = {"random": 0, "chain": 1, "star": 2}[recipe.topology]
    sequence = np.random.SeedSequence(
        [int(suite_seed), topology_code, int(recipe.overlap_budget)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def overlap24_cases(*, suite_seed: int = OVERLAP24_DEFAULT_SEED) -> tuple[Overlap24CaseSpec, ...]:
    """Return the fixed 24-case matrix without constructing objectives.

    ``suite_seed`` identifies the benchmark release.  It is an *instance*
    seed source; optimisation ``run_seed`` values are intentionally not part
    of this API and must be supplied to the algorithm separately.
    """

    if isinstance(suite_seed, bool) or not isinstance(suite_seed, int) or suite_seed < 0:
        raise ValueError("suite_seed must be a non-negative integer")

    cases: list[Overlap24CaseSpec] = []
    case_number = 1
    for base_function in OVERLAP24_BASE_FUNCTIONS:
        for recipe in OVERLAP24_RECIPES:
            cases.append(
                Overlap24CaseSpec(
                    case_id=f"O{case_number:02d}",
                    base_function=base_function,
                    recipe=recipe.name,
                    conflict_mode=recipe.conflict_mode,
                    topology=recipe.topology,
                    overlap_budget=recipe.overlap_budget,
                    instance_seed=_recipe_instance_seed(suite_seed, recipe),
                )
            )
            case_number += 1
    return tuple(cases)


def _connected_components(group_count: int, edges: set[tuple[int, int]]) -> int:
    adjacency = [set() for _ in range(group_count)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[int] = set()
    components = 0
    for start in range(group_count):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return components


def _truth_edges(objective: OverlapObjective) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for owners in objective.structure.membership:
        edges.update(tuple(sorted(pair)) for pair in combinations(owners, 2))
    return tuple(sorted(edges))


def build_overlap24_case_manifest(
    spec: Overlap24CaseSpec,
    objective: OverlapObjective,
    *,
    include_variable_ids: bool = False,
) -> dict[str, Any]:
    """Build an offline truth manifest from a constructed objective.

    The default manifest contains aggregate structure only.  Set
    ``include_variable_ids=True`` when writing a protected audit artifact that
    needs exact groups/shared-variable IDs; never pass that artifact to ARAC.
    """

    structure = objective.structure
    edges = _truth_edges(objective)
    component_count = _connected_components(len(structure.groups), set(edges))
    payload: dict[str, Any] = {
        "schema_version": OVERLAP24_CASE_SCHEMA,
        "case_id": spec.case_id,
        "base_function": spec.base_function,
        "recipe": spec.recipe,
        "conflict_mode": spec.conflict_mode,
        "topology": spec.topology,
        "overlap_budget": spec.overlap_budget,
        "instance_seed": spec.instance_seed,
        "dimension": structure.grouping.dimension,
        "group_count": len(structure.groups),
        "group_sizes": list(structure.group_sizes),
        "overlap_slots": int(sum(structure.group_sizes) - structure.grouping.dimension),
        "shared_variable_count": len(structure.shared_variables),
        "graph_edges": [list(edge) for edge in edges],
        "component_count": component_count,
        "graph_connected": component_count == 1,
        "optimum": float(objective.optimum),
        "optimum_is_attainable": bool(objective.optimum_is_attainable),
    }
    if include_variable_ids:
        payload["groups"] = [list(group) for group in structure.groups]
        payload["shared_variables"] = list(structure.shared_variables)
    return payload


class Overlap24Benchmark:
    """Load one case from the cross-suite 24-function benchmark."""

    def __init__(
        self,
        *,
        suite_seed: int = OVERLAP24_DEFAULT_SEED,
        dimension: int = OVERLAP24_DIMENSION,
        num_groups: int = OVERLAP24_NUM_GROUPS,
        min_group_size: int = OVERLAP24_MIN_GROUP_SIZE,
        max_group_size: int = OVERLAP24_MAX_GROUP_SIZE,
        bounds: float = OVERLAP24_BOUNDS,
        rotation: bool = True,
        transforms: bool = True,
        interaction_strength: float = OVERLAP24_INTERACTION_STRENGTH,
        contiguous: bool = False,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups < 2:
            raise ValueError("num_groups must be an integer >= 2")
        if isinstance(min_group_size, bool) or not isinstance(min_group_size, int) or min_group_size <= 0:
            raise ValueError("min_group_size must be a positive integer")
        if isinstance(max_group_size, bool) or not isinstance(max_group_size, int):
            raise ValueError("max_group_size must be an integer")
        if max_group_size < min_group_size:
            raise ValueError("max_group_size must be >= min_group_size")
        if num_groups * min_group_size > dimension or num_groups * max_group_size < dimension:
            raise ValueError("group size range cannot cover the requested dimension")
        if isinstance(suite_seed, bool) or not isinstance(suite_seed, int) or suite_seed < 0:
            raise ValueError("suite_seed must be a non-negative integer")

        self._suite_seed = suite_seed
        self._dimension = dimension
        self._num_groups = num_groups
        self._min_group_size = min_group_size
        self._max_group_size = max_group_size
        self._bounds = float(bounds)
        self._rotation = bool(rotation)
        self._transforms = bool(transforms)
        self._interaction_strength = float(interaction_strength)
        self._contiguous = bool(contiguous)
        self._cases = overlap24_cases(suite_seed=suite_seed)
        self._by_id = {case.case_id: case for case in self._cases}
        self._recipe_by_name = {recipe.name: recipe for recipe in OVERLAP24_RECIPES}

        if max(recipe.overlap_budget for recipe in OVERLAP24_RECIPES) >= dimension:
            raise ValueError("overlap budgets must be smaller than dimension")

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self._cases)

    @property
    def cases(self) -> tuple[Overlap24CaseSpec, ...]:
        return self._cases

    def spec(self, case_id: str) -> Overlap24CaseSpec:
        normalized = str(case_id).strip().upper()
        try:
            return self._by_id[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown overlap24 case_id: {case_id!r}") from exc

    def load(self, case_id: str) -> OptimizationProblem:
        """Return only the identity-free problem consumed by an algorithm."""

        problem, _truth, _spec = self.load_with_truth(case_id)
        return problem

    def load_with_truth(
        self, case_id: str
    ) -> tuple[OptimizationProblem, OverlapObjective, Overlap24CaseSpec]:
        spec = self.spec(case_id)
        recipe = self._recipe_by_name[spec.recipe]
        problem, truth = build_overlap_problem(
            self._dimension,
            overlap_budget=recipe.overlap_budget,
            min_group_size=self._min_group_size,
            max_group_size=self._max_group_size,
            num_groups=self._num_groups,
            base_function=spec.base_function,
            conflict_mode=recipe.conflict_mode,
            bounds=self._bounds,
            contiguous=self._contiguous,
            rotation=self._rotation,
            transforms=self._transforms,
            seed=spec.instance_seed,
            topology=recipe.topology,
            interaction_strength=self._interaction_strength,
        )
        return problem, truth, spec

    def truth_manifest(
        self, case_id: str, *, include_variable_ids: bool = False
    ) -> dict[str, Any]:
        _problem, truth, spec = self.load_with_truth(case_id)
        return build_overlap24_case_manifest(
            spec, truth, include_variable_ids=include_variable_ids
        )

    def manifest(self, *, include_variable_ids: bool = False) -> tuple[dict[str, Any], ...]:
        """Return one offline manifest row per case in case-ID order."""

        return tuple(
            self.truth_manifest(case.case_id, include_variable_ids=include_variable_ids)
            for case in self._cases
        )


def build_overlap24_case(
    case_id: str,
    *,
    suite_seed: int = OVERLAP24_DEFAULT_SEED,
    **benchmark_options: Any,
) -> tuple[OptimizationProblem, OverlapObjective, Overlap24CaseSpec]:
    """Convenience wrapper around :class:`Overlap24Benchmark.load_with_truth`."""

    return Overlap24Benchmark(
        suite_seed=suite_seed, **benchmark_options
    ).load_with_truth(case_id)
