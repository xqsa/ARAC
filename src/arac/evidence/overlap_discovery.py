"""Counted black-box discovery of variable-level overlap evidence."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.overlap_adapter import Phase1OverlapEvidence
from arac.runtime.ledger import EvaluationLedger


DEFAULT_EDGE_THRESHOLD = 1.0e-15
DEFAULT_MIN_SUPPORT = 1.0


@dataclass(frozen=True)
class OverlapDiscoveryResult:
    """Auditable interaction graph and inferred variable memberships."""

    evidence: Phase1OverlapEvidence
    edge_scores: tuple[tuple[int, int, float, float], ...]
    consumed_fes: int
    anchor_count: int

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((left, right) for left, right, _, _ in self.edge_scores)

    @property
    def identifiable(self) -> bool:
        """Whether the observed graph contains at least one nontrivial edge."""

        return bool(self.edges)


def _validate_anchor(
    problem: OptimizationProblem,
    anchor: Sequence[float],
) -> np.ndarray:
    point = np.asarray(anchor, dtype=float)
    if point.shape != (problem.dimension,) or not np.all(np.isfinite(point)):
        raise ValueError("anchor must be a finite vector matching the problem dimension")
    if np.any(point < problem.lower_array) or np.any(point > problem.upper_array):
        raise ValueError("anchor must stay inside the problem bounds")
    return point


def _maximal_cliques(
    dimension: int,
    edges: set[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    """Return maximal cliques in deterministic lexicographic order."""

    adjacency = {variable: set() for variable in range(dimension)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    found: set[tuple[int, ...]] = set()

    def visit(
        current: tuple[int, ...],
        candidates: set[int],
        excluded: set[int],
    ) -> None:
        if not candidates and not excluded:
            if current:
                found.add(tuple(sorted(current)))
            return
        for variable in sorted(tuple(candidates)):
            visit(
                (*current, variable),
                candidates.intersection(adjacency[variable]),
                excluded.intersection(adjacency[variable]),
            )
            candidates.remove(variable)
            excluded.add(variable)

    visit((), set(range(dimension)), set())
    return tuple(sorted(found, key=lambda clique: (clique[0], len(clique), clique)))


def _candidate_matrix(
    anchor: np.ndarray,
    step: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    dimension = anchor.size
    rows = [anchor.copy()]
    for variable in range(dimension):
        candidate = anchor.copy()
        candidate[variable] += step[variable]
        rows.append(candidate)
    pairs: list[tuple[int, int]] = []
    for left in range(dimension):
        for right in range(left + 1, dimension):
            candidate = anchor.copy()
            candidate[left] += step[left]
            candidate[right] += step[right]
            rows.append(candidate)
            pairs.append((left, right))
    return np.asarray(rows, dtype=float), tuple(pairs)


def discover_overlap(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    anchors: Iterable[Sequence[float]],
    step: float | Sequence[float],
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    min_support: float = DEFAULT_MIN_SUPPORT,
) -> OverlapDiscoveryResult:
    """Infer overlap groups from repeated two-variable mixed differences.

    The probe consumes ``A * (1 + d + d*(d-1)/2)`` evaluations for ``A`` anchors.
    It only emits an edge when its normalized mixed difference exceeds
    ``edge_threshold`` on at least ``min_support`` of the anchors.  Maximal
    cliques of the resulting graph become inferred groups; clique intersections
    are the inferred shared-variable memberships.
    """

    if not isinstance(ledger, EvaluationLedger) or ledger.problem is not problem:
        raise ValueError("overlap discovery requires the ledger for the same problem")
    if not math.isfinite(float(edge_threshold)) or edge_threshold <= 0.0:
        raise ValueError("edge_threshold must be finite and positive")
    if not math.isfinite(float(min_support)) or not 0.0 < min_support <= 1.0:
        raise ValueError("min_support must be in (0, 1]")
    if isinstance(step, bool):
        raise ValueError("step must be positive")
    if np.isscalar(step):
        step_vector = np.full(problem.dimension, float(step))
    else:
        step_vector = np.asarray(step, dtype=float)
    if step_vector.shape != (problem.dimension,) or not np.all(np.isfinite(step_vector)):
        raise ValueError("step must be a finite scalar or dimension-sized vector")
    if np.any(step_vector <= 0.0):
        raise ValueError("step must be positive")

    anchor_points = tuple(_validate_anchor(problem, anchor) for anchor in anchors)
    if not anchor_points:
        raise ValueError("at least one anchor is required")
    scores_by_pair: dict[tuple[int, int], list[float]] = {}
    started = ledger.count
    for anchor in anchor_points:
        candidates, pairs = _candidate_matrix(anchor, step_vector)
        if np.any(candidates < problem.lower_array) or np.any(candidates > problem.upper_array):
            raise ValueError("probe step escaped the problem bounds")
        values = np.asarray(ledger.evaluate(candidates), dtype=float)
        base = float(values[0])
        singles = values[1 : 1 + problem.dimension]
        pair_values = values[1 + problem.dimension :]
        for index, (left, right) in enumerate(pairs):
            joint = float(pair_values[index])
            scale = abs(base) + abs(float(singles[left])) + abs(float(singles[right])) + abs(joint) + 1.0
            mixed = abs(joint - float(singles[left]) - float(singles[right]) + base) / scale
            scores_by_pair.setdefault((left, right), []).append(float(mixed))

    edges: set[tuple[int, int]] = set()
    edge_rows: list[tuple[int, int, float, float]] = []
    for pair in sorted(scores_by_pair):
        scores = np.asarray(scores_by_pair[pair], dtype=float)
        support = float(np.mean(scores > edge_threshold))
        strength = float(np.median(scores))
        if strength > edge_threshold and support >= min_support:
            edges.add(pair)
            edge_rows.append((pair[0], pair[1], strength, support))

    groups = list(_maximal_cliques(problem.dimension, edges))
    covered = {variable for group in groups for variable in group}
    groups.extend((variable,) for variable in range(problem.dimension) if variable not in covered)
    groups = sorted(set(groups), key=lambda group: (group[0], len(group), group))
    memberships = tuple(
        tuple(group_index for group_index, group in enumerate(groups) if variable in group)
        for variable in range(problem.dimension)
    )
    support_lookup = {(left, right): support for left, right, _, support in edge_rows}
    confidences: list[tuple[int, int, float]] = []
    for variable, owners in enumerate(memberships):
        for group_index in owners:
            group = groups[group_index]
            incident = [
                support_lookup[tuple(sorted((variable, other)))]
                for other in group
                if other != variable
            ]
            confidence = min(incident, default=1.0)
            confidences.append((variable, group_index, float(confidence)))
    evidence = Phase1OverlapEvidence(
        dimension=problem.dimension,
        groups=tuple(groups),
        memberships=memberships,
        membership_confidences=tuple(confidences),
        complete=True,
    )
    return OverlapDiscoveryResult(
        evidence=evidence,
        edge_scores=tuple(edge_rows),
        consumed_fes=ledger.count - started,
        anchor_count=len(anchor_points),
    )


__all__ = [
    "DEFAULT_EDGE_THRESHOLD",
    "DEFAULT_MIN_SUPPORT",
    "OverlapDiscoveryResult",
    "discover_overlap",
]
