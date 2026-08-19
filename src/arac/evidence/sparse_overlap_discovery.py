"""Sparse, fail-closed variable-level overlap discovery."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.overlap_adapter import Phase1OverlapEvidence
from arac.evidence.overlap_discovery import (
    DEFAULT_EDGE_THRESHOLD,
    _maximal_cliques,
)
from arac.runtime.ledger import EvaluationLedger


DEFAULT_SPARSE_ROUNDS = 8
DEFAULT_BUCKET_SIZE = 8
DEFAULT_MAX_CANDIDATE_PAIRS = 4096
DEFAULT_SPARSE_MIN_SUPPORT = 1.0


@dataclass(frozen=True)
class SparseOverlapDiscoveryResult:
    """Auditable result of bucket screening and candidate refinement."""

    evidence: Phase1OverlapEvidence
    screened_bucket_edges: tuple[tuple[int, int, float, float], ...]
    refined_edge_scores: tuple[tuple[int, int, float, float], ...]
    consumed_fes: int
    expected_fes: int
    separated_pair_fraction: float
    candidate_pair_count: int
    complete_reason: str

    @property
    def complete(self) -> bool:
        return self.evidence.complete

    @property
    def inferred_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((left, right) for left, right, _, _ in self.refined_edge_scores)


def _validate_anchor(problem: OptimizationProblem, anchor: Sequence[float]) -> np.ndarray:
    point = np.asarray(anchor, dtype=float)
    if point.shape != (problem.dimension,) or not np.all(np.isfinite(point)):
        raise ValueError("anchor must be a finite vector matching the problem dimension")
    if np.any(point < problem.lower_array) or np.any(point > problem.upper_array):
        raise ValueError("anchor must stay inside the problem bounds")
    return point


def _singleton_evidence(dimension: int, *, complete: bool) -> Phase1OverlapEvidence:
    groups = tuple((variable,) for variable in range(dimension))
    memberships = tuple((variable,) for variable in range(dimension))
    confidences = tuple((variable, variable, 1.0) for variable in range(dimension))
    return Phase1OverlapEvidence(
        dimension=dimension,
        groups=groups,
        memberships=memberships,
        membership_confidences=confidences,
        complete=complete,
    )


def _bucket_layout(
    dimension: int,
    bucket_size: int,
    *,
    seed: int,
    round_index: int,
) -> tuple[tuple[int, ...], ...]:
    rng = np.random.default_rng(seed ^ 0xB4C6_2026 ^ (round_index * 0x9E3779B1))
    permutation = rng.permutation(dimension)
    return tuple(
        tuple(sorted(int(variable) for variable in chunk))
        for chunk in np.array_split(permutation, math.ceil(dimension / bucket_size))
    )


def _screen_candidates(
    anchor: np.ndarray,
    buckets: tuple[tuple[int, ...], ...],
    step: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    rows = [anchor.copy()]
    for bucket in buckets:
        candidate = anchor.copy()
        candidate[list(bucket)] += step[list(bucket)]
        rows.append(candidate)
    pairs: list[tuple[int, int]] = []
    for left in range(len(buckets)):
        for right in range(left + 1, len(buckets)):
            candidate = anchor.copy()
            candidate[list(buckets[left])] += step[list(buckets[left])]
            candidate[list(buckets[right])] += step[list(buckets[right])]
            rows.append(candidate)
            pairs.append((left, right))
    return np.asarray(rows, dtype=float), tuple(pairs)


def _mixed_score(base: float, left: float, right: float, joint: float) -> float:
    scale = abs(base) + abs(left) + abs(right) + abs(joint) + 1.0
    return abs(joint - left - right + base) / scale


def _build_evidence(
    dimension: int,
    edges: set[tuple[int, int]],
    supports: dict[tuple[int, int], float],
) -> Phase1OverlapEvidence:
    groups = list(_maximal_cliques(dimension, edges))
    covered = {variable for group in groups for variable in group}
    groups.extend((variable,) for variable in range(dimension) if variable not in covered)
    groups = sorted(set(groups), key=lambda group: (group[0], len(group), group))
    memberships = tuple(
        tuple(group for group, variables in enumerate(groups) if variable in variables)
        for variable in range(dimension)
    )
    confidences: list[tuple[int, int, float]] = []
    for variable, owners in enumerate(memberships):
        for owner in owners:
            group = groups[owner]
            incident = [
                supports[tuple(sorted((variable, other)))]
                for other in group
                if other != variable
            ]
            confidences.append((variable, owner, min(incident, default=1.0)))
    return Phase1OverlapEvidence(
        dimension=dimension,
        groups=tuple(groups),
        memberships=memberships,
        membership_confidences=tuple(confidences),
        complete=True,
    )


def discover_overlap_sparse(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    anchors: Iterable[Sequence[float]],
    step: float | Sequence[float],
    run_seed: int = 0,
    rounds: int = DEFAULT_SPARSE_ROUNDS,
    bucket_size: int = DEFAULT_BUCKET_SIZE,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    min_support: float = DEFAULT_SPARSE_MIN_SUPPORT,
) -> SparseOverlapDiscoveryResult:
    """Discover overlap with bucket screening and fail closed on incomplete evidence."""

    if not isinstance(ledger, EvaluationLedger) or ledger.problem is not problem:
        raise ValueError("sparse discovery requires the ledger for the same problem")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    if isinstance(bucket_size, bool) or not isinstance(bucket_size, int) or bucket_size <= 0:
        raise ValueError("bucket_size must be a positive integer")
    if isinstance(max_candidate_pairs, bool) or not isinstance(max_candidate_pairs, int) or max_candidate_pairs <= 0:
        raise ValueError("max_candidate_pairs must be a positive integer")
    if not math.isfinite(float(edge_threshold)) or edge_threshold <= 0.0:
        raise ValueError("edge_threshold must be finite and positive")
    if not math.isfinite(float(min_support)) or not 0.0 < min_support <= 1.0:
        raise ValueError("min_support must be in (0, 1]")
    if isinstance(step, bool):
        raise ValueError("step must be positive")
    step_vector = np.full(problem.dimension, float(step)) if np.isscalar(step) else np.asarray(step, dtype=float)
    if step_vector.shape != (problem.dimension,) or not np.all(np.isfinite(step_vector)):
        raise ValueError("step must be a finite scalar or dimension-sized vector")
    if np.any(step_vector <= 0.0):
        raise ValueError("step must be positive")
    anchor_points = tuple(_validate_anchor(problem, anchor) for anchor in anchors)
    if not anchor_points:
        raise ValueError("at least one anchor is required")

    bucket_count = math.ceil(problem.dimension / bucket_size)
    screen_per_anchor = 1 + bucket_count + bucket_count * (bucket_count - 1) // 2
    expected_screen_fes = len(anchor_points) * rounds * screen_per_anchor
    expected_fes = expected_screen_fes
    if ledger.remaining < expected_screen_fes:
        return SparseOverlapDiscoveryResult(
            evidence=_singleton_evidence(problem.dimension, complete=False),
            screened_bucket_edges=(),
            refined_edge_scores=(),
            consumed_fes=0,
            expected_fes=expected_fes,
            separated_pair_fraction=0.0,
            candidate_pair_count=0,
            complete_reason="budget_insufficient_for_screening",
        )

    started = ledger.count
    bucket_scores: dict[tuple[int, int], list[float]] = {}
    separated_pairs: set[tuple[int, int]] = set()
    candidate_seen: dict[tuple[int, int], int] = {}
    candidate_hits: dict[tuple[int, int], int] = {}
    for anchor_index, anchor in enumerate(anchor_points):
        for round_index in range(rounds):
            buckets = _bucket_layout(
                problem.dimension,
                bucket_size,
                seed=int(run_seed) ^ (anchor_index * 0x45D9F3B) ^ 0xA17E5D,
                round_index=round_index,
            )
            for left in range(len(buckets)):
                for right in range(left + 1, len(buckets)):
                    separated_pairs.update(
                        (min(variable, other), max(variable, other))
                        for variable in buckets[left]
                        for other in buckets[right]
                    )
            candidates, bucket_pairs = _screen_candidates(anchor, buckets, step_vector)
            if np.any(candidates < problem.lower_array) or np.any(candidates > problem.upper_array):
                raise ValueError("probe step escaped the problem bounds")
            values = np.asarray(ledger.evaluate(candidates), dtype=float)
            base = float(values[0])
            singles = values[1 : 1 + len(buckets)]
            joint_values = values[1 + len(buckets) :]
            for index, (left, right) in enumerate(bucket_pairs):
                score = _mixed_score(base, float(singles[left]), float(singles[right]), float(joint_values[index]))
                bucket_scores.setdefault((left, right), []).append(score)

            # Treat each significant bucket pair as a group-test result.  A
            # variable pair is a candidate only when it is repeatedly separated
            # by significant bucket pairs across the randomized layouts.
            for left, right in bucket_pairs:
                score = bucket_scores[(left, right)][-1]
                for variable in buckets[left]:
                    for other in buckets[right]:
                        pair = (min(variable, other), max(variable, other))
                        candidate_seen[pair] = candidate_seen.get(pair, 0) + 1
                        if score > edge_threshold:
                            candidate_hits[pair] = candidate_hits.get(pair, 0) + 1

    all_pairs = problem.dimension * (problem.dimension - 1) // 2
    separation_fraction = len(separated_pairs) / max(all_pairs, 1)
    candidate_pairs = {
        pair
        for pair, seen in candidate_seen.items()
        if candidate_hits.get(pair, 0) / seen >= min_support
    }
    bucket_edge_rows = tuple(
        (
            left,
            right,
            float(np.median(scores)),
            float(np.mean(np.asarray(scores) > edge_threshold)),
        )
        for (left, right), scores in sorted(bucket_scores.items())
        if float(np.median(scores)) > edge_threshold
        and float(np.mean(np.asarray(scores) > edge_threshold)) >= min_support
    )
    if separation_fraction < 1.0:
        return SparseOverlapDiscoveryResult(
            evidence=_singleton_evidence(problem.dimension, complete=False),
            screened_bucket_edges=bucket_edge_rows,
            refined_edge_scores=(),
            consumed_fes=ledger.count - started,
            expected_fes=expected_fes,
            separated_pair_fraction=separation_fraction,
            candidate_pair_count=len(candidate_pairs),
            complete_reason="separation_coverage_incomplete",
        )
    if len(candidate_pairs) > max_candidate_pairs:
        return SparseOverlapDiscoveryResult(
            evidence=_singleton_evidence(problem.dimension, complete=False),
            screened_bucket_edges=bucket_edge_rows,
            refined_edge_scores=(),
            consumed_fes=ledger.count - started,
            expected_fes=expected_fes,
            separated_pair_fraction=separation_fraction,
            candidate_pair_count=len(candidate_pairs),
            complete_reason="candidate_pair_cap_exceeded",
        )

    refine_per_anchor = 1 + problem.dimension + len(candidate_pairs)
    expected_refine_fes = len(anchor_points) * refine_per_anchor
    expected_fes += expected_refine_fes
    if ledger.remaining < expected_refine_fes:
        return SparseOverlapDiscoveryResult(
            evidence=_singleton_evidence(problem.dimension, complete=False),
            screened_bucket_edges=bucket_edge_rows,
            refined_edge_scores=(),
            consumed_fes=ledger.count - started,
            expected_fes=expected_fes,
            separated_pair_fraction=separation_fraction,
            candidate_pair_count=len(candidate_pairs),
            complete_reason="budget_insufficient_for_refinement",
        )

    refined_scores: dict[tuple[int, int], list[float]] = {pair: [] for pair in sorted(candidate_pairs)}
    for anchor in anchor_points:
        rows = [anchor.copy()]
        for variable in range(problem.dimension):
            candidate = anchor.copy()
            candidate[variable] += step_vector[variable]
            rows.append(candidate)
        for left, right in sorted(candidate_pairs):
            candidate = anchor.copy()
            candidate[left] += step_vector[left]
            candidate[right] += step_vector[right]
            rows.append(candidate)
        if np.any(np.asarray(rows) < problem.lower_array) or np.any(np.asarray(rows) > problem.upper_array):
            raise ValueError("refinement step escaped the problem bounds")
        values = np.asarray(ledger.evaluate(np.asarray(rows, dtype=float)), dtype=float)
        base = float(values[0])
        singles = values[1 : 1 + problem.dimension]
        for index, pair in enumerate(sorted(candidate_pairs)):
            joint = float(values[1 + problem.dimension + index])
            refined_scores[pair].append(_mixed_score(base, float(singles[pair[0]]), float(singles[pair[1]]), joint))

    refined_edges = {
        pair for pair, scores in refined_scores.items()
        if float(np.median(scores)) > edge_threshold and float(np.mean(np.asarray(scores) > edge_threshold)) >= min_support
    }
    refined_rows = tuple(
        (pair[0], pair[1], float(np.median(refined_scores[pair])), float(np.mean(np.asarray(refined_scores[pair]) > edge_threshold)))
        for pair in sorted(refined_edges)
    )
    supports = {(left, right): support for left, right, _, support in refined_rows}
    evidence = _build_evidence(problem.dimension, refined_edges, supports)
    return SparseOverlapDiscoveryResult(
        evidence=evidence,
        screened_bucket_edges=bucket_edge_rows,
        refined_edge_scores=refined_rows,
        consumed_fes=ledger.count - started,
        expected_fes=expected_fes,
        separated_pair_fraction=separation_fraction,
        candidate_pair_count=len(candidate_pairs),
        complete_reason="complete",
    )


__all__ = [
    "DEFAULT_BUCKET_SIZE",
    "DEFAULT_MAX_CANDIDATE_PAIRS",
    "DEFAULT_SPARSE_MIN_SUPPORT",
    "DEFAULT_SPARSE_ROUNDS",
    "SparseOverlapDiscoveryResult",
    "discover_overlap_sparse",
]
