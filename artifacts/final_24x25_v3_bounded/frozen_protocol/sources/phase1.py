"""A fixed-budget black-box probe for evidence-driven action selection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.structural import StructuralEvidence, infer_structure
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort


PHASE1_PROTOCOL = "arac-identity-blind-evidence-v3"
LANDSCAPE_PROBE_FES = 240
PHASE1_MIN_FES = LANDSCAPE_PROBE_FES
PHASE1_FES = 180_000
STRUCTURE_MAX_FES = 174_000
STRUCTURE_MIN_FES = 2_000
BLOCK_COUNT = 20
RELATION_PROBE_COUNT = 39
LINE_COUNT = 4
LINE_NONZERO_OFFSET_COUNT = 40
_BLOCK_SCALE = 0.025
_LINE_SCALE = 0.035
_SEED_NAMESPACE = 0xA4AC2026
_EPSILON = 1e-300

_LINE_METRICS = (
    "residual_std",
    "high_frequency_fraction",
    "second_difference_total",
    "residual_max",
    "third_difference_std",
)
_AGGREGATES = ("median", "std", "minimum", "maximum")
LINE_FEATURE_NAMES = tuple(
    f"line_{metric}_{aggregate}"
    for metric in _LINE_METRICS
    for aggregate in _AGGREGATES
)
LANDSCAPE_FEATURE_NAMES = (
    "log10_center_error",
    "log10_best_probe_error",
    "probe_log_relative_std",
    "probe_log_relative_iqr",
    "block_response_median",
    "block_response_cv",
    "block_asymmetry_median",
    "relation_strength_median",
    "relation_strength_maximum",
    "significant_relation_fraction",
    *LINE_FEATURE_NAMES,
)
PHASE1_FEATURE_NAMES = (
    *LANDSCAPE_FEATURE_NAMES,
    "structural_block_count_normalized",
    "structural_largest_block_fraction",
    "structural_relation_density",
    "structural_probe_fraction",
    "structural_inference_complete",
    "phase1_log10_improvement",
)


def phase1_budget(total_budget_fes: int) -> int:
    if isinstance(total_budget_fes, bool) or total_budget_fes <= PHASE1_MIN_FES:
        raise ValueError("total budget must exceed the minimum Phase-I probe")
    return min(PHASE1_FES, max(PHASE1_MIN_FES, total_budget_fes // 16))


@dataclass(frozen=True)
class Phase1Probe:
    checkpoint: PhaseCheckpoint
    fitness_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.checkpoint.protocol != PHASE1_PROTOCOL:
            raise ValueError("Phase-I protocol drifted")
        if self.checkpoint.phase1_fes != phase1_budget(self.checkpoint.total_budget_fes):
            raise ValueError("Phase-I FE count drifted")
        if len(self.fitness_values) != LANDSCAPE_PROBE_FES:
            raise ValueError("Phase-I landscape trace length drifted")
        if not all(math.isfinite(value) for value in self.fitness_values):
            raise ValueError("Phase-I trace must be finite")


def _partition(dimension: int, rng: np.random.Generator) -> tuple[tuple[int, ...], ...]:
    if dimension < BLOCK_COUNT:
        raise ValueError(f"ARAC requires at least {BLOCK_COUNT} dimensions")
    permutation = rng.permutation(dimension)
    blocks = tuple(
        tuple(sorted(int(value) for value in block))
        for block in np.array_split(permutation, BLOCK_COUNT)
    )
    if any(not block for block in blocks):
        raise RuntimeError("evidence block construction produced an empty block")
    return blocks


def _relation_pairs() -> tuple[tuple[int, int], ...]:
    adjacent = tuple((index, index + 1) for index in range(BLOCK_COUNT - 1))
    cross = tuple(
        tuple(sorted((index, (index + BLOCK_COUNT // 4) % BLOCK_COUNT)))
        for index in range(BLOCK_COUNT)
    )
    pairs = adjacent + cross
    if len(pairs) != RELATION_PROBE_COUNT or len(set(pairs)) != len(pairs):
        raise RuntimeError("relation probe design drifted")
    return pairs


def _probe_candidates(
    problem: OptimizationProblem,
    blocks: tuple[tuple[int, ...], ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[int, ...]]:
    center = (problem.lower_array + problem.upper_array) / 2.0
    span = problem.upper_array - problem.lower_array
    candidates = [center.copy()]

    for block in blocks:
        step = _BLOCK_SCALE * span[np.asarray(block)] / math.sqrt(len(block))
        positive = center.copy()
        negative = center.copy()
        positive[np.asarray(block)] += step
        negative[np.asarray(block)] -= step
        candidates.extend((positive, negative))

    pairs = _relation_pairs()
    for left, right in pairs:
        mixed = center.copy()
        indices = np.asarray(blocks[left] + blocks[right])
        mixed[indices] += _BLOCK_SCALE * span[indices] / np.sqrt(len(indices))
        candidates.append(mixed)

    line_axes = tuple(
        int(value) for value in rng.choice(problem.dimension, size=LINE_COUNT, replace=False)
    )
    offsets = np.linspace(-_LINE_SCALE, _LINE_SCALE, LINE_NONZERO_OFFSET_COUNT + 1)
    offsets = offsets[offsets != 0.0]
    if len(offsets) != LINE_NONZERO_OFFSET_COUNT:
        raise RuntimeError("line offset design drifted")
    for axis in line_axes:
        for offset in offsets:
            point = center.copy()
            point[axis] += offset * span[axis]
            candidates.append(point)

    matrix = np.asarray(candidates, dtype=float)
    if matrix.shape != (LANDSCAPE_PROBE_FES, problem.dimension):
        raise RuntimeError("Phase-I candidate count drifted")
    if np.any(matrix < problem.lower_array) or np.any(matrix > problem.upper_array):
        raise RuntimeError("Phase-I candidate escaped the public search bounds")
    return matrix, pairs, line_axes


def _line_signature(center_error: float, line_errors: np.ndarray) -> tuple[float, ...]:
    if line_errors.shape != (LINE_COUNT, LINE_NONZERO_OFFSET_COUNT):
        raise ValueError("line error matrix has the wrong shape")
    normalized_offsets = np.linspace(-1.0, 1.0, LINE_NONZERO_OFFSET_COUNT + 1)
    zero_index = LINE_NONZERO_OFFSET_COUNT // 2
    metric_rows = []
    for raw_line in line_errors:
        line = np.insert(raw_line, zero_index, center_error)
        magnitude = max(float(np.max(np.abs(line))), _EPSILON)
        scaled = line / magnitude
        line_range = max(float(np.ptp(scaled)), 64.0 * np.finfo(float).eps)
        coefficients = np.polynomial.chebyshev.chebfit(normalized_offsets, scaled, 5)
        trend = np.polynomial.chebyshev.chebval(normalized_offsets, coefficients)
        residual = (scaled - trend) / line_range
        centered = residual - np.mean(residual)
        power = np.square(np.abs(np.fft.rfft(centered)))
        metric_rows.append(
            (
                float(np.std(residual)),
                float(np.sum(power[8:]) / (np.sum(power[1:]) + _EPSILON)),
                float(np.sum(np.abs(np.diff(residual, n=2)))),
                float(np.max(np.abs(residual))),
                float(np.std(np.diff(residual, n=3))),
            )
        )
    matrix = np.asarray(metric_rows, dtype=float)
    signature = tuple(
        value
        for column in matrix.T
        for value in (
            float(np.median(column)),
            float(np.std(column)),
            float(np.min(column)),
            float(np.max(column)),
        )
    )
    if len(signature) != len(LINE_FEATURE_NAMES) or not all(
        math.isfinite(value) and value >= 0.0 for value in signature
    ):
        raise RuntimeError("line signature is invalid")
    return signature


def _derive_relations(
    errors: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[RelationEvidence, ...], np.ndarray]:
    center = float(errors[0])
    scale = abs(center) + 1.0
    positive = errors[1 : 1 + 2 * BLOCK_COUNT : 2]
    mixed = errors[1 + 2 * BLOCK_COUNT : 1 + 2 * BLOCK_COUNT + len(pairs)]
    strengths = np.asarray(
        [
            abs(
                float(mixed[index])
                - float(positive[left])
                - float(positive[right])
                + center
            )
            / scale
            for index, (left, right) in enumerate(pairs)
        ],
        dtype=float,
    )
    median = float(np.median(strengths))
    mad = float(np.median(np.abs(strengths - median)))
    threshold = max(1e-10, median + 3.0 * mad)
    relations = tuple(
        RelationEvidence(
            left_block=left,
            right_block=right,
            strength=float(strengths[index]),
            disagreement=abs(float(positive[left]) - float(positive[right])) / scale,
        )
        for index, (left, right) in enumerate(pairs)
        if strengths[index] > threshold
    )
    return relations, strengths


def _feature_vector(
    values: np.ndarray,
    optimum: float,
    pairs: tuple[tuple[int, int], ...],
    relations: tuple[RelationEvidence, ...],
    relation_strengths: np.ndarray,
) -> tuple[float, ...]:
    errors = np.maximum(values - optimum, 0.0)
    center = float(errors[0])
    log_relative = np.log10(errors + _EPSILON) - math.log10(center + _EPSILON)
    block_pairs = errors[1 : 1 + 2 * BLOCK_COUNT].reshape(BLOCK_COUNT, 2)
    block_response = np.mean(np.abs(block_pairs - center), axis=1) / (abs(center) + 1.0)
    block_asymmetry = np.abs(block_pairs[:, 0] - block_pairs[:, 1]) / (abs(center) + 1.0)
    line_start = 1 + 2 * BLOCK_COUNT + len(pairs)
    line_errors = errors[line_start:].reshape(LINE_COUNT, LINE_NONZERO_OFFSET_COUNT)
    features = (
        math.log10(center + _EPSILON),
        math.log10(float(np.min(errors)) + _EPSILON),
        float(np.std(log_relative)),
        float(np.quantile(log_relative, 0.9) - np.quantile(log_relative, 0.1)),
        float(np.median(block_response)),
        float(np.std(block_response) / (np.mean(block_response) + _EPSILON)),
        float(np.median(block_asymmetry)),
        float(np.median(relation_strengths)),
        float(np.max(relation_strengths)),
        len(relations) / len(pairs),
        *_line_signature(center, line_errors),
    )
    if len(features) != len(LANDSCAPE_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in features
    ):
        raise RuntimeError("Phase-I feature vector is invalid")
    return tuple(float(value) for value in features)


def _structural_features(
    evidence: StructuralEvidence,
    *,
    dimension: int,
    structural_budget: int,
    probe_best_error: float,
    phase1_best_error: float,
) -> tuple[float, ...]:
    block_count = len(evidence.blocks)
    possible_relations = block_count * (block_count - 1) // 2
    return (
        block_count / dimension,
        max(len(block) for block in evidence.blocks) / dimension,
        len(evidence.relations) / max(possible_relations, 1),
        evidence.consumed_fes / max(structural_budget, 1),
        float(evidence.completed),
        math.log10((probe_best_error + 1.0) / (phase1_best_error + 1.0)),
    )


def run_phase1(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    run_seed: int,
) -> Phase1Probe:
    """Collect identity-blind landscape and structural evidence at a fixed boundary."""

    if problem is not ledger.problem:
        raise ValueError("Phase-I problem and ledger problem must be identical")
    target_fes = phase1_budget(ledger.total_budget)
    if ledger.count != 0 or ledger.remaining < target_fes:
        raise ValueError("Phase I requires a fresh ledger with its full target budget")
    if isinstance(run_seed, bool) or run_seed < 0:
        raise ValueError("run_seed must be a non-negative integer")
    rng = np.random.default_rng(int(run_seed) ^ _SEED_NAMESPACE)
    blocks = _partition(problem.dimension, rng)
    candidates, pairs, _ = _probe_candidates(problem, blocks, rng)
    values = np.asarray(ledger.evaluate(candidates), dtype=float)
    errors = np.maximum(values - problem.optimum, 0.0)
    fallback_relations, strengths = _derive_relations(errors, pairs)
    landscape_features = _feature_vector(
        values,
        problem.optimum,
        pairs,
        fallback_relations,
        strengths,
    )

    available = target_fes - ledger.count
    if available >= STRUCTURE_MIN_FES:
        structural_budget = min(STRUCTURE_MAX_FES, available)
        structural = infer_structure(
            problem,
            ledger,
            base=candidates[0],
            base_value=float(values[0]),
            run_seed=run_seed,
            max_fes=structural_budget,
            fallback_blocks=blocks,
            fallback_relations=fallback_relations,
        )
    else:
        structural_budget = 0
        structural = StructuralEvidence(
            blocks=blocks,
            relations=fallback_relations,
            consumed_fes=0,
            interaction_tests=0,
            completed=False,
        )

    if ledger.count < target_fes:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(value) for value in ledger.best_x),
            sigma=0.5,
            seed=int(run_seed) ^ 0xE71D3A,
            budget_fes=target_fes - ledger.count,
            population_size=24,
            restart=False,
        )
    if ledger.count != target_fes:
        raise RuntimeError("Phase-I ledger did not stop at the frozen FE boundary")

    features = landscape_features + _structural_features(
        structural,
        dimension=problem.dimension,
        structural_budget=structural_budget,
        probe_best_error=float(np.min(errors)),
        phase1_best_error=ledger.best_error,
    )
    checkpoint = PhaseCheckpoint(
        protocol=PHASE1_PROTOCOL,
        run_seed=int(run_seed),
        total_budget_fes=ledger.total_budget,
        phase1_fes=ledger.count,
        incumbent=tuple(float(value) for value in ledger.best_x),
        incumbent_error=float(ledger.best_error),
        feature_names=PHASE1_FEATURE_NAMES,
        feature_values=features,
        blocks=structural.blocks,
        relations=structural.relations,
    )
    return Phase1Probe(
        checkpoint=checkpoint,
        fitness_values=tuple(float(value) for value in values),
    )


__all__ = [
    "BLOCK_COUNT",
    "LANDSCAPE_PROBE_FES",
    "LANDSCAPE_FEATURE_NAMES",
    "LINE_COUNT",
    "LINE_FEATURE_NAMES",
    "LINE_NONZERO_OFFSET_COUNT",
    "PHASE1_FEATURE_NAMES",
    "PHASE1_FES",
    "PHASE1_PROTOCOL",
    "PHASE1_MIN_FES",
    "Phase1Probe",
    "phase1_budget",
    "run_phase1",
]
