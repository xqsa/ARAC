"""Counted black-box structural evidence for Phase I."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import RelationEvidence
from arac.runtime.ledger import EvaluationLedger


# Phase I exposes a fixed-width, identity-blind decomposition to every action.
# Allowing hundreds of singleton components changed the action semantics between
# seeds and made the selector learn decomposition artifacts instead of landscape
# behavior.
MAX_BLOCK_COUNT = 20
MAX_BLOCK_SIZE = 128
_SUBSET_BATCH_SIZE = 128
_ROUND_OFF_FACTOR = 64.0
_MIN_RELATIVE_INTERACTION = 1e-11
_OVERSIZED_REFINEMENT_ANCHORS = MAX_BLOCK_COUNT
_SPARSE_RELATION_REFINEMENT_MAX = 2


@dataclass(frozen=True)
class StructuralEvidence:
    blocks: tuple[tuple[int, ...], ...]
    relations: tuple[RelationEvidence, ...]
    consumed_fes: int
    interaction_tests: int
    completed: bool


def _interaction(
    base_value: float,
    left_value: float,
    right_value: float,
    joint_value: float,
) -> tuple[bool, float]:
    delta = abs((left_value - base_value) - (joint_value - right_value))
    scale = abs(base_value) + abs(left_value) + abs(right_value) + abs(joint_value) + 1.0
    threshold = max(
        1e-12,
        _ROUND_OFF_FACTOR * np.finfo(float).eps * scale,
    )
    strength = float(delta / scale)
    return delta > threshold and strength > _MIN_RELATIVE_INTERACTION, strength


def _perturbed(base: np.ndarray, step: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    candidate = base.copy()
    selected = np.asarray(indices, dtype=int)
    candidate[selected] += step[selected]
    return candidate


def _bounded_steps(
    problem: OptimizationProblem,
    base: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose fixed probe steps that stay feasible around an arbitrary anchor."""

    span = problem.upper_array - problem.lower_array
    primary_sign = np.where(
        problem.upper_array - base >= base - problem.lower_array,
        1.0,
        -1.0,
    )
    primary = 0.10 * span * primary_sign

    secondary = 0.061 * span * rng.choice(
        np.asarray([-1.0, 1.0]),
        size=problem.dimension,
    )
    infeasible = (base + secondary < problem.lower_array) | (
        base + secondary > problem.upper_array
    )
    secondary[infeasible] *= -1.0
    if any(
        np.any(base + step < problem.lower_array)
        or np.any(base + step > problem.upper_array)
        for step in (primary, secondary)
    ):
        raise RuntimeError("structural probe step escaped the public bounds")
    return primary, secondary


def _discover_neighbors(
    *,
    seed: int,
    candidates: tuple[int, ...],
    base: np.ndarray,
    base_value: float,
    steps: tuple[np.ndarray, np.ndarray],
    ledger: EvaluationLedger,
    stop_count: int,
) -> tuple[tuple[int, ...], int, bool]:
    if not candidates:
        return (), 0, True
    if ledger.count >= stop_count:
        return (), 0, False

    if ledger.count + 1 > stop_count:
        return (), 0, False
    primary_step = steps[0]
    left_value = float(ledger.evaluate(_perturbed(base, primary_step, (seed,))))
    pending = deque((candidates,))
    neighbors: list[int] = []
    tests = 0

    while pending:
        capacity = (stop_count - ledger.count) // 2
        if capacity <= 0:
            return tuple(sorted(neighbors)), tests, False
        active = [
            pending.popleft()
            for _ in range(min(len(pending), _SUBSET_BATCH_SIZE, capacity))
        ]
        primary_rows = []
        for subset in active:
            right = _perturbed(base, primary_step, subset)
            joint = right.copy()
            joint[seed] += primary_step[seed]
            primary_rows.extend((right, joint))
        primary_values = np.asarray(
            ledger.evaluate(np.asarray(primary_rows, dtype=float)),
            dtype=float,
        )
        for index in range(len(active)):
            significant, _ = _interaction(
                base_value,
                left_value,
                float(primary_values[2 * index]),
                float(primary_values[2 * index + 1]),
            )
            if not significant:
                continue
            subset = active[index]
            if len(subset) == 1:
                neighbors.append(subset[0])
                continue
            middle = len(subset) // 2
            pending.extend((subset[:middle], subset[middle:]))
        tests += len(active)
    return tuple(sorted(neighbors)), tests, True


def _discover_partition(
    *,
    order: tuple[int, ...],
    base: np.ndarray,
    base_value: float,
    steps: tuple[np.ndarray, np.ndarray],
    ledger: EvaluationLedger,
    stop_count: int,
) -> tuple[tuple[tuple[int, ...], ...], int, bool]:
    """Cover variables with direct neighborhoods without closing overlap chains."""

    rank = {variable: index for index, variable in enumerate(order)}
    unassigned = set(order)
    neighborhoods: dict[int, frozenset[int]] = {}
    candidates: list[frozenset[int]] = []
    tests = 0

    def neighborhood(anchor: int) -> tuple[frozenset[int], bool]:
        nonlocal tests
        if anchor not in neighborhoods:
            neighbors, used_tests, completed = _discover_neighbors(
                seed=anchor,
                candidates=tuple(variable for variable in order if variable != anchor),
                base=base,
                base_value=base_value,
                steps=steps,
                ledger=ledger,
                stop_count=stop_count,
            )
            tests += used_tests
            if not completed:
                return frozenset((anchor,)), False
            neighborhoods[anchor] = frozenset((anchor, *neighbors))
        return neighborhoods[anchor], True

    while unassigned:
        seed = next(variable for variable in order if variable in unassigned)
        candidate, completed = neighborhood(seed)
        if not completed:
            return (), tests, False
        candidate = frozenset(candidate & unassigned)
        detached = []
        if len(candidate) > MAX_BLOCK_SIZE:
            refinement_anchors = sorted(
                (variable for variable in candidate if variable != seed),
                key=rank.__getitem__,
            )[:_OVERSIZED_REFINEMENT_ANCHORS]
            for anchor in refinement_anchors:
                if anchor not in candidate:
                    continue
                alternate, completed = neighborhood(anchor)
                if not completed:
                    return (), tests, False
                overlap = frozenset(candidate & alternate)
                if seed in overlap and len(overlap) < len(candidate):
                    candidate = overlap
                elif seed not in overlap and overlap:
                    detached.append(overlap)
                    candidate = frozenset(candidate - overlap)
                if len(candidate) <= MAX_BLOCK_SIZE:
                    break
        for block in detached:
            candidates.append(block)
            unassigned.difference_update(block)
        candidate = frozenset(candidate & unassigned)
        if seed not in candidate:
            raise RuntimeError("oversized refinement dropped the active seed")
        candidates.append(candidate)
        unassigned.difference_update(candidate)

    assigned: set[int] = set()
    blocks = []
    for candidate in candidates:
        block = tuple(sorted(candidate - assigned))
        if block:
            blocks.append(block)
            assigned.update(block)
    if assigned != set(order):
        raise RuntimeError("direct neighborhoods did not cover every variable")
    return tuple(blocks), tests, True


def _split_large_blocks(
    blocks: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    split = []
    for block in blocks:
        pieces = max(1, math.ceil(len(block) / MAX_BLOCK_SIZE))
        split.extend(
            tuple(int(value) for value in piece)
            for piece in np.array_split(np.asarray(block, dtype=int), pieces)
        )
    return tuple(split)


def _coalesce_blocks(
    blocks: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    active = [tuple(block) for block in blocks]
    while len(active) > MAX_BLOCK_COUNT:
        order = sorted(range(len(active)), key=lambda index: (len(active[index]), index))
        left, right = sorted(order[:2])
        merged = tuple(sorted(active[left] + active[right]))
        active = [block for index, block in enumerate(active) if index not in {left, right}]
        active.append(merged)
    return tuple(sorted((tuple(sorted(block)) for block in active), key=lambda block: block[0]))


def _derive_block_relations(
    *,
    blocks: tuple[tuple[int, ...], ...],
    base: np.ndarray,
    base_value: float,
    steps: tuple[np.ndarray, np.ndarray],
    ledger: EvaluationLedger,
    stop_count: int,
) -> tuple[tuple[RelationEvidence, ...], int, bool]:
    block_count = len(blocks)
    pairs = tuple(
        (left, right)
        for left in range(block_count)
        for right in range(left + 1, block_count)
    )
    required = block_count + len(pairs)
    if ledger.count + required > stop_count:
        return (), 0, False

    primary_step = steps[0]
    primary_points = np.asarray(
        [_perturbed(base, primary_step, block) for block in blocks]
    )
    primary_values = np.asarray(ledger.evaluate(primary_points), dtype=float)
    primary_joint_values = np.asarray(
        ledger.evaluate(
            np.asarray(
                [
                    _joint_block_point(primary_points[left], blocks[right], primary_step)
                    for left, right in pairs
                ]
            )
        ),
        dtype=float,
    ) if pairs else np.asarray([], dtype=float)
    outcomes_by_probe = [
        tuple(
            _interaction(
                base_value,
                float(primary_values[left]),
                float(primary_values[right]),
                float(primary_joint_values[index]),
            )
            for index, (left, right) in enumerate(pairs)
        )
    ]

    if any(outcome[0] for outcome in outcomes_by_probe[0]):
        secondary_step = steps[1]
        if ledger.count + required <= stop_count:
            secondary_points = np.asarray(
                [_perturbed(base, secondary_step, block) for block in blocks]
            )
            secondary_values = np.asarray(ledger.evaluate(secondary_points), dtype=float)
            secondary_joint_values = np.asarray(
                ledger.evaluate(
                    np.asarray(
                        [
                            _joint_block_point(
                                secondary_points[left],
                                blocks[right],
                                secondary_step,
                            )
                            for left, right in pairs
                        ]
                    )
                ),
                dtype=float,
            ) if pairs else np.asarray([], dtype=float)
            outcomes_by_probe.append(
                tuple(
                    _interaction(
                        base_value,
                        float(secondary_values[left]),
                        float(secondary_values[right]),
                        float(secondary_joint_values[index]),
                    )
                    for index, (left, right) in enumerate(pairs)
                )
            )

    relations = []
    for index, (left, right) in enumerate(pairs):
        outcomes = tuple(probe[index] for probe in outcomes_by_probe)
        significant = any(item[0] for item in outcomes)
        strength = max(item[1] for item in outcomes)
        if significant:
            scale = abs(base_value) + abs(primary_values[left]) + abs(primary_values[right]) + 1.0
            relations.append(
                RelationEvidence(
                    left_block=left,
                    right_block=right,
                    strength=strength,
                    disagreement=float(abs(primary_values[left] - primary_values[right]) / scale),
                )
            )
    return tuple(relations), len(pairs), True


def _joint_block_point(
    left_point: np.ndarray,
    right_block: tuple[int, ...],
    step: np.ndarray,
) -> np.ndarray:
    joint = left_point.copy()
    indices = np.asarray(right_block, dtype=int)
    joint[indices] += step[indices]
    return joint


def infer_structure(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    base: np.ndarray,
    base_value: float,
    run_seed: int,
    max_fes: int,
    fallback_blocks: tuple[tuple[int, ...], ...],
    fallback_relations: tuple[RelationEvidence, ...],
) -> StructuralEvidence:
    """Infer a partition and cross-block relations using counted objective responses."""

    if problem is not ledger.problem:
        raise ValueError("structural probe problem and ledger problem must be identical")
    anchor = np.asarray(base, dtype=float)
    if anchor.shape != (problem.dimension,) or not np.all(np.isfinite(anchor)):
        raise ValueError("structural probe base is invalid")
    if not math.isfinite(float(base_value)):
        raise ValueError("structural probe base value must be finite")
    if max_fes <= 0 or max_fes > ledger.remaining:
        raise ValueError("structural probe budget is invalid")

    started = ledger.count
    stop_count = started + int(max_fes)
    rng = np.random.default_rng(int(run_seed) ^ 0x57A6C7)
    steps = _bounded_steps(problem, anchor, rng)
    order = tuple(int(value) for value in rng.permutation(problem.dimension))
    blocks, tests, completed = _discover_partition(
        order=order,
        base=anchor,
        base_value=float(base_value),
        steps=steps,
        ledger=ledger,
        stop_count=stop_count,
    )
    if not completed:
        return StructuralEvidence(
            blocks=fallback_blocks,
            relations=fallback_relations,
            consumed_fes=ledger.count - started,
            interaction_tests=tests,
            completed=False,
        )

    inferred = _coalesce_blocks(_split_large_blocks(blocks))
    relations, relation_tests, relations_complete = _derive_block_relations(
        blocks=inferred,
        base=anchor,
        base_value=float(base_value),
        steps=steps,
        ledger=ledger,
        stop_count=stop_count,
    )
    tests += relation_tests
    if not relations_complete:
        return StructuralEvidence(
            blocks=fallback_blocks,
            relations=fallback_relations,
            consumed_fes=ledger.count - started,
            interaction_tests=tests,
            completed=False,
        )

    if 0 < len(relations) <= _SPARSE_RELATION_REFINEMENT_MAX:
        related_indices = {
            index
            for relation in relations
            for index in (relation.left_block, relation.right_block)
        }
        related_variables = {
            variable
            for index in related_indices
            for variable in inferred[index]
        }
        local_order = tuple(
            variable for variable in order if variable in related_variables
        )
        refined, refinement_tests, refinement_complete = _discover_partition(
            order=local_order,
            base=anchor,
            base_value=float(base_value),
            steps=(steps[1], steps[0]),
            ledger=ledger,
            stop_count=stop_count,
        )
        tests += refinement_tests
        if refinement_complete and len(refined) == len(related_indices):
            untouched = tuple(
                block
                for index, block in enumerate(inferred)
                if index not in related_indices
            )
            inferred = _coalesce_blocks(_split_large_blocks((*untouched, *refined)))
            relations, relation_tests, relations_complete = _derive_block_relations(
                blocks=inferred,
                base=anchor,
                base_value=float(base_value),
                steps=steps,
                ledger=ledger,
                stop_count=stop_count,
            )
            tests += relation_tests
            if not relations_complete:
                return StructuralEvidence(
                    blocks=fallback_blocks,
                    relations=fallback_relations,
                    consumed_fes=ledger.count - started,
                    interaction_tests=tests,
                    completed=False,
                )
    return StructuralEvidence(
        blocks=inferred,
        relations=relations,
        consumed_fes=ledger.count - started,
        interaction_tests=tests,
        completed=True,
    )


__all__ = [
    "MAX_BLOCK_COUNT",
    "MAX_BLOCK_SIZE",
    "StructuralEvidence",
    "infer_structure",
]
