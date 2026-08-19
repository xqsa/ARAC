"""Counted black-box structural evidence for Phase I."""

from __future__ import annotations

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
_CLOSURE_ANCHORS = 3


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

    if ledger.count + 2 > stop_count:
        return (), 0, False
    left_points = np.asarray([_perturbed(base, step, (seed,)) for step in steps])
    left_values = np.asarray(ledger.evaluate(left_points), dtype=float)
    pending = [candidates]
    neighbors: list[int] = []
    tests = 0

    while pending:
        capacity = (stop_count - ledger.count) // 4
        if capacity <= 0:
            return tuple(sorted(neighbors)), tests, False
        active = pending[: min(_SUBSET_BATCH_SIZE, capacity)]
        del pending[: len(active)]
        primary_rows = []
        for subset in active:
            right = _perturbed(base, steps[0], subset)
            joint = right.copy()
            joint[seed] += steps[0][seed]
            primary_rows.extend((right, joint))
        primary_values = np.asarray(
            ledger.evaluate(np.asarray(primary_rows, dtype=float)),
            dtype=float,
        )
        primary_significant = []
        for index in range(len(active)):
            significant, _ = _interaction(
                base_value,
                float(left_values[0]),
                float(primary_values[2 * index]),
                float(primary_values[2 * index + 1]),
            )
            primary_significant.append(significant)
        negative_indices = [
            index for index, significant in enumerate(primary_significant) if not significant
        ]
        secondary_significant: dict[int, bool] = {}
        if negative_indices:
            secondary_rows = []
            for index in negative_indices:
                subset = active[index]
                right = _perturbed(base, steps[1], subset)
                joint = right.copy()
                joint[seed] += steps[1][seed]
                secondary_rows.extend((right, joint))
            secondary_values = np.asarray(
                ledger.evaluate(np.asarray(secondary_rows, dtype=float)),
                dtype=float,
            )
            for offset, index in enumerate(negative_indices):
                significant, _ = _interaction(
                    base_value,
                    float(left_values[1]),
                    float(secondary_values[2 * offset]),
                    float(secondary_values[2 * offset + 1]),
                )
                secondary_significant[index] = significant
        tests += len(active)
        for index, subset in enumerate(active):
            significant = primary_significant[index] or secondary_significant.get(index, False)
            if not significant:
                continue
            if len(subset) == 1:
                neighbors.append(subset[0])
                continue
            middle = len(subset) // 2
            pending.extend((subset[:middle], subset[middle:]))
    return tuple(sorted(neighbors)), tests, True


def _discover_component(
    *,
    seed: int,
    candidates: tuple[int, ...],
    base: np.ndarray,
    base_value: float,
    steps: tuple[np.ndarray, np.ndarray],
    ledger: EvaluationLedger,
    stop_count: int,
) -> tuple[tuple[int, ...], int, bool]:
    component = {seed}
    remaining = set(candidates)
    anchors = (seed,)
    tests = 0
    while anchors and remaining:
        discovered: set[int] = set()
        for anchor in anchors:
            neighbors, used_tests, completed = _discover_neighbors(
                seed=anchor,
                candidates=tuple(sorted(remaining)),
                base=base,
                base_value=base_value,
                steps=steps,
                ledger=ledger,
                stop_count=stop_count,
            )
            tests += used_tests
            if not completed:
                return tuple(sorted(component)), tests, False
            discovered.update(neighbors)
            remaining.difference_update(neighbors)
        if not discovered:
            break
        component.update(discovered)
        anchors = tuple(sorted(discovered))[:_CLOSURE_ANCHORS]
        if len(component) >= 2 * MAX_BLOCK_SIZE:
            break
    return tuple(sorted(component)), tests, True


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


def _merge_related_blocks(
    blocks: tuple[tuple[int, ...], ...],
    relations: tuple[RelationEvidence, ...],
) -> tuple[tuple[int, ...], ...]:
    parents = list(range(len(blocks)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for relation in relations:
        left = root(relation.left_block)
        right = root(relation.right_block)
        if left != right:
            parents[right] = left
    components: dict[int, list[int]] = {}
    for index, block in enumerate(blocks):
        components.setdefault(root(index), []).extend(block)
    return tuple(
        sorted(
            (tuple(sorted(values)) for values in components.values()),
            key=lambda block: block[0],
        )
    )


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
    required = 2 * (block_count + block_count * (block_count - 1) // 2)
    if ledger.count + required > stop_count:
        return (), 0, False

    block_points_by_probe = tuple(
        np.asarray([_perturbed(base, step, block) for block in blocks])
        for step in steps
    )
    block_values_by_probe = tuple(
        np.asarray(ledger.evaluate(points), dtype=float)
        for points in block_points_by_probe
    )
    pairs = tuple(
        (left, right)
        for left in range(block_count)
        for right in range(left + 1, block_count)
    )
    joint_values_by_probe = tuple(
        np.asarray(
            ledger.evaluate(
                np.asarray(
                    [
                        _joint_block_point(
                            block_points_by_probe[probe_index][left],
                            blocks[right],
                            steps[probe_index],
                        )
                        for left, right in pairs
                    ]
                )
            ),
            dtype=float,
        )
        if pairs
        else np.asarray([], dtype=float)
        for probe_index in range(2)
    )

    relations = []
    for index, (left, right) in enumerate(pairs):
        outcomes = tuple(
            _interaction(
                base_value,
                float(block_values_by_probe[probe_index][left]),
                float(block_values_by_probe[probe_index][right]),
                float(joint_values_by_probe[probe_index][index]),
            )
            for probe_index in range(2)
        )
        significant = any(item[0] for item in outcomes)
        strength = max(item[1] for item in outcomes)
        if significant:
            primary_values = block_values_by_probe[0]
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
    span = problem.upper_array - problem.lower_array
    steps = (
        0.10 * span,
        0.061 * span * rng.choice(np.asarray([-1.0, 1.0]), size=problem.dimension),
    )
    order = tuple(int(value) for value in rng.permutation(problem.dimension))
    unassigned = set(order)
    blocks = []
    tests = 0
    completed = True

    while unassigned:
        seed = next(value for value in order if value in unassigned)
        candidates = tuple(value for value in order if value in unassigned and value != seed)
        block, used_tests, finished = _discover_component(
            seed=seed,
            candidates=candidates,
            base=anchor,
            base_value=float(base_value),
            steps=steps,
            ledger=ledger,
            stop_count=stop_count,
        )
        tests += used_tests
        if not finished:
            completed = False
            break
        blocks.append(block)
        unassigned.difference_update(block)

    if not completed:
        return StructuralEvidence(
            blocks=fallback_blocks,
            relations=fallback_relations,
            consumed_fes=ledger.count - started,
            interaction_tests=tests,
            completed=False,
        )

    raw_blocks = _coalesce_blocks(tuple(blocks))
    raw_relations, relation_tests, relations_complete = _derive_block_relations(
        blocks=raw_blocks,
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
    merged = _merge_related_blocks(raw_blocks, raw_relations)
    oversized = any(len(block) > MAX_BLOCK_SIZE for block in merged)
    inferred = _split_large_blocks(merged)
    relations: tuple[RelationEvidence, ...] = ()
    for _ in range(4):
        relations, final_relation_tests, relations_complete = _derive_block_relations(
            blocks=inferred,
            base=anchor,
            base_value=float(base_value),
            steps=steps,
            ledger=ledger,
            stop_count=stop_count,
        )
        tests += final_relation_tests
        if not relations_complete:
            return StructuralEvidence(
                blocks=fallback_blocks,
                relations=fallback_relations,
                consumed_fes=ledger.count - started,
                interaction_tests=tests,
                completed=False,
            )
        if not relations or oversized:
            break
        merged = _merge_related_blocks(inferred, relations)
        oversized = any(len(block) > MAX_BLOCK_SIZE for block in merged)
        inferred = _split_large_blocks(merged)
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
