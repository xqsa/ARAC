"""Rolling, reference-blind features for structural overlap relations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .overlap_relation_builder import OverlapRelation


@dataclass(frozen=True)
class TrajectoryFeatures:
    delta_momentum: float
    conflict_trend: float
    stagnation_score: float


class TrajectoryAccumulator:
    """Maintain a bounded history for each structural relation within one run."""

    def __init__(self, window: int = 4) -> None:
        if isinstance(window, bool) or int(window) != window or int(window) <= 0:
            raise ValueError("trajectory window must be a positive integer")
        self._window = int(window)
        self._history: dict[
            tuple[str, int, int, tuple[int, ...]], deque[OverlapRelation]
        ] = {}

    def update(self, relation: OverlapRelation) -> TrajectoryFeatures:
        if not isinstance(relation, OverlapRelation):
            raise TypeError("relation must be an OverlapRelation")
        key = (
            relation.problem_id,
            relation.group_left,
            relation.group_right,
            relation.shared_vars,
        )
        history = self._history.setdefault(key, deque(maxlen=self._window))
        features = self._compute(relation, history)
        history.append(relation)
        return features

    def _compute(
        self,
        current: OverlapRelation,
        history: deque[OverlapRelation],
    ) -> TrajectoryFeatures:
        if not history:
            return TrajectoryFeatures(0.0, 0.0, 0.0)

        previous_signs = [
            1.0 if relation.delta_signed_gap > 0.0 else -1.0
            for relation in history
            if relation.delta_signed_gap != 0.0
        ]
        if previous_signs and current.delta_signed_gap != 0.0:
            current_sign = 1.0 if current.delta_signed_gap > 0.0 else -1.0
            agreement = sum(sign == current_sign for sign in previous_signs)
            delta_momentum = 2.0 * agreement / len(previous_signs) - 1.0
        else:
            delta_momentum = 0.0

        previous_conflict = math.fsum(
            relation.delta_ratio_gap for relation in history
        ) / len(history)
        conflict_trend = math.tanh(
            5.0 * (current.delta_ratio_gap - previous_conflict)
        )

        snapshots = (*history, current)
        stagnant_tail = 0
        for relation in reversed(snapshots):
            if relation.previous_delta != 0.0 or relation.current_delta != 0.0:
                break
            stagnant_tail += 1
        stagnation_score = min(stagnant_tail, self._window) / self._window

        return TrajectoryFeatures(
            delta_momentum=delta_momentum,
            conflict_trend=conflict_trend,
            stagnation_score=stagnation_score,
        )
