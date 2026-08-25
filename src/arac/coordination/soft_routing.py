"""Gated soft-routing activity for an already selected CTP/GSS host.

This module is intentionally not imported by the production planner.  It is a
small, deterministic utility for the post-M1 soft-routing gate.  Its output may
only rank patch scopes and bound patch strength/radius; callers must keep the
outer action route and FE reservations unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


EPS = 1e-12


def _finite_values(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def rank_normalize(values: Iterable[float]) -> tuple[float, ...]:
    """Map values to [0, 1] ranks with deterministic average ties."""

    values_tuple = _finite_values(values, "values")
    order = sorted(range(len(values_tuple)), key=lambda index: (values_tuple[index], index))
    ranks = [0.0] * len(values_tuple)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values_tuple[order[end]] == values_tuple[order[index]]:
            end += 1
        average_rank = 0.5 * (index + end - 1)
        for position in range(index, end):
            ranks[order[position]] = average_rank
        index = end
    denominator = max(1, len(values_tuple) - 1)
    return tuple(rank / denominator for rank in ranks)


@dataclass(frozen=True)
class SoftRoutingDecision:
    disagreement_rank: tuple[float, ...]
    progress_rank: tuple[float, ...]
    activity: tuple[float, ...]
    scope_order: tuple[int, ...]

    @property
    def candidate_strength(self) -> tuple[float, ...]:
        return self.activity

    @property
    def radius_upper_bound(self) -> tuple[float, ...]:
        return self.activity


def compute_activity(
    owner_disagreement: Iterable[float],
    recent_progress_residual: Iterable[float],
    *,
    eps: float = EPS,
) -> SoftRoutingDecision:
    """Compute ``a_j = d_j / (d_j + q_j + eps)`` after rank normalization."""

    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    disagreement_rank = rank_normalize(owner_disagreement)
    progress_rank = rank_normalize(recent_progress_residual)
    if len(disagreement_rank) != len(progress_rank):
        raise ValueError("owner disagreement and progress residual widths must match")
    activity = tuple(
        min(1.0, max(0.0, disagreement / (disagreement + progress + eps)))
        for disagreement, progress in zip(disagreement_rank, progress_rank, strict=True)
    )
    scope_order = tuple(sorted(range(len(activity)), key=lambda index: (-activity[index], index)))
    return SoftRoutingDecision(disagreement_rank, progress_rank, activity, scope_order)


__all__ = ["EPS", "SoftRoutingDecision", "compute_activity", "rank_normalize"]
