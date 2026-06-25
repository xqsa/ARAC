"""HCC backbone extraction helpers.

This module is the first clean ARAC extraction layer for the historical
``E:\\HCC-main`` work. It models the data ARAC needs from HCC grouping and
optimization traces without importing legacy milestone runners or mutating the
HCC baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arac.action_space import ActionFamily
from arac.backend_adapter import BackendSemanticsDiff
from arac.evidence import EvidenceProfile, validate_runtime_payload
from arac.policy import ActionDecision


@dataclass(frozen=True)
class HccGroupSignal:
    """Reference-blind signal exposed by one HCC decomposition group."""

    group_id: str
    fitness_delta: float
    rank: int
    shared_variable_count: int = 0


@dataclass(frozen=True)
class HccBackboneSnapshot:
    """Minimal HCC grouping/optimization state needed by ARAC.

    The snapshot deliberately excludes final error, oracle labels, reported
    baselines, problem-family labels, and prior outcome fields. ``problem_id``
    is retained only as execution identity and artifact grouping.
    """

    run_id: str
    problem_id: str
    seed: int
    dimension: int
    group_count: int
    overlap_group_count: int
    overlapping_element_count: int
    budget_remaining_ratio: float
    groups: tuple[HccGroupSignal, ...]
    runtime_payload_extra: dict[str, object] = field(default_factory=dict)


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _rank_stability(groups: tuple[HccGroupSignal, ...]) -> float:
    if len(groups) <= 1:
        return 1.0
    ranks = [group.rank for group in groups]
    if min(ranks) < 1:
        return 0.0
    unique_ratio = len(set(ranks)) / len(ranks)
    return _clamp_ratio(unique_ratio)


def _priority_spread(groups: tuple[HccGroupSignal, ...]) -> float:
    if not groups:
        return 0.0
    ranks = [group.rank for group in groups]
    span = max(ranks) - min(ranks)
    return _clamp_ratio(_safe_divide(span, max(len(groups), 1)))


def _gain_asymmetry(groups: tuple[HccGroupSignal, ...]) -> float:
    if not groups:
        return 0.0
    gains = [max(0.0, group.fitness_delta) for group in groups]
    return _clamp_ratio(_safe_divide(max(gains) - min(gains), max(gains) + 1e-12))


def _direction_disagreement(groups: tuple[HccGroupSignal, ...]) -> float:
    if not groups:
        return 0.0
    positives = sum(1 for group in groups if group.fitness_delta > 0)
    non_positives = len(groups) - positives
    minority = min(positives, non_positives)
    return _clamp_ratio(_safe_divide(minority, len(groups)))


def build_hcc_evidence_profile(snapshot: HccBackboneSnapshot) -> EvidenceProfile:
    """Convert HCC grouping/trace state into a runtime-legal ARAC evidence row."""

    payload = {
        "run_id": snapshot.run_id,
        "problem_id": snapshot.problem_id,
        "seed": snapshot.seed,
        "dimension": snapshot.dimension,
        "group_count": snapshot.group_count,
        "overlap_group_count": snapshot.overlap_group_count,
        "overlapping_element_count": snapshot.overlapping_element_count,
        "budget_remaining_ratio": snapshot.budget_remaining_ratio,
        **snapshot.runtime_payload_extra,
    }
    validate_runtime_payload(payload)

    overlap_degree = _clamp_ratio(
        _safe_divide(snapshot.overlap_group_count, max(snapshot.group_count, 1))
    )
    shared_var_support_ratio = _clamp_ratio(
        _safe_divide(snapshot.overlapping_element_count, max(snapshot.dimension, 1))
    )
    group_gain_asymmetry = _gain_asymmetry(snapshot.groups)
    priority_spread = _priority_spread(snapshot.groups)
    direction_disagreement = _direction_disagreement(snapshot.groups)
    harmful_coord_score = _clamp_ratio(
        max(overlap_degree, shared_var_support_ratio) * max(group_gain_asymmetry, 0.1)
    )

    return EvidenceProfile(
        run_id=snapshot.run_id,
        problem_id=snapshot.problem_id,
        seed=snapshot.seed,
        unit_type="problem",
        unit_id=f"hcc_backbone:{snapshot.problem_id}",
        feature_coverage=1.0 if snapshot.groups else 0.5,
        overlap_degree=overlap_degree,
        shared_var_support_ratio=shared_var_support_ratio,
        direction_disagreement=direction_disagreement,
        harmful_coord_score=harmful_coord_score,
        group_gain_asymmetry=group_gain_asymmetry,
        priority_spread=priority_spread,
        rank_stability=_rank_stability(snapshot.groups),
        budget_remaining_ratio=_clamp_ratio(snapshot.budget_remaining_ratio),
        fallback_margin_proxy=_clamp_ratio(1.0 - harmful_coord_score),
    )


def hcc_backend_semantics_for(decision: ActionDecision) -> BackendSemanticsDiff:
    """Map clean ARAC actions onto HCC optimizer-consumed semantic surfaces."""

    if decision.action_family == ActionFamily.ISOLATE:
        return BackendSemanticsDiff(relation_handling_changed=True)
    if decision.action_family == ActionFamily.PROTECT:
        return BackendSemanticsDiff(budget_allocation_changed=True)
    if decision.action_family == ActionFamily.REASSIGN_REPAIR:
        return BackendSemanticsDiff(variable_owner_changed=True)
    if decision.action_family == ActionFamily.COORDINATE:
        return BackendSemanticsDiff(coordination_mode_changed=True)
    return BackendSemanticsDiff()
