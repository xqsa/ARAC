"""Evidence-to-intervention policy scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from ..actions import ActionDecision, ActionFamily
from ..evidence import EvidenceProfile
from .overlap_hypergraph import (
    ClosedOwnerCredit,
    CompletedSweepSnapshot,
    DelayedHyperedgeCredit,
    DirectionalSurvival,
    GroupCycleObservation,
    HyperedgeCycleState,
    HyperedgeFocalScope,
    HyperedgeScore,
    OverlapHypergraphTopology,
    SharedProposal,
    SharedVariableStar,
    build_closed_owner_credit,
    build_delayed_hyperedge_credit,
    build_group_cycle_observation,
    build_hyperedge_cycle_states,
    build_overlap_hypergraph,
    directional_survival,
    score_hyperedge_states,
)


__all__ = [
    "ClosedOwnerCredit",
    "CompletedSweepSnapshot",
    "DelayedHyperedgeCredit",
    "DirectionalSurvival",
    "GroupCycleObservation",
    "HyperedgeCycleState",
    "HyperedgeFocalScope",
    "HyperedgeScore",
    "OverlapHypergraphTopology",
    "PolicyConfig",
    "SharedProposal",
    "SharedVariableStar",
    "build_closed_owner_credit",
    "build_delayed_hyperedge_credit",
    "build_group_cycle_observation",
    "build_hyperedge_cycle_states",
    "build_overlap_hypergraph",
    "decide_action",
    "directional_survival",
    "score_hyperedge_states",
]


@dataclass(frozen=True)
class PolicyConfig:
    min_feature_coverage: float = 0.80
    min_budget_remaining_ratio: float = 0.20
    high_conflict_threshold: float = 0.75
    high_shared_support_threshold: float = 0.75
    high_priority_spread_threshold: float = 0.55
    safe_fallback_margin_threshold: float = 0.50
    low_conflict_coordinate_threshold: float = 0.20
    low_shared_coordinate_threshold: float = 0.35
    coordinate_rank_stability_threshold: float = 0.80
    coordinate_fallback_margin_threshold: float = 0.70
    tie_band_conflict_threshold: float = 0.15
    tie_band_shared_threshold: float = 0.15
    tie_band_fallback_margin_threshold: float = 0.90
    tie_band_utility_threshold: float = 0.70


def decide_action(evidence: EvidenceProfile, config: PolicyConfig | None = None) -> ActionDecision:
    """Map legal evidence to a backend intervention decision."""

    cfg = config or PolicyConfig()
    if (
        evidence.feature_coverage < cfg.min_feature_coverage
        or evidence.budget_remaining_ratio < cfg.min_budget_remaining_ratio
    ):
        return ActionDecision(
            ActionFamily.FALLBACK,
            "conservative_no_action",
            "fallback",
            "insufficient_feature_or_budget_coverage",
            0.0,
        )

    conflict_signal = max(evidence.direction_disagreement, evidence.harmful_coord_score)
    shared_signal = max(evidence.shared_var_support_ratio, evidence.overlap_degree)
    protect_signal = evidence.priority_spread * evidence.rank_stability
    coordinate_signal = (
        (1.0 - conflict_signal)
        * (1.0 - shared_signal)
        * evidence.rank_stability
        * evidence.fallback_margin_proxy
    )

    if (
        conflict_signal <= cfg.tie_band_conflict_threshold
        and shared_signal <= cfg.tie_band_shared_threshold
        and evidence.fallback_margin_proxy >= cfg.tie_band_fallback_margin_threshold
        and coordinate_signal < cfg.tie_band_utility_threshold
    ):
        return ActionDecision(
            ActionFamily.FALLBACK,
            "conservative_no_action",
            "fallback",
            "tie_band_no_harm_gate",
            0.0,
        )

    if conflict_signal >= cfg.high_conflict_threshold:
        utility = conflict_signal - (1.0 - evidence.fallback_margin_proxy)
        if utility > 0:
            return ActionDecision(
                ActionFamily.ISOLATE,
                "isolate_conflicting_relation",
                "allow",
                "high_conflict_signal_with_safe_fallback_margin",
                utility,
            )

    if (
        shared_signal >= cfg.high_shared_support_threshold
        and evidence.fallback_margin_proxy >= cfg.safe_fallback_margin_threshold
    ):
        utility = shared_signal + evidence.fallback_margin_proxy - 1.0
        if utility > 0:
            return ActionDecision(
                ActionFamily.REASSIGN_REPAIR,
                "repair_shared_variable_binding",
                "allow",
                "shared_variable_support_with_positive_fallback_margin",
                utility,
            )

    if protect_signal >= cfg.high_priority_spread_threshold:
        utility = protect_signal - 0.10
        if utility > 0:
            return ActionDecision(
                ActionFamily.PROTECT,
                "protect_high_margin_group",
                "allow",
                "stable_priority_spread_supports_resource_protection",
                utility,
            )

    if (
        conflict_signal <= cfg.low_conflict_coordinate_threshold
        and shared_signal <= cfg.low_shared_coordinate_threshold
        and evidence.rank_stability >= cfg.coordinate_rank_stability_threshold
        and evidence.fallback_margin_proxy >= cfg.coordinate_fallback_margin_threshold
    ):
        utility = coordinate_signal - 0.25
        if utility > 0:
            return ActionDecision(
                ActionFamily.COORDINATE,
                "allow_beneficial_coordination",
                "allow",
                "stable_low_conflict_evidence_supports_coordination",
                utility,
            )

    return ActionDecision(
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "fallback",
        "no_positive_reference_blind_action_margin",
        0.0,
    )
