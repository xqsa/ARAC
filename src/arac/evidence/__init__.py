"""Reference-blind evidence profile definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceProfile:
    """Runtime-legal Phase-I evidence.

    Do not add final errors, relative gains, oracle labels, reported baselines,
    or problem-family labels to this dataclass.
    """

    run_id: str
    problem_id: str
    seed: int
    unit_type: str
    unit_id: str
    feature_coverage: float
    overlap_degree: float
    shared_var_support_ratio: float
    direction_disagreement: float
    harmful_coord_score: float
    group_gain_asymmetry: float
    priority_spread: float
    rank_stability: float
    budget_remaining_ratio: float
    fallback_margin_proxy: float


FORBIDDEN_RUNTIME_FIELDS = {
    "final_error",
    "relative_gain",
    "oracle",
    "oracle_best_action",
    "user_reference_best",
    "reported_baseline",
    "reported_mean",
    "reported_std",
    "paper_reported_mean",
    "paper_reported_std",
    "paper_reported_time",
    "problem_family_label",
    "problem_family",
    "base_function",
    "prior_final_outcome",
    "prior_pilot_outcome",
}


def validate_runtime_payload(payload: dict) -> None:
    forbidden = sorted(FORBIDDEN_RUNTIME_FIELDS.intersection(payload))
    if forbidden:
        raise ValueError(f"forbidden runtime fields present: {', '.join(forbidden)}")
