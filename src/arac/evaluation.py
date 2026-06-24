"""Same-budget and utility helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SameBudgetLedger:
    phase_i_fe: int
    phase_ii_fe: int
    budget_limit: int
    fresh_execution: bool

    @property
    def total_fe(self) -> int:
        return self.phase_i_fe + self.phase_ii_fe

    @property
    def violation(self) -> bool:
        return self.total_fe > self.budget_limit


def relative_gain(fallback_error: float, action_error: float) -> float:
    denominator = max(abs(fallback_error), 1e-12)
    return (fallback_error - action_error) / denominator


def classify_utility(
    fallback_error: float,
    action_error: float,
    meaningful_gain_threshold: float = 0.05,
    catastrophic_loss_threshold: float = -0.20,
) -> str:
    gain = relative_gain(fallback_error, action_error)
    if gain >= meaningful_gain_threshold:
        return "meaningful_win"
    if gain <= catastrophic_loss_threshold:
        return "catastrophic_loss"
    return "tie_or_small_effect"

