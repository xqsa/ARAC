"""Offline UCB-1 prototype for future selector evaluation.

This module is intentionally not wired into the HCC runner during Action
Validation. It can consume ActionOutcomeLedger credits in isolated tests, but
it does not select runtime actions or replace the current relation policy.

The v1-v26 rule-tree recommendation is used as a prior (pseudo-counts) so
the bandit starts with a sensible warm start instead of a cold uniform prior.

Evidence context bucketing:
  "conflict"     — delta_ratio_gap >= 0.3
  "coordinated"  — delta_ratio_gap <  0.3

Per-bucket, per-action statistics: (sum_credit, count) with UCB exploration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# Prior pseudo-count: one virtual observation per rule-tree recommendation.
# Keeps UCB from diverging on cold arms during the first few sweeps.
_PRIOR_CREDIT = 0.02       # assumed credit for the v26-recommended action
_PRIOR_PSEUDOCOUNT = 3     # equivalent to 3 prior observations
_UCB_EXPLORATION = 0.5     # C in UCB1: higher → more exploration


def _context_bucket(evidence: dict[str, float]) -> str:
    gap = evidence.get("delta_ratio_gap", 0.0)
    return "conflict" if gap >= 0.3 else "coordinated"


@dataclass
class _ArmStats:
    sum_credit: float = 0.0
    count: int = 0

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_credit / self.count

    def ucb(self, total_pulls: int, exploration: float) -> float:
        if self.count == 0:
            return math.inf
        return self.mean() + exploration * math.sqrt(math.log(total_pulls + 1) / self.count)

    def update(self, credit: float) -> None:
        self.sum_credit += credit
        self.count += 1


@dataclass
class ActionBandit:
    """Offline UCB-1 scorer over shared-variable-value actions.

    Usage:
        bandit = ActionBandit(candidate_actions=["native_eq8", "repair_...", ...])
        # at action selection time:
        arm = bandit.select(evidence_snapshot, prior_recommendation="repair_...")
        # after ActionOutcomeLedger closes:
        bandit.update(arm, evidence_snapshot, penalized_credit)
    """

    candidate_actions: list[str]
    _stats: dict[tuple[str, str], _ArmStats] = field(
        default_factory=dict, init=False, repr=False
    )
    _bucket_pulls: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.candidate_actions or any(not arm for arm in self.candidate_actions):
            raise ValueError("candidate actions must be non-empty strings")
        if len(set(self.candidate_actions)) != len(self.candidate_actions):
            raise ValueError("candidate actions must be unique")

    def _get_stats(self, arm: str, bucket: str) -> _ArmStats:
        key = (arm, bucket)
        if key not in self._stats:
            self._stats[key] = _ArmStats()
        return self._stats[key]

    def _inject_prior(self, arm: str, bucket: str) -> None:
        """Inject one pseudo-count for the given arm/bucket if not yet initialised."""
        stats = self._get_stats(arm, bucket)
        if stats.count == 0:
            stats.sum_credit = _PRIOR_CREDIT * _PRIOR_PSEUDOCOUNT
            stats.count = _PRIOR_PSEUDOCOUNT
            self._bucket_pulls[bucket] = (
                self._bucket_pulls.get(bucket, 0) + _PRIOR_PSEUDOCOUNT
            )

    def select(
        self,
        evidence_snapshot: dict[str, float],
        *,
        prior_recommendation: str | None = None,
        candidate_override: Sequence[str] | None = None,
    ) -> str:
        """Select an action using UCB-1 in the evidence context bucket.

        prior_recommendation: the v26-rule-tree answer — gets a warm prior.
        candidate_override: restrict candidates for this call.
        """
        bucket = _context_bucket(evidence_snapshot)
        candidates = list(candidate_override or self.candidate_actions)
        if not candidates or any(arm not in self.candidate_actions for arm in candidates):
            raise ValueError("candidate override must select registered actions")
        # Inject warm prior for the rule-tree recommendation
        if prior_recommendation and prior_recommendation in candidates:
            self._inject_prior(prior_recommendation, bucket)
        total = self._bucket_pulls.get(bucket, 0)
        best_arm = candidates[0]
        best_score = -math.inf
        for arm in candidates:
            score = self._get_stats(arm, bucket).ucb(total, _UCB_EXPLORATION)
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

    def update(self, arm: str, evidence_snapshot: dict[str, float], credit: float) -> None:
        """Record an outcome credit for arm in the evidence context bucket."""
        if arm not in self.candidate_actions:
            raise ValueError("cannot update an unregistered action")
        if not math.isfinite(float(credit)):
            raise ValueError("action credit must be finite")
        bucket = _context_bucket(evidence_snapshot)
        stats = self._get_stats(arm, bucket)
        stats.update(float(credit))
        self._bucket_pulls[bucket] = self._bucket_pulls.get(bucket, 0) + 1

    def arm_mean_credit(self, arm: str, evidence_snapshot: dict[str, float]) -> float | None:
        bucket = _context_bucket(evidence_snapshot)
        stats = self._stats.get((arm, bucket))
        if stats is None or stats.count == 0:
            return None
        return stats.mean()

    def summary(self) -> dict[str, dict[str, float]]:
        """Return {arm: {bucket: mean_credit}} for all observed (arm, bucket) pairs."""
        result: dict[str, dict[str, float]] = {}
        for (arm, bucket), stats in self._stats.items():
            if stats.count > 0:
                result.setdefault(arm, {})[bucket] = stats.mean()
        return result
