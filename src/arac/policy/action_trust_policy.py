"""Runtime-only trust guard for overlap-relation actions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ActionTrustConfig:
    initial_strength: float = 1.0
    probation_strength: float = 0.20
    trusted_strength: float = 1.0
    positive_credit_threshold: float = 1e-6
    promotion_streak: int = 2
    quarantine_streak: int = 2
    cooldown_steps: int = 2
    exposure_cap: float = 4.0
    ewma_alpha: float = 0.50
    risk_beta: float = 0.50

    def __post_init__(self) -> None:
        for name in ("initial_strength", "probation_strength", "trusted_strength"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.positive_credit_threshold < 0.0:
            raise ValueError("positive_credit_threshold must be non-negative")
        if self.promotion_streak < 1 or self.quarantine_streak < 1:
            raise ValueError("streak thresholds must be positive")
        if self.cooldown_steps < 1:
            raise ValueError("cooldown_steps must be positive")
        if not math.isfinite(self.exposure_cap) or self.exposure_cap <= 0.0:
            raise ValueError("exposure_cap must be finite and positive")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not math.isfinite(self.risk_beta) or self.risk_beta < 0.0:
            raise ValueError("risk_beta must be finite and non-negative")


@dataclass
class ActionTrustState:
    phase: str = "probation"
    attempt_count: int = 0
    observation_count: int = 0
    gain_mean: float = 0.0
    gain_variance: float = 0.0
    positive_streak: int = 0
    low_gain_streak: int = 0
    instability_count: int = 0
    cooldown_remaining: int = 0
    exposure: float = 0.0
    last_credit: float | None = None
    last_unstable: bool = False
    has_observed_risk: bool = False


@dataclass(frozen=True)
class ActionTrustDecision:
    key: str
    phase: str
    allow_intervention: bool
    blend_strength: float
    reason: str
    trust_score: float
    attempt_count: int
    exposure: float
    cooldown_remaining: int


class ActionTrustPolicy:
    """Keep bounded trust state for one optimizer run."""

    def __init__(self, config: ActionTrustConfig | None = None) -> None:
        self.config = config or ActionTrustConfig()
        self._states: dict[str, ActionTrustState] = {}

    def state_for(self, key: str) -> ActionTrustState:
        """Return a snapshot so callers cannot mutate policy state."""

        return replace(self._state(key))

    def decide(self, key: str) -> ActionTrustDecision:
        if not key:
            raise ValueError("action trust key must not be empty")
        state = self._state(key)

        if state.phase == "quarantined" and state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1
            decision = self._decision(
                key,
                state,
                allow_intervention=False,
                blend_strength=0.0,
                reason="cooldown_active",
            )
            if state.cooldown_remaining == 0:
                state.phase = "probation"
                state.positive_streak = 0
                state.low_gain_streak = 0
            return decision

        if state.phase == "trusted":
            strength = self.config.trusted_strength
            reason = "trusted"
        elif state.has_observed_risk:
            strength = self.config.probation_strength
            reason = "probation_limited"
        else:
            strength = self.config.initial_strength
            reason = "probation_shadow"
        if state.exposure + strength > self.config.exposure_cap + 1e-12:
            state.phase = "quarantined"
            return self._decision(
                key,
                state,
                allow_intervention=False,
                blend_strength=0.0,
                reason="exposure_cap_reached",
            )

        state.attempt_count += 1
        state.exposure += strength
        return self._decision(
            key,
            state,
            allow_intervention=True,
            blend_strength=strength,
            reason=reason,
        )

    def observe(
        self,
        key: str,
        *,
        credit: float,
        unstable: bool,
    ) -> ActionTrustState:
        credit = float(credit)
        if not math.isfinite(credit):
            raise ValueError("credit must be finite")
        state = self._state(key)
        state.last_credit = credit
        state.last_unstable = bool(unstable)
        state.observation_count += 1

        if state.observation_count == 1:
            state.gain_mean = credit
            state.gain_variance = 0.0
        else:
            difference = credit - state.gain_mean
            alpha = self.config.ewma_alpha
            state.gain_mean += alpha * difference
            state.gain_variance = max(
                0.0,
                (1.0 - alpha)
                * (state.gain_variance + alpha * difference * difference),
            )

        material_positive = (
            credit > self.config.positive_credit_threshold and not unstable
        )
        if material_positive:
            state.positive_streak += 1
            state.low_gain_streak = 0
            if state.positive_streak >= self.config.promotion_streak:
                state.phase = "trusted"
        else:
            state.positive_streak = 0
            state.low_gain_streak += 1
            state.has_observed_risk = True
            if unstable:
                state.instability_count += 1
            if state.low_gain_streak >= self.config.quarantine_streak:
                state.phase = "quarantined"
                state.cooldown_remaining = self.config.cooldown_steps
        return replace(state)

    def rollback_decision(
        self,
        decision: ActionTrustDecision,
    ) -> ActionTrustState:
        """Undo the latest unobserved intervention decision for a no-op."""

        if not decision.allow_intervention:
            raise ValueError("only an allowed intervention can be rolled back")
        state = self._states.get(decision.key)
        if state is None:
            raise RuntimeError("cannot roll back an unknown trust decision")
        if (
            state.attempt_count != decision.attempt_count
            or not math.isclose(
                state.exposure,
                decision.exposure,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("only the latest unobserved trust decision can be rolled back")
        state.attempt_count -= 1
        state.exposure = max(0.0, state.exposure - decision.blend_strength)
        return replace(state)

    def _state(self, key: str) -> ActionTrustState:
        return self._states.setdefault(key, ActionTrustState())

    def _decision(
        self,
        key: str,
        state: ActionTrustState,
        *,
        allow_intervention: bool,
        blend_strength: float,
        reason: str,
    ) -> ActionTrustDecision:
        trust_score = state.gain_mean - self.config.risk_beta * math.sqrt(
            max(0.0, state.gain_variance)
        )
        return ActionTrustDecision(
            key=key,
            phase=state.phase,
            allow_intervention=allow_intervention,
            blend_strength=blend_strength,
            reason=reason,
            trust_score=trust_score,
            attempt_count=state.attempt_count,
            exposure=state.exposure,
            cooldown_remaining=state.cooldown_remaining,
        )


def make_action_key(
    *,
    group_left: int,
    group_right: int,
    shared_vars: tuple[int, ...],
    canonical_action_name: str,
) -> str:
    if not canonical_action_name:
        raise ValueError("canonical_action_name must not be empty")
    normalized_vars = tuple(sorted(set(int(value) for value in shared_vars)))
    payload = ";".join(str(value) for value in normalized_vars).encode("utf-8")
    shared_hash = hashlib.sha256(payload).hexdigest()[:16]
    return f"{int(group_left)}:{int(group_right)}:{shared_hash}:{canonical_action_name}"


def robust_damped_writeback(
    *,
    current_values: np.ndarray,
    proposed_values: np.ndarray,
    blend_strength: float,
    max_delta_norm: float,
) -> np.ndarray:
    current = np.asarray(current_values, dtype=float)
    proposal = np.asarray(proposed_values, dtype=float)
    if current.shape != proposal.shape:
        raise ValueError("current and proposed values must have the same shape")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(proposal)):
        raise ValueError("writeback values must be finite")
    strength = float(blend_strength)
    max_norm = float(max_delta_norm)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("blend_strength must be finite and in [0, 1]")
    if not math.isfinite(max_norm) or max_norm < 0.0:
        raise ValueError("max_delta_norm must be finite and non-negative")

    step = strength * (proposal - current)
    step_norm = float(np.linalg.norm(step))
    if step_norm > max_norm and step_norm > 0.0:
        step = step * (max_norm / step_norm)
    adjusted = current + step
    if not np.all(np.isfinite(adjusted)):
        raise ValueError("robust writeback produced non-finite values")
    return adjusted


def normalized_objective_credit(
    baseline_fitness: float,
    downstream_fitness: float,
) -> float:
    baseline = float(baseline_fitness)
    downstream = float(downstream_fitness)
    if not math.isfinite(baseline) or not math.isfinite(downstream):
        raise ValueError("objective credit inputs must be finite")
    scale = max(abs(baseline), abs(downstream))
    if scale <= 1e-12:
        return 0.0
    return float(np.clip((baseline - downstream) / scale, -1.0, 1.0))
