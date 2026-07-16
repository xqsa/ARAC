"""Derived-response gate for the one-shot precision response probe."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, fields
from typing import Mapping, Sequence


PRECISION_RESPONSE_PROTOCOL_VERSION = "precision-response-loop-v1"
PRECISION_PROBE_GATE_SCHEMA_VERSION = "precision-probe-gate-v1"
PRECISION_PROBE_GATE_FIELDS = (
    "valid",
    "invalid_reason",
    "pair_count",
    "direction_hash_match",
    "paired_win_count",
    "paired_win_rate",
    "paired_win_lcb",
    "median_relative_advantage",
    "large_loss_count",
    "large_loss_rate",
    "large_loss_ucb",
    "standardized_diversity_ratio",
    "normal_boundary_hit_count",
    "precision_boundary_hit_count",
    "normal_boundary_hit_rate",
    "precision_boundary_hit_rate",
    "precision_best_relative_gain",
)
FORBIDDEN_GATE_FIELD_FRAGMENTS = (
    "case",
    "problem",
    "seed",
    "family",
    "group",
    "component",
    "fingerprint",
    "objective",
    "incumbent",
    "paper",
    "outcome",
    "terminal",
)


@dataclass(frozen=True)
class PrecisionProbeConfig:
    pair_count: int
    probe_fe: int
    boundary_mode: str
    precision_to_normal_sigma_ratio: float
    confidence_level: float
    win_lcb_threshold: float
    large_loss_multiplier: float
    large_loss_ucb_threshold: float
    median_relative_advantage_threshold: float
    standardized_diversity_ratio_min: float
    precision_boundary_hit_rate_max: float
    precision_best_relative_gain_threshold: float


@dataclass(frozen=True)
class PrecisionProbeGateState:
    """Immutable derived response; raw objective values cannot enter this type."""

    valid: bool
    invalid_reason: str
    pair_count: int
    direction_hash_match: bool
    paired_win_count: int
    paired_win_rate: float
    paired_win_lcb: float
    median_relative_advantage: float
    large_loss_count: int
    large_loss_rate: float
    large_loss_ucb: float
    standardized_diversity_ratio: float
    normal_boundary_hit_count: int
    precision_boundary_hit_count: int
    normal_boundary_hit_rate: float
    precision_boundary_hit_rate: float
    precision_best_relative_gain: float

    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PrecisionProbeGateDecision:
    release: bool
    reason: str
    state_sha256: str


def precision_probe_config_from_mapping(
    payload: Mapping[str, object],
) -> PrecisionProbeConfig:
    if payload.get("protocol_version") != PRECISION_RESPONSE_PROTOCOL_VERSION:
        raise ValueError("unsupported precision response protocol")
    probe = payload.get("probe")
    gate = payload.get("gate")
    if not isinstance(probe, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("precision response config requires probe and gate mappings")
    config = PrecisionProbeConfig(
        pair_count=int(probe["pair_count"]),
        probe_fe=int(probe["probe_fe"]),
        boundary_mode=str(probe["boundary_mode"]),
        precision_to_normal_sigma_ratio=float(
            probe["precision_to_normal_sigma_ratio"]
        ),
        confidence_level=float(gate["confidence_level"]),
        win_lcb_threshold=float(gate["win_lcb_threshold"]),
        large_loss_multiplier=float(gate["large_loss_multiplier"]),
        large_loss_ucb_threshold=float(gate["large_loss_ucb_threshold"]),
        median_relative_advantage_threshold=float(
            gate["median_relative_advantage_threshold"]
        ),
        standardized_diversity_ratio_min=float(
            gate["standardized_diversity_ratio_min"]
        ),
        precision_boundary_hit_rate_max=float(
            gate["precision_boundary_hit_rate_max"]
        ),
        precision_best_relative_gain_threshold=float(
            gate["precision_best_relative_gain_threshold"]
        ),
    )
    if config.pair_count != 16 or config.probe_fe != 2 * config.pair_count:
        raise ValueError("v1 requires exactly 16 pairs and 32 probe FE")
    if config.boundary_mode != "legacy_none":
        raise ValueError("v1 requires legacy_none boundary behavior")
    if not 0.0 < config.precision_to_normal_sigma_ratio < 1.0:
        raise ValueError("precision sigma ratio must be in (0, 1)")
    if not 0.5 < config.confidence_level < 1.0:
        raise ValueError("confidence level must be in (0.5, 1)")
    return config


def _wilson_bounds(successes: int, total: int, confidence: float) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    z = statistics.NormalDist().inv_cdf(confidence)
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def build_precision_probe_gate_state(
    *,
    normal_errors: Sequence[float],
    precision_errors: Sequence[float],
    checkpoint_error: float,
    direction_hash_match: bool,
    standardized_diversity_ratio: float,
    normal_boundary_hit_count: int,
    precision_boundary_hit_count: int,
    config: PrecisionProbeConfig,
) -> PrecisionProbeGateState:
    normal = tuple(float(value) for value in normal_errors)
    precision = tuple(float(value) for value in precision_errors)
    pair_count = len(normal)
    invalid_reason = ""
    if pair_count != config.pair_count or len(precision) != pair_count:
        invalid_reason = "invalid_pair_count"
    elif not all(math.isfinite(value) and value >= 0.0 for value in (*normal, *precision)):
        invalid_reason = "invalid_probe_error"
    elif not math.isfinite(float(checkpoint_error)) or float(checkpoint_error) < 0.0:
        invalid_reason = "invalid_checkpoint_error"
    elif not math.isfinite(float(standardized_diversity_ratio)):
        invalid_reason = "invalid_standardized_diversity"
    elif not 0 <= int(normal_boundary_hit_count) <= pair_count:
        invalid_reason = "invalid_normal_boundary_hits"
    elif not 0 <= int(precision_boundary_hit_count) <= pair_count:
        invalid_reason = "invalid_precision_boundary_hits"

    if invalid_reason:
        return PrecisionProbeGateState(
            valid=False,
            invalid_reason=invalid_reason,
            pair_count=pair_count,
            direction_hash_match=bool(direction_hash_match),
            paired_win_count=0,
            paired_win_rate=0.0,
            paired_win_lcb=0.0,
            median_relative_advantage=0.0,
            large_loss_count=0,
            large_loss_rate=0.0,
            large_loss_ucb=1.0,
            standardized_diversity_ratio=0.0,
            normal_boundary_hit_count=max(0, int(normal_boundary_hit_count)),
            precision_boundary_hit_count=max(0, int(precision_boundary_hit_count)),
            normal_boundary_hit_rate=1.0,
            precision_boundary_hit_rate=1.0,
            precision_best_relative_gain=0.0,
        )

    wins = sum(p_value < b_value for b_value, p_value in zip(normal, precision))
    large_losses = sum(
        p_value >= config.large_loss_multiplier * max(b_value, 1e-300)
        for b_value, p_value in zip(normal, precision)
    )
    relative_advantages = tuple(
        max(
            -1.0,
            min(
                1.0,
                (b_value - p_value)
                / max(abs(b_value), abs(p_value), 1e-300),
            ),
        )
        for b_value, p_value in zip(normal, precision)
    )
    win_lcb, _ = _wilson_bounds(wins, pair_count, config.confidence_level)
    _, loss_ucb = _wilson_bounds(
        large_losses,
        pair_count,
        config.confidence_level,
    )
    checkpoint = float(checkpoint_error)
    precision_best_gain = (checkpoint - min(precision)) / max(abs(checkpoint), 1e-300)
    return PrecisionProbeGateState(
        valid=True,
        invalid_reason="",
        pair_count=pair_count,
        direction_hash_match=bool(direction_hash_match),
        paired_win_count=wins,
        paired_win_rate=wins / pair_count,
        paired_win_lcb=win_lcb,
        median_relative_advantage=float(statistics.median(relative_advantages)),
        large_loss_count=large_losses,
        large_loss_rate=large_losses / pair_count,
        large_loss_ucb=loss_ucb,
        standardized_diversity_ratio=float(standardized_diversity_ratio),
        normal_boundary_hit_count=int(normal_boundary_hit_count),
        precision_boundary_hit_count=int(precision_boundary_hit_count),
        normal_boundary_hit_rate=int(normal_boundary_hit_count) / pair_count,
        precision_boundary_hit_rate=int(precision_boundary_hit_count) / pair_count,
        precision_best_relative_gain=precision_best_gain,
    )


def decide_precision_probe(
    state: PrecisionProbeGateState,
    config: PrecisionProbeConfig,
) -> PrecisionProbeGateDecision:
    checks = (
        (state.valid, state.invalid_reason or "invalid_response_state"),
        (state.pair_count == config.pair_count, "invalid_pair_count"),
        (state.direction_hash_match, "paired_direction_mismatch"),
        (state.paired_win_lcb > config.win_lcb_threshold, "win_lcb_not_positive"),
        (
            state.large_loss_ucb <= config.large_loss_ucb_threshold,
            "large_loss_ucb_exceeded",
        ),
        (
            state.median_relative_advantage
            > config.median_relative_advantage_threshold,
            "median_advantage_not_positive",
        ),
        (
            state.standardized_diversity_ratio
            >= config.standardized_diversity_ratio_min,
            "standardized_diversity_collapsed",
        ),
        (
            state.precision_boundary_hit_rate
            <= config.precision_boundary_hit_rate_max,
            "precision_boundary_rate_exceeded",
        ),
        (
            state.precision_boundary_hit_rate <= state.normal_boundary_hit_rate,
            "precision_boundary_rate_worse_than_normal",
        ),
        (
            state.precision_best_relative_gain
            > config.precision_best_relative_gain_threshold,
            "precision_best_not_improving",
        ),
    )
    for passed, reason in checks:
        if not passed:
            return PrecisionProbeGateDecision(False, reason, state.sha256())
    return PrecisionProbeGateDecision(True, "precision_response_gate_passed", state.sha256())


def assert_gate_schema_has_no_forbidden_fields() -> None:
    names = {item.name for item in fields(PrecisionProbeGateState)}
    if names != set(PRECISION_PROBE_GATE_FIELDS):
        raise RuntimeError("precision probe gate schema drift")
    leaked = sorted(
        name
        for name in names
        if any(fragment in name.lower() for fragment in FORBIDDEN_GATE_FIELD_FRAGMENTS)
    )
    if leaked:
        raise RuntimeError(f"forbidden precision gate fields: {','.join(leaked)}")


assert_gate_schema_has_no_forbidden_fields()
