"""Native binary cooperative-coevolution backend for the LSGO benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from arac.benchmarks.binary_lsgo import BinaryLsgoProblem, BinaryLsgoTopology
from arac.evidence import EvidenceProfile, validate_runtime_payload


@dataclass(frozen=True)
class BinaryLsgoGroupStats:
    group_index: int
    proposed: int = 0
    accepted: int = 0
    gain: float = 0.0
    early_gain: float = 0.0
    late_gain: float = 0.0


@dataclass(frozen=True)
class BinaryLsgoSnapshot:
    run_id: str
    lane_id: str
    problem_id: str
    optimizer_seed: int
    consumed_fes: int
    total_fes: int
    group_stats: tuple[BinaryLsgoGroupStats, ...]
    shared_proposals: int
    rejected_shared_proposals: int
    conflicting_shared_variables: int
    rank_stability: float
    topology: BinaryLsgoTopology


@dataclass(frozen=True)
class BinaryLsgoExecutionRequest:
    problem: BinaryLsgoProblem
    optimizer_seed: int
    total_fes: int = 2_000
    phase_one_fraction: float = 0.20
    run_id: str = "binary_lsgo_arac"
    lane_id: str = "arac_policy"

    def __post_init__(self) -> None:
        if isinstance(self.optimizer_seed, bool) or not isinstance(self.optimizer_seed, int):
            raise ValueError("optimizer_seed must be an integer")
        if self.optimizer_seed < 0:
            raise ValueError("optimizer_seed must be non-negative")
        if isinstance(self.total_fes, bool) or not isinstance(self.total_fes, int):
            raise ValueError("total_fes must be an integer")
        if self.total_fes < 2:
            raise ValueError("total_fes must be at least 2")
        if isinstance(self.phase_one_fraction, bool) or not isinstance(
            self.phase_one_fraction, (int, float)
        ):
            raise ValueError("phase_one_fraction must be a real number")
        if not 0.0 < self.phase_one_fraction < 1.0:
            raise ValueError("phase_one_fraction must be in (0, 1)")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.lane_id.strip():
            raise ValueError("lane_id must be non-empty")

    @property
    def phase_one_fes(self) -> int:
        return max(1, min(self.total_fes - 1, round(self.total_fes * self.phase_one_fraction)))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def build_binary_lsgo_evidence_profile(snapshot: BinaryLsgoSnapshot) -> EvidenceProfile:
    payload = {
        "run_id": snapshot.run_id,
        "lane_id": snapshot.lane_id,
        "problem_id": snapshot.problem_id,
        "optimizer_seed": snapshot.optimizer_seed,
        "consumed_fes": snapshot.consumed_fes,
        "total_fes": snapshot.total_fes,
    }
    validate_runtime_payload(payload)

    gains = [max(0.0, item.gain) for item in snapshot.group_stats]
    maximum_gain = max(gains, default=0.0)
    gain_asymmetry = _ratio(maximum_gain - min(gains, default=0.0), maximum_gain + 1e-12)
    group_count = len(snapshot.topology.groups)
    possible_pairs = group_count * (group_count - 1) / 2
    overlap_degree = _ratio(len(snapshot.topology.adjacency_pairs), possible_pairs)
    shared_support = _ratio(
        snapshot.topology.shared_variable_count,
        snapshot.topology.decision_dimension,
    )
    harmful = _ratio(snapshot.rejected_shared_proposals, snapshot.shared_proposals)
    conflict = _ratio(
        snapshot.conflicting_shared_variables,
        snapshot.topology.shared_variable_count,
    )
    covered_groups = sum(item.proposed > 0 for item in snapshot.group_stats)

    return EvidenceProfile(
        run_id=snapshot.run_id,
        problem_id=snapshot.problem_id,
        seed=snapshot.optimizer_seed,
        unit_type="problem",
        unit_id=f"binary_lsgo_backend:{snapshot.problem_id}",
        feature_coverage=_ratio(covered_groups, group_count),
        overlap_degree=overlap_degree,
        shared_var_support_ratio=shared_support,
        direction_disagreement=conflict,
        harmful_coord_score=harmful,
        group_gain_asymmetry=gain_asymmetry,
        priority_spread=gain_asymmetry,
        rank_stability=_ratio(snapshot.rank_stability, 1.0),
        budget_remaining_ratio=_ratio(
            snapshot.total_fes - snapshot.consumed_fes,
            snapshot.total_fes,
        ),
        fallback_margin_proxy=1.0 - harmful,
    )


__all__ = [
    "BinaryLsgoExecutionRequest",
    "BinaryLsgoGroupStats",
    "BinaryLsgoSnapshot",
    "build_binary_lsgo_evidence_profile",
]
