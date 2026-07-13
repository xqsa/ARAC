"""Native binary cooperative-coevolution backend for the LSGO benchmark."""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, replace
from numbers import Real
from typing import Callable

from arac.action_space import ActionFamily
from arac.benchmarks.binary_lsgo import BinaryLsgoProblem, BinaryLsgoTopology
from arac.evaluation import SameBudgetLedger
from arac.evidence import EvidenceProfile, validate_runtime_payload
from arac.policy import ActionDecision, decide_action


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
        if isinstance(self.phase_one_fraction, bool) or not isinstance(self.phase_one_fraction, Real):
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


@dataclass(frozen=True)
class BinaryBackendSemanticsDiff:
    variable_owner_changed: bool = False
    relation_handling_changed: bool = False
    coordination_mode_changed: bool = False
    budget_allocation_changed: bool = False

    @property
    def changed(self) -> bool:
        return any(
            (
                self.variable_owner_changed,
                self.relation_handling_changed,
                self.coordination_mode_changed,
                self.budget_allocation_changed,
            )
        )


@dataclass(frozen=True)
class BinaryLsgoActionTrace:
    action_name: str
    decision: str
    trigger_reason: str
    phase: str
    affected_group_count: int
    affected_shared_variable_count: int
    allocated_fe: int
    consumed_fe: int


@dataclass(frozen=True)
class BinaryLsgoExecutionResult:
    run_id: str
    lane_id: str
    problem_id: str
    optimizer_seed: int
    initial_vector_hash: str
    phase_one_objective: float
    final_objective: float
    final_vector: tuple[int, ...]
    evidence: EvidenceProfile
    decision: ActionDecision
    semantics: BinaryBackendSemanticsDiff
    ledger: SameBudgetLedger
    action_trace: BinaryLsgoActionTrace
    optimizer_consumed: bool


SUPPORTED_ACTIONS = {
    "conservative_no_action": "native_round_robin",
    "allow_beneficial_coordination": "prioritize_related_groups_after_shared_accept",
    "isolate_conflicting_relation": "owner_only_shared_write",
    "repair_shared_variable_binding": "gain_ranked_owner_only_shared_write",
    "protect_high_margin_group": "gain_weighted_group_schedule",
}


@dataclass
class _MutableGroupStats:
    proposed: int = 0
    accepted: int = 0
    gain: float = 0.0
    early_gain: float = 0.0
    late_gain: float = 0.0


def _rank_stability(stats: list[_MutableGroupStats]) -> float:
    if len(stats) < 2:
        return 1.0
    early_order = sorted(range(len(stats)), key=lambda index: (-stats[index].early_gain, index))
    late_order = sorted(range(len(stats)), key=lambda index: (-stats[index].late_gain, index))
    concordant = 0
    total = 0
    for left in range(len(stats)):
        for right in range(left + 1, len(stats)):
            early_sign = early_order.index(left) < early_order.index(right)
            late_sign = late_order.index(left) < late_order.index(right)
            concordant += early_sign == late_sign
            total += 1
    return _ratio(concordant, total)


def _initial_vector_hash(vector: list[int]) -> str:
    return hashlib.sha256(bytes(vector)).hexdigest()


def _shared_variables(topology: BinaryLsgoTopology) -> dict[int, tuple[int, ...]]:
    return dict(topology.shared_variable_groups)


def _stats_snapshot(stats: list[_MutableGroupStats]) -> tuple[BinaryLsgoGroupStats, ...]:
    return tuple(
        BinaryLsgoGroupStats(
            group_index=index,
            proposed=item.proposed,
            accepted=item.accepted,
            gain=item.gain,
            early_gain=item.early_gain,
            late_gain=item.late_gain,
        )
        for index, item in enumerate(stats)
    )


def _action_semantics(action_name: str) -> BinaryBackendSemanticsDiff:
    if action_name == "allow_beneficial_coordination":
        return BinaryBackendSemanticsDiff(coordination_mode_changed=True)
    if action_name == "isolate_conflicting_relation":
        return BinaryBackendSemanticsDiff(relation_handling_changed=True)
    if action_name == "repair_shared_variable_binding":
        return BinaryBackendSemanticsDiff(variable_owner_changed=True)
    if action_name == "protect_high_margin_group":
        return BinaryBackendSemanticsDiff(budget_allocation_changed=True)
    if action_name == "conservative_no_action":
        return BinaryBackendSemanticsDiff()
    raise ValueError(f"unsupported binary LSGO action: {action_name}")


def _choose_group_variable(
    group: tuple[int, ...],
    *,
    action_name: str,
    group_index: int,
    owners: dict[int, int],
    shared_variables: dict[int, tuple[int, ...]],
    rng: random.Random,
) -> int | None:
    if action_name not in {"isolate_conflicting_relation", "repair_shared_variable_binding"}:
        eligible = list(group)
    else:
        eligible = [
            variable
            for variable in group
            if variable not in shared_variables or owners[variable] == group_index
        ]
    return None if not eligible else rng.choice(eligible)


def _protected_schedule(stats: list[_MutableGroupStats]) -> list[int]:
    if not stats:
        return []
    ranked = sorted(range(len(stats)), key=lambda index: (-stats[index].gain, index))
    protected_count = max(1, (len(ranked) + 3) // 4)
    protected = ranked[:protected_count]
    return protected + list(range(len(stats)))


def run_binary_lsgo(
    request: BinaryLsgoExecutionRequest,
    *,
    decision_override: ActionDecision | None = None,
    decision_provider: Callable[[EvidenceProfile], ActionDecision] = decide_action,
) -> BinaryLsgoExecutionResult:
    """Run the exact two-phase binary backend for one lane."""

    problem = request.problem
    topology = problem.topology
    group_count = len(topology.groups)
    if group_count == 0:
        raise ValueError("binary LSGO topology must contain at least one group")

    rng = random.Random(request.optimizer_seed)
    vector = [rng.randrange(2) for _ in range(problem.decision_dimension)]
    initial_hash = _initial_vector_hash(vector)
    current_objective = problem.evaluate(tuple(vector))
    consumed_fes = 1
    phase_one_target = request.phase_one_fes
    stats = [_MutableGroupStats() for _ in topology.groups]
    shared_variables = _shared_variables(topology)
    shared_proposals = 0
    rejected_shared_proposals = 0
    shared_outcomes: dict[int, dict[int, bool]] = {}
    first_phase_split = 1 + (phase_one_target - 1) // 2

    while consumed_fes < phase_one_target:
        group_index = (consumed_fes - 1) % group_count
        group = topology.groups[group_index]
        variable = rng.choice(group)
        candidate = list(vector)
        candidate[variable] = 1 - candidate[variable]
        candidate_objective = problem.evaluate(tuple(candidate))
        consumed_fes += 1
        improvement = current_objective - candidate_objective
        item = stats[group_index]
        item.proposed += 1
        is_shared = variable in shared_variables
        if is_shared:
            shared_proposals += 1
        accepted = candidate_objective < current_objective
        if accepted:
            vector = candidate
            current_objective = candidate_objective
            item.accepted += 1
            item.gain += improvement
            if consumed_fes <= first_phase_split:
                item.early_gain += improvement
            else:
                item.late_gain += improvement
        elif is_shared:
            rejected_shared_proposals += 1
        if is_shared:
            shared_outcomes.setdefault(variable, {})[group_index] = accepted

    conflicting_shared_variables = sum(
        len(set(outcomes.values())) > 1 for outcomes in shared_outcomes.values()
    )
    phase_one_objective = current_objective
    snapshot = BinaryLsgoSnapshot(
        run_id=request.run_id,
        lane_id=request.lane_id,
        problem_id=problem.spec.problem_id,
        optimizer_seed=request.optimizer_seed,
        consumed_fes=consumed_fes,
        total_fes=request.total_fes,
        group_stats=_stats_snapshot(stats),
        shared_proposals=shared_proposals,
        rejected_shared_proposals=rejected_shared_proposals,
        conflicting_shared_variables=conflicting_shared_variables,
        rank_stability=_rank_stability(stats),
        topology=topology,
    )
    if request.lane_id == "shuffled_evidence_negative_control":
        shuffle_rng = random.Random(request.optimizer_seed + 1_000_003)
        shuffled = list(snapshot.group_stats)
        shuffle_rng.shuffle(shuffled)
        snapshot = replace(snapshot, group_stats=tuple(shuffled))
    evidence = build_binary_lsgo_evidence_profile(snapshot)

    if decision_override is not None:
        decision = decision_override
    elif request.lane_id == "native_baseline":
        decision = ActionDecision(
            ActionFamily.FALLBACK,
            "conservative_no_action",
            "fallback",
            "native_baseline_lane",
            0.0,
        )
    else:
        decision = decision_provider(evidence)
    semantics = _action_semantics(decision.action_name)

    owners = {
        variable: min(group_indices)
        for variable, group_indices in shared_variables.items()
    }
    if decision.action_name == "repair_shared_variable_binding":
        owners = {
            variable: max(
                group_indices,
                key=lambda index: (stats[index].gain, -index),
            )
            for variable, group_indices in shared_variables.items()
        }

    phase_two_budget = request.total_fes - phase_one_target
    group_schedule = (
        _protected_schedule(stats)
        if decision.action_name == "protect_high_margin_group"
        else list(range(group_count))
    )
    if not group_schedule:
        raise ValueError("binary LSGO topology produced an empty Phase-II schedule")
    schedule_cursor = 0
    empty_scans = 0
    coordinated_queue: deque[int] = deque()
    while consumed_fes < request.total_fes:
        if coordinated_queue:
            group_index = coordinated_queue.popleft()
        else:
            group_index = group_schedule[schedule_cursor % len(group_schedule)]
            schedule_cursor += 1
        group = topology.groups[group_index]
        variable = _choose_group_variable(
            group,
            action_name=decision.action_name,
            group_index=group_index,
            owners=owners,
            shared_variables=shared_variables,
            rng=rng,
        )
        if variable is None:
            empty_scans += 1
            if empty_scans >= len(group_schedule) and not coordinated_queue:
                raise ValueError("binary LSGO action left no eligible variables")
            continue
        empty_scans = 0
        candidate = list(vector)
        candidate[variable] = 1 - candidate[variable]
        candidate_objective = problem.evaluate(tuple(candidate))
        consumed_fes += 1
        if candidate_objective < current_objective:
            vector = candidate
            current_objective = candidate_objective
            if decision.action_name == "allow_beneficial_coordination" and variable in shared_variables:
                coordinated_queue.extend(
                    other for other in shared_variables[variable] if other != group_index
                )

    ledger = SameBudgetLedger(
        phase_i_fe=phase_one_target,
        phase_ii_fe=phase_two_budget,
        budget_limit=request.total_fes,
        fresh_execution=True,
    )
    if ledger.total_fe != request.total_fes or ledger.violation:
        raise RuntimeError("binary LSGO FE accounting violated the configured budget")
    action_trace = BinaryLsgoActionTrace(
        action_name=decision.action_name,
        decision=decision.decision,
        trigger_reason=decision.trigger_reason,
        phase="phase_ii",
        affected_group_count=(
            len(_protected_schedule(stats))
            if decision.action_name == "protect_high_margin_group"
            else group_count
        ),
        affected_shared_variable_count=(
            len(shared_variables)
            if semantics.changed
            else 0
        ),
        allocated_fe=phase_two_budget,
        consumed_fe=phase_two_budget,
    )
    return BinaryLsgoExecutionResult(
        run_id=request.run_id,
        lane_id=request.lane_id,
        problem_id=problem.spec.problem_id,
        optimizer_seed=request.optimizer_seed,
        initial_vector_hash=initial_hash,
        phase_one_objective=phase_one_objective,
        final_objective=current_objective,
        final_vector=tuple(vector),
        evidence=evidence,
        decision=decision,
        semantics=semantics,
        ledger=ledger,
        action_trace=action_trace,
        optimizer_consumed=semantics.changed,
    )


__all__ = [
    "BinaryLsgoExecutionRequest",
    "BinaryLsgoGroupStats",
    "BinaryLsgoSnapshot",
    "BinaryBackendSemanticsDiff",
    "BinaryLsgoActionTrace",
    "BinaryLsgoExecutionResult",
    "SUPPORTED_ACTIONS",
    "build_binary_lsgo_evidence_profile",
    "run_binary_lsgo",
]
