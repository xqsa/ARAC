"""HCC component-horizon primitives for ARAC-CAR-W.

The functions here know how to apply one frozen writeback plan and how to run
one complete component horizon.  Branch creation, paired ordering, gating, and
the global FE ledger remain owned by the policy-level CAR executor.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from arac.policy.action_trust_policy import robust_damped_writeback
from arac.policy.counterfactual_action_racing import (
    BranchState,
    DispatchEvidence,
    ProbeSeedDescriptor,
    fingerprint_branch_state,
)


@dataclass(frozen=True)
class CARRelationProposal:
    """Identity-free v31 proposal captured at a component barrier."""

    sweep_index: int
    group_left: int
    group_right: int
    shared_indices: tuple[int, ...]
    target_values: tuple[float, ...]
    action_name: str
    action_family: str
    overlap_strength: float
    feature_coverage: float
    writeback_norm: float

    def __post_init__(self) -> None:
        if int(self.sweep_index) < 0:
            raise ValueError("sweep_index must be non-negative")
        if int(self.group_left) < 0 or int(self.group_right) <= int(self.group_left):
            raise ValueError("proposal group indices must be ordered")
        if not self.shared_indices or len(self.shared_indices) != len(self.target_values):
            raise ValueError("proposal shared indices and targets are invalid")
        if len(set(self.shared_indices)) != len(self.shared_indices):
            raise ValueError("proposal shared indices must be unique")
        if not self.action_name or not self.action_family:
            raise ValueError("proposal action metadata is required")
        for value in (
            *self.target_values,
            self.overlap_strength,
            self.feature_coverage,
            self.writeback_norm,
        ):
            if not math.isfinite(float(value)):
                raise ValueError("proposal values must be finite")
        if not 0.0 <= float(self.feature_coverage) <= 1.0:
            raise ValueError("proposal feature coverage must be within [0, 1]")


@dataclass(frozen=True)
class CARPlanDecision:
    plan: "CARWritebackPlan | None"
    evidence: DispatchEvidence | None
    abstain_reason: str


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _graph_fingerprint(
    grouping_result: tuple[tuple[int, ...], ...],
    overlapping_elements: tuple[tuple[int, ...], ...],
) -> str:
    return _stable_hash(
        {
            "groups": [list(group) for group in grouping_result],
            "overlaps": [list(shared) for shared in overlapping_elements],
        }
    )


def _component_groups(
    group_count: int,
    overlapping_elements: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    parent = list(range(group_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge_index, shared in enumerate(overlapping_elements):
        if shared and edge_index + 1 < group_count:
            union(edge_index, edge_index + 1)
    components: dict[int, list[int]] = {}
    for index in range(group_count):
        components.setdefault(find(index), []).append(index)
    return tuple(tuple(values) for values in sorted(components.values()))


def freeze_component_writeback_plan(
    *,
    grouping_result: tuple[tuple[int, ...], ...],
    overlapping_elements: tuple[tuple[int, ...], ...],
    group_population_sizes: tuple[int, ...],
    proposal_sweeps: tuple[tuple[CARRelationProposal, ...], ...],
    lower: float,
    upper: float,
) -> CARPlanDecision:
    """Freeze one identity-free component plan after two complete sweeps."""

    groups = tuple(tuple(int(value) for value in group) for group in grouping_result)
    overlaps = tuple(tuple(sorted(int(value) for value in shared)) for shared in overlapping_elements)
    if len(proposal_sweeps) < 2:
        return CARPlanDecision(None, None, "insufficient_complete_evidence_sweeps")
    if len(groups) != len(group_population_sizes) or len(overlaps) != max(0, len(groups) - 1):
        raise ValueError("group and overlap metadata lengths are inconsistent")
    graph_fp = _graph_fingerprint(groups, overlaps)
    latest_sweeps = proposal_sweeps[-2:]
    proposals_by_sweep: list[dict[tuple[int, int, tuple[int, ...]], CARRelationProposal]] = []
    for expected_sweep, sweep in enumerate(latest_sweeps):
        mapping: dict[tuple[int, int, tuple[int, ...]], CARRelationProposal] = {}
        for proposal in sweep:
            if proposal.sweep_index != expected_sweep:
                raise ValueError("proposal sweep index does not match the barrier order")
            key = (
                int(proposal.group_left),
                int(proposal.group_right),
                tuple(proposal.shared_indices),
            )
            mapping[key] = proposal
        proposals_by_sweep.append(mapping)

    candidates: list[tuple[tuple[int, ...], CARWritebackPlan, DispatchEvidence]] = []
    rejection_reasons: list[str] = []
    for component in _component_groups(len(groups), overlaps):
        component_edges = [
            edge
            for edge in range(len(overlaps))
            if overlaps[edge] and edge in component and edge + 1 in component
        ]
        if not component_edges:
            continue
        edge_pairs: list[tuple[CARRelationProposal, CARRelationProposal]] = []
        for edge in component_edges:
            key = (edge, edge + 1, overlaps[edge])
            pair = tuple(mapping.get(key) for mapping in proposals_by_sweep)
            if any(item is None for item in pair):
                rejection_reasons.append("missing_component_evidence")
                break
            edge_pairs.append((pair[0], pair[1]))
        if len(edge_pairs) != len(component_edges):
            continue
        stable_pairs = [
            pair
            for pair in edge_pairs
            if pair[0].action_family == pair[1].action_family
            and pair[0].action_name == pair[1].action_name
        ]
        active_pairs = [
            pair for pair in stable_pairs if pair[1].action_family != "fallback"
        ]
        if not active_pairs:
            rejection_reasons.append("no_stable_non_fallback_action")
            continue
        families = {pair[1].action_family for pair in active_pairs}
        names = {pair[1].action_name for pair in active_pairs}
        if len(families) != 1 or len(names) != 1:
            rejection_reasons.append("mixed_non_fallback_action_family")
            continue
        if any(
            first.feature_coverage < 1.0 or second.feature_coverage < 1.0
            for first, second in active_pairs
        ):
            rejection_reasons.append("incomplete_candidate_evidence")
            continue
        latest = [pair[1] for pair in active_pairs]
        target_by_variable: dict[int, list[float]] = {}
        for item in latest:
            for variable, target in zip(item.shared_indices, item.target_values, strict=True):
                target_by_variable.setdefault(int(variable), []).append(float(target))
        shared_indices = tuple(sorted(target_by_variable))
        target_values = tuple(
            sum(target_by_variable[index]) / len(target_by_variable[index])
            for index in shared_indices
        )
        component_fp = _stable_hash(
            {
                "groups": list(component),
                "edges": [list(overlaps[edge]) for edge in component_edges],
            }
        )
        action_family = next(iter(families))
        action_name = next(iter(names))
        overlap_strength = sum(item.overlap_strength for item in latest) / len(latest)
        writeback_norm = math.sqrt(sum(item.writeback_norm**2 for item in latest))
        evidence = DispatchEvidence(
            graph_fingerprint=graph_fp,
            component_fingerprint=component_fp,
            candidate_action_name=action_name,
            candidate_action_family=action_family,
            overlap_strength=overlap_strength,
            shared_variable_count=len(shared_indices),
            evidence_sweep_count=2,
            evidence_coverage=min(item.feature_coverage for item in latest),
            writeback_norm=writeback_norm,
        )
        plan = CARWritebackPlan(
            graph_fingerprint=graph_fp,
            component_fingerprint=component_fp,
            action_name=action_name,
            action_family=action_family,
            group_indices=component,
            group_dims=tuple(groups[index] for index in component),
            group_population_sizes=tuple(group_population_sizes[index] for index in component),
            shared_indices=shared_indices,
            target_values=target_values,
            lower=float(lower),
            upper=float(upper),
            max_delta_norm=2.5 if action_family == "coordinate" else 0.5,
        )
        candidates.append((component, plan, evidence))

    if not candidates:
        reason = rejection_reasons[0] if rejection_reasons else "no_overlap_component_candidate"
        return CARPlanDecision(None, None, reason)
    _, plan, evidence = sorted(
        candidates,
        key=lambda item: (-item[2].shared_variable_count, item[1].component_fingerprint),
    )[0]
    return CARPlanDecision(plan, evidence, "")


@dataclass(frozen=True)
class CARWritebackPlan:
    graph_fingerprint: str
    component_fingerprint: str
    action_name: str
    action_family: str
    group_indices: tuple[int, ...]
    group_dims: tuple[tuple[int, ...], ...]
    group_population_sizes: tuple[int, ...]
    shared_indices: tuple[int, ...]
    target_values: tuple[float, ...]
    lower: float
    upper: float
    alpha: float = 0.20
    max_delta_norm: float = 0.5

    def __post_init__(self) -> None:
        if not self.graph_fingerprint or not self.component_fingerprint:
            raise ValueError("graph and component fingerprints are required")
        if not self.action_name or not self.action_family or self.action_family == "fallback":
            raise ValueError("a non-fallback candidate action is required")
        group_count = len(self.group_indices)
        if group_count == 0:
            raise ValueError("component must contain at least one group")
        if len(self.group_dims) != group_count or len(self.group_population_sizes) != group_count:
            raise ValueError("component group metadata lengths must match")
        if any(not dims for dims in self.group_dims):
            raise ValueError("component group dimensions must not be empty")
        if any(int(size) <= 0 for size in self.group_population_sizes):
            raise ValueError("group population sizes must be positive")
        if len(self.shared_indices) != len(self.target_values):
            raise ValueError("shared indices and target values must have equal lengths")
        if len(set(self.shared_indices)) != len(self.shared_indices):
            raise ValueError("shared indices must be unique")
        numeric = (
            *self.target_values,
            self.lower,
            self.upper,
            self.alpha,
            self.max_delta_norm,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("writeback plan values must be finite")
        if float(self.lower) >= float(self.upper):
            raise ValueError("writeback bounds are invalid")
        if float(self.alpha) != 0.20:
            raise ValueError("CAR-W candidate alpha is frozen at 0.20")
        if float(self.max_delta_norm) < 0.0:
            raise ValueError("max_delta_norm must be non-negative")


def shuffled_component_writeback_plan(plan: CARWritebackPlan) -> CARWritebackPlan:
    """Deterministically break variable-target pairing for a graph control."""

    count = len(plan.target_values)
    if count < 2:
        return plan
    digest = hashlib.sha256(
        f"{plan.graph_fingerprint}|{plan.component_fingerprint}|shuffled-graph".encode(
            "utf-8"
        )
    ).digest()
    offset = 1 + (int.from_bytes(digest[:8], "big") % (count - 1))
    targets = plan.target_values[offset:] + plan.target_values[:offset]
    return replace(plan, target_values=targets)


@dataclass(frozen=True)
class GroupOptimizationResult:
    best_x: np.ndarray
    best_y: float
    actual_fes: int


def allocate_component_horizon_budgets(
    *,
    max_arm_fes: int,
    population_sizes: tuple[int, ...],
) -> tuple[int, ...]:
    """Use the largest full-population component horizon within one arm cap.

    One FE is reserved for evaluating the branch start after candidate
    writeback. Every group then receives at least one complete population.
    """

    arm_cap = int(max_arm_fes)
    populations = tuple(int(size) for size in population_sizes)
    if arm_cap <= 0 or not populations or any(size <= 0 for size in populations):
        return ()
    minimum = 1 + sum(populations)
    if minimum > arm_cap:
        return ()
    budgets = list(populations)
    remaining = arm_cap - minimum
    while True:
        changed = False
        for index, population in enumerate(populations):
            if population <= remaining:
                budgets[index] += population
                remaining -= population
                changed = True
        if not changed:
            break
    return tuple(budgets)


def apply_candidate_writeback(
    incumbent: np.ndarray,
    plan: CARWritebackPlan,
) -> tuple[np.ndarray, float]:
    candidate = np.asarray(incumbent, dtype=float).reshape(-1).copy()
    indices = np.asarray(plan.shared_indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= candidate.size):
        raise ValueError("shared writeback index is outside the incumbent")
    current = candidate[indices]
    target = np.asarray(plan.target_values, dtype=float)
    adjusted = robust_damped_writeback(
        current_values=current,
        proposed_values=target,
        blend_strength=plan.alpha,
        max_delta_norm=plan.max_delta_norm,
    )
    adjusted = np.clip(adjusted, plan.lower, plan.upper)
    candidate[indices] = adjusted
    return candidate, float(np.linalg.norm(adjusted - current))


def _group_seed(descriptor: ProbeSeedDescriptor, group_position: int) -> int:
    payload = f"{descriptor.canonical_key}|group={int(group_position)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def run_component_horizon(
    *,
    checkpoint: BranchState,
    evaluator: object,
    seed_descriptor: ProbeSeedDescriptor,
    requested_fes: int,
    plan: CARWritebackPlan,
    apply_candidate: bool,
    optimize_group: Callable[..., GroupOptimizationResult],
) -> BranchState:
    budgets = allocate_component_horizon_budgets(
        max_arm_fes=int(requested_fes),
        population_sizes=plan.group_population_sizes,
    )
    if not budgets or 1 + sum(budgets) != int(requested_fes):
        raise ValueError("requested FE does not form a complete component horizon")
    record = getattr(evaluator, "fitness_record", None)
    if not isinstance(record, list) or record:
        raise ValueError("component horizon requires a fresh branch-local evaluator")

    incumbent = np.asarray(checkpoint.incumbent, dtype=float).reshape(-1).copy()
    writeback_norm = 0.0
    if apply_candidate:
        incumbent, writeback_norm = apply_candidate_writeback(incumbent, plan)
    initial_value = np.asarray(evaluator(incumbent), dtype=float).reshape(-1)
    if initial_value.size != 1 or not np.isfinite(initial_value[0]):
        raise ValueError("branch start evaluation must return one finite value")
    best_fitness = float(initial_value[0])

    for group_position, (dims, population_size, budget) in enumerate(
        zip(
            plan.group_dims,
            plan.group_population_sizes,
            budgets,
            strict=True,
        )
    ):
        result = optimize_group(
            evaluator=evaluator,
            background=incumbent.copy(),
            dims=dims,
            requested_fes=budget,
            population_size=population_size,
            seed=_group_seed(seed_descriptor, group_position),
        )
        best_x = np.asarray(result.best_x, dtype=float).reshape(-1)
        best_y = float(result.best_y)
        if int(result.actual_fes) != budget:
            raise ValueError("group optimizer did not consume its complete population budget")
        if best_x.size != len(dims) or not np.all(np.isfinite(best_x)):
            raise ValueError("group optimizer returned an invalid candidate")
        if not math.isfinite(best_y):
            raise ValueError("group optimizer returned a non-finite objective")
        if best_y < best_fitness:
            incumbent[np.asarray(dims, dtype=int)] = best_x
            best_fitness = best_y

    if len(record) != int(requested_fes):
        raise ValueError("component horizon evaluator FE does not match its reservation")
    payload = copy.deepcopy(checkpoint.state_payload)
    payload.update(
        {
            "car_component_fingerprint": plan.component_fingerprint,
            "car_graph_fingerprint": plan.graph_fingerprint,
            "car_action_family": plan.action_family,
            "car_action_applied": bool(apply_candidate),
            "car_writeback_norm": writeback_norm,
            "car_probe_seed": seed_descriptor.seed,
            "car_group_budgets": list(budgets),
        }
    )
    state = BranchState(
        incumbent=tuple(float(value) for value in incumbent),
        committed_fitness=best_fitness,
        evaluator_record=[float(value) for value in record],
        state_fingerprint="",
        state_payload=payload,
    )
    state.state_fingerprint = fingerprint_branch_state(state)
    return state
