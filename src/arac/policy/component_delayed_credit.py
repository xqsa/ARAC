"""Trace-only component state for action-specific delayed credit audits.

The tracker observes runtime state and mutates trace dictionaries only. It has
no dispatch API and must not change candidates, optimizer state, RNG, or FE.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .action_trust_policy import normalized_objective_credit


COMPONENT_CREDIT_TRACE_FIELDS = (
    "component_id",
    "component_group_count",
    "component_shared_var_count",
    "component_action_id",
    "component_action_scope",
    "component_credit_status",
    "component_decision_fe",
    "component_remaining_budget_ratio",
    "component_resolution_fe",
    "component_resolution_delay_fe",
    "component_resolution_window",
    "component_pending_before",
    "component_lock_conflict",
    "component_proposal_disagreement",
    "component_local_gain",
    "component_gain",
    "component_neighbor_gain",
    "component_neighbor_spillover",
    "shared_var_overwrite_rate",
    "shared_var_survival_rate",
    "component_credit_reason",
)


@dataclass(frozen=True)
class OverlapComponent:
    component_id: str
    group_indices: tuple[int, ...]
    shared_variables: tuple[int, ...]


@dataclass(frozen=True)
class OverlapComponentTopology:
    components: tuple[OverlapComponent, ...]
    group_component_ids: tuple[str, ...]
    group_shared_variables: tuple[tuple[int, ...], ...]

    def for_group(self, group_index: int) -> OverlapComponent:
        index = int(group_index)
        if index < 0 or index >= len(self.group_component_ids):
            raise IndexError(f"group index out of range: {group_index}")
        component_id = self.group_component_ids[index]
        return next(
            component
            for component in self.components
            if component.component_id == component_id
        )

    def shared_for_group(self, group_index: int) -> tuple[int, ...]:
        index = int(group_index)
        if index < 0 or index >= len(self.group_shared_variables):
            raise IndexError(f"group index out of range: {group_index}")
        return self.group_shared_variables[index]


@dataclass
class _PendingComponentAction:
    action_id: str
    component: OverlapComponent
    group_index: int
    decision_fe: int
    pre_action_fitness: float
    post_action_fitness: float
    shared_indices: tuple[int, ...]
    pre_shared_values: np.ndarray
    post_shared_values: np.ndarray
    trace_row: dict[str, str]


def _component_id(
    group_indices: tuple[int, ...],
    groups: tuple[tuple[int, ...], ...],
) -> str:
    payload = "|".join(
        f"{index}:" + ",".join(str(value) for value in groups[index])
        for index in group_indices
    )
    return "component_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_overlap_components(
    grouping_result: list[list[int]] | tuple[tuple[int, ...], ...],
) -> OverlapComponentTopology:
    groups = tuple(
        tuple(sorted({int(value) for value in group}))
        for group in grouping_result
    )
    if not groups or any(not group for group in groups):
        raise ValueError("grouping result must contain non-empty groups")

    parent = list(range(len(groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    group_sets = [set(group) for group in groups]
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            if group_sets[left].intersection(group_sets[right]):
                union(left, right)

    members_by_root: dict[int, list[int]] = {}
    for index in range(len(groups)):
        members_by_root.setdefault(find(index), []).append(index)

    components: list[OverlapComponent] = []
    group_component_ids = [""] * len(groups)
    group_shared_variables: list[tuple[int, ...]] = [()] * len(groups)
    for members in sorted(members_by_root.values(), key=lambda values: values[0]):
        group_indices = tuple(members)
        variable_counts: dict[int, int] = {}
        for index in group_indices:
            for variable in groups[index]:
                variable_counts[variable] = variable_counts.get(variable, 0) + 1
        shared_variables = tuple(
            sorted(variable for variable, count in variable_counts.items() if count > 1)
        )
        component_id = _component_id(group_indices, groups)
        component = OverlapComponent(
            component_id=component_id,
            group_indices=group_indices,
            shared_variables=shared_variables,
        )
        components.append(component)
        for index in group_indices:
            group_component_ids[index] = component_id
            group_shared_variables[index] = tuple(
                sorted(group_sets[index].intersection(shared_variables))
            )

    return OverlapComponentTopology(
        components=tuple(components),
        group_component_ids=tuple(group_component_ids),
        group_shared_variables=tuple(group_shared_variables),
    )


def _format_float(value: float) -> str:
    return f"{float(value):.6e}"


class ComponentDelayedCreditTrace:
    """Observe component state without changing controller behavior."""

    def __init__(
        self,
        grouping_result: list[list[int]] | tuple[tuple[int, ...], ...],
        *,
        lower: float,
        upper: float,
    ) -> None:
        self.topology = build_overlap_components(grouping_result)
        self.lower = float(lower)
        self.upper = float(upper)
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("component bounds must be finite")
        if self.upper <= self.lower:
            raise ValueError("component upper bound must exceed lower bound")
        self._pending_by_group: dict[int, _PendingComponentAction] = {}
        self._sweep_disagreements: dict[tuple[int, str], list[float]] = {}
        self._latest_completed_disagreement: dict[str, float] = {}
        self._action_sequence = 0

    def _component_values(self, component: OverlapComponent) -> dict[str, str]:
        return {
            "component_id": component.component_id,
            "component_group_count": str(len(component.group_indices)),
            "component_shared_var_count": str(len(component.shared_variables)),
        }

    def _remaining_ratio(self, decision_fe: int, max_fes: int) -> float:
        if max_fes <= 0:
            raise ValueError("max_fes must be positive")
        return max(0.0, min(1.0, (int(max_fes) - int(decision_fe)) / int(max_fes)))

    def _proposal_disagreement(
        self,
        previous_values: np.ndarray,
        current_values: np.ndarray,
    ) -> float:
        previous = np.asarray(previous_values, dtype=float).reshape(-1)
        current = np.asarray(current_values, dtype=float).reshape(-1)
        if previous.shape != current.shape or previous.size == 0:
            raise ValueError("proposal vectors must be non-empty and shape-aligned")
        if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
            raise ValueError("proposal vectors must be finite")
        scale = (self.upper - self.lower) * math.sqrt(previous.size)
        return min(1.0, float(np.linalg.norm(previous - current)) / scale)

    def annotate_relation_observation(
        self,
        trace_row: dict[str, str],
        *,
        outer_iter: int,
        group_left: int,
        group_right: int,
        previous_values: np.ndarray,
        current_values: np.ndarray,
        decision_fe: int,
        max_fes: int,
    ) -> float:
        component = self.topology.for_group(group_right)
        if self.topology.for_group(group_left).component_id != component.component_id:
            raise ValueError("overlap relation crosses disconnected components")
        disagreement = self._proposal_disagreement(previous_values, current_values)
        self._sweep_disagreements.setdefault(
            (int(outer_iter), component.component_id), []
        ).append(disagreement)
        trace_row.update(
            {
                **self._component_values(component),
                "component_action_scope": "shared_relation_observation",
                "component_credit_status": "relation_observation",
                "component_decision_fe": str(int(decision_fe)),
                "component_remaining_budget_ratio": _format_float(
                    self._remaining_ratio(decision_fe, max_fes)
                ),
                "component_proposal_disagreement": _format_float(disagreement),
                "component_credit_reason": "paired_shared_value_proposals_observed",
            }
        )
        return disagreement

    def complete_sweep(self, *, outer_iter: int, optimized_group_count: int) -> None:
        if int(optimized_group_count) != len(self.topology.group_component_ids):
            return
        for component in self.topology.components:
            values = self._sweep_disagreements.pop(
                (int(outer_iter), component.component_id), []
            )
            if values:
                self._latest_completed_disagreement[component.component_id] = (
                    sum(values) / len(values)
                )

    def register_search_action(
        self,
        trace_row: dict[str, str],
        *,
        action_name: str,
        outer_iter: int,
        group_index: int,
        decision_fe: int,
        max_fes: int,
        pre_action_fitness: float,
        post_action_fitness: float,
        pre_action_candidate: np.ndarray,
        post_action_candidate: np.ndarray,
    ) -> str:
        action = str(action_name).strip()
        if not action:
            raise ValueError("action_name must be non-empty")
        group = int(group_index)
        if group in self._pending_by_group:
            raise RuntimeError("group already has an unresolved component action")
        component = self.topology.for_group(group)
        pending_before = sum(
            pending.component.component_id == component.component_id
            for pending in self._pending_by_group.values()
        )
        pre_candidate = np.asarray(pre_action_candidate, dtype=float).reshape(-1)
        post_candidate = np.asarray(post_action_candidate, dtype=float).reshape(-1)
        if pre_candidate.shape != post_candidate.shape:
            raise ValueError("action candidates must be shape-aligned")
        if not np.all(np.isfinite(pre_candidate)) or not np.all(np.isfinite(post_candidate)):
            raise ValueError("action candidates must be finite")
        shared_indices = self.topology.shared_for_group(group)
        self._action_sequence += 1
        action_id = (
            f"{action}:{component.component_id}:{int(outer_iter)}:{group}:"
            f"{self._action_sequence}"
        )
        pre_fitness = float(pre_action_fitness)
        post_fitness = float(post_action_fitness)
        local_gain = normalized_objective_credit(pre_fitness, post_fitness)
        proposal_disagreement = self._latest_completed_disagreement.get(
            component.component_id
        )
        trace_row.update(
            {
                **self._component_values(component),
                "component_action_id": action_id,
                "component_action_scope": "group_search_start_component_credit",
                "component_credit_status": "pending",
                "component_decision_fe": str(int(decision_fe)),
                "component_remaining_budget_ratio": _format_float(
                    self._remaining_ratio(decision_fe, max_fes)
                ),
                "component_resolution_window": "next_canonical_group_revisit",
                "component_pending_before": str(pending_before),
                "component_lock_conflict": str(int(pending_before > 0)),
                "component_proposal_disagreement": (
                    ""
                    if proposal_disagreement is None
                    else _format_float(proposal_disagreement)
                ),
                "component_local_gain": _format_float(local_gain),
                "component_credit_reason": "awaiting_next_canonical_group_revisit",
            }
        )
        self._pending_by_group[group] = _PendingComponentAction(
            action_id=action_id,
            component=component,
            group_index=group,
            decision_fe=int(decision_fe),
            pre_action_fitness=pre_fitness,
            post_action_fitness=post_fitness,
            shared_indices=shared_indices,
            pre_shared_values=pre_candidate[list(shared_indices)].copy(),
            post_shared_values=post_candidate[list(shared_indices)].copy(),
            trace_row=trace_row,
        )
        return action_id

    def resolve_group_revisit(
        self,
        *,
        group_index: int,
        resolution_fe: int,
        current_fitness: float,
        current_candidate: np.ndarray,
    ) -> int:
        pending = self._pending_by_group.pop(int(group_index), None)
        if pending is None:
            return 0
        if int(resolution_fe) < pending.decision_fe:
            self._pending_by_group[int(group_index)] = pending
            raise ValueError("resolution_fe must not precede decision_fe")
        candidate = np.asarray(current_candidate, dtype=float).reshape(-1)
        if not np.all(np.isfinite(candidate)):
            raise ValueError("resolution candidate must be finite")
        component_gain = normalized_objective_credit(
            pending.pre_action_fitness,
            float(current_fitness),
        )
        neighbor_gain = normalized_objective_credit(
            pending.post_action_fitness,
            float(current_fitness),
        )
        overwrite_rate = ""
        survival_rate = ""
        reason = "resolved_next_canonical_group_revisit"
        if pending.shared_indices:
            action_displacement = np.abs(
                pending.post_shared_values - pending.pre_shared_values
            )
            changed_by_action = action_displacement > (self.upper - self.lower) * 1e-12
            if np.any(changed_by_action):
                current_shared = candidate[list(pending.shared_indices)]
                overwritten = np.abs(current_shared - pending.post_shared_values) > (
                    (self.upper - self.lower) * 1e-12
                )
                rate = float(np.mean(overwritten[changed_by_action]))
                overwrite_rate = _format_float(rate)
                survival_rate = _format_float(1.0 - rate)
            else:
                reason = "resolved_no_shared_variable_displacement"
        else:
            reason = "resolved_no_shared_variables"
        pending.trace_row.update(
            {
                "component_credit_status": "resolved",
                "component_resolution_fe": str(int(resolution_fe)),
                "component_resolution_delay_fe": str(
                    int(resolution_fe) - pending.decision_fe
                ),
                "component_gain": _format_float(component_gain),
                "component_neighbor_gain": _format_float(neighbor_gain),
                "component_neighbor_spillover": _format_float(
                    max(0.0, -neighbor_gain)
                ),
                "shared_var_overwrite_rate": overwrite_rate,
                "shared_var_survival_rate": survival_rate,
                "component_credit_reason": reason,
            }
        )
        return 1

    def finalize_unresolved(self, *, resolution_fe: int) -> int:
        pending_actions = list(self._pending_by_group.values())
        self._pending_by_group.clear()
        for pending in pending_actions:
            pending.trace_row.update(
                {
                    "component_credit_status": "unresolved_run_end",
                    "component_resolution_fe": str(int(resolution_fe)),
                    "component_resolution_delay_fe": str(
                        int(resolution_fe) - pending.decision_fe
                    ),
                    "component_credit_reason": (
                        "budget_ended_before_next_group_revisit"
                    ),
                }
            )
        return len(pending_actions)
