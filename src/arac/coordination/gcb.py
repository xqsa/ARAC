"""Graph-conditioned scheduling for oracle overlap components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from arac.coordination.overlap import (
    ConflictLevel,
    CoordinationResult,
    LocalProposal,
    OverlapCoordinator,
    OverlapStructure,
    ProposalResidual,
    compute_proposal_residuals,
)


@dataclass(frozen=True)
class ComponentPriority:
    """Auditable GCB priority for one overlap-connected group component."""

    component: tuple[int, ...]
    shared_variables: tuple[int, ...]
    conflict_level: ConflictLevel
    max_conflict_score: float
    mean_conflict_score: float
    overlap_load: int
    topology_factor: float
    proposal_contribution: float
    conflict_streak: int
    priority_score: float


@dataclass(frozen=True)
class DispatchEvent:
    component: tuple[int, ...]
    priority_score: float
    requested_ctp_fes: int
    consumed_ctp_fes: int
    ledger_fes: int
    best_error_before: float
    best_error_after: float
    accepted_candidate: str | None


@dataclass(frozen=True)
class ComponentValueProbe:
    """Two counted objective probes used to estimate component repair value."""

    component: tuple[int, ...]
    conflict_level: ConflictLevel
    probe_errors: tuple[float, float]
    best_error_before: float
    estimated_gain: float
    consumed_fes: int


@dataclass(frozen=True)
class GcbDispatchResult:
    priorities: tuple[ComponentPriority, ...]
    priming_results: tuple[CoordinationResult, ...]
    events: tuple[DispatchEvent, ...]
    total_ctp_budget_fes: int
    consumed_ctp_fes: int
    unspent_ctp_fes: int
    ledger_fes: int
    value_probes: tuple[ComponentValueProbe, ...] = ()
    selection_mode: str = "structural"


class GraphCoordinationScheduler:
    """Rank persistent overlap conflicts and dispatch bounded CTP repairs."""

    def __init__(self, coordinator: OverlapCoordinator) -> None:
        if not isinstance(coordinator, OverlapCoordinator):
            raise TypeError("coordinator must be OverlapCoordinator")
        self.coordinator = coordinator
        self._priming_results: tuple[CoordinationResult, ...] = ()

    @property
    def overlap_components(self) -> tuple[tuple[int, ...], ...]:
        structure = self.coordinator.structure
        return tuple(
            component
            for component in structure.connected_components()
            if any(
                set(structure.owners(variable)).issubset(component)
                for variable in structure.shared_variables
            )
        )

    def _component_proposals(
        self,
        component: tuple[int, ...],
        proposals: tuple[LocalProposal, ...],
    ) -> tuple[LocalProposal, ...]:
        selected = tuple(proposal for proposal in proposals if proposal.group in component)
        if {proposal.group for proposal in selected} != set(component):
            raise ValueError("proposals must cover every group in each overlap component")
        return selected

    def _shared_variables(self, component: tuple[int, ...]) -> tuple[int, ...]:
        structure = self.coordinator.structure
        component_set = set(component)
        return tuple(
            variable
            for variable in structure.shared_variables
            if set(structure.owners(variable)).issubset(component_set)
            and component_set.intersection(structure.owners(variable))
        )

    def _level(self, residuals: Iterable[ProposalResidual]) -> ConflictLevel:
        maximum = max((item.conflict_score for item in residuals), default=0.0)
        if maximum >= self.coordinator.high_threshold:
            return ConflictLevel.HIGH
        if maximum >= self.coordinator.medium_threshold:
            return ConflictLevel.MEDIUM
        return ConflictLevel.LOW

    def prioritize(self, proposals: Iterable[LocalProposal]) -> tuple[ComponentPriority, ...]:
        proposal_list = tuple(proposals)
        priorities = []
        structure = self.coordinator.structure
        for component in self.overlap_components:
            variables = self._shared_variables(component)
            residuals = compute_proposal_residuals(
                structure,
                self._component_proposals(component, proposal_list),
                variables=variables,
                epsilon=self.coordinator.epsilon,
            )
            scores = tuple(item.conflict_score for item in residuals.values())
            component_proposals = self._component_proposals(component, proposal_list)
            proposal_contribution = sum(
                max(0.0, proposal.improvement) for proposal in component_proposals
            ) / len(component_proposals)
            overlap_load = sum(len(structure.owners(variable)) - 1 for variable in variables)
            topology_factor = 1.0 + overlap_load / len(component)
            streak = self.coordinator.conflict_streak(component)
            persistence_factor = 1.0 + 0.25 * max(0, streak - 1)
            maximum = max(scores, default=0.0)
            priorities.append(
                ComponentPriority(
                    component=component,
                    shared_variables=variables,
                    conflict_level=self._level(residuals.values()),
                    max_conflict_score=maximum,
                    mean_conflict_score=sum(scores) / len(scores),
                    overlap_load=overlap_load,
                    topology_factor=topology_factor,
                    proposal_contribution=proposal_contribution,
                    conflict_streak=streak,
                    priority_score=(
                        maximum
                        * topology_factor
                        * persistence_factor
                        * (1.0 + proposal_contribution)
                    ),
                )
            )
        return tuple(
            sorted(
                priorities,
                key=lambda item: (-item.priority_score, item.component),
            )
        )

    def prime(self, proposals: Iterable[LocalProposal]) -> tuple[CoordinationResult, ...]:
        """Record one residual observation per component without CTP repair."""

        if self._priming_results:
            raise RuntimeError("GCB scheduler has already been primed")
        proposal_list = tuple(proposals)
        self._priming_results = tuple(
            self.coordinator.coordinate(
                component,
                self._component_proposals(component, proposal_list),
            )
            for component in self.overlap_components
        )
        return self._priming_results

    def dispatch(
        self,
        proposals: Iterable[LocalProposal],
        *,
        total_ctp_budget_fes: int,
        max_components: int = 1,
        seed: int = 0,
    ) -> GcbDispatchResult:
        """Dispatch repair budget to the highest-priority persistent conflicts."""

        if not self._priming_results:
            raise RuntimeError("GCB scheduler must be primed before dispatch")
        if (
            isinstance(total_ctp_budget_fes, bool)
            or not isinstance(total_ctp_budget_fes, int)
            or total_ctp_budget_fes < 0
        ):
            raise ValueError("total_ctp_budget_fes must be a non-negative integer")
        if isinstance(max_components, bool) or not isinstance(max_components, int) or max_components <= 0:
            raise ValueError("max_components must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        proposal_list = tuple(proposals)
        priorities = self.prioritize(proposal_list)
        eligible = (
            tuple(
                priority
                for priority in priorities
                if priority.conflict_level is ConflictLevel.HIGH
                and priority.conflict_streak >= 1
            )[:max_components]
            if total_ctp_budget_fes > 0
            else ()
        )
        if eligible:
            arbitration_fes = 4 * len(eligible)
            if arbitration_fes + total_ctp_budget_fes > self.coordinator.ledger.remaining:
                raise ValueError("dispatch budget does not fit after candidate arbitration")
        remaining = total_ctp_budget_fes
        events = []
        for index, priority in enumerate(eligible):
            slots_left = len(eligible) - index
            requested = remaining // slots_left if slots_left else 0
            before_count = self.coordinator.ledger.count
            result = self.coordinator.coordinate(
                priority.component,
                self._component_proposals(priority.component, proposal_list),
                ctp_budget_fes=requested,
                ctp_seed=seed + index,
            )
            remaining -= result.ctp_consumed_fes
            events.append(
                DispatchEvent(
                    component=priority.component,
                    priority_score=priority.priority_score,
                    requested_ctp_fes=requested,
                    consumed_ctp_fes=result.ctp_consumed_fes,
                    ledger_fes=self.coordinator.ledger.count - before_count,
                    best_error_before=result.best_error_before,
                    best_error_after=result.best_error_after,
                    accepted_candidate=result.accepted_candidate,
                )
            )
        consumed = total_ctp_budget_fes - remaining
        return GcbDispatchResult(
            priorities=priorities,
            priming_results=self._priming_results,
            events=tuple(events),
            total_ctp_budget_fes=total_ctp_budget_fes,
            consumed_ctp_fes=consumed,
            unspent_ctp_fes=remaining,
            ledger_fes=self.coordinator.ledger.count,
        )

    def _probe_candidates(
        self,
        component: tuple[int, ...],
        proposals: tuple[LocalProposal, ...],
        base: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build symmetric complete candidates around the frozen archive."""

        variables = self._shared_variables(component)
        residuals = compute_proposal_residuals(
            self.coordinator.structure,
            self._component_proposals(component, proposals),
            variables=variables,
            epsilon=self.coordinator.epsilon,
        )
        by_group = {proposal.group: proposal for proposal in proposals}
        plus = np.asarray(base, dtype=float).copy()
        minus = np.asarray(base, dtype=float).copy()
        for variable in variables:
            residual = residuals[variable]
            spread = max(
                abs(by_group[group].value(variable) - residual.weighted_mean)
                for group in self.coordinator.structure.owners(variable)
            )
            step = max(self.coordinator.epsilon, spread / 4.0)
            plus[variable] = residual.weighted_mean + step
            minus[variable] = residual.weighted_mean - step
        lower = self.coordinator.ledger.problem.lower_array
        upper = self.coordinator.ledger.problem.upper_array
        return np.clip(plus, lower, upper), np.clip(minus, lower, upper)

    def value_probe(
        self,
        proposals: Iterable[LocalProposal],
    ) -> tuple[ComponentValueProbe, ...]:
        """Probe every persistent high-conflict component from one frozen archive."""

        if not self._priming_results:
            raise RuntimeError("GCB scheduler must be primed before value probes")
        proposal_list = tuple(proposals)
        priorities = self.prioritize(proposal_list)
        eligible = tuple(
            priority
            for priority in priorities
            if priority.conflict_level is ConflictLevel.HIGH and priority.conflict_streak >= 1
        )
        if not eligible:
            return ()
        base = self.coordinator.ledger.best_x
        base_error = float(self.coordinator.ledger.best_error)
        candidates = [
            candidate
            for priority in eligible
            for candidate in self._probe_candidates(
                priority.component,
                proposal_list,
                base,
            )
        ]
        if len(candidates) != 2 * len(eligible):
            raise RuntimeError("value probe construction did not produce two candidates per component")
        errors = np.asarray(self.coordinator.ledger.evaluate(np.asarray(candidates)), dtype=float)
        probes = []
        for index, priority in enumerate(eligible):
            pair = tuple(float(value) for value in errors[2 * index : 2 * index + 2])
            probes.append(
                ComponentValueProbe(
                    component=priority.component,
                    conflict_level=priority.conflict_level,
                    probe_errors=pair,
                    best_error_before=base_error,
                    estimated_gain=max(0.0, base_error - min(pair)),
                    consumed_fes=2,
                )
            )
        return tuple(probes)

    def dispatch_value_probe(
        self,
        proposals: Iterable[LocalProposal],
        *,
        total_ctp_budget_fes: int,
        forced_component: tuple[int, ...] | None = None,
        seed: int = 0,
    ) -> GcbDispatchResult:
        """Select a persistent conflict by counted probe gain, then run CTP."""

        if (
            isinstance(total_ctp_budget_fes, bool)
            or not isinstance(total_ctp_budget_fes, int)
            or total_ctp_budget_fes < 0
        ):
            raise ValueError("total_ctp_budget_fes must be a non-negative integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if forced_component is not None:
            forced_component = tuple(forced_component)
        proposal_list = tuple(proposals)
        search_base = self.coordinator.ledger.best_x
        probes = self.value_probe(proposal_list)
        priorities = self.prioritize(proposal_list)
        by_component = {priority.component: priority for priority in priorities}
        if forced_component is None:
            selected = max(
                probes,
                key=lambda item: (
                    item.estimated_gain,
                    by_component[item.component].priority_score,
                    tuple(-value for value in item.component),
                ),
            ).component if probes else None
            selection_mode = "value_probe"
        else:
            if forced_component not in by_component:
                raise ValueError("forced_component is not an overlap component")
            selected = forced_component
            selection_mode = "value_probe_control"
        if selected is None or total_ctp_budget_fes == 0:
            return GcbDispatchResult(
                priorities=priorities,
                priming_results=self._priming_results,
                events=(),
                total_ctp_budget_fes=total_ctp_budget_fes,
                consumed_ctp_fes=0,
                unspent_ctp_fes=total_ctp_budget_fes,
                ledger_fes=self.coordinator.ledger.count,
                value_probes=probes,
                selection_mode=selection_mode,
            )
        if not any(item.component == selected for item in probes):
            raise ValueError("selected component is not persistently high-conflict")
        if 4 + total_ctp_budget_fes > self.coordinator.ledger.remaining:
            raise ValueError("value-probe dispatch budget does not fit after probes")
        priority = by_component[selected]
        before_count = self.coordinator.ledger.count
        result = self.coordinator.coordinate(
            selected,
            self._component_proposals(selected, proposal_list),
            ctp_budget_fes=total_ctp_budget_fes,
            ctp_seed=seed,
            search_base=search_base,
        )
        event = DispatchEvent(
            component=selected,
            priority_score=priority.priority_score,
            requested_ctp_fes=total_ctp_budget_fes,
            consumed_ctp_fes=result.ctp_consumed_fes,
            ledger_fes=self.coordinator.ledger.count - before_count,
            best_error_before=result.best_error_before,
            best_error_after=result.best_error_after,
            accepted_candidate=result.accepted_candidate,
        )
        return GcbDispatchResult(
            priorities=priorities,
            priming_results=self._priming_results,
            events=(event,),
            total_ctp_budget_fes=total_ctp_budget_fes,
            consumed_ctp_fes=result.ctp_consumed_fes,
            unspent_ctp_fes=total_ctp_budget_fes - result.ctp_consumed_fes,
            ledger_fes=self.coordinator.ledger.count,
            value_probes=probes,
            selection_mode=selection_mode,
        )


DISPATCH_NEIGHBORHOOD = "neighborhood"
DISPATCH_COORDINATE_CTP = "coordinate_ctp"
DISPATCH_JOINT_CTP = "joint_ctp"
DISPATCH_JOINT_CMAES = "joint_cmaes_escalation"

DISPATCH_ACTIONS = (
    DISPATCH_NEIGHBORHOOD,
    DISPATCH_COORDINATE_CTP,
    DISPATCH_JOINT_CTP,
    DISPATCH_JOINT_CMAES,
)

DISPATCH_STRATEGIES = {
    DISPATCH_COORDINATE_CTP: "sequential_coordinate_patch",
    DISPATCH_JOINT_CTP: "sequential_joint_patch",
    DISPATCH_JOINT_CMAES: "joint_cmaes",
}


@dataclass(frozen=True)
class GcbDispatchConfig:
    """Fixed dispatch thresholds; only offline calibration may change them.

    v1 (Gate 37) used the absolute hub degree with threshold 3 and failed
    because Phase-I inferred structures are far denser than truth topologies.
    v2 (Gate 38) adds the relative hub degree (hub / (component groups - 1)),
    calibrated offline on fresh seeds: inferred star topologies saturate at
    1.0 while chain/random stay near 0.7.
    """

    persistent_streak: int = 2
    escalation_streak: int = 6
    hub_mode: str = "absolute"
    complex_hub_degree: int = 3
    complex_hub_ratio: float = 0.9
    stall_cap: int = 2
    cooldown_cycles: int = 1

    def __post_init__(self) -> None:
        for name in ("persistent_streak", "escalation_streak", "stall_cap", "cooldown_cycles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.escalation_streak < self.persistent_streak:
            raise ValueError("escalation_streak must not precede persistent_streak")
        if isinstance(self.complex_hub_degree, bool) or not isinstance(self.complex_hub_degree, int):
            raise ValueError("complex_hub_degree must be an integer")
        if self.complex_hub_degree < 2:
            raise ValueError("complex_hub_degree must be at least 2")
        if self.hub_mode not in ("absolute", "relative"):
            raise ValueError("hub_mode must be 'absolute' or 'relative'")
        if (
            isinstance(self.complex_hub_ratio, bool)
            or not isinstance(self.complex_hub_ratio, float)
            or not 0.0 < self.complex_hub_ratio <= 1.0
        ):
            raise ValueError("complex_hub_ratio must be a float in (0, 1]")


@dataclass(frozen=True)
class ComponentDispatchState:
    stall_count: int = 0
    cooldown_until_cycle: int = -1
    escalation_used: bool = False
    dispatch_enabled: bool = True


@dataclass(frozen=True)
class DispatchPlan:
    component: tuple[int, ...]
    action: str
    reserved_fes: int
    reason: str
    conflict_streak: int
    hub_degree: int
    relative_hub: float
    stall_count: int
    cooldown_until_cycle: int
    escalation_used: bool


@dataclass(frozen=True)
class DispatchReceipt:
    cycle_index: int
    component: tuple[int, ...]
    action: str
    reason: str
    conflict_streak: int
    hub_degree: int
    relative_hub: float
    reserved_fes: int
    consumed_fes: int
    gained: bool
    best_error_before: float
    best_error_after: float


class GcbDispatchPlanner:
    """Pre-registered streak-and-topology dispatch policy for the overlap loop.

    This is the minimal production GCB kernel: it owns the per-component
    coordination envelope, plans every operator reservation explicitly, and
    never lets an operator silently reduce another budget category. Component
    priority ordering, counted conflict probes and budget pulses remain future
    work under the ARAC-OC contract.
    """

    def __init__(
        self,
        structure: "OverlapStructure",
        components: Iterable[tuple[int, ...]],
        *,
        envelope_fes: int,
        config: GcbDispatchConfig | None = None,
    ) -> None:
        if (
            isinstance(envelope_fes, bool)
            or not isinstance(envelope_fes, int)
            or envelope_fes <= 0
        ):
            raise ValueError("envelope_fes must be a positive integer")
        if config is None:
            config = GcbDispatchConfig()
        if not isinstance(config, GcbDispatchConfig):
            raise TypeError("config must be GcbDispatchConfig")
        self.structure = structure
        self.envelope_fes = int(envelope_fes)
        self.config = config
        self._states: dict[tuple[int, ...], ComponentDispatchState] = {}
        self._hub_degrees: dict[tuple[int, ...], int] = {}
        self._relative_hubs: dict[tuple[int, ...], float] = {}
        for component in components:
            component = tuple(component)
            if component in self._states:
                raise ValueError("dispatch components must be unique")
            self._states[component] = ComponentDispatchState()
            self._hub_degrees[component] = self._compute_hub_degree(component)
            self._relative_hubs[component] = self._hub_degrees[component] / max(
                1, len(component) - 1
            )

    def _compute_hub_degree(self, component: tuple[int, ...]) -> int:
        """Max number of distinct overlap partners of one group in a component."""

        component_set = set(component)
        partners: dict[int, set[int]] = {group: set() for group in component}
        for variable in self.structure.shared_variables:
            owners = self.structure.owners(variable)
            if not set(owners).issubset(component_set):
                continue
            for group in owners:
                partners[group].update(other for other in owners if other != group)
        return max((len(partner_set) for partner_set in partners.values()), default=0)

    def hub_degree(self, component: tuple[int, ...]) -> int:
        return self._hub_degrees[tuple(component)]

    def relative_hub(self, component: tuple[int, ...]) -> float:
        return self._relative_hubs[tuple(component)]

    def _is_complex_topology(self, component: tuple[int, ...]) -> bool:
        if self.config.hub_mode == "relative":
            return self._relative_hubs[component] >= self.config.complex_hub_ratio
        return self._hub_degrees[component] >= self.config.complex_hub_degree

    def plan(
        self,
        component: tuple[int, ...],
        *,
        cycle_index: int,
        conflict_streak: int,
    ) -> DispatchPlan:
        if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0:
            raise ValueError("cycle_index must be a non-negative integer")
        if isinstance(conflict_streak, bool) or not isinstance(conflict_streak, int) or conflict_streak < 0:
            raise ValueError("conflict_streak must be a non-negative integer")
        component = tuple(component)
        state = self._states[component]
        if not state.dispatch_enabled:
            action, reason = DISPATCH_NEIGHBORHOOD, "stalled_out"
        elif cycle_index < state.cooldown_until_cycle:
            action, reason = DISPATCH_NEIGHBORHOOD, "cooldown"
        elif (
            conflict_streak >= self.config.escalation_streak
            and not state.escalation_used
        ):
            action, reason = DISPATCH_JOINT_CMAES, "persistent_escalation"
        elif conflict_streak >= self.config.persistent_streak:
            if self._is_complex_topology(component):
                action, reason = DISPATCH_JOINT_CTP, "complex_topology"
            else:
                action, reason = DISPATCH_COORDINATE_CTP, "pairwise_topology"
        else:
            action, reason = DISPATCH_NEIGHBORHOOD, "not_persistent"
        return DispatchPlan(
            component=component,
            action=action,
            reserved_fes=self.envelope_fes if action != DISPATCH_NEIGHBORHOOD else 0,
            reason=reason,
            conflict_streak=conflict_streak,
            hub_degree=self._hub_degrees[component],
            relative_hub=self._relative_hubs[component],
            stall_count=state.stall_count,
            cooldown_until_cycle=state.cooldown_until_cycle,
            escalation_used=state.escalation_used,
        )

    def record_outcome(
        self,
        component: tuple[int, ...],
        *,
        cycle_index: int,
        action: str,
        gained: bool,
    ) -> None:
        if action not in DISPATCH_STRATEGIES:
            raise ValueError(f"outcome requires a dispatch action, got: {action}")
        if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0:
            raise ValueError("cycle_index must be a non-negative integer")
        if not isinstance(gained, bool):
            raise TypeError("gained must be a boolean")
        component = tuple(component)
        state = self._states[component]
        updated = {
            "cooldown_until_cycle": cycle_index + 1 + self.config.cooldown_cycles,
            "escalation_used": state.escalation_used or action == DISPATCH_JOINT_CMAES,
            "stall_count": 0 if gained else state.stall_count + 1,
        }
        if not gained and updated["stall_count"] >= self.config.stall_cap:
            updated["dispatch_enabled"] = False
        self._states[component] = ComponentDispatchState(
            stall_count=updated["stall_count"],
            cooldown_until_cycle=updated["cooldown_until_cycle"],
            escalation_used=updated["escalation_used"],
            dispatch_enabled=updated.get("dispatch_enabled", state.dispatch_enabled),
        )


__all__ = [
    "DISPATCH_ACTIONS",
    "DISPATCH_COORDINATE_CTP",
    "DISPATCH_JOINT_CMAES",
    "DISPATCH_JOINT_CTP",
    "DISPATCH_NEIGHBORHOOD",
    "DISPATCH_STRATEGIES",
    "ComponentDispatchState",
    "ComponentPriority",
    "ComponentValueProbe",
    "DispatchEvent",
    "DispatchPlan",
    "DispatchReceipt",
    "GcbDispatchConfig",
    "GcbDispatchPlanner",
    "GcbDispatchResult",
    "GraphCoordinationScheduler",
]
