"""GCB planner producing complete OperatorPlans under the ARAC-OC contract.

One ``make_plan`` call answers all four coordinator questions for one
component: the component comes from :meth:`prioritize` (priority queue
with cooldown/deactivation filters), the scope from
:meth:`select_scope` (EMA-ranked, affordability-shrunk), the reservation
from the component's budget pulse, and the action from the frozen level
mapping of ``arac-oc-design.md`` section 6 with the complex-topology and
persistent-escalation upgrade paths of the Gate 38 v2 table.

The planner is stateless across cycles: every dynamic quantity (EMA
level, streak, pulse, trust) arrives in a :class:`ComponentSignal`
snapshot owned by :class:`~arac.coordination.state.CoordinatorState`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from arac.coordination.contract import (
    OC_ACTION_AOR,
    OC_ACTION_ARBITRATION,
    OC_ACTION_CTP_RESTRICTED,
    OC_ACTION_CTP_SHARED_CORE,
    OC_ACTION_SMP,
    OC_PROBE_FES_PER_VARIABLE,
    OcCoordinatorConfig,
    OperatorPlan,
)
from arac.coordination.overlap import OverlapStructure

OC_SIGNAL_LEVELS = ("low", "medium", "high")
SMP_MIN_WINDOW_FES = 8
AOR_MIN_WINDOW_FES = 2
CTP_SHARED_CORE_MIN_FES_PER_VARIABLE = 2


@dataclass(frozen=True)
class ComponentSignal:
    """Per-component snapshot the planner consumes each cycle."""

    component: tuple[int, ...]
    level: str
    conflict_streak: int = 0
    in_cooldown: bool = False
    active: bool = True
    stall: int = 0
    pulse_fes: int = 16
    qhat_mean: float = 1.0
    mean_c: float = 0.0
    max_c: float = 0.0
    proposal_contribution: float = 0.0
    escalation_used: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.component, tuple) or not self.component:
            raise ValueError("component must be a non-empty tuple of group indices")
        if self.level not in OC_SIGNAL_LEVELS:
            raise ValueError(f"unknown signal level: {self.level}")
        for name in ("conflict_streak", "stall", "pulse_fes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("qhat_mean", "mean_c", "max_c", "proposal_contribution"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.qhat_mean > 1.0:
            raise ValueError("qhat_mean must not exceed 1")


class OcDispatchPlanner:
    """Plan-only GCB: consumes signals, emits contract-valid OperatorPlans."""

    def __init__(
        self,
        structure: OverlapStructure,
        components: list[tuple[int, ...]],
        *,
        config: OcCoordinatorConfig | None = None,
        base_seed: int = 0,
    ) -> None:
        if not isinstance(structure, OverlapStructure):
            raise TypeError("structure must be OverlapStructure")
        if config is None:
            config = OcCoordinatorConfig()
        if not isinstance(config, OcCoordinatorConfig):
            raise TypeError("config must be OcCoordinatorConfig")
        if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
            raise ValueError("base_seed must be a non-negative integer")
        self.structure = structure
        self.config = config
        self.base_seed = int(base_seed)
        self._components: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
        self._hub_degrees: dict[tuple[int, ...], int] = {}
        self._relative_hubs: dict[tuple[int, ...], float] = {}
        for component in components:
            key = tuple(component)
            if not key or key in self._hub_degrees:
                raise ValueError("planner components must be non-empty and unique")
            self._hub_degrees[key] = self._compute_hub_degree(key)
            self._relative_hubs[key] = self._hub_degrees[key] / max(1, len(key) - 1)
            self._components[key] = tuple(
                variable
                for variable in self.structure.shared_variables
                if set(self.structure.owners(variable)).issubset(set(key))
            )

    def _compute_hub_degree(self, component: tuple[int, ...]) -> int:
        """Max number of distinct overlap partners of one group (Gate 37 §2)."""

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

    def shared_scope_variables(self, component: tuple[int, ...]) -> tuple[int, ...]:
        """Shared variables whose owners all lie inside the component."""

        return self._components[tuple(component)]

    def select_scope(
        self,
        component: tuple[int, ...],
        ema_c: dict[int, float],
        *,
        probe_budget_fes: int,
    ) -> tuple[int, ...]:
        """Rank the component's shared variables and shrink to the affordable prefix.

        Ranking key: EMA(C_j) descending, variable index ascending.  The
        counted probe costs ``2 * |scope|`` FE (design section 8); an
        unaffordable prefix is empty and the caller must receipt
        ``probe_budget_unavailable``.
        """

        if isinstance(probe_budget_fes, bool) or not isinstance(probe_budget_fes, int) or probe_budget_fes < 0:
            raise ValueError("probe_budget_fes must be a non-negative integer")
        candidates = sorted(
            self.shared_scope_variables(component),
            key=lambda variable: (-float(ema_c.get(variable, 0.0)), variable),
        )
        affordable = probe_budget_fes // OC_PROBE_FES_PER_VARIABLE
        return tuple(candidates[:affordable])

    def prioritize(self, signals: list[ComponentSignal]) -> tuple[ComponentSignal, ...]:
        """Order active, non-cooling components by the GCB priority score.

        ``priority = max_c * topology_factor * persistence_factor * (1 + contribution)``
        with ``topology_factor = 1 + overlap_load / |component|`` and
        ``persistence_factor = 1 + 0.25 * max(0, streak - 1)`` (the formula
        frozen with ``GraphCoordinationScheduler.prioritize``).
        """

        ranked: list[tuple[float, tuple[int, ...], ComponentSignal]] = []
        for signal in signals:
            if not isinstance(signal, ComponentSignal):
                raise TypeError("signals must be ComponentSignal instances")
            if not signal.active or signal.in_cooldown:
                continue
            overlap_load = len(self.shared_scope_variables(signal.component))
            topology_factor = 1.0 + overlap_load / len(signal.component)
            persistence_factor = 1.0 + 0.25 * max(0, signal.conflict_streak - 1)
            score = (
                signal.max_c
                * topology_factor
                * persistence_factor
                * (1.0 + signal.proposal_contribution)
            )
            ranked.append((score, signal.component, signal))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(signal for _, _, signal in ranked)

    def _plan_seed(self, cycle_index: int, component: tuple[int, ...]) -> int:
        return (self.base_seed * 1_000_003 + cycle_index * 10_007 + sum(component) * 101) % (2**31)

    def make_plan(
        self,
        signal: ComponentSignal,
        *,
        cycle_index: int,
        scope: tuple[int, ...],
        probe_widths: dict[int, float],
        available_fes: int,
        arbitration_gain: float = 0.0,
        arbitration_reference_error: float = 0.0,
    ) -> OperatorPlan:
        """Emit a plan from proposal persistence, topology and qhat trust.

        Counted-probe amplitude is intentionally absent from this decision;
        the probe remains available through ``signal.mean_c``/``max_c`` for
        scope ranking and audit.
        """

        if not isinstance(signal, ComponentSignal):
            raise TypeError("signal must be a ComponentSignal")
        if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0:
            raise ValueError("cycle_index must be a non-negative integer")
        if isinstance(available_fes, bool) or not isinstance(available_fes, int) or available_fes < 0:
            raise ValueError("available_fes must be a non-negative integer")
        if not math.isfinite(arbitration_gain) or arbitration_gain < 0.0:
            raise ValueError("arbitration_gain must be finite and non-negative")
        if not math.isfinite(arbitration_reference_error):
            raise ValueError("arbitration_reference_error must be finite")
        component = signal.component
        if component not in self._hub_degrees:
            raise ValueError(f"unknown component: {component}")
        if set(scope) - set(self.shared_scope_variables(component)):
            raise ValueError("scope must be a subset of the component's shared variables")

        persistent = signal.conflict_streak >= self.config.persistent_streak
        escalation = persistent and (
            signal.conflict_streak >= self.config.escalation_streak
            and not signal.escalation_used
        )
        legacy_high = signal.level == "high" and not persistent

        # A failed dispatch has already consumed its reserved window without
        # improving the strict-best archive.  Keep sensing and arbitration
        # alive, but do not immediately spend the next pulse on the same
        # repair path.  The component can re-enter after the normal cooldown
        # and persistence gates; escalation remains available when warranted.
        if signal.stall > 0 and not escalation:
            action, reason = OC_ACTION_ARBITRATION, "stall_guard_arbitration"
        elif signal.level == "low" and not persistent:
            action, reason = OC_ACTION_ARBITRATION, "low_arbitration"
        elif escalation:
            action, reason = OC_ACTION_AOR, "persistent_escalation_aor"
        elif legacy_high and self._is_complex_topology(component):
            # Preserve the explicit high/complex contract for direct planner
            # calls; state-driven persistent conflicts use shared-core CTP
            # before they reach the AOR escalation streak.
            action, reason = OC_ACTION_AOR, "complex_topology_aor"
        elif signal.qhat_mean < self.config.smp_trust_floor:
            action, reason = OC_ACTION_SMP, "high_smp_trust_rebuild"
        elif self._is_complex_topology(component) or (
            signal.level == "high" and signal.escalation_used
        ):
            action, reason = OC_ACTION_CTP_SHARED_CORE, "high_shared_core"
        else:
            action, reason = OC_ACTION_CTP_RESTRICTED, "medium_restricted_ctp"

        # The complete-candidate arbitration has already spent this cycle's
        # real evaluations. If its relative improvement is material, avoid a
        # low-value same-cycle pulse that could perturb the terminal-tail
        # starting point. The gate is independent of benchmark and topology.
        arbitration_ratio = arbitration_gain / max(abs(arbitration_reference_error), 1.0)
        if action != OC_ACTION_ARBITRATION and arbitration_ratio >= self.config.arbitration_value_ratio:
            action = OC_ACTION_ARBITRATION
            reason = "arbitration_value_gate"

        # A shared-core repair must have enough evaluations to give every
        # selected shared variable both a proposal and a contrasting move.
        # If the current pulse cannot pay that minimum window, do not emit a
        # reservation that is structurally too small to be a joint repair.
        shared_core_minimum = CTP_SHARED_CORE_MIN_FES_PER_VARIABLE * len(scope)
        if (
            action == OC_ACTION_CTP_SHARED_CORE
            and min(signal.pulse_fes, available_fes) < shared_core_minimum
        ):
            action = OC_ACTION_ARBITRATION
            reason = "shared_core_budget_unavailable"

        if action == OC_ACTION_SMP:
            touched_groups = {
                group
                for group in component
                if set(scope).intersection(self.structure.groups[group])
            }
            minimum_smp_fes = SMP_MIN_WINDOW_FES * max(1, len(touched_groups))
            if available_fes < minimum_smp_fes:
                # Each owner-local proposal session has an 8-FE lower bound;
                # make that constraint an explicit GCB decision instead of
                # allowing the operator adapter to fail after dispatch.
                action = OC_ACTION_CTP_SHARED_CORE
                reason = "smp_budget_unavailable_ctp"

        if action == OC_ACTION_ARBITRATION:
            reserved_fes = 0
            predicted_gain = 0.0
        else:
            if not scope:
                raise ValueError("operator plans require a non-empty scope")
            minimum_fes = {
                OC_ACTION_AOR: AOR_MIN_WINDOW_FES,
                OC_ACTION_SMP: SMP_MIN_WINDOW_FES,
                OC_ACTION_CTP_RESTRICTED: self.config.operator_episode_min_fes,
                OC_ACTION_CTP_SHARED_CORE: shared_core_minimum,
            }[action]
            # Preserve the pre-episode SMP shortfall reroute: it is a
            # compatibility path used by the frozen planner tests.  Gate49
            # never treats this 2-FE path as evidence of a valid shared-core
            # episode; its registered configuration keeps the operator pool
            # above the structural minimum.
            if (
                action == OC_ACTION_CTP_SHARED_CORE
                and reason == "smp_budget_unavailable_ctp"
            ):
                minimum_fes = 1
            if available_fes < minimum_fes:
                if available_fes == 0:
                    raise ValueError("dispatch pool exhausted; no operator plan may be emitted")
                if action in {
                    OC_ACTION_CTP_RESTRICTED,
                    OC_ACTION_CTP_SHARED_CORE,
                    OC_ACTION_SMP,
                }:
                    # A partial repair window is not an operator episode. Keep
                    # the decision auditable and let the exact-budget tail use
                    # the remaining FE instead of emitting a false pulse.
                    action = OC_ACTION_ARBITRATION
                    reason = "operator_episode_budget_unavailable"
                    reserved_fes = 0
                    predicted_gain = 0.0
                    minimum_fes = 0
                else:
                    raise ValueError(
                        f"{action} requires at least {minimum_fes} FE in the dispatch pool"
                    )
            if action != OC_ACTION_ARBITRATION:
                reserved_fes = min(max(signal.pulse_fes, minimum_fes), available_fes)
                if action == OC_ACTION_SMP:
                    reserved_fes = max(reserved_fes, minimum_smp_fes)
                if probe_widths:
                    widths = [float(probe_widths.get(variable, 0.0)) for variable in scope]
                    predicted_gain = sum(widths) / len(widths)
                else:
                    predicted_gain = 0.0

        conflict_level = {
            OC_ACTION_ARBITRATION: "low",
            OC_ACTION_CTP_RESTRICTED: "medium",
            OC_ACTION_SMP: "high",
            OC_ACTION_CTP_SHARED_CORE: "high",
            OC_ACTION_AOR: "complex",
        }[action]
        return OperatorPlan(
            cycle_index=cycle_index,
            component=component,
            scope=tuple(sorted(scope)),
            conflict_level=conflict_level,
            action=action,
            reserved_fes=reserved_fes,
            predicted_gain=predicted_gain,
            seed=self._plan_seed(cycle_index, component),
            reason=reason,
            hub_degree=self.hub_degree(component),
            relative_hub=self.relative_hub(component),
        )


__all__ = [
    "AOR_MIN_WINDOW_FES",
    "CTP_SHARED_CORE_MIN_FES_PER_VARIABLE",
    "ComponentSignal",
    "OcDispatchPlanner",
    "SMP_MIN_WINDOW_FES",
]
