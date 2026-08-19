"""Frozen ARAC-OC operator contract (design doc sections 2, 3, 6, 7, 8).

This module is the field-level freeze of the unified operator contract:
``GCB.make_plan`` emits an :class:`OperatorPlan`, one selected operator
executes it under a bounded reservation, and the outcome is always an
:class:`OperatorReceipt` with exact FE parity and a canonical hash.

Semantics frozen here (arac-oc-design.md):

- AOR is a pre-registered escalation action with the same contract as
  SMP/CTP; needing it is not a failure (section 2.1).
- Operator exceptions fail closed: evaluations already made stay in the
  ledger, the receipt records the action, consumed FE, remaining FE and
  the exception, and no retry or silent hand-off to another action may
  occur (section 2.2).
- Completing without strict-best gain is normal: status ``no_gain``,
  which feeds stall/cooldown state (section 2.3).
- An operator consumes exactly its reservation; no implicit encroachment
  on sense/probe/neighborhood budgets (section 7.1).
- Every receipt carries both topology signals (absolute and relative hub)
  and the coordinator state hash (section 3, Gate 38 section 5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from arac.runtime.contracts import canonical_sha256

OC_PLAN_SCHEMA = "arac-oc-operator-plan-v1"
OC_RECEIPT_SCHEMA = "arac-oc-operator-receipt-v1"
OC_CONFIG_SCHEMA = "arac-oc-coordinator-config-v3"

# Counted two-sided probe cost per scope variable; f(x0) reuses the ledger
# incumbent and is never re-billed (design section 5).
OC_PROBE_FES_PER_VARIABLE = 2

OC_ACTION_ARBITRATION = "arbitration_only"
OC_ACTION_SMP = "smp"
OC_ACTION_CTP_RESTRICTED = "ctp_restricted"
OC_ACTION_CTP_SHARED_CORE = "ctp_shared_core"
OC_ACTION_AOR = "aor"

OC_ACTIONS = (
    OC_ACTION_ARBITRATION,
    OC_ACTION_SMP,
    OC_ACTION_CTP_RESTRICTED,
    OC_ACTION_CTP_SHARED_CORE,
    OC_ACTION_AOR,
)

# Operator actions that must be dispatched through an operator execution
# (everything except the LOW level, whose gain comes solely from candidate
# arbitration -- design section 6).
OC_OPERATOR_ACTIONS = (
    OC_ACTION_SMP,
    OC_ACTION_CTP_RESTRICTED,
    OC_ACTION_CTP_SHARED_CORE,
    OC_ACTION_AOR,
)

OC_CONFLICT_LEVELS = ("low", "medium", "high", "complex")

# Fixed level -> action mapping (design section 6); runtime retuning by
# benchmark identity or final results is forbidden.  ``high`` admits the
# SMP alternative: when runtime trust in the scope owners' state memory
# decays below ``smp_trust_floor``, the plan dispatches a state-memory
# rebuild instead of a shared-core repair (design section 4).
OC_LEVEL_ACTION_MAP = {
    "low": OC_ACTION_ARBITRATION,
    "medium": OC_ACTION_CTP_RESTRICTED,
    "high": OC_ACTION_CTP_SHARED_CORE,
    "complex": OC_ACTION_AOR,
}

OC_LEVEL_ALLOWED_ACTIONS = {
    "low": (OC_ACTION_ARBITRATION,),
    "medium": (OC_ACTION_CTP_RESTRICTED,),
    "high": (OC_ACTION_CTP_SHARED_CORE, OC_ACTION_SMP),
    "complex": (OC_ACTION_AOR,),
}

OC_STATUS_COMPLETED = "completed"
OC_STATUS_NO_GAIN = "no_gain"
OC_STATUS_OPERATOR_FAILED = "operator_failed"

# Neutral placeholder constants marked uncalibrated: they only exist so the
# unified loop can run end-to-end; any comparative claim requires the
# pre-registered offline calibration gate first (design section 6).
UNCALIBRATED_FIELDS = frozenset(
    {
        "ema_alpha",
        "tau_enter",
        "tau_exit",
        "k_enter",
        "k_exit",
        "gamma_up",
        "gamma_down",
        "pulse_min_fes",
        "pulse_max_fes",
        "gain_floor",
        "k_window",
        "probe_budget_share",
        "smp_trust_floor",
        "arbitration_value_ratio",
        "operator_value_ratio",
        "operator_episode_min_fes",
    }
)


@dataclass(frozen=True)
class OcCoordinatorConfig:
    """Versioned coordinator constants.

    The streak/hub/cooldown block carries the Gate 38 v2 frozen values
    (calibrated offline on 36 fresh-seed structures).  Every field listed
    in :data:`UNCALIBRATED_FIELDS` holds a neutral default that has NOT
    passed the calibration gate; receipts must expose ``calibration_status``.
    """

    config_version: str = OC_CONFIG_SCHEMA
    # --- calibrated (Gate 38 v2 freeze) ---
    persistent_streak: int = 2
    escalation_streak: int = 6
    hub_mode: str = "relative"
    complex_hub_degree: int = 3
    complex_hub_ratio: float = 0.9
    stall_cap: int = 2
    cooldown_cycles: int = 1
    # --- uncalibrated placeholders ---
    ema_alpha: float = 0.3
    tau_enter: float = 0.5
    tau_exit: float = 0.2
    k_enter: int = 3
    k_exit: int = 3
    gamma_up: float = 1.5
    gamma_down: float = 0.5
    pulse_min_fes: int = 8
    pulse_max_fes: int = 64
    gain_floor: float = 1e-12
    k_window: int = 4
    probe_budget_share: float = 0.25
    smp_trust_floor: float = 0.5
    # Suppress an operator pulse when same-cycle arbitration already produced
    # a material relative improvement. This remains uncalibrated until the
    # fresh-seed value gate is confirmed.
    arbitration_value_ratio: float = 0.01
    # The same scale is applied after an operator window so a tiny strict-best
    # move cannot perturb the terminal-tail starting point.
    operator_value_ratio: float = 0.01
    # Minimum reservation for a real CTP episode.  The legacy default keeps
    # the frozen contract executable; production gates must register a larger
    # value before making comparative claims.
    operator_episode_min_fes: int = 8
    calibration_status: str = "partial-v3-uncalibrated"

    def __post_init__(self) -> None:
        for name in ("persistent_streak", "escalation_streak", "stall_cap", "cooldown_cycles",
                     "k_enter", "k_exit", "k_window", "pulse_min_fes", "pulse_max_fes",
                     "complex_hub_degree", "operator_episode_min_fes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.escalation_streak < self.persistent_streak:
            raise ValueError("escalation_streak must not precede persistent_streak")
        if self.hub_mode not in ("absolute", "relative"):
            raise ValueError("hub_mode must be 'absolute' or 'relative'")
        if not 0.0 < self.complex_hub_ratio <= 1.0:
            raise ValueError("complex_hub_ratio must be in (0, 1]")
        for name in ("ema_alpha",):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not self.tau_exit < self.tau_enter:
            raise ValueError("hysteresis requires tau_exit < tau_enter")
        for name in (
            "tau_enter",
            "tau_exit",
            "gain_floor",
            "probe_budget_share",
            "smp_trust_floor",
            "arbitration_value_ratio",
            "operator_value_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.probe_budget_share > 1.0:
            raise ValueError("probe_budget_share must not exceed 1")
        if self.smp_trust_floor > 1.0:
            raise ValueError("smp_trust_floor must not exceed 1")
        if self.gamma_up <= 1.0:
            raise ValueError("gamma_up must exceed 1")
        if not 0.0 < self.gamma_down < 1.0:
            raise ValueError("gamma_down must be in (0, 1)")
        if self.pulse_min_fes > self.pulse_max_fes:
            raise ValueError("pulse_min_fes must not exceed pulse_max_fes")
        if not self.calibration_status:
            raise ValueError("calibration_status must be a non-empty string")

    @property
    def config_hash(self) -> str:
        return canonical_sha256(asdict(self))


def _validate_variable_tuple(scope: tuple[int, ...], name: str) -> None:
    if not isinstance(scope, tuple):
        raise TypeError(f"{name} must be a tuple")
    seen: set[int] = set()
    for variable in scope:
        if isinstance(variable, bool) or not isinstance(variable, int) or variable < 0:
            raise ValueError(f"{name} must contain non-negative integer variable indices")
        if variable in seen:
            raise ValueError(f"{name} must not repeat a variable")
        seen.add(variable)
    if list(scope) != sorted(scope):
        raise ValueError(f"{name} must be sorted")


def _validate_group_tuple(component: tuple[int, ...]) -> None:
    if not isinstance(component, tuple) or not component:
        raise ValueError("component must be a non-empty tuple of group indices")
    if list(component) != sorted(component):
        raise ValueError("component must be sorted")
    if len(set(component)) != len(component):
        raise ValueError("component must not repeat a group")
    if any(isinstance(group, bool) or not isinstance(group, int) or group < 0 for group in component):
        raise ValueError("component must contain non-negative integer group indices")


@dataclass(frozen=True)
class OperatorPlan:
    """One GCB dispatch decision: where, what scope, how much, which action."""

    cycle_index: int
    component: tuple[int, ...]
    scope: tuple[int, ...]
    conflict_level: str
    action: str
    reserved_fes: int
    predicted_gain: float
    seed: int
    reason: str
    hub_degree: int
    relative_hub: float

    def __post_init__(self) -> None:
        if isinstance(self.cycle_index, bool) or not isinstance(self.cycle_index, int) or self.cycle_index < 0:
            raise ValueError("cycle_index must be a non-negative integer")
        _validate_group_tuple(self.component)
        _validate_variable_tuple(self.scope, "scope")
        if self.conflict_level not in OC_CONFLICT_LEVELS:
            raise ValueError(f"unknown conflict level: {self.conflict_level}")
        if self.action not in OC_ACTIONS:
            raise ValueError(f"unknown operator action: {self.action}")
        if self.action not in OC_LEVEL_ALLOWED_ACTIONS[self.conflict_level]:
            raise ValueError(
                f"action {self.action!r} is not admissible at conflict level "
                f"{self.conflict_level!r}"
            )
        if isinstance(self.reserved_fes, bool) or not isinstance(self.reserved_fes, int):
            raise ValueError("reserved_fes must be an integer")
        if self.action == OC_ACTION_ARBITRATION:
            if self.reserved_fes != 0:
                raise ValueError("arbitration-only plans reserve no operator FE")
        elif self.reserved_fes <= 0:
            raise ValueError("operator plans must reserve a positive FE budget")
        for name in ("predicted_gain", "relative_hub"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.relative_hub > 1.0:
            raise ValueError("relative_hub must not exceed 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.hub_degree, bool) or not isinstance(self.hub_degree, int) or self.hub_degree < 0:
            raise ValueError("hub_degree must be a non-negative integer")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")

    def payload(self) -> dict[str, object]:
        body = asdict(self)
        body["schema_version"] = OC_PLAN_SCHEMA
        return body

    @property
    def plan_hash(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True)
class OperatorReceipt:
    """Auditable outcome of one operator (or arbitration-only) execution."""

    plan_hash: str
    cycle_index: int
    component: tuple[int, ...]
    action: str
    conflict_level: str
    reason: str
    hub_degree: int
    relative_hub: float
    reserved_fes: int
    actual_fes: int
    status: str
    realized_gain: float
    best_error_before: float
    best_error_after: float
    candidates: tuple[tuple[float, ...], ...] = ()
    state_hash: str = ""
    remaining_fes: int = 0
    exception_name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.plan_hash, str) or len(self.plan_hash) != 64:
            raise ValueError("plan_hash must be a 64-character hash")
        if any(character not in "0123456789abcdef" for character in self.plan_hash):
            raise ValueError("plan_hash must be lowercase hexadecimal")
        _validate_group_tuple(self.component)
        if self.action not in OC_ACTIONS:
            raise ValueError(f"unknown operator action: {self.action}")
        if self.status not in (OC_STATUS_COMPLETED, OC_STATUS_NO_GAIN, OC_STATUS_OPERATOR_FAILED):
            raise ValueError(f"unknown operator status: {self.status}")
        for name in ("actual_fes", "reserved_fes", "cycle_index", "remaining_fes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("realized_gain", "best_error_before", "best_error_after", "relative_hub"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.realized_gain < 0.0:
            raise ValueError("realized_gain must be non-negative under strict-best")
        if self.best_error_after > self.best_error_before:
            raise ValueError("receipt best_error_after must not exceed best_error_before")
        if not self.state_hash:
            raise ValueError("receipts must carry a coordinator state hash")
        if len(self.state_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.state_hash
        ):
            raise ValueError("state_hash must be lowercase hexadecimal")
        if self.status == OC_STATUS_OPERATOR_FAILED:
            if not self.exception_name:
                raise ValueError("operator_failed receipts must name the exception")
            if self.actual_fes > self.reserved_fes:
                raise ValueError("a failed operator cannot exceed its reservation")
        else:
            if self.exception_name:
                raise ValueError("exception_name is reserved for operator_failed")
            if self.actual_fes != self.reserved_fes:
                raise ValueError("normal completion requires exact FE parity")
            if (self.realized_gain > 0.0) != (self.status == OC_STATUS_COMPLETED):
                raise ValueError("status and realized_gain disagree")
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple")

    def payload(self) -> dict[str, object]:
        body = asdict(self)
        body["schema_version"] = OC_RECEIPT_SCHEMA
        return body

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256(self.payload())


def receipt_from_plan(
    plan: OperatorPlan,
    *,
    actual_fes: int,
    best_error_before: float,
    best_error_after: float,
    candidates: tuple[tuple[float, ...], ...] = (),
    state_hash: str = "",
    remaining_fes: int = 0,
    exception_name: str = "",
) -> OperatorReceipt:
    """Build the receipt for one executed plan, enforcing status semantics."""

    if exception_name:
        status = OC_STATUS_OPERATOR_FAILED
    else:
        status = (
            OC_STATUS_COMPLETED
            if best_error_before - best_error_after > 0.0
            else OC_STATUS_NO_GAIN
        )
    return OperatorReceipt(
        plan_hash=plan.plan_hash,
        cycle_index=plan.cycle_index,
        component=plan.component,
        action=plan.action,
        conflict_level=plan.conflict_level,
        reason=plan.reason,
        hub_degree=plan.hub_degree,
        relative_hub=plan.relative_hub,
        reserved_fes=plan.reserved_fes,
        actual_fes=actual_fes,
        status=status,
        realized_gain=max(0.0, best_error_before - best_error_after),
        best_error_before=best_error_before,
        best_error_after=best_error_after,
        candidates=candidates,
        state_hash=state_hash,
        remaining_fes=remaining_fes,
        exception_name=exception_name,
    )


__all__ = [
    "OC_ACTIONS",
    "OC_ACTION_AOR",
    "OC_ACTION_ARBITRATION",
    "OC_ACTION_CTP_RESTRICTED",
    "OC_ACTION_CTP_SHARED_CORE",
    "OC_ACTION_SMP",
    "OC_CONFLICT_LEVELS",
    "OC_LEVEL_ACTION_MAP",
    "OC_LEVEL_ALLOWED_ACTIONS",
    "OC_OPERATOR_ACTIONS",
    "OC_PROBE_FES_PER_VARIABLE",
    "OC_STATUS_COMPLETED",
    "OC_STATUS_NO_GAIN",
    "OC_STATUS_OPERATOR_FAILED",
    "OcCoordinatorConfig",
    "OperatorPlan",
    "OperatorReceipt",
    "UNCALIBRATED_FIELDS",
    "receipt_from_plan",
]
