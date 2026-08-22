"""Runtime CoordinatorState: proposal persistence, probe telemetry, trust and pulses.

Implements design section 3 (checkpoint/feedback state) and section 7
(anti-oscillation) as a resumable object: every update lands in the
snapshot payload, and deterministic replay is guaranteed by the tuple
``(checkpoint_hash, state_hash, seed, config_hash)``.

v1 declared separation (Gate 47-R):

- Counted-probe ``C_j`` is telemetry for scope ranking and probe diagnostics.
  It does not decide whether an operator is dispatched; Gate 47 found no
  topology-separating threshold gap in its amplitude.
- Proposal residual persistence is the dispatch gate.  A component becomes
  dispatchable after ``persistent_streak`` consecutive high residual reports.
- qhat credit for a multi-variable scope is spread uniformly over the
  ``(variable, group)`` membership pairs of the scope (per-variable
  attribution is deferred to a v2 ablation).
- The initial budget pulse is ``pulse_min_fes`` (conservative start,
  grows by ``gamma_up`` on gain).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from arac.coordination.contract import OcCoordinatorConfig
from arac.coordination.contract import OC_ACTION_ARBITRATION
from arac.coordination.overlap import OverlapStructure
from arac.runtime.contracts import canonical_sha256
from arac.coordination.planner import ComponentSignal

OC_STATE_SCHEMA = "arac-oc-coordinator-state-v2"
OC_STATE_SCHEMA_V1 = "arac-oc-coordinator-state-v1"


@dataclass(frozen=True)
class OcStateSnapshot:
    """Canonical, hash-chained snapshot of the coordinator state."""

    payload: bytes
    state_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("payload must be non-empty bytes")
        if self.state_hash != hashlib.sha256(self.payload).hexdigest():
            raise ValueError("state_hash must be the sha256 of the payload")


class CoordinatorState:
    """Mutable per-run feedback state; never touches the Phase-I checkpoint."""

    def __init__(
        self,
        structure: OverlapStructure,
        components: list[tuple[int, ...]],
        *,
        config: OcCoordinatorConfig | None = None,
        checkpoint_hash: str = "",
    ) -> None:
        if not isinstance(structure, OverlapStructure):
            raise TypeError("structure must be OverlapStructure")
        if config is None:
            config = OcCoordinatorConfig()
        if not isinstance(config, OcCoordinatorConfig):
            raise TypeError("config must be OcCoordinatorConfig")
        self.structure = structure
        self.config = config
        if checkpoint_hash and (
            len(checkpoint_hash) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint_hash)
        ):
            raise ValueError("checkpoint_hash must be lowercase hexadecimal")
        self.checkpoint_hash = checkpoint_hash
        self.shared_patch: dict[str, object] = {}
        self.structure_hash = canonical_sha256(
            {
                "dimension": structure.dimension,
                "groups": structure.groups,
                "member_confidences": structure.member_confidences,
            }
        )
        self.qhat: dict[tuple[int, int], float] = {
            (int(variable), int(group)): float(confidence)
            for variable, group, confidence in structure.member_confidences
        }
        self.ema_c: dict[int, float] = {}
        self.level: dict[tuple[int, ...], str] = {}
        self.enter_count: dict[tuple[int, ...], int] = {}
        self.exit_count: dict[tuple[int, ...], int] = {}
        self.conflict_streak: dict[tuple[int, ...], int] = {}
        self.cooldown_until: dict[tuple[int, ...], int] = {}
        self.stall: dict[tuple[int, ...], int] = {}
        self.escalation_used: dict[tuple[int, ...], bool] = {}
        self.pulse_fes: dict[tuple[int, ...], int] = {}
        for component in components:
            key = tuple(component)
            if not key or key in self.level:
                raise ValueError("state components must be non-empty and unique")
            self.level[key] = "low"
            self.enter_count[key] = 0
            self.exit_count[key] = 0
            self.conflict_streak[key] = 0
            self.cooldown_until[key] = -1
            self.stall[key] = 0
            self.escalation_used[key] = False
            self.pulse_fes[key] = config.pulse_min_fes

    # ------------------------------------------------------------------
    # sensing side: diagnostic EMA plus proposal-residual dispatch gate
    # ------------------------------------------------------------------
    def observe_probes(
        self,
        component: tuple[int, ...],
        scope: tuple[int, ...],
        conflicts: dict[int, float],
    ) -> None:
        """Fold one counted-probe sweep into diagnostic EMA(C_j) only."""

        component = tuple(component)
        alpha = self.config.ema_alpha
        for variable in scope:
            raw = float(conflicts[variable])
            if not math.isfinite(raw) or raw < 0.0:
                raise ValueError("conflict scores must be finite and non-negative")
            previous = self.ema_c.get(variable)
            self.ema_c[variable] = raw if previous is None else (1.0 - alpha) * previous + alpha * raw
        # Deliberately no level transition here.  C_j remains available to
        # GCB for scope ranking and audit, but cannot open an operator window.

    def observe_proposal_conflict(
        self,
        component: tuple[int, ...],
        *,
        high_conflict: bool,
    ) -> None:
        """Update the dispatch persistence signal from proposal residuals."""

        component = tuple(component)
        if not isinstance(high_conflict, bool):
            raise TypeError("high_conflict must be a bool")
        if high_conflict:
            self.conflict_streak[component] += 1
        else:
            self.conflict_streak[component] = 0
        self.level[component] = (
            "medium"
            if self.conflict_streak[component] >= self.config.persistent_streak
            else "low"
        )

    def signal(
        self,
        component: tuple[int, ...],
        *,
        cycle_index: int,
        proposal_contribution: float = 0.0,
    ) -> ComponentSignal:
        component = tuple(component)
        scope_variables = [
            variable
            for variable in self.structure.shared_variables
            if set(self.structure.owners(variable)).issubset(set(component))
        ]
        values = [self.ema_c.get(variable, 0.0) for variable in scope_variables] or [0.0]
        pairs = [
            (variable, group)
            for variable in scope_variables
            for group in self.structure.owners(variable)
        ]
        trust = [self.qhat.get(pair, 1.0) for pair in pairs] or [1.0]
        return ComponentSignal(
            component=component,
            level=self.level[component],
            conflict_streak=self.conflict_streak[component],
            in_cooldown=cycle_index < self.cooldown_until[component],
            active=self.stall[component] < self.config.stall_cap,
            stall=self.stall[component],
            pulse_fes=self.pulse_fes[component],
            qhat_mean=sum(trust) / len(trust),
            mean_c=sum(values) / len(values),
            max_c=max(values),
            proposal_contribution=proposal_contribution,
            escalation_used=self.escalation_used[component],
        )

    # ------------------------------------------------------------------
    # feedback side: credit, pulse, cooldown, stall, deactivation
    # ------------------------------------------------------------------
    def update_dispatch(
        self,
        component: tuple[int, ...],
        *,
        cycle_index: int,
        action: str,
        gained: bool,
        scope: tuple[int, ...],
        realized_gain: float,
        predicted_gain: float,
    ) -> None:
        """Apply the section-3/section-7 feedback rules after one dispatch."""

        component = tuple(component)
        if action == OC_ACTION_ARBITRATION:
            # Arbitration evaluates candidates but does not dispatch an operator;
            # it must not consume dispatch stall, cooldown, pulse, or qhat credit.
            return
        alpha = self.config.ema_alpha
        if scope:
            credit = min(
                1.0,
                max(0.0, realized_gain / max(predicted_gain, self.config.gain_floor)),
            )
            pairs = [
                (variable, group)
                for variable in scope
                for group in self.structure.owners(variable)
            ]
            for pair in pairs:
                self.qhat[pair] = min(
                    1.0, max(0.0, (1.0 - alpha) * self.qhat.get(pair, 1.0) + alpha * credit)
                )
        pulse = self.pulse_fes[component]
        if gained:
            pulse = int(math.floor(pulse * self.config.gamma_up))
        else:
            pulse = int(math.ceil(pulse * self.config.gamma_down))
        self.pulse_fes[component] = min(
            self.config.pulse_max_fes, max(self.config.pulse_min_fes, pulse)
        )
        self.stall[component] = 0 if gained else self.stall[component] + 1
        self.cooldown_until[component] = cycle_index + 1 + self.config.cooldown_cycles
        if action == "aor":
            self.escalation_used[component] = True

    def record_stall_guard(self, component: tuple[int, ...], *, cycle_index: int) -> None:
        """Close a failed repair path after a guarded retry.

        A guarded arbitration is intentionally not normal dispatch feedback:
        it must not change pulse or qhat.  It does, however, consume the
        component's remaining retry opportunity so repeated high residual
        observations cannot escalate a known failed repair into another
        operator window.  SMP sensing remains enabled through
        :meth:`sensing_components`.
        """

        component = tuple(component)
        if component not in self.stall:
            raise ValueError("unknown component")
        if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0:
            raise ValueError("cycle_index must be a non-negative integer")
        if self.stall[component] <= 0:
            raise ValueError("stall guard requires a previously stalled component")
        self.stall[component] = self.config.stall_cap
        self.conflict_streak[component] = 0
        self.level[component] = "low"
        self.cooldown_until[component] = cycle_index + 1 + self.config.cooldown_cycles

    def active_components(self) -> tuple[tuple[int, ...], ...]:
        """Return components still eligible for operator dispatch."""

        return tuple(
            component
            for component in self.level
            if self.stall[component] < self.config.stall_cap
        )

    def sensing_components(self) -> tuple[tuple[int, ...], ...]:
        """Return all Phase-I components eligible for persistent SMP sensing.

        Dispatch stall and cooldown are operator lifecycle controls.  They do not
        terminate the owner-local proposal lane, which must retain its reserved
        Phase-II budget after a low-value operator window.
        """

        return tuple(self.level)

    # ------------------------------------------------------------------
    # snapshot / restore
    # ------------------------------------------------------------------
    def payload(self) -> dict[str, object]:
        return {
            "schema_version": OC_STATE_SCHEMA,
            "checkpoint_hash": self.checkpoint_hash,
            "config_hash": self.config.config_hash,
            "structure_hash": self.structure_hash,
            "qhat": sorted([v, g, x] for (v, g), x in self.qhat.items()),
            "ema_c": sorted([v, x] for v, x in self.ema_c.items()),
            "level": sorted([list(k), lv] for k, lv in self.level.items()),
            "enter_count": sorted([list(k), v] for k, v in self.enter_count.items()),
            "exit_count": sorted([list(k), v] for k, v in self.exit_count.items()),
            "conflict_streak": sorted([list(k), v] for k, v in self.conflict_streak.items()),
            "cooldown_until": sorted([list(k), v] for k, v in self.cooldown_until.items()),
            "stall": sorted([list(k), v] for k, v in self.stall.items()),
            "escalation_used": sorted([list(k), v] for k, v in self.escalation_used.items()),
            "pulse_fes": sorted([list(k), v] for k, v in self.pulse_fes.items()),
            "shared_patch": self.shared_patch,
        }

    def snapshot(self) -> OcStateSnapshot:
        payload = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return OcStateSnapshot(payload=payload, state_hash=hashlib.sha256(payload).hexdigest())

    def restore(self, snapshot: OcStateSnapshot) -> None:
        if not isinstance(snapshot, OcStateSnapshot):
            raise TypeError("snapshot must be OcStateSnapshot")
        data = json.loads(snapshot.payload.decode("utf-8"))
        if data.get("schema_version") not in (OC_STATE_SCHEMA, OC_STATE_SCHEMA_V1):
            raise ValueError("unsupported coordinator state schema")
        if data.get("checkpoint_hash", "") != self.checkpoint_hash:
            raise ValueError("coordinator state checkpoint hash does not match")
        if data.get("config_hash") != self.config.config_hash:
            raise ValueError("coordinator state config hash does not match")
        if data.get("structure_hash") != self.structure_hash:
            raise ValueError("coordinator state structure hash does not match")
        self.qhat = {(int(v), int(g)): float(x) for v, g, x in data["qhat"]}
        self.ema_c = {int(v): float(x) for v, x in data["ema_c"]}
        for name in (
            "level",
            "enter_count",
            "exit_count",
            "conflict_streak",
            "cooldown_until",
            "stall",
            "escalation_used",
            "pulse_fes",
        ):
                setattr(self, name, {tuple(int(g) for g in key): value for key, value in data[name]})
        self.shared_patch = dict(data.get("shared_patch", {})) or {}


__all__ = ["CoordinatorState", "OcStateSnapshot", "OC_STATE_SCHEMA", "OC_STATE_SCHEMA_V1"]
