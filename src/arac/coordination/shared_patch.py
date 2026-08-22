"""Stateful Shared-Patch Kernel for CTP-class operators (revised plan 2026-08-22).

Implements ``docs/arac-oc-shared-patch-completion-plan.md`` (修订版) exactly;
frozen parameters (section 14):

    k_patch                  = 8 FE (4 rounds x 2 candidates)
    radius_success_scale     = 1.25
    radius_failure_scale     = 0.50
    radius_upper_bound       = 4 * base_radius
    u_decay                  = 0.80
    u_context_decay          = 0.25
    u_upper_bound            = 4.0
    min_relative_gain        = 1e-6
    min_absolute_gain        = 1e-12
    conforming_threshold      = 1e-12 * variable_range

Mechanism highlights (revised):

- owner weights ``w_g = max(I_g, 0)`` (uniform when all zero); consensus and
  disagreement are computed from the IMPROVEMENT-WEIGHTED consensus value;
- base radius uses conforming silence: variables whose owner disagreement
  is at or below ``1e-12 * range`` collapse to radius ``eps`` (the patch is
  inert there without any runtime consistency classifier);
- context hashes are PER-VARIABLE and LOCAL: they hash only the incumbent
  coordinates OUTSIDE the write set (write set := current scope plus the
  component's own coordinates), so a patch's own acceptance never triggers
  a reset while external context changes do.  Reset reasons are explicit
  (none / external_context_change / scope_change / checkpoint_change /
  restore);
- ``u_j`` is updated on every scope visit even when the lane budget is
  unavailable (z/r untouched then); its ONLY causal channel is the
  ``(-u_j, -proposal_priority_j, j)`` scope ordering inside the kernel.

Mode semantics (ablation section 9):

    v2          owner-conditioned candidates only, incumbent centre,
                fixed base radius                (A1)
    candidates  + consensus / disagreement candidates, incumbent centre,
                fixed base radius                (A2)
    state       full candidate set, persistent centre z_j, FIXED base
                radius                            (A3)
    full        full candidate set, persistent z_j and adaptive r_j (A4)
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from arac.coordination.overlap import LocalProposal, OverlapStructure
from arac.runtime.ledger import EvaluationLedger

PATCH_MODES = ("v2", "candidates", "state", "full")
K_PATCH_FES = 8
PATCH_ROUNDS = 4
RADIUS_SUCCESS_SCALE = 1.25
RADIUS_FAILURE_SCALE = 0.50
RADIUS_UPPER_MULTIPLE = 4.0
U_DECAY = 0.80
U_CONTEXT_DECAY = 0.25
U_UPPER_BOUND = 4.0
MIN_RELATIVE_GAIN = 1e-6
MIN_ABSOLUTE_GAIN = 1e-12
CONFORMING_THRESHOLD_COEFF = 1e-12
EPS = 1e-12
RESET_REASONS = (
    "none",
    "external_context_change",
    "scope_change",
    "checkpoint_change",
    "restore",
)


@dataclass
class SharedVariableState:
    """Persistent per-shared-variable patch state (z, u, r + local context)."""

    center: float
    radius: float
    base_radius: float
    u: float = 0.0
    context_hash: str = ""
    last_scope: tuple[int, ...] = ()


@dataclass(frozen=True)
class PatchRoundTrace:
    round_index: int
    candidate_names: tuple[str, str]
    errors: tuple[float, float]
    accepted: bool
    radii: tuple[float, ...]


@dataclass(frozen=True)
class SharedPatchResult:
    component: tuple[int, ...]
    scope: tuple[int, ...]
    mode: str
    candidate_trace: tuple[PatchRoundTrace, ...]
    consumed_fes: int
    best_error_before: float
    best_error_after: float
    accepted_candidate: str | None
    context_reset: bool
    context_reset_reason: str
    context_hash_before: str
    context_hash_after: str
    state_hash: str
    radius_trace: tuple[float, ...]
    u_trace: tuple[float, ...]
    budget_status: str
    reset_count: int


def _owner_statistics(structure, proposals, scope):
    """Improvement-weighted consensus / disagreement per shared variable."""

    by_group = {proposal.group: proposal for proposal in proposals}
    stats = {}
    for variable in scope:
        owners = {
            group: float(by_group[group].value(variable))
            for group in structure.owners(variable)
            if group in by_group
        }
        weights = {
            group: max(float(by_group[group].improvement), 0.0)
            for group in owners
        }
        if owners and sum(weights.values()) > 0.0:
            total = sum(weights.values())
            consensus = sum(weights[g] * value for g, value in owners.items()) / total
        elif owners:
            consensus = float(np.mean(list(owners.values())))
        else:
            consensus = None
        sigmas = [float(by_group[group].sigma(variable)) for group in owners]
        disagreement = (
            max(abs(value - consensus) for value in owners.values())
            if owners and consensus is not None
            else 0.0
        )
        stats[variable] = {
            "owners": owners,
            "weights": weights,
            "consensus": consensus,
            "disagreement": disagreement,
            "max_sigma": max(sigmas) if sigmas else 0.0,
        }
    return stats


def _ranked_owners(stats, variable):
    owners = stats[variable]["owners"]
    return sorted(
        owners,
        key=lambda group: (-stats[variable]["weights"].get(group, 0.0), group),
    )


class SharedPatchKernel:
    """Persistent shared-variable patch kernel mounted inside CTP operators."""

    def __init__(self) -> None:
        self.variables: dict[int, SharedVariableState] = {}
        self.stable_hash: str = ""
        self.reset_count: int = 0

    # ------------------------------------------------------------------
    # state serialization (feeds CoordinatorState v2 payload)
    # ------------------------------------------------------------------
    def payload(self) -> dict[str, object]:
        return {
            "variables": {
                str(variable): {
                    "center": float(state.center),
                    "radius": float(state.radius),
                    "base_radius": float(state.base_radius),
                    "u": float(state.u),
                    "context_hash": state.context_hash,
                    "last_scope": list(state.last_scope),
                }
                for variable, state in sorted(self.variables.items())
            },
            "stable_hash": self.stable_hash,
            "reset_count": int(self.reset_count),
        }

    def load(self, payload: dict[str, object] | None) -> None:
        if not payload:
            return
        self.variables = {
            int(variable): SharedVariableState(
                center=float(state["center"]),
                radius=float(state["radius"]),
                base_radius=float(state["base_radius"]),
                u=float(state.get("u", 0.0)),
                context_hash=str(state.get("context_hash", "")),
                last_scope=tuple(int(v) for v in state.get("last_scope", ())),
            )
            for variable, state in payload.get("variables", {}).items()
        }
        self.stable_hash = str(payload.get("stable_hash", ""))
        self.reset_count = int(payload.get("reset_count", 0))

    def state_hash(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def _local_context_hashes(
        self,
        structure: OverlapStructure,
        incumbent: np.ndarray,
        scope,
        component,
        stable_hash: str,
    ) -> dict[int, str]:
        """Per-variable local hash over the incumbent OUTSIDE the write set.

        write set := current shared scope + the component's own coordinates
        (the documented operationalization of "scope + the coordinates this
        CTP round modifies"); therefore the hash covers the remaining
        (external) coordinates: a patch's own acceptance cannot change it,
        while external components' accepted moves do.
        """

        write_set = set(scope)
        for group in component:
            write_set.update(int(v) for v in structure.groups[group])
        external = [
            float(incumbent[variable])
            for variable in range(structure.dimension)
            if variable not in write_set
        ]
        return {
            variable: hashlib.sha256(
                json.dumps(
                    {
                        "stable": stable_hash,
                        "component": sorted(component),
                        "external": external,
                        "variable": int(variable),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for variable in scope
        }

    # ------------------------------------------------------------------
    def apply(
        self,
        component: tuple[int, ...],
        proposals,
        scope,
        context_hash: str,
        *,
        structure: OverlapStructure,
        ledger: EvaluationLedger,
        budget_fes: int = K_PATCH_FES,
        seed: int = 0,
        mode: str = "full",
    ) -> SharedPatchResult:
        if mode not in PATCH_MODES:
            raise ValueError(f"unknown patch mode: {mode}")
        if budget_fes != K_PATCH_FES:
            raise ValueError(f"patch lane is frozen at {K_PATCH_FES} FE")
        scope = tuple(sorted(set(int(v) for v in scope)))
        if not scope:
            raise ValueError("patch scope must be non-empty")
        shared = set(structure.shared_variables)
        if any(variable not in shared for variable in scope):
            raise ValueError("patch scope must contain only shared variables")
        component = tuple(sorted(set(int(g) for g in component)))

        def span_of(variable: int) -> float:
            return float(
                ledger.problem.upper_array[variable] - ledger.problem.lower_array[variable]
            )

        incumbent_now = np.asarray(ledger.best_x, dtype=float).copy()
        stats = _owner_statistics(structure, tuple(proposals), scope)
        stable_changed = bool(self.stable_hash and self.stable_hash != context_hash)

        # per-variable local context check + reset (section 6.1 order:
        # context check BEFORE the u update)
        local_hashes = self._local_context_hashes(
            structure, incumbent_now, scope, component, context_hash
        )
        reset_reason = "none"
        any_reset = stable_changed
        for variable in scope:
            state = self.variables.get(variable)
            if state is None:
                continue
            if stable_changed:
                reason = "checkpoint_change"
            elif state.last_scope != scope:
                reason = "scope_change"
            elif state.context_hash != local_hashes[variable]:
                reason = "external_context_change"
            else:
                reason = "none"
            if reason != "none":
                any_reset = True
                if reset_reason == "none":
                    reset_reason = reason
                state.center = float(incumbent_now[variable])
                base = self._base_radius(stats, variable, span_of(variable))
                state.base_radius = base
                state.radius = base
                state.u *= U_CONTEXT_DECAY
        if stable_changed:
            self.reset_count += 1
            if reset_reason == "none":
                reset_reason = "checkpoint_change"
        self.stable_hash = context_hash

        for variable in scope:
            if variable not in self.variables:
                base = self._base_radius(stats, variable, span_of(variable))
                self.variables[variable] = SharedVariableState(
                    center=float(incumbent_now[variable]),
                    radius=base,
                    base_radius=base,
                    u=0.0,
                    context_hash=local_hashes[variable],
                    last_scope=scope,
                )
            else:
                self.variables[variable].context_hash = local_hashes[variable]
                self.variables[variable].last_scope = scope

        # u update on EVERY scope visit (even when the lane cannot run)
        for variable in scope:
            stat = stats[variable]
            normalizer = max(span_of(variable), stat["max_sigma"], EPS)
            normalized = min(1.0, stat["disagreement"] / normalizer)
            self.variables[variable].u = min(
                U_UPPER_BOUND, U_DECAY * self.variables[variable].u + normalized
            )

        f0 = float(ledger.best_error)
        if budget_fes > ledger.remaining:
            return SharedPatchResult(
                component=component,
                scope=scope,
                mode=mode,
                candidate_trace=(),
                consumed_fes=0,
                best_error_before=f0,
                best_error_after=f0,
                accepted_candidate=None,
                context_reset=any_reset,
                context_reset_reason=reset_reason,
                context_hash_before=self.stable_hash,
                context_hash_after=self.stable_hash,
                state_hash=self.state_hash(),
                radius_trace=(),
                u_trace=tuple(self.variables[v].u for v in scope),
                budget_status="patch_budget_unavailable",
                reset_count=self.reset_count,
            )

        # u's single causal channel: the (-u, -priority, j) scope ordering
        priority = {
            variable: max(stats[variable]["weights"].values(), default=0.0)
            for variable in scope
        }
        ordered_scope = tuple(
            sorted(scope, key=lambda v: (-self.variables[v].u, -priority[v], v))
        )

        rng = np.random.default_rng(seed)
        gain_threshold = max(MIN_ABSOLUTE_GAIN, MIN_RELATIVE_GAIN * max(abs(f0), 1.0))
        rounds: list[PatchRoundTrace] = []
        radius_trace: list[float] = []
        u_trace: list[float] = []
        accepted_candidate: str | None = None
        start = ledger.count
        incumbent = incumbent_now

        for round_index in range(PATCH_ROUNDS):
            names, vectors = self._candidate_pair(
                round_index, mode, stats, ordered_scope, incumbent, rng
            )
            batch = np.asarray(vectors, dtype=float)
            np.clip(
                batch, ledger.problem.lower_array, ledger.problem.upper_array, out=batch
            )
            before = float(ledger.best_error)
            errors = np.asarray(ledger.evaluate(batch), dtype=float)
            best_index = int(np.argmin(errors))
            gain = before - float(ledger.best_error)
            accepted = bool(gain >= gain_threshold and float(ledger.best_error) < before - EPS)
            if accepted:
                accepted_candidate = names[best_index]
                incumbent = np.asarray(ledger.best_x, dtype=float).copy()
                if mode == "full":
                    for variable in ordered_scope:
                        state = self.variables[variable]
                        state.center = float(incumbent[variable])
                        state.radius = min(
                            state.base_radius * RADIUS_UPPER_MULTIPLE,
                            state.radius * RADIUS_SUCCESS_SCALE,
                        )
            elif mode == "full":
                for variable in ordered_scope:
                    state = self.variables[variable]
                    state.radius = max(EPS, state.radius * RADIUS_FAILURE_SCALE)
            rounds.append(
                PatchRoundTrace(
                    round_index=round_index,
                    candidate_names=names,
                    errors=(float(errors[0]), float(errors[1])),
                    accepted=accepted,
                    radii=tuple(self.variables[v].radius for v in ordered_scope),
                )
            )
            radius_trace.append(float(np.mean([self.variables[v].radius for v in ordered_scope])))
            u_trace.append(float(np.mean([self.variables[v].u for v in ordered_scope])))

        if ledger.count - start != budget_fes:
            raise RuntimeError("patch lane FE accounting drifted")
        return SharedPatchResult(
            component=component,
            scope=scope,
            mode=mode,
            candidate_trace=tuple(rounds),
            consumed_fes=ledger.count - start,
            best_error_before=f0,
            best_error_after=float(ledger.best_error),
            accepted_candidate=accepted_candidate,
            context_reset=any_reset,
            context_reset_reason=reset_reason,
            context_hash_before=self.stable_hash,
            context_hash_after=self.stable_hash,
            state_hash=self.state_hash(),
            radius_trace=tuple(radius_trace),
            u_trace=tuple(u_trace),
            budget_status="executed",
            reset_count=self.reset_count,
        )

    # ------------------------------------------------------------------
    def _base_radius(self, stats, variable: int, span_j: float) -> float:
        stat = stats[variable]
        if stat["disagreement"] <= CONFORMING_THRESHOLD_COEFF * span_j:
            return EPS
        return float(min(max(stat["disagreement"], stat["max_sigma"], EPS), span_j))

    def _candidate_pair(
        self,
        round_index: int,
        mode: str,
        stats,
        scope,
        incumbent: np.ndarray,
        rng,
    ):
        style = round_index % 3 if mode in ("candidates", "state", "full") else 0
        if style == 1:
            name = "consensus"
        elif style == 2:
            name = "disagreement"
        else:
            name = "owner_conditioned"

        first = incumbent.copy()
        second = incumbent.copy()
        for variable in scope:
            stat = stats[variable]
            state = self.variables[variable]
            center = (
                float(incumbent[variable]) if mode in ("v2", "candidates") else state.center
            )
            radius = state.radius if mode == "full" else state.base_radius
            owners = stat["owners"]
            ranked = _ranked_owners(stats, variable)
            if name == "owner_conditioned":
                first[variable] = owners[ranked[0]]
                second[variable] = owners[ranked[min(1, len(ranked) - 1)]]
                noise = rng.normal(0.0, max(EPS, radius))
                first[variable] += noise
                second[variable] -= noise
            elif name == "consensus":
                value = stat["consensus"] if stat["consensus"] is not None else center
                noise = rng.normal(0.0, max(EPS, radius))
                first[variable] = value + noise
                second[variable] = value - noise
            else:
                if len(ranked) >= 2:
                    direction = owners[ranked[0]] - owners[ranked[1]]
                else:
                    direction = radius
                scale = radius / max(abs(direction), EPS) if abs(direction) > EPS else 1.0
                step = direction * scale
                first[variable] = center + step
                second[variable] = center - step
        return (f"{name}+", f"{name}-"), (first, second)


def patch_stable_hash(
    checkpoint_hash: str, selector_input_hash: str, selector_output_hash: str
) -> str:
    """Combine the STABLE hash ingredients (no incumbent!) per section 5.1."""

    return hashlib.sha256(
        json.dumps(
            {
                "checkpoint": checkpoint_hash,
                "selector_input": selector_input_hash,
                "selector_output": selector_output_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CONFORMING_THRESHOLD_COEFF",
    "K_PATCH_FES",
    "PATCH_MODES",
    "RESET_REASONS",
    "PatchRoundTrace",
    "SharedPatchKernel",
    "SharedPatchResult",
    "SharedVariableState",
    "patch_stable_hash",
]
