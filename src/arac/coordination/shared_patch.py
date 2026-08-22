"""Stateful Shared-Patch Kernel for CTP-class operators (plan 2026-08-22).

Implements the pre-registered shared-patch mechanism exactly as specified in
the user plan ("ARAC-OC Shared-Patch 完整落地方案"), frozen parameters:

    k_patch              = 8 FE (4 rounds x 2 candidates)
    radius_success_scale = 1.25
    radius_failure_scale = 0.50
    radius_upper_bound   = 4 * base_radius
    u_decay              = 0.80
    u_context_decay      = 0.25
    u_upper_bound        = 4.0
    min_relative_gain    = 1e-6
    min_absolute_gain    = 1e-12

Candidate directions come ONLY from owner proposal differences; ``u_j`` is
never used for candidate direction, action type, or the outer selector.
The lane is carved from the CTP operator reservation; with fewer than
``k_patch`` FE available the kernel refuses with an explicit
``patch_budget_unavailable`` status (the caller must surface it in the
receipt and may fall back to the plain CTP baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

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
EPS = 1e-12


@dataclass
class SharedVariableState:
    """Persistent per-shared-variable patch state (z, u, r)."""

    center: float
    radius: float
    base_radius: float
    u: float = 0.0


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
    context_hash_before: str
    context_hash_after: str
    state_hash: str
    radius_trace: tuple[float, ...]
    u_trace: tuple[float, ...]
    budget_status: str
    reset_count: int


def _owner_views(structure: OverlapStructure, proposals, scope) -> dict[int, dict[int, float]]:
    by_group = {proposal.group: proposal for proposal in proposals}
    views: dict[int, dict[int, float]] = {}
    for variable in scope:
        owners = {}
        for group in structure.owners(variable):
            proposal = by_group.get(group)
            if proposal is not None:
                owners[group] = float(proposal.value(variable))
        views[variable] = owners
    return views


def _base_radius(views, variable, span_j: float) -> float:
    values = list(views[variable].values())
    disagreement = max(values) - min(values) if len(values) >= 2 else 0.0
    return float(min(max(EPS, disagreement, 1e-6 * span_j), span_j))


class SharedPatchKernel:
    """Persistent shared-variable patch kernel mounted inside CTP operators."""

    def __init__(self) -> None:
        self.variables: dict[int, SharedVariableState] = {}
        self.context_hash: str = ""
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
                }
                for variable, state in sorted(self.variables.items())
            },
            "context_hash": self.context_hash,
            "reset_count": int(self.reset_count),
        }

    def load(self, payload: dict[str, object] | None) -> None:
        if not payload:
            return
        variables = payload.get("variables", {})
        self.variables = {
            int(variable): SharedVariableState(
                center=float(state["center"]),
                radius=float(state["radius"]),
                base_radius=float(state["base_radius"]),
                u=float(state.get("u", 0.0)),
            )
            for variable, state in variables.items()
        }
        self.context_hash = str(payload.get("context_hash", ""))
        self.reset_count = int(payload.get("reset_count", 0))

    def state_hash(self) -> str:
        import hashlib
        import json

        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # kernel application
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
        if budget_fes > ledger.remaining:
            return SharedPatchResult(
                component=tuple(component),
                scope=scope,
                mode=mode,
                candidate_trace=(),
                consumed_fes=0,
                best_error_before=float(ledger.best_error),
                best_error_after=float(ledger.best_error),
                accepted_candidate=None,
                context_reset=False,
                context_hash_before=self.context_hash,
                context_hash_after=self.context_hash,
                state_hash=self.state_hash(),
                radius_trace=(),
                u_trace=(),
                budget_status="patch_budget_unavailable",
                reset_count=self.reset_count,
            )

        component = tuple(component)
        proposal_list = tuple(proposals)
        views = _owner_views(structure, proposal_list, scope)
        lower = ledger.problem.lower_array
        upper = ledger.problem.upper_array
        span = upper - lower
        incumbent = np.asarray(ledger.best_x, dtype=float).copy()
        f0 = float(ledger.best_error)

        context_reset = bool(self.context_hash and self.context_hash != context_hash)
        hash_before = self.context_hash
        if context_reset:
            self.reset_count += 1
        # initialize / context-reset per-variable state
        for variable in scope:
            span_j = float(span[variable])
            base = _base_radius(views, variable, span_j)
            state = self.variables.get(variable)
            if state is None:
                self.variables[variable] = SharedVariableState(
                    center=float(incumbent[variable]), radius=base, base_radius=base, u=0.0
                )
            elif context_reset:
                state.center = float(incumbent[variable])
                state.radius = base
                state.u *= U_CONTEXT_DECAY
                state.base_radius = base
        self.context_hash = context_hash

        # disagreement integrator (diagnostics/scope only, never direction)
        normalized_disagreement = float(
            np.mean(
                [
                    min(
                        1.0,
                        (max(views[v].values()) - min(views[v].values())) / float(span[v]),
                    )
                    if len(views[v]) >= 2
                    else 0.0
                    for v in scope
                ]
            )
        )
        for variable in scope:
            state = self.variables[variable]
            state.u = min(U_UPPER_BOUND, U_DECAY * state.u + normalized_disagreement)

        rng = np.random.default_rng(seed)
        gain_threshold = max(MIN_ABSOLUTE_GAIN, MIN_RELATIVE_GAIN * max(abs(f0), 1.0))
        rounds: list[PatchRoundTrace] = []
        radius_trace: list[float] = []
        u_trace: list[float] = []
        accepted_candidate: str | None = None
        start = ledger.count

        for round_index in range(PATCH_ROUNDS):
            pair = self._candidate_pair(
                round_index,
                mode,
                views,
                scope,
                incumbent,
                structure,
                proposal_list,
                span,
                rng,
            )
            names, vectors = pair
            batch = np.asarray(vectors, dtype=float)
            np.clip(batch, lower, upper, out=batch)
            before = float(ledger.best_error)
            errors = np.asarray(ledger.evaluate(batch), dtype=float)
            best_index = int(np.argmin(errors))
            gain = before - float(ledger.best_error)
            accepted = bool(
                float(errors[best_index]) < before - EPS
                and gain >= gain_threshold
                and float(ledger.best_error) < before - EPS
            )
            if accepted:
                accepted_candidate = names[best_index]
                incumbent = np.asarray(ledger.best_x, dtype=float).copy()
                if mode == "full":
                    for variable in scope:
                        state = self.variables[variable]
                        state.center = float(incumbent[variable])
                        state.radius = min(
                            state.base_radius * RADIUS_UPPER_MULTIPLE,
                            state.radius * RADIUS_SUCCESS_SCALE,
                        )
            else:
                if mode == "full":
                    for variable in scope:
                        state = self.variables[variable]
                        state.radius = max(EPS, state.radius * RADIUS_FAILURE_SCALE)
            rounds.append(
                PatchRoundTrace(
                    round_index=round_index,
                    candidate_names=names,
                    errors=(float(errors[0]), float(errors[1])),
                    accepted=accepted,
                    radii=tuple(self.variables[v].radius for v in scope),
                )
            )
            radius_trace.append(float(np.mean([self.variables[v].radius for v in scope])))
            u_trace.append(float(np.mean([self.variables[v].u for v in scope])))

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
            context_reset=context_reset,
            context_hash_before=hash_before,
            context_hash_after=self.context_hash,
            state_hash=self.state_hash(),
            radius_trace=tuple(radius_trace),
            u_trace=tuple(u_trace),
            budget_status="executed",
            reset_count=self.reset_count,
        )

    # ------------------------------------------------------------------
    def _candidate_pair(
        self,
        round_index: int,
        mode: str,
        views,
        scope,
        incumbent: np.ndarray,
        structure: OverlapStructure,
        proposal_list,
        span: np.ndarray,
        rng,
    ) -> tuple[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
        """Two candidates for one round, from owner values / consensus / disagreement."""

        by_group = {proposal.group: proposal for proposal in proposal_list}
        style = round_index % 3 if mode in ("candidates", "full") else 0
        if style == 1:
            name = "consensus"
        elif style == 2:
            name = "disagreement"
        else:
            name = "owner_conditioned"

        def radius_of(variable: int) -> float:
            state = self.variables[variable]
            return state.base_radius if mode in ("v2", "candidates") else state.radius

        def center_of(variable: int) -> float:
            state = self.variables[variable]
            return float(incumbent[variable]) if mode in ("v2", "candidates") else state.center

        def owner_value(variable: int, rank: int = 0) -> float:
            owners = views[variable]
            if not owners:
                return float(incumbent[variable])
            ranked = sorted(
                owners,
                key=lambda group: (
                    -float(by_group[group].improvement) if group in by_group else 0.0,
                    group,
                ),
            )
            return owners[ranked[min(rank, len(ranked) - 1)]]

        first = incumbent.copy()
        second = incumbent.copy()
        for variable in scope:
            center = center_of(variable)
            radius = radius_of(variable)
            owners = views[variable]
            if name == "owner_conditioned":
                first[variable] = owner_value(variable, rank=0)
                second[variable] = owner_value(variable, rank=1 if len(owners) >= 2 else 0)
                noise = rng.normal(0.0, max(EPS, radius))
                first[variable] += noise
                second[variable] -= noise
            elif name == "consensus":
                value = float(np.mean(list(owners.values()))) if owners else center
                noise = rng.normal(0.0, max(EPS, radius))
                first[variable] = value + noise
                second[variable] = value - noise
            else:  # disagreement
                values = sorted(owners.values()) if owners else [center]
                direction = (values[-1] - values[0]) if len(values) >= 2 else radius
                scale = radius / max(abs(direction), EPS) if abs(direction) > EPS else 1.0
                step = direction * scale
                first[variable] = center + step
                second[variable] = center - step
        return (f"{name}+", f"{name}-"), (first, second)


__all__ = [
    "K_PATCH_FES",
    "PATCH_MODES",
    "PatchRoundTrace",
    "SharedPatchKernel",
    "SharedPatchResult",
    "SharedVariableState",
]
