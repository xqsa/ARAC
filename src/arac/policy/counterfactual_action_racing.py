"""Pure contracts for the ARAC-CAR writeback calibration protocol.

This module deliberately has no HCC imports.  It defines the small, auditable
boundary between runtime evidence and the branch executor.  Problem identity,
historical outcomes, and final results belong to :class:`AuditEnvelope`, never
to :class:`DispatchEvidence`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Mapping


class CARInvariantError(ValueError):
    """Raised when a CAR hard gate would otherwise be bypassed."""


def _finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise CARInvariantError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True)
class AuditEnvelope:
    """Identity fields retained for audit output, outside runtime dispatch."""

    run_id: str
    problem_id: str
    seed: int


@dataclass(frozen=True)
class DispatchEvidence:
    """Identity-free evidence that is allowed to select a CAR candidate."""

    graph_fingerprint: str
    component_fingerprint: str
    candidate_action_name: str
    candidate_action_family: str
    overlap_strength: float
    shared_variable_count: int
    evidence_sweep_count: int
    evidence_coverage: float
    writeback_norm: float

    _RUNTIME_FIELDS: ClassVar[tuple[str, ...]] = (
        "graph_fingerprint",
        "component_fingerprint",
        "candidate_action_name",
        "candidate_action_family",
        "overlap_strength",
        "shared_variable_count",
        "evidence_sweep_count",
        "evidence_coverage",
        "writeback_norm",
    )
    _FORBIDDEN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "case",
            "case_label",
            "problem_id",
            "run_id",
            "seed",
            "lane",
            "algorithm",
            "function_name",
            "function_family",
            "paper_best",
            "historical_best",
            "historical_optimum",
            "final_outcome",
            "final_result",
            "final_error",
            "result",
            "fitness",
            "objective",
            "baseline",
        }
    )

    def __post_init__(self) -> None:
        if not self.graph_fingerprint or not self.component_fingerprint:
            raise CARInvariantError("graph and component fingerprints are required")
        if not self.candidate_action_name or not self.candidate_action_family:
            raise CARInvariantError("candidate action identity is required")
        if self.candidate_action_family == "fallback":
            raise CARInvariantError("candidate action family cannot be fallback")
        if int(self.shared_variable_count) < 0:
            raise CARInvariantError("shared_variable_count must be non-negative")
        if int(self.evidence_sweep_count) < 2:
            raise CARInvariantError("at least two complete evidence sweeps are required")
        coverage = _finite("evidence_coverage", self.evidence_coverage)
        if not 0.0 <= coverage <= 1.0:
            raise CARInvariantError("evidence_coverage must be within [0, 1]")
        _finite("overlap_strength", self.overlap_strength)
        _finite("writeback_norm", self.writeback_norm)

    @classmethod
    def runtime_field_names(cls) -> tuple[str, ...]:
        return cls._RUNTIME_FIELDS

    @classmethod
    def forbidden_field_names(cls) -> tuple[str, ...]:
        """Return audit-only names rejected at the runtime boundary."""

        return tuple(sorted(cls._FORBIDDEN_FIELDS))

    @classmethod
    def from_runtime_payload(cls, payload: Mapping[str, object]) -> "DispatchEvidence":
        keys = {str(key) for key in payload}
        forbidden = sorted(keys & cls._FORBIDDEN_FIELDS)
        if forbidden:
            raise CARInvariantError(
                "forbidden runtime field(s): " + ", ".join(forbidden)
            )
        unknown = sorted(keys - set(cls._RUNTIME_FIELDS))
        if unknown:
            raise CARInvariantError("unknown runtime field(s): " + ", ".join(unknown))
        missing = sorted(set(cls._RUNTIME_FIELDS) - keys)
        if missing:
            raise CARInvariantError("missing runtime field(s): " + ", ".join(missing))
        return cls(**{name: payload[name] for name in cls._RUNTIME_FIELDS})


@dataclass(frozen=True)
class ProbeSeedDescriptor:
    """Counter-based common-random-number descriptor shared by both arms."""

    seed: int
    canonical_key: str


def derive_probe_seed(
    *,
    base_seed: int,
    sweep_index: int,
    component_fingerprint: str,
    pair_index: int,
) -> ProbeSeedDescriptor:
    if int(sweep_index) < 0 or int(pair_index) < 0:
        raise CARInvariantError("sweep_index and pair_index must be non-negative")
    if not component_fingerprint:
        raise CARInvariantError("component_fingerprint is required")
    canonical_key = (
        f"arac-car|base={int(base_seed)}|sweep={int(sweep_index)}|"
        f"component={component_fingerprint}|pair={int(pair_index)}"
    )
    digest = hashlib.sha256(canonical_key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") & 0x7FFFFFFF
    return ProbeSeedDescriptor(seed=seed, canonical_key=canonical_key)


@dataclass
class CARBudgetLedger:
    """Single FE ledger for committed trajectory and both probe arms."""

    max_fes: int
    probe_fe_limit: int
    committed_fe: int = 0
    probe_fe: int = 0
    _charged_pairs: set[int] = field(default_factory=set, init=False, repr=False)
    _pair_reservations: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.max_fes) <= 0:
            raise CARInvariantError("max_fes must be positive")
        if int(self.probe_fe_limit) < 0:
            raise CARInvariantError("probe_fe_limit must be non-negative")
        if int(self.committed_fe) < 0 or int(self.probe_fe) < 0:
            raise CARInvariantError("ledger FE counters must be non-negative")
        if self.total_fe > int(self.max_fes):
            raise CARInvariantError("initial ledger exceeds total FE budget")
        if self.probe_fe > int(self.probe_fe_limit):
            raise CARInvariantError("initial ledger exceeds probe FE limit")

    @property
    def total_fe(self) -> int:
        return int(self.committed_fe) + int(self.probe_fe)

    @property
    def remaining_probe_fe(self) -> int:
        return int(self.probe_fe_limit) - int(self.probe_fe)

    @property
    def remaining_total_fe(self) -> int:
        return int(self.max_fes) - self.total_fe

    def charge_committed(self, *, stage: str, actual_fe: int) -> None:
        if not stage:
            raise CARInvariantError("ledger stage is required")
        amount = int(actual_fe)
        if amount < 0:
            raise CARInvariantError("actual FE must be non-negative")
        if self.total_fe + amount > int(self.max_fes):
            raise CARInvariantError("total FE budget exceeded")
        self.committed_fe += amount

    def charge_pair(self, *, pair_index: int, fallback_fe: int, candidate_fe: int) -> None:
        index = int(pair_index)
        fallback = int(fallback_fe)
        candidate = int(candidate_fe)
        if index < 0 or index in self._charged_pairs or index in self._pair_reservations:
            raise CARInvariantError("pair index must be unique and non-negative")
        if fallback < 0 or candidate < 0:
            raise CARInvariantError("arm FE must be non-negative")
        if fallback != candidate:
            raise CARInvariantError("paired arms must have equal actual FE")
        self.reserve_pair(pair_index=index, arm_fe=fallback)
        self.commit_reserved_pair(
            pair_index=index,
            fallback_fe=fallback,
            candidate_fe=candidate,
        )

    def reserve_pair(self, *, pair_index: int, arm_fe: int) -> None:
        index = int(pair_index)
        arm = int(arm_fe)
        if index < 0 or index in self._charged_pairs or index in self._pair_reservations:
            raise CARInvariantError("pair index must be unique and non-negative")
        if arm <= 0:
            raise CARInvariantError("reserved arm FE must be positive")
        pair_fe = 2 * arm
        if self.probe_fe + pair_fe > int(self.probe_fe_limit):
            raise CARInvariantError("probe FE limit exceeded")
        if self.total_fe + pair_fe > int(self.max_fes):
            raise CARInvariantError("total FE budget exceeded")
        self.probe_fe += pair_fe
        self._pair_reservations[index] = arm

    def commit_reserved_pair(
        self,
        *,
        pair_index: int,
        fallback_fe: int,
        candidate_fe: int,
    ) -> None:
        index = int(pair_index)
        reserved = self._pair_reservations.get(index)
        if reserved is None:
            raise CARInvariantError("pair FE must be reserved before execution")
        fallback = int(fallback_fe)
        candidate = int(candidate_fe)
        if fallback != candidate:
            raise CARInvariantError("paired arms must have equal actual FE")
        if fallback != reserved:
            raise CARInvariantError("actual arm FE must match the reserved component horizon")
        del self._pair_reservations[index]
        self._charged_pairs.add(index)


@dataclass(frozen=True)
class PairedProbeObservation:
    pair_index: int
    phase1_probe_fitness_before: float
    fallback_after: float
    candidate_after: float
    fallback_fe: int
    candidate_fe: int
    graph_fingerprint: str
    component_fingerprint: str
    action_family: str
    fallback_start_state_fingerprint: str
    candidate_start_state_fingerprint: str
    fallback_evaluator_id: str
    candidate_evaluator_id: str
    seed_descriptor: ProbeSeedDescriptor
    normalized_delta: float = field(init=False)

    @classmethod
    def create(cls, **kwargs: object) -> "PairedProbeObservation":
        return cls(**kwargs)

    def __post_init__(self) -> None:
        if int(self.pair_index) < 0:
            raise CARInvariantError("pair_index must be non-negative")
        start = _finite("phase1_probe_fitness_before", self.phase1_probe_fitness_before)
        fallback = _finite("fallback_after", self.fallback_after)
        candidate = _finite("candidate_after", self.candidate_after)
        if int(self.fallback_fe) != int(self.candidate_fe):
            raise CARInvariantError("paired arms must have equal actual FE")
        if int(self.fallback_fe) < 0:
            raise CARInvariantError("arm FE must be non-negative")
        if (
            not self.fallback_start_state_fingerprint
            or self.fallback_start_state_fingerprint
            != self.candidate_start_state_fingerprint
        ):
            raise CARInvariantError("paired arms must start from an identical checkpoint")
        if not self.graph_fingerprint or not self.component_fingerprint:
            raise CARInvariantError("probe graph and component fingerprints are required")
        if not self.action_family or self.action_family == "fallback":
            raise CARInvariantError("probe action family is invalid")
        if not self.fallback_evaluator_id or not self.candidate_evaluator_id:
            raise CARInvariantError("branch-local evaluator ids are required")
        if self.fallback_evaluator_id == self.candidate_evaluator_id:
            raise CARInvariantError("probe arms must use distinct evaluator records")
        denominator = max(abs(start), abs(fallback), abs(candidate), 1e-12)
        object.__setattr__(self, "normalized_delta", (fallback - candidate) / denominator)


@dataclass(frozen=True)
class RiskGateResult:
    mean: float
    sample_std: float
    lcb: float
    tail: float
    committed: bool
    adopted_arm: str
    abstain_reasons: tuple[str, ...]


def evaluate_risk_gate(
    observations: tuple[PairedProbeObservation, ...],
    *,
    epsilon: float = 1e-12,
) -> RiskGateResult:
    """Evaluate the frozen K=3 runtime safety score and worst replicate.

    ``lcb`` is a stable trace field name, not a statistical confidence bound.
    With three within-run checkpoints it is only a conservative dispatch score.
    """

    if len(observations) != 3:
        raise CARInvariantError("CAR-W requires exactly 3 paired probes")
    if not math.isfinite(float(epsilon)) or float(epsilon) < 0.0:
        raise CARInvariantError("epsilon must be finite and non-negative")
    ordered = tuple(sorted(observations, key=lambda item: item.pair_index))
    if tuple(item.pair_index for item in ordered) != (0, 1, 2):
        raise CARInvariantError("paired probe indices must be exactly (0, 1, 2)")
    deltas = tuple(item.normalized_delta for item in ordered)
    mean = statistics.fmean(deltas)
    sample_std = statistics.stdev(deltas)
    lcb = mean - sample_std / math.sqrt(3.0)
    tail = min(deltas)
    reasons: list[str] = []
    if len({item.graph_fingerprint for item in ordered}) != 1:
        reasons.append("unstable_graph_fingerprint")
    if len({item.component_fingerprint for item in ordered}) != 1:
        reasons.append("unstable_component_fingerprint")
    if len({item.action_family for item in ordered}) != 1:
        reasons.append("unstable_action_family")
    if ordered[-1].candidate_after > ordered[-1].phase1_probe_fitness_before + float(epsilon):
        reasons.append("candidate_endpoint_worse_than_start")
    if lcb <= float(epsilon):
        reasons.append("lcb_not_positive")
    if tail < 0.0:
        reasons.append("lower_tail_negative")
    committed = not reasons
    return RiskGateResult(
        mean=mean,
        sample_std=sample_std,
        lcb=lcb,
        tail=tail,
        committed=committed,
        adopted_arm="candidate" if committed else "fallback",
        abstain_reasons=tuple(reasons),
    )


def evaluate_first_pair_futility(
    observation: PairedProbeObservation,
    *,
    epsilon: float = 1e-12,
) -> RiskGateResult:
    """Fail closed after the first pair when a candidate is already futile."""

    if not math.isfinite(float(epsilon)) or float(epsilon) < 0.0:
        raise CARInvariantError("epsilon must be finite and non-negative")
    reasons: list[str] = []
    if observation.normalized_delta <= float(epsilon):
        reasons.append("futility_pair_not_positive")
    if observation.candidate_after > (
        observation.phase1_probe_fitness_before + float(epsilon)
    ):
        reasons.append("candidate_endpoint_worse_than_start")
    if not reasons:
        raise CARInvariantError("first pair is not futile")
    return RiskGateResult(
        mean=observation.normalized_delta,
        sample_std=0.0,
        lcb=observation.normalized_delta,
        tail=observation.normalized_delta,
        committed=False,
        adopted_arm="fallback",
        abstain_reasons=tuple(reasons),
    )


@dataclass
class BranchState:
    """Small pure state carrier used to prove adoption does not alias a branch."""

    incumbent: tuple[float, ...]
    committed_fitness: float
    evaluator_record: list[float]
    state_fingerprint: str
    state_payload: dict[str, object] = field(default_factory=dict)

    def clone(self) -> "BranchState":
        return copy.deepcopy(self)


def fingerprint_branch_state(state: BranchState) -> str:
    """Hash only committed optimizer state, never the mutable evaluator record."""

    payload = {
        "incumbent": [float(value) for value in state.incumbent],
        "committed_fitness": _finite("committed_fitness", state.committed_fitness),
        "state_payload": state.state_payload,
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CARInvariantError("branch state payload must be finite JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BranchManifest:
    pair_index: int
    arm: str
    evaluator_id: str
    requested_fe: int
    actual_fe: int
    record_sha256: str
    record_best: float
    state_fingerprint_before: str
    state_fingerprint_after: str
    seed_descriptor: ProbeSeedDescriptor


@dataclass(frozen=True)
class CARProbeExecutionResult:
    observations: tuple[PairedProbeObservation, ...]
    gate: RiskGateResult
    adopted_state: BranchState
    branch_manifests: tuple[BranchManifest, ...]
    accounting_record: tuple[float, ...]


class CARProbeExecutor:
    """Execute one complete component horizon with isolated paired branches."""

    def __init__(
        self,
        *,
        evaluator_factory: Callable[[], object],
        ledger: CARBudgetLedger,
        base_seed: int,
        sweep_index: int,
        graph_fingerprint: str,
        component_fingerprint: str,
        action_family: str,
        arm_fes: int,
    ) -> None:
        if not graph_fingerprint or not component_fingerprint:
            raise CARInvariantError("probe graph and component fingerprints are required")
        if not action_family or action_family == "fallback":
            raise CARInvariantError("probe action family is invalid")
        if int(arm_fes) <= 0:
            raise CARInvariantError("arm_fes must be positive")
        self._evaluator_factory = evaluator_factory
        self._ledger = ledger
        self._base_seed = int(base_seed)
        self._sweep_index = int(sweep_index)
        self._graph_fingerprint = graph_fingerprint
        self._component_fingerprint = component_fingerprint
        self._action_family = action_family
        self._arm_fes = int(arm_fes)

    def execute(
        self,
        *,
        initial_checkpoint: BranchState,
        fallback_transition: Callable[[BranchState, object, ProbeSeedDescriptor, int], BranchState],
        candidate_transition: Callable[[BranchState, object, ProbeSeedDescriptor, int], BranchState],
        branch_order: tuple[str, str] = ("fallback", "candidate"),
        early_futility_abort: bool = False,
    ) -> CARProbeExecutionResult:
        if set(branch_order) != {"fallback", "candidate"} or len(branch_order) != 2:
            raise CARInvariantError("branch_order must contain fallback and candidate exactly once")
        current_checkpoint = initial_checkpoint.clone()
        observations: list[PairedProbeObservation] = []
        manifests: list[BranchManifest] = []
        branch_evaluators: list[object] = []
        final_pair_states: dict[str, BranchState] = {}
        pair_states_history: list[dict[str, BranchState]] = []

        for pair_index in range(3):
            seed_descriptor = derive_probe_seed(
                base_seed=self._base_seed,
                sweep_index=self._sweep_index,
                component_fingerprint=self._component_fingerprint,
                pair_index=pair_index,
            )
            pair_states: dict[str, BranchState] = {}
            pair_fitness: dict[str, float] = {}
            pair_fe: dict[str, int] = {}
            self._ledger.reserve_pair(pair_index=pair_index, arm_fe=self._arm_fes)
            for arm in branch_order:
                evaluator = self._evaluator_factory()
                if any(evaluator is existing for existing in branch_evaluators):
                    raise CARInvariantError("each arm needs a fresh branch-local evaluator")
                branch_evaluators.append(evaluator)
                record = getattr(evaluator, "fitness_record", None)
                if not isinstance(record, list) or record:
                    raise CARInvariantError("each arm needs a fresh branch-local evaluator")
                transition = fallback_transition if arm == "fallback" else candidate_transition
                start = current_checkpoint.clone()
                state = transition(start, evaluator, seed_descriptor, self._arm_fes)
                if not isinstance(state, BranchState):
                    raise CARInvariantError("component transition must return BranchState")
                actual_fe = len(record)
                if actual_fe != self._arm_fes:
                    raise CARInvariantError(
                        "complete component horizon must consume the requested FE"
                    )
                if state.evaluator_record != record:
                    raise CARInvariantError("branch state must retain only its local evaluator record")
                if state.state_fingerprint != fingerprint_branch_state(state):
                    raise CARInvariantError("branch state fingerprint is invalid")
                evaluator_id = f"pair-{pair_index}-{arm}"
                record_sha256 = hashlib.sha256(
                    repr(tuple(float(value) for value in record)).encode("utf-8")
                ).hexdigest()
                manifests.append(
                    BranchManifest(
                        pair_index=pair_index,
                        arm=arm,
                        evaluator_id=evaluator_id,
                        requested_fe=self._arm_fes,
                        actual_fe=actual_fe,
                        record_sha256=record_sha256,
                        record_best=min(float(value) for value in record),
                        state_fingerprint_before=current_checkpoint.state_fingerprint,
                        state_fingerprint_after=state.state_fingerprint,
                        seed_descriptor=seed_descriptor,
                    )
                )
                pair_states[arm] = state
                pair_fitness[arm] = float(state.committed_fitness)
                pair_fe[arm] = actual_fe
            self._ledger.commit_reserved_pair(
                pair_index=pair_index,
                fallback_fe=pair_fe["fallback"],
                candidate_fe=pair_fe["candidate"],
            )
            observations.append(
                PairedProbeObservation.create(
                    pair_index=pair_index,
                    phase1_probe_fitness_before=current_checkpoint.committed_fitness,
                    fallback_after=pair_fitness["fallback"],
                    candidate_after=pair_fitness["candidate"],
                    fallback_fe=pair_fe["fallback"],
                    candidate_fe=pair_fe["candidate"],
                    graph_fingerprint=self._graph_fingerprint,
                    component_fingerprint=self._component_fingerprint,
                    action_family=self._action_family,
                    fallback_start_state_fingerprint=current_checkpoint.state_fingerprint,
                    candidate_start_state_fingerprint=current_checkpoint.state_fingerprint,
                    fallback_evaluator_id=f"pair-{pair_index}-fallback",
                    candidate_evaluator_id=f"pair-{pair_index}-candidate",
                    seed_descriptor=seed_descriptor,
                )
            )
            first_pair_futile = (
                pair_index == 0
                and (
                    observations[0].normalized_delta <= 1e-12
                    or observations[0].candidate_after
                    > observations[0].phase1_probe_fitness_before + 1e-12
                )
            )
            if early_futility_abort and first_pair_futile:
                gate = evaluate_first_pair_futility(observations[0])
                pair_states_history.append(
                    {arm: state.clone() for arm, state in pair_states.items()}
                )
                selected = pair_states["fallback"]
                accounting_record = tuple(
                    float(value) for value in selected.evaluator_record
                ) + (float(selected.committed_fitness),) * self._arm_fes
                return CARProbeExecutionResult(
                    observations=tuple(observations),
                    gate=gate,
                    adopted_state=selected.clone(),
                    branch_manifests=tuple(manifests),
                    accounting_record=accounting_record,
                )
            if pair_index < 2:
                current_checkpoint = pair_states["fallback"].clone()
                current_checkpoint.evaluator_record = []
            else:
                final_pair_states = pair_states
            pair_states_history.append(
                {arm: state.clone() for arm, state in pair_states.items()}
            )

        gate = evaluate_risk_gate(tuple(observations))
        adopted = adopt_final_pair_branch(
            gate=gate,
            fallback=final_pair_states["fallback"],
            candidate=final_pair_states["candidate"],
        )
        accounting_record: list[float] = []
        for pair_index, pair_states in enumerate(pair_states_history):
            selected_arm = "fallback" if pair_index < 2 else gate.adopted_arm
            selected = pair_states[selected_arm]
            accounting_record.extend(float(value) for value in selected.evaluator_record)
            accounting_record.extend(
                [float(selected.committed_fitness)] * self._arm_fes
            )
        return CARProbeExecutionResult(
            observations=tuple(observations),
            gate=gate,
            adopted_state=adopted,
            branch_manifests=tuple(manifests),
            accounting_record=tuple(accounting_record),
        )


def adopt_final_pair_branch(
    *,
    gate: RiskGateResult,
    fallback: BranchState,
    candidate: BranchState,
) -> BranchState:
    """Adopt only the predeclared final arm; never select the best probe."""

    selected = candidate if gate.committed and gate.adopted_arm == "candidate" else fallback
    return selected.clone()
