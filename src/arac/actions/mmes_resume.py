"""Frozen contract and deterministic executor for Phase1 MMES resumption."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from arac.actions.action_spec import ActionSpec


PHASE1_MMES_RESUME_ACTION = "phase1_mmes_resume"
PHASE1_MMES_RESUME_SCHEMA = "arac.action.phase1_mmes_resume"
PHASE1_MMES_RESUME_SCHEMA_VERSION = 1
MMES_STATE_HASH_SCHEMA = "mmes-continuation-v1"
MMES_RUN_BLOCK_REFERENCE_VERSION = "arac.vendor.hcc.mmes.run_block-v1"
MMES_PARAMETER_SCHEMA = "arac.vendor.hcc.mmes.parameters-v1"
MMES_VENDOR_TYPE = "HCC.NDAs.MMES.mmes.MMES"
MMES_VENDOR_STATE_TYPE = "HCC.NDAs.MMES.state.MMESState"
NO_RESTART_POLICY = "none"
RUNTIME_TERMINATION_DISABLED = "disabled"
STRICT_IMPROVEMENT_ACCEPTANCE = "strict_improvement"
TRIGGER_SCOPE_PHASE_BOUNDARY = "phase_boundary"

_POSITIVE_INFINITY = "positive_infinity"
_NEGATIVE_INFINITY = "negative_infinity"
_BLOCK_LIMIT = "block_start_plus_budget"

PHASE1_MMES_RESUME_ACTION_SPEC = ActionSpec(
    name=PHASE1_MMES_RESUME_ACTION,
    semantic_surface="phase1_mmes_continuation",
    parameter_names=(
        "state_hash",
        "state_payload_hash",
        "population_size",
        "budget_fes",
        "optimizer_parameters",
        "optimizer_parameter_hash",
        "run_block_reference_version",
        "restart_policy",
        "runtime_termination_policy",
        "acceptance_rule",
        "trigger_scope",
    ),
)

_HASH_LENGTH = 64
_EXECUTION_STATUSES = frozenset({"issued", "running", "completed", "abstained", "failed"})


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hash(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _array_payload(value: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _state_type_name(state: object) -> str:
    state_type = type(state)
    return f"{state_type.__module__}.{state_type.__qualname__}"


def canonical_mmes_state_hash(state: object) -> str:
    """Hash every continuation field except nondeterministic wall-clock counters."""

    validate = getattr(state, "validate", None)
    if not callable(validate):
        raise TypeError("state must implement validate()")
    validate()
    try:
        payload: dict[str, object] = {
            "schema": MMES_STATE_HASH_SCHEMA,
            "state_type": _state_type_name(state),
            "arrays": {
                name: _array_payload(getattr(state, name))
                for name in ("x", "mean", "p", "q", "t", "v", "y", "best_so_far_x")
            },
            "scalars": {
                name: getattr(state, name)
                for name in (
                    "w",
                    "sigma",
                    "sigma_bak",
                    "n_individuals",
                    "n_parents",
                    "n_mirror_sampling",
                    "n_generations",
                    "n_restart",
                    "best_so_far_y",
                    "n_function_evaluations",
                    "termination_signal",
                    "counter_early_stopping",
                    "base_early_stopping",
                    "printed_evaluations",
                    "pending_distribution_update",
                )
            },
            "initial_mean": (
                None
                if getattr(state, "initial_mean") is None
                else _array_payload(getattr(state, "initial_mean"))
            ),
            "pending_y_bak": (
                None
                if getattr(state, "pending_y_bak") is None
                else _array_payload(getattr(state, "pending_y_bak"))
            ),
            "list_generations": list(getattr(state, "list_generations")),
            "list_fitness": list(getattr(state, "list_fitness")),
            "list_initial_mean": [
                _array_payload(value) for value in getattr(state, "list_initial_mean")
            ],
            "fitness": list(getattr(state, "fitness")),
            "recent_best": [list(value) for value in getattr(state, "recent_best")],
            "rng_initialization_state": _jsonable(getattr(state, "rng_initialization_state")),
            "rng_optimization_state": _jsonable(getattr(state, "rng_optimization_state")),
        }
    except AttributeError as error:
        raise TypeError("state is missing an MMES continuation field") from error
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mmes_vector_hash(values: Sequence[float] | np.ndarray) -> str:
    """Hash an exact finite one-dimensional MMES vector."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("MMES vector must be a non-empty finite one-dimensional vector")
    return _canonical_sha256(
        {
            "dtype": vector.dtype.str,
            "shape": list(vector.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest(),
        }
    )


def mmes_resume_anchor_hash(
    problem_id: str,
    incumbent: Sequence[float] | np.ndarray,
    incumbent_fitness: float,
) -> str:
    """Bind a resume action to the incumbent visible at its target checkpoint."""

    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("problem_id must be a non-empty string")
    fitness = _finite(incumbent_fitness, "incumbent_fitness")
    return _canonical_sha256(
        {
            "action": PHASE1_MMES_RESUME_ACTION,
            "schema": PHASE1_MMES_RESUME_SCHEMA,
            "schema_version": PHASE1_MMES_RESUME_SCHEMA_VERSION,
            "trigger_scope": TRIGGER_SCOPE_PHASE_BOUNDARY,
            "problem_id": problem_id,
            "incumbent_hash": mmes_vector_hash(incumbent),
            "incumbent_fitness": fitness,
        }
    )


def _derived_mmes_parameter_payload(
    *,
    population_size: int,
    n_parents: int,
    c_c: float,
    ms: int,
    c_s: float,
    gamma: float,
) -> dict[str, object]:
    w_base = np.log((population_size + 1.0) / 2.0)
    log_ranks = np.log(np.arange(n_parents) + 1.0)
    weights = (w_base - log_ranks) / (n_parents * w_base - np.sum(log_ranks))
    mu_eff = 1.0 / np.sum(np.square(weights))
    return {
        "z_1": float(np.sqrt(1.0 - gamma)),
        "z_2": float(np.sqrt(gamma / ms)),
        "p_1": 1.0 - c_c,
        "p_2": float(np.sqrt(c_c * (2.0 - c_c))),
        "w_1": 1.0 - c_s,
        "w_2": float(np.sqrt(c_s * (2.0 - c_s))),
        "recombination_weights": [float(value) for value in weights],
        "mu_eff": float(mu_eff),
    }


@dataclass(frozen=True)
class FrozenMmesParameters:
    """Complete static parameter surface for the pinned vendor ``run_block``."""

    ndim_problem: int
    population_size: int
    n_parents: int
    m: int
    c_c: float
    ms: int
    c_s: float
    a_z: float
    distance: int
    c_a: float
    gamma: float
    schema: str = MMES_PARAMETER_SCHEMA
    reference_version: str = MMES_RUN_BLOCK_REFERENCE_VERSION
    optimizer_type: str = MMES_VENDOR_TYPE
    max_function_evaluations: str = _BLOCK_LIMIT
    max_runtime: str = _POSITIVE_INFINITY
    fitness_threshold: str = _NEGATIVE_INFINITY
    is_restart: bool = False
    early_stopping_evaluations: str = _POSITIVE_INFINITY
    early_stopping_threshold: float = 0.0
    saving_fitness: int = 0
    verbose: int = 0

    def __post_init__(self) -> None:
        for name in (
            "ndim_problem",
            "population_size",
            "n_parents",
            "m",
            "ms",
            "distance",
        ):
            _integer(getattr(self, name), name, minimum=1)
        for name in ("saving_fitness", "verbose"):
            _integer(getattr(self, name), name)
        if self.n_parents > self.population_size:
            raise ValueError("n_parents must not exceed population_size")
        for name in ("c_c", "c_s", "a_z", "c_a", "gamma"):
            value = _finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.early_stopping_threshold != 0.0:
            raise ValueError("early_stopping_threshold must be zero")
        if self.schema != MMES_PARAMETER_SCHEMA:
            raise ValueError("unsupported MMES parameter schema")
        if self.reference_version != MMES_RUN_BLOCK_REFERENCE_VERSION:
            raise ValueError("unsupported MMES run_block reference version")
        if self.optimizer_type != MMES_VENDOR_TYPE:
            raise ValueError("unsupported MMES optimizer type")
        if self.max_function_evaluations != _BLOCK_LIMIT:
            raise ValueError("MMES max_function_evaluations must equal the block limit")
        if self.max_runtime != _POSITIVE_INFINITY:
            raise ValueError("MMES max_runtime must be disabled")
        if self.fitness_threshold != _NEGATIVE_INFINITY:
            raise ValueError("MMES fitness_threshold must be disabled")
        if self.is_restart is not False:
            raise ValueError("MMES restart must be disabled")
        if self.early_stopping_evaluations != _POSITIVE_INFINITY:
            raise ValueError("MMES early stopping must be disabled")
        if self.saving_fitness != 0 or self.verbose != 0:
            raise ValueError("MMES action requires quiet non-saving execution")

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "reference_version": self.reference_version,
            "optimizer_type": self.optimizer_type,
            "ndim_problem": self.ndim_problem,
            "population_size": self.population_size,
            "n_parents": self.n_parents,
            "m": self.m,
            "c_c": self.c_c,
            "ms": self.ms,
            "c_s": self.c_s,
            "a_z": self.a_z,
            "distance": self.distance,
            "c_a": self.c_a,
            "gamma": self.gamma,
            "max_function_evaluations": self.max_function_evaluations,
            "max_runtime": self.max_runtime,
            "fitness_threshold": self.fitness_threshold,
            "is_restart": self.is_restart,
            "early_stopping_evaluations": self.early_stopping_evaluations,
            "early_stopping_threshold": self.early_stopping_threshold,
            "saving_fitness": self.saving_fitness,
            "verbose": self.verbose,
            "derived": _derived_mmes_parameter_payload(
                population_size=self.population_size,
                n_parents=self.n_parents,
                c_c=self.c_c,
                ms=self.ms,
                c_s=self.c_s,
                gamma=self.gamma,
            ),
        }

    @property
    def parameter_hash(self) -> str:
        return _canonical_sha256(self.audit_payload())


def canonical_mmes_parameters(state: object) -> FrozenMmesParameters:
    """Derive the canonical static MMES parameters paired with a frozen state."""

    validate = getattr(state, "validate", None)
    if not callable(validate):
        raise TypeError("state must implement validate()")
    validate()
    ndim = int(np.asarray(getattr(state, "best_so_far_x")).size)
    population = int(getattr(state, "n_individuals"))
    n_parents = int(getattr(state, "n_parents"))
    m = int(np.asarray(getattr(state, "q")).shape[0])
    c_c = float(0.4 / np.sqrt(ndim))
    c_a = float(3.8 / ndim)
    if not 0.0 < c_a <= 1.0:
        raise ValueError("canonical MMES requires ndim_problem >= 4")
    return FrozenMmesParameters(
        ndim_problem=ndim,
        population_size=population,
        n_parents=n_parents,
        m=m,
        c_c=c_c,
        ms=4,
        c_s=0.3,
        a_z=0.05,
        distance=int(np.ceil(1.0 / c_c)),
        c_a=c_a,
        gamma=float(1.0 - np.power(1.0 - c_a, m)),
    )


@dataclass(frozen=True)
class FrozenMmesState:
    """Immutable in-process snapshot of one complete vendor ``MMESState``."""

    schema: str
    state_type: str
    canonical_hash: str
    payload_hash: str
    _payload: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != MMES_STATE_HASH_SCHEMA:
            raise ValueError("unsupported MMES state hash schema")
        if not isinstance(self.state_type, str) or not self.state_type.strip():
            raise ValueError("state_type must be a non-empty string")
        if self.state_type != MMES_VENDOR_STATE_TYPE:
            raise ValueError("unsupported MMES state type")
        _validate_hash(self.canonical_hash, "canonical_hash")
        _validate_hash(self.payload_hash, "payload_hash")
        if not isinstance(self._payload, bytes) or not self._payload:
            raise ValueError("MMES state payload must be non-empty bytes")
        if hashlib.sha256(self._payload).hexdigest() != self.payload_hash:
            raise ValueError("MMES state payload hash mismatch")
        state = self._decode()
        if _state_type_name(state) != self.state_type:
            raise ValueError("MMES state payload type mismatch")
        if canonical_mmes_state_hash(state) != self.canonical_hash:
            raise ValueError("MMES state canonical hash mismatch")

    @classmethod
    def capture(cls, state: object) -> FrozenMmesState:
        clone = getattr(state, "clone", None)
        if not callable(clone):
            raise TypeError("state must implement clone()")
        frozen_state = clone()
        canonical_hash = canonical_mmes_state_hash(frozen_state)
        payload = pickle.dumps(frozen_state, protocol=pickle.HIGHEST_PROTOCOL)
        return cls(
            schema=MMES_STATE_HASH_SCHEMA,
            state_type=_state_type_name(frozen_state),
            canonical_hash=canonical_hash,
            payload_hash=hashlib.sha256(payload).hexdigest(),
            _payload=payload,
        )

    def _decode(self) -> object:
        return pickle.loads(self._payload)  # noqa: S301 - trusted in-process action snapshot

    def clone_state(self) -> object:
        """Return a verified deep clone for exactly one executor invocation."""

        if hashlib.sha256(self._payload).hexdigest() != self.payload_hash:
            raise RuntimeError("frozen MMES payload changed")
        state = self._decode()
        if canonical_mmes_state_hash(state) != self.canonical_hash:
            raise RuntimeError("frozen MMES state changed")
        clone = getattr(state, "clone", None)
        if not callable(clone):
            raise TypeError("frozen state must implement clone()")
        return clone()


@dataclass(frozen=True)
class Phase1MmesResumeAction:
    """One complete MMES continuation consumed at the Phase1 boundary."""

    problem_id: str
    run_seed: int
    checkpoint_fe: int
    dispatch_checkpoint_hash: str
    anchor_hash: str
    state_snapshot: FrozenMmesState
    state_dimension: int
    population_size: int
    budget_fes: int
    optimizer_parameters: FrozenMmesParameters
    optimizer_parameter_hash: str
    seed_namespace: str
    acceptance_fitness: float
    issued_sweep: int
    target_sweep: int
    ttl_sweeps: int
    expires_sweep: int
    restart_policy: str = NO_RESTART_POLICY
    runtime_termination_policy: str = RUNTIME_TERMINATION_DISABLED
    acceptance_rule: str = STRICT_IMPROVEMENT_ACCEPTANCE
    run_block_reference_version: str = MMES_RUN_BLOCK_REFERENCE_VERSION
    trigger_scope: str = TRIGGER_SCOPE_PHASE_BOUNDARY
    schema: str = PHASE1_MMES_RESUME_SCHEMA
    schema_version: int = PHASE1_MMES_RESUME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")
        if not isinstance(self.seed_namespace, str) or not self.seed_namespace.strip():
            raise ValueError("seed_namespace must be a non-empty string")
        if self.schema != PHASE1_MMES_RESUME_SCHEMA:
            raise ValueError("unsupported MMES resume action schema")
        if self.schema_version != PHASE1_MMES_RESUME_SCHEMA_VERSION:
            raise ValueError("unsupported MMES resume action schema version")
        if self.run_block_reference_version != MMES_RUN_BLOCK_REFERENCE_VERSION:
            raise ValueError("unsupported MMES run_block reference version")
        _integer(self.run_seed, "run_seed")
        _integer(self.checkpoint_fe, "checkpoint_fe")
        _validate_hash(self.dispatch_checkpoint_hash, "dispatch_checkpoint_hash")
        _validate_hash(self.anchor_hash, "anchor_hash")
        _validate_hash(self.optimizer_parameter_hash, "optimizer_parameter_hash")
        if not isinstance(self.state_snapshot, FrozenMmesState):
            raise TypeError("state_snapshot must be FrozenMmesState")
        if not isinstance(self.optimizer_parameters, FrozenMmesParameters):
            raise TypeError("optimizer_parameters must be FrozenMmesParameters")
        if self.trigger_scope != TRIGGER_SCOPE_PHASE_BOUNDARY:
            raise ValueError("MMES resume actions require phase_boundary trigger_scope")

        state = self.state_snapshot.clone_state()
        dimension = _integer(self.state_dimension, "state_dimension", minimum=1)
        population = _integer(self.population_size, "population_size", minimum=2)
        state_dimension = int(np.asarray(getattr(state, "best_so_far_x")).size)
        if dimension != state_dimension:
            raise ValueError("state_dimension does not match the frozen MMES state")
        if population != int(getattr(state, "n_individuals")):
            raise ValueError("population_size does not match the frozen MMES state")
        expected_parameters = canonical_mmes_parameters(state)
        if self.optimizer_parameters != expected_parameters:
            raise ValueError("optimizer_parameters do not match the frozen MMES state")
        if self.optimizer_parameter_hash != expected_parameters.parameter_hash:
            raise ValueError("optimizer_parameter_hash does not match canonical MMES parameters")
        state_best_x = np.asarray(getattr(state, "best_so_far_x"), dtype=np.float64)
        state_best_y = _finite(getattr(state, "best_so_far_y"), "state best_so_far_y")
        expected_anchor = mmes_resume_anchor_hash(
            self.problem_id,
            state_best_x,
            state_best_y,
        )
        if self.anchor_hash != expected_anchor:
            raise ValueError("anchor_hash does not match the frozen MMES best")

        budget = _integer(self.budget_fes, "budget_fes", minimum=population)
        if budget % population:
            raise ValueError("budget_fes must be a whole number of MMES populations")
        acceptance = _finite(self.acceptance_fitness, "acceptance_fitness")
        if acceptance < 0.0:
            raise ValueError("acceptance_fitness must be non-negative")
        if acceptance != state_best_y:
            raise ValueError("acceptance_fitness must equal the frozen MMES best")
        object.__setattr__(self, "acceptance_fitness", acceptance)

        issued = _integer(self.issued_sweep, "issued_sweep")
        target = _integer(self.target_sweep, "target_sweep")
        ttl = _integer(self.ttl_sweeps, "ttl_sweeps")
        expires = _integer(self.expires_sweep, "expires_sweep")
        if ttl != 0:
            raise ValueError("phase-boundary MMES actions must have ttl_sweeps=0")
        if target != issued:
            raise ValueError("phase-boundary MMES actions must target the issued sweep")
        if expires != target:
            raise ValueError("expires_sweep must equal target_sweep")
        if self.restart_policy != NO_RESTART_POLICY:
            raise ValueError("MMES resume actions prohibit restart")
        if self.runtime_termination_policy != RUNTIME_TERMINATION_DISABLED:
            raise ValueError("MMES resume actions require disabled runtime termination")
        if self.acceptance_rule != STRICT_IMPROVEMENT_ACCEPTANCE:
            raise ValueError("unsupported MMES resume acceptance_rule")

    @property
    def state_hash(self) -> str:
        return self.state_snapshot.canonical_hash

    def audit_payload(self) -> dict[str, object]:
        return {
            "action": PHASE1_MMES_RESUME_ACTION,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "run_seed": self.run_seed,
            "checkpoint_fe": self.checkpoint_fe,
            "dispatch_checkpoint_hash": self.dispatch_checkpoint_hash,
            "anchor_hash": self.anchor_hash,
            "state_hash_schema": self.state_snapshot.schema,
            "state_type": self.state_snapshot.state_type,
            "state_hash": self.state_hash,
            "state_payload_hash": self.state_snapshot.payload_hash,
            "state_dimension": self.state_dimension,
            "population_size": self.population_size,
            "budget_fes": self.budget_fes,
            "optimizer_parameters": self.optimizer_parameters.audit_payload(),
            "optimizer_parameter_hash": self.optimizer_parameter_hash,
            "run_block_reference_version": self.run_block_reference_version,
            "seed_namespace": self.seed_namespace,
            "acceptance_fitness": self.acceptance_fitness,
            "issued_sweep": self.issued_sweep,
            "target_sweep": self.target_sweep,
            "ttl_sweeps": self.ttl_sweeps,
            "expires_sweep": self.expires_sweep,
            "restart_policy": self.restart_policy,
            "runtime_termination_policy": self.runtime_termination_policy,
            "acceptance_rule": self.acceptance_rule,
            "trigger_scope": self.trigger_scope,
        }

    @property
    def action_hash(self) -> str:
        return _canonical_sha256(self.audit_payload())


def _optimizer_limit_payload(value: object, *, expected: int | None = None) -> object:
    numeric = float(value)
    if expected is not None and numeric == expected:
        return _BLOCK_LIMIT
    if math.isinf(numeric):
        return _POSITIVE_INFINITY if numeric > 0.0 else _NEGATIVE_INFINITY
    return numeric


def _optimizer_parameter_receipt(
    optimizer: object,
    *,
    expected_max_fes: int,
) -> dict[str, object]:
    try:
        population = int(getattr(optimizer, "n_individuals"))
        n_parents = int(getattr(optimizer, "n_parents"))
        c_c = float(getattr(optimizer, "c_c"))
        ms = int(getattr(optimizer, "ms"))
        c_s = float(getattr(optimizer, "c_s"))
        gamma = float(getattr(optimizer, "gamma"))
        derived = {
            "z_1": float(getattr(optimizer, "_z_1")),
            "z_2": float(getattr(optimizer, "_z_2")),
            "p_1": float(getattr(optimizer, "_p_1")),
            "p_2": float(getattr(optimizer, "_p_2")),
            "w_1": float(getattr(optimizer, "_w_1")),
            "w_2": float(getattr(optimizer, "_w_2")),
            "recombination_weights": [
                float(value) for value in np.asarray(getattr(optimizer, "_w"))
            ],
            "mu_eff": float(getattr(optimizer, "_mu_eff")),
        }
        return {
            "schema": MMES_PARAMETER_SCHEMA,
            "reference_version": MMES_RUN_BLOCK_REFERENCE_VERSION,
            "optimizer_type": _state_type_name(optimizer),
            "ndim_problem": int(getattr(optimizer, "ndim_problem")),
            "population_size": population,
            "n_parents": n_parents,
            "m": int(getattr(optimizer, "m")),
            "c_c": c_c,
            "ms": ms,
            "c_s": c_s,
            "a_z": float(getattr(optimizer, "a_z")),
            "distance": int(getattr(optimizer, "distance")),
            "c_a": float(getattr(optimizer, "c_a")),
            "gamma": gamma,
            "max_function_evaluations": _optimizer_limit_payload(
                getattr(optimizer, "max_function_evaluations"),
                expected=expected_max_fes,
            ),
            "max_runtime": _optimizer_limit_payload(getattr(optimizer, "max_runtime")),
            "fitness_threshold": _optimizer_limit_payload(getattr(optimizer, "fitness_threshold")),
            "is_restart": getattr(optimizer, "is_restart"),
            "early_stopping_evaluations": _optimizer_limit_payload(
                getattr(optimizer, "early_stopping_evaluations")
            ),
            "early_stopping_threshold": float(getattr(optimizer, "early_stopping_threshold")),
            "saving_fitness": int(getattr(optimizer, "saving_fitness")),
            "verbose": int(getattr(optimizer, "verbose")),
            "derived": derived,
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("mmes_factory returned an incompatible optimizer") from error


def build_canonical_mmes_optimizer(
    factory: Callable[[dict[str, object], dict[str, object]], object],
    *,
    objective: Callable[..., object],
    action: Phase1MmesResumeAction,
    state: object,
) -> object:
    """Build and verify the only vendor MMES instance authorized by the action."""

    if not callable(factory):
        raise TypeError("mmes_factory must be callable")
    if not callable(objective):
        raise TypeError("objective must be callable")
    parameters = action.optimizer_parameters
    initial_fes = int(getattr(state, "n_function_evaluations"))
    max_fes = initial_fes + action.budget_fes
    optimizer = factory(
        {
            "fitness_function": objective,
            "ndim_problem": parameters.ndim_problem,
            "lower_boundary": None,
            "upper_boundary": None,
        },
        {
            "max_function_evaluations": max_fes,
            "max_runtime": np.inf,
            "fitness_threshold": -np.inf,
            "mean": np.copy(getattr(state, "mean")),
            "sigma": float(getattr(state, "sigma")),
            "n_individuals": parameters.population_size,
            "n_parents": parameters.n_parents,
            "m": parameters.m,
            "c_c": parameters.c_c,
            "ms": parameters.ms,
            "c_s": parameters.c_s,
            "a_z": parameters.a_z,
            "distance": parameters.distance,
            "c_a": parameters.c_a,
            "gamma": parameters.gamma,
            "seed_rng": action.run_seed,
            "is_restart": False,
            "early_stopping_evaluations": np.inf,
            "early_stopping_threshold": 0.0,
            "saving_fitness": 0,
            "verbose": 0,
        },
    )
    if not callable(getattr(optimizer, "run_block", None)):
        raise TypeError("canonical MMES optimizer must implement run_block")
    receipt = _optimizer_parameter_receipt(optimizer, expected_max_fes=max_fes)
    if receipt != parameters.audit_payload():
        raise ValueError("MMES optimizer parameter receipt mismatch")
    if _canonical_sha256(receipt) != action.optimizer_parameter_hash:
        raise ValueError("MMES optimizer parameter hash mismatch")
    return optimizer


class _CountingObjective:
    def __init__(self, objective: Callable[..., object]) -> None:
        self.objective = objective
        self.evaluations = 0

    def __call__(self, values: object, *args: object, **kwargs: object) -> object:
        array = np.asarray(values)
        if array.ndim == 1:
            count = 1
        elif array.ndim == 2:
            count = int(array.shape[0])
        else:
            raise ValueError("MMES objective input must be one- or two-dimensional")
        if count <= 0:
            raise ValueError("MMES objective input must contain at least one candidate")
        self.evaluations += count
        return self.objective(values, *args, **kwargs)


@dataclass
class MmesResumeExecutionState:
    """Mutable one-shot lifecycle record; optimizer state stays in snapshots."""

    action_hash: str
    initial_state_hash: str
    status: str = "issued"
    consumed_fes: int = 0
    unused_fes: int = 0
    started_fe: int | None = None
    completed_fe: int | None = None
    final_state_hash: str | None = None
    invalidation_reason: str = ""

    @classmethod
    def for_action(cls, action: Phase1MmesResumeAction) -> MmesResumeExecutionState:
        return cls(action_hash=action.action_hash, initial_state_hash=action.state_hash)

    def validate_for(self, action: Phase1MmesResumeAction) -> None:
        _validate_hash(self.action_hash, "action_hash")
        _validate_hash(self.initial_state_hash, "initial_state_hash")
        if self.action_hash != action.action_hash or self.initial_state_hash != action.state_hash:
            raise ValueError("MMES resume lifecycle does not match the action")
        if self.status not in _EXECUTION_STATUSES:
            raise ValueError("unsupported MMES resume lifecycle status")
        _integer(self.consumed_fes, "consumed_fes")
        _integer(self.unused_fes, "unused_fes")
        if not isinstance(self.invalidation_reason, str):
            raise ValueError("invalidation_reason must be a string")
        if self.started_fe is not None:
            _integer(self.started_fe, "started_fe")
        if self.completed_fe is not None:
            _integer(self.completed_fe, "completed_fe")
        if self.final_state_hash is not None:
            _validate_hash(self.final_state_hash, "final_state_hash")
        if self.status == "issued":
            if (
                self.consumed_fes
                or self.unused_fes
                or self.started_fe is not None
                or self.completed_fe is not None
                or self.final_state_hash is not None
                or self.invalidation_reason
            ):
                raise ValueError("issued MMES resume lifecycle contains outcome data")
        elif self.status == "running":
            if (
                self.started_fe is None
                or self.consumed_fes
                or self.unused_fes
                or self.completed_fe is not None
                or self.final_state_hash is not None
                or self.invalidation_reason
            ):
                raise ValueError("running MMES resume lifecycle is inconsistent")
        elif self.status == "completed":
            if (
                self.started_fe is None
                or self.completed_fe is None
                or self.final_state_hash is None
                or self.invalidation_reason
                or self.consumed_fes != action.budget_fes
                or self.unused_fes != 0
                or self.completed_fe - self.started_fe != self.consumed_fes
            ):
                raise ValueError("completed MMES resume lifecycle is inconsistent")
        elif self.status == "abstained":
            if (
                self.consumed_fes
                or self.unused_fes
                or self.started_fe is not None
                or self.completed_fe is not None
                or self.final_state_hash is not None
                or not self.invalidation_reason
            ):
                raise ValueError("abstained MMES resume lifecycle is inconsistent")
        elif (
            self.started_fe is None
            or self.completed_fe is None
            or self.final_state_hash is not None
            or not self.invalidation_reason
            or self.completed_fe - self.started_fe != self.consumed_fes
            or self.unused_fes != max(action.budget_fes - self.consumed_fes, 0)
        ):
            raise ValueError("failed MMES resume lifecycle is inconsistent")

    def start(self, action: Phase1MmesResumeAction, *, current_fe: int) -> None:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("only an issued MMES resume action can start")
        self.started_fe = _integer(current_fe, "current_fe")
        self.status = "running"

    def complete(
        self,
        action: Phase1MmesResumeAction,
        *,
        consumed_fes: int,
        unused_fes: int,
        final_state_hash: str,
    ) -> None:
        self.validate_for(action)
        if self.status != "running" or self.started_fe is None:
            raise ValueError("only a running MMES resume action can complete")
        consumed = _integer(consumed_fes, "consumed_fes", minimum=1)
        unused = _integer(unused_fes, "unused_fes")
        final_hash = _validate_hash(final_state_hash, "final_state_hash")
        if consumed != action.budget_fes or unused != 0:
            raise ValueError("MMES resume must consume its exact frozen FE budget")
        self.consumed_fes = consumed
        self.unused_fes = unused
        self.completed_fe = self.started_fe + consumed
        self.final_state_hash = final_hash
        self.status = "completed"
        self.validate_for(action)

    def fail(
        self,
        action: Phase1MmesResumeAction,
        *,
        consumed_fes: int,
        reason: str,
    ) -> None:
        self.validate_for(action)
        if self.status != "running" or self.started_fe is None:
            raise ValueError("only a running MMES resume action can fail")
        consumed = _integer(consumed_fes, "consumed_fes")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason must be a non-empty string")
        self.consumed_fes = consumed
        self.unused_fes = max(action.budget_fes - consumed, 0)
        self.completed_fe = self.started_fe + consumed
        self.invalidation_reason = reason
        self.status = "failed"
        self.validate_for(action)

    def abstain(self, action: Phase1MmesResumeAction, *, reason: str) -> None:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("only an issued MMES resume action can abstain")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abstain reason must be a non-empty string")
        self.invalidation_reason = reason
        self.status = "abstained"
        self.validate_for(action)

    def audit_payload(self, action: Phase1MmesResumeAction) -> dict[str, object]:
        self.validate_for(action)
        return {
            "action": PHASE1_MMES_RESUME_ACTION,
            "schema": action.schema,
            "schema_version": action.schema_version,
            "run_block_reference_version": action.run_block_reference_version,
            "action_hash": self.action_hash,
            "initial_state_hash": self.initial_state_hash,
            "status": self.status,
            "consumed_fes": self.consumed_fes,
            "unused_fes": self.unused_fes,
            "started_fe": self.started_fe,
            "completed_fe": self.completed_fe,
            "final_state_hash": self.final_state_hash,
            "invalidation_reason": self.invalidation_reason,
        }

    def state_hash(self, action: Phase1MmesResumeAction) -> str:
        return _canonical_sha256(self.audit_payload(action))


@dataclass(frozen=True)
class MmesResumeExecutionContext:
    """Phase-boundary checkpoint and dependencies for canonical MMES construction."""

    current_fe: int
    current_sweep: int
    dispatch_checkpoint_hash: str
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    required_seed_namespace: str
    objective: Callable[..., object]
    mmes_factory: Callable[[dict[str, object], dict[str, object]], object]

    def __post_init__(self) -> None:
        _integer(self.current_fe, "current_fe")
        _integer(self.current_sweep, "current_sweep")
        _validate_hash(self.dispatch_checkpoint_hash, "dispatch_checkpoint_hash")
        if (
            not isinstance(self.required_seed_namespace, str)
            or not self.required_seed_namespace.strip()
        ):
            raise ValueError("required_seed_namespace must be a non-empty string")
        if not callable(self.objective):
            raise TypeError("objective must be callable")
        if not callable(self.mmes_factory):
            raise TypeError("mmes_factory must be callable")
        incumbent = np.asarray(self.incumbent, dtype=np.float64)
        if incumbent.ndim != 1 or incumbent.size == 0 or not np.all(np.isfinite(incumbent)):
            raise ValueError("incumbent must be a non-empty finite one-dimensional vector")
        object.__setattr__(self, "incumbent", tuple(float(value) for value in incumbent))
        object.__setattr__(
            self,
            "incumbent_fitness",
            _finite(self.incumbent_fitness, "incumbent_fitness"),
        )


@dataclass(frozen=True)
class MmesResumeRejectionResult:
    """Auditable no-execution result for abstention or a repeated invocation."""

    disposition: str
    reason: str
    action_hash: str
    lifecycle: MmesResumeExecutionState
    lifecycle_hash: str | None
    counterfactual_applied: bool = False
    resume_native: bool = True

    def __post_init__(self) -> None:
        if self.disposition not in {"abstained", "rejected"}:
            raise ValueError("unsupported MMES rejection disposition")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("MMES rejection reason must be non-empty")
        _validate_hash(self.action_hash, "action_hash")
        if self.lifecycle_hash is not None:
            _validate_hash(self.lifecycle_hash, "lifecycle_hash")


@dataclass(frozen=True)
class MmesResumeExecutionResult:
    """Auditable outcome of one exact ``clone -> run_block`` invocation."""

    incumbent: tuple[float, ...]
    incumbent_fitness: float
    candidate: tuple[float, ...]
    candidate_fitness: float
    accepted: bool
    requested_fes: int
    consumed_fes: int
    unused_fes: int
    termination_reason: str
    initial_state_hash: str
    final_state_snapshot: FrozenMmesState
    final_state_hash: str
    action_hash: str
    lifecycle: MmesResumeExecutionState
    lifecycle_hash: str
    candidate_hash: str
    post_incumbent_hash: str
    resume_native: bool = True


def _preflight_mmes_resume_action(
    action: Phase1MmesResumeAction,
    context: MmesResumeExecutionContext,
) -> None:
    if len(context.incumbent) != action.state_dimension:
        raise ValueError("incumbent dimension does not match the MMES action")
    if context.current_sweep > action.expires_sweep:
        raise ValueError("MMES resume action phase boundary expired")
    if context.current_sweep != action.target_sweep:
        raise ValueError("current_sweep does not match target_sweep")
    if context.current_fe != action.checkpoint_fe:
        raise ValueError("current_fe does not match checkpoint_fe")
    if context.dispatch_checkpoint_hash != action.dispatch_checkpoint_hash:
        raise ValueError("dispatch_checkpoint_hash mismatch")
    if context.required_seed_namespace != action.seed_namespace:
        raise ValueError("MMES resume seed namespace mismatch")
    if (
        mmes_resume_anchor_hash(
            action.problem_id,
            context.incumbent,
            context.incumbent_fitness,
        )
        != action.anchor_hash
    ):
        raise ValueError("MMES resume anchor mismatch")
    if context.incumbent_fitness != action.acceptance_fitness:
        raise ValueError("incumbent fitness does not match acceptance_fitness")


def _execution_reason(stage: str, error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    return f"{stage}:{type(error).__name__}:{detail}"


def _rejection_result(
    action: Phase1MmesResumeAction,
    execution_state: MmesResumeExecutionState,
    *,
    disposition: str,
    reason: str,
    lifecycle_hash: str | None,
) -> MmesResumeRejectionResult:
    return MmesResumeRejectionResult(
        disposition=disposition,
        reason=reason,
        action_hash=action.action_hash,
        lifecycle=execution_state,
        lifecycle_hash=lifecycle_hash,
    )


def execute_phase1_mmes_resume_action(
    action: Phase1MmesResumeAction,
    context: MmesResumeExecutionContext,
    execution_state: MmesResumeExecutionState,
) -> MmesResumeExecutionResult | MmesResumeRejectionResult:
    """Consume one exact canonical MMES block at the Phase1 boundary."""

    if not isinstance(action, Phase1MmesResumeAction):
        raise TypeError("action must be Phase1MmesResumeAction")
    if not isinstance(context, MmesResumeExecutionContext):
        raise TypeError("context must be MmesResumeExecutionContext")
    if not isinstance(execution_state, MmesResumeExecutionState):
        raise TypeError("execution_state must be MmesResumeExecutionState")

    try:
        execution_state.validate_for(action)
    except ValueError as error:
        return _rejection_result(
            action,
            execution_state,
            disposition="rejected",
            reason=_execution_reason("lifecycle_mismatch", error),
            lifecycle_hash=None,
        )
    if execution_state.status != "issued":
        return _rejection_result(
            action,
            execution_state,
            disposition="rejected",
            reason=f"execution_state_already_{execution_state.status}",
            lifecycle_hash=execution_state.state_hash(action),
        )

    try:
        _preflight_mmes_resume_action(action, context)
        state = action.state_snapshot.clone_state()
        initial_hash = canonical_mmes_state_hash(state)
        if initial_hash != action.state_hash:
            raise RuntimeError("frozen MMES state hash drifted")
        initial_fes = int(getattr(state, "n_function_evaluations"))
        initial_restart_count = int(getattr(state, "n_restart"))
        initial_best = _finite(getattr(state, "best_so_far_y"), "MMES best_before")
        fingerprint = getattr(state, "fingerprint", None)
        if not callable(fingerprint):
            raise TypeError("frozen state must implement fingerprint()")
        initial_vendor_fingerprint = _validate_hash(
            fingerprint(),
            "initial MMES vendor fingerprint",
        )
    except Exception as error:
        reason = _execution_reason("preflight", error)
        execution_state.abstain(action, reason=reason)
        return _rejection_result(
            action,
            execution_state,
            disposition="abstained",
            reason=reason,
            lifecycle_hash=execution_state.state_hash(action),
        )

    counted_objective = _CountingObjective(context.objective)
    try:
        optimizer = build_canonical_mmes_optimizer(
            context.mmes_factory,
            objective=counted_objective,
            action=action,
            state=state,
        )
    except Exception as error:
        if counted_objective.evaluations:
            execution_state.start(action, current_fe=context.current_fe)
            execution_state.fail(
                action,
                consumed_fes=counted_objective.evaluations,
                reason=_execution_reason("optimizer_build", error),
            )
            raise
        reason = _execution_reason("optimizer_preflight", error)
        execution_state.abstain(action, reason=reason)
        return _rejection_result(
            action,
            execution_state,
            disposition="abstained",
            reason=reason,
            lifecycle_hash=execution_state.state_hash(action),
        )

    execution_state.start(action, current_fe=context.current_fe)
    try:
        block = optimizer.run_block(state, action.budget_fes)
        try:
            requested = block.requested_fes
            consumed = block.actual_fes
            unused = block.unused_fes
            best_before = _finite(block.best_before, "MMES block best_before")
            best_after = _finite(block.best_after, "MMES block best_after")
            termination_reason = block.termination_reason
            fingerprint_before = block.state_fingerprint_before
            fingerprint_after = block.state_fingerprint_after
            final_state = block.state
        except AttributeError as error:
            raise TypeError("run_block must return an MMESBlockResult-like object") from error

        _integer(requested, "MMES block requested_fes")
        _integer(consumed, "MMES block actual_fes")
        _integer(unused, "MMES block unused_fes")
        _validate_hash(fingerprint_before, "MMES block state_fingerprint_before")
        _validate_hash(fingerprint_after, "MMES block state_fingerprint_after")
        if fingerprint_before != initial_vendor_fingerprint:
            raise RuntimeError("MMES block before fingerprint does not match frozen state")
        if requested != action.budget_fes:
            raise RuntimeError("MMES block requested_fes drifted")
        if consumed != action.budget_fes or unused != 0:
            raise RuntimeError("MMES block did not consume its exact frozen budget")
        if counted_objective.evaluations != consumed:
            raise RuntimeError("MMES objective FE receipt does not match actual_fes")
        if termination_reason != "block_complete":
            raise RuntimeError("MMES block returned an invalid termination_reason")
        canonical_mmes_state_hash(final_state)
        final_fingerprint = getattr(final_state, "fingerprint", None)
        if not callable(final_fingerprint):
            raise TypeError("final state must implement fingerprint()")
        if fingerprint_after != final_fingerprint():
            raise RuntimeError("MMES block after fingerprint does not match final state")
        if int(getattr(final_state, "n_individuals")) != action.population_size:
            raise RuntimeError("MMES population changed during resume")
        if np.asarray(getattr(final_state, "best_so_far_x")).size != action.state_dimension:
            raise RuntimeError("MMES state dimension changed during resume")
        if int(getattr(final_state, "n_function_evaluations")) - initial_fes != consumed:
            raise RuntimeError("MMES state FE delta does not match actual_fes")
        if int(getattr(final_state, "n_restart")) != initial_restart_count:
            raise RuntimeError("MMES resume attempted a restart")
        if best_before != initial_best or best_after != float(
            getattr(final_state, "best_so_far_y")
        ):
            raise RuntimeError("MMES block best values do not match its states")
        if best_after > best_before:
            raise RuntimeError("MMES best-so-far fitness regressed")

        final_snapshot = FrozenMmesState.capture(final_state)
        candidate_array = np.asarray(
            getattr(final_state, "best_so_far_x"),
            dtype=np.float64,
        )
        candidate = tuple(float(value) for value in candidate_array)
        accepted = best_after < action.acceptance_fitness
        incumbent = candidate if accepted else context.incumbent
        incumbent_fitness = best_after if accepted else action.acceptance_fitness
        candidate_hash = mmes_vector_hash(candidate)
        post_incumbent_hash = mmes_vector_hash(incumbent)
    except Exception as error:
        execution_state.fail(
            action,
            consumed_fes=counted_objective.evaluations,
            reason=_execution_reason("execution", error),
        )
        raise

    execution_state.complete(
        action,
        consumed_fes=consumed,
        unused_fes=unused,
        final_state_hash=final_snapshot.canonical_hash,
    )
    return MmesResumeExecutionResult(
        incumbent=incumbent,
        incumbent_fitness=incumbent_fitness,
        candidate=candidate,
        candidate_fitness=best_after,
        accepted=accepted,
        requested_fes=requested,
        consumed_fes=consumed,
        unused_fes=unused,
        termination_reason=termination_reason,
        initial_state_hash=initial_hash,
        final_state_snapshot=final_snapshot,
        final_state_hash=final_snapshot.canonical_hash,
        action_hash=action.action_hash,
        lifecycle=execution_state,
        lifecycle_hash=execution_state.state_hash(action),
        candidate_hash=candidate_hash,
        post_incumbent_hash=post_incumbent_hash,
    )


__all__ = [
    "MMES_PARAMETER_SCHEMA",
    "MMES_RUN_BLOCK_REFERENCE_VERSION",
    "MMES_STATE_HASH_SCHEMA",
    "MMES_VENDOR_TYPE",
    "MMES_VENDOR_STATE_TYPE",
    "NO_RESTART_POLICY",
    "PHASE1_MMES_RESUME_ACTION",
    "PHASE1_MMES_RESUME_ACTION_SPEC",
    "PHASE1_MMES_RESUME_SCHEMA",
    "PHASE1_MMES_RESUME_SCHEMA_VERSION",
    "RUNTIME_TERMINATION_DISABLED",
    "STRICT_IMPROVEMENT_ACCEPTANCE",
    "TRIGGER_SCOPE_PHASE_BOUNDARY",
    "FrozenMmesState",
    "FrozenMmesParameters",
    "MmesResumeExecutionContext",
    "MmesResumeRejectionResult",
    "MmesResumeExecutionResult",
    "MmesResumeExecutionState",
    "Phase1MmesResumeAction",
    "build_canonical_mmes_optimizer",
    "canonical_mmes_parameters",
    "canonical_mmes_state_hash",
    "execute_phase1_mmes_resume_action",
    "mmes_resume_anchor_hash",
    "mmes_vector_hash",
]
