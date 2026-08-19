"""Shared numerical machinery for the four ARAC actions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import hashlib
import importlib.metadata
import math
import time

import numpy as np
from pypop7.optimizers.es.cmaes import CMAES
from pypop7.optimizers.core.optimizer import Optimizer

from arac.runtime.contracts import ActionContext, ActionResult
from arac.runtime.optimizers import OptimizationRun, PypopOptimizerPort


BLOCK_POPULATION_SIZE = 12
FULL_SPACE_POPULATION_SIZE = 24
DEFAULT_SIGMA = 0.5
EARLY_STOPPING_EVALUATIONS = 1000
STATE_STALE_WINDOW = 3
STATE_MATERIAL_LOG_GAIN = math.log1p(0.01)
HISTORICAL_MATERIAL_LOG_GAIN = math.log(1.01)
STATE_RESCUE_FRACTION = 0.20
STATE_RESCUE_MIN_FES = 100_000
STATE_RESCUE_VISIT_FES = 35_000
STATE_RESCUE_ATTEMPTS = 8
STATE_RESCUE_DIRECTION_FRACTION = 1e-4
STATE_RESCUE_MAX_BOUND_FRACTION = 0.16
ZERO_RELATION_COVERAGE_FRACTION = 0.10
ZERO_RELATION_RESCUE_SIGMA_DIVISORS = (1.0,) * 4 + (2.0,) * 4 + (4.0,) * 4 + (8.0,) * 4
ZERO_RELATION_PERSISTENT_ATTEMPTS = 4
ZERO_RELATION_RETIRE_AFTER = 2


def derived_seed(context: ActionContext, namespace: str, index: int = 0) -> int:
    payload = (
        f"arac-action-seed-v1:{context.action_name}:{context.action_seed}:{namespace}:{index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


@dataclass
class _PersistentBlockSession:
    context: ActionContext
    block: tuple[int, ...]
    block_index: int
    budget_fes: int
    population_size: int = BLOCK_POPULATION_SIZE
    seed_namespace: str = "persistent-block"
    initial_sigma: float = DEFAULT_SIGMA
    seed_factory: Callable[[int], int] | None = None
    stage_index: int = 1
    clip_offspring: bool = True

    def __post_init__(self) -> None:
        dimensions = np.asarray(self.block, dtype=int)
        if self.population_size < 2:
            raise ValueError("population_size must be at least two")
        if not math.isfinite(self.initial_sigma) or self.initial_sigma <= 0.0:
            raise ValueError("initial_sigma must be finite and positive")
        self._dimensions = dimensions
        self.lower = self.context.problem.lower_array[dimensions]
        self.upper = self.context.problem.upper_array[dimensions]
        self.restart_count = 0
        self.consumed_fes = 0
        self._initialize_optimizer()

    def _initialize_optimizer(self) -> None:
        self._visit_base = self.context.ledger.best_x
        mean = self._visit_base[self._dimensions]

        def objective(candidate: np.ndarray) -> float | np.ndarray:
            values = np.asarray(candidate, dtype=float)
            if values.ndim == 1:
                full = self._visit_base.copy()
                full[self._dimensions] = values
                return float(self.context.ledger.evaluate(full))
            full = np.repeat(self._visit_base[None, :], len(values), axis=0)
            full[:, self._dimensions] = values
            return self.context.ledger.evaluate(full)

        seed_rng = (
            self.seed_factory(self.stage_index)
            if self.seed_factory is not None
            else derived_seed(
                self.context,
                f"{self.seed_namespace}-restart-{self.restart_count}",
                self.block_index,
            )
        )
        self.optimizer = CMAES(
            {
                "fitness_function": objective,
                "ndim_problem": len(self._dimensions),
                "lower_boundary": self.lower,
                "upper_boundary": self.upper,
            },
            {
                "max_function_evaluations": self.budget_fes,
                "mean": mean,
                "sigma": self.initial_sigma,
                "seed_rng": seed_rng,
                "n_individuals": self.population_size,
                "is_restart": False,
                "verbose": 0,
            },
        )
        Optimizer.optimize(self.optimizer)
        (
            self.x,
            self.mean,
            self.path_sigma,
            self.path_covariance,
            self.covariance,
            self.eigenvectors,
            self.eigenvalues,
            self.fitness,
            self.steps,
        ) = self.optimizer.initialize()
        self._prepared = False
        self._pending_values = []

    @property
    def complete(self) -> bool:
        return self.consumed_fes == self.budget_fes

    @property
    def early_stopped(self) -> bool:
        return self.optimizer._counter_early_stopping >= EARLY_STOPPING_EVALUATIONS

    def begin_visit(self) -> None:
        """Restore distribution state around one frozen incumbent slice."""

        self._visit_base = self.context.ledger.best_x
        self.mean = self._visit_base[self._dimensions]
        self.optimizer.best_so_far_x = None
        self.optimizer.best_so_far_y = math.inf
        self.optimizer._base_early_stopping = math.inf
        self.optimizer._counter_early_stopping = 0

    def restart(self, *, stage_index: int | None = None) -> None:
        """Discard a stale covariance state while preserving consumed FE."""

        self.restart_count += 1
        if stage_index is not None:
            self.stage_index = int(stage_index)
        self._initialize_optimizer()

    def prepare_generation(self) -> None:
        """Draw one offspring population without evaluating it (resumable split)."""

        if self._prepared:
            raise RuntimeError("a prepared generation is still pending")
        noise = self.optimizer.rng_optimization.standard_normal(
            (self.population_size, self.optimizer.ndim_problem)
        )
        transform = np.diag(self.eigenvalues) @ self.eigenvectors.T
        self.steps[:] = noise @ transform
        # Overflowing proposals are immediately repaired at the public bounds;
        # keep that deliberate numerical boundary visible without emitting a
        # warning for every repaired offspring.
        with np.errstate(over="ignore", invalid="ignore"):
            self.x[:] = self.mean + self.optimizer.sigma * self.steps
        escaped = (~np.isfinite(self.x)) | (self.x < self.lower) | (self.x > self.upper)
        if np.any(~np.isfinite(self.x)):
            self.x[:] = np.nan_to_num(
                self.x,
                nan=self.mean,
                posinf=self.upper,
                neginf=self.lower,
            )
        if self.clip_offspring and np.any(escaped):
            np.clip(self.x, self.lower, self.upper, out=self.x)
            repaired_steps = (self.x - self.mean) / self.optimizer.sigma
            self.steps[escaped] = repaired_steps[escaped]
        self._pending_values = []
        self._prepared = True

    def evaluate_pending_batch(self) -> None:
        """Evaluate the whole prepared population as one batch call.

        The vendor objectives are batch-shape sensitive at the last bit, so
        episodes must never split a generation's evaluation: a full-population
        batch reproduces the legacy trajectory bitwise, while row-wise or
        partial evaluation does not.
        """

        if not self._prepared:
            raise RuntimeError("no prepared generation is pending")
        started = time.time()
        values = np.asarray(
            self.optimizer.fitness_function(self.x),
            dtype=float,
        ).reshape(-1)
        self.optimizer.time_function_evaluations += time.time() - started
        if values.shape != (self.population_size,) or not np.all(np.isfinite(values)):
            raise RuntimeError("block objective returned an invalid population")
        self._pending_values = [float(value) for value in values]

    def finalize_generation(self, *, adapt_on_early_stop: bool = True) -> None:
        if not self._prepared or len(self._pending_values) != self.population_size:
            raise RuntimeError("prepared generation is incomplete")
        values = np.asarray(self._pending_values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("block objective returned an invalid population")
        self.fitness[:] = values
        for candidate, value in zip(self.x, values, strict=True):
            numeric = float(value)
            self.optimizer.n_function_evaluations += 1
            if numeric < self.optimizer.best_so_far_y:
                self.optimizer.best_so_far_x = candidate.copy()
                self.optimizer.best_so_far_y = numeric
            if (
                self.optimizer._base_early_stopping - numeric
                <= self.optimizer.early_stopping_threshold
            ):
                self.optimizer._counter_early_stopping += 1
            else:
                self.optimizer._counter_early_stopping = 0
                self.optimizer._base_early_stopping = numeric
        self.consumed_fes += self.population_size
        self._prepared = False
        self._pending_values = []
        if self.early_stopped and not adapt_on_early_stop:
            return
        self.optimizer._n_generations += 1
        (
            self.mean,
            self.path_sigma,
            self.path_covariance,
            self.covariance,
            self.eigenvectors,
            self.eigenvalues,
        ) = self.optimizer.update_distribution(
            self.x,
            self.path_sigma,
            self.path_covariance,
            self.covariance,
            self.eigenvectors,
            self.eigenvalues,
            self.fitness,
            self.steps,
        )

    def advance(self, *, adapt_on_early_stop: bool = True) -> None:
        if self.complete:
            raise RuntimeError("completed block session cannot advance")
        if self.consumed_fes + self.population_size > self.budget_fes:
            raise RuntimeError("block session cannot consume a partial population")
        if self.population_size > self.context.ledger.remaining:
            raise RuntimeError("block population exceeds the remaining terminal budget")
        self.prepare_generation()
        self.evaluate_pending_batch()
        self.finalize_generation(adapt_on_early_stop=adapt_on_early_stop)

    def state_dict(self) -> dict[str, object]:
        from arac.runtime.optimizers import _jsonable

        return _jsonable(
            {
                "restart_count": self.restart_count,
                "consumed_fes": self.consumed_fes,
                "stage_index": self.stage_index,
                "visit_base": self._visit_base,
                "prepared": self._prepared,
                "pending_values": self._pending_values,
                "arrays": {
                    # x/steps are scratch until a generation is prepared;
                    # fitness is scratch until finalize writes it.  Their
                    # pre-initialization contents are uninitialized memory,
                    # so serialize them only when meaningful and skip them
                    # on restore otherwise.
                    "x": self.x if self._prepared else None,
                    "mean": self.mean,
                    "path_sigma": self.path_sigma,
                    "path_covariance": self.path_covariance,
                    "covariance": self.covariance,
                    "eigenvectors": self.eigenvectors,
                    "eigenvalues": self.eigenvalues,
                    "fitness": None,
                    "steps": self.steps if self._prepared else None,
                },
                "optimizer": {
                    "sigma": self.optimizer.sigma,
                    "n_function_evaluations": self.optimizer.n_function_evaluations,
                    "best_so_far_x": self.optimizer.best_so_far_x,
                    "best_so_far_y": self.optimizer.best_so_far_y,
                    "base_early_stopping": self.optimizer._base_early_stopping,
                    "counter_early_stopping": self.optimizer._counter_early_stopping,
                    "n_generations": self.optimizer._n_generations,
                    "rng": self.optimizer.rng.bit_generator.state,
                    "rng_initialization": self.optimizer.rng_initialization.bit_generator.state,
                    "rng_optimization": self.optimizer.rng_optimization.bit_generator.state,
                },
            }
        )

    def restore_state_dict(self, payload: dict[str, object]) -> None:
        arrays = payload.get("arrays")
        optimizer_state = payload.get("optimizer")
        if not isinstance(arrays, dict) or not isinstance(optimizer_state, dict):
            raise ValueError("block session payload is missing arrays or optimizer state")
        self.restart_count = int(payload["restart_count"])
        self.stage_index = int(payload["stage_index"])
        self._initialize_optimizer()
        self.consumed_fes = int(payload["consumed_fes"])
        self._visit_base = np.asarray(payload["visit_base"], dtype=float)
        if self._visit_base.shape != (self.context.problem.dimension,):
            raise ValueError("restored visit base has the wrong dimension")
        self._prepared = bool(payload["prepared"])
        self._pending_values = [float(value) for value in payload["pending_values"]]
        if not self._prepared and self._pending_values:
            raise ValueError("pending offspring values without a prepared generation")
        if len(self._pending_values) > self.population_size:
            raise ValueError("pending offspring values exceed the population")
        for name in (
            "x",
            "mean",
            "path_sigma",
            "path_covariance",
            "covariance",
            "eigenvectors",
            "eigenvalues",
            "fitness",
            "steps",
        ):
            raw = arrays[name]
            if raw is None:
                continue
            restored = np.asarray(raw, dtype=float)
            current = getattr(self, name)
            if restored.shape != current.shape or not np.all(np.isfinite(restored)):
                raise ValueError(f"invalid restored block session array: {name}")
            setattr(self, name, restored)
        self.optimizer.sigma = float(optimizer_state["sigma"])
        self.optimizer.n_function_evaluations = int(optimizer_state["n_function_evaluations"])
        best_x = optimizer_state["best_so_far_x"]
        self.optimizer.best_so_far_x = (
            None if best_x is None else np.asarray(best_x, dtype=float)
        )
        self.optimizer.best_so_far_y = float(optimizer_state["best_so_far_y"])
        self.optimizer._base_early_stopping = float(optimizer_state["base_early_stopping"])
        self.optimizer._counter_early_stopping = int(optimizer_state["counter_early_stopping"])
        self.optimizer._n_generations = int(optimizer_state["n_generations"])
        self.optimizer.rng.bit_generator.state = optimizer_state["rng"]
        self.optimizer.rng_initialization.bit_generator.state = optimizer_state["rng_initialization"]
        self.optimizer.rng_optimization.bit_generator.state = optimizer_state["rng_optimization"]


def _block_population_size(dimension: int) -> int:
    if dimension <= 0:
        raise ValueError("block dimension must be positive")
    return 4 + 3 * math.ceil(math.log(dimension))


def _aligned_visit_budget(requested_fes: int, remaining_fes: int, population_size: int) -> int:
    available = min(int(requested_fes), int(remaining_fes))
    return available - available % population_size


def _log_improvement(before: float, after: float) -> float:
    if after >= before:
        return 0.0
    floor = np.finfo(float).tiny
    return max(0.0, math.log(max(before, floor) / max(after, floor)))


def _historical_log_improvement(before: float, after: float) -> float:
    epsilon = 1e-300
    return math.log((before + epsilon) / (after + epsilon)) if after < before else 0.0


def _run_block_visit(session: _PersistentBlockSession, budget_fes: int) -> int:
    aligned = _aligned_visit_budget(
        budget_fes,
        session.context.ledger.remaining,
        session.population_size,
    )
    if aligned == 0:
        return 0
    session.begin_visit()
    consumed_before = session.consumed_fes
    visit_target = consumed_before + aligned
    while session.consumed_fes - consumed_before < aligned and not session.early_stopped:
        completes_visit = session.consumed_fes + session.population_size >= visit_target
        session.advance(adapt_on_early_stop=completes_visit)
    return session.consumed_fes - consumed_before


def _run_stateful_block_visits(
    context: ActionContext,
    *,
    requested_fes: int,
    block_order: tuple[int, ...] | None = None,
    seed_factory: Callable[[int], int] | None = None,
    clip_offspring: bool = True,
    precheck_incumbent: bool = False,
    strict_material_gain: bool = False,
    event_trace: list[dict[str, object]] | None = None,
) -> tuple[int, int, int, tuple[_PersistentBlockSession, ...]]:
    """Run incumbent-recentered visits while retaining useful block distributions."""

    blocks = context.checkpoint.blocks
    order = tuple(range(len(blocks))) if block_order is None else block_order
    if sorted(order) != list(range(len(blocks))):
        raise ValueError("block_order must be a complete block permutation")
    stage_slots = {index: position + 1 for position, index in enumerate(order)}
    action_budget = min(int(requested_fes), context.ledger.remaining)
    populations = tuple(_block_population_size(len(block)) for block in blocks)
    sessions = tuple(
        _PersistentBlockSession(
            context,
            block,
            index,
            action_budget,
            population_size=populations[index],
            seed_namespace="stateful-block",
            seed_factory=seed_factory,
            stage_index=stage_slots[index],
            clip_offspring=clip_offspring,
        )
        for index, block in enumerate(blocks)
    )
    stale_streaks = [0] * len(blocks)
    count_before = context.ledger.count
    target_count = count_before + action_budget
    visit_count = 0
    restart_count = 0
    outer_iter = 0
    cold_start = [True] * len(blocks)
    while context.ledger.count < target_count:
        sweep_remaining = target_count - context.ledger.count
        requested_per_block = math.ceil(sweep_remaining / len(blocks))
        sweep_before = context.ledger.count
        for index in order:
            start_fes = context.ledger.count
            remaining = target_count - context.ledger.count
            population = populations[index]
            if remaining < population or (precheck_incumbent and remaining <= population):
                break
            precheck_error: float | None = None
            if precheck_incumbent:
                precheck_error = context.ledger.evaluate_incumbent(refresh_error=True)
                remaining = target_count - context.ledger.count
            visit_budget = _aligned_visit_budget(
                max(requested_per_block, population),
                remaining,
                population,
            )
            before = context.ledger.best_error if precheck_error is None else precheck_error
            route = "cold_start" if cold_start[index] else "restore"
            consumed = _run_block_visit(sessions[index], visit_budget)
            if consumed == 0:
                continue
            visit_count += 1
            gain = (
                _historical_log_improvement(before, context.ledger.best_error)
                if strict_material_gain
                else _log_improvement(before, context.ledger.best_error)
            )
            material = (
                gain > HISTORICAL_MATERIAL_LOG_GAIN
                if strict_material_gain
                else gain >= STATE_MATERIAL_LOG_GAIN
            )
            stale_streaks[index] = (
                0
                if material
                else stale_streaks[index] + 1
            )
            reset = stale_streaks[index] >= STATE_STALE_WINDOW
            if event_trace is not None:
                event_trace.append(
                    {
                        "outer_iter": outer_iter,
                        "group_index": index,
                        "stage_index": outer_iter * len(blocks) + stage_slots[index],
                        "route": route,
                        "start_fes": start_fes,
                        "precheck_fes": int(precheck_incumbent),
                        "group_fes": consumed,
                        "end_fes": context.ledger.count,
                        "log_gain": gain,
                        "material_improvement": material,
                        "stagnation_streak": stale_streaks[index],
                        "reset": reset,
                    }
                )
            cold_start[index] = False
            if reset:
                sessions[index].restart(
                    stage_index=(outer_iter + 1) * len(blocks) + stage_slots[index]
                    if seed_factory is not None
                    else None
                )
                stale_streaks[index] = 0
                restart_count += 1
                cold_start[index] = True
        if context.ledger.count == sweep_before:
            break
        outer_iter += 1
    return (
        context.ledger.count - count_before,
        visit_count,
        restart_count,
        sessions,
    )


def run_stateful_block_visits(
    context: ActionContext,
    *,
    requested_fes: int,
    block_order: tuple[int, ...] | None = None,
    seed_factory: Callable[[int], int] | None = None,
    clip_offspring: bool = True,
    precheck_incumbent: bool = False,
    strict_material_gain: bool = False,
    event_trace: list[dict[str, object]] | None = None,
) -> tuple[int, int, int]:
    """Run incumbent-recentered visits while retaining useful block distributions."""

    consumed, visit_count, restart_count, _ = _run_stateful_block_visits(
        context,
        requested_fes=requested_fes,
        block_order=block_order,
        seed_factory=seed_factory,
        clip_offspring=clip_offspring,
        precheck_incumbent=precheck_incumbent,
        strict_material_gain=strict_material_gain,
        event_trace=event_trace,
    )
    return consumed, visit_count, restart_count


def run_stateful_block_visits_with_sessions(
    context: ActionContext,
    *,
    requested_fes: int,
    block_order: tuple[int, ...] | None = None,
    seed_factory: Callable[[int], int] | None = None,
    clip_offspring: bool = True,
    precheck_incumbent: bool = False,
    strict_material_gain: bool = False,
    event_trace: list[dict[str, object]] | None = None,
) -> tuple[int, int, int, tuple[_PersistentBlockSession, ...]]:
    """Run stateful visits and retain their distributions for zero-relation rescue."""

    return _run_stateful_block_visits(
        context,
        requested_fes=requested_fes,
        block_order=block_order,
        seed_factory=seed_factory,
        clip_offspring=clip_offspring,
        precheck_incumbent=precheck_incumbent,
        strict_material_gain=strict_material_gain,
        event_trace=event_trace,
    )


def _rank_blocks_by_directional_slope(
    context: ActionContext,
    *,
    cycle_index: int,
    excluded: frozenset[int] = frozenset(),
) -> tuple[tuple[int, ...], int]:
    """Rank unprocessed blocks by identity-blind predicted local decrease."""

    blocks = context.checkpoint.blocks
    active = tuple(index for index in range(len(blocks)) if index not in excluded)
    required = 2 * len(active)
    if not active:
        return (), 0
    if context.ledger.remaining < required:
        return (), 0
    base = context.ledger.best_x
    base_error = context.ledger.best_error
    scores: list[tuple[float, int]] = []
    count_before = context.ledger.count
    for index in active:
        block = blocks[index]
        dimensions = np.asarray(block, dtype=int)
        span = context.problem.upper_array[dimensions] - context.problem.lower_array[dimensions]
        step = STATE_RESCUE_DIRECTION_FRACTION * float(np.min(span))
        rng = np.random.default_rng(
            derived_seed(context, f"state-rescue-probe-{cycle_index}", index)
        )
        direction = rng.standard_normal(len(dimensions))
        direction /= np.linalg.norm(direction)
        candidates = np.repeat(base[None, :], 2, axis=0)
        candidates[0, dimensions] += step * direction
        candidates[1, dimensions] -= step * direction
        candidates[:, dimensions] = np.clip(
            candidates[:, dimensions],
            context.problem.lower_array[dimensions],
            context.problem.upper_array[dimensions],
        )
        values = np.asarray(context.ledger.evaluate(candidates), dtype=float)
        gradient = (float(values[0]) - float(values[1])) / (2.0 * step)
        curvature = (float(values[0]) + float(values[1]) - 2.0 * base_error) / step**2
        predicted_decrease = (
            gradient**2 / (2.0 * curvature)
            if curvature > np.finfo(float).eps
            else max(0.0, base_error - float(np.min(values)))
        )
        scores.append((predicted_decrease, index))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return tuple(index for _, index in scores), context.ledger.count - count_before


def run_stalled_block_rescue(
    context: ActionContext,
    *,
    requested_fes: int,
) -> tuple[int, int, int]:
    """Spend a bounded tail budget on blocks with the largest residual slope."""

    blocks = context.checkpoint.blocks
    target_count = context.ledger.count + min(int(requested_fes), context.ledger.remaining)
    count_before = context.ledger.count
    probe_fes = 0
    visit_count = 0
    cycle_index = 0
    processed: set[int] = set()
    while context.ledger.count < target_count:
        remaining = target_count - context.ledger.count
        if remaining < 2 * len(blocks):
            break
        ranking, consumed = _rank_blocks_by_directional_slope(
            context,
            cycle_index=cycle_index,
            excluded=frozenset(processed),
        )
        probe_fes += consumed
        if not ranking:
            break
        index = ranking[0]
        block = blocks[index]
        population = _block_population_size(len(block))
        bound_span = float(
            np.min(
                context.problem.upper_array[np.asarray(block, dtype=int)]
                - context.problem.lower_array[np.asarray(block, dtype=int)]
            )
        )
        maximum_sigma = STATE_RESCUE_MAX_BOUND_FRACTION * bound_span
        error_before = context.ledger.best_error
        for attempt in range(STATE_RESCUE_ATTEMPTS):
            remaining = target_count - context.ledger.count
            visit_budget = _aligned_visit_budget(
                min(STATE_RESCUE_VISIT_FES, remaining),
                context.ledger.remaining,
                population,
            )
            if visit_budget == 0:
                break
            session = _PersistentBlockSession(
                context,
                block,
                index,
                visit_budget,
                population_size=population,
                seed_namespace=f"state-rescue-{cycle_index}-{attempt}",
                initial_sigma=maximum_sigma if attempt < 4 else maximum_sigma / 2.0,
            )
            _run_block_visit(session, visit_budget)
            visit_count += 1
        if _log_improvement(error_before, context.ledger.best_error) < STATE_MATERIAL_LOG_GAIN:
            processed.add(index)
        cycle_index += 1
    return context.ledger.count - count_before, probe_fes, visit_count


def run_zero_relation_hybrid_rescue(
    context: ActionContext,
    *,
    requested_fes: int,
    sessions: tuple[_PersistentBlockSession, ...],
) -> tuple[int, int, int, int, int]:
    """Combine shallow coverage with productive multiscale residual rescue."""

    blocks = context.checkpoint.blocks
    if len(sessions) != len(blocks) or any(
        session.context is not context or session.block_index != index
        for index, session in enumerate(sessions)
    ):
        raise ValueError("sessions must match the active zero-relation context")
    target_count = context.ledger.count + min(int(requested_fes), context.ledger.remaining)
    count_before = context.ledger.count
    coverage_cap = int((target_count - count_before) * ZERO_RELATION_COVERAGE_FRACTION)
    coverage_fes, _ = run_cold_start_block_sweeps(
        context,
        requested_fes=coverage_cap,
        sweep_limit=1,
        namespace="zero-relation-coverage",
    )
    probe_fes = 0
    cold_visit_count = 0
    persistent_visit_count = 0
    cycle_index = 0
    stale_strikes = [0] * len(blocks)
    retired: set[int] = set()
    while context.ledger.count < target_count:
        remaining = target_count - context.ledger.count
        active_count = len(blocks) - len(retired)
        if active_count == 0 or remaining < 2 * active_count:
            break
        ranking, consumed = _rank_blocks_by_directional_slope(
            context,
            cycle_index=cycle_index,
            excluded=frozenset(retired),
        )
        probe_fes += consumed
        if not ranking:
            break
        index = ranking[0]
        block = blocks[index]
        population = _block_population_size(len(block))
        dimensions = np.asarray(block, dtype=int)
        bound_span = float(
            np.min(
                context.problem.upper_array[dimensions] - context.problem.lower_array[dimensions]
            )
        )
        maximum_sigma = STATE_RESCUE_MAX_BOUND_FRACTION * bound_span
        error_before = context.ledger.best_error
        remaining = target_count - context.ledger.count
        persistent_reserve = min(STATE_RESCUE_VISIT_FES, remaining // 5)
        cold_target = target_count - persistent_reserve
        for attempt, divisor in enumerate(ZERO_RELATION_RESCUE_SIGMA_DIVISORS):
            remaining = cold_target - context.ledger.count
            visit_budget = _aligned_visit_budget(
                min(STATE_RESCUE_VISIT_FES, remaining),
                context.ledger.remaining,
                population,
            )
            if visit_budget == 0:
                break
            session = _PersistentBlockSession(
                context,
                block,
                index,
                visit_budget,
                population_size=population,
                seed_namespace=f"zero-relation-rescue-{cycle_index}-{attempt}",
                initial_sigma=maximum_sigma / divisor,
            )
            _run_block_visit(session, visit_budget)
            cold_visit_count += 1

        cold_gain = _log_improvement(error_before, context.ledger.best_error)
        if cold_gain < STATE_MATERIAL_LOG_GAIN:
            for _ in range(ZERO_RELATION_PERSISTENT_ATTEMPTS):
                remaining = target_count - context.ledger.count
                visit_budget = _aligned_visit_budget(
                    min(STATE_RESCUE_VISIT_FES, remaining),
                    context.ledger.remaining,
                    sessions[index].population_size,
                )
                if visit_budget == 0:
                    break
                visit_before = context.ledger.best_error
                consumed = _run_block_visit(sessions[index], visit_budget)
                if consumed == 0:
                    break
                persistent_visit_count += 1
                if (
                    _log_improvement(visit_before, context.ledger.best_error)
                    < STATE_MATERIAL_LOG_GAIN
                ):
                    break

        gain = _log_improvement(error_before, context.ledger.best_error)
        stale_strikes[index] = 0 if gain >= STATE_MATERIAL_LOG_GAIN else stale_strikes[index] + 1
        if stale_strikes[index] >= ZERO_RELATION_RETIRE_AFTER:
            retired.add(index)
        cycle_index += 1
    return (
        context.ledger.count - count_before,
        probe_fes,
        coverage_fes,
        cold_visit_count,
        persistent_visit_count,
    )


def run_cold_start_block_sweeps(
    context: ActionContext,
    *,
    requested_fes: int,
    sweep_limit: int | None = None,
    block_order: tuple[int, ...] | None = None,
    namespace: str,
    seed_factory: Callable[[int], int] | None = None,
    start_sweep_index: int = 0,
    event_trace: list[dict[str, object]] | None = None,
) -> tuple[int, tuple[int, ...]]:
    """Run independent block-CMA visits and report each completed sweep cost."""

    blocks = context.checkpoint.blocks
    order = tuple(range(len(blocks))) if block_order is None else block_order
    if sorted(order) != list(range(len(blocks))):
        raise ValueError("block_order must be a complete block permutation")
    if sweep_limit is not None and sweep_limit <= 0:
        raise ValueError("sweep_limit must be positive")
    if start_sweep_index < 0:
        raise ValueError("start_sweep_index must be non-negative")
    action_budget = min(int(requested_fes), context.ledger.remaining)
    target_count = context.ledger.count + action_budget
    population_sizes = tuple(_block_population_size(len(block)) for block in blocks)
    sweep_costs: list[int] = []
    while context.ledger.count < target_count and (
        sweep_limit is None or len(sweep_costs) < sweep_limit
    ):
        requested_per_block = math.ceil((target_count - context.ledger.count) / len(blocks))
        sweep_before = context.ledger.count
        sweep_index = start_sweep_index + len(sweep_costs)
        for index in order:
            visit_start = context.ledger.count
            remaining = target_count - context.ledger.count
            population = population_sizes[index]
            if remaining < population:
                break
            visit_budget = _aligned_visit_budget(
                max(requested_per_block, population),
                remaining,
                population,
            )
            session = _PersistentBlockSession(
                context,
                blocks[index],
                index,
                visit_budget,
                population_size=population,
                seed_namespace=f"{namespace}-sweep-{sweep_index}",
                seed_factory=seed_factory,
                stage_index=sweep_index * len(blocks) + index + 1,
            )
            consumed = _run_block_visit(session, visit_budget)
            if event_trace is not None:
                event_trace.append(
                    {
                        "event": "cold_group_visit",
                        "namespace": namespace,
                        "sweep_index": sweep_index,
                        "group_index": index,
                        "stage_index": sweep_index * len(blocks) + index + 1,
                        "seed": (
                            seed_factory(sweep_index * len(blocks) + index + 1)
                            if seed_factory is not None
                            else derived_seed(
                                context,
                                f"{namespace}-sweep-{sweep_index}-restart-0",
                                index,
                            )
                        ),
                        "start_fes": visit_start,
                        "requested_fes": visit_budget,
                        "actual_fes": consumed,
                        "end_fes": context.ledger.count,
                        "cold_start": True,
                        "state_restored": False,
                    }
                )
        sweep_cost = context.ledger.count - sweep_before
        if sweep_cost == 0:
            break
        sweep_costs.append(sweep_cost)
    return sum(sweep_costs), tuple(sweep_costs)


def _allocate_block_budgets(
    blocks: tuple[tuple[int, ...], ...],
    total_fes: int,
    *,
    equal_generations: bool,
) -> tuple[int, ...]:
    generation_count = int(total_fes) // BLOCK_POPULATION_SIZE
    if generation_count < len(blocks) or generation_count % len(blocks):
        raise ValueError("block budget must contain a whole common sweep")
    if equal_generations:
        per_block = generation_count // len(blocks)
        return (per_block * BLOCK_POPULATION_SIZE,) * len(blocks)

    allocations = np.ones(len(blocks), dtype=int)
    remaining = generation_count - len(blocks)
    dimensions = np.asarray([len(block) for block in blocks], dtype=float)
    weighted = remaining * dimensions / np.sum(dimensions)
    additions = np.floor(weighted).astype(int)
    allocations += additions
    residue = remaining - int(np.sum(additions))
    priority = sorted(
        range(len(blocks)),
        key=lambda index: (-(weighted[index] - additions[index]), -len(blocks[index]), index),
    )
    for index in priority[:residue]:
        allocations[index] += 1
    budgets = tuple(int(value) * BLOCK_POPULATION_SIZE for value in allocations)
    if sum(budgets) != total_fes:
        raise RuntimeError("dimension-weighted block allocation drifted")
    return budgets


def run_persistent_blocks(
    context: ActionContext,
    *,
    requested_fes: int,
    sweep_limit: int | None = None,
    block_order: tuple[int, ...] | None = None,
    blocks: tuple[tuple[int, ...], ...] | None = None,
) -> int:
    """Interleave upstream block-CMA states and return the aligned FE consumed."""

    active_blocks = context.checkpoint.blocks if blocks is None else blocks
    if not active_blocks:
        raise ValueError("blocks must be non-empty")
    order = tuple(range(len(active_blocks))) if block_order is None else block_order
    if sorted(order) != list(range(len(active_blocks))):
        raise ValueError("block_order must be a complete block permutation")
    sweep_cost = len(active_blocks) * BLOCK_POPULATION_SIZE
    aligned = min(int(requested_fes), context.ledger.remaining)
    aligned -= aligned % sweep_cost
    if sweep_limit is not None:
        if sweep_limit <= 0:
            raise ValueError("sweep_limit must be positive")
        aligned = min(aligned, sweep_limit * sweep_cost)
    if aligned == 0:
        return 0
    budgets = _allocate_block_budgets(
        active_blocks,
        aligned,
        equal_generations=sweep_limit is not None,
    )
    sessions = tuple(
        _PersistentBlockSession(context, block, index, budgets[index])
        for index, block in enumerate(active_blocks)
    )
    count_before = context.ledger.count
    while not all(session.complete for session in sessions):
        for index in order:
            session = sessions[index]
            if not session.complete:
                session.advance()
    consumed = context.ledger.count - count_before
    if consumed != aligned:
        raise RuntimeError("persistent block execution drifted from its aligned FE budget")
    return consumed


def run_sequential_blocks(
    context: ActionContext,
    *,
    requested_fes: int,
    block_order: tuple[int, ...] | None = None,
    blocks: tuple[tuple[int, ...], ...] | None = None,
) -> int:
    """Polish each evidence block to its aligned allocation before moving on."""

    active_blocks = context.checkpoint.blocks if blocks is None else blocks
    if not active_blocks:
        raise ValueError("blocks must be non-empty")
    order = tuple(range(len(active_blocks))) if block_order is None else block_order
    if sorted(order) != list(range(len(active_blocks))):
        raise ValueError("block_order must be a complete block permutation")
    aligned = min(int(requested_fes), context.ledger.remaining)
    aligned -= aligned % (len(active_blocks) * BLOCK_POPULATION_SIZE)
    if aligned == 0:
        return 0
    budgets = _allocate_block_budgets(
        active_blocks,
        aligned,
        equal_generations=False,
    )
    count_before = context.ledger.count
    for index in order:
        session = _PersistentBlockSession(
            context,
            active_blocks[index],
            index,
            budgets[index],
        )
        while not session.complete:
            session.advance()
    consumed = context.ledger.count - count_before
    if consumed != aligned:
        raise RuntimeError("sequential block execution drifted from its aligned FE budget")
    return consumed


def run_full_space(
    context: ActionContext,
    *,
    algorithm: str,
    budget_fes: int | None = None,
    namespace: str,
) -> OptimizationRun:
    budget = context.ledger.remaining if budget_fes is None else int(budget_fes)
    return PypopOptimizerPort().run(
        algorithm,
        problem=context.problem,
        ledger=context.ledger,
        initial_mean=context.ledger.best_x,
        sigma=DEFAULT_SIGMA,
        seed=derived_seed(context, namespace),
        budget_fes=budget,
        population_size=FULL_SPACE_POPULATION_SIZE,
        restart=False,
    )


def terminal_result(
    context: ActionContext,
    *,
    route: str,
    optimizer_package: str = "pypop7",
    optimizer_version: str | None = None,
) -> ActionResult:
    if context.ledger.remaining != 0:
        raise RuntimeError("action returned before consuming the terminal FE budget")
    if context.ledger.best_error > context.checkpoint.incumbent_error:
        raise RuntimeError("action degraded the shared strict-best archive")
    version = (
        importlib.metadata.version(optimizer_package)
        if optimizer_version is None
        else optimizer_version
    )
    return ActionResult(
        action_name=context.action_name,
        checkpoint_hash=context.checkpoint.checkpoint_hash,
        action_seed=context.action_seed,
        consumed_fes=context.checkpoint.remaining_fes,
        terminal_fes=context.ledger.count,
        incumbent=tuple(float(value) for value in context.ledger.best_x),
        final_error=float(context.ledger.best_error),
        route=route,
        optimizer_package=optimizer_package,
        optimizer_version=version,
    )


__all__ = [
    "BLOCK_POPULATION_SIZE",
    "DEFAULT_SIGMA",
    "FULL_SPACE_POPULATION_SIZE",
    "derived_seed",
    "run_cold_start_block_sweeps",
    "run_full_space",
    "run_persistent_blocks",
    "run_sequential_blocks",
    "run_stalled_block_rescue",
    "run_stateful_block_visits",
    "run_stateful_block_visits_with_sessions",
    "run_zero_relation_hybrid_rescue",
    "terminal_result",
]
