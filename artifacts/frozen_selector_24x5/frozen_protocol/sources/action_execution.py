"""Shared numerical machinery for the four ARAC actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import time

import numpy as np
from pypop7.optimizers.es.cmaes import CMAES
from pypop7.optimizers.core.optimizer import Optimizer

from arac.runtime.contracts import ActionContext, ActionResult
from arac.runtime.optimizers import OptimizationRun, PypopOptimizerPort


BLOCK_POPULATION_SIZE = 12
FULL_SPACE_POPULATION_SIZE = 24
DEFAULT_SIGMA = 0.5


def derived_seed(context: ActionContext, namespace: str, index: int = 0) -> int:
    payload = (
        f"{context.checkpoint.checkpoint_hash}:{context.action_seed}:{namespace}:{index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & (
        (1 << 63) - 1
    )


@dataclass
class _PersistentBlockSession:
    context: ActionContext
    block: tuple[int, ...]
    block_index: int
    budget_fes: int

    def __post_init__(self) -> None:
        dimensions = np.asarray(self.block, dtype=int)
        mean = self.context.ledger.best_x[dimensions]
        lower = self.context.problem.lower_array[dimensions]
        upper = self.context.problem.upper_array[dimensions]

        def objective(candidate: np.ndarray) -> float | np.ndarray:
            values = np.asarray(candidate, dtype=float)
            if values.ndim == 1:
                full = self.context.ledger.best_x
                full[dimensions] = values
                return float(self.context.ledger.evaluate(full))
            full = np.repeat(self.context.ledger.best_x[None, :], len(values), axis=0)
            full[:, dimensions] = values
            return self.context.ledger.evaluate(full)

        self.optimizer = CMAES(
            {
                "fitness_function": objective,
                "ndim_problem": len(self.block),
                "lower_boundary": lower,
                "upper_boundary": upper,
            },
            {
                "max_function_evaluations": self.budget_fes,
                "mean": mean,
                "sigma": DEFAULT_SIGMA,
                "seed_rng": derived_seed(self.context, "persistent-block", self.block_index),
                "n_individuals": BLOCK_POPULATION_SIZE,
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
        self.consumed_fes = 0

    @property
    def complete(self) -> bool:
        return self.consumed_fes == self.budget_fes

    def advance(self) -> None:
        if self.complete:
            raise RuntimeError("completed block session cannot advance")
        transform = self.eigenvectors @ np.diag(self.eigenvalues)
        for index in range(BLOCK_POPULATION_SIZE):
            noise = self.optimizer.rng_optimization.standard_normal(
                (self.optimizer.ndim_problem,)
            )
            self.steps[index] = transform @ noise
            self.x[index] = self.mean + self.optimizer.sigma * self.steps[index]
        started = time.time()
        values = np.asarray(
            self.optimizer.fitness_function(self.x),
            dtype=float,
        ).reshape(-1)
        self.optimizer.time_function_evaluations += time.time() - started
        if values.shape != (BLOCK_POPULATION_SIZE,) or not np.all(np.isfinite(values)):
            raise RuntimeError("block objective returned an invalid population")
        self.fitness[:] = values
        for candidate, value in zip(self.x, values, strict=True):
            numeric = float(value)
            self.optimizer.n_function_evaluations += 1
            if numeric < self.optimizer.best_so_far_y:
                self.optimizer.best_so_far_x = candidate.copy()
                self.optimizer.best_so_far_y = numeric
        consumed = BLOCK_POPULATION_SIZE
        self.consumed_fes += consumed
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
        self.optimizer._n_generations += 1


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
) -> int:
    """Interleave upstream block-CMA states and return the aligned FE consumed."""

    blocks = context.checkpoint.blocks
    order = tuple(range(len(blocks))) if block_order is None else block_order
    if sorted(order) != list(range(len(blocks))):
        raise ValueError("block_order must be a complete block permutation")
    sweep_cost = len(blocks) * BLOCK_POPULATION_SIZE
    aligned = min(int(requested_fes), context.ledger.remaining)
    aligned -= aligned % sweep_cost
    if sweep_limit is not None:
        if sweep_limit <= 0:
            raise ValueError("sweep_limit must be positive")
        aligned = min(aligned, sweep_limit * sweep_cost)
    if aligned == 0:
        return 0
    budgets = _allocate_block_budgets(
        blocks,
        aligned,
        equal_generations=sweep_limit is not None,
    )
    sessions = tuple(
        _PersistentBlockSession(context, block, index, budgets[index])
        for index, block in enumerate(blocks)
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
) -> int:
    """Polish each evidence block to its aligned allocation before moving on."""

    blocks = context.checkpoint.blocks
    order = tuple(range(len(blocks))) if block_order is None else block_order
    if sorted(order) != list(range(len(blocks))):
        raise ValueError("block_order must be a complete block permutation")
    aligned = min(int(requested_fes), context.ledger.remaining)
    aligned -= aligned % (len(blocks) * BLOCK_POPULATION_SIZE)
    if aligned == 0:
        return 0
    budgets = _allocate_block_budgets(
        blocks,
        aligned,
        equal_generations=False,
    )
    count_before = context.ledger.count
    for index in order:
        session = _PersistentBlockSession(
            context,
            blocks[index],
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
    "run_full_space",
    "run_persistent_blocks",
    "run_sequential_blocks",
    "terminal_result",
]
