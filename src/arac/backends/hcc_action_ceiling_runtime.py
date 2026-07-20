"""Dependency-injected HCC runtime adapter for action-ceiling branches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from arac.backends.hcc_action_ceiling import (
    ActionExecutionRequest,
    NativeContinuationState,
    OptimizationResult,
    branch_horizon_errors,
    execute_action_ceiling_arm,
    paired_arm_rows,
    run_native_group_cycle,
    run_native_continuation,
    selector_arm_for_context,
)
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    ACTION_CEILING_PROTOCOL_VERSION,
    RelationActionSet,
)


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CapturedActionCeilingContext:
    context_row: dict[str, str]
    arm_rows: tuple[dict[str, str], ...]
    expected_native_record: tuple[float, ...]
    expected_native_incumbent: tuple[float, ...]
    expected_native_incumbent_hash: str


@dataclass(frozen=True)
class HccActionCeilingRuntime:
    benchmark_factory: Callable[..., object]
    cmaes_factory: Callable[..., object]
    combine: Callable[..., np.ndarray]
    derive_seed: Callable[[int, str, int, int, int], int]
    fun_name: str
    fun_id: int
    output_path: Path
    data_root: Path
    sigma: float
    cmaes_restart: bool
    early_stopping_evaluations: int
    lower: float
    upper: float
    dimension: int

    def _fresh_objective(self) -> object:
        benchmark = self.benchmark_factory(
            str(self.output_path) + "/",
            data_dir=self.data_root,
        )
        objective = benchmark.get_function(self.fun_name, self.fun_id)
        record = getattr(objective, "fitness_record", None)
        if not isinstance(record, list) or record:
            raise RuntimeError("action-ceiling branch evaluator must be fresh")
        return objective

    def _group_optimizer(
        self,
        objective: object,
        evaluate: Callable[[np.ndarray], np.ndarray],
        *,
        initial_means: Mapping[int, Sequence[float]] | None = None,
    ) -> Callable[..., OptimizationResult]:
        pending_means = {
            int(group_index): np.asarray(mean, dtype=float).reshape(-1).copy()
            for group_index, mean in (initial_means or {}).items()
        }

        def optimize(**kwargs: object) -> OptimizationResult:
            background = np.asarray(kwargs["background"], dtype=float)
            dims = tuple(int(value) for value in kwargs["dims"])
            group_index = int(kwargs["group_index"])
            requested_mean = np.asarray(kwargs["mean"], dtype=float).reshape(-1)
            mean = pending_means.pop(
                group_index,
                requested_mean,
            )
            if mean.shape != (len(dims),) or not np.all(np.isfinite(mean)):
                raise ValueError("group optimizer mean must be finite and match group dims")
            sigma = float(kwargs["sigma"])
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("group optimizer sigma must be finite and positive")

            def fitness(x_batch: np.ndarray) -> np.ndarray:
                return evaluate(self.combine(x_batch, background, dims))

            record = getattr(objective, "fitness_record")
            before = len(record)
            result = self.cmaes_factory(
                {
                    "fitness_function": fitness,
                    "ndim_problem": len(dims),
                    "lower_boundary": self.lower * np.ones((len(dims),)),
                    "upper_boundary": self.upper * np.ones((len(dims),)),
                },
                {
                    "max_function_evaluations": int(kwargs["requested_fes"]),
                    "mean": (mean,),
                    "sigma": sigma,
                    "n_individuals": int(kwargs["population_size"]),
                    "is_restart": self.cmaes_restart,
                    "verbose": 0,
                    "early_stopping_evaluations": self.early_stopping_evaluations,
                    "seed_rng": int(kwargs["seed"]),
                },
            ).optimize()
            return OptimizationResult(
                tuple(float(value) for value in result["best_so_far_x"]),
                float(result["best_so_far_y"]),
                len(record) - before,
            )

        return optimize

    def capture(
        self,
        *,
        action_set: RelationActionSet,
        cohort: str,
        problem_id: str,
        seed: int,
        dispatch_fe: int,
        outer_iter: int,
        group_index: int,
        incumbent: Sequence[float],
        incumbent_fitness: float,
        previous_values: Sequence[float],
        current_values: Sequence[float],
        previous_delta: float,
        current_delta: float,
        completed_group_deltas: Sequence[float],
        completed_group_actual_fes: Sequence[int],
        group_dims: Sequence[Sequence[int]],
        overlapping_elements: Sequence[Sequence[int]],
        population_sizes: Sequence[int],
        optimizer_budgets: Sequence[int],
        efficiency_ewma: Sequence[float],
        completed_efficiency_sweeps: int,
        stagnation_streaks: Sequence[int],
        fitness_prefix: Sequence[float],
        topology_hash: str,
        order_hash: str,
    ) -> CapturedActionCeilingContext:
        incumbent_array = np.asarray(incumbent, dtype=float).reshape(-1)
        prefix = tuple(float(value) for value in fitness_prefix)
        if incumbent_array.size != self.dimension or not prefix:
            raise ValueError("action-ceiling capture state is incomplete")
        relation = action_set.relation
        dispatch_checkpoint_hash = _sha256(
            {
                "problem_id": problem_id,
                "seed": int(seed),
                "dispatch_fe": int(dispatch_fe),
                "outer_iter": int(outer_iter),
                "group_index": int(group_index),
                "relation": {
                    "owners": relation.owner_group_indices,
                    "shared": relation.shared_variable_indices,
                },
                "incumbent_hash": _sha256(incumbent_array.tolist()),
                "fitness_prefix_hash": _sha256(prefix),
                "topology_hash": topology_hash,
                "order_hash": order_hash,
                "action_set_hash": action_set.action_set_hash,
                "previous_values": list(previous_values),
                "current_values": list(current_values),
                "previous_delta": float(previous_delta),
                "current_delta": float(current_delta),
                "completed_group_deltas": list(completed_group_deltas),
                "completed_group_actual_fes": list(completed_group_actual_fes),
                "efficiency_ewma": list(efficiency_ewma),
                "completed_efficiency_sweeps": int(completed_efficiency_sweeps),
                "stagnation_streaks": list(stagnation_streaks),
            }
        )
        context_id = (
            f"{cohort}:{problem_id}:seed{seed}:s{outer_iter}:"
            f"g{relation.owner_group_indices[0]}-{relation.owner_group_indices[1]}:"
            f"{dispatch_checkpoint_hash[:12]}"
        )
        selector_arm = selector_arm_for_context(
            action_set,
            relation=relation,
            current_sweep=outer_iter,
            checkpoint_hash=action_set.checkpoint_hash,
            current_shared_values=current_values,
        )
        selector_reason = (
            action_set.selector_reason
            if selector_arm != "true_no_writeback"
            else "g0_validation_abstain"
        )
        continuation_state = NativeContinuationState(
            incumbent=tuple(float(value) for value in incumbent_array),
            sweep_index=outer_iter,
            next_group_index=group_index + 1,
            completed_group_deltas=tuple(float(value) for value in completed_group_deltas),
            completed_group_actual_fes=tuple(
                int(value) for value in completed_group_actual_fes
            ),
            group_dims=tuple(tuple(int(value) for value in group) for group in group_dims),
            overlapping_elements=tuple(
                tuple(int(value) for value in overlap) for overlap in overlapping_elements
            ),
            population_sizes=tuple(int(value) for value in population_sizes),
            optimizer_budgets=tuple(int(value) for value in optimizer_budgets),
            efficiency_ewma=tuple(float(value) for value in efficiency_ewma),
            completed_efficiency_sweeps=int(completed_efficiency_sweeps),
            stagnation_streaks=tuple(int(value) for value in stagnation_streaks),
            stagnation_cooldowns=tuple(0 for _ in group_dims),
            lower_bound=self.lower,
            upper_bound=self.upper,
            sigma=self.sigma,
        )
        owner_group_dimensions = tuple(
            tuple(int(value) for value in group_dims[owner_group])
            for owner_group in relation.owner_group_indices
        )
        owner_optimizer_means = tuple(
            tuple(float(incumbent_array[dimension]) for dimension in dimensions)
            for dimensions in owner_group_dimensions
        )
        def group_seed(sweep: int, group: int) -> int:
            return self.derive_seed(
                int(seed),
                self.fun_name,
                self.fun_id,
                0,
                sweep * len(group_dims) + group + 1,
            )

        def start_branch(arm: str):
            objective = self._fresh_objective()
            record = getattr(objective, "fitness_record")

            def evaluate(values: np.ndarray) -> np.ndarray:
                return np.asarray(objective(values), dtype=float)

            action_result = execute_action_ceiling_arm(
                ActionExecutionRequest(
                    arm=arm,
                    context_hash=dispatch_checkpoint_hash,
                    action_set=action_set,
                    incumbent=tuple(float(value) for value in incumbent_array),
                    incumbent_fitness=float(incumbent_fitness),
                    previous_values=tuple(float(value) for value in previous_values),
                    current_values=tuple(float(value) for value in current_values),
                    previous_delta=float(previous_delta),
                    current_delta=float(current_delta),
                    owner_group_dimensions=owner_group_dimensions,
                    owner_optimizer_means=owner_optimizer_means,
                ),
                evaluate=evaluate,
            )
            branch_optimizer = self._group_optimizer(
                objective,
                evaluate,
                initial_means=(
                    dict(
                        zip(
                            relation.owner_group_indices,
                            action_result.owner_optimizer_means,
                            strict=True,
                        )
                    )
                    if action_result.optimizer_mean_mutation_norm > 0.0
                    else None
                ),
            )
            return objective, record, action_result, evaluate, branch_optimizer

        branch_results: dict[str, dict[str, object]] = {}
        (
            native_objective,
            native_record,
            native_action,
            native_evaluate,
            native_optimizer,
        ) = start_branch("native_eq8")
        native_cycle = run_native_group_cycle(
            replace(continuation_state, incumbent=native_action.incumbent),
            evaluate=native_evaluate,
            fitness_record=native_record,
            optimize_group=native_optimizer,
            group_seed=group_seed,
        )
        horizon_fe = len(native_cycle.fitness_record)
        if horizon_fe <= 0:
            raise RuntimeError("native action-ceiling cycle consumed no FEs")
        native_continuation = run_native_continuation(
            replace(
                continuation_state,
                incumbent=native_cycle.incumbent,
                sweep_index=native_cycle.sweep_index,
                next_group_index=native_cycle.next_group_index,
                completed_group_deltas=native_cycle.completed_group_deltas,
                completed_group_actual_fes=native_cycle.completed_group_actual_fes,
                efficiency_ewma=native_cycle.efficiency_ewma,
                completed_efficiency_sweeps=(
                    native_cycle.completed_efficiency_sweeps
                ),
                stagnation_streaks=native_cycle.stagnation_streaks,
                stagnation_cooldowns=native_cycle.stagnation_cooldowns,
            ),
            evaluate=native_evaluate,
            fitness_record=native_record,
            optimize_group=native_optimizer,
            group_seed=group_seed,
            target_relative_fe=3 * horizon_fe,
        )
        native_continuation = replace(
            native_continuation,
            execution_sweep_trace=(
                native_cycle.execution_sweep_trace
                + native_continuation.execution_sweep_trace
            ),
            execution_order_trace=(
                native_cycle.execution_order_trace
                + native_continuation.execution_order_trace
            ),
            group_budget_trace=(
                native_cycle.group_budget_trace
                + native_continuation.group_budget_trace
            ),
            group_start_fe_trace=(
                native_cycle.group_start_fe_trace
                + native_continuation.group_start_fe_trace
            ),
        )
        branch_results["native_eq8"] = {
            "action": native_action,
            "record": native_continuation.fitness_record,
            "continuation": native_continuation,
            "errors": branch_horizon_errors(
                prefix_best_error=min(prefix),
                post_checkpoint_record=native_continuation.fitness_record,
                sweep_horizon_fe=horizon_fe,
            ),
        }

        for arm in ACTION_CEILING_ARMS:
            if arm == "native_eq8":
                continue
            objective, record, action_result, evaluate, branch_optimizer = start_branch(
                arm
            )
            continuation = run_native_continuation(
                replace(continuation_state, incumbent=action_result.incumbent),
                evaluate=evaluate,
                fitness_record=record,
                optimize_group=branch_optimizer,
                group_seed=group_seed,
                target_relative_fe=3 * horizon_fe,
                continuation_arm=arm,
            )
            branch_results[arm] = {
                "action": action_result,
                "record": continuation.fitness_record,
                "continuation": continuation,
                "errors": branch_horizon_errors(
                    prefix_best_error=min(prefix),
                    post_checkpoint_record=continuation.fitness_record,
                    sweep_horizon_fe=horizon_fe,
                ),
            }

        context_row = {
            "protocol_version": ACTION_CEILING_PROTOCOL_VERSION,
            "cohort": cohort,
            "problem_id": problem_id,
            "seed": str(seed),
            "context_id": context_id,
            "relation_id": "g{}:v{}".format(
                "-".join(str(value) for value in relation.owner_group_indices),
                "-".join(str(value) for value in relation.shared_variable_indices),
            ),
            "action_set_hash": action_set.action_set_hash,
            "checkpoint_hash": action_set.checkpoint_hash,
            "dispatch_checkpoint_hash": dispatch_checkpoint_hash,
            "phase_boundary_fe": str(action_set.checkpoint_fe),
            "dispatch_fe": str(dispatch_fe),
            "issued_sweep": str(action_set.issued_sweep),
            "target_sweep": str(action_set.target_sweep),
            "group_index": str(group_index),
            "efficiency_ewma": json.dumps(list(efficiency_ewma)),
            "completed_efficiency_sweeps": str(completed_efficiency_sweeps),
            "stagnation_streaks": json.dumps(list(stagnation_streaks)),
            "population_sizes": json.dumps(list(population_sizes)),
            "uniform_group_budgets": json.dumps(list(optimizer_budgets)),
            "horizon_fe": str(horizon_fe),
            "selector_arm": selector_arm,
            "selector_reason": selector_reason,
            "anchor_values": json.dumps(action_set.anchor.shared_values),
            "left_values": json.dumps(action_set.left_owner.shared_values),
            "right_values": json.dumps(action_set.right_owner.shared_values),
            "bridge_values": json.dumps(action_set.bridge.shared_values),
            "bridge_weights": json.dumps(
                {
                    "left_owner": action_set.bridge_weights.left_owner,
                    "right_owner": action_set.bridge_weights.right_owner,
                },
                sort_keys=True,
            ),
            "native_parity": "",
            "runtime_authorized": "0",
            "status": "pending_native_parity",
            "invalidation_reason": "",
        }

        native = branch_results["native_eq8"]
        targets = {"immediate": 1, "sweep_1": horizon_fe, "sweep_3": 3 * horizon_fe}
        arm_rows: list[dict[str, str]] = []
        for arm in ACTION_CEILING_ARMS:
            branch = branch_results[arm]
            action_result = branch["action"]
            continuation = branch["continuation"]
            for row in paired_arm_rows(native["errors"], branch["errors"]):
                horizon = str(row["horizon"])
                target_relative_fe = targets[horizon]
                trace_count = sum(
                    start_fe <= target_relative_fe
                    for start_fe in continuation.group_start_fe_trace
                )
                continuation_applied = any(
                    event_fe <= target_relative_fe
                    for event_fe in continuation.policy_application_fes
                )
                warm_shifts = tuple(
                    shift
                    for event_fe, shift in zip(
                        continuation.warm_start_event_fes,
                        continuation.warm_start_shift_norms,
                        strict=True,
                    )
                    if event_fe <= target_relative_fe
                )
                arm_rows.append(
                    {
                        "protocol_version": ACTION_CEILING_PROTOCOL_VERSION,
                        "cohort": cohort,
                        "problem_id": problem_id,
                        "seed": str(seed),
                        "context_id": context_id,
                        "arm": arm,
                        "horizon": horizon,
                        "target_fe": str(dispatch_fe + targets[horizon]),
                        "natural_endpoint_fe": str(dispatch_fe + len(branch["record"])),
                        "native_error": f"{float(row['native_error']):.17e}",
                        "arm_error": f"{float(row['arm_error']):.17e}",
                        "delta": f"{float(row['delta']):.17e}",
                        "extra_fes": str(action_result.extra_fes),
                        "counterfactual_applied": str(
                            int(
                                action_result.counterfactual_applied
                                or continuation_applied
                            )
                        ),
                        "mutation_norm": f"{action_result.mutation_norm:.17e}",
                        "optimizer_mean_mutation_norm": (
                            f"{action_result.optimizer_mean_mutation_norm:.17e}"
                        ),
                        "continuation_policy_applied": str(
                            int(continuation_applied)
                        ),
                        "execution_sweep_trace": json.dumps(
                            list(continuation.execution_sweep_trace[:trace_count])
                        ),
                        "execution_order_trace": json.dumps(
                            list(continuation.execution_order_trace[:trace_count])
                        ),
                        "group_budget_trace": json.dumps(
                            list(continuation.group_budget_trace[:trace_count])
                        ),
                        "warm_start_trigger_count": str(
                            len(warm_shifts)
                        ),
                        "warm_start_mean_shift_norm": (
                            f"{float(np.linalg.norm(warm_shifts)):.17e}"
                        ),
                        "selected_candidate": action_result.selected_candidate,
                        "runtime_authorized": "0",
                        "status": "pending_native_parity",
                        "invalidation_reason": "",
                    }
                )
        return CapturedActionCeilingContext(
            context_row=context_row,
            arm_rows=tuple(arm_rows),
            expected_native_record=tuple(
                float(value) for value in native["record"][:horizon_fe]
            ),
            expected_native_incumbent=tuple(native["action"].incumbent),
            expected_native_incumbent_hash=_sha256(
                list(native["action"].incumbent)
            ),
        )
