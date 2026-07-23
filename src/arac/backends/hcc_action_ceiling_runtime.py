"""Dependency-injected HCC runtime adapter for action-ceiling branches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from arac.actions.budget_reallocation import (
    FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
    BudgetAllocationAction,
)
from arac.actions.gcb import (
    FULL_SPACE_DIMENSION,
    GCB_ACTION,
    GCB_SEED_NAMESPACE,
    TRIGGER_SCOPE_RELATION_DISPATCH,
    GcbAction,
    build_gcb_optimizer,
)
from arac.actions.gcb import GcbExecutionContext
from arac.actions.runtime_dispatcher import DEFAULT_RUNTIME_ACTION_DISPATCHER
from arac.actions.shrunk_budget_pulse import (
    SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
    ShrunkEfficiencyBudgetPulseAction,
)
from arac.backends.hcc_action_ceiling import (
    ActionExecutionRequest,
    NativeContinuationState,
    OptimizationResult,
    branch_horizon_errors,
    execute_action_ceiling_arm,
    freeze_efficiency_budget_action,
    freeze_shrunk_efficiency_budget_pulse_action,
    paired_arm_rows,
    run_native_group_cycle,
    run_native_continuation,
    selector_arm_for_context,
)
from arac.backends.hcc_gcb import (
    compile_gcb_relation_action,
    gcb_optimizer_seed,
)
from arac.backends.hcc_phase2_action_context import phase2_relation_hash
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    ACTION_CEILING_PROTOCOL_VERSION,
    RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION,
    RS_FAMILY_RASTRIGIN_ARMS,
    RS_FAMILY_SCHWEFEL_ARMS,
    S_FAMILY_BUDGET_PULSE_ARMS,
    S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION,
    RelationActionSet,
)
from arac.policy.evidence_overlay import runtime_probe_anchor_hash


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_AOB_PROBLEM_FAMILIES = {
    "elliptic": "E",
    "ackley": "A",
    "rastrigin": "R",
    "schwefel": "S",
}


def _runtime_problem_id(fun_name: str, fun_id: int) -> str:
    family = _AOB_PROBLEM_FAMILIES.get(fun_name)
    if family is None or type(fun_id) is not int or fun_id not in range(1, 7):
        raise ValueError("action-ceiling runtime requires an AOB family/id 1 through 6")
    return f"{family}{fun_id}"


@dataclass(frozen=True)
class CapturedActionCeilingContext:
    context_row: dict[str, str]
    arm_rows: tuple[dict[str, str], ...]
    expected_native_record: tuple[float, ...]
    expected_native_incumbent: tuple[float, ...]
    expected_native_incumbent_hash: str
    expected_native_cycle_incumbent: tuple[float, ...]
    expected_native_cycle_incumbent_hash: str
    expected_native_cycle_sweep_trace: tuple[int, ...]
    expected_native_cycle_order_trace: tuple[int, ...]
    expected_native_cycle_budget_trace: tuple[int, ...]
    expected_native_cycle_start_fe_trace: tuple[int, ...]


@dataclass(frozen=True)
class HccActionCeilingRuntime:
    benchmark_factory: Callable[..., object]
    cmaes_factory: Callable[..., object]
    sepcmaes_factory: Callable[..., object]
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
    capture_arms: tuple[str, ...] = ACTION_CEILING_ARMS
    artifact_protocol_version: str = ACTION_CEILING_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        arms = tuple(self.capture_arms)
        problem_id = _runtime_problem_id(self.fun_name, self.fun_id)
        full_matrix = (
            self.artifact_protocol_version == ACTION_CEILING_PROTOCOL_VERSION
            and arms == ACTION_CEILING_ARMS
        )
        rastrigin_target = (
            self.artifact_protocol_version
            == RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
            and arms == RS_FAMILY_RASTRIGIN_ARMS
            and problem_id.startswith("R")
        )
        schwefel_target = (
            self.artifact_protocol_version
            == RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
            and arms == RS_FAMILY_SCHWEFEL_ARMS
            and problem_id.startswith("S")
        )
        schwefel_budget_pulse = (
            self.artifact_protocol_version
            == S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION
            and arms == S_FAMILY_BUDGET_PULSE_ARMS
            and problem_id.startswith("S")
        )
        if not (
            full_matrix
            or rastrigin_target
            or schwefel_target
            or schwefel_budget_pulse
        ):
            raise ValueError("unsupported action-ceiling runtime protocol/arms/family")
        object.__setattr__(self, "capture_arms", arms)

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
        if problem_id != _runtime_problem_id(self.fun_name, self.fun_id):
            raise ValueError("capture problem id does not match runtime function")
        incumbent_array = np.asarray(incumbent, dtype=float).reshape(-1)
        prefix = tuple(float(value) for value in fitness_prefix)
        if incumbent_array.size != self.dimension or not prefix:
            raise ValueError("action-ceiling capture state is incomplete")
        relation = action_set.relation
        dispatch_checkpoint_payload = {
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
        dispatch_checkpoint_hash = _sha256(dispatch_checkpoint_payload)
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
        raw_budget_action: BudgetAllocationAction | None = None
        shrunk_budget_action: ShrunkEfficiencyBudgetPulseAction | None = None
        if any(
            arm in self.capture_arms
            for arm in (
                FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
                SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
            )
        ):
            if int(outer_iter) != action_set.target_sweep:
                raise ValueError(
                    "frozen budget action must be compiled at its target sweep"
                )
            raw_budget_action = freeze_efficiency_budget_action(
                problem_id=problem_id,
                run_seed=int(seed),
                checkpoint_fe=int(dispatch_fe),
                dispatch_checkpoint_hash=dispatch_checkpoint_hash,
                source_efficiency_ewma=efficiency_ewma,
                population_sizes=population_sizes,
                uniform_group_budgets=optimizer_budgets,
                issued_sweep=int(outer_iter),
                target_sweep=int(outer_iter) + 1,
            )
            if SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION in self.capture_arms:
                shrunk_budget_action = (
                    freeze_shrunk_efficiency_budget_pulse_action(
                        problem_id=problem_id,
                        run_seed=int(seed),
                        checkpoint_fe=int(dispatch_fe),
                        dispatch_checkpoint_hash=dispatch_checkpoint_hash,
                        raw_group_budgets=raw_budget_action.group_budgets,
                        population_sizes=population_sizes,
                        uniform_group_budgets=optimizer_budgets,
                        issued_sweep=int(outer_iter),
                        target_sweep=int(outer_iter) + 1,
                    )
                )
        budget_actions = {
            FROZEN_EFFICIENCY_BUDGET_REALLOCATION_ACTION: raw_budget_action,
            SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION: shrunk_budget_action,
        }

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
                    dispatch_fe=int(dispatch_fe),
                    action_set=action_set,
                    incumbent=tuple(float(value) for value in incumbent_array),
                    incumbent_fitness=float(incumbent_fitness),
                    previous_values=tuple(float(value) for value in previous_values),
                    current_values=tuple(float(value) for value in current_values),
                    previous_delta=float(previous_delta),
                    current_delta=float(current_delta),
                    owner_group_dimensions=owner_group_dimensions,
                    owner_optimizer_means=owner_optimizer_means,
                    budget_action=budget_actions.get(arm),
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

        gcb_action: GcbAction | None = None
        for arm in self.capture_arms:
            if arm == "native_eq8":
                continue
            objective, record, action_result, evaluate, branch_optimizer = start_branch(
                arm
            )
            if arm == GCB_ACTION:
                if self.dimension != FULL_SPACE_DIMENSION:
                    raise ValueError("GCB requires the 1000D AOB space")
                trigger_context_hash = phase2_relation_hash(
                    relation.owner_group_indices,
                    relation.shared_variable_indices,
                )
                optimizer_seed = gcb_optimizer_seed(
                    dispatch_checkpoint_hash
                )
                prepared_optimizer = build_gcb_optimizer(
                    self.sepcmaes_factory,
                    objective=evaluate,
                    initial_mean=action_result.incumbent,
                    initial_sigma=self.sigma,
                    lower_bound=self.lower,
                    upper_bound=self.upper,
                    budget_fes=horizon_fe,
                    optimizer_seed=optimizer_seed,
                )
                gcb_action = compile_gcb_relation_action(
                    problem_id=problem_id,
                    run_seed=int(seed),
                    dispatch_fe=int(dispatch_fe),
                    dispatch_checkpoint_hash=dispatch_checkpoint_hash,
                    owner_group_indices=relation.owner_group_indices,
                    shared_variable_indices=relation.shared_variable_indices,
                    incumbent=action_result.incumbent,
                    acceptance_fitness=float(action_result.incumbent_fitness),
                    sigma=self.sigma,
                    lower=self.lower,
                    upper=self.upper,
                    budget_fes=horizon_fe,
                    issued_sweep=action_set.issued_sweep,
                    target_sweep=action_set.target_sweep,
                    objective=evaluate,
                    sepcmaes_factory=self.sepcmaes_factory,
                    prepared_optimizer=prepared_optimizer,
                )
                before = len(record)
                execution_result = DEFAULT_RUNTIME_ACTION_DISPATCHER.execute(
                    gcb_action,
                    GcbExecutionContext(
                        objective=evaluate,
                        sepcmaes_factory=self.sepcmaes_factory,
                        current_fe=int(dispatch_fe),
                        current_sweep=int(outer_iter),
                        dispatch_checkpoint_hash=dispatch_checkpoint_hash,
                        trigger_context_hash=trigger_context_hash,
                        trigger_scope=TRIGGER_SCOPE_RELATION_DISPATCH,
                        incumbent=tuple(action_result.incumbent),
                        required_seed_namespace=(
                            GCB_SEED_NAMESPACE
                        ),
                        prepared_optimizer=prepared_optimizer,
                    ),
                )
                observed_fes = len(record) - before
                if observed_fes != execution_result.consumed_fes:
                    raise RuntimeError(
                        "GCB did not consume its frozen FE budget"
                    )
                lifecycle_payload = execution_result.lifecycle.audit_payload(
                    gcb_action
                )

                candidate_fitness = execution_result.candidate_fitness
                post_action_incumbent = np.asarray(
                    execution_result.incumbent,
                    dtype=float,
                )
                action_result = replace(
                    action_result,
                    incumbent=tuple(float(value) for value in post_action_incumbent),
                    incumbent_fitness=execution_result.incumbent_fitness,
                    action_budget_fes=gcb_action.budget_fes,
                    action_actual_fes=observed_fes,
                    action_instance_hash=gcb_action.action_hash,
                    action_lifecycle_payload=json.dumps(
                        lifecycle_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    action_lifecycle_hash=execution_result.lifecycle_hash,
                    action_accepted=execution_result.accepted,
                    action_candidate_hash=execution_result.candidate_hash,
                    action_candidate_fitness=candidate_fitness,
                    action_post_incumbent_hash=execution_result.post_incumbent_hash,
                    optimizer_scope="full_space",
                    optimizer_parameter_hash=(
                        gcb_action.canonical_parameters_hash
                    ),
                    optimizer_initial_state_hash=gcb_action.initial_state_hash,
                    optimizer_final_state_hash=execution_result.final_state_hash,
                    optimizer_population_size=gcb_action.population_size,
                    optimizer_generation_count=(
                        execution_result.optimizer_generation_count
                    ),
                    counterfactual_applied=True,
                    mutation_norm=float(
                        np.linalg.norm(post_action_incumbent - incumbent_array)
                    ),
                    applied_values_hash=_sha256(post_action_incumbent.tolist()),
                )
                continuation = run_native_continuation(
                    replace(
                        continuation_state,
                        incumbent=action_result.incumbent,
                    ),
                    evaluate=evaluate,
                    fitness_record=record,
                    optimize_group=branch_optimizer,
                    group_seed=group_seed,
                    target_relative_fe=3 * horizon_fe,
                    continuation_arm="native_eq8",
                )
                continuation = replace(
                    continuation,
                    policy_application_fes=(1,),
                    continuation_policy_applied=True,
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
                continue
            continuation = run_native_continuation(
                replace(continuation_state, incumbent=action_result.incumbent),
                evaluate=evaluate,
                fitness_record=record,
                optimize_group=branch_optimizer,
                group_seed=group_seed,
                target_relative_fe=3 * horizon_fe,
                continuation_arm=arm,
                frozen_budget_action=budget_actions.get(arm),
            )
            if arm in budget_actions:
                budget_action = budget_actions[arm]
                if budget_action is None:
                    raise RuntimeError("frozen budget branch lost its action instance")
                if (
                    continuation.budget_action_instance_hash
                    != budget_action.action_hash
                    or not continuation.budget_action_lifecycle_payload
                    or not continuation.budget_action_lifecycle_hash
                ):
                    raise RuntimeError("frozen budget lifecycle is incomplete")
                execution_payload = json.loads(
                    continuation.budget_action_lifecycle_payload
                )
                if execution_payload.get("status") != "consumed":
                    raise RuntimeError(
                        "frozen budget action was not consumed at the target sweep"
                    )
                lifecycle_payload = {
                    "action": arm,
                    "instance": budget_action.audit_payload(),
                    "instance_hash": budget_action.action_hash,
                    "execution": execution_payload,
                    "execution_hash": continuation.budget_action_lifecycle_hash,
                }
                action_result = replace(
                    action_result,
                    action_instance_hash=budget_action.action_hash,
                    action_lifecycle_payload=json.dumps(
                        lifecycle_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    action_lifecycle_hash=_sha256(lifecycle_payload),
                    action_accepted=True,
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

        if (
            GCB_ACTION in self.capture_arms
            and gcb_action is None
        ):
            raise RuntimeError("GCB action arm was not executed")
        context_row = {
            "protocol_version": self.artifact_protocol_version,
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
            "dispatch_anchor_hash": runtime_probe_anchor_hash(
                relation,
                current_values,
            ),
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
            "gcb_action_hash": (
                "" if gcb_action is None else gcb_action.action_hash
            ),
            "gcb_action_payload": (
                ""
                if gcb_action is None
                else json.dumps(
                    gcb_action.audit_payload(),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
            "gcb_initial_mean_hash": (
                "" if gcb_action is None else gcb_action.initial_mean_hash
            ),
            "gcb_parameter_hash": (
                ""
                if gcb_action is None
                else gcb_action.canonical_parameters_hash
            ),
            "gcb_optimizer_seed": (
                "" if gcb_action is None else str(gcb_action.optimizer_seed)
            ),
            "gcb_population_size": (
                ""
                if gcb_action is None
                else str(gcb_action.population_size)
            ),
            "gcb_budget_fes": (
                "" if gcb_action is None else str(gcb_action.budget_fes)
            ),
            "gcb_acceptance_fitness": (
                ""
                if gcb_action is None
                else f"{gcb_action.acceptance_fitness:.17e}"
            ),
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
        for arm in self.capture_arms:
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
                        "protocol_version": self.artifact_protocol_version,
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
                        "action_budget_fes": str(
                            action_result.action_budget_fes
                        ),
                        "action_actual_fes": str(
                            action_result.action_actual_fes
                        ),
                        "action_instance_hash": (
                            action_result.action_instance_hash
                        ),
                        "action_lifecycle_payload": (
                            action_result.action_lifecycle_payload
                        ),
                        "action_lifecycle_hash": (
                            action_result.action_lifecycle_hash
                        ),
                        "action_accepted": str(
                            int(action_result.action_accepted)
                        ),
                        "action_candidate_hash": (
                            action_result.action_candidate_hash
                        ),
                        "action_candidate_fitness": (
                            ""
                            if action_result.action_candidate_fitness is None
                            else f"{action_result.action_candidate_fitness:.17e}"
                        ),
                        "action_post_incumbent_hash": (
                            _sha256(list(action_result.incumbent))
                            if self.artifact_protocol_version
                            in {
                                RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION,
                                S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION,
                            }
                            and arm == "native_eq8"
                            else action_result.action_post_incumbent_hash
                        ),
                        "optimizer_scope": action_result.optimizer_scope,
                        "optimizer_parameter_hash": (
                            action_result.optimizer_parameter_hash
                        ),
                        "optimizer_initial_state_hash": (
                            action_result.optimizer_initial_state_hash
                        ),
                        "optimizer_final_state_hash": (
                            action_result.optimizer_final_state_hash
                        ),
                        "optimizer_population_size": str(
                            action_result.optimizer_population_size
                        ),
                        "optimizer_generation_count": str(
                            action_result.optimizer_generation_count
                        ),
                        "counterfactual_applied": str(
                            int(
                                action_result.counterfactual_applied
                                or action_result.action_actual_fes > 0
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
                        "execution_start_fe_trace": json.dumps(
                            list(continuation.group_start_fe_trace[:trace_count])
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
            expected_native_cycle_incumbent=native_cycle.incumbent,
            expected_native_cycle_incumbent_hash=_sha256(
                list(native_cycle.incumbent)
            ),
            expected_native_cycle_sweep_trace=(
                native_cycle.execution_sweep_trace
            ),
            expected_native_cycle_order_trace=(
                native_cycle.execution_order_trace
            ),
            expected_native_cycle_budget_trace=(
                native_cycle.group_budget_trace
            ),
            expected_native_cycle_start_fe_trace=(
                native_cycle.group_start_fe_trace
            ),
        )
