from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from arac.actions.full_space_sep_cma import (
    FULL_SPACE_SEP_CMA_ACTION,
    full_space_vector_hash,
)
from arac.backends.hcc_action_ceiling import native_eq8_values
from arac.backends.hcc_action_ceiling_runtime import HccActionCeilingRuntime
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    ACTION_CEILING_HORIZONS,
    FrozenProbeCandidate,
    RelationActionSet,
)
from arac.policy.evidence_overlay import (
    BridgeWeights,
    ProbeUtilities,
    RelationKey,
    runtime_probe_anchor_hash,
    runtime_probe_shared_values_hash,
)
from scripts import hcc_smoke_runner as runner


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _action_set() -> RelationActionSet:
    relation = RelationKey((0, 1), (4, 5))

    def candidate(name: str, values: tuple[float, ...], fitness: float):
        return FrozenProbeCandidate(
            name=name,
            shared_values=values,
            shared_values_hash=(
                runtime_probe_anchor_hash(relation, values)
                if name == "anchor"
                else runtime_probe_shared_values_hash(relation, values)
            ),
            candidate_hash=_hash(name),
            fitness=fitness,
            utility=math.log(10.0 / fitness),
        )

    return RelationActionSet(
        relation=relation,
        anchor=candidate("anchor", (0.0, 0.0), 10.0),
        left_owner=candidate("left_owner", (1.0, 2.0), 8.0),
        right_owner=candidate("right_owner", (3.0, 4.0), 9.0),
        bridge=candidate("bridge", (2.0, 3.0), 7.0),
        bridge_weights=BridgeWeights(0.5, 0.5),
        probe_utilities=ProbeUtilities(math.log(10 / 8), math.log(10 / 9), math.log(10 / 7)),
        selector_winner="bridge",
        selector_utility=math.log(10 / 7),
        selector_reason="unique_probe_winner_above_one_percent",
        checkpoint_fe=100,
        checkpoint_hash=_hash("checkpoint"),
        issued_sweep=2,
        target_sweep=3,
    )


class Sphere:
    def __init__(self) -> None:
        self.fitness_record: list[float] = []

    def __call__(self, values):
        rows = np.asarray(values, dtype=float)
        if rows.ndim == 1:
            rows = rows[None, :]
        result = np.sum(np.square(rows), axis=1)
        self.fitness_record.extend(float(value) for value in result)
        return result


class FakeBenchmark:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_function(self, fun_name: str, fun_id: int) -> Sphere:
        return Sphere()


class FakeOptimizer:
    def __init__(self, problem, options) -> None:
        self.problem = problem
        self.options = options

    def optimize(self):
        requested = int(self.options["max_function_evaluations"])
        mean = np.asarray(self.options["mean"][0], dtype=float)
        batch = np.repeat(mean[None, :], requested, axis=0)
        values = np.asarray(self.problem["fitness_function"](batch), dtype=float)
        return {
            "best_so_far_x": mean,
            "best_so_far_y": float(np.min(values)),
            "n_function_evaluations": requested,
        }


class EarlyStoppingGroupOptimizer(FakeOptimizer):
    def optimize(self):
        requested = int(self.options["max_function_evaluations"])
        if requested != 24:
            return super().optimize()
        mean = np.asarray(self.options["mean"][0], dtype=float)
        batch = np.repeat(mean[None, :], 12, axis=0)
        values = np.asarray(self.problem["fitness_function"](batch), dtype=float)
        return {
            "best_so_far_x": mean,
            "best_so_far_y": float(np.min(values)),
            "n_function_evaluations": 12,
        }


def test_group_optimizer_consumes_owner_context_mean_once(tmp_path: Path) -> None:
    observed_means: list[tuple[float, ...]] = []

    class RecordingOptimizer(FakeOptimizer):
        def __init__(self, problem, options) -> None:
            observed_means.append(
                tuple(float(value) for value in options["mean"][0])
            )
            super().__init__(problem, options)

    runtime = HccActionCeilingRuntime(
        benchmark_factory=FakeBenchmark,
        cmaes_factory=RecordingOptimizer,
        sepcmaes_factory=runner.SEPCMAES,
        combine=lambda x, background, dims: runner.combine(x, background, dims),
        derive_seed=runner.derive_optimizer_seed,
        fun_name="elliptic",
        fun_id=3,
        output_path=tmp_path,
        data_root=tmp_path,
        sigma=0.5,
        cmaes_restart=True,
        early_stopping_evaluations=1000,
        lower=-100.0,
        upper=100.0,
        dimension=8,
    )
    objective = Sphere()
    optimize = runtime._group_optimizer(
        objective,
        lambda values: np.asarray(objective(values), dtype=float),
        initial_means={
            0: (10.0, 11.0, 3.0, 4.0),
            1: (20.0, 21.0, 3.0, 4.0),
        },
    )
    background = np.ones(8)
    common = {
        "background": background,
        "requested_fes": 2,
        "population_size": 2,
        "seed": 117,
    }

    optimize(
        group_index=0,
        dims=(0, 1, 4, 5),
        mean=np.ones(4),
        sigma=0.5,
        **common,
    )
    optimize(
        group_index=1,
        dims=(2, 3, 4, 5),
        mean=np.ones(4),
        sigma=0.5,
        **common,
    )
    optimize(
        group_index=0,
        dims=(0, 1, 4, 5),
        mean=np.ones(4),
        sigma=0.5,
        **common,
    )

    assert observed_means == [
        (10.0, 11.0, 3.0, 4.0),
        (20.0, 21.0, 3.0, 4.0),
        (1.0, 1.0, 1.0, 1.0),
    ]


def test_runtime_capture_executes_all_arms_from_one_context(tmp_path: Path) -> None:
    runtime = HccActionCeilingRuntime(
        benchmark_factory=FakeBenchmark,
        cmaes_factory=FakeOptimizer,
        sepcmaes_factory=runner.SEPCMAES,
        combine=lambda x, background, dims: runner.combine(x, background, dims),
        derive_seed=runner.derive_optimizer_seed,
        fun_name="elliptic",
        fun_id=3,
        output_path=tmp_path,
        data_root=tmp_path,
        sigma=0.5,
        cmaes_restart=True,
        early_stopping_evaluations=1000,
        lower=-100.0,
        upper=100.0,
        dimension=1000,
    )
    incumbent = np.ones(1000)
    incumbent[4:6] = 0.0
    captured = runtime.capture(
        action_set=_action_set(),
        cohort="real_aob",
        problem_id="E3",
        seed=117,
        dispatch_fe=120,
        outer_iter=3,
        group_index=1,
        incumbent=incumbent,
        incumbent_fitness=998.0,
        previous_values=(1.0, 1.0),
        current_values=(0.0, 0.0),
        previous_delta=1.0,
        current_delta=2.0,
        completed_group_deltas=(1.0, 2.0),
        completed_group_actual_fes=(13, 13),
        group_dims=((0, 1, 4, 5), (2, 3, 4, 5)),
        overlapping_elements=((4, 5),),
        population_sizes=(2, 2),
        optimizer_budgets=(12, 12),
        efficiency_ewma=(0.1, 0.2),
        completed_efficiency_sweeps=3,
        stagnation_streaks=(0, 0),
        fitness_prefix=(1200.0, 1000.0, 998.0),
        topology_hash=_hash("topology"),
        order_hash=_hash("order"),
    )

    assert captured.context_row["status"] == "pending_native_parity"
    assert captured.context_row["selector_arm"] == "exact_bridge"
    assert len(captured.expected_native_record) == 26
    assert captured.expected_native_cycle_sweep_trace == (4, 4)
    assert captured.expected_native_cycle_order_trace == (0, 1)
    assert captured.expected_native_cycle_budget_trace == (12, 12)
    assert captured.expected_native_cycle_start_fe_trace == (1, 14)
    assert captured.expected_native_cycle_incumbent_hash == (
        runner._canonical_payload_sha256(
            list(captured.expected_native_cycle_incumbent)
        )
    )
    assert len(captured.arm_rows) == len(ACTION_CEILING_ARMS) * len(ACTION_CEILING_HORIZONS)
    assert {
        row["action_actual_fes"]
        for row in captured.arm_rows
        if row["arm"] != FULL_SPACE_SEP_CMA_ACTION
    } == {"0"}
    assert {
        row["action_actual_fes"]
        for row in captured.arm_rows
        if row["arm"] == FULL_SPACE_SEP_CMA_ACTION
    } == {"26"}
    assert all(row["runtime_authorized"] == "0" for row in captured.arm_rows)
    assert all(
        json.loads(row["execution_order_trace"])
        for row in captured.arm_rows
        if not (
            row["arm"] == FULL_SPACE_SEP_CMA_ACTION
            and row["horizon"] in {"immediate", "sweep_1"}
        )
    )
    assert all(
        json.loads(row["group_budget_trace"])
        for row in captured.arm_rows
        if not (
            row["arm"] == FULL_SPACE_SEP_CMA_ACTION
            and row["horizon"] in {"immediate", "sweep_1"}
        )
    )
    applied = {
        row["arm"]: row["counterfactual_applied"]
        for row in captured.arm_rows
        if row["horizon"] == "sweep_1"
    }
    assert applied == {
        "native_eq8": "1",
        "true_no_writeback": "0",
        "exact_left": "1",
        "exact_right": "1",
        "exact_bridge": "1",
        "efficiency_budget_reallocation": "1",
        "delta_priority_scan": "1",
        "stagnation_cross_group_warm_start": "1",
        FULL_SPACE_SEP_CMA_ACTION: "1",
    }
    no_trigger_warm_start = [
        row
        for row in captured.arm_rows
        if row["arm"] == "stagnation_cross_group_warm_start"
    ]
    assert all(row["continuation_policy_applied"] == "0" for row in no_trigger_warm_start)
    assert all(float(row["delta"]) == 0.0 for row in no_trigger_warm_start)


def test_native_cycle_parity_requires_exact_relation_trace_and_horizon_fe() -> None:
    relation = RelationKey((0, 1), (4, 5))
    expected_record = tuple(float(value) for value in range(26))
    incumbent = tuple(float(value) for value in range(8))
    pending = runner.PendingActionCeilingParity(
        relation=relation,
        expected_sweep=4,
        start_fe=120,
        horizon_fe=26,
        expected_record=expected_record,
        expected_incumbent=incumbent,
        expected_incumbent_hash=_hash("incumbent"),
        expected_sweep_trace=(4, 4),
        expected_order_trace=(0, 1),
        expected_budget_trace=(12, 12),
        expected_start_fe_trace=(1, 14),
        context_row={},
        arm_rows=[],
        actual_sweep_trace=[4, 4],
        actual_order_trace=[0, 1],
        actual_budget_trace=[12, 12],
        actual_start_fe_trace=[1, 14],
    )
    fitness_record = [999.0] * 120 + list(expected_record)

    actual_record, reason = (
        runner._action_ceiling_native_cycle_prewriteback_parity(
            pending,
            relation=relation,
            current_sweep=4,
            current_fe=146,
            fitness_record=fitness_record,
        )
    )
    assert actual_record == expected_record
    assert reason == ""

    fitness_record.append(1.0)
    _, reason = runner._action_ceiling_native_cycle_prewriteback_parity(
        pending,
        relation=relation,
        current_sweep=4,
        current_fe=147,
        fitness_record=fitness_record,
    )
    assert reason == "native_horizon_fe_parity_mismatch"

    _, reason = runner._action_ceiling_native_cycle_prewriteback_parity(
        pending,
        relation=RelationKey((1, 2), (5, 6)),
        current_sweep=4,
        current_fe=146,
        fitness_record=fitness_record,
    )
    assert reason == "native_relation_parity_mismatch"


def test_full_space_sep_cma_strict_gate_and_frozen_fe_cost(tmp_path: Path) -> None:
    runtime = HccActionCeilingRuntime(
        benchmark_factory=FakeBenchmark,
        cmaes_factory=FakeOptimizer,
        sepcmaes_factory=runner.SEPCMAES,
        combine=lambda x, background, dims: runner.combine(x, background, dims),
        derive_seed=runner.derive_optimizer_seed,
        fun_name="elliptic",
        fun_id=3,
        output_path=tmp_path,
        data_root=tmp_path,
        sigma=0.5,
        cmaes_restart=True,
        early_stopping_evaluations=1000,
        lower=-100.0,
        upper=100.0,
        dimension=1000,
    )
    incumbent = np.ones(1000)
    incumbent[4:6] = 0.0
    post_eq8 = incumbent.copy()
    post_eq8[4:6] = native_eq8_values(
        (1.0, 1.0),
        (0.0, 0.0),
        1.0,
        2.0,
    )

    for acceptance_fitness, expected_accepted in ((1.0e9, "1"), (0.0, "0")):
        captured = runtime.capture(
            action_set=_action_set(),
            cohort="real_aob",
            problem_id="E3",
            seed=117,
            dispatch_fe=120,
            outer_iter=3,
            group_index=1,
            incumbent=incumbent,
            incumbent_fitness=acceptance_fitness,
            previous_values=(1.0, 1.0),
            current_values=(0.0, 0.0),
            previous_delta=1.0,
            current_delta=2.0,
            completed_group_deltas=(1.0, 2.0),
            completed_group_actual_fes=(13, 13),
            group_dims=((0, 1, 4, 5), (2, 3, 4, 5)),
            overlapping_elements=((4, 5),),
            population_sizes=(2, 2),
            optimizer_budgets=(12, 12),
            efficiency_ewma=(0.1, 0.2),
            completed_efficiency_sweeps=3,
            stagnation_streaks=(0, 0),
            fitness_prefix=(1200.0, 1000.0, 998.0),
            topology_hash=_hash("topology"),
            order_hash=_hash("order"),
        )
        horizon_fe = int(captured.context_row["horizon_fe"])
        sep_rows = [
            row
            for row in captured.arm_rows
            if row["arm"] == FULL_SPACE_SEP_CMA_ACTION
        ]
        sweep_3 = next(row for row in sep_rows if row["horizon"] == "sweep_3")
        candidate_fitness = float(sweep_3["action_candidate_fitness"])

        assert float(
            captured.context_row["full_space_acceptance_fitness"]
        ) == acceptance_fitness
        assert captured.context_row["full_space_initial_mean_hash"] == (
            full_space_vector_hash(post_eq8)
        )
        assert all(row["action_actual_fes"] == str(horizon_fe) for row in sep_rows)
        assert all(row["action_accepted"] == expected_accepted for row in sep_rows)
        assert (candidate_fitness < acceptance_fitness) == (
            expected_accepted == "1"
        )
        assert sweep_3["action_post_incumbent_hash"] == (
            sweep_3["action_candidate_hash"]
            if expected_accepted == "1"
            else captured.context_row["full_space_initial_mean_hash"]
        )
        assert all(row["counterfactual_applied"] == "1" for row in sep_rows)
        assert all(
            json.loads(row["execution_start_fe_trace"]) == []
            for row in sep_rows
            if row["horizon"] in {"immediate", "sweep_1"}
        )
        assert json.loads(sweep_3["execution_start_fe_trace"])[0] == (
            horizon_fe + 1
        )
        assert int(sweep_3["natural_endpoint_fe"]) - 120 == 3 * horizon_fe


def test_runtime_horizon_uses_actual_native_cycle_fe(tmp_path: Path) -> None:
    runtime = HccActionCeilingRuntime(
        benchmark_factory=FakeBenchmark,
        cmaes_factory=EarlyStoppingGroupOptimizer,
        sepcmaes_factory=runner.SEPCMAES,
        combine=lambda x, background, dims: runner.combine(x, background, dims),
        derive_seed=runner.derive_optimizer_seed,
        fun_name="elliptic",
        fun_id=3,
        output_path=tmp_path,
        data_root=tmp_path,
        sigma=0.5,
        cmaes_restart=True,
        early_stopping_evaluations=1000,
        lower=-100.0,
        upper=100.0,
        dimension=1000,
    )
    incumbent = np.ones(1000)
    incumbent[4:6] = 0.0
    captured = runtime.capture(
        action_set=_action_set(),
        cohort="real_aob",
        problem_id="E3",
        seed=117,
        dispatch_fe=120,
        outer_iter=3,
        group_index=1,
        incumbent=incumbent,
        incumbent_fitness=998.0,
        previous_values=(1.0, 1.0),
        current_values=(0.0, 0.0),
        previous_delta=1.0,
        current_delta=2.0,
        completed_group_deltas=(1.0, 2.0),
        completed_group_actual_fes=(25, 25),
        group_dims=((0, 1, 4, 5), (2, 3, 4, 5)),
        overlapping_elements=((4, 5),),
        population_sizes=(2, 2),
        optimizer_budgets=(24, 24),
        efficiency_ewma=(0.1, 0.2),
        completed_efficiency_sweeps=3,
        stagnation_streaks=(0, 0),
        fitness_prefix=(1200.0, 1000.0, 998.0),
        topology_hash=_hash("topology"),
        order_hash=_hash("order"),
    )

    assert captured.context_row["horizon_fe"] == "26"
    assert len(captured.expected_native_record) == 26


def test_relation_context_uses_structural_shared_variable_order() -> None:
    relation = runner.build_overlap_relation_for_pair(
        problem_id="A4",
        outer_iter=3,
        grouping_result=[[0, 4, 5], [1, 4, 5]],
        overlapping_elements=[[5, 4]],
        fitness_delta_list=[1.0, 2.0],
        group_right=1,
        budget_remaining_ratio=0.5,
    )
    original = np.arange(10, dtype=float)
    current = original + 100.0

    context = runner.build_relation_execution_context(
        relation=relation,
        original_best=original,
        current_best=current,
        previous_delta=1.0,
        current_delta=2.0,
    )

    assert relation.shared_vars == (4, 5)
    assert context.overlap_indices == [4, 5]
    assert context.previous_values.tolist() == [4.0, 5.0]
    assert context.current_values.tolist() == [104.0, 105.0]


def test_action_ceiling_cli_requires_explicit_capture_flag() -> None:
    common = [
        "--functions",
        "ackley",
        "--ids",
        "4",
        "--output-root",
        "results/test",
        "--seed",
        "1",
        "--max-fes",
        "100000",
        "--arac-action",
        runner.EVIDENCE_ACTION_CONTROLLER_V37,
        "--enable-relation-dispatch",
        "--relation-policy",
        runner.ACTION_CEILING_POLICY,
        "--evidence-overlay-mode",
        "paired_owner",
    ]

    try:
        runner.parse_args(common)
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("action_ceiling without capture flag must fail")

    args = runner.parse_args(
        common
        + [
            "--action-ceiling-capture",
            "--action-ceiling-cohort",
            "real_aob",
        ]
    )
    assert args.action_ceiling_capture is True
    assert args.relation_policy == runner.ACTION_CEILING_POLICY
