from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

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


def test_runtime_capture_executes_all_arms_from_one_context(tmp_path: Path) -> None:
    runtime = HccActionCeilingRuntime(
        benchmark_factory=FakeBenchmark,
        cmaes_factory=FakeOptimizer,
        mmes_factory=FakeOptimizer,
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
        group_dims=((0, 1, 4, 5), (2, 3, 4, 5)),
        overlapping_elements=((4, 5),),
        population_sizes=(2, 2),
        optimizer_budgets=(2, 2),
        fitness_prefix=(1200.0, 1000.0, 998.0),
        topology_hash=_hash("topology"),
        order_hash=_hash("order"),
    )

    assert captured.context_row["status"] == "pending_native_parity"
    assert captured.context_row["selector_arm"] == "exact_bridge"
    assert len(captured.expected_native_record) == 6
    assert len(captured.arm_rows) == len(ACTION_CEILING_ARMS) * len(ACTION_CEILING_HORIZONS)
    rescue = [
        row for row in captured.arm_rows if row["arm"] == "non_decomposition_rescue"
    ]
    assert {row["extra_fes"] for row in rescue} == {"97"}
    assert all(row["runtime_authorized"] == "0" for row in captured.arm_rows)
    assert all(row["counterfactual_applied"] == "1" for row in captured.arm_rows)


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
