from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from arac.actions.gcb import (
    GCB_ACTION,
    full_space_vector_hash,
)
from arac.actions.shrunk_budget_pulse import (
    SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
    allocate_shrunk_efficiency_budgets,
)
from arac.backends.hcc_action_ceiling import native_eq8_values
from arac.backends.hcc_action_ceiling_runtime import HccActionCeilingRuntime
from arac.policy.action_ceiling import (
    ACTION_CEILING_FULL_MATRIX_PROFILE,
    ACTION_CEILING_ARMS,
    ACTION_CEILING_HORIZONS,
    ACTION_CEILING_KNOWN_ARMS,
    ACTION_CEILING_PROTOCOL_VERSION,
    ActionCeilingObservation,
    FrozenProbeCandidate,
    RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION,
    RS_FAMILY_RASTRIGIN_ARMS,
    RS_FAMILY_SCHWEFEL_ARMS,
    RS_FAMILY_TARGET_PROFILE,
    S_FAMILY_BUDGET_PULSE_ARMS,
    S_FAMILY_BUDGET_PULSE_PROFILE,
    S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION,
    RelationActionSet,
    action_ceiling_capture_contract,
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


class ProgressiveGroupOptimizer(FakeOptimizer):
    def optimize(self):
        requested = int(self.options["max_function_evaluations"])
        candidate = 0.5 * np.asarray(self.options["mean"][0], dtype=float)
        batch = np.repeat(candidate[None, :], requested, axis=0)
        values = np.asarray(self.problem["fitness_function"](batch), dtype=float)
        return {
            "best_so_far_x": candidate,
            "best_so_far_y": float(np.min(values)),
            "n_function_evaluations": requested,
        }


def _capture_target_context(
    runtime: HccActionCeilingRuntime,
    *,
    problem_id: str,
):
    incumbent = np.ones(1000)
    incumbent[4:6] = 0.0
    return runtime.capture(
        action_set=_action_set(),
        cohort="real_aob",
        problem_id=problem_id,
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


def _target_runtime(
    tmp_path: Path,
    *,
    problem_id: str,
    sepcmaes_factory,
) -> HccActionCeilingRuntime:
    contract = action_ceiling_capture_contract(
        RS_FAMILY_TARGET_PROFILE,
        problem_id,
    )
    return HccActionCeilingRuntime(
        benchmark_factory=FakeBenchmark,
        cmaes_factory=FakeOptimizer,
        sepcmaes_factory=sepcmaes_factory,
        combine=lambda x, background, dims: runner.combine(x, background, dims),
        derive_seed=runner.derive_optimizer_seed,
        fun_name="rastrigin" if problem_id.startswith("R") else "schwefel",
        fun_id=int(problem_id[1:]),
        output_path=tmp_path,
        data_root=tmp_path,
        sigma=0.5,
        cmaes_restart=True,
        early_stopping_evaluations=1000,
        lower=-100.0,
        upper=100.0,
        dimension=1000,
        capture_arms=contract.arms,
        artifact_protocol_version=contract.protocol_version,
    )


def _budget_pulse_runtime(
    tmp_path: Path,
    *,
    sepcmaes_factory,
) -> HccActionCeilingRuntime:
    contract = action_ceiling_capture_contract(
        S_FAMILY_BUDGET_PULSE_PROFILE,
        "S5",
    )
    return HccActionCeilingRuntime(
        benchmark_factory=FakeBenchmark,
        cmaes_factory=FakeOptimizer,
        sepcmaes_factory=sepcmaes_factory,
        combine=lambda x, background, dims: runner.combine(x, background, dims),
        derive_seed=runner.derive_optimizer_seed,
        fun_name="schwefel",
        fun_id=5,
        output_path=tmp_path,
        data_root=tmp_path,
        sigma=0.5,
        cmaes_restart=True,
        early_stopping_evaluations=1000,
        lower=-100.0,
        upper=100.0,
        dimension=1000,
        capture_arms=contract.arms,
        artifact_protocol_version=contract.protocol_version,
    )


@pytest.mark.parametrize(
    ("capture_arms", "protocol_version", "fun_name"),
    [
        (("gcb", "native_eq8"), ACTION_CEILING_PROTOCOL_VERSION, "rastrigin"),
        (("native_eq8", "native_eq8"), ACTION_CEILING_PROTOCOL_VERSION, "rastrigin"),
        (("native_eq8", "unknown_arm"), ACTION_CEILING_PROTOCOL_VERSION, "rastrigin"),
        (("native_eq8",), RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION, "rastrigin"),
        (RS_FAMILY_RASTRIGIN_ARMS, RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION, "schwefel"),
        (RS_FAMILY_SCHWEFEL_ARMS, RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION, "rastrigin"),
        (S_FAMILY_BUDGET_PULSE_ARMS, S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION, "rastrigin"),
        (RS_FAMILY_SCHWEFEL_ARMS, S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION, "schwefel"),
    ],
)
def test_runtime_rejects_invalid_contract_combinations(
    tmp_path: Path,
    capture_arms: tuple[str, ...],
    protocol_version: str,
    fun_name: str,
) -> None:
    with pytest.raises(ValueError):
        HccActionCeilingRuntime(
            benchmark_factory=FakeBenchmark,
            cmaes_factory=FakeOptimizer,
            sepcmaes_factory=runner.SEPCMAES,
            combine=lambda x, background, dims: runner.combine(x, background, dims),
            derive_seed=runner.derive_optimizer_seed,
            fun_name=fun_name,
            fun_id=2,
            output_path=tmp_path,
            data_root=tmp_path,
            sigma=0.5,
            cmaes_restart=True,
            early_stopping_evaluations=1000,
            lower=-100.0,
            upper=100.0,
            dimension=1000,
            capture_arms=capture_arms,
            artifact_protocol_version=protocol_version,
        )


def test_action_ceiling_profiles_freeze_default_and_family_target_arms() -> None:
    full = action_ceiling_capture_contract(
        ACTION_CEILING_FULL_MATRIX_PROFILE,
        "E3",
    )
    rastrigin = action_ceiling_capture_contract(RS_FAMILY_TARGET_PROFILE, "R2")
    schwefel = action_ceiling_capture_contract(RS_FAMILY_TARGET_PROFILE, "S6")
    budget_pulse = action_ceiling_capture_contract(
        S_FAMILY_BUDGET_PULSE_PROFILE,
        "S5",
    )

    assert full.arms == ACTION_CEILING_ARMS
    assert ACTION_CEILING_KNOWN_ARMS[: len(ACTION_CEILING_ARMS)] == (
        ACTION_CEILING_ARMS
    )
    assert len(full.arms) == 13
    assert full.protocol_version == "exp019-action-ceiling-v7"
    assert rastrigin.arms == RS_FAMILY_RASTRIGIN_ARMS
    assert schwefel.arms == RS_FAMILY_SCHWEFEL_ARMS
    assert budget_pulse.arms == S_FAMILY_BUDGET_PULSE_ARMS
    assert rastrigin.protocol_version == RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
    assert schwefel.protocol_version == RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
    assert budget_pulse.protocol_version == S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION
    assert action_ceiling_capture_contract(RS_FAMILY_TARGET_PROFILE, "r2") == (
        rastrigin
    )
    assert action_ceiling_capture_contract(
        S_FAMILY_BUDGET_PULSE_PROFILE,
        "s5",
    ) == budget_pulse
    for invalid_problem_id in ("R0", "R7", "R01", "S", "E3"):
        with pytest.raises(ValueError):
            action_ceiling_capture_contract(
                RS_FAMILY_TARGET_PROFILE,
                invalid_problem_id,
            )
    for invalid_problem_id in ("S0", "S7", "S01", "R3", "A4"):
        with pytest.raises(ValueError):
            action_ceiling_capture_contract(
                S_FAMILY_BUDGET_PULSE_PROFILE,
                invalid_problem_id,
            )

    frozen_observation = ActionCeilingObservation(
        context_id="context",
        cohort="real_aob",
        problem_id="S6",
        seed=117,
        arm=RS_FAMILY_SCHWEFEL_ARMS[1],
        horizon="sweep_1",
        delta=0.1,
        selector_arm="native_eq8",
    )
    assert frozen_observation.arm == RS_FAMILY_SCHWEFEL_ARMS[1]


def test_schwefel_target_capture_never_constructs_sep_cma(tmp_path: Path) -> None:
    def forbidden_sep_cma(*args, **kwargs):
        raise AssertionError("Schwefel target capture constructed Sep-CMA")

    captured = _capture_target_context(
        _target_runtime(
            tmp_path,
            problem_id="S6",
            sepcmaes_factory=forbidden_sep_cma,
        ),
        problem_id="S6",
    )

    assert {row["arm"] for row in captured.arm_rows} == set(
        RS_FAMILY_SCHWEFEL_ARMS
    )
    assert len(captured.arm_rows) == 2 * len(ACTION_CEILING_HORIZONS)
    assert captured.context_row["protocol_version"] == (
        RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
    )
    assert all(
        captured.context_row[field] == ""
        for field in (
            "gcb_action_hash",
            "gcb_action_payload",
            "gcb_initial_mean_hash",
            "gcb_parameter_hash",
            "gcb_optimizer_seed",
            "gcb_population_size",
            "gcb_budget_fes",
            "gcb_acceptance_fitness",
        )
    )
    assert all(
        row["protocol_version"] == RS_FAMILY_ACTION_CEILING_PROTOCOL_VERSION
        for row in captured.arm_rows
    )
    budget_rows = [
        row
        for row in captured.arm_rows
        if row["arm"] == RS_FAMILY_SCHWEFEL_ARMS[1]
    ]
    lifecycle = json.loads(budget_rows[0]["action_lifecycle_payload"])
    assert lifecycle["instance"]["checkpoint_fe"] == 120
    assert lifecycle["instance"]["checkpoint_fe"] != int(
        captured.context_row["phase_boundary_fe"]
    )
    native_post_hashes = {
        row["horizon"]: row["action_post_incumbent_hash"]
        for row in captured.arm_rows
        if row["arm"] == "native_eq8"
    }
    assert all(
        row["action_post_incumbent_hash"] == native_post_hashes[row["horizon"]]
        for row in budget_rows
    )


def test_s_budget_pulse_capture_pairs_raw_and_shrunk_typed_actions(
    tmp_path: Path,
) -> None:
    def forbidden_sep_cma(*args, **kwargs):
        raise AssertionError("S budget pulse capture constructed Sep-CMA")

    captured = _capture_target_context(
        _budget_pulse_runtime(
            tmp_path,
            sepcmaes_factory=forbidden_sep_cma,
        ),
        problem_id="S5",
    )

    assert captured.context_row["protocol_version"] == (
        S_FAMILY_BUDGET_PULSE_PROTOCOL_VERSION
    )
    assert {row["arm"] for row in captured.arm_rows} == set(
        S_FAMILY_BUDGET_PULSE_ARMS
    )
    assert len(captured.arm_rows) == (
        len(S_FAMILY_BUDGET_PULSE_ARMS) * len(ACTION_CEILING_HORIZONS)
    )

    lifecycle_by_arm = {}
    for arm in (
        RS_FAMILY_SCHWEFEL_ARMS[1],
        SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
    ):
        rows = [row for row in captured.arm_rows if row["arm"] == arm]
        assert len({row["context_id"] for row in rows}) == 1
        assert len({row["action_instance_hash"] for row in rows}) == 1
        assert len({row["action_lifecycle_payload"] for row in rows}) == 1
        lifecycle_by_arm[arm] = json.loads(rows[0]["action_lifecycle_payload"])

    raw = lifecycle_by_arm[RS_FAMILY_SCHWEFEL_ARMS[1]]
    shrunk = lifecycle_by_arm[SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION]
    assert raw["instance_hash"] != shrunk["instance_hash"]
    assert raw["instance"]["dispatch_checkpoint_hash"] == (
        captured.context_row["dispatch_checkpoint_hash"]
    )
    assert shrunk["instance"]["dispatch_checkpoint_hash"] == (
        captured.context_row["dispatch_checkpoint_hash"]
    )
    assert shrunk["instance"]["raw_group_budgets"] == raw["instance"][
        "group_budgets"
    ]
    assert tuple(shrunk["instance"]["group_budgets"]) == (
        allocate_shrunk_efficiency_budgets(
            raw["instance"]["group_budgets"],
            shrunk["instance"]["uniform_group_budgets"],
            shrunk["instance"]["population_sizes"],
        )
    )
    for lifecycle in (raw, shrunk):
        assert lifecycle["instance"]["issued_sweep"] == 3
        assert lifecycle["instance"]["target_sweep"] == 4
        assert lifecycle["instance"]["ttl_sweeps"] == 1
        assert lifecycle["execution"]["status"] == "consumed"
        assert lifecycle["execution"]["consumed_sweep"] == 4
        assert lifecycle["execution"]["applied_group_budgets"] == lifecycle[
            "instance"
        ]["group_budgets"]
        assert lifecycle["execution"]["application_fe"] == (
            int(captured.context_row["dispatch_fe"]) + 1
        )

    shrunk_sweep_3 = next(
        row
        for row in captured.arm_rows
        if row["arm"] == SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION
        and row["horizon"] == "sweep_3"
    )
    budget_trace = json.loads(shrunk_sweep_3["group_budget_trace"])
    order_trace = json.loads(shrunk_sweep_3["execution_order_trace"])
    sweep_trace = json.loads(shrunk_sweep_3["execution_sweep_trace"])
    target_budgets = tuple(shrunk["instance"]["group_budgets"])
    uniform_budgets = tuple(shrunk["instance"]["uniform_group_budgets"])
    assert tuple(budget_trace[:2]) == target_budgets
    assert sweep_trace[:2] == [4, 4]
    assert all(
        budget == uniform_budgets[group]
        for budget, group, sweep in zip(
            budget_trace[2:],
            order_trace[2:],
            sweep_trace[2:],
            strict=True,
        )
        if sweep > 4
    )


def test_capture_exposes_incumbent_after_full_native_continuation(
    tmp_path: Path,
) -> None:
    captured = _capture_target_context(
        replace(
            _target_runtime(
                tmp_path,
                problem_id="S6",
                sepcmaes_factory=runner.SEPCMAES,
            ),
            cmaes_factory=ProgressiveGroupOptimizer,
        ),
        problem_id="S6",
    )

    assert captured.expected_native_incumbent[:4] == (1.0, 1.0, 1.0, 1.0)
    assert captured.expected_native_continuation_incumbent[:4] == (
        0.125,
        0.125,
        0.125,
        0.125,
    )


def test_rastrigin_target_capture_constructs_sep_cma_once(tmp_path: Path) -> None:
    sep_cma_calls = 0

    def recording_sep_cma(*args, **kwargs):
        nonlocal sep_cma_calls
        sep_cma_calls += 1
        return runner.SEPCMAES(*args, **kwargs)

    captured = _capture_target_context(
        _target_runtime(
            tmp_path,
            problem_id="R2",
            sepcmaes_factory=recording_sep_cma,
        ),
        problem_id="R2",
    )

    assert sep_cma_calls == 1
    assert {row["arm"] for row in captured.arm_rows} == set(
        RS_FAMILY_RASTRIGIN_ARMS
    )
    assert captured.context_row["gcb_action_hash"]


def test_full_and_target_rastrigin_share_checkpoint_and_sep_seed(
    tmp_path: Path,
) -> None:
    target_runtime = _target_runtime(
        tmp_path,
        problem_id="R2",
        sepcmaes_factory=runner.SEPCMAES,
    )
    full_runtime = replace(
        target_runtime,
        capture_arms=ACTION_CEILING_ARMS,
        artifact_protocol_version=ACTION_CEILING_PROTOCOL_VERSION,
    )

    target = _capture_target_context(target_runtime, problem_id="R2")
    full = _capture_target_context(full_runtime, problem_id="R2")

    assert target.context_row["dispatch_checkpoint_hash"] == (
        full.context_row["dispatch_checkpoint_hash"]
    )
    assert target.context_row["gcb_optimizer_seed"] == (
        full.context_row["gcb_optimizer_seed"]
    )


def test_capture_rejects_problem_id_mismatch(tmp_path: Path) -> None:
    runtime = _target_runtime(
        tmp_path,
        problem_id="R2",
        sepcmaes_factory=runner.SEPCMAES,
    )

    with pytest.raises(ValueError, match="problem id does not match"):
        _capture_target_context(runtime, problem_id="R3")


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
    action_set = _action_set()
    captured = runtime.capture(
        action_set=action_set,
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

    legacy_checkpoint_payload = {
        "problem_id": "E3",
        "seed": 117,
        "dispatch_fe": 120,
        "outer_iter": 3,
        "group_index": 1,
        "relation": {"owners": (0, 1), "shared": (4, 5)},
        "incumbent_hash": runner._canonical_payload_sha256(incumbent.tolist()),
        "fitness_prefix_hash": runner._canonical_payload_sha256(
            (1200.0, 1000.0, 998.0)
        ),
        "topology_hash": _hash("topology"),
        "order_hash": _hash("order"),
        "action_set_hash": action_set.action_set_hash,
        "previous_values": [1.0, 1.0],
        "current_values": [0.0, 0.0],
        "previous_delta": 1.0,
        "current_delta": 2.0,
        "completed_group_deltas": [1.0, 2.0],
        "completed_group_actual_fes": [13, 13],
        "efficiency_ewma": [0.1, 0.2],
        "completed_efficiency_sweeps": 3,
        "stagnation_streaks": [0, 0],
    }
    assert captured.context_row["dispatch_checkpoint_hash"] == (
        runner._canonical_payload_sha256(legacy_checkpoint_payload)
    )
    assert captured.context_row["status"] == "pending_native_parity"
    assert captured.context_row["selector_arm"] == "exact_bridge"
    assert len(captured.expected_native_record) == 78
    assert captured.expected_native_continuation_sweep_trace == (
        4,
        4,
        5,
        5,
        6,
        6,
    )
    assert captured.expected_native_continuation_order_trace == (0, 1) * 3
    assert captured.expected_native_continuation_budget_trace == (12, 12) * 3
    assert captured.expected_native_continuation_start_fe_trace == (
        1,
        14,
        27,
        40,
        53,
        66,
    )
    assert captured.expected_native_continuation_incumbent_hash == (
        runner._canonical_payload_sha256(
            list(captured.expected_native_continuation_incumbent)
        )
    )
    assert len(captured.arm_rows) == len(ACTION_CEILING_ARMS) * len(ACTION_CEILING_HORIZONS)
    assert {
        row["action_actual_fes"]
        for row in captured.arm_rows
        if row["arm"] not in {GCB_ACTION, "guarded_eq8_writeback"}
    } == {"0"}
    assert {
        row["action_actual_fes"]
        for row in captured.arm_rows
        if row["arm"] == "guarded_eq8_writeback"
    } == {"2"}
    assert {
        row["action_actual_fes"]
        for row in captured.arm_rows
        if row["arm"] == GCB_ACTION
    } == {"26"}
    assert all(row["runtime_authorized"] == "0" for row in captured.arm_rows)
    assert all(
        json.loads(row["execution_order_trace"])
        for row in captured.arm_rows
        if not (
            (
                row["arm"] == GCB_ACTION
                and row["horizon"] in {"immediate", "sweep_1"}
            )
            or (row["arm"] == "guarded_eq8_writeback" and row["horizon"] == "immediate")
        )
    )
    assert all(
        json.loads(row["group_budget_trace"])
        for row in captured.arm_rows
        if not (
            (
                row["arm"] == GCB_ACTION
                and row["horizon"] in {"immediate", "sweep_1"}
            )
            or (row["arm"] == "guarded_eq8_writeback" and row["horizon"] == "immediate")
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
        "guarded_eq8_writeback": "1",
        "stagnation_guard_writeback": "1",
        "contribution_owner_writeback": "0",
        "contribution_owner_reverse_writeback": "1",
        "efficiency_budget_reallocation": "1",
        "delta_priority_scan": "1",
        "stagnation_cross_group_warm_start": "1",
        GCB_ACTION: "1",
    }
    no_trigger_warm_start = [
        row
        for row in captured.arm_rows
        if row["arm"] == "stagnation_cross_group_warm_start"
    ]
    assert all(row["continuation_policy_applied"] == "0" for row in no_trigger_warm_start)
    assert all(float(row["delta"]) == 0.0 for row in no_trigger_warm_start)


def test_native_continuation_parity_covers_full_three_sweep_horizon() -> None:
    relation = RelationKey((0, 1), (4, 5))
    expected_record = tuple(float(value) for value in range(78))
    incumbent = tuple(float(value) for value in range(8))
    pending = runner.PendingActionCeilingParity(
        relation=relation,
        expected_sweep=6,
        start_fe=120,
        continuation_fe=78,
        expected_record=expected_record,
        expected_incumbent=incumbent,
        expected_incumbent_hash=_hash("incumbent"),
        expected_sweep_trace=(4, 4, 5, 5, 6, 6),
        expected_order_trace=(0, 1) * 3,
        expected_budget_trace=(12, 12) * 3,
        expected_start_fe_trace=(1, 14, 27, 40, 53, 66),
        context_row={},
        arm_rows=[],
        actual_sweep_trace=[4, 4, 5, 5, 6, 6],
        actual_order_trace=[0, 1] * 3,
        actual_budget_trace=[12, 12] * 3,
        actual_start_fe_trace=[1, 14, 27, 40, 53, 66],
    )
    fitness_record = [999.0] * 120 + list(expected_record)

    actual_record, reason = (
        runner._action_ceiling_native_continuation_prewriteback_parity(
            pending,
            relation=relation,
            current_sweep=6,
            current_fe=198,
            fitness_record=fitness_record,
        )
    )
    assert actual_record == expected_record
    assert reason == ""

    fitness_record[120 + 40] = -1.0
    _, reason = runner._action_ceiling_native_continuation_prewriteback_parity(
        pending,
        relation=relation,
        current_sweep=6,
        current_fe=198,
        fitness_record=fitness_record,
    )
    assert reason == "native_prefix_parity_mismatch"
    fitness_record[120 + 40] = expected_record[40]

    pending.actual_order_trace[4] = 1
    _, reason = runner._action_ceiling_native_continuation_prewriteback_parity(
        pending,
        relation=relation,
        current_sweep=6,
        current_fe=198,
        fitness_record=fitness_record,
    )
    assert reason == "native_dispatch_trace_parity_mismatch"
    pending.actual_order_trace[4] = 0

    fitness_record.append(1.0)
    _, reason = runner._action_ceiling_native_continuation_prewriteback_parity(
        pending,
        relation=relation,
        current_sweep=6,
        current_fe=199,
        fitness_record=fitness_record,
    )
    assert reason == "native_horizon_fe_parity_mismatch"

    _, reason = runner._action_ceiling_native_continuation_prewriteback_parity(
        pending,
        relation=RelationKey((1, 2), (5, 6)),
        current_sweep=6,
        current_fe=198,
        fitness_record=fitness_record,
    )
    assert reason == "native_relation_parity_mismatch"


def test_gcb_strict_gate_and_frozen_fe_cost(tmp_path: Path) -> None:
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
            if row["arm"] == GCB_ACTION
        ]
        sweep_3 = next(row for row in sep_rows if row["horizon"] == "sweep_3")
        candidate_fitness = float(sweep_3["action_candidate_fitness"])

        assert float(
            captured.context_row["gcb_acceptance_fitness"]
        ) == acceptance_fitness
        assert captured.context_row["gcb_initial_mean_hash"] == (
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
            else captured.context_row["gcb_initial_mean_hash"]
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
    assert len(captured.expected_native_record) == 78


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


def test_action_ceiling_cli_requires_explicit_capture_flag(tmp_path: Path) -> None:
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

    target_common = common.copy()
    target_common[1] = "rastrigin"
    target_common[3] = "2"
    target_args = runner.parse_args(
        target_common
        + [
            "--action-ceiling-capture",
            "--action-ceiling-profile",
            RS_FAMILY_TARGET_PROFILE,
        ]
    )
    assert target_args.action_ceiling_profile == RS_FAMILY_TARGET_PROFILE

    s_common = common.copy()
    s_common[1] = "schwefel"
    s_common[3] = "5"
    s_args = runner.parse_args(
        s_common
        + [
            "--action-ceiling-capture",
            "--action-ceiling-profile",
            S_FAMILY_BUDGET_PULSE_PROFILE,
        ]
    )
    assert s_args.action_ceiling_profile == S_FAMILY_BUDGET_PULSE_PROFILE

    with pytest.raises(SystemExit) as error:
        runner.parse_args(
            target_common
            + [
                "--action-ceiling-capture",
                "--action-ceiling-profile",
                S_FAMILY_BUDGET_PULSE_PROFILE,
            ]
        )
    assert error.value.code == 2

    with pytest.raises(SystemExit) as error:
        runner.parse_args(
            s_common
            + [
                "--action-ceiling-capture",
                "--action-ceiling-profile",
                S_FAMILY_BUDGET_PULSE_PROFILE,
                "--action-ceiling-cohort",
                "synthetic_conflict",
            ]
        )
    assert error.value.code == 2

    with pytest.raises(SystemExit) as error:
        runner.parse_args(
            target_common
            + [
                "--action-ceiling-capture",
                "--action-ceiling-profile",
                RS_FAMILY_TARGET_PROFILE,
                "--action-ceiling-cohort",
                "synthetic_conflict",
            ]
        )
    assert error.value.code == 2

    non_capture = target_common.copy()
    non_capture[non_capture.index(runner.ACTION_CEILING_POLICY)] = "controller_v31"
    non_capture.extend(
        ["--action-ceiling-profile", RS_FAMILY_TARGET_PROFILE]
    )
    with pytest.raises(SystemExit) as error:
        runner.parse_args(non_capture)
    assert error.value.code == 2

    synthetic_config = runner.SmokeConfig(
        max_fes=100_000,
        seed=117,
        arac_action=runner.EVIDENCE_ACTION_CONTROLLER_V37,
        enable_relation_dispatch=True,
        relation_policy_mode=runner.ACTION_CEILING_POLICY,
        evidence_overlay_mode="paired_owner",
        action_ceiling_capture=True,
        action_ceiling_cohort="synthetic_conflict",
        action_ceiling_profile=RS_FAMILY_TARGET_PROFILE,
    )
    with pytest.raises(ValueError, match="requires the real_aob cohort"):
        runner.run_problem("rastrigin", 2, tmp_path, synthetic_config)
