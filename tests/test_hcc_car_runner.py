from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arac.backends.hcc_car import CARPlanDecision, CARWritebackPlan
from arac.policy.counterfactual_action_racing import (
    BranchState,
    DispatchEvidence,
    fingerprint_branch_state,
)
from scripts import hcc_smoke_runner as runner


def make_decision() -> CARPlanDecision:
    plan = CARWritebackPlan(
        graph_fingerprint="graph-a",
        component_fingerprint="component-a",
        action_name="isolate_conflicting_relation",
        action_family="isolate",
        group_indices=(0, 1),
        group_dims=((0, 1), (2, 3)),
        group_population_sizes=(4, 6),
        shared_indices=(0,),
        target_values=(0.0,),
        lower=-5.0,
        upper=5.0,
        max_delta_norm=0.5,
    )
    evidence = DispatchEvidence(
        graph_fingerprint=plan.graph_fingerprint,
        component_fingerprint=plan.component_fingerprint,
        candidate_action_name=plan.action_name,
        candidate_action_family=plan.action_family,
        overlap_strength=1.0,
        shared_variable_count=1,
        evidence_sweep_count=2,
        evidence_coverage=1.0,
        writeback_norm=0.5,
    )
    return CARPlanDecision(plan=plan, evidence=evidence, abstain_reason="")


class CountingSphere:
    instances: list["CountingSphere"] = []

    def __init__(self) -> None:
        self.fitness_record: list[float] = []
        self.instances.append(self)

    def __call__(self, x_batch):
        values = np.asarray(x_batch, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        result = np.sum(np.square(values), axis=1)
        self.fitness_record.extend(result.tolist())
        return result


class FakeBenchmark:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_function(self, fun_name, fun_id):
        return CountingSphere()


class FakeCMAES:
    def __init__(self, problem, options) -> None:
        self.problem = problem
        self.options = options

    def optimize(self):
        requested = int(self.options["max_function_evaluations"])
        mean = np.asarray(self.options["mean"][0], dtype=float).reshape(-1)
        values = self.problem["fitness_function"](np.tile(mean, (requested, 1)))
        return {
            "best_so_far_x": mean.copy(),
            "best_so_far_y": float(np.min(values)),
            "n_function_evaluations": requested,
        }


def make_checkpoint() -> BranchState:
    state = BranchState(
        incumbent=(4.0, 1.0, 2.0, 3.0),
        committed_fitness=30.0,
        evaluator_record=[],
        state_fingerprint="",
        state_payload={"rng": {"counter": 3}, "cache": {}},
    )
    state.state_fingerprint = fingerprint_branch_state(state)
    return state


def test_car_action_is_registered_in_runner_cli() -> None:
    args = runner.parse_args(
        [
            "--functions",
            "elliptic",
            "--ids",
            "2",
            "--output-root",
            "results/car-cli",
            "--max-fes",
            "5000",
            "--arac-action",
            runner.CAR_W_ACTION,
            "--enable-relation-dispatch",
            "--relation-policy",
            "controller_v31",
        ]
    )

    assert args.arac_action == "arac_counterfactual_action_racing_w"


def test_car_run_rejects_missing_controller_v31_dispatch(tmp_path: Path) -> None:
    config = runner.SmokeConfig(
        max_fes=5_000,
        seed=1,
        arac_action=runner.CAR_W_ACTION,
        enable_relation_dispatch=False,
        relation_policy_mode="rule",
    )

    with pytest.raises(ValueError, match="requires relation dispatch"):
        runner.run_problem("elliptic", 2, tmp_path, config)


def test_car_barrier_abstains_when_total_remaining_fe_cannot_fit_pairs(
    tmp_path: Path,
) -> None:
    config = runner.SmokeConfig(
        max_fes=10_000,
        seed=7,
        arac_action=runner.CAR_W_ACTION,
        enable_relation_dispatch=True,
        relation_policy_mode="controller_v31",
        verbose=0,
    )

    result = runner.execute_car_w_probe_at_barrier(
        decision=make_decision(),
        checkpoint=make_checkpoint(),
        checkpoint_fe=9_950,
        fun_name="elliptic",
        fun_id=2,
        output_path=tmp_path,
        info={"lower": -5.0, "upper": 5.0},
        config=config,
        problem_id="E2",
    )

    assert result.adopted_state is None
    assert result.probe_fe == 0
    assert result.abstain_reason == (
        "remaining_total_budget_cannot_fit_complete_component_horizon"
    )


@pytest.mark.parametrize(
    "branch_order",
    [("fallback", "candidate"), ("candidate", "fallback")],
)
def test_runner_barrier_executes_isolated_equal_fe_pairs(
    monkeypatch,
    tmp_path: Path,
    branch_order: tuple[str, str],
) -> None:
    CountingSphere.instances = []
    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    config = runner.SmokeConfig(
        max_fes=10_000,
        seed=7,
        arac_action=runner.CAR_W_ACTION,
        enable_relation_dispatch=True,
        relation_policy_mode="controller_v31",
        verbose=0,
    )

    result = runner.execute_car_w_probe_at_barrier(
        decision=make_decision(),
        checkpoint=make_checkpoint(),
        checkpoint_fe=100,
        fun_name="elliptic",
        fun_id=2,
        output_path=tmp_path,
        info={"lower": -5.0, "upper": 5.0},
        config=config,
        problem_id="E2",
        branch_order=branch_order,
    )

    assert result.adopted_state is not None
    assert result.state_ledger_rows[0]["adopted_branch"] == "candidate"
    assert len(result.probe_trace_rows) == 3
    assert len(result.branch_manifest_rows) == 6
    assert len(CountingSphere.instances) == 6
    assert len({id(item.fitness_record) for item in CountingSphere.instances}) == 6
    assert len(result.accounting_record) == result.probe_fe
    assert all(
        row["fallback_fe"] == row["candidate_fe"]
        for row in result.probe_trace_rows
    )
    assert result.adopted_state.committed_fitness == pytest.approx(26.25)
    assert min(result.accounting_record) == pytest.approx(26.25)
