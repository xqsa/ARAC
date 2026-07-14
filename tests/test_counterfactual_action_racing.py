from __future__ import annotations

import pytest

from arac.policy.counterfactual_action_racing import (
    AuditEnvelope,
    BranchState,
    CARBudgetLedger,
    CARInvariantError,
    CARProbeExecutor,
    DispatchEvidence,
    PairedProbeObservation,
    adopt_final_pair_branch,
    derive_probe_seed,
    evaluate_risk_gate,
    fingerprint_branch_state,
)


def make_observation(
    pair_index: int,
    *,
    start: float = 100.0,
    fallback_after: float = 90.0,
    candidate_after: float = 80.0,
    fallback_fe: int = 12,
    candidate_fe: int = 12,
    graph_fingerprint: str = "graph-a",
    component_fingerprint: str = "component-a",
    action_family: str = "coordinate",
    fallback_start_state_fingerprint: str | None = None,
    candidate_start_state_fingerprint: str | None = None,
) -> PairedProbeObservation:
    start_fingerprint = f"state-{pair_index}"
    return PairedProbeObservation.create(
        pair_index=pair_index,
        phase1_probe_fitness_before=start,
        fallback_after=fallback_after,
        candidate_after=candidate_after,
        fallback_fe=fallback_fe,
        candidate_fe=candidate_fe,
        graph_fingerprint=graph_fingerprint,
        component_fingerprint=component_fingerprint,
        action_family=action_family,
        fallback_start_state_fingerprint=(
            fallback_start_state_fingerprint or start_fingerprint
        ),
        candidate_start_state_fingerprint=(
            candidate_start_state_fingerprint or start_fingerprint
        ),
        fallback_evaluator_id=f"fallback-{pair_index}",
        candidate_evaluator_id=f"candidate-{pair_index}",
        seed_descriptor=derive_probe_seed(
            base_seed=17,
            sweep_index=2,
            component_fingerprint=component_fingerprint,
            pair_index=pair_index,
        ),
    )


def test_dispatch_evidence_rejects_identity_and_outcome_fields() -> None:
    valid = {
        "graph_fingerprint": "graph-a",
        "component_fingerprint": "component-a",
        "candidate_action_name": "allow_beneficial_coordination",
        "candidate_action_family": "coordinate",
        "overlap_strength": 0.75,
        "shared_variable_count": 3,
        "evidence_sweep_count": 2,
        "evidence_coverage": 1.0,
        "writeback_norm": 0.25,
    }

    evidence = DispatchEvidence.from_runtime_payload(valid)

    assert evidence.evidence_sweep_count == 2
    assert not hasattr(evidence, "problem_id")
    assert not hasattr(evidence, "final_outcome")
    for forbidden in (
        "case_label",
        "problem_id",
        "function_family",
        "paper_best",
        "historical_best",
        "final_outcome",
        "seed",
    ):
        with pytest.raises(CARInvariantError, match="forbidden runtime field"):
            DispatchEvidence.from_runtime_payload({**valid, forbidden: "leak"})


def test_audit_identity_is_kept_outside_dispatch_evidence() -> None:
    envelope = AuditEnvelope(run_id="run-1", problem_id="E2", seed=3)

    assert envelope.problem_id == "E2"
    assert set(DispatchEvidence.runtime_field_names()).isdisjoint(
        {"run_id", "problem_id", "seed"}
    )
    assert set(DispatchEvidence.runtime_field_names()).isdisjoint(
        DispatchEvidence.forbidden_field_names()
    )
    assert {"run_id", "problem_id", "seed"}.issubset(
        DispatchEvidence.forbidden_field_names()
    )


def test_budget_ledger_charges_both_arms_and_prevents_overrun() -> None:
    ledger = CARBudgetLedger(max_fes=5_000, probe_fe_limit=150, committed_fe=4_700)

    ledger.charge_pair(pair_index=0, fallback_fe=20, candidate_fe=20)

    assert ledger.probe_fe == 40
    assert ledger.total_fe == 4_740
    assert ledger.remaining_probe_fe == 110
    with pytest.raises(CARInvariantError, match="equal actual FE"):
        ledger.charge_pair(pair_index=1, fallback_fe=20, candidate_fe=19)
    with pytest.raises(CARInvariantError, match="probe FE limit"):
        ledger.charge_pair(pair_index=1, fallback_fe=60, candidate_fe=60)
    with pytest.raises(CARInvariantError, match="total FE budget"):
        ledger.charge_committed(stage="phase_ii", actual_fe=300)


def test_probe_seed_is_deterministic_and_shared_by_both_arms() -> None:
    first = derive_probe_seed(
        base_seed=23,
        sweep_index=2,
        component_fingerprint="component-a",
        pair_index=1,
    )
    replay = derive_probe_seed(
        base_seed=23,
        sweep_index=2,
        component_fingerprint="component-a",
        pair_index=1,
    )
    different_pair = derive_probe_seed(
        base_seed=23,
        sweep_index=2,
        component_fingerprint="component-a",
        pair_index=2,
    )

    assert first == replay
    assert first.seed == replay.seed
    assert first != different_pair
    assert "arm" not in first.canonical_key
    assert "problem" not in first.canonical_key


def test_pair_creation_rejects_unequal_fe_and_mismatched_start_state() -> None:
    with pytest.raises(CARInvariantError, match="equal actual FE"):
        make_observation(0, fallback_fe=12, candidate_fe=11)

    with pytest.raises(CARInvariantError, match="identical checkpoint"):
        make_observation(
            0,
            fallback_start_state_fingerprint="fallback-state",
            candidate_start_state_fingerprint="candidate-state",
        )


def test_k3_empirical_gate_commits_only_the_predeclared_final_candidate() -> None:
    observations = (
        make_observation(0, fallback_after=91.0, candidate_after=88.0),
        make_observation(1, fallback_after=84.0, candidate_after=80.0),
        make_observation(2, fallback_after=78.0, candidate_after=72.0),
    )

    gate = evaluate_risk_gate(observations, epsilon=1e-12)

    assert gate.committed is True
    assert gate.adopted_arm == "candidate"
    assert gate.tail == min(item.normalized_delta for item in observations)
    assert gate.lcb > 0.0


def test_k3_empirical_gate_abstains_on_negative_lower_tail() -> None:
    observations = (
        make_observation(0, fallback_after=91.0, candidate_after=88.0),
        make_observation(1, fallback_after=84.0, candidate_after=80.0),
        make_observation(2, fallback_after=72.0, candidate_after=74.0),
    )

    gate = evaluate_risk_gate(observations, epsilon=1e-12)

    assert gate.committed is False
    assert gate.adopted_arm == "fallback"
    assert gate.tail < 0.0
    assert any(reason.startswith("lower_tail") for reason in gate.abstain_reasons)


def test_gate_abstains_when_action_family_or_graph_is_unstable() -> None:
    family_gate = evaluate_risk_gate(
        (
            make_observation(0),
            make_observation(1),
            make_observation(2, action_family="reassign_repair"),
        )
    )
    graph_gate = evaluate_risk_gate(
        (
            make_observation(0),
            make_observation(1),
            make_observation(2, graph_fingerprint="graph-b"),
        )
    )

    assert "unstable_action_family" in family_gate.abstain_reasons
    assert "unstable_graph_fingerprint" in graph_gate.abstain_reasons


def test_gate_requires_exactly_three_pairs() -> None:
    with pytest.raises(CARInvariantError, match="exactly 3"):
        evaluate_risk_gate((make_observation(0), make_observation(1)))


def test_discarded_branch_does_not_pollute_adopted_state_or_record() -> None:
    fallback = BranchState(
        incumbent=(1.0, 2.0),
        committed_fitness=90.0,
        evaluator_record=[100.0, 90.0],
        state_fingerprint="fallback-final",
        state_payload={"rng": {"counter": 1}},
    )
    candidate = BranchState(
        incumbent=(3.0, 4.0),
        committed_fitness=80.0,
        evaluator_record=[100.0, 80.0],
        state_fingerprint="candidate-final",
        state_payload={"rng": {"counter": 1}},
    )
    gate = evaluate_risk_gate(
        (make_observation(0), make_observation(1), make_observation(2))
    )

    adopted = adopt_final_pair_branch(
        gate=gate,
        fallback=fallback,
        candidate=candidate,
    )
    fallback.evaluator_record.append(1.0)
    candidate.evaluator_record.append(2.0)

    assert adopted.committed_fitness == 80.0
    assert adopted.evaluator_record == [100.0, 80.0]
    assert adopted.evaluator_record is not candidate.evaluator_record


class FakeEvaluator:
    def __init__(self) -> None:
        self.fitness_record: list[float] = []

    def evaluate(self, values: list[float]) -> None:
        self.fitness_record.extend(values)


def make_checkpoint(fitness: float = 100.0) -> BranchState:
    checkpoint = BranchState(
        incumbent=(fitness, 0.0),
        committed_fitness=fitness,
        evaluator_record=[],
        state_fingerprint="",
        state_payload={"rng": {"counter": 0}, "cache": [1.0, 2.0]},
    )
    checkpoint.state_fingerprint = fingerprint_branch_state(checkpoint)
    return checkpoint


def deterministic_transition(improvement: float, calls: list[tuple[str, int]]):
    def transition(
        checkpoint: BranchState,
        evaluator: FakeEvaluator,
        seed_descriptor,
        requested_fes: int,
    ) -> BranchState:
        calls.append((seed_descriptor.canonical_key, seed_descriptor.seed))
        after = checkpoint.committed_fitness - improvement
        evaluator.evaluate([checkpoint.committed_fitness] * (requested_fes - 1) + [after])
        state = BranchState(
            incumbent=(after, improvement),
            committed_fitness=after,
            evaluator_record=list(evaluator.fitness_record),
            state_fingerprint="",
            state_payload={**checkpoint.state_payload, "probe_seed": seed_descriptor.seed},
        )
        state.state_fingerprint = fingerprint_branch_state(state)
        return state

    return transition


@pytest.mark.parametrize(
    "branch_order",
    [("fallback", "candidate"), ("candidate", "fallback")],
)
def test_probe_executor_is_branch_order_invariant(branch_order) -> None:
    calls: list[tuple[str, int]] = []
    ledger = CARBudgetLedger(max_fes=1_000, probe_fe_limit=120, committed_fe=100)
    executor = CARProbeExecutor(
        evaluator_factory=FakeEvaluator,
        ledger=ledger,
        base_seed=31,
        sweep_index=2,
        graph_fingerprint="graph-a",
        component_fingerprint="component-a",
        action_family="coordinate",
        arm_fes=10,
    )

    result = executor.execute(
        initial_checkpoint=make_checkpoint(),
        fallback_transition=deterministic_transition(1.0, calls),
        candidate_transition=deterministic_transition(2.0, calls),
        branch_order=branch_order,
    )

    assert result.gate.committed is True
    assert result.adopted_state.committed_fitness == pytest.approx(96.0)
    assert ledger.probe_fe == 60
    assert len(result.observations) == 3
    assert len(result.branch_manifests) == 6
    assert len(result.accounting_record) == 60
    assert min(result.accounting_record) == pytest.approx(96.0)
    assert all(item.actual_fe == 10 for item in result.branch_manifests)
    for pair_offset in range(0, len(calls), 2):
        assert calls[pair_offset] == calls[pair_offset + 1]


def test_probe_executor_rejects_reused_evaluator_record() -> None:
    shared = FakeEvaluator()
    executor = CARProbeExecutor(
        evaluator_factory=lambda: shared,
        ledger=CARBudgetLedger(max_fes=1_000, probe_fe_limit=120),
        base_seed=31,
        sweep_index=2,
        graph_fingerprint="graph-a",
        component_fingerprint="component-a",
        action_family="coordinate",
        arm_fes=10,
    )

    with pytest.raises(CARInvariantError, match="fresh branch-local evaluator"):
        executor.execute(
            initial_checkpoint=make_checkpoint(),
            fallback_transition=deterministic_transition(1.0, []),
            candidate_transition=deterministic_transition(2.0, []),
        )


def test_probe_executor_rejects_incomplete_component_horizon() -> None:
    def short_transition(checkpoint, evaluator, seed_descriptor, requested_fes):
        evaluator.evaluate([checkpoint.committed_fitness] * (requested_fes - 1))
        state = checkpoint.clone()
        state.evaluator_record = list(evaluator.fitness_record)
        state.state_fingerprint = fingerprint_branch_state(state)
        return state

    ledger = CARBudgetLedger(max_fes=1_000, probe_fe_limit=120)
    executor = CARProbeExecutor(
        evaluator_factory=FakeEvaluator,
        ledger=ledger,
        base_seed=31,
        sweep_index=2,
        graph_fingerprint="graph-a",
        component_fingerprint="component-a",
        action_family="coordinate",
        arm_fes=10,
    )

    with pytest.raises(CARInvariantError, match="complete component horizon"):
        executor.execute(
            initial_checkpoint=make_checkpoint(),
            fallback_transition=short_transition,
            candidate_transition=short_transition,
        )
    assert ledger.probe_fe == 20


def test_branch_fingerprint_covers_rng_and_cache_payload() -> None:
    checkpoint = make_checkpoint()
    changed = checkpoint.clone()
    changed.state_payload["rng"]["counter"] = 1

    assert fingerprint_branch_state(changed) != checkpoint.state_fingerprint
