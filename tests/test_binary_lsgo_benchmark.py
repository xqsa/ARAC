from __future__ import annotations

from dataclasses import fields, replace

import pytest

from arac.benchmarks.binary_lsgo import (
    BinaryLsgoSpec,
    BinaryLsgoTopology,
    generate_binary_lsgo,
    standard_binary_lsgo_specs,
)


def test_spec_rejects_invalid_generation_inputs() -> None:
    with pytest.raises(ValueError, match="nominal_dimension"):
        BinaryLsgoSpec("bad", 4, 4, 1, 2, True, 0.1, 0.5, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="alpha"):
        BinaryLsgoSpec("bad", 10, 1, 2, 5, True, 0.9, 0.5, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="ratio"):
        BinaryLsgoSpec("bad", 10, 1, 2, 5, True, 0.1, 0.0, 0.5, 0.5, 1)


def test_spec_rejects_ambiguous_input_types() -> None:
    valid = BinaryLsgoSpec("valid", 10, 1, 2, 5, True, 0.1, 0.5, 0.5, 0.5, 1)

    with pytest.raises(ValueError, match="problem_id"):
        replace(valid, problem_id=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="continuous_groups"):
        replace(valid, continuous_groups=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        replace(valid, alpha="0.1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ratio"):
        replace(valid, related_group_ratio="0.5")  # type: ignore[arg-type]


def test_generated_problem_exposes_explicit_dimension_semantics() -> None:
    spec = BinaryLsgoSpec("small", 20, 4, 2, 5, True, 0.1, 0.5, 0.5, 0.5, 7)
    problem = generate_binary_lsgo(spec)

    assert problem.decision_dimension == 16
    assert problem.topology.nominal_dimension == 20
    assert problem.topology.membership_count == 20
    assert len(problem.template) == 16


def test_topology_has_valid_groups_and_exact_overlap_slots() -> None:
    problem = generate_binary_lsgo(
        BinaryLsgoSpec("topology", 40, 8, 2, 5, True, 0.5, 0.5, 0.5, 0.5, 11)
    )
    topology = problem.topology

    assert sum(topology.group_sizes) == 40
    assert all(2 <= size <= 5 for size in topology.group_sizes)
    assert topology.membership_count == 40
    assert topology.overlap_slot_count == 8
    assert sum(topology.variable_occurrence_counts.values()) == 40
    assert all(0 <= index < 32 for group in topology.groups for index in group)
    assert all(len(group) == len(set(group)) for group in topology.groups)
    assert set(topology.variable_occurrence_counts) == set(range(32))
    assert all(count >= 1 for count in topology.variable_occurrence_counts.values())
    assert topology.max_variable_occurrence_count >= 2


def test_equal_groups_and_continuous_order_are_preserved() -> None:
    problem = generate_binary_lsgo(
        BinaryLsgoSpec("equal", 20, 0, 5, 5, True, 0.1, 0.5, 0.5, 0.5, 3)
    )

    assert problem.topology.group_sizes == (5, 5, 5, 5)
    assert tuple(index for group in problem.topology.groups for index in group) == tuple(range(20))
    assert problem.topology.shared_variable_count == 0
    assert problem.topology.adjacency_pairs == ()


def test_shuffled_order_is_seeded_and_not_continuous() -> None:
    spec = BinaryLsgoSpec("shuffled", 20, 0, 5, 5, False, 0.1, 0.5, 0.5, 0.5, 19)

    first = generate_binary_lsgo(spec)
    second = generate_binary_lsgo(spec)
    flattened = tuple(index for group in first.topology.groups for index in group)

    assert first == second
    assert flattened != tuple(range(20))
    assert sorted(flattened) == list(range(20))


def test_spec_rejects_group_size_larger_than_decision_dimension() -> None:
    with pytest.raises(ValueError, match="decision_dimension"):
        BinaryLsgoSpec("impossible", 10, 8, 2, 5, True, 0.1, 0.5, 0.5, 0.5, 1)


def _reference_contribution(group_size: int, matching: int, alpha: float) -> float:
    if alpha == 0:
        return float(matching)
    local_optimum = 0.9 * group_size
    deception_point = 10 * alpha * group_size / 9
    if matching < deception_point:
        return -(local_optimum / deception_point) * matching + local_optimum
    return (
        (group_size / (group_size - deception_point)) * matching
        - (group_size * deception_point / (group_size - deception_point))
    )


@pytest.mark.parametrize("alpha", [0.1, 0.8])
def test_objective_matches_piecewise_deception_formula(alpha: float) -> None:
    problem = generate_binary_lsgo(
        BinaryLsgoSpec("objective", 20, 4, 2, 5, True, alpha, 0.5, 0.5, 0.5, 3)
    )
    assert problem.evaluate(problem.template) == pytest.approx(-20.0)

    complement = tuple(1 - bit for bit in problem.template)
    assert problem.evaluate(complement) == pytest.approx(-18.0)

    mutated = list(problem.template)
    mutated[0] = 1 - mutated[0]
    expected = 0.0
    for group in problem.topology.groups:
        matching = len(group) - (1 if 0 in group else 0)
        expected += _reference_contribution(len(group), matching, alpha)
    assert problem.evaluate(mutated) == pytest.approx(-expected)


def test_alpha_zero_uses_matching_count_without_deception() -> None:
    problem = generate_binary_lsgo(
        BinaryLsgoSpec("no-deception", 20, 0, 5, 5, True, 0.0, 0.5, 0.5, 0.5, 5)
    )
    complement = tuple(1 - bit for bit in problem.template)

    assert problem.evaluate(problem.template) == pytest.approx(-20.0)
    assert problem.evaluate(complement) == pytest.approx(0.0)


def test_objective_rejects_non_binary_or_wrong_length_vectors() -> None:
    problem = generate_binary_lsgo(
        BinaryLsgoSpec("validation", 20, 4, 2, 5, True, 0.1, 0.5, 0.5, 0.5, 4)
    )
    with pytest.raises(ValueError, match="length"):
        problem.evaluate((0, 1))
    with pytest.raises(ValueError, match="binary"):
        problem.evaluate((2,) + problem.template[1:])
    with pytest.raises(ValueError, match="binary"):
        problem.evaluate((0.0,) + problem.template[1:])


def test_batch_evaluation_matches_scalar_evaluation() -> None:
    problem = generate_binary_lsgo(
        BinaryLsgoSpec("batch", 20, 4, 2, 5, True, 0.5, 0.5, 0.5, 0.5, 6)
    )
    complement = tuple(1 - bit for bit in problem.template)

    assert problem.evaluate_batch([problem.template, complement]) == (
        problem.evaluate(problem.template),
        problem.evaluate(complement),
    )
    assert problem.evaluate_batch([]) == ()


def test_standard_suite_matches_the_inherited_18_case_matrix() -> None:
    specs = standard_binary_lsgo_specs()

    assert len(specs) == 18
    assert [spec.problem_id for spec in specs] == [f"BLSGO-F{index:02d}" for index in range(1, 19)]
    assert len({spec.problem_id for spec in specs}) == 18
    assert [spec.alpha for spec in specs] == [0.1] * 6 + [0.5] * 6 + [0.8] * 6
    assert [(spec.min_group_size, spec.max_group_size) for spec in specs] == (
        [(5, 5)] * 3
        + [(2, 5)] * 3
        + [(5, 5)] * 3
        + [(2, 5)] * 3
        + [(5, 5)] * 3
        + [(2, 5)] * 3
    )
    assert [spec.overlap_count for spec in specs] == [100, 200, 300] * 6
    assert [spec.decision_dimension for spec in specs] == [900, 800, 700] * 6
    assert [spec.seed for spec in specs] == list(range(1, 19))
    assert all(spec.nominal_dimension == 1000 for spec in specs)
    assert all(spec.continuous_groups for spec in specs)


def test_standard_suite_generates_valid_topologies() -> None:
    for spec in standard_binary_lsgo_specs():
        problem = generate_binary_lsgo(spec)
        assert problem.topology.membership_count == 1000
        assert problem.topology.overlap_slot_count == spec.overlap_count
        assert problem.decision_dimension == spec.decision_dimension


def test_generation_is_reproducible_and_seed_scoped() -> None:
    spec = BinaryLsgoSpec("seeded", 40, 8, 2, 5, False, 0.5, 0.5, 0.5, 0.5, 23)

    first = generate_binary_lsgo(spec)
    second = generate_binary_lsgo(spec)
    changed_seed = generate_binary_lsgo(replace(spec, seed=24))

    assert first == second
    assert (first.template, first.topology.groups) != (
        changed_seed.template,
        changed_seed.topology.groups,
    )


def test_topology_metadata_excludes_runtime_outcome_fields() -> None:
    field_names = {field.name for field in fields(BinaryLsgoTopology)}
    forbidden = {
        "final_error",
        "relative_gain",
        "oracle",
        "reported_baseline",
        "prior_final_outcome",
        "prior_pilot_outcome",
    }

    assert field_names.isdisjoint(forbidden)
