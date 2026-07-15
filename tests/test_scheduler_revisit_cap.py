from __future__ import annotations

import itertools
import math

from arac.policy.component_delayed_credit import (
    calculate_scheduler_revisit_cap,
)


def _bounded_budget(requested: int, remaining: int, population: int) -> int:
    usable = min(requested, remaining)
    return 0 if usable <= 0 else (usable // population) * population


def test_scheduler_cap_covers_e2_seed32_end_budget_state() -> None:
    cap = calculate_scheduler_revisit_cap(
        sweep_start_fe=2_935_070,
        decision_fe=2_987_567,
        cc_budget_limit_fe=3_000_000,
        current_group_index=18,
        current_sweep_group_budget_fe=3_247,
        current_optimizer_budget_fe=3_232,
        group_population_sizes=(16,) * 20,
    )

    assert cap.reachable is True
    assert cap.cap_fe == 11_871
    assert cap.current_tail_cap_fe == 6_465
    assert cap.next_sweep_min_group_budget_fe == 299
    assert cap.reason == "scheduler_revisit_cap_available"


def test_scheduler_cap_rejects_a_tail_that_can_exhaust_strict_guard() -> None:
    cap = calculate_scheduler_revisit_cap(
        sweep_start_fe=88,
        decision_fe=93,
        cc_budget_limit_fe=100,
        current_group_index=1,
        current_sweep_group_budget_fe=4,
        current_optimizer_budget_fe=4,
        group_population_sizes=(4, 4, 4),
    )

    assert cap.reachable is False
    assert cap.cap_fe is None
    assert cap.reason == "current_sweep_tail_not_guaranteed"


def test_scheduler_cap_bounds_all_discrete_early_stop_paths() -> None:
    populations = (4, 4, 4, 4)
    decision_fe = 60
    budget_limit = 100
    cap = calculate_scheduler_revisit_cap(
        sweep_start_fe=50,
        decision_fe=decision_fe,
        cc_budget_limit_fe=budget_limit,
        current_group_index=2,
        current_sweep_group_budget_fe=13,
        current_optimizer_budget_fe=12,
        group_population_sizes=populations,
    )
    observed_delays: list[int] = []

    for current_used, final_group_used in itertools.product((4, 8, 12), repeat=2):
        fe = decision_fe + current_used
        assert budget_limit - fe > populations[3]
        fe += 1 + final_group_used
        next_group_budget = math.ceil((budget_limit - fe) / len(populations))
        prefix_caps = [
            _bounded_budget(
                max(next_group_budget, populations[index]),
                budget_limit - fe - 1,
                populations[index],
            )
            for index in range(2)
        ]
        for prefix_used in itertools.product(
            *(
                range(population, cap_fe + 1, population)
                for population, cap_fe in zip(
                    populations[:2], prefix_caps, strict=True
                )
            )
        ):
            prefix_fe = fe
            for used in prefix_used:
                prefix_fe += 1 + used
            assert budget_limit - prefix_fe > populations[2]
            observed_delays.append(prefix_fe + 1 - decision_fe)

    assert cap.reachable is True
    assert cap.cap_fe == 36
    assert max(observed_delays) == 36
    assert all(delay <= cap.cap_fe for delay in observed_delays)


def test_scheduler_cap_rejects_optimizer_budget_not_derived_from_schedule() -> None:
    cap = calculate_scheduler_revisit_cap(
        sweep_start_fe=64,
        decision_fe=80,
        cc_budget_limit_fe=100,
        current_group_index=1,
        current_sweep_group_budget_fe=12,
        current_optimizer_budget_fe=8,
        group_population_sizes=(4, 4, 4),
    )

    assert cap.reachable is False
    assert cap.reason == "current_optimizer_budget_not_scheduler_derived"
