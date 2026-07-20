"""Deterministic execution of a frozen group-budget allocation."""

from __future__ import annotations

from collections.abc import Sequence

from arac.actions.action_spec import ActionSpec


BUDGET_REALLOCATION_ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        name="efficiency_budget_reallocation",
        semantic_surface="budget_allocation",
        parameter_names=("group_budgets", "population_sizes", "frozen_total"),
    ),
)


def apply_budget_reallocation_action(
    action_name: str,
    group_budgets: Sequence[int],
    population_sizes: Sequence[int],
    frozen_total: int,
) -> tuple[int, ...]:
    """Validate and return the exact budget vector selected upstream."""

    if action_name != "efficiency_budget_reallocation":
        raise ValueError(f"unsupported budget reallocation action: {action_name!r}")
    raw_budgets = tuple(group_budgets)
    raw_populations = tuple(population_sizes)
    budgets = tuple(int(value) for value in raw_budgets)
    populations = tuple(int(value) for value in raw_populations)
    if not budgets or len(budgets) != len(populations):
        raise ValueError("budget action vectors must be non-empty and aligned")
    if any(
        isinstance(value, bool) or int(value) != value
        for value in (*raw_budgets, *raw_populations)
    ):
        raise ValueError("budget action values must be integers")
    if any(budget < population or population <= 0 for budget, population in zip(
        budgets, populations, strict=True
    )):
        raise ValueError("each group budget must cover one positive population")
    if isinstance(frozen_total, bool) or int(frozen_total) != frozen_total:
        raise ValueError("frozen_total must be an integer")
    if sum(budgets) != int(frozen_total):
        raise ValueError("group budgets must preserve the frozen FE total")
    return budgets
