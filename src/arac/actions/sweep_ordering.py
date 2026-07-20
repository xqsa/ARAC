"""Deterministic execution of a frozen group permutation."""

from __future__ import annotations

from collections.abc import Sequence

from arac.actions.action_spec import ActionSpec


SWEEP_ORDERING_ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        name="delta_priority_scan",
        semantic_surface="sweep_ordering",
        parameter_names=("group_order", "group_count"),
    ),
)


def apply_sweep_ordering_action(
    action_name: str,
    group_order: Sequence[int],
    group_count: int,
) -> tuple[int, ...]:
    """Validate and return the exact permutation selected upstream."""

    if action_name != "delta_priority_scan":
        raise ValueError(f"unsupported sweep ordering action: {action_name!r}")
    if isinstance(group_count, bool) or int(group_count) != group_count or group_count <= 0:
        raise ValueError("group_count must be a positive integer")
    raw_order = tuple(group_order)
    if any(
        isinstance(value, bool) or int(value) != value for value in raw_order
    ):
        raise ValueError("group order values must be integers")
    order = tuple(int(value) for value in raw_order)
    if len(order) != int(group_count) or set(order) != set(range(int(group_count))):
        raise ValueError("group order must be a complete permutation")
    return order
