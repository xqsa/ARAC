"""Build overlap-relation evidence rows from HCC trace-like payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


@dataclass
class OverlapRelation:
    relation_id: str
    problem_id: str
    outer_iter: int
    group_left: int
    group_right: int
    shared_vars: tuple[int, ...]
    overlap_strength: float
    delta_signal: float
    rank_signal: float
    budget_remaining_ratio: float


_ITERATION_KEYS = ("iterations", "outer_iterations", "trace_windows")
_GROUP_KEYS = (
    "groups",
    "grouping_result",
    "grouping",
    "topology_groups",
    "decomposition_groups",
)
_OVERLAP_KEYS = (
    "overlapping_elements",
    "overlap_groups",
    "shared_variables",
    "shared_vars",
)
_DELTA_KEYS = (
    "fitness_deltas",
    "fitness_delta_list",
    "group_deltas",
    "deltas",
    "score_deltas",
    "scores",
    "fitness_scores",
    "group_scores",
)
_GROUP_DELTA_KEYS = ("fitness_delta", "delta", "score", "fitness_score", "group_score")
_RANK_KEYS = ("group_ranks", "ranks", "rankings", "priority_ranks")
_GROUP_RANK_KEYS = ("rank", "priority_rank", "group_rank")
_BUDGET_RATIO_KEYS = (
    "budget_remaining_ratio",
    "remaining_budget_ratio",
    "budget_ratio_remaining",
)
_FE_USED_KEYS = ("fe_used", "sum_fes", "current_fes")
_BUDGET_LIMIT_KEYS = ("budget_limit", "max_fes", "total_fe")


def build_overlap_relations(
    hcc_trace: dict,
    problem_id: str,
) -> list[OverlapRelation]:
    """Extract deterministic adjacent-group overlap relation evidence."""

    if not isinstance(hcc_trace, dict):
        raise TypeError("hcc_trace must be a dict")

    relations: list[OverlapRelation] = []
    for default_outer_iter, payload in _iter_payloads(hcc_trace):
        outer_iter = _as_int(
            _first_present(payload, ("outer_iter", "iteration", "iter")),
            default_outer_iter,
        )
        groups = _extract_groups(payload)
        overlap_groups = _extract_overlap_groups(payload)
        pair_count = max(max(len(groups) - 1, 0), len(overlap_groups))
        deltas = _extract_numeric_sequence(payload, _DELTA_KEYS, _GROUP_DELTA_KEYS)
        ranks = _extract_numeric_sequence(payload, _RANK_KEYS, _GROUP_RANK_KEYS)
        budget_remaining_ratio = _budget_remaining_ratio(payload)

        for group_left in range(pair_count):
            group_right = group_left + 1
            shared_vars = _shared_vars_for_pair(groups, overlap_groups, group_left, group_right)
            relations.append(
                OverlapRelation(
                    relation_id=f"O{outer_iter}_{group_left}_{group_right}",
                    problem_id=problem_id,
                    outer_iter=outer_iter,
                    group_left=group_left,
                    group_right=group_right,
                    shared_vars=shared_vars,
                    overlap_strength=float(len(shared_vars)),
                    delta_signal=_pair_abs_difference(deltas, group_left, group_right),
                    rank_signal=_rank_stability_proxy(ranks, group_left, group_right),
                    budget_remaining_ratio=budget_remaining_ratio,
                )
            )

    return relations


def _iter_payloads(hcc_trace: dict) -> list[tuple[int, dict]]:
    iteration_payloads = _first_present(hcc_trace, _ITERATION_KEYS)
    if iteration_payloads is None:
        return [(0, hcc_trace)]
    if not isinstance(iteration_payloads, (list, tuple)):
        raise TypeError("iteration payload must be a list or tuple of dicts")
    return [
        (index, payload)
        for index, payload in enumerate(iteration_payloads)
        if isinstance(payload, dict)
    ]


def _extract_groups(payload: dict) -> list[tuple[int, ...]]:
    raw_groups = _first_present(payload, _GROUP_KEYS)
    if raw_groups is None:
        return []
    return [_group_vars(group) for group in _ordered_values(raw_groups)]


def _extract_overlap_groups(payload: dict) -> list[tuple[int, ...]]:
    raw_overlap_groups = _first_present(payload, _OVERLAP_KEYS)
    if raw_overlap_groups is None:
        return []
    return [_coerce_int_tuple(group) for group in _ordered_values(raw_overlap_groups)]


def _extract_numeric_sequence(
    payload: dict,
    sequence_keys: tuple[str, ...],
    group_field_keys: tuple[str, ...],
) -> list[float]:
    explicit = _first_present(payload, sequence_keys)
    if explicit is not None:
        return [_as_float(value, 0.0) for value in _ordered_values(explicit)]

    raw_groups = _first_present(payload, _GROUP_KEYS)
    if raw_groups is None:
        return []

    values: list[float] = []
    for group in _ordered_values(raw_groups):
        if not isinstance(group, dict):
            return []
        value = _first_present(group, group_field_keys)
        if value is None:
            return []
        values.append(_as_float(value, 0.0))
    return values


def _shared_vars_for_pair(
    groups: list[tuple[int, ...]],
    overlap_groups: list[tuple[int, ...]],
    group_left: int,
    group_right: int,
) -> tuple[int, ...]:
    if group_left < len(overlap_groups):
        return overlap_groups[group_left]
    if group_right < len(groups):
        return tuple(sorted(set(groups[group_left]) & set(groups[group_right])))
    return tuple()


def _group_vars(group: Any) -> tuple[int, ...]:
    if isinstance(group, dict):
        raw_vars = _first_present(
            group,
            ("variables", "vars", "members", "dims", "indices", "elements", "values"),
        )
        return tuple() if raw_vars is None else _coerce_int_tuple(raw_vars)
    return _coerce_int_tuple(group)


def _ordered_values(raw: Any) -> list[Any]:
    if isinstance(raw, dict):
        return [raw[key] for key in sorted(raw, key=_sort_key)]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _sort_key(value: Any) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _coerce_int_tuple(raw: Any) -> tuple[int, ...]:
    values: list[int] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            values.append(int(value))
            return
        if isinstance(value, Integral):
            values.append(int(value))
            return
        if isinstance(value, Real):
            numeric = float(value)
            if not numeric.is_integer():
                raise ValueError(f"variable index must be integer-like: {value!r}")
            values.append(int(numeric))
            return
        if isinstance(value, str):
            for token in re.split(r"[\s,;]+", value.strip()):
                if token:
                    numeric = float(token)
                    if not numeric.is_integer():
                        raise ValueError(f"variable index must be integer-like: {token!r}")
                    values.append(int(numeric))
            return
        try:
            iterator = iter(value)
        except TypeError as exc:
            raise TypeError(f"cannot read variable index from {value!r}") from exc
        for item in iterator:
            visit(item)

    visit(raw)
    return tuple(sorted(set(values)))


def _pair_abs_difference(values: list[float], group_left: int, group_right: int) -> float:
    if group_right >= len(values):
        return 0.0
    return abs(values[group_left] - values[group_right])


def _rank_stability_proxy(values: list[float], group_left: int, group_right: int) -> float:
    if group_right >= len(values):
        return 0.0
    denominator = max(len(values) - 1, max(values) - min(values), 1.0)
    return _clamp_ratio(1.0 - (abs(values[group_left] - values[group_right]) / denominator))


def _budget_remaining_ratio(payload: dict) -> float:
    explicit = _first_present(payload, _BUDGET_RATIO_KEYS)
    if explicit is not None:
        return _clamp_ratio(_as_float(explicit, 1.0))

    fe_used = _first_present(payload, _FE_USED_KEYS)
    budget_limit = _first_present(payload, _BUDGET_LIMIT_KEYS)
    if fe_used is None or budget_limit is None:
        return 1.0

    limit = _as_float(budget_limit, 0.0)
    if limit <= 0:
        return 1.0
    return _clamp_ratio((limit - _as_float(fe_used, 0.0)) / limit)


def _first_present(payload: dict, keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(float(value))


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))
