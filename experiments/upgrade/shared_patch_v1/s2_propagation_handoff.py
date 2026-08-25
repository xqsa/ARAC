"""S2 propagation handoff for the shared_patch_v1 upgrade candidate.

Mechanism (docs/arac-oc-stepwise-upgrade-plan-v2.1.md, stage S2): inside one
action episode a single transient field ``last_improved_scope`` is kept; when
a scope produces a strict-best acceptance, the next executable slot prefers a
neighbouring scope that shares variables with it, otherwise the slot falls
back to the S1 order.  The state never crosses a checkpoint.

Because the frozen sweep functions only accept a static ``block_order``, S2
re-implements the two slot-structured orchestration loops here (in the
upgrade namespace) while reusing every frozen numeric component unchanged:
``_PersistentBlockSession``, ``_run_block_visit``, ``_allocate_block_budgets``,
``_block_population_size`` and ``_aligned_visit_budget``.  With no relations
there are no neighbours, so slot selection degenerates to the static order and
the loop reproduces the frozen schedule exactly.

Preregistered slot granularity:

- gcb host: every cold block visit is a slot; the handoff state spans all
  sweeps of the episode; neighbours are checkpoint relation edges.
- ctp host: the interleaved coverage segment has no per-scope slots and keeps
  the S1 static order; the polish segment (one full block polish per slot) is
  slot-structured and carries the handoff state from its first slot on;
  neighbours are variable-set intersections over the polish cover (base blocks
  plus merged relation blocks), because merged covers literally share the
  base blocks' variables.

Receipt semantics (v2.1): ``no_acceptance_event`` (host reachable, no
acceptance to hand off from) is recorded separately from
``host_unreachable`` (no slot-structured scope access at all).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import arac.actions._execution as execution_module
import arac.actions.ctp as ctp_module
import arac.actions.gcb as gcb_module
from experiments.upgrade.shared_patch_v1.conflicting_generator import relation_leverage
from experiments.upgrade.shared_patch_v1.s1_leverage_sweep import LeverageOrderShim, s1_order


REASONS = ("neighbor_of_improved_scope", "no_remaining_neighbor", "no_acceptance_event", "static_order_fallback")


class HandoffState:
    """Transient per-episode propagation state over slot-structured sweeps."""

    def __init__(self, neighbor_fn: Callable[[int, int], bool], *, block_count: int) -> None:
        self._neighbor_fn = neighbor_fn
        self._block_count = int(block_count)
        self.last_improved: int | None = None
        self.selections: list[dict[str, Any]] = []
        self.acceptances = 0
        self.handoff_selections = 0
        self.no_acceptance_events = 0

    def select_next(self, remaining: Iterable[int], static_order: Sequence[int]) -> tuple[int, dict[str, Any]]:
        remaining_set = sorted(int(index) for index in remaining)
        if not remaining_set:
            raise ValueError("slot selection requested with no remaining scope")
        static_rank = {int(index): rank for rank, index in enumerate(static_order)}
        source = self.last_improved
        neighbors: list[int] = []
        if source is not None:
            neighbors = sorted(index for index in remaining_set if self._neighbor_fn(source, index))
        if source is not None and neighbors:
            selected = min(neighbors, key=lambda index: static_rank[index])
            reason = "neighbor_of_improved_scope"
            self.handoff_selections += 1
        else:
            selected = min(remaining_set, key=lambda index: static_rank[index])
            if source is None:
                reason = "no_acceptance_event"
                self.no_acceptance_events += 1
            else:
                reason = "no_remaining_neighbor"
        record = {
            "handoff_reason": reason,
            "selected_scope": int(selected),
            "scope_rank_before": static_rank[selected],
            "handoff_source_scope": None if source is None else int(source),
            "shared_neighbor_count": len(neighbors),
        }
        self.selections.append(record)
        return selected, record

    def report(self, index: int, improved: bool) -> None:
        if improved:
            self.last_improved = int(index)
            self.acceptances += 1
        else:
            self.last_improved = None

    def payload(self) -> dict[str, Any]:
        return {
            "selection_count": len(self.selections),
            "acceptance_count": self.acceptances,
            "handoff_selection_count": self.handoff_selections,
            "no_acceptance_event_count": self.no_acceptance_events,
            "handoff_trace_nonempty": self.handoff_selections > 0,
            "selections": [dict(record) for record in self.selections],
        }


def _cold_sweep_s2(
    context,
    *,
    requested_fes: int,
    sweep_limit: int | None,
    block_order: tuple[int, ...] | None,
    namespace: str,
    seed_factory: Callable[[int], int] | None,
    start_sweep_index: int,
    event_trace: list[dict[str, object]] | None,
    handoff: HandoffState,
    static_order: Sequence[int],
) -> tuple[int, tuple[int, ...]]:
    """Fork of ``run_cold_start_block_sweeps`` with per-slot handoff selection."""

    blocks = context.checkpoint.blocks
    if sorted(static_order) != list(range(len(blocks))):
        raise ValueError("static order must be a complete block permutation")
    if sweep_limit is not None and sweep_limit <= 0:
        raise ValueError("sweep_limit must be positive")
    if start_sweep_index < 0:
        raise ValueError("start_sweep_index must be non-negative")
    action_budget = min(int(requested_fes), context.ledger.remaining)
    target_count = context.ledger.count + action_budget
    population_sizes = tuple(execution_module._block_population_size(len(block)) for block in blocks)
    sweep_costs: list[int] = []
    while context.ledger.count < target_count and (sweep_limit is None or len(sweep_costs) < sweep_limit):
        requested_per_block = math.ceil((target_count - context.ledger.count) / len(blocks))
        sweep_before = context.ledger.count
        sweep_index = start_sweep_index + len(sweep_costs)
        visited: set[int] = set()
        while len(visited) < len(blocks):
            index, selection = handoff.select_next(set(range(len(blocks))) - visited, static_order)
            visit_start = context.ledger.count
            remaining = target_count - context.ledger.count
            population = population_sizes[index]
            if remaining < population:
                break
            visit_budget = execution_module._aligned_visit_budget(
                max(requested_per_block, population),
                remaining,
                population,
            )
            error_before = context.ledger.best_error
            session = execution_module._PersistentBlockSession(
                context,
                blocks[index],
                index,
                visit_budget,
                population_size=population,
                seed_namespace=f"{namespace}-sweep-{sweep_index}",
                seed_factory=seed_factory,
                stage_index=sweep_index * len(blocks) + index + 1,
            )
            consumed = execution_module._run_block_visit(session, visit_budget)
            improved = context.ledger.best_error < error_before
            handoff.report(index, improved)
            visited.add(index)
            if event_trace is not None:
                event_trace.append(
                    {
                        "event": "cold_group_visit",
                        "namespace": namespace,
                        "sweep_index": sweep_index,
                        "group_index": index,
                        "stage_index": sweep_index * len(blocks) + index + 1,
                        "start_fes": visit_start,
                        "requested_fes": visit_budget,
                        "actual_fes": consumed,
                        "end_fes": context.ledger.count,
                        "cold_start": True,
                        "state_restored": False,
                        "handoff_reason": selection["handoff_reason"],
                        "handoff_source_scope": selection["handoff_source_scope"],
                        "improved": improved,
                    }
                )
        sweep_cost = context.ledger.count - sweep_before
        if sweep_cost == 0:
            break
        sweep_costs.append(sweep_cost)
    return sum(sweep_costs), tuple(sweep_costs)


def _sequential_blocks_s2(
    context,
    *,
    requested_fes: int,
    blocks: tuple[tuple[int, ...], ...] | None,
    handoff: HandoffState,
    static_order: Sequence[int],
) -> int:
    """Fork of ``run_sequential_blocks`` with per-slot handoff selection."""

    active_blocks = context.checkpoint.blocks if blocks is None else blocks
    if sorted(static_order) != list(range(len(active_blocks))):
        raise ValueError("static order must be a complete block permutation")
    aligned = min(int(requested_fes), context.ledger.remaining)
    aligned -= aligned % (len(active_blocks) * execution_module.BLOCK_POPULATION_SIZE)
    if aligned == 0:
        return 0
    budgets = execution_module._allocate_block_budgets(
        active_blocks,
        aligned,
        equal_generations=False,
    )
    count_before = context.ledger.count
    visited: set[int] = set()
    while len(visited) < len(active_blocks):
        index, selection = handoff.select_next(set(range(len(active_blocks))) - visited, static_order)
        error_before = context.ledger.best_error
        session = execution_module._PersistentBlockSession(
            context,
            active_blocks[index],
            index,
            budgets[index],
        )
        while not session.complete:
            session.advance()
        handoff.report(index, context.ledger.best_error < error_before)
        visited.add(index)
    consumed = context.ledger.count - count_before
    if consumed != aligned:
        raise RuntimeError("sequential block execution drifted from its aligned FE budget")
    return consumed


class S2Shim:
    """Install S1's coverage reorder plus the S2 slot handoff for one host."""

    def __init__(self, host: str) -> None:
        if host not in ("ctp", "gcb"):
            raise ValueError(f"unsupported S2 host: {host}")
        self.host = host
        self.handoff: HandoffState | None = None
        self.checkpoint_hash: str | None = None
        self.slot_segments = 0
        self.s1_order_records: list[dict[str, Any]] = []
        self._s1_shim: LeverageOrderShim | None = None

    def _ensure_handoff(self, context, block_count: int) -> HandoffState:
        if self.handoff is None:
            self.checkpoint_hash = context.checkpoint.checkpoint_hash
            self.handoff = HandoffState(self._neighbor_fn(context), block_count=block_count)
        elif context.checkpoint.checkpoint_hash != self.checkpoint_hash:
            raise RuntimeError("S2 handoff state observed a different checkpoint")
        return self.handoff

    def _neighbor_fn(self, context) -> Callable[[int, int], bool]:
        relations = context.checkpoint.relations
        edges = {frozenset((relation.left_block, relation.right_block)) for relation in relations}

        def neighbor(left: int, right: int) -> bool:
            return frozenset((left, right)) in edges

        return neighbor

    def _gcb_static_order(self, context, block_order: tuple[int, ...] | None, *, first_call: bool) -> tuple[int, ...]:
        blocks = context.checkpoint.blocks
        baseline = tuple(range(len(blocks))) if block_order is None else tuple(block_order)
        if not first_call:
            return baseline
        leverage = relation_leverage(blocks, context.checkpoint.relations)
        applied = s1_order(len(blocks), leverage, baseline)
        self.s1_order_records.append(
            {
                "host": "gcb",
                "namespace": "gcb-source",
                "block_count": len(blocks),
                "leverage_per_block": list(leverage),
                "baseline_order": [int(index) for index in baseline],
                "applied_order": [int(index) for index in applied],
                "order_changed": tuple(baseline) != applied,
            }
        )
        return applied

    def install(self) -> None:
        if self.host == "ctp":
            self._s1_shim = LeverageOrderShim("ctp")
            self._s1_shim.install()
            original_sequential = ctp_module.run_sequential_blocks

            def sequential_wrapper(context, **kwargs):
                active_blocks = kwargs.get("blocks") or context.checkpoint.blocks
                baseline = tuple(kwargs.get("block_order") or range(len(active_blocks)))
                if len(baseline) != len(active_blocks):
                    raise ValueError("ctp polish block_order must cover the polish cover")
                handoff = self._ensure_handoff_cover(context, len(active_blocks))
                self.slot_segments += 1
                return _sequential_blocks_s2(
                    context,
                    requested_fes=kwargs["requested_fes"],
                    blocks=kwargs.get("blocks"),
                    handoff=handoff,
                    static_order=baseline,
                )

            self._restore_sequential = original_sequential
            ctp_module.run_sequential_blocks = sequential_wrapper
        else:
            original_cold = gcb_module.run_cold_start_block_sweeps
            state = {"calls": 0}

            def cold_wrapper(context, **kwargs):
                first_call = state["calls"] == 0
                state["calls"] += 1
                static_order = self._gcb_static_order(context, kwargs.get("block_order"), first_call=first_call)
                handoff = self._ensure_handoff(context, len(context.checkpoint.blocks))
                self.slot_segments += 1
                return _cold_sweep_s2(
                    context,
                    requested_fes=kwargs["requested_fes"],
                    sweep_limit=kwargs.get("sweep_limit"),
                    block_order=kwargs.get("block_order"),
                    namespace=str(kwargs.get("namespace", "")),
                    seed_factory=kwargs.get("seed_factory"),
                    start_sweep_index=int(kwargs.get("start_sweep_index", 0)),
                    event_trace=kwargs.get("event_trace"),
                    handoff=handoff,
                    static_order=static_order,
                )

            self._restore_cold = original_cold
            gcb_module.run_cold_start_block_sweeps = cold_wrapper

    def _ensure_handoff_cover(self, context, block_count: int) -> HandoffState:
        if self.handoff is None:
            self.checkpoint_hash = context.checkpoint.checkpoint_hash
            active = _polish_cover(context.checkpoint.blocks, context.checkpoint.relations)
            variable_sets = [frozenset(block) for block in active]

            def neighbor(left: int, right: int) -> bool:
                return bool(variable_sets[left] & variable_sets[right])

            self.handoff = HandoffState(neighbor, block_count=block_count)
        elif context.checkpoint.checkpoint_hash != self.checkpoint_hash:
            raise RuntimeError("S2 handoff state observed a different checkpoint")
        return self.handoff

    def uninstall(self) -> None:
        if self._s1_shim is not None:
            self.s1_order_records.extend(dict(record) for record in self._s1_shim.records)
            self._s1_shim.uninstall()
            self._s1_shim = None
        if self.host == "ctp" and getattr(self, "_restore_sequential", None) is not None:
            ctp_module.run_sequential_blocks = self._restore_sequential
            self._restore_sequential = None
        if self.host == "gcb" and getattr(self, "_restore_cold", None) is not None:
            gcb_module.run_cold_start_block_sweeps = self._restore_cold
            self._restore_cold = None

    def payload(self) -> dict[str, Any]:
        handoff_payload = self.handoff.payload() if self.handoff is not None else None
        return {
            "host": self.host,
            "slot_segment_count": self.slot_segments,
            "handoff": handoff_payload,
            "s1_order_records": [dict(record) for record in self.s1_order_records],
        }


def _polish_cover(blocks, relations) -> tuple[tuple[int, ...], ...]:
    """Reproduce the frozen ctp positive-relation polish cover."""

    base_blocks = tuple(tuple(block) for block in blocks)
    relation_blocks = []
    for relation in sorted(
        relations,
        key=lambda item: (-item.strength * (1.0 + item.disagreement), item.left_block, item.right_block),
    ):
        merged = tuple(sorted(set(base_blocks[relation.left_block]) | set(base_blocks[relation.right_block])))
        if merged not in relation_blocks:
            relation_blocks.append(merged)
    return base_blocks + tuple(relation_blocks)


__all__ = ["HandoffState", "S2Shim", "_cold_sweep_s2", "_sequential_blocks_s2"]
