"""Non-invasive sweep instrumentation for the recovered ctp/gcb hosts.

The frozen executors are never edited.  In the experiment process only, the
three block-sweep entry points that the ctp/gcb executors call are wrapped
with recording shims that delegate to the originals unchanged:

- ``run_cold_start_block_sweeps`` (gcb source/native/continuation sweeps)
  is recorded through the frozen ``event_trace`` channel, which yields one
  record per cold block visit;
- ``run_persistent_blocks`` (ctp coverage) and ``run_sequential_blocks``
  (ctp polish) are recorded at call granularity; each block owns exactly
  one optimizer session per call, so per-block visits are 1 per block when
  the call consumed a non-zero budget.

Bit-identity of the instrumented trajectory is verified against the frozen
recovery-screen receipts by the caller, so any accidental perturbation is
caught by the gate rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import arac.actions._execution as execution_module
import arac.actions.ctp as ctp_module
import arac.actions.gcb as gcb_module


@dataclass
class SweepRecorder:
    """Per-arm record of every sweep-capable segment the host executed."""

    segments: list[dict[str, Any]] = field(default_factory=list)

    def _wrap_persistent(self, original: Callable[..., int], kind: str) -> Callable[..., int]:
        def wrapper(context, **kwargs):
            start_fes = context.ledger.count
            consumed = original(context, **kwargs)
            blocks = kwargs.get("blocks") or context.checkpoint.blocks
            order = tuple(kwargs.get("block_order") or range(len(blocks)))
            # keys stay strings so the canonical hash of a receipt is stable
            # across an in-memory build and a JSON round trip
            per_block = {str(int(index)): 1 for index in order} if consumed > 0 else {}
            self.segments.append(
                {
                    "kind": kind,
                    "start_fes": start_fes,
                    "end_fes": context.ledger.count,
                    "requested_fes": int(kwargs["requested_fes"]),
                    "consumed_fes": int(consumed),
                    "block_count": len(blocks),
                    "block_order": [int(index) for index in order],
                    "per_block_visits": per_block,
                }
            )
            return consumed

        return wrapper

    def _wrap_cold(self, original: Callable[..., tuple[int, tuple[int, ...]]]) -> Callable[..., tuple[int, tuple[int, ...]]]:
        def wrapper(context, **kwargs):
            start_fes = context.ledger.count
            trace = kwargs.get("event_trace")
            owns_trace = trace is None
            if owns_trace:
                trace = []
                kwargs["event_trace"] = trace
            consumed, sweeps = original(context, **kwargs)
            events = [dict(event) for event in trace] if owns_trace else []
            per_block: dict[str, int] = {}
            visit_rows = []
            for event in events:
                if event.get("event") != "cold_group_visit":
                    continue
                index = int(event["group_index"])
                key = str(index)
                per_block[key] = per_block.get(key, 0) + 1
                visit_rows.append(
                    {
                        "group_index": index,
                        "sweep_index": int(event["sweep_index"]),
                        "requested_fes": int(event["requested_fes"]),
                        "actual_fes": int(event["actual_fes"]),
                    }
                )
            order = tuple(kwargs.get("block_order") or range(len(context.checkpoint.blocks)))
            self.segments.append(
                {
                    "kind": "cold_start_sweeps",
                    "namespace": str(kwargs.get("namespace", "")),
                    "start_fes": start_fes,
                    "end_fes": context.ledger.count,
                    "requested_fes": int(kwargs["requested_fes"]),
                    "consumed_fes": int(consumed),
                    "sweep_count": len(sweeps),
                    "block_count": len(context.checkpoint.blocks),
                    "block_order": [int(index) for index in order],
                    "per_block_visits": per_block,
                    "visits": visit_rows,
                }
            )
            return consumed, sweeps

        return wrapper

    def install(self) -> None:
        self._originals = (
            execution_module.run_persistent_blocks,
            execution_module.run_sequential_blocks,
            execution_module.run_cold_start_block_sweeps,
            ctp_module.run_persistent_blocks,
            ctp_module.run_sequential_blocks,
            gcb_module.run_cold_start_block_sweeps,
        )
        ctp_module.run_persistent_blocks = self._wrap_persistent(execution_module.run_persistent_blocks, "persistent_coverage")
        ctp_module.run_sequential_blocks = self._wrap_persistent(execution_module.run_sequential_blocks, "sequential_polish")
        gcb_module.run_cold_start_block_sweeps = self._wrap_cold(execution_module.run_cold_start_block_sweeps)

    def uninstall(self) -> None:
        (
            execution_module.run_persistent_blocks,
            execution_module.run_sequential_blocks,
            execution_module.run_cold_start_block_sweeps,
            ctp_module.run_persistent_blocks,
            ctp_module.run_sequential_blocks,
            gcb_module.run_cold_start_block_sweeps,
        ) = self._originals

    def summary(self, leverage: tuple[int, ...]) -> dict[str, Any]:
        total_visits = 0
        leverage_visits = 0
        kinds: dict[str, int] = {}
        per_block: dict[str, int] = {}
        for segment in self.segments:
            kinds[segment["kind"]] = kinds.get(segment["kind"], 0) + 1
            for index, visits in segment["per_block_visits"].items():
                key = str(int(index))
                per_block[key] = per_block.get(key, 0) + visits
                numeric = int(index)
                total_visits += visits
                if numeric < len(leverage) and leverage[numeric] > 0:
                    leverage_visits += visits
        return {
            "segment_count": len(self.segments),
            "segment_kinds": kinds,
            "total_block_visits": total_visits,
            "leverage_positive_block_visits": leverage_visits,
            "per_block_visits": {str(index): visits for index, visits in sorted(per_block.items())},
        }


def new_recorder() -> SweepRecorder:
    return SweepRecorder()


__all__ = ["SweepRecorder", "new_recorder"]
