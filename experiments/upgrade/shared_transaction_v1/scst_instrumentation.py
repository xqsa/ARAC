"""Non-invasive transaction audit instrumentation for shared_transaction_v1.

Observes three things without touching any frozen source file, any random
state, or any evaluation order:

1. **Phase boundaries.**  The phase runner functions re-exported by the
   action modules (``run_persistent_blocks``, ``run_sequential_blocks``,
   ``run_stalled_block_rescue``, ``run_stateful_block_visits``,
   ``run_stateful_block_visits_with_sessions``, ``run_cold_start_block_sweeps``,
   ``run_full_space``) are wrapped to record entry/exit FE, the ledger
   incumbent hash at entry, and (for ``run_full_space``) the namespace.
2. **Block-session windows.**  ``_PersistentBlockSession.__init__`` /
   ``.advance`` are patched at class level to record each session's birth
   (block index, construction anchor hash) and every strict-best improvement
   inside an ``advance`` call (the SCST proposal definition: the committed
   incumbent value of the last strict-best writeback of a block inside its
   source phase).
3. **Ledger timeline.**  The arm's ledger instance gets an ``evaluate`` /
   ``evaluate_incumbent`` wrapper that appends ``(fes, error, hash)`` rows
   for every strict-best improvement - the propagation evidence backbone.

Every wrapper restores the original on ``uninstall``.  Patches observe only:
they never rewrite arguments, reorder calls, or consume evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable

import numpy as np

import arac.actions._execution as execution_module
import arac.actions.ctp as ctp_module
import arac.actions.gcb as gcb_module
import arac.actions.phase2_v2 as phase2_v2_module
import arac.actions.smp as smp_module

_PHASE_RUNNERS = (
    "run_persistent_blocks",
    "run_sequential_blocks",
    "run_stalled_block_rescue",
    "run_stateful_block_visits",
    "run_stateful_block_visits_with_sessions",
    "run_cold_start_block_sweeps",
    "run_full_space",
)

_PHASE_MODULES = (
    execution_module,
    ctp_module,
    smp_module,
    gcb_module,
    phase2_v2_module,
)


def _hash_vector(vector: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vector, dtype=float).tobytes()).hexdigest()


@dataclass
class PhaseRecord:
    tag: str
    entry_fes: int
    exit_fes: int | None
    incumbent_hash_at_entry: str
    incumbent_error_at_entry: float
    namespace: str | None = None


@dataclass
class SessionRecord:
    phase_tag: str
    block_index: int
    block_size: int
    birth_fes: int
    anchor_hash: str


@dataclass
class ImprovementRecord:
    phase_tag: str
    block_index: int | None
    commit_fes: int
    incumbent_hash_after: str
    error: float
    committed_x: np.ndarray | None = None  # in-memory only; never serialized


@dataclass
class TimelineRecord:
    fes: int
    error: float
    incumbent_hash: str


class TransactionAuditRecorder:
    """Install/uninstall audit patches; collect the census payloads."""

    def __init__(self) -> None:
        self.phases: list[PhaseRecord] = []
        self.sessions: list[SessionRecord] = []
        self.improvements: list[ImprovementRecord] = []
        self.timeline: list[TimelineRecord] = []
        self._phase_stack: list[PhaseRecord] = []
        self._current_phase_tag: str = "outside"
        self._orig_runners: dict[tuple[int, str], Callable[..., Any]] = {}
        self._orig_session_init: Callable[..., None] | None = None
        self._orig_session_advance: Callable[..., Any] | None = None
        self._orig_evaluate: Callable[..., Any] | None = None
        self._orig_evaluate_incumbent: Callable[..., Any] | None = None
        self._ledger: Any = None
        self._installed = False

    # -- installation -----------------------------------------------------

    def install(self, ledger: Any) -> None:
        if self._installed:
            raise RuntimeError("recorder already installed")
        self._ledger = ledger
        self._orig_session_init = execution_module._PersistentBlockSession.__init__
        self._orig_session_advance = execution_module._PersistentBlockSession.advance
        recorder = self

        def patched_init(session, context, block, index, budget_fes, **kwargs):
            recorder._orig_session_init(session, context, block, index, budget_fes, **kwargs)
            recorder.sessions.append(
                SessionRecord(
                    phase_tag=recorder._current_phase_tag,
                    block_index=int(index),
                    block_size=len(tuple(block)),
                    birth_fes=int(context.ledger.count),
                    anchor_hash=_hash_vector(context.ledger.best_x),
                )
            )

        def patched_advance(session, *args, **kwargs):
            ledger_now = session.context.ledger
            error_before = ledger_now.best_error
            consumed = recorder._orig_session_advance(session, *args, **kwargs)
            if ledger_now.best_error < error_before:
                recorder.improvements.append(
                    ImprovementRecord(
                        phase_tag=recorder._current_phase_tag,
                        block_index=int(session.block_index),
                        commit_fes=int(ledger_now.count),
                        incumbent_hash_after=_hash_vector(ledger_now.best_x),
                        error=float(ledger_now.best_error),
                        committed_x=ledger_now.best_x.copy(),
                    )
                )
            return consumed

        execution_module._PersistentBlockSession.__init__ = patched_init
        execution_module._PersistentBlockSession.advance = patched_advance

        for module in _PHASE_MODULES:
            for name in _PHASE_RUNNERS:
                original = getattr(module, name, None)
                if original is None or getattr(original, "_scst_audit_wrapped", False):
                    continue
                wrapper = self._wrap_runner(original, name)
                setattr(module, name, wrapper)
                self._orig_runners[(id(module), name)] = original

        original_evaluate = ledger.evaluate
        original_incumbent = ledger.evaluate_incumbent
        self._orig_evaluate = original_evaluate
        self._orig_evaluate_incumbent = original_incumbent

        def patched_evaluate(candidate):
            result = original_evaluate(candidate)
            self._note_ledger()
            return result

        def patched_incumbent(**kwargs):
            result = original_incumbent(**kwargs)
            self._note_ledger()
            return result

        ledger.evaluate = patched_evaluate
        ledger.evaluate_incumbent = patched_incumbent
        # Seed the timeline with the install-time incumbent so that
        # ``incumbent_hash_at`` always resolves for positions at or after
        # installation, even before the first strict-best improvement.
        self.timeline.append(
            TimelineRecord(
                fes=int(ledger.count),
                error=float(ledger.best_error),
                incumbent_hash=_hash_vector(ledger.best_x),
            )
        )
        self._installed = True

    def _wrap_runner(self, original: Callable[..., Any], name: str) -> Callable[..., Any]:
        recorder = self

        def wrapper(*args, **kwargs):
            context = args[0] if args else kwargs.get("context")
            ledger = getattr(context, "ledger", None)
            record = PhaseRecord(
                tag=name,
                entry_fes=int(ledger.count) if ledger is not None else -1,
                exit_fes=None,
                incumbent_hash_at_entry=_hash_vector(ledger.best_x) if ledger is not None else "",
                incumbent_error_at_entry=float(ledger.best_error) if ledger is not None else float("nan"),
                namespace=kwargs.get("namespace"),
            )
            recorder.phases.append(record)
            recorder._phase_stack.append(record)
            previous_tag = recorder._current_phase_tag
            recorder._current_phase_tag = name
            try:
                return original(*args, **kwargs)
            finally:
                recorder._current_phase_tag = previous_tag
                recorder._phase_stack.pop()
                record.exit_fes = int(ledger.count) if ledger is not None else -1

        wrapper._scst_audit_wrapped = True
        return wrapper

    def _note_ledger(self) -> None:
        ledger = self._ledger
        error = float(ledger.best_error)
        if not self.timeline or error < self.timeline[-1].error:
            self.timeline.append(
                TimelineRecord(
                    fes=int(ledger.count),
                    error=error,
                    incumbent_hash=_hash_vector(ledger.best_x),
                )
            )

    def uninstall(self) -> None:
        if not self._installed:
            return
        execution_module._PersistentBlockSession.__init__ = self._orig_session_init
        execution_module._PersistentBlockSession.advance = self._orig_session_advance
        for (module_id, name), original in self._orig_runners.items():
            for module in _PHASE_MODULES:
                if id(module) == module_id:
                    setattr(module, name, original)
        self._orig_runners.clear()
        if self._ledger is not None:
            self._ledger.evaluate = self._orig_evaluate
            self._ledger.evaluate_incumbent = self._orig_evaluate_incumbent
        self._installed = False

    # -- census -----------------------------------------------------------

    def proposals_by_block(self, phase_tag: str) -> dict[int, ImprovementRecord]:
        """Last strict-best writeback per block inside one source phase."""

        result: dict[int, ImprovementRecord] = {}
        for record in self.improvements:
            if record.phase_tag == phase_tag and record.block_index is not None:
                result[record.block_index] = record
        return result

    def first_session_after(self, phase_tag: str) -> SessionRecord | None:
        """First block session born inside the given phase."""

        for session in self.sessions:
            if session.phase_tag == phase_tag:
                return session
        return None

    def phase(self, tag: str) -> list[PhaseRecord]:
        return [record for record in self.phases if record.tag == tag]

    def incumbent_hash_at(self, fes: int) -> str | None:
        """Incumbent hash at a ledger position, per the improvement timeline.

        The row with the largest position <= ``fes`` is the incumbent a live
        consumer (block session, full-space run) would read at that moment;
        ``None`` means no strict-best improvement had been recorded yet, so
        the consumer reads the initial incumbent.
        """

        current: TimelineRecord | None = None
        for row in self.timeline:
            if row.fes <= fes:
                current = row
            else:
                break
        return current.incumbent_hash if current is not None else None

    def census_payload(self) -> dict[str, Any]:
        return {
            "phases": [
                {
                    "tag": record.tag,
                    "entry_fes": record.entry_fes,
                    "exit_fes": record.exit_fes,
                    "incumbent_hash_at_entry": record.incumbent_hash_at_entry,
                    "incumbent_error_at_entry": record.incumbent_error_at_entry,
                    "namespace": record.namespace,
                }
                for record in self.phases
            ],
            "session_count": len(self.sessions),
            "improvement_count": len(self.improvements),
            "timeline_rows": len(self.timeline),
        }


__all__ = ["TransactionAuditRecorder", "ImprovementRecord", "PhaseRecord", "SessionRecord"]
