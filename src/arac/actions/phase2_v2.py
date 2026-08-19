"""Interruptible Phase-II v2 action states.

The legacy action executors still expose the original one-shot optimizers.  This
module supplies the common v2 lifecycle used by the next protocol revision:
``initialize -> step -> snapshot -> restore``.  Each state owns a deterministic
proposal stream.  The proposal at cursor ``k`` depends only on the immutable
checkpoint and action seed, so splitting a run cannot change its prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json

import numpy as np

from arac.actions._execution import (
    DEFAULT_SIGMA,
    FULL_SPACE_POPULATION_SIZE,
    HISTORICAL_MATERIAL_LOG_GAIN,
    STATE_STALE_WINDOW,
    _PersistentBlockSession,
    _aligned_visit_budget,
    _block_population_size,
    _historical_log_improvement,
    derived_seed,
    terminal_result,
)
from arac.runtime.contracts import (
    ActionContext,
    ActionResult,
    Phase2Snapshot,
    Phase2StepResult,
)
from arac.runtime.optimizers import ResumableOptimizerSession
from arac.evidence.mechanism_features import summarize_relation_topology
from arac.runtime.phase2 import (
    EpisodeProgress,
    Phase2StateError,
    ResumablePhase2State,
    validate_snapshot_context,
)


def _check_maturity_window(maturity_window_fes: int) -> None:
    if (
        isinstance(maturity_window_fes, bool)
        or not isinstance(maturity_window_fes, int)
        or maturity_window_fes <= 0
    ):
        raise ValueError("maturity_window_fes must be a positive integer")


def _finite_vector(values: object, dimension: int, name: str) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise Phase2StateError(f"{name} is not a finite problem-sized vector")
    return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class _Segment:
    algorithm: str
    dimensions: tuple[int, ...] | None
    budget_fes: int
    seed: int
    namespace: str

    def payload(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "dimensions": None if self.dimensions is None else list(self.dimensions),
            "budget_fes": self.budget_fes,
            "seed": self.seed,
            "namespace": self.namespace,
        }


def _balanced_budgets(total_fes: int, count: int) -> tuple[int, ...]:
    if total_fes <= 0 or count <= 0:
        return ()
    active = min(total_fes, count)
    base, remainder = divmod(total_fes, active)
    return tuple(base + (1 if index < remainder else 0) for index in range(active))


def _block_segments(
    context: ActionContext,
    *,
    total_fes: int,
    blocks: tuple[tuple[int, ...], ...],
    algorithm: str,
    namespace: str,
) -> tuple[_Segment, ...]:
    budgets = _balanced_budgets(total_fes, len(blocks))
    return tuple(
        _Segment(
            algorithm=algorithm,
            dimensions=tuple(blocks[index]),
            budget_fes=budget,
            seed=derived_seed(context, namespace, index),
            namespace=f"{namespace}-{index}",
        )
        for index, budget in enumerate(budgets)
    )


class Phase2V2State(ResumablePhase2State):
    """Shared deterministic state machine for all four v2 action paths."""

    def __init__(
        self,
        context: ActionContext,
        *,
        cursor: int = 0,
        anchor: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(context)
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise Phase2StateError("Phase-II cursor must be a non-negative integer")
        if cursor != self.consumed_fes:
            raise Phase2StateError("Phase-II cursor does not match the ledger position")
        self.cursor = cursor
        self.anchor = (
            tuple(float(value) for value in context.checkpoint.incumbent)
            if anchor is None
            else _finite_vector(anchor, context.problem.dimension, "Phase-II anchor")
        )
        if not np.all(
            (np.asarray(self.anchor) >= context.problem.lower_array)
            & (np.asarray(self.anchor) <= context.problem.upper_array)
        ):
            raise Phase2StateError("Phase-II anchor escaped the public bounds")
        self.best_trace = (
            [float(context.ledger.best_error)] if context.retain_trajectory else []
        )

    @property
    def route(self) -> str:
        if self.action_name == "aor":
            return f"phase2_v2_evidence_routed_{self._aor_algorithm()}"
        if self.action_name == "ctp":
            return "phase2_v2_coverage_then_polish"
        if self.action_name == "smp":
            return "phase2_v2_stateful_memory"
        relation_mode = (
            "zero_relation" if self.context.checkpoint.overlap_relation_count == 0 else "positive_relation_graph"
        )
        return f"phase2_v2_{relation_mode}"

    @property
    def optimizer_package(self) -> str:
        return "arac-phase2-v2"

    @property
    def optimizer_version(self) -> str:
        return "1"

    @property
    def numerical_repair_count(self) -> int:
        return 0

    def _aor_algorithm(self) -> str:
        features = dict(
            zip(
                self.context.checkpoint.feature_names,
                self.context.checkpoint.feature_values,
                strict=True,
            )
        )
        roughness = float(features.get("line_high_frequency_fraction_median", 0.3))
        return "sepcmaes" if roughness < 0.3 else "mmes"

    def _noise(self, namespace: str, size: int) -> np.ndarray:
        return np.random.default_rng(
            derived_seed(self.context, f"phase2-v2-{namespace}", self.cursor)
        ).standard_normal(size)

    def _proposal(self) -> np.ndarray:
        """Return the candidate at the current cursor without consulting history."""

        anchor = np.asarray(self.anchor, dtype=float)
        span = self.context.problem.upper_array - self.context.problem.lower_array
        if self.action_name == "aor":
            scale = 0.12 if self._aor_algorithm() == "sepcmaes" else 0.18
            candidate = anchor + scale * span * self._noise("aor", anchor.size)
        elif self.action_name == "ctp":
            blocks = self._ctp_cover()
            block = np.asarray(blocks[self.cursor % len(blocks)], dtype=int)
            candidate = anchor.copy()
            candidate[block] += 0.10 * span[block] * self._noise("ctp", len(block))
        elif self.action_name == "smp":
            blocks = self.context.checkpoint.blocks
            block = np.asarray(blocks[self.cursor % len(blocks)], dtype=int)
            cycle = (self.cursor // len(blocks)) % 4
            scale = (0.18, 0.09, 0.045, 0.0225)[cycle]
            candidate = anchor.copy()
            candidate[block] += scale * span[block] * self._noise("smp", len(block))
        else:
            blocks = self._gcb_blocks()
            block = np.asarray(blocks[self.cursor % len(blocks)], dtype=int)
            candidate = anchor.copy()
            candidate[block] += 0.14 * span[block] * self._noise("gcb", len(block))
        return np.clip(candidate, self.context.problem.lower_array, self.context.problem.upper_array)

    def _ctp_cover(self) -> tuple[tuple[int, ...], ...]:
        base = tuple(tuple(block) for block in self.context.checkpoint.blocks)
        merged: list[tuple[int, ...]] = []
        for relation in sorted(
            self.context.checkpoint.relations,
            key=lambda item: (
                -item.strength * (1.0 + item.disagreement),
                item.left_block,
                item.right_block,
            ),
        ):
            candidate = tuple(
                sorted(
                    set(base[relation.left_block]) | set(base[relation.right_block])
                )
            )
            if candidate not in merged:
                merged.append(candidate)
            if len(merged) >= len(base):
                break
        return base + tuple(merged)

    def _gcb_blocks(self) -> tuple[tuple[int, ...], ...]:
        base = tuple(tuple(block) for block in self.context.checkpoint.blocks)
        scores = [0.0] * len(base)
        for relation in self.context.checkpoint.relations:
            score = relation.strength * (1.0 + relation.disagreement)
            scores[relation.left_block] += score
            scores[relation.right_block] += score
        ordered = [base[index] for index in sorted(range(len(base)), key=lambda i: (-scores[i], i))]
        if self.context.checkpoint.overlap_relation_count == 0:
            return tuple(ordered)
        for relation in self.context.checkpoint.relations:
            merged = tuple(
                sorted(
                    set(base[relation.left_block]) | set(base[relation.right_block])
                )
            )
            if merged not in ordered:
                ordered.append(merged)
        return tuple(ordered)

    def _advance(self, budget_fes: int) -> None:
        for _ in range(budget_fes):
            self.context.ledger.evaluate(self._proposal())
            self.cursor += 1
            if self.context.retain_trajectory:
                self.best_trace.append(float(self.context.ledger.best_error))

    def _private_payload(self) -> dict[str, object]:
        return {}

    def _restore_private(self, payload: dict[str, object]) -> None:
        if payload:
            raise Phase2StateError("unexpected private Phase-II state payload")

    def _snapshot_payload(self) -> bytes:
        payload = {
            "schema": "arac-phase2-v2-state-v1",
            "action_name": self.action_name,
            "cursor": self.cursor,
            "anchor": list(self.anchor),
            "best_trace": self.best_trace if self.context.retain_trajectory else None,
            "private": self._private_payload(),
            "route": self.route,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def restore(cls, context: ActionContext, snapshot: Phase2Snapshot) -> Phase2V2State:
        validate_snapshot_context(context, snapshot)
        try:
            payload = json.loads(snapshot.state_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Phase2StateError("Phase-II v2 state payload is not valid JSON") from exc
        if payload.get("schema") != "arac-phase2-v2-state-v1":
            raise Phase2StateError("Phase-II v2 state schema drifted")
        if payload.get("action_name") != context.action_name:
            raise Phase2StateError("Phase-II v2 state action does not match context")
        raw_cursor = payload.get("cursor")
        raw_anchor = payload.get("anchor")
        raw_trace = payload.get("best_trace")
        if isinstance(raw_cursor, bool) or not isinstance(raw_cursor, int):
            raise Phase2StateError("Phase-II v2 cursor is not an integer")
        if not isinstance(raw_anchor, list):
            raise Phase2StateError("Phase-II v2 anchor payload is invalid")
        if context.retain_trajectory:
            if (
                not isinstance(raw_trace, list)
                or len(raw_trace) != raw_cursor + 1
                or any(
                    not isinstance(value, (int, float)) or not np.isfinite(value)
                    for value in raw_trace
                )
            ):
                raise Phase2StateError("Phase-II v2 best trace is invalid")
        elif raw_trace is not None:
            raise Phase2StateError("compact Phase-II snapshot unexpectedly carries a trace")
        state = cls(
            context,
            cursor=raw_cursor,
            anchor=tuple(float(value) for value in raw_anchor),
        )
        state.best_trace = (
            [float(value) for value in raw_trace] if context.retain_trajectory else []
        )
        raw_private = payload.get("private", {})
        if not isinstance(raw_private, dict):
            raise Phase2StateError("Phase-II private state payload is invalid")
        state._restore_private(raw_private)
        if payload.get("route") != state.route or state.snapshot().state_hash != snapshot.state_hash:
            raise Phase2StateError("Phase-II v2 state payload does not match snapshot")
        return state

    def result(self) -> ActionResult:
        if not self.complete:
            raise Phase2StateError("Phase-II v2 action is not complete")
        return terminal_result(
            self.context,
            route=self.route,
            optimizer_package=self.optimizer_package,
            optimizer_version=self.optimizer_version,
        )


class AorPhase2State(Phase2V2State):
    """Resumable AOR state."""

    def __init__(
        self,
        context: ActionContext,
        *,
        cursor: int = 0,
        anchor: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(context, cursor=cursor, anchor=anchor)
        algorithm = self._aor_algorithm()
        self.optimizer_session = ResumableOptimizerSession(
            algorithm,
            problem=context.problem,
            ledger=context.ledger,
            initial_mean=self.anchor,
            sigma=0.5,
            seed=derived_seed(context, f"aor-{algorithm}"),
            budget_fes=self.total_fes - self.start_fes,
            population_size=24,
            initial_consumed=cursor,
        )

    def _advance(self, budget_fes: int) -> None:
        before = self.optimizer_session.consumed_fes
        trace = self.optimizer_session.step(
            budget_fes, record_trace=self.context.retain_trajectory
        )
        self.cursor += self.optimizer_session.consumed_fes - before
        if self.context.retain_trajectory:
            self.best_trace.extend(trace)

    def _private_payload(self) -> dict[str, object]:
        import json

        return json.loads(self.optimizer_session.json_payload().decode("utf-8"))

    def _restore_private(self, payload: dict[str, object]) -> None:
        self.optimizer_session.restore_state_dict(payload)

    @property
    def optimizer_package(self) -> str:
        return self.optimizer_session.package

    @property
    def optimizer_version(self) -> str:
        return self.optimizer_session.package_version

    @property
    def numerical_repair_count(self) -> int:
        return self.optimizer_session.sigma_floor_repair_count


class RecoveredAorPhase2State(Phase2V2State):
    """Resumable recovered AOR: fresh zero-mean Sep-CMA over the full space.

    Mirrors the legacy :class:`~arac.actions.recovered.RecoveredAorExecutor`
    one-shot (zero initial mean, ``DEFAULT_SIGMA``, the raw action seed and
    the 24-member population) so segmented episodes reproduce the recovered
    route; only the lifecycle (step/snapshot/restore) is new.
    """

    def __init__(
        self,
        context: ActionContext,
        *,
        cursor: int = 0,
        anchor: tuple[float, ...] | None = None,
    ) -> None:
        zero_anchor = tuple(0.0 for _ in range(context.problem.dimension))
        super().__init__(context, cursor=cursor, anchor=zero_anchor if anchor is None else anchor)
        if self.anchor != zero_anchor:
            raise Phase2StateError("recovered AOR state anchors at the zero vector")
        self.optimizer_session = ResumableOptimizerSession(
            "sepcmaes",
            problem=context.problem,
            ledger=context.ledger,
            initial_mean=self.anchor,
            sigma=DEFAULT_SIGMA,
            seed=context.action_seed,
            budget_fes=self.total_fes - self.start_fes,
            population_size=FULL_SPACE_POPULATION_SIZE,
            initial_consumed=cursor,
        )

    def _advance(self, budget_fes: int) -> None:
        before = self.optimizer_session.consumed_fes
        trace = self.optimizer_session.step(
            budget_fes, record_trace=self.context.retain_trajectory
        )
        self.cursor += self.optimizer_session.consumed_fes - before
        if self.context.retain_trajectory:
            self.best_trace.extend(trace)

    def _private_payload(self) -> dict[str, object]:
        import json

        return json.loads(self.optimizer_session.json_payload().decode("utf-8"))

    def _restore_private(self, payload: dict[str, object]) -> None:
        self.optimizer_session.restore_state_dict(payload)

    @property
    def route(self) -> str:
        return f"recovered_fresh_zero_mean_sepcmaes_{self.total_fes - self.start_fes}"

    def progress(self, *, maturity_window_fes: int = 20_000) -> EpisodeProgress:
        """AOR contract: one ``global_correction`` regime.

        Protocol maturity = one independent correction window consumed
        (clamped to the action budget).  The session is FE-granular, so
        any budget is executable and any FE is a legal switch point.
        """

        _check_maturity_window(maturity_window_fes)
        window = min(maturity_window_fes, self.total_fes - self.start_fes)
        return EpisodeProgress(
            episode=self.action_name,
            phase="global_correction",
            consumed_fes=self.consumed_fes,
            next_boundary_fes=min(maturity_window_fes, self.context.ledger.remaining),
            min_step_fes=1,
            maturity_target_fes=window,
            protocol_mature=self.consumed_fes >= window,
            contract="aor-fresh-correction-v1",
        )

    @property
    def optimizer_package(self) -> str:
        return self.optimizer_session.package

    @property
    def optimizer_version(self) -> str:
        return self.optimizer_session.package_version

    @property
    def numerical_repair_count(self) -> int:
        return self.optimizer_session.sigma_floor_repair_count


class RecoveredSmpPhase2State(Phase2V2State):
    """Resumable recovered SMP: stateful block visits, generation-aligned.

    Reimplements the legacy ``run_stateful_block_visits_with_sessions`` loop
    (clip_offspring=False, precheck, strict material gain) as a pausable
    state machine.  Step budgets must decompose into the units the loop
    consumes -- one precheck FE, whole population generations, or terminal
    noop evaluations; a misaligned remainder raises instead of splitting a
    generation, because the vendor objectives are batch-shape sensitive at
    the last bit and a split would silently change the trajectory.
    """


    def __init__(
        self,
        context: ActionContext,
        *,
        cursor: int = 0,
        anchor: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(context, cursor=cursor, anchor=anchor)
        blocks = context.checkpoint.blocks
        self.order = tuple(range(len(blocks)))
        self.populations = tuple(_block_population_size(len(block)) for block in blocks)
        budget = self.total_fes - self.start_fes
        self.action_budget = budget
        self.target_count = self.start_fes + budget
        self.sessions = tuple(
            _PersistentBlockSession(
                context,
                block,
                index,
                budget,
                population_size=self.populations[index],
                seed_namespace="stateful-block",
                stage_index=index + 1,
                clip_offspring=False,
            )
            for index, block in enumerate(blocks)
        )
        self.stale_streaks = [0] * len(blocks)
        self.cold_start = [True] * len(blocks)
        self.visit_count = 0
        self.restart_count = 0
        self.outer_iter = 0
        self.noop_count = 0
        self.phase = "sweeps"
        self.sweep_active = False
        self.sweep_requested_per_block = 0
        self.sweep_index = 0
        self.sweep_start_count = context.ledger.count
        self.visit: dict[str, object] | None = None

    def step(self, budget_fes: int) -> Phase2StepResult:
        """Consume the largest generation-aligned portion of ``budget_fes``.

        The actual consumption is reported in ``step_fes``; callers may
        re-step with the remainder at any time.  Mid-generation splits are
        refused (vendor objectives are batch-shape sensitive), so partial
        consumption is the episode contract, not an error.
        """

        if self.complete:
            raise Phase2StateError("completed Phase-II state cannot advance")
        if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes <= 0:
            raise ValueError("budget_fes must be a positive integer")
        remaining_budget = self.total_fes - self.context.ledger.count
        before = self.context.ledger.count
        self._advance(min(budget_fes, remaining_budget))
        consumed = self.context.ledger.count - before
        if consumed == 0:
            raise Phase2StateError(
                "recovered SMP step budget is below the next generation-aligned "
                f"unit; request at least {min(self.populations) + 1} FE or the "
                "remaining budget"
            )
        snapshot = self.snapshot()
        self._last_step_snapshot = snapshot
        return Phase2StepResult(
            action_name=self.action_name,
            checkpoint_hash=self.checkpoint_hash,
            action_seed=self.action_seed,
            step_fes=consumed,
            consumed_fes=self.consumed_fes,
            total_fes=self.total_fes - self.start_fes,
            best_error=float(self.context.ledger.best_error),
            complete=self.complete,
            state_hash=snapshot.state_hash,
        )

    def _advance(self, budget_fes: int) -> None:
        remaining = budget_fes
        transitions = 0
        # Each generation costs ~3 transitions for ~13 FE, so the stall
        # guard must scale with the step budget instead of a flat cap.
        transition_limit = 10_000 + 20 * budget_fes
        while remaining > 0:
            consumed = self._advance_unit(remaining)
            if consumed is None:
                break
            if consumed and self.context.retain_trajectory:
                self.best_trace.extend(
                    [float(self.context.ledger.best_error)] * consumed
                )
            remaining -= consumed
            transitions += 1
            if transitions > transition_limit:
                raise Phase2StateError("recovered SMP state machine stalled")
        self.cursor = self.consumed_fes

    def _advance_unit(self, max_fe: int) -> int:
        ledger = self.context.ledger
        if self.phase == "drain":
            ledger.evaluate(ledger.best_x)
            self.noop_count += 1
            return 1
        if ledger.count >= self.target_count:
            self.phase = "drain"
            return 0
        if not self.sweep_active:
            sweep_remaining = self.target_count - ledger.count
            self.sweep_requested_per_block = -(-sweep_remaining // len(self.order))
            self.sweep_index = 0
            self.sweep_start_count = ledger.count
            self.sweep_active = True
            return 0
        if self.visit is None:
            if self.sweep_index >= len(self.order):
                if ledger.count == self.sweep_start_count:
                    self.phase = "drain"
                    return 0
                self.outer_iter += 1
                self.sweep_active = False
                return 0
            block_index = self.order[self.sweep_index]
            population = self.populations[block_index]
            remaining_to_target = self.target_count - ledger.count
            if remaining_to_target <= population:
                # Legacy precheck guard: no room for a precheck plus one generation.
                self.sweep_index = len(self.order)
                return 0
            precheck_error = ledger.evaluate_incumbent(refresh_error=True)
            self.visit = {
                "block": block_index,
                "aligned": None,
                "consumed": 0,
                "before": precheck_error,
            }
            return 1
        block_index = int(self.visit["block"])
        session = self.sessions[block_index]
        population = self.populations[block_index]
        if self.visit["aligned"] is None:
            remaining_to_target = self.target_count - ledger.count
            aligned = _aligned_visit_budget(
                max(self.sweep_requested_per_block, population),
                remaining_to_target,
                population,
            )
            self.visit["aligned"] = aligned
            if aligned == 0:
                self._finish_visit(0)
                return 0
            session.begin_visit()
            return 0
        if session._prepared:
            if max_fe < population:
                return None
            completes_visit = self.visit["consumed"] + population >= self.visit["aligned"]
            session.evaluate_pending_batch()
            session.finalize_generation(adapt_on_early_stop=completes_visit)
            self.visit["consumed"] += population
            return population
        if self.visit["consumed"] < self.visit["aligned"] and not session.early_stopped:
            if session.consumed_fes + population > session.budget_fes:
                raise Phase2StateError("block session cannot consume a partial population")
            if population > ledger.remaining:
                raise Phase2StateError("block population exceeds the remaining terminal budget")
            session.prepare_generation()
            return 0
        self._finish_visit(int(self.visit["consumed"]))
        return 0

    def _finish_visit(self, consumed: int) -> None:
        if consumed > 0:
            block_index = int(self.visit["block"])
            self.visit_count += 1
            gain = _historical_log_improvement(
                float(self.visit["before"]), self.context.ledger.best_error
            )
            material = gain > HISTORICAL_MATERIAL_LOG_GAIN
            self.stale_streaks[block_index] = (
                0 if material else self.stale_streaks[block_index] + 1
            )
            reset = self.stale_streaks[block_index] >= STATE_STALE_WINDOW
            self.cold_start[block_index] = False
            if reset:
                self.sessions[block_index].restart(stage_index=None)
                self.stale_streaks[block_index] = 0
                self.restart_count += 1
                self.cold_start[block_index] = True
        self.sweep_index += 1
        self.visit = None

    def _private_payload(self) -> dict[str, object]:
        return {
            "action_budget": self.action_budget,
            "stale_streaks": list(self.stale_streaks),
            "cold_start": list(self.cold_start),
            "visit_count": self.visit_count,
            "restart_count": self.restart_count,
            "outer_iter": self.outer_iter,
            "noop_count": self.noop_count,
            "phase": self.phase,
            "sweep_active": self.sweep_active,
            "sweep_requested_per_block": self.sweep_requested_per_block,
            "sweep_index": self.sweep_index,
            "sweep_start_count": self.sweep_start_count,
            "visit": self.visit,
            "sessions": [session.state_dict() for session in self.sessions],
        }

    def _restore_private(self, payload: dict[str, object]) -> None:
        if int(payload["action_budget"]) != self.action_budget:
            raise Phase2StateError("recovered SMP action budget drifted")
        self.stale_streaks = [int(value) for value in payload["stale_streaks"]]
        self.cold_start = [bool(value) for value in payload["cold_start"]]
        self.visit_count = int(payload["visit_count"])
        self.restart_count = int(payload["restart_count"])
        self.outer_iter = int(payload["outer_iter"])
        self.noop_count = int(payload["noop_count"])
        self.phase = str(payload["phase"])
        self.sweep_active = bool(payload["sweep_active"])
        self.sweep_requested_per_block = int(payload["sweep_requested_per_block"])
        self.sweep_index = int(payload["sweep_index"])
        self.sweep_start_count = int(payload["sweep_start_count"])
        self.visit = payload["visit"]
        sessions = payload["sessions"]
        if not isinstance(sessions, list) or len(sessions) != len(self.sessions):
            raise Phase2StateError("recovered SMP session payload is invalid")
        for session, session_payload in zip(self.sessions, sessions, strict=True):
            session.restore_state_dict(session_payload)

    @property
    def route(self) -> str:
        # Match the legacy route accounting: the visit-loop consumption
        # includes the per-visit precheck FE, i.e. everything except the
        # terminal noop drain.
        consumed = self.total_fes - self.start_fes - self.noop_count
        return (
            f"recovered_stateful_visits_{consumed}_visits_{self.visit_count}_"
            f"stale_resets_{self.restart_count}_noop_{self.noop_count}"
        )

    def progress(self, *, maturity_window_fes: int = 20_000) -> EpisodeProgress:
        """SMP contract: ``block_sweep -> visit``.

        Protocol maturity = one full block visit completed plus one
        maturity window consumed (clamped to the action budget).  Steps
        are generation-aligned: the smallest executable request is one
        precheck FE plus the smallest block population, and a visit is
        never split mid-generation.
        """

        _check_maturity_window(maturity_window_fes)
        window = min(maturity_window_fes, self.action_budget)
        if self.phase == "drain":
            phase = "drain"
            boundary = self.context.ledger.remaining
        elif self.visit is not None:
            phase = "visit"
            block_index = int(self.visit["block"])
            aligned = self.visit["aligned"]
            boundary = (
                self.populations[block_index]
                if aligned is None
                else max(int(aligned) - int(self.visit["consumed"]), 0)
            )
        else:
            phase = "block_sweep"
            boundary = 1
        return EpisodeProgress(
            episode=self.action_name,
            phase=phase,
            consumed_fes=self.consumed_fes,
            next_boundary_fes=boundary,
            min_step_fes=min(self.populations) + 1,
            maturity_target_fes=window,
            protocol_mature=self.visit_count >= 1 and self.consumed_fes >= window,
            contract="smp-stateful-visits-v1",
        )

    @property
    def optimizer_package(self) -> str:
        return "arac-phase2-v2"

    @property
    def optimizer_version(self) -> str:
        return "1"

    @property
    def numerical_repair_count(self) -> int:
        return 0


class _BlockScheduledPhase2State(Phase2V2State):
    """Run action-specific fixed segments backed by resumable pypop states."""

    def __init__(
        self,
        context: ActionContext,
        *,
        cursor: int = 0,
        anchor: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(context, cursor=cursor, anchor=anchor)
        self._segments = self._build_segments()
        expected = self.total_fes - self.start_fes
        if sum(segment.budget_fes for segment in self._segments) != expected:
            raise Phase2StateError("Phase-II segment budgets do not cover the terminal budget")
        self._segment_index = 0
        self._segment_consumed = 0
        self._segment_session: ResumableOptimizerSession | None = None
        self._completed_sigma_floor_repairs = 0

    def _build_segments(self) -> tuple[_Segment, ...]:
        raise NotImplementedError

    def _start_segment(self) -> None:
        if self._segment_index >= len(self._segments):
            raise Phase2StateError("Phase-II segment schedule is already complete")
        segment = self._segments[self._segment_index]
        current = self.context.ledger.best_x
        dimensions = segment.dimensions
        initial_mean = (
            current
            if dimensions is None
            else tuple(float(current[index]) for index in dimensions)
        )
        self._segment_session = ResumableOptimizerSession(
            segment.algorithm,
            problem=self.context.problem,
            ledger=self.context.ledger,
            initial_mean=initial_mean,
            sigma=0.5,
            seed=segment.seed,
            budget_fes=segment.budget_fes,
            population_size=24 if dimensions is None else 12,
            dimensions=dimensions,
            anchor=tuple(float(value) for value in current),
        )

    def _prefix_budget(self, prefix: str) -> int:
        """Cumulative budget of every segment whose namespace carries ``prefix``."""

        return sum(
            segment.budget_fes
            for segment in self._segments
            if segment.namespace.startswith(prefix)
        )

    def _current_segment_boundary(self) -> int:
        """FE from the current position to the end of the active segment.

        Structural milestone only: block-scheduled segments are resumable
        at any FE inside them, so every FE is a legal switch point; the
        boundary reports where the current semantic unit ends.
        """

        if self._segment_index >= len(self._segments):
            return 0
        return self._segments[self._segment_index].budget_fes - self._segment_consumed

    def _advance(self, budget_fes: int) -> None:
        remaining = budget_fes
        while remaining:
            if self._segment_session is None:
                self._start_segment()
            segment = self._segments[self._segment_index]
            segment_remaining = segment.budget_fes - self._segment_consumed
            chunk = min(remaining, segment_remaining)
            before = self._segment_session.consumed_fes
            trace = self._segment_session.step(
                chunk, record_trace=self.context.retain_trajectory
            )
            consumed = self._segment_session.consumed_fes - before
            self.cursor += consumed
            if self.context.retain_trajectory:
                self.best_trace.extend(trace)
            self._segment_consumed += consumed
            remaining -= consumed
            if self._segment_consumed == segment.budget_fes:
                self._completed_sigma_floor_repairs += (
                    self._segment_session.sigma_floor_repair_count
                )
                self._segment_index += 1
                self._segment_consumed = 0
                self._segment_session = None

    def _private_payload(self) -> dict[str, object]:
        import json

        return {
            "segments": [segment.payload() for segment in self._segments],
            "segment_index": self._segment_index,
            "segment_consumed": self._segment_consumed,
            "completed_sigma_floor_repairs": self._completed_sigma_floor_repairs,
            "session": (
                None
                if self._segment_session is None
                else json.loads(self._segment_session.json_payload().decode("utf-8"))
            ),
        }

    def _restore_private(self, payload: dict[str, object]) -> None:
        if payload.get("segments") != [segment.payload() for segment in self._segments]:
            raise Phase2StateError("Phase-II segment schedule drifted")
        raw_index = payload.get("segment_index")
        raw_consumed = payload.get("segment_consumed")
        raw_repairs = payload.get("completed_sigma_floor_repairs")
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or not 0 <= raw_index <= len(self._segments)
            or isinstance(raw_consumed, bool)
            or not isinstance(raw_consumed, int)
            or raw_consumed < 0
            or isinstance(raw_repairs, bool)
            or not isinstance(raw_repairs, int)
            or raw_repairs < 0
        ):
            raise Phase2StateError("Phase-II segment position is invalid")
        self._segment_index = raw_index
        self._segment_consumed = raw_consumed
        self._completed_sigma_floor_repairs = raw_repairs
        raw_session = payload.get("session")
        if self._segment_index == len(self._segments):
            if self._segment_consumed != 0 or raw_session is not None:
                raise Phase2StateError("completed Phase-II segments carry pending state")
            return
        segment = self._segments[self._segment_index]
        if self._segment_consumed >= segment.budget_fes:
            raise Phase2StateError("pending Phase-II segment state is invalid")
        # A snapshot may land exactly between two segments.  In that state the
        # next optimizer has not sampled anything yet, so there is no session
        # payload to restore; _advance will create it deterministically.
        if self._segment_consumed == 0 and raw_session is None:
            return
        if not isinstance(raw_session, dict):
            raise Phase2StateError("pending Phase-II segment state is invalid")
        raw_dimensions = raw_session.get("dimensions")
        dimensions = None if raw_dimensions is None else tuple(int(value) for value in raw_dimensions)
        # A full-space segment stores ``dimensions=None`` while the session
        # always persists the expanded coordinate tuple; compare in the
        # expanded form so mid-segment snapshots of full-space segments
        # restore instead of reporting a coordinate drift.
        expected_dimensions = (
            segment.dimensions
            if segment.dimensions is not None
            else tuple(range(self.context.problem.dimension))
        )
        if dimensions != expected_dimensions:
            raise Phase2StateError("pending Phase-II segment coordinates drifted")
        raw_mean = raw_session.get("initial_mean")
        raw_anchor = raw_session.get("anchor")
        self._segment_session = ResumableOptimizerSession(
            segment.algorithm,
            problem=self.context.problem,
            ledger=self.context.ledger,
            initial_mean=tuple(float(value) for value in raw_mean),
            sigma=float(raw_session.get("sigma", 0.5)),
            seed=int(raw_session.get("seed")),
            budget_fes=segment.budget_fes,
            # Mirror _start_segment: full-space segments carry the 24-member
            # population regardless of how the session payload serializes
            # its expanded coordinate tuple.
            population_size=24 if segment.dimensions is None else 12,
            initial_consumed=self._segment_consumed,
            dimensions=dimensions,
            anchor=tuple(float(value) for value in raw_anchor),
        )
        self._segment_session.restore_state_dict(raw_session)

    @property
    def optimizer_package(self) -> str:
        return "pypop7"

    @property
    def optimizer_version(self) -> str:
        return importlib.metadata.version("pypop7")

    @property
    def numerical_repair_count(self) -> int:
        active = (
            0
            if self._segment_session is None
            else self._segment_session.sigma_floor_repair_count
        )
        return self._completed_sigma_floor_repairs + active


class CtpPhase2State(_BlockScheduledPhase2State):
    """Resumable CTP coverage followed by full-space polish."""

    def progress(self, *, maturity_window_fes: int = 20_000) -> EpisodeProgress:
        """CTP contract: ``coverage -> polish``.

        Protocol maturity = coverage regime complete plus one polish
        window (clamped to the polish budget).  A probe that stops
        exactly at the coverage boundary is NOT mature -- the polish
        window must actually run.
        """

        _check_maturity_window(maturity_window_fes)
        total = self.total_fes - self.start_fes
        coverage = self._prefix_budget("ctp-v2-coverage")
        window = min(maturity_window_fes, total - coverage)
        target = coverage + window
        return EpisodeProgress(
            episode=self.action_name,
            phase="coverage" if self.consumed_fes < coverage else "polish",
            consumed_fes=self.consumed_fes,
            next_boundary_fes=self._current_segment_boundary(),
            min_step_fes=1,
            maturity_target_fes=target,
            protocol_mature=self.consumed_fes >= target,
            contract="ctp-coverage-polish-v1",
        )

    def _build_segments(self) -> tuple[_Segment, ...]:
        total = self.total_fes - self.start_fes
        base = tuple(tuple(block) for block in self.context.checkpoint.blocks)
        relation_ratio = self.context.checkpoint.overlap_relation_count / len(base)
        coverage_fraction = max(0.10, 0.20 / (1.0 + relation_ratio))
        coverage = min(total, max(len(base) * 12, int(total * coverage_fraction)))
        cover = self._ctp_cover()
        segments = list(
            _block_segments(
                self.context,
                total_fes=coverage,
                blocks=cover,
                algorithm="cmaes",
                namespace="ctp-v2-coverage",
            )
        )
        if total - coverage:
            segments.append(
                _Segment(
                    "mmes",
                    None,
                    total - coverage,
                    derived_seed(self.context, "ctp-v2-polish"),
                    "ctp-v2-polish",
                )
            )
        return tuple(segments)


class SmpPhase2State(_BlockScheduledPhase2State):
    """Resumable SMP block memory followed by global polish."""

    def _build_segments(self) -> tuple[_Segment, ...]:
        total = self.total_fes - self.start_fes
        if self.context.checkpoint.overlap_relation_count == 0:
            global_fraction = 0.0
        else:
            _, entropy, largest_component = summarize_relation_topology(
                self.context.checkpoint.blocks,
                self.context.checkpoint.relations,
            )
            global_fraction = min(0.50, 0.20 + 0.30 * entropy * largest_component)
        global_budget = int(total * global_fraction)
        segments = list(
            _block_segments(
                self.context,
                total_fes=total - global_budget,
                blocks=tuple(tuple(block) for block in self.context.checkpoint.blocks),
                algorithm="cmaes",
                namespace="smp-v2-block",
            )
        )
        if global_budget:
            segments.append(
                _Segment(
                    "mmes",
                    None,
                    global_budget,
                    derived_seed(self.context, "smp-v2-global"),
                    "smp-v2-global",
                )
            )
        return tuple(segments)


class GcbPhase2State(_BlockScheduledPhase2State):
    """Resumable GCB warmup, coordination, and continuation schedule."""

    def progress(self, *, maturity_window_fes: int = 20_000) -> EpisodeProgress:
        """GSS contract: ``warmup -> coordination/continuation``.

        Protocol maturity = the warmup block sweep complete plus one
        window consumed in the post-warmup regime (coordination or
        continuation), clamped to what remains after warmup.
        """

        _check_maturity_window(maturity_window_fes)
        total = self.total_fes - self.start_fes
        warmup = self._prefix_budget("gcb-v2-warmup")
        coordination = self._prefix_budget("gcb-v2-coordination")
        window = min(maturity_window_fes, total - warmup)
        target = warmup + window
        consumed = self.consumed_fes
        if consumed < warmup:
            phase = "warmup"
        elif consumed < warmup + coordination:
            phase = "coordination"
        else:
            phase = "continuation"
        return EpisodeProgress(
            episode=self.action_name,
            phase=phase,
            consumed_fes=consumed,
            next_boundary_fes=self._current_segment_boundary(),
            min_step_fes=1,
            maturity_target_fes=target,
            protocol_mature=consumed >= target,
            contract="gss-warmup-coordination-continuation-v1",
        )

    def _ordered_blocks(self) -> tuple[tuple[int, ...], ...]:
        base = tuple(tuple(block) for block in self.context.checkpoint.blocks)
        scores = [0.0] * len(base)
        for relation in self.context.checkpoint.relations:
            score = relation.strength * (1.0 + relation.disagreement)
            scores[relation.left_block] += score
            scores[relation.right_block] += score
        return tuple(base[index] for index in sorted(range(len(base)), key=lambda i: (-scores[i], i)))

    def _build_segments(self) -> tuple[_Segment, ...]:
        total = self.total_fes - self.start_fes
        warmup = int(total * 0.20)
        if self.context.checkpoint.overlap_relation_count == 0:
            coordination_fraction = 0.0
        else:
            _, entropy, largest_component = summarize_relation_topology(
                self.context.checkpoint.blocks,
                self.context.checkpoint.relations,
            )
            coordination_fraction = min(0.40, 0.10 + 0.30 * entropy * largest_component)
        coordination = min(total - warmup, int(total * coordination_fraction))
        continuation = total - warmup - coordination
        ordered = self._ordered_blocks()
        segments = list(
            _block_segments(
                self.context,
                total_fes=warmup,
                blocks=ordered,
                algorithm="cmaes",
                namespace="gcb-v2-warmup",
            )
        )
        if coordination:
            segments.append(
                _Segment(
                    "sepcmaes",
                    None,
                    coordination,
                    derived_seed(self.context, "gcb-v2-coordination"),
                    "gcb-v2-coordination",
                )
            )
        segments.extend(
            _block_segments(
                self.context,
                total_fes=continuation,
                blocks=ordered,
                algorithm="cmaes",
                namespace="gcb-v2-continuation",
            )
        )
        return tuple(segments)


def state_type(action_name: str) -> type[Phase2V2State]:
    states = {
        "aor": AorPhase2State,
        "ctp": CtpPhase2State,
        "smp": SmpPhase2State,
        "gcb": GcbPhase2State,
    }
    try:
        return states[action_name]
    except KeyError as exc:
        raise ValueError(f"unsupported Phase-II action: {action_name}") from exc


def initialize_state(context: ActionContext) -> Phase2V2State:
    """Create the action-specific v2 state after validating the context type."""

    if not isinstance(context, ActionContext):
        raise TypeError("context must be ActionContext")
    return state_type(context.action_name)(context)


def restore_state(context: ActionContext, snapshot: Phase2Snapshot) -> Phase2V2State:
    """Restore an action-specific v2 state with public binding checks."""

    if not isinstance(context, ActionContext):
        raise TypeError("context must be ActionContext")
    return state_type(context.action_name).restore(context, snapshot)


__all__ = [
    "AorPhase2State",
    "CtpPhase2State",
    "GcbPhase2State",
    "Phase2V2State",
    "initialize_state",
    "restore_state",
    "SmpPhase2State",
    "state_type",
]
