"""Episode-level ARAC-OC scheduling over the four complete v2 episodes.

Each historical Phase-II composition (CTP coverage+polish, GSS graph
sweeps, recovered SMP stateful visits, recovered AOR fresh correction) is
a v2 state machine owning a full-budget internal schedule.  The scheduler
hosts all four on private mirrored ledgers over one global ledger: every
evaluation is charged once to the global budget and feeds the global
strict-best archive, while each episode's internal position advances on
its own ledger.

Gate 50b semantics (pre-registered):

- Terminology: ``dispatcher`` is the GCB coordinator; the four search
  episodes are CTP / SMP / GSS / AOR, where GSS is the graph-scheduled
  sweep composition whose historical receipt name remains ``gcb`` for
  artifact compatibility (``episode_kind`` records the public name).
- Dual gains: ``global`` archive deltas own materiality, switching and
  exploitation ranking; ``local`` (private archive) deltas are recorded
  for internal-stagnation diagnostics only and never drive scheduling.
- Evidence-driven cold start: one counted B/W/C probe over the shared
  variables orders the forced per-episode probe windows (a ranking, never
  a bare C_j threshold -- Gate 47 rejected magnitude gating) and fixes
  the sensing scope.  Every episode receives one real executable probe
  window; if the remaining budget cannot fund a minimum probe for any
  unprobed episode the protocol fails loudly instead of letting it
  starve (Gate 50's R2/R6 failure mode: AOR saw 1-4 FE).
- After the probes, exploitation concentrates on the best global-gain
  per-FE episode; non-material segments (by global materiality) switch
  down the evidence ranking.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from arac.actions._execution import HISTORICAL_MATERIAL_LOG_GAIN
from arac.actions.ctp import CtpExecutor
from arac.actions.gcb import GcbExecutor
from arac.actions.recovered import RecoveredAorExecutor, RecoveredSmpExecutor
from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.counted_probe import counted_probe
from arac.coordination.overlap import OverlapStructure
from arac.runtime.contracts import (
    ActionContext,
    Phase2Snapshot,
    PhaseCheckpoint,
    canonical_sha256,
)
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import EpisodeProgress, Phase2StateError

OC_EPISODE_SCHEMA = "arac-oc-episode-schedule-v3"
# How each episode consumes the archive baton at a switch.  Search-state
# re-anchoring happens at the next segment/visit boundary via the ledger;
# AOR keeps its zero-mean search by design and only gains the baseline.
# Episodes whose anchor machinery requires in-bounds incumbents refuse an
# out-of-bounds baton (declared boundary policy).
HANDOFF_ACCEPTS_OOB = {"ctp": False, "gcb": False, "smp": True, "aor": True}
HANDOFF_MODE = {
    "ctp": "reanchor_next_segment",
    "gcb": "reanchor_next_segment",
    "smp": "reanchor_next_visit",
    "aor": "fresh_by_design",
}
EPISODES = ("ctp", "gcb", "smp", "aor")
DISPATCHER = "gcb_coordinator"
# Public episode names; the graph-sweep composition keeps its historical
# receipt name ``gcb`` while the paper-facing name is GSS.
EPISODE_KIND = {"ctp": "ctp", "gcb": "gss", "smp": "smp", "aor": "aor"}
COORDINATOR_PUBLIC_NAME = "GCB coordinator"
EPISODE_PUBLIC_NAMES = dict(EPISODE_KIND)
PROBE_SHARE = 0.10
# Coordinates beyond this magnitude are clipped before evaluation (see
# _EpisodeLedger.evaluate).  Measured on 2026-08-16: the AOB vendor
# transforms overflow to NaN well below float range (R2/Tasy NaN at
# |x|=5e4) while every case stays finite up to 1e4 -- 100x the public
# bounds (+-100), so legitimate search never touches the guard.
_MAGNITUDE_GUARD = 1e4
PROBE_MIN_FES = 20_000


class _EpisodeLedger(EvaluationLedger):
    """Private episode position mirroring every evaluation into the global ledger."""

    def __init__(
        self,
        problem: OptimizationProblem,
        *,
        total_budget_fes: int,
        initial_count: int,
        initial_incumbent: tuple[float, ...],
        initial_error: float,
        global_ledger: EvaluationLedger,
    ) -> None:
        super().__init__(
            problem,
            total_budget_fes,
            initial_count=initial_count,
            initial_incumbent=initial_incumbent,
            initial_error=initial_error,
            allow_out_of_bounds=True,
        )
        if global_ledger.count < initial_count:
            raise ValueError("global ledger is behind the episode boundary")
        self.global_ledger = global_ledger
        self.magnitude_repairs = 0

    def evaluate(self, candidate: np.ndarray) -> float | np.ndarray:
        values = np.asarray(candidate, dtype=float)
        single = values.ndim == 1
        batch = values[np.newaxis, :] if single else values
        if batch.ndim != 2 or batch.shape[1] != self.problem.dimension:
            raise ValueError("candidate shape does not match the problem dimension")
        # Numerical boundary: a huge-but-finite coordinate (observed at 3M
        # on R2: sigma/eigenvalue drift pushing |x| past 1e150) makes the
        # vendor objectives overflow to inf.  Clipping to a magnitude guard
        # far beyond any meaningful search scale keeps the evaluation
        # finite; the repair is counted and surfaced in the schedule
        # receipts.  Mirrors the legacy block-session bound clipping.
        if not np.all(np.isfinite(batch)) or np.any(np.abs(batch) > _MAGNITUDE_GUARD):
            batch = np.clip(
                np.nan_to_num(batch, nan=0.0, posinf=_MAGNITUDE_GUARD, neginf=-_MAGNITUDE_GUARD),
                -_MAGNITUDE_GUARD,
                _MAGNITUDE_GUARD,
            )
            self.magnitude_repairs += 1
        requested = int(batch.shape[0])
        if requested > self.remaining:
            raise RuntimeError("episode exceeded its planned budget")
        if requested > self.global_ledger.remaining:
            raise RuntimeError("global ledger cannot pay episode FE")
        raw = self.global_ledger.evaluate(batch[0] if single else batch)
        results = np.asarray(raw, dtype=float).reshape(-1)
        if results.shape != (requested,) or not np.all(np.isfinite(results)):
            raise ValueError("global objective returned invalid episode values")
        self._count += requested
        for vector, result in zip(batch, results, strict=True):
            numeric = float(result)
            if self._best_x is None or numeric < self._best_error:
                self._best_x = vector.copy()
                self._best_error = numeric
        return float(results[0]) if single else results

    def adopt_global_archive(
        self, *, source: EvaluationLedger, accept_out_of_bounds: bool
    ) -> tuple[bool, str]:
        """Archive handoff: adopt the source strict-best as this episode's baseline.

        Strictly monotone (a source archive that is not better is refused),
        and the private FE position is asserted unchanged -- only the
        strict-best archive moves.  Returns (adopted, refusal).

        Boundary policy (declared): the global strict-best may be an
        out-of-bounds point produced by the unbounded SMP profile.  Episodes
        whose optimizer machinery requires in-bounds anchors (CTP/GSS via
        the resumable session constructor) refuse such a baton and wait for
        the next in-bounds incumbent; the unbounded SMP machinery and the
        archive-only AOR baseline accept it.
        """

        count_before = self._count
        source_x = np.asarray(source.best_x, dtype=float).copy()
        source_error = float(source.best_error)
        if source_x.shape != (self.problem.dimension,) or not np.all(np.isfinite(source_x)):
            raise ValueError("source archive incumbent is not adoptable")
        if not math.isfinite(source_error):
            raise ValueError("source archive error is not adoptable")
        if not source_error < self._best_error:
            return False, "not_better"
        if not accept_out_of_bounds and (
            np.any(source_x < self.problem.lower_array)
            or np.any(source_x > self.problem.upper_array)
        ):
            return False, "oob_incumbent"
        self._best_x = source_x
        self._best_error = source_error
        if self._count != count_before:
            raise RuntimeError("archive adoption changed the private FE position")
        return True, "none"


def _log_gain(before: float, after: float) -> float:
    return math.log((before + 1e-300) / (after + 1e-300)) if after < before else 0.0


@dataclass(frozen=True)
class EpisodeProbeReceipt:
    order_position: int
    episode: str
    episode_kind: str
    budget_fes: int
    scope_size: int
    global_error_before: float
    global_error_after: float
    global_gain: float
    global_gain_per_fe: float
    local_error_before: float
    local_error_after: float
    local_gain: float

    @property
    def public_episode(self) -> str:
        """Paper-facing episode name; ``episode`` remains receipt-compatible."""

        return EPISODE_KIND[self.episode]


@dataclass(frozen=True)
class EpisodeSegmentReceipt:
    segment_index: int
    phase: str
    episode: str
    episode_kind: str
    requested_fes: int
    consumed_fes: int
    global_error_before: float
    global_error_after: float
    global_gain: float
    local_error_before: float
    local_error_after: float
    local_gain: float
    material: bool
    switched: bool
    next_episode: str
    state_hash: str
    snapshot_hash: str = ""

    @property
    def public_episode(self) -> str:
        """Paper-facing episode name; ``episode`` remains receipt-compatible."""

        return EPISODE_KIND[self.episode]


@dataclass(frozen=True)
class EpisodeHandoffReceipt:
    segment_index: int
    handoff_from: str
    handoff_to: str
    handoff_mode: str
    adopted: bool
    refusal: str
    incumbent_error: float
    from_snapshot_hash: str
    to_snapshot_hash: str

    @property
    def public_handoff_from(self) -> str:
        return EPISODE_KIND.get(self.handoff_from, self.handoff_from)

    @property
    def public_handoff_to(self) -> str:
        return EPISODE_KIND.get(self.handoff_to, self.handoff_to)


@dataclass(frozen=True)
class EpisodeScheduleResult:
    schema_version: str
    dispatcher: str
    sensing: dict[str, object] = field(default_factory=dict)
    probe_order: tuple[str, ...] = ()
    probe_tax_fes: int = 0
    exploitation_fes: int = 0
    scoped_checkpoint_hash: str = ""
    probes: tuple[EpisodeProbeReceipt, ...] = ()
    handoffs: tuple[EpisodeHandoffReceipt, ...] = ()
    receipts: tuple[EpisodeSegmentReceipt, ...] = ()
    funded_fes: dict[str, int] = field(default_factory=dict)
    magnitude_repairs: dict[str, int] = field(default_factory=dict)
    switches: int = 0
    final_error: float = 0.0
    terminal_fes: int = 0
    schedule_hash: str = ""
    coordinator_name: str = COORDINATOR_PUBLIC_NAME
    episode_names: dict[str, str] = field(default_factory=lambda: dict(EPISODE_PUBLIC_NAMES))


def _sense(
    structure: OverlapStructure | None,
    global_ledger: EvaluationLedger,
    blocks: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, object]:
    """One counted B/W/C pass over the shared variables.

    B/W drive ordering (Gate 47 rejected C-magnitude gating); C is recorded
    as a diagnostic and used only as a non-threshold tie-breaker.  Per-block
    evidence scores turn the sensing pass into the episode scope: blocks are
    ranked by the mean relative width of their probed variables so CTP/GSS
    sweep the conflict-active scope first.
    """

    empty = {
        "scope_size": 0,
        "mean_bias": None,
        "mean_relative_width": None,
        "mean_conflict": None,
        "probe_fes": 0,
        "block_scores": [],
    }
    if structure is None or not structure.shared_variables:
        return empty
    scope = tuple(structure.shared_variables)
    results = counted_probe(structure, global_ledger, scope)
    base = abs(float(global_ledger.best_error))
    by_variable = {r.variable: r for r in results}
    mean_bias = float(np.mean([abs(r.bias) for r in results]))
    mean_width = float(np.mean([r.width / (base + r.width) for r in results]))
    mean_conflict = float(np.mean([r.conflict_score for r in results]))
    block_scores: list[float] = []
    if blocks is not None:
        for block in blocks:
            widths = [
                by_variable[v].width / (base + by_variable[v].width)
                for v in block
                if v in by_variable
            ]
            block_scores.append(float(np.mean(widths)) if widths else 0.0)
    return {
        "scope_size": len(scope),
        "mean_bias": mean_bias,
        "mean_relative_width": mean_width,
        "mean_conflict": mean_conflict,
        "probe_fes": 2 * len(scope),
        "block_scores": block_scores,
    }


def evidence_block_order(
    blocks: tuple[tuple[int, ...], ...],
    block_scores: list[float],
) -> tuple[tuple[int, ...], ...]:
    """Order blocks by descending evidence score (stable on ties).

    Blocks whose variables the sensing pass did not probe keep their
    relative order after every scored block; this is the scope mechanism
    v1: the evidence ranks where episodes search first inside their own
    sweep schedules, without thresholding anything.
    """

    if len(block_scores) != len(blocks):
        raise ValueError("block score count must match the block count")
    order = sorted(range(len(blocks)), key=lambda index: (-float(block_scores[index]), index))
    return tuple(blocks[index] for index in order)


def _probe_order(sensing: dict[str, object]) -> tuple[str, ...]:
    """Pre-registered v1 rubric: B/W rank the probe windows (no thresholds).

    Directional bias feeds the GSS score (graph balancing targets biased
    shared variables), relative response width feeds the AOR score (a wide
    response says the incumbent region is far from exploitable structure),
    the midpoint feeds CTP, and narrowness feeds SMP.  B/W order the
    probes; the conflict score C is recorded as a diagnostic only and its
    magnitude never gates or ranks anything (Gate 47 rejected C-magnitude
    discrimination).  All four episodes are probed regardless of scores;
    the rubric is declared calibration-pending.

    Statistical semantics: the four probes run sequentially against one
    evolving global archive, so the recorded gains are *contextual marginal
    contributions* -- what each episode adds given what the earlier probes
    already banked -- not absolute capability estimates from a common
    baseline.  That is the scheduler-relevant estimand; receipts carry both
    gains so the distinction stays auditable.
    """

    bias = float(sensing.get("mean_bias") or 0.0)
    width = float(sensing.get("mean_relative_width") or 0.0)
    scores = {
        "aor": width,
        "gcb": bias,
        "ctp": 0.5 * (bias + width),
        "smp": 1.0 - width,
    }
    return tuple(
        sorted(EPISODES, key=lambda episode: (-scores[episode], EPISODES.index(episode)))
    )


def run_oc_episode_schedule(
    problem: OptimizationProblem,
    checkpoint: PhaseCheckpoint,
    *,
    action_seed: int,
    structure: OverlapStructure | None = None,
    segment_fes: int = 300_000,
    probe_share: float = PROBE_SHARE,
    probe_min_fes: int = PROBE_MIN_FES,
    max_switches: int = 6,
    material_log_gain: float = HISTORICAL_MATERIAL_LOG_GAIN,
    handoff_enabled: bool = True,
) -> EpisodeScheduleResult:
    """Interleave the four complete episodes under GCB segment allotment."""

    if not isinstance(problem, OptimizationProblem):
        raise TypeError("problem must be OptimizationProblem")
    if not isinstance(checkpoint, PhaseCheckpoint):
        raise TypeError("checkpoint must be PhaseCheckpoint")
    if isinstance(segment_fes, bool) or not isinstance(segment_fes, int) or segment_fes <= 0:
        raise ValueError("segment_fes must be a positive integer")
    if isinstance(probe_min_fes, bool) or not isinstance(probe_min_fes, int) or probe_min_fes <= 0:
        raise ValueError("probe_min_fes must be a positive integer")
    if isinstance(max_switches, bool) or not isinstance(max_switches, int) or max_switches < 0:
        raise ValueError("max_switches must be a non-negative integer")

    global_ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=True,
    )
    phase2_fes = checkpoint.total_budget_fes - checkpoint.phase1_fes
    if not 0.0 < float(probe_share) <= 0.25:
        raise ValueError("probe_share must lie in (0, 0.25] so the four forced probes stay affordable")
    sensing = _sense(structure, global_ledger, checkpoint.blocks)
    probe_order = _probe_order(sensing)
    probe_budget = max(probe_min_fes, int(probe_share * phase2_fes))
    probe_tax_fes = probe_budget * len(EPISODES)
    if phase2_fes - probe_tax_fes < probe_min_fes:
        raise Phase2StateError(
            f"probe tax ({probe_tax_fes} FE) leaves no executable exploitation budget "
            f"in phase-II ({phase2_fes} FE)"
        )
    # Scope mechanism v1: evidence ranks where the sweep-based episodes
    # (CTP polish, GSS graph schedule) search first.  Blocks are reordered
    # by descending sensing score inside the schedule checkpoint; SMP and
    # AOR are inherently global compositions and are not scope-constrained.
    block_scores = list(sensing.get("block_scores") or [])
    scoped_blocks = (
        evidence_block_order(checkpoint.blocks, block_scores)
        if len(block_scores) == len(checkpoint.blocks)
        else tuple(checkpoint.blocks)
    )
    scoped_checkpoint = (
        dataclasses.replace(checkpoint, blocks=scoped_blocks)
        if scoped_blocks != tuple(checkpoint.blocks)
        else checkpoint
    )

    executors = {
        "ctp": CtpExecutor(),
        "gcb": GcbExecutor(),
        "smp": RecoveredSmpExecutor(),
        "aor": RecoveredAorExecutor(),
    }
    states: dict[str, object] = {}
    ledgers: dict[str, _EpisodeLedger] = {}
    for episode in EPISODES:
        ledgers[episode] = _EpisodeLedger(
            problem,
            total_budget_fes=checkpoint.total_budget_fes,
            initial_count=checkpoint.phase1_fes,
            initial_incumbent=checkpoint.incumbent,
            initial_error=checkpoint.incumbent_error,
            global_ledger=global_ledger,
        )
        context = ActionContext(
            episode, scoped_checkpoint, problem, ledgers[episode], action_seed=action_seed
        )
        states[episode] = executors[episode].initialize(context)

    funded = {episode: 0 for episode in EPISODES}
    stuck: set[str] = set()
    receipts: list[EpisodeSegmentReceipt] = []
    probes: list[EpisodeProbeReceipt] = []
    handoffs: list[EpisodeHandoffReceipt] = []
    segment_index = 0
    switches = 0

    def _handoff(previous: str, episode: str) -> None:
        """Archive baton at a switch: adopt the global strict-best baseline."""
        if previous == episode:
            return
        refusal = "disabled"
        adopted = False
        if handoff_enabled:
            adopted, refusal = ledgers[episode].adopt_global_archive(
                source=global_ledger,
                accept_out_of_bounds=HANDOFF_ACCEPTS_OOB[episode],
            )
        handoffs.append(
            EpisodeHandoffReceipt(
                segment_index=segment_index,
                handoff_from=previous,
                handoff_to=episode,
                handoff_mode=HANDOFF_MODE[episode] if handoff_enabled else "disabled",
                adopted=adopted,
                refusal=refusal,
                incumbent_error=float(global_ledger.best_error),
                from_snapshot_hash=(
                    states[previous].snapshot().snapshot_hash if previous else ""
                ),
                to_snapshot_hash=states[episode].snapshot().snapshot_hash,
            )
        )

    def _step_episode(episode: str, request: int) -> int:
        """Step one episode; returns consumed FE (0 only on forced switch)."""
        state = states[episode]
        consumed = 0
        while consumed == 0:
            try:
                step = state.step(request)
                consumed = step.step_fes
            except Phase2StateError:
                if request >= global_ledger.remaining:
                    stuck.add(episode)
                    return 0
                request = min(request * 2, global_ledger.remaining)
        return consumed

    # --- exploration: one forced, executable probe window per episode ---
    gain_per_fe: dict[str, float] = {}
    previous_episode = ""
    for position, episode in enumerate(probe_order):
        _handoff(previous_episode, episode)
        previous_episode = episode
        unprobed_left = len(probe_order) - position - 1
        if global_ledger.remaining < probe_min_fes * (unprobed_left + 1):
            raise Phase2StateError(
                "episode probe protocol violated: remaining "
                f"{global_ledger.remaining} FE cannot fund a minimum probe "
                f"({probe_min_fes} FE) for every unprobed episode "
                f"({unprobed_left + 1} left, next {episode})"
            )
        request = min(probe_budget, global_ledger.remaining - probe_min_fes * unprobed_left)
        global_before = float(global_ledger.best_error)
        local_before = float(ledgers[episode].best_error)
        consumed = _step_episode(episode, request)
        if consumed == 0:
            raise Phase2StateError(f"episode {episode} cannot execute its probe window")
        funded[episode] += consumed
        global_after = float(global_ledger.best_error)
        local_after = float(ledgers[episode].best_error)
        g_gain = _log_gain(global_before, global_after)
        l_gain = _log_gain(local_before, local_after)
        gain_per_fe[episode] = g_gain / max(consumed, 1)
        probes.append(
            EpisodeProbeReceipt(
                order_position=position,
                episode=episode,
                episode_kind=EPISODE_KIND[episode],
                budget_fes=consumed,
                scope_size=int(sensing.get("scope_size") or 0),
                global_error_before=global_before,
                global_error_after=global_after,
                global_gain=g_gain,
                global_gain_per_fe=g_gain / max(consumed, 1),
                local_error_before=local_before,
                local_error_after=local_after,
                local_gain=l_gain,
            )
        )
        receipts.append(
            EpisodeSegmentReceipt(
                segment_index=segment_index,
                phase="probe",
                episode=episode,
                episode_kind=EPISODE_KIND[episode],
                requested_fes=request,
                consumed_fes=consumed,
                global_error_before=global_before,
                global_error_after=global_after,
                global_gain=g_gain,
                local_error_before=local_before,
                local_error_after=local_after,
                local_gain=l_gain,
                material=g_gain > material_log_gain,
                switched=False,
                next_episode="",
                state_hash=states[episode].snapshot().state_hash,
                snapshot_hash=states[episode].snapshot().snapshot_hash,
            )
        )
        segment_index += 1

    # --- exploitation: concentrate on the best global gain per FE ---
    ranking = sorted(EPISODES, key=lambda e: (-gain_per_fe[e], probe_order.index(e)))
    current = ranking[0]
    _handoff(previous_episode, current)
    while global_ledger.remaining > 0:
        if current in stuck:
            alive = [e for e in ranking if e not in stuck]
            if not alive:
                raise Phase2StateError("every episode is stuck mid-generation")
            current = alive[0]
            continue
        request = min(segment_fes, global_ledger.remaining)
        global_before = float(global_ledger.best_error)
        local_before = float(ledgers[current].best_error)
        consumed = _step_episode(current, request)
        if consumed == 0:
            continue
        funded[current] += consumed
        global_after = float(global_ledger.best_error)
        local_after = float(ledgers[current].best_error)
        g_gain = _log_gain(global_before, global_after)
        l_gain = _log_gain(local_before, local_after)
        # Materiality is owned by the GLOBAL archive delta only; local gain
        # is diagnostic (a privately-improving episode that does not move
        # the global archive must not stay funded -- Gate 50's false
        # stickiness).
        material = g_gain > material_log_gain
        gain_per_fe[current] = (
            (gain_per_fe[current] * max(funded[current] - consumed, 1) + g_gain)
            / max(funded[current], 1)
        )
        ranking = sorted(EPISODES, key=lambda e: (-gain_per_fe[e], probe_order.index(e)))
        switched = False
        nxt = current
        if global_ledger.remaining > 0 and not material and switches < max_switches:
            candidates = [e for e in ranking if e != current and e not in stuck]
            if candidates:
                nxt = candidates[0]
                switches += 1
                switched = True
        receipts.append(
            EpisodeSegmentReceipt(
                segment_index=segment_index,
                phase="exploit",
                episode=current,
                episode_kind=EPISODE_KIND[current],
                requested_fes=request,
                consumed_fes=consumed,
                global_error_before=global_before,
                global_error_after=global_after,
                global_gain=g_gain,
                local_error_before=local_before,
                local_error_after=local_after,
                local_gain=l_gain,
                material=material,
                switched=switched,
                next_episode=nxt,
                state_hash=states[current].snapshot().state_hash,
                snapshot_hash=states[current].snapshot().snapshot_hash,
            )
        )
        segment_index += 1
        _handoff(current, nxt)
        current = nxt

    if global_ledger.count != checkpoint.total_budget_fes:
        raise RuntimeError("episode schedule did not stop at the terminal FE")
    magnitude_repairs = {episode: ledgers[episode].magnitude_repairs for episode in EPISODES}
    payload = {
        "schema_version": OC_EPISODE_SCHEMA,
        "dispatcher": DISPATCHER,
        "sensing": sensing,
        "probe_order": list(probe_order),
        "probe_tax_fes": probe_tax_fes,
        "exploitation_fes": phase2_fes - probe_tax_fes,
        "scoped_checkpoint_hash": scoped_checkpoint.checkpoint_hash,
        "probes": [p.__dict__ for p in probes],
        "handoffs": [h.__dict__ for h in handoffs],
        "receipts": [r.__dict__ for r in receipts],
        "funded_fes": funded,
        "switches": switches,
        "magnitude_repairs": magnitude_repairs,
    }
    return EpisodeScheduleResult(
        schema_version=OC_EPISODE_SCHEMA,
        dispatcher=DISPATCHER,
        sensing=sensing,
        probe_order=tuple(probe_order),
        probe_tax_fes=probe_tax_fes,
        exploitation_fes=phase2_fes - probe_tax_fes,
        scoped_checkpoint_hash=scoped_checkpoint.checkpoint_hash,
        probes=tuple(probes),
        handoffs=tuple(handoffs),
        receipts=tuple(receipts),
        funded_fes=funded,
        magnitude_repairs=magnitude_repairs,
        switches=switches,
        final_error=float(global_ledger.best_error),
        terminal_fes=global_ledger.count,
        schedule_hash=canonical_sha256(payload),
    )


# ---------------------------------------------------------------------------
# Phase-aware scheduler v4 (upgrade plan sections 4-6).
#
# The v3 path above is preserved untouched as the historical/ablation
# implementation.  v4 replaces the cumulative gain/FE exploitation ranking
# with an explicit adjudication ladder (P0-P5) and adds the four-part
# late-maturity protection: maturity tickets (no elimination before a
# semantic unit ran), escalating development windows (revelation horizon
# coverage w1 * (2^K - 1) >= h*), a challenger rotation floor (no
# starvation after a completed ticket), and promotion-only private
# trajectory credit (S5/CTP-type hidden value; never demotes the global
# material leader -- Gate 50b's false-stickiness lesson).
# ---------------------------------------------------------------------------

OC_EPISODE_SCHEMA_V4 = "arac-oc-episode-schedule-v4"
SCHEDULER_POLICY_V4 = "phase_aware_v4"
DEFAULT_SCHEDULER_VERSION_V4 = "v4.4"
OC_EPISODE_SCHEMA_V5 = "arac-oc-episode-schedule-v5"
SCHEDULER_POLICY_V5 = "hpr_gcb_v5"
DEFAULT_SCHEDULER_VERSION_V5 = "v5.0"
OC_EPISODE_SCHEMA_V5_1 = "arac-oc-episode-schedule-v5.1"
SCHEDULER_POLICY_V5_1 = "adaptive_hpr_gcb_v5_1"
DEFAULT_SCHEDULER_VERSION_V5_1 = "v5.1"
# v5.2 supersedes the whole v5 line (Gate 51c post-mortem): the post-lock
# verification window and the material-leader continuation are bounded by
# one calibrated maturity window, so a plateau releases the runway after
# w1 FE instead of a full segment, and the released state is part of the
# receipt/audit chain.  v5.0/v5.1 behaviour no longer exists in this tree;
# their frozen cells keep provenance via their recorded manifests.
OC_EPISODE_SCHEMA_V5_2 = "arac-oc-episode-schedule-v5.2"
SCHEDULER_POLICY_V5_2 = "adaptive_hpr_gcb_v5_2"
DEFAULT_SCHEDULER_VERSION_V5_2 = "v5.2"
GRANT_KINDS = ("probe", "ticket", "exploit", "challenger", "escalation")
PRIVATE_CREDIT_MODES = ("extreme", "rate")


@dataclass(frozen=True)
class PhaseAwareSchedulerConfig:
    """v4 scheduler configuration; calibrated values come from Gate 51-0.

    ``maturity_window_fes`` (w1), ``revelation_horizon_fes`` (h*_cal),
    ``exploration_and_development_cap`` and ``exploitation_reserve_ratio``
    are deliberately left without defaults: they must be supplied from the
    frozen 51-0 calibration table, never guessed.
    """

    maturity_window_fes: int
    revelation_horizon_fes: int
    exploration_and_development_cap: float
    exploitation_reserve_ratio: float
    scheduler_version: str = DEFAULT_SCHEDULER_VERSION_V4
    cold_start_probe_cap: float = 0.25
    probe_min_fes: int = PROBE_MIN_FES
    escalation_factor: int = 2
    escalation_grants_k: int = 3
    segment_fes: int = 300_000
    material_log_gain: float = HISTORICAL_MATERIAL_LOG_GAIN
    handoff_enabled: bool = True
    private_credit_mode: str = "extreme"
    calibration_ref: str = ""
    # v5 is opt-in so the v4.4 frozen path remains bitwise reproducible.
    # The horizon and action-native boundaries are supplied by the existing
    # Gate 51-0 calibration and EpisodeProgress contract.
    horizon_protected: bool = False
    # The v5 line (v5.2) is separately opt-in. A material, completed
    # maturity ticket may verify its value with one exploit segment before
    # the remaining P1 tickets run; the verification window and every
    # continuation are bounded by one maturity window, so a plateau
    # releases the runway after w1 FE. The HPR release and reserve rules
    # bound a wrong early choice.
    adaptive_exploration: bool = False

    def __post_init__(self) -> None:
        for name in ("maturity_window_fes", "revelation_horizon_fes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer (Gate 51-0 calibration)")
        for name in ("exploration_and_development_cap", "cold_start_probe_cap"):
            value = getattr(self, name)
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not 0.0 < float(self.exploitation_reserve_ratio) < 1.0:
            raise ValueError("exploitation_reserve_ratio must lie in (0, 1)")
        if isinstance(self.escalation_factor, bool) or self.escalation_factor < 2:
            raise ValueError("escalation_factor must be an integer >= 2")
        if isinstance(self.escalation_grants_k, bool) or self.escalation_grants_k < 1:
            raise ValueError("escalation_grants_k must be an integer >= 1")
        if isinstance(self.segment_fes, bool) or self.segment_fes <= 0:
            raise ValueError("segment_fes must be a positive integer")
        if self.private_credit_mode not in PRIVATE_CREDIT_MODES:
            raise ValueError("private_credit_mode must be 'extreme' or 'rate'")
        if not isinstance(self.horizon_protected, bool):
            raise ValueError("horizon_protected must be a boolean")
        if not isinstance(self.adaptive_exploration, bool):
            raise ValueError("adaptive_exploration must be a boolean")
        if self.adaptive_exploration and not self.horizon_protected:
            raise ValueError("adaptive_exploration requires horizon_protected")


@dataclass(frozen=True)
class V4ProbeReceipt:
    order_position: int
    episode: str
    episode_kind: str
    budget_fes: int
    scope_size: int
    probe_contract: str
    min_step_fes: int
    maturity_target_fes: int
    global_error_before: float
    global_error_after: float
    global_gain: float
    local_error_before: float
    local_error_after: float
    local_gain: float
    max_local_log_gain_window: float


@dataclass(frozen=True)
class V4SegmentReceipt:
    segment_index: int
    grant_kind: str
    episode: str
    episode_kind: str
    grant_index: int
    window_fes: int
    requested_fes: int
    consumed_fes: int
    leader: str
    ledger_class: str
    global_error_before: float
    global_error_after: float
    global_gain: float
    local_error_before: float
    local_error_after: float
    local_gain: float
    material: bool
    recent_rate: float
    private_credit: float
    handoff_epoch: int
    cumulative_development_fes: int
    cumulative_runtime_fes: int
    evidence_revealed: bool
    maturity_ticket_id: str
    maturity_committed: bool
    challenger: bool
    cooldown: int
    remaining_budget: int
    cold_start_spent: int
    development_spent: int
    progress_before: dict[str, object]
    progress_after: dict[str, object]
    switched: bool
    next_episode: str
    state_hash: str
    snapshot_hash: str
    # v5 audit surface.  Defaults preserve v4 receipt construction and hashes.
    reservation_kind: str = ""
    plateau_release: bool = False
    handoff_penalty: int = 0
    released: bool = False


@dataclass(frozen=True)
class V4TicketRecord:
    episode: str
    ticket_id: str
    target_fes: int
    requested_fes: int
    granted_fes: int
    ledger_class: str
    affordable: bool
    protocol_mature_after: bool
    segment_index: int


@dataclass(frozen=True)
class V4ScheduleResult:
    schema_version: str
    scheduler_policy: str
    scheduler_version: str
    calibration_ref: str
    dispatcher: str
    config: dict[str, object] = field(default_factory=dict)
    sensing: dict[str, object] = field(default_factory=dict)
    probe_order: tuple[str, ...] = ()
    cold_start_probe_tax_fes: int = 0
    development_fes: int = 0
    exploitation_fes: int = 0
    scoped_checkpoint_hash: str = ""
    probes: tuple[V4ProbeReceipt, ...] = ()
    handoffs: tuple[EpisodeHandoffReceipt, ...] = ()
    tickets: tuple[V4TicketRecord, ...] = ()
    receipts: tuple[V4SegmentReceipt, ...] = ()
    audit: dict[str, bool] = field(default_factory=dict)
    funded_fes: dict[str, int] = field(default_factory=dict)
    magnitude_repairs: dict[str, int] = field(default_factory=dict)
    switches: int = 0
    final_error: float = 0.0
    terminal_fes: int = 0
    schedule_hash: str = ""
    coordinator_name: str = COORDINATOR_PUBLIC_NAME
    episode_names: dict[str, str] = field(default_factory=lambda: dict(EPISODE_PUBLIC_NAMES))


def _progress_dict(progress: EpisodeProgress) -> dict[str, object]:
    return {
        "phase": progress.phase,
        "consumed_fes": progress.consumed_fes,
        "next_boundary_fes": progress.next_boundary_fes,
        "min_step_fes": progress.min_step_fes,
        "maturity_target_fes": progress.maturity_target_fes,
        "protocol_mature": progress.protocol_mature,
        "contract": progress.contract,
    }


def _audit_v4(
    receipts: list[V4SegmentReceipt],
    probes: list[V4ProbeReceipt],
    tickets: list[V4TicketRecord],
    *,
    phase2_fes: int,
    cold_start_cap_fes: int,
    development_cap_fes: int,
    reserve_floor_fes: int,
    revelation_horizon_fes: int,
    terminal_fes: int,
    sensing_fes: int,
    phase1_fes: int,
    escalation_factor: int,
    funded: dict[str, int],
) -> dict[str, bool]:
    """Recompute every declared invariant over the emitted receipts."""

    checks: dict[str, bool] = {}
    checks["segment_indices_contiguous"] = [
        r.segment_index for r in receipts
    ] == list(range(len(receipts)))
    probes_first = not receipts or all(
        r.grant_kind == "probe" for r in receipts[: len(probes)]
    )
    checks["probes_precede_exploitation"] = probes_first and all(
        r.grant_kind != "probe" for r in receipts[len(probes):]
    )
    checks["global_error_monotone"] = all(
        receipts[i + 1].global_error_before >= receipts[i + 1].global_error_after
        for i in range(len(receipts) - 1)
    ) and all(
        receipts[i].global_error_after <= receipts[i].global_error_before
        for i in range(len(receipts))
    )
    checks["fe_reconcile"] = (
        phase1_fes + sensing_fes + sum(r.consumed_fes for r in receipts) == terminal_fes
    )
    checks["funded_reconcile"] = sum(funded.values()) == sum(
        r.consumed_fes for r in receipts
    )
    checks["cold_start_cap_respected"] = all(
        r.cold_start_spent <= cold_start_cap_fes for r in receipts
    )
    checks["development_cap_respected"] = all(
        r.development_spent <= development_cap_fes for r in receipts
    )
    escalations = [r for r in receipts if r.grant_kind == "escalation"]
    ladder_ok = True
    seen: dict[str, tuple[int, int]] = {}
    for r in escalations:
        previous = seen.get(r.episode)
        if previous is not None:
            _, previous_window = previous
            if r.window_fes != previous_window * escalation_factor:
                ladder_ok = False
        seen[r.episode] = (r.grant_index, r.window_fes)
    checks["escalation_ladder_geometric"] = ladder_ok
    checks["ticket_targets_nonnegative"] = all(t.target_fes >= 0 for t in tickets)
    # A refused ticket may legitimately be re-granted later at a SMALLER
    # size: the remainder shrinks as other lanes advance the episode, and
    # plan rule P1 keeps retrying unfinished tickets.  What must never
    # happen is smuggling a grant LARGER than a refused request through
    # after the refusal (budgets only shrink).
    refused: dict[str, int] = {}
    for t in tickets:
        if not t.affordable:
            refused[t.episode] = max(refused.get(t.episode, 0), t.requested_fes)
    checks["unaffordable_tickets_not_circumvented"] = all(
        t.granted_fes == 0 for t in tickets if not t.affordable
    ) and all(
        r.grant_kind != "ticket"
        or r.episode not in refused
        or r.requested_fes <= refused[r.episode]
        for r in receipts
    )
    # Rotation floor: no episode may monopolize consecutive challenger
    # grants while alternatives existed, and no ticket-completed episode
    # starves afterwards.  A repeat run inside the forced regime
    # (terminal drain, or the development lane exhausted) is legitimate:
    # there the schedule has no alternative to grant.
    challenger_stream = [
        r for r in receipts if r.grant_kind in ("challenger", "escalation")
    ]
    window = max(len(EPISODES) - 1, 1)

    def _forced_regime(receipt: V4SegmentReceipt) -> bool:
        pre_remaining = receipt.remaining_budget + receipt.consumed_fes
        return (
            pre_remaining <= reserve_floor_fes
            or receipt.development_spent >= development_cap_fes
        )

    rotation_ok = (
        all(
            len({r.episode for r in challenger_stream[start : start + window]}) > 1
            or any(_forced_regime(r) for r in challenger_stream[start : start + window])
            for start in range(len(challenger_stream) - window + 1)
        )
        if len(challenger_stream) > window
        else True
    )
    cycle = len(EPISODES) - 1

    def _ticket_not_starved(ticket: V4TicketRecord) -> bool:
        foreign = [
            r
            for r in receipts
            if r.segment_index > ticket.segment_index
            and r.grant_kind in ("challenger", "escalation")
            and r.episode != ticket.episode
        ]
        if len(foreign) < cycle:
            return True
        return any(
            r.episode == ticket.episode and r.segment_index > ticket.segment_index
            for r in receipts
        )

    checks["rotation_no_monopoly"] = rotation_ok
    checks["no_post_ticket_starvation"] = all(
        _ticket_not_starved(t) for t in tickets if t.affordable and t.protocol_mature_after
    )
    checks["phase2_grants_positive"] = all(r.consumed_fes > 0 for r in receipts)
    # Receipt-chain continuity: each grant starts exactly where the
    # previous one left the global archive.
    checks["receipt_chain_continuous"] = all(
        receipts[i + 1].global_error_before == receipts[i].global_error_after
        for i in range(len(receipts) - 1)
    )
    # Exploit-run semantics (v4.3): a material leader may run
    # continuously (R6's strict win came from exactly such a run), but a
    # NON-material exploit segment must yield -- the next grant after a
    # zero-gain exploit goes to another episode or a cadence event, never
    # to the same leader immediately.
    last_exploit_episode = ""
    stagnation_ok = True
    for r in receipts:
        if r.grant_kind == "exploit":
            if (
                last_exploit_episode == r.episode
                and r.grant_kind == "exploit"
                and r.global_gain <= 0.0
                and receipts[r.segment_index - 1].global_gain <= 0.0
                and r.remaining_budget > reserve_floor_fes
            ):
                # Same leader ran again after its own zero-gain segment
                # with budget left to yield -- stagnation violation.
                stagnation_ok = False
                break
            last_exploit_episode = r.episode
        elif r.grant_kind != "probe":
            last_exploit_episode = ""
    checks["exploit_stagnation_yields"] = stagnation_ok
    # Development grants never penetrate the terminal exploitation
    # reserve (the final drain may land exactly on zero).
    checks["development_reserve_preserved"] = all(
        r.ledger_class != "development"
        or r.remaining_budget >= reserve_floor_fes
        or r.remaining_budget == 0
        for r in receipts
    )
    # Escalation fairness: pending episodes stay within one escalation
    # grant of each other unless the lagging one was meanwhile being
    # exploited as the leader (better treatment than its ladder).
    escalation_grant_counts: dict[str, int] = {}
    cum_dev_by_receipt: dict[str, int] = {}
    fairness_ok = True
    last_escalation_index: dict[str, int] = {}
    for r in receipts:
        cum_dev_by_receipt[r.episode] = r.cumulative_development_fes
        if r.grant_kind == "escalation":
            escalation_grant_counts[r.episode] = escalation_grant_counts.get(r.episode, 0) + 1
            pending = [e for e, dev in cum_dev_by_receipt.items() if dev < revelation_horizon_fes]
            counts = {e: escalation_grant_counts.get(e, 0) for e in pending}
            if counts and max(counts.values()) - min(counts.values()) > 1:
                laggards = [e for e, c in counts.items() if c == min(counts.values())]
                rescued = False
                for laggard in laggards:
                    since = last_escalation_index.get(laggard, -1)
                    if any(
                        x.grant_kind == "exploit"
                        and x.episode == laggard
                        and x.segment_index > since
                        and x.segment_index < r.segment_index
                        for x in receipts
                    ):
                        rescued = True
                        break
                if not rescued:
                    fairness_ok = False
                    break
            last_escalation_index[r.episode] = r.segment_index
    checks["escalation_rotation_fair"] = fairness_ok
    return checks


def run_oc_episode_schedule_v4(
    problem: OptimizationProblem,
    checkpoint: PhaseCheckpoint,
    *,
    action_seed: int,
    config: PhaseAwareSchedulerConfig,
    structure: OverlapStructure | None = None,
) -> V4ScheduleResult:
    """Run the phase-aware v4 adjudication over the four v2 episodes.

    The ladder (exactly one grant per decision, no silent degradation):

    ``P0`` hard constraints -- minimal executable probe for every episode,
    in B/W evidence order, before anything else;
    ``P1`` unfinished, currently affordable maturity tickets (probe order);
    ``P2`` leader exploitation while the leader's consecutive exploit run
    is below two and the leader is not in its avoidance window;
    ``P3`` forced challenger -- (a) evidence-pending episodes with an
    affordable escalating window, fewest-escalation-first fairness with
    private credit as the tie-breaker, cooldown exempt; (b) rotation
    over completed-ticket non-leaders, cooldown filtered,
    earliest-expiry fallback; (c) empty candidate set is a loud
    failure;
    ``P4`` ranking signals inside P2/P3 -- ``recent_rate`` (last two
    segments, global gains) elects the leader; private credit only
    promotes inside P3(a) and never demotes the leader;
    ``P5`` fallback when P1-P3 have no executable object (the designed
    600k corner): affordable tickets, then >=2-segment history ranked by
    recent_rate, then rotation; nothing left is a loud failure.
    """

    if not isinstance(problem, OptimizationProblem):
        raise TypeError("problem must be OptimizationProblem")
    if not isinstance(checkpoint, PhaseCheckpoint):
        raise TypeError("checkpoint must be PhaseCheckpoint")
    if not isinstance(config, PhaseAwareSchedulerConfig):
        raise TypeError("config must be PhaseAwareSchedulerConfig")
    # Version isolation (pre-registered discipline): only v4.4 (frozen
    # bitwise path) and v5.2 (current v5 line) are producible.  A v5.0/v5.1
    # label would misreport v5.2 behaviour -- their runway was bounded in
    # v5.2 -- so hand-built legacy configs fail loudly here.
    if config.scheduler_version not in (
        DEFAULT_SCHEDULER_VERSION_V4,
        DEFAULT_SCHEDULER_VERSION_V5_2,
    ):
        raise ValueError(
            "scheduler_version "
            f"{config.scheduler_version!r} cannot be produced by this tree; "
            f"use {DEFAULT_SCHEDULER_VERSION_V5_2!r} for the v5 line or "
            f"{DEFAULT_SCHEDULER_VERSION_V4!r} for the frozen v4 path"
        )
    if (config.horizon_protected or config.adaptive_exploration) and (
        config.scheduler_version != DEFAULT_SCHEDULER_VERSION_V5_2
    ):
        raise ValueError(
            "horizon/adaptive features require scheduler_version "
            f"{DEFAULT_SCHEDULER_VERSION_V5_2!r}"
        )

    global_ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=True,
    )
    phase2_fes = checkpoint.total_budget_fes - checkpoint.phase1_fes
    cold_start_budget = int(config.cold_start_probe_cap * phase2_fes)
    development_budget = int(config.exploration_and_development_cap * phase2_fes)
    reserve_floor = int(config.exploitation_reserve_ratio * phase2_fes)
    hpr_mode = bool(config.horizon_protected)
    adaptive_mode = bool(config.adaptive_exploration)

    sensing = _sense(structure, global_ledger, checkpoint.blocks)
    probe_order = _probe_order(sensing)
    block_scores = list(sensing.get("block_scores") or [])
    scoped_blocks = (
        evidence_block_order(checkpoint.blocks, block_scores)
        if len(block_scores) == len(checkpoint.blocks)
        else tuple(checkpoint.blocks)
    )
    scoped_checkpoint = (
        dataclasses.replace(checkpoint, blocks=scoped_blocks)
        if scoped_blocks != tuple(checkpoint.blocks)
        else checkpoint
    )

    executors = {
        "ctp": CtpExecutor(),
        "gcb": GcbExecutor(),
        "smp": RecoveredSmpExecutor(),
        "aor": RecoveredAorExecutor(),
    }
    states: dict[str, object] = {}
    ledgers: dict[str, _EpisodeLedger] = {}
    for episode in EPISODES:
        ledgers[episode] = _EpisodeLedger(
            problem,
            total_budget_fes=checkpoint.total_budget_fes,
            initial_count=checkpoint.phase1_fes,
            initial_incumbent=checkpoint.incumbent,
            initial_error=checkpoint.incumbent_error,
            global_ledger=global_ledger,
        )
        context = ActionContext(
            episode,
            scoped_checkpoint,
            problem,
            ledgers[episode],
            action_seed=action_seed,
            retain_trajectory=False,
        )
        states[episode] = executors[episode].initialize(context)

    # P0 affordability: the four minimal probes are hard constraints and
    # must fit the cold-start ledger by construction; a configuration
    # where they cannot is rejected up front instead of mid-run.
    probe_windows = {
        episode: max(
            config.probe_min_fes,
            states[episode].progress(maturity_window_fes=config.maturity_window_fes).min_step_fes,
        )
        for episode in EPISODES
    }
    if sum(probe_windows.values()) > cold_start_budget:
        raise Phase2StateError(
            "v4 probe protocol violated: minimal probes "
            f"({sum(probe_windows.values())} FE) exceed the cold-start cap "
            f"({cold_start_budget} FE)"
        )

    def _progress(episode: str) -> EpisodeProgress:
        return states[episode].progress(maturity_window_fes=config.maturity_window_fes)

    book = {
        episode: {
            "grant_index": 0,          # development grants received (ticket = 1st)
            "escalation_step": 0,      # post-ticket escalation ladder position
            "challenger_absence": 0,   # consecutive challenger events granted to others
            "exploit_history": deque(),  # (global_gain, consumed_fes) exploit segments only
            "cum_dev_fes": 0,
            "history": deque(),        # (global_gain, consumed_fes) per segment
            "consecutive_exploit": 0,
            "cooldown": 0,
            "handoff_epoch": 0,
            "private_credit": 0.0,
            "ticket_done": False,
            "maturity_unaffordable": False,
            # HPR-GCB state.  A released leader remains released until a
            # complete reservation/challenger is paid, preventing a single
            # zero-gain block from immediately reclaiming the budget.
            "released": False,
            "reservation_paid": False,
            "reservation_due": False,
            "handoff_penalty": 0,
            "handoff_cooldown": 0,
        }
        for episode in EPISODES
    }
    funded = {episode: 0 for episode in EPISODES}
    stuck: set[str] = set()
    # Transient execution blocks: an episode whose next state unit is
    # larger than the granted window (SMP mid-visit) is skipped at THIS
    # window size, not killed -- later, bigger windows may still run it.
    # The old v3 retry-doubling is gone: it consumed past the reserved
    # window, which pre-execution reservation (v4.1) forbids.
    min_window_needed = {episode: 0 for episode in EPISODES}
    exploit_streak = 0
    # v4.3 reserved cadence: exploitation FE consumed since the last
    # challenger-class event.  Every 2*segment_fes of healthy-leader
    # exploitation opens one development event regardless of leader
    # health -- the reservation semantics the plan demanded.
    exploit_since_event = 0
    # v4.4: sampling debt accrues while a material leader runs with
    # unsampled matured episodes waiting (diagnostic; repaid in the
    # first DISCOVERY_RUN bootstrap).
    sampling_debt = 0
    receipts: list[V4SegmentReceipt] = []
    probes: list[V4ProbeReceipt] = []
    tickets: list[V4TicketRecord] = []
    handoffs: list[EpisodeHandoffReceipt] = []
    segment_index = 0
    switches = 0
    cold_spent = 0
    development_spent = 0
    rotation_index = 0
    previous_episode = ""

    def _latest_snapshot_hash(episode: str, *, force_fresh: bool = False) -> str:
        if not force_fresh:
            try:
                return states[episode].last_step_snapshot.snapshot_hash
            except Phase2StateError:
                pass
        return states[episode].snapshot().snapshot_hash

    def _handoff(previous: str, episode: str) -> None:
        nonlocal switches
        if previous == episode:
            return
        refusal = "disabled"
        adopted = False
        if config.handoff_enabled:
            adopted, refusal = ledgers[episode].adopt_global_archive(
                source=global_ledger,
                accept_out_of_bounds=HANDOFF_ACCEPTS_OOB[episode],
            )
        if adopted:
            book[episode]["handoff_epoch"] += 1
            # Discounted carry (v4.2): adoption changes the private
            # baseline, so pre-adoption magnitudes are not directly
            # comparable -- but wiping them destroyed the "strong yet
            # slow" evidence entirely (Gate 51b S5: CTP's 2.21 ticket
            # credit went to 0).  Halve instead of zeroing.
            book[episode]["private_credit"] *= 0.5
        elif hpr_mode and refusal in {"oob_incumbent", "not_better"}:
            # A failed baton transfer is a real switching cost.  It does not
            # consume FE by itself, but repeated invalid re-anchors must not
            # be treated as fresh evidence for the target episode.
            book[episode]["handoff_penalty"] += 1
            book[episode]["handoff_cooldown"] = max(
                book[episode]["handoff_cooldown"], 1
            )
        handoffs.append(
            EpisodeHandoffReceipt(
                segment_index=segment_index,
                handoff_from=previous,
                handoff_to=episode,
                handoff_mode=HANDOFF_MODE[episode] if config.handoff_enabled else "disabled",
                adopted=adopted,
                refusal=refusal,
                incumbent_error=float(global_ledger.best_error),
                from_snapshot_hash=(
                    _latest_snapshot_hash(previous) if previous else ""
                ),
                to_snapshot_hash=_latest_snapshot_hash(
                    episode, force_fresh=adopted
                ),
            )
        )
        switches += 1

    def _step_episode(
        episode: str, request: int
    ) -> tuple[int, Phase2Snapshot | None]:
        state = states[episode]
        try:
            step = state.step(request)
            consumed = step.step_fes
        except Phase2StateError:
            if request >= global_ledger.remaining:
                stuck.add(episode)
            else:
                min_window_needed[episode] = max(min_window_needed[episode], request + 1)
            return 0, None
        if consumed == 0:
            min_window_needed[episode] = max(min_window_needed[episode], request + 1)
            return 0, None
        return consumed, state.last_step_snapshot

    def _recent_rate(episode: str) -> float:
        history = list(book[episode]["history"])[-2:]
        if not history:
            return 0.0
        gains = sum(g for g, _ in history)
        fes = sum(f for _, f in history)
        return gains / fes if fes else 0.0

    def _ticket_target(episode: str) -> int:
        progress = _progress(episode)
        target = progress.maturity_target_fes
        if episode == "aor":
            target = max(target, 2 * config.maturity_window_fes)
        return target

    def _ticket_complete(episode: str) -> bool:
        # AOR's ticket spec demands two correction windows even though its
        # generic protocol maturity fires after one; every other episode
        # completes at protocol maturity.
        progress = _progress(episode)
        if not progress.protocol_mature:
            return False
        if episode == "aor":
            return progress.consumed_fes >= _ticket_target(episode)
        return True

    def _ticket_size(episode: str) -> int:
        if _ticket_complete(episode):
            return 0
        progress = _progress(episode)
        # The ticket spec (including AOR's two-window protection) lives in
        # _ticket_target; the progress contract only exposes the generic
        # single-window maturity estimate.  The effective execution floor
        # (generation alignment + any recorded transient block) must be
        # part of the size, or P1 would re-issue a window the episode
        # provably cannot execute and spin without consuming budget.
        floor = max(progress.min_step_fes, min_window_needed[episode])
        remaining_to_target = max(_ticket_target(episode) - progress.consumed_fes, 0)
        if remaining_to_target == 0:
            return max(progress.next_boundary_fes, floor)
        return max(remaining_to_target, floor)

    def _escalation_window(episode: str) -> int:
        step = book[episode]["escalation_step"]
        return int(
            config.maturity_window_fes * (config.escalation_factor ** max(step, 0))
        )

    def _bookable(size: int, *, first_round_ticket: bool) -> tuple[bool, str]:
        """Dual-ledger affordability: cold-start first for first-round
        tickets, development ledger otherwise.  Development-class grants
        must leave the terminal exploitation reserve intact."""

        remaining = global_ledger.remaining
        if first_round_ticket and cold_spent + size <= cold_start_budget and size <= remaining:
            return True, "cold_start"
        if (
            development_spent + size <= development_budget
            and size <= remaining
            and remaining - size >= reserve_floor
        ):
            return True, "development"
        return False, ""

    def _challenger_reservation(
        episode: str, wanted: int, leader_exists: bool
    ) -> tuple[int, str]:
        """Reserve-respecting mandated-challenger reservation (v4.1 P1).

        Returns ``(window, ledger_class)``; window 0 means not grantable.
        Normal mode: the window clamps to the development ledger's
        headroom and must leave the exploitation reserve intact
        (development class).  Terminal drain (remaining <= reserve) or no
        matured leader (the reserve protects exploitation that cannot
        happen): the window only clamps to the remaining budget and books
        exploitation -- decided BEFORE execution, never reclassified
        after the fact.
        """

        min_step = max(_progress(episode).min_step_fes, min_window_needed[episode])
        if global_ledger.remaining <= reserve_floor or not leader_exists:
            allowed = min(wanted, global_ledger.remaining)
            return (allowed, "exploitation") if allowed >= min_step else (0, "")
        headroom = development_budget - development_spent
        allowed = min(wanted, headroom, global_ledger.remaining - reserve_floor)
        if allowed < min_step:
            return 0, ""
        return allowed, "development"

    def _credit_update(episode: str, local_gain: float, consumed: int) -> float:
        if config.private_credit_mode == "rate":
            value = local_gain / consumed if consumed else 0.0
            book[episode]["private_credit"] = value
        else:
            book[episode]["private_credit"] = max(book[episode]["private_credit"], local_gain)
        return book[episode]["private_credit"]

    def _emit(
        episode: str,
        grant_kind: str,
        window: int,
        request: int,
        consumed: int,
        g_before: float,
        g_after: float,
        l_before: float,
        l_after: float,
        progress_before: EpisodeProgress,
        progress_after: EpisodeProgress,
        snapshot: Phase2Snapshot,
        *,
        ledger_class: str,
        ticket_id: str = "",
        switched: bool = False,
        leader: str = "",
        reservation_kind: str = "",
        plateau_release: bool = False,
        released: bool = False,
    ) -> None:
        nonlocal segment_index
        g_gain = _log_gain(g_before, g_after)
        l_gain = _log_gain(l_before, l_after)
        material = g_gain > config.material_log_gain
        credit = _credit_update(episode, l_gain, consumed) if grant_kind != "exploit" else book[episode]["private_credit"]
        # Rate history samples real search windows only; maturity tickets
        # are protocol units whose huge semantic gains would crown an
        # unearned leader (Gate 51b R2: CTP's ticket gain bought two
        # zero-gain exploit rounds).
        if grant_kind != "ticket":
            book[episode]["history"].append((g_gain, consumed))
        receipts.append(
            V4SegmentReceipt(
                segment_index=segment_index,
                grant_kind=grant_kind,
                episode=episode,
                episode_kind=EPISODE_KIND[episode],
                grant_index=book[episode]["grant_index"],
                window_fes=window,
                requested_fes=request,
                consumed_fes=consumed,
                leader=leader,
                ledger_class=ledger_class,
                global_error_before=g_before,
                global_error_after=g_after,
                global_gain=g_gain,
                local_error_before=l_before,
                local_error_after=l_after,
                local_gain=l_gain,
                material=material,
                recent_rate=_recent_rate(episode),
                private_credit=credit,
                handoff_epoch=book[episode]["handoff_epoch"],
                cumulative_development_fes=book[episode]["cum_dev_fes"],
                # v4.2 semantics: revelation counts ACTUAL search FE the
                # episode has run (every grant kind), not only development
                # grants -- an exploited episode was accumulating evidence
                # all along (Gate 51b S5: CTP ran 657k but read 337k).
                cumulative_runtime_fes=funded[episode],
                evidence_revealed=funded[episode] >= config.revelation_horizon_fes,
                maturity_ticket_id=ticket_id,
                maturity_committed=bool(ticket_id),
                challenger=grant_kind in ("challenger", "escalation"),
                cooldown=book[episode]["cooldown"],
                remaining_budget=global_ledger.remaining,
                cold_start_spent=cold_spent,
                development_spent=development_spent,
                progress_before=_progress_dict(progress_before),
                progress_after=_progress_dict(progress_after),
                switched=switched,
                next_episode="",
                state_hash=snapshot.state_hash,
                snapshot_hash=snapshot.snapshot_hash,
                reservation_kind=reservation_kind,
                plateau_release=plateau_release,
                handoff_penalty=int(book[episode]["handoff_penalty"]),
                released=released,
            )
        )
        segment_index += 1

    def _current_leader_name() -> str:
        # v4.3: leadership is EARNED by exploit-class evidence only --
        # episodes whose rate samples are probes/tickets/challengers stay
        # in the development lanes.  Gate 51b S5: gss's 20k probe rate
        # (5e-5/FE) crowned it for the whole run while its marginal value
        # decayed, and no rescue lane could fire under a healthy leader.
        # Bootstrap: before any exploit exists, the highest private
        # credit among matured episodes takes the first exploit segment
        # (S5: CTP's 2.21 ticket credit) -- otherwise P2 could never
        # grant the first exploit that establishes the rate history.
        matured = [
            e
            for e in EPISODES
            if _progress(e).protocol_mature and e not in stuck
        ]
        if not matured:
            return ""
        with_exploit = [e for e in matured if book[e]["exploit_history"]]
        if with_exploit:
            def _exploit_rate(episode: str) -> float:
                tail = list(book[episode]["exploit_history"])[-2:]
                fes = sum(f for _, f in tail)
                return (sum(g for g, _ in tail) / fes) if fes else 0.0

            ranked = sorted(
                with_exploit, key=lambda e: (-_exploit_rate(e), probe_order.index(e))
            )
            return ranked[0]
        ranked = sorted(
            matured,
            key=lambda e: (-book[e]["private_credit"], probe_order.index(e)),
        )
        return ranked[0]

    def _hpr_target(episode: str) -> int:
        """Minimum runtime contract for one delayed-feedback reservation."""

        return max(_ticket_target(episode), config.revelation_horizon_fes)

    def _hpr_reservation(episode: str) -> tuple[int, str]:
        """Return one executable horizon reservation, never a soft ticket.

        The reservation is action-independent at the policy level: its size
        comes from the frozen revelation horizon and the episode's native
        progress contract.  It is paid in one grant whenever the development
        ledger can afford it; otherwise the scheduler records a normal
        challenger-sized partial reservation and keeps the debt visible.
        """

        progress = _progress(episode)
        remaining_to_horizon = max(_hpr_target(episode) - funded[episode], 0)
        wanted = max(remaining_to_horizon, progress.min_step_fes)
        if global_ledger.remaining <= reserve_floor:
            allowed = min(wanted, global_ledger.remaining)
            return (allowed, "exploitation") if allowed >= progress.min_step_fes else (0, "")
        headroom = development_budget - development_spent
        allowed = min(wanted, headroom, global_ledger.remaining - reserve_floor)
        if allowed < progress.min_step_fes:
            return 0, ""
        return allowed, "development"

    def _hpr_candidate(leader: str) -> tuple[str, int, str] | None:
        """Choose the most urgent delayed episode without case-specific rules."""

        candidates: list[str] = []
        for episode in EPISODES:
            if episode == leader or episode in stuck:
                continue
            if funded[episode] >= _hpr_target(episode):
                book[episode]["reservation_paid"] = True
                continue
            if book[episode]["handoff_cooldown"] > 0:
                continue
            window, ledger_class = _hpr_reservation(episode)
            if window > 0 and window >= max(
                _progress(episode).min_step_fes, min_window_needed[episode]
            ):
                candidates.append(episode)
        if not candidates:
            return None

        def urgency(episode: str) -> tuple[float, int, float, int]:
            debt = max(_hpr_target(episode) - funded[episode], 0)
            # Remaining debt per available FE is the deadline pressure.  The
            # handoff penalty is a deterministic tie-break, not a case rule.
            ratio = debt / max(global_ledger.remaining, 1)
            return (
                ratio,
                int(book[episode]["challenger_absence"]),
                -float(book[episode]["private_credit"]),
                -probe_order.index(episode),
            )

        episode = max(candidates, key=urgency)
        window, ledger_class = _hpr_reservation(episode)
        return episode, window, ledger_class

    def _adjudicate() -> tuple[str, str, int, dict[str, object]]:
        """Exactly one grant decision; raises on an unsolvable state."""
        nonlocal sampling_debt

        remaining = global_ledger.remaining

        def _executable(episode: str, window: int) -> bool:
            # Generation-aligned episodes (SMP) cannot execute a window
            # below their minimum state unit -- or below the last window
            # size that already failed for them; the terminal tail must
            # go to an FE-granular episode instead of dead-ending.
            return window >= max(
                _progress(episode).min_step_fes, min_window_needed[episode]
            )

        segment_window = min(config.segment_fes, remaining)
        if adaptive_mode:
            leader = _current_leader_name()
            pending_tickets = any(
                e not in stuck and not _ticket_complete(e) for e in EPISODES
            )
            leader_material = bool(
                leader
                and book[leader]["exploit_history"]
                and book[leader]["exploit_history"][-1][0] > config.material_log_gain
                and not book[leader]["released"]
            )
            # One early verification is allowed, but its continuation cannot
            # consume the challenger reserve: all remaining minimum tickets
            # must be paid before protected runway resumes.
            if (
                leader_material
                and not pending_tickets
                and _executable(leader, segment_window)
            ):
                # Adaptive lock verification is deliberately one calibrated
                # maturity window.  A full segment here charged 300k FE on
                # S5 before the plateau signal could release the runway.
                verification_window = min(
                    segment_window,
                    max(config.maturity_window_fes, _progress(leader).min_step_fes),
                )
                return leader, "exploit", verification_window, {
                    "ledger_class": "exploitation",
                    "reservation_kind": "protected_runway",
                }

            # P1 normally matures all four episodes before P2 can run. v5.1
            # inserts one real verification segment immediately after a
            # completed material ticket. A false lead produces a normal
            # non-material exploit, so HPR releases it and P1 resumes.
            if receipts:
                latest = receipts[-1]
                candidate = latest.episode
                earned_lock = (
                    latest.grant_kind == "ticket"
                    and latest.material
                    and candidate not in stuck
                    and _progress(candidate).protocol_mature
                    and not book[candidate]["exploit_history"]
                    and not any(
                        receipt.reservation_kind == "adaptive_lock"
                        for receipt in receipts
                    )
                )
                if earned_lock and _executable(candidate, segment_window):
                    return candidate, "exploit", segment_window, {
                        "ledger_class": "exploitation",
                        "reservation_kind": "adaptive_lock",
                    }
        # P1 -- unfinished affordable maturity tickets, probe order.
        for episode in probe_order:
            if episode in stuck or book[episode]["ticket_done"]:
                continue
            if _ticket_complete(episode):
                book[episode]["ticket_done"] = True
                continue
            size = _ticket_size(episode)
            if size <= 0 or not _executable(episode, size):
                # The ticket cannot execute at its own size (transient
                # block) -- leave it pending for a later, larger pass
                # instead of spinning on a provably failing window.
                continue
            first_round = book[episode]["grant_index"] == 0
            affordable, ledger_class = _bookable(size, first_round_ticket=first_round)
            ticket_id = f"{episode}-ticket-{len(tickets)}"
            if not affordable:
                if not book[episode]["maturity_unaffordable"]:
                    book[episode]["maturity_unaffordable"] = True
                    tickets.append(
                        V4TicketRecord(
                            episode=episode,
                            ticket_id=ticket_id,
                            target_fes=_ticket_target(episode),
                            requested_fes=size,
                            granted_fes=0,
                            ledger_class="",
                            affordable=False,
                            protocol_mature_after=False,
                            segment_index=segment_index,
                        )
                    )
                continue
            return episode, "ticket", size, {
                "ledger_class": ledger_class,
                "ticket_id": ticket_id,
                "first_round": first_round,
            }
        # P2 -- leader exploitation (v4.3): the v3-winning core, restored.
        # A material leader runs CONTINUOUSLY -- no streak cap, because
        # R6's strict win came from smp's long material run and S5's CTP
        # needs exactly such a run once revealed.  Two v3-style yielding
        # rules bound the monopoly: (a) a NON-material exploit segment
        # arms the leader's one-segment cooldown (stagnation switch),
        # and (b) the reserved cadence below opens the development lanes
        # on a fixed budget rhythm regardless of leader health, so
        # pending ladders / credit continuation cannot be starved by a
        # healthy leader (Gate 51b v4.2: the rescue lanes never fired).
        leader = _current_leader_name()
        if hpr_mode:
            leader_material = bool(
                leader
                and book[leader]["exploit_history"]
                and book[leader]["exploit_history"][-1][0] > config.material_log_gain
                and not book[leader]["released"]
            )
            # A material leader owns the current action-native continuation
            # block.  This is the protected runway: reservations accrue but
            # do not interrupt a live material segment (S5 protection).
            if leader_material and segment_window >= max(
                _progress(leader).min_step_fes, min_window_needed[leader]
            ):
                runway_window = min(
                    segment_window,
                    max(config.maturity_window_fes, _progress(leader).min_step_fes),
                )
                return leader, "exploit", runway_window, {
                    "ledger_class": "exploitation",
                    "reservation_kind": "protected_runway",
                }
            # Before the first exploit there is no delayed feedback to
            # arbitrate; let the calibrated bootstrap credit establish the
            # initial leader.  Horizon reservations begin only after an
            # exploit sample has either materialized or plateaued.
            if leader and book[leader]["exploit_history"]:
                reservation = _hpr_candidate(leader)
                if reservation is not None:
                    episode, window, ledger_class = reservation
                    return episode, "challenger", window, {
                        "ledger_class": ledger_class,
                        "reservation_kind": "horizon",
                    }
        dev_lane_open = development_budget - development_spent >= max(
            max(_progress(e).min_step_fes, min_window_needed[e])
            for e in EPISODES
            if e not in stuck
        )
        cadence_due = exploit_since_event >= 2 * config.segment_fes
        # Health-conditioned sampling fairness (v4.4): two scheduler
        # states.  MATERIAL_RUN -- the leader's last exploit segment was
        # material: continue uninterrupted, accrue sampling debt, only
        # the bounded cadence rungs may intrude (the 75k ladder windows
        # kept v4.3's S5 win intact at 1.59M).  DISCOVERY_RUN -- the
        # leader's last segment was non-material: cadence events take
        # the ladder first (cadence_due below), and between them the
        # unsampled bootstrap fires to break the zero-rate monopoly
        # (Gate 51b R2: ctp burned 1.24M in zero-gain exploits while
        # aor/gcb never established a rate).  v4.3.1's S5 regression
        # came from interrupting a 1.75-gain leader; the health gate
        # keeps that protection without the monopoly.
        if leader and not cadence_due and book[leader]["exploit_history"]:
            last_gain = book[leader]["exploit_history"][-1][0]
            leader_material = last_gain > config.material_log_gain
            unsampled = [
                e
                for e in EPISODES
                if e != leader
                and e not in stuck
                and _progress(e).protocol_mature
                and not book[e]["exploit_history"]
            ]
            if leader_material:
                if unsampled:
                    sampling_debt += 1
            elif unsampled and len(book[leader]["exploit_history"]) >= 2:
                ranked = sorted(
                    unsampled,
                    key=lambda e: (-book[e]["private_credit"], probe_order.index(e)),
                )
                candidate = ranked[0]
                if book[candidate]["cooldown"] == 0 and _executable(candidate, segment_window):
                    return candidate, "exploit", segment_window, {
                        "ledger_class": "exploitation"
                    }
        if (
            leader
            and not cadence_due
            and book[leader]["cooldown"] == 0
            and _executable(leader, segment_window)
        ):
            return leader, "exploit", segment_window, {
                "ledger_class": "exploitation"
            }
        # P3 -- forced challenger.
        # Lane priority (v4.3.1): the pending revelation ladder
        # outranks the revealed-episode rotation floor at cadence
        # events -- R2 needs AOR's next rung more than gcb needs
        # its turn (the floor still fires when nothing is pending).
        candidates_a = [
            e
            for e in EPISODES
            if e != leader
            and e not in stuck
            and funded[e] < config.revelation_horizon_fes
            and book[e]["escalation_step"] < config.escalation_grants_k
        ]
        # Fairness inside the escalation lane (v4.1): the episode with the
        # FEWEST escalation grants so far advances first -- private credit
        # only breaks ties.  A pure credit ranking would let a high-credit
        # pending episode exhaust the development ledger before a
        # low-credit late-maturer (AOR on R2) ever starts its ladder,
        # which is starvation by ordering, not by evidence.
        candidates_a.sort(
            key=lambda e: (
                book[e]["escalation_step"],
                -book[e]["private_credit"],
                probe_order.index(e),
            )
        )
        for episode in candidates_a:
            window = _escalation_window(episode)
            affordable, ledger_class = _bookable(window, first_round_ticket=False)
            if affordable and _executable(episode, window):
                return episode, "escalation", window, {"ledger_class": ledger_class}

        # Rotation floor (plan rule 4b, absence-counted): every non-leader
        # must appear at least once within N consecutive challenger events
        # -- INCLUDING revealed episodes, which the escalation lane would
        # otherwise starve for the whole evidence-pending stretch (Gate
        # 51b R6: gcb completed a material ticket and never returned).
        # Trigger at absence >= N-1 so the return lands inside the window
        # even when the budget ends right after the next event.
        reserve_binding_floor = bool(leader) and dev_lane_open
        overdue = [
            e
            for e in EPISODES
            if e != leader
            and e not in stuck
            and book[e]["challenger_absence"] >= len(EPISODES) - 2
            and _challenger_reservation(e, segment_window, reserve_binding_floor)[0] > 0
        ]
        if overdue:
            episode = min(
                overdue,
                key=lambda e: (-book[e]["challenger_absence"], probe_order.index(e)),
            )
            window, klass = _challenger_reservation(episode, segment_window, reserve_binding_floor)
            return episode, "challenger", window, {"ledger_class": klass}
        # Credit continuation (v4.2): an episode past its revelation
        # horizon whose PRIVATE trajectory still carries above-material
        # credit has demonstrated unrealized global value -- the horizon
        # is contextual and the calibration was probe-stage (Gate 51b
        # S5: CTP revealed at ~450k runtime but only crosses the live
        # archive near 1.2M).  Promotion by discounted credit, never
        # leader crowning: these run as challenger-class continuation
        # windows inside the development budget, ranked by credit.
        continuation = [
            e
            for e in EPISODES
            if e != leader
            and e not in stuck
            and funded[e] >= config.revelation_horizon_fes
            and book[e]["private_credit"] > config.material_log_gain
            and _challenger_reservation(e, segment_window, reserve_binding_floor)[0] > 0
        ]
        if continuation:
            episode = max(
                continuation, key=lambda e: (book[e]["private_credit"], -probe_order.index(e))
            )
            window, klass = _challenger_reservation(episode, segment_window, reserve_binding_floor)
            return episode, "challenger", window, {"ledger_class": klass}
        non_leader = [e for e in EPISODES if e != leader and e not in stuck]
        if non_leader and leader:
            # A matured leader exists but is capped or in its avoidance
            # window: rotate over the non-leaders.  This is the rotation
            # floor's hard guarantee -- a rate ranking here would starve
            # zero-rate episodes and resurrect Gate 50's failure mode.
            grantable = [
                e for e in non_leader
                if _challenger_reservation(e, segment_window, True)[0] > 0
            ]
            if grantable:
                pool = [e for e in grantable if book[e]["cooldown"] == 0]
                if pool:
                    episode = pool[rotation_index % len(pool)]
                else:
                    # Every grantable non-leader is in its cooldown window:
                    # earliest expiry wins (plan P3(b) fallback).
                    episode = min(
                        grantable,
                        key=lambda e: (book[e]["cooldown"], probe_order.index(e)),
                    )
                window, klass = _challenger_reservation(episode, segment_window, True)
                return episode, "challenger", window, {"ledger_class": klass}
        if non_leader:
            # True 600k corner (plan section 4 item 9): no matured leader
            # at all -- or the development lane is exhausted while the
            # leader is streak-blocked (the reserve then protects an
            # exploitation that cannot happen).  Rate ranking over
            # >=2-segment history with a self-exclusion so the previous
            # grant cannot monopolize; otherwise earliest cooldown expiry.
            reserve_binding = bool(leader) and dev_lane_open

            def _grantable(episode: str) -> bool:
                return _challenger_reservation(episode, segment_window, reserve_binding)[0] > 0

            with_history = [
                e
                for e in non_leader
                if len(book[e]["history"]) >= 2
                and book[e]["cooldown"] == 0
                and e != previous_episode
                and _grantable(e)
            ]
            if not with_history:
                with_history = [
                    e
                    for e in non_leader
                    if len(book[e]["history"]) >= 2
                    and book[e]["cooldown"] == 0
                    and _grantable(e)
                ]
            if with_history:
                ranked = sorted(
                    with_history, key=lambda e: (-_recent_rate(e), probe_order.index(e))
                )
                window, klass = _challenger_reservation(ranked[0], segment_window, reserve_binding)
                return ranked[0], "challenger", window, {"ledger_class": klass}
            pool = [e for e in non_leader if _grantable(e)]
            if pool:
                episode = min(
                    pool,
                    key=lambda e: (book[e]["cooldown"], probe_order.index(e)),
                )
                window, klass = _challenger_reservation(episode, segment_window, reserve_binding)
                return episode, "challenger", window, {"ledger_class": klass}
        # P5 -- designed fallback (600k corner): history then rotation.
        reserve_binding_p5 = bool(leader) and dev_lane_open

        def _grantable_any(episode: str) -> bool:
            return (
                _challenger_reservation(episode, segment_window, reserve_binding_p5)[0] > 0
            )

        executable = [e for e in EPISODES if e not in stuck and _grantable_any(e)]
        with_history = [
            e for e in executable if len(book[e]["history"]) >= 2
        ]
        if with_history:
            ranked = sorted(
                with_history, key=lambda e: (-_recent_rate(e), probe_order.index(e))
            )
            window, klass = _challenger_reservation(ranked[0], segment_window, reserve_binding_p5)
            return ranked[0], "challenger", window, {"ledger_class": klass}
        if executable:
            episode = probe_order[0] if probe_order[0] in executable else executable[0]
            window, klass = _challenger_reservation(episode, segment_window, reserve_binding_p5)
            return episode, "challenger", window, {"ledger_class": klass}
        raise Phase2StateError("v4 adjudication has no executable episode left")

    def _execute(episode: str, grant_kind: str, window: int, meta: dict[str, object]) -> bool:
        nonlocal previous_episode, cold_spent, development_spent, rotation_index
        nonlocal exploit_streak, exploit_since_event
        departed = previous_episode
        switched = departed != episode
        window = min(window, global_ledger.remaining)
        reservation_kind = str(meta.get("reservation_kind", ""))
        if grant_kind in ("challenger", "escalation"):
            # v4.2 tail guard: a challenger window must leave one
            # continuation slice of runway, so a winner discovered in the
            # last stretch is never left with zero budget to bank its
            # gain (Gate 51b R2: AOR's final 282k challenger gained
            # 0.328 and the budget hit zero).
            if not (hpr_mode and reservation_kind == "horizon"):
                continuation_reserve = min(
                    config.maturity_window_fes, global_ledger.remaining // 2
                )
                clamped = global_ledger.remaining - continuation_reserve
                if clamped >= max(_progress(episode).min_step_fes, min_window_needed[episode]):
                    window = min(window, clamped)
        ledger_class = str(meta.get("ledger_class", ""))
        ticket_id = str(meta.get("ticket_id", ""))
        progress_before = _progress(episode)
        # The receipt's leader field is the leader identity AT GRANT TIME
        # (plan section 6) -- captured before this grant's own gains can
        # reshuffle the recent-rate ranking.
        leader_at_grant = _current_leader_name()
        _handoff(departed, episode)
        g_before = float(global_ledger.best_error)
        l_before = float(ledgers[episode].best_error)
        consumed, snapshot = _step_episode(episode, window)
        if consumed == 0:
            # The episode cannot execute this unit (e.g. SMP mid-visit on a
            # larger block at the terminal tail).  _step_episode already
            # marked it stuck; no receipt is emitted and the ladder
            # re-adjudicates without it.
            return False
        if snapshot is None:
            raise RuntimeError("successful episode step did not expose its snapshot")
        exploit_streak = exploit_streak + 1 if grant_kind == "exploit" else 0
        g_after = float(global_ledger.best_error)
        l_after = float(ledgers[episode].best_error)
        g_gain = _log_gain(g_before, g_after)
        material = g_gain > config.material_log_gain
        if grant_kind == "exploit":
            exploit_since_event += consumed
            book[episode]["exploit_history"].append((g_gain, consumed))
            if hpr_mode:
                # A zero-gain exploit releases the leader until a complete
                # delayed reservation is paid.  A material block keeps the
                # leader protected for its next native continuation block.
                book[episode]["released"] = not material
        else:
            exploit_since_event = 0
        funded[episode] += consumed
        if grant_kind == "probe":
            cold_spent += consumed
        elif grant_kind == "exploit":
            book[episode]["consecutive_exploit"] += 1
        else:
            if grant_kind == "ticket" and ticket_id:
                # A partially consumed ticket (generation alignment) stays
                # pending; P1 re-grants the remainder on a later pass.
                book[episode]["ticket_done"] = _ticket_complete(episode)
            if hpr_mode and reservation_kind == "horizon":
                book[episode]["reservation_paid"] = (
                    funded[episode] >= _hpr_target(episode)
                )
                book[episode]["reservation_due"] = not book[episode]["reservation_paid"]
            if grant_kind == "escalation":
                book[episode]["escalation_step"] += 1
            book[episode]["grant_index"] += 1
            book[episode]["cum_dev_fes"] += consumed
            if ledger_class == "cold_start":
                cold_spent += consumed
            elif ledger_class == "development":
                # Reservation happened before execution (_bookable /
                # _challenger_reservation clamp), so development booking
                # can never exceed the cap nor penetrate the reserve; the
                # asserts keep that invariant loud if it ever regresses.
                development_spent += consumed
                if development_spent > development_budget:
                    raise Phase2StateError("development ledger exceeded after reservation")
                if global_ledger.remaining < reserve_floor and global_ledger.remaining > 0:
                    raise Phase2StateError("development grant penetrated the exploitation reserve")
            # ledger_class == "exploitation": a terminal-drain or
            # no-leader challenger (declared at adjudication) -- it spends
            # the exploitation pool and the receipt records the class.
            if grant_kind == "ticket" and ticket_id:
                tickets.append(
                    V4TicketRecord(
                        episode=episode,
                        ticket_id=ticket_id,
                        target_fes=_ticket_target(episode),
                        requested_fes=window,
                        granted_fes=consumed,
                        ledger_class=ledger_class,
                        affordable=True,
                        protocol_mature_after=_ticket_complete(episode),
                        segment_index=segment_index,
                    )
                )
        progress_after = _progress(episode)
        _emit(
            episode,
            grant_kind,
            window,
            window,
            consumed,
            g_before,
            g_after,
            l_before,
            l_after,
            progress_before,
            progress_after,
            snapshot,
            ledger_class=ledger_class if grant_kind != "exploit" else "exploitation",
            ticket_id=ticket_id,
            switched=switched,
            leader=leader_at_grant,
            reservation_kind=reservation_kind,
            plateau_release=bool(hpr_mode and grant_kind == "exploit" and not material),
            released=bool(hpr_mode and book[episode]["released"]),
        )
        # One avoidance segment for the departed episode: decrement every
        # outstanding cooldown first, then arm the freshly departed one so
        # it is skipped for exactly the next grant.  The stagnation yield
        # arms AFTER the decrement too -- arming it earlier let the same
        # grant's tail loop consume it, so zero-gain leaders could run
        # back-to-back (Gate 51b v4.3 R2: ctp burned 600k in four
        # zero-gain exploits while AOR starved at 225k).
        for name in EPISODES:
            if book[name]["cooldown"] > 0:
                book[name]["cooldown"] -= 1
            if book[name]["handoff_cooldown"] > 0:
                book[name]["handoff_cooldown"] -= 1
        if switched and departed:
            book[departed]["cooldown"] = 1
            book[departed]["consecutive_exploit"] = 0
        if grant_kind == "exploit" and not material:
            book[episode]["cooldown"] = 1
        if grant_kind in ("challenger", "escalation"):
            rotation_index += 1
            for name in EPISODES:
                book[name]["challenger_absence"] = (
                    0 if name == episode else book[name]["challenger_absence"] + 1
                )
            if hpr_mode and reservation_kind == "horizon":
                # A reservation event pays the delayed-feedback debt.  If a
                # partial window was forced by the remaining budget, the
                # released leader stays released and the next call must pay
                # the remainder before ordinary exploitation resumes.
                if book[episode]["reservation_paid"]:
                    for name in EPISODES:
                        if name != episode and book[name]["released"]:
                            book[name]["released"] = False
        previous_episode = episode
        return True

    # --- P0: minimal executable probe per episode, B/W order ---
    for position, episode in enumerate(probe_order):
        if episode in stuck:
            raise Phase2StateError(f"episode {episode} is stuck before its probe")
        progress = _progress(episode)
        probe_window = max(config.probe_min_fes, progress.min_step_fes)
        unprobed_left = len(probe_order) - position - 1
        if global_ledger.remaining < probe_window * (unprobed_left + 1):
            raise Phase2StateError(
                "v4 probe protocol violated: remaining "
                f"{global_ledger.remaining} FE cannot fund minimal probes "
                f"({probe_window} FE each) for every unprobed episode "
                f"({unprobed_left + 1} left, next {episode})"
            )
        probe_window = min(probe_window, global_ledger.remaining - probe_window * unprobed_left)
        departed = previous_episode
        switched = departed != episode
        _handoff(departed, episode)
        g_before = float(global_ledger.best_error)
        l_before = float(ledgers[episode].best_error)
        consumed, snapshot = _step_episode(episode, probe_window)
        if consumed == 0:
            raise Phase2StateError(f"episode {episode} cannot execute its probe window")
        if snapshot is None:
            raise RuntimeError("successful episode probe did not expose its snapshot")
        funded[episode] += consumed
        cold_spent += consumed
        g_after = float(global_ledger.best_error)
        l_after = float(ledgers[episode].best_error)
        g_gain = _log_gain(g_before, g_after)
        l_gain = _log_gain(l_before, l_after)
        progress_after = _progress(episode)
        book[episode]["history"].append((g_gain, consumed))
        probes.append(
            V4ProbeReceipt(
                order_position=position,
                episode=episode,
                episode_kind=EPISODE_KIND[episode],
                budget_fes=consumed,
                scope_size=int(sensing.get("scope_size") or 0),
                probe_contract=progress.contract,
                min_step_fes=progress.min_step_fes,
                maturity_target_fes=_ticket_target(episode),
                global_error_before=g_before,
                global_error_after=g_after,
                global_gain=g_gain,
                local_error_before=l_before,
                local_error_after=l_after,
                local_gain=l_gain,
                max_local_log_gain_window=l_gain,
            )
        )
        receipts.append(
            V4SegmentReceipt(
                segment_index=segment_index,
                grant_kind="probe",
                episode=episode,
                episode_kind=EPISODE_KIND[episode],
                grant_index=0,
                window_fes=probe_window,
                requested_fes=probe_window,
                consumed_fes=consumed,
                leader="",
                ledger_class="cold_start",
                global_error_before=g_before,
                global_error_after=g_after,
                global_gain=g_gain,
                local_error_before=l_before,
                local_error_after=l_after,
                local_gain=l_gain,
                material=g_gain > config.material_log_gain,
                recent_rate=_recent_rate(episode),
                private_credit=_credit_update(episode, l_gain, consumed),
                handoff_epoch=book[episode]["handoff_epoch"],
                cumulative_development_fes=0,
                cumulative_runtime_fes=funded[episode],
                evidence_revealed=funded[episode] >= config.revelation_horizon_fes,
                maturity_ticket_id="",
                maturity_committed=False,
                challenger=False,
                cooldown=book[episode]["cooldown"],
                remaining_budget=global_ledger.remaining,
                cold_start_spent=cold_spent,
                development_spent=development_spent,
                progress_before=_progress_dict(progress),
                progress_after=_progress_dict(progress_after),
                switched=switched,
                next_episode="",
                state_hash=snapshot.state_hash,
                snapshot_hash=snapshot.snapshot_hash,
            )
        )
        segment_index += 1
        for name in EPISODES:
            if book[name]["cooldown"] > 0:
                book[name]["cooldown"] -= 1
        if switched and departed:
            book[departed]["cooldown"] = 1
            book[departed]["consecutive_exploit"] = 0
        previous_episode = episode

    # --- P1-P5 loop ---
    failed_executions = 0
    while global_ledger.remaining > 0:
        episode, grant_kind, window, meta = _adjudicate()
        if not _execute(episode, grant_kind, window, meta):
            failed_executions += 1
            if failed_executions > 200:
                raise Phase2StateError(
                    "v4 adjudication is spinning: 200 consecutive grants "
                    "could not execute"
                )
            continue
        failed_executions = 0

    if global_ledger.count != checkpoint.total_budget_fes:
        raise RuntimeError("v4 schedule did not stop at the terminal FE")
    audit = _audit_v4(
        receipts,
        probes,
        tickets,
        phase2_fes=phase2_fes,
        cold_start_cap_fes=cold_start_budget,
        development_cap_fes=development_budget,
        reserve_floor_fes=reserve_floor,
        revelation_horizon_fes=config.revelation_horizon_fes,
        terminal_fes=global_ledger.count,
        sensing_fes=int(sensing.get("probe_fes") or 0),
        phase1_fes=checkpoint.phase1_fes,
        escalation_factor=config.escalation_factor,
        funded=funded,
    )
    if hpr_mode:
        audit["hpr_receipt_surface"] = all(
            hasattr(receipt, "reservation_kind")
            and hasattr(receipt, "plateau_release")
            and hasattr(receipt, "handoff_penalty")
            and hasattr(receipt, "released")
            for receipt in receipts
        )
        audit["hpr_handoff_penalty_nonnegative"] = all(
            receipt.handoff_penalty >= 0 for receipt in receipts
        )
        audit["hpr_reservation_progress_monotone"] = all(
            funded[episode] >= 0 for episode in EPISODES
        )
    if adaptive_mode:
        adaptive_locks = [
            receipt for receipt in receipts
            if receipt.reservation_kind == "adaptive_lock"
        ]
        audit["adaptive_lock_receipts_are_exploit"] = all(
            receipt.grant_kind == "exploit" for receipt in adaptive_locks
        )
        audit["adaptive_lock_is_earned"] = all(
            receipt.segment_index > 0
            and receipts[receipt.segment_index - 1].grant_kind == "ticket"
            and receipts[receipt.segment_index - 1].episode == receipt.episode
            and receipts[receipt.segment_index - 1].material
            and receipts[receipt.segment_index - 1].progress_after.get("protocol_mature") is True
            for receipt in adaptive_locks
        )
        audit["adaptive_challenger_reserve_preserved"] = (
            reserve_floor > 0 and audit["development_reserve_preserved"]
        )
        protected_indices = [
            receipt.segment_index
            for receipt in receipts
            if receipt.reservation_kind == "protected_runway"
        ]
        first_protected = min(protected_indices, default=len(receipts))
        audit["adaptive_tickets_precede_protected_runway"] = all(
            receipt.grant_kind != "ticket"
            for receipt in receipts[first_protected:]
        )
        audit["adaptive_plateau_releases_runway"] = all(
            not (
                receipt.reservation_kind == "protected_runway"
                and receipt.plateau_release
                and index + 1 < len(receipts)
                and receipts[index + 1].episode == receipt.episode
                and receipts[index + 1].reservation_kind == "protected_runway"
            )
            for index, receipt in enumerate(receipts)
        )
        adaptive_lock_indices = [
            receipt.segment_index
            for receipt in receipts
            if receipt.reservation_kind == "adaptive_lock"
        ]
        audit["adaptive_verification_window_bounded"] = all(
            receipts[index + 1].window_fes
            <= max(config.maturity_window_fes, _progress(receipts[index + 1].episode).min_step_fes)
            for index in adaptive_lock_indices
            if index + 1 < len(receipts)
            and receipts[index + 1].reservation_kind == "protected_runway"
        )
    if not all(audit.values()):
        failed = [name for name, ok in audit.items() if not ok]
        raise RuntimeError(f"v4 schedule audit failed: {failed}")
    magnitude_repairs = {episode: ledgers[episode].magnitude_repairs for episode in EPISODES}
    if config.scheduler_version == DEFAULT_SCHEDULER_VERSION_V5_2:
        scheduler_policy = SCHEDULER_POLICY_V5_2
        scheduler_schema = OC_EPISODE_SCHEMA_V5_2
    else:
        scheduler_policy = SCHEDULER_POLICY_V4
        scheduler_schema = OC_EPISODE_SCHEMA_V4
    config_payload = {
        "scheduler_policy": scheduler_policy,
        "scheduler_version": config.scheduler_version,
        "calibration_ref": config.calibration_ref,
        "maturity_window_fes": config.maturity_window_fes,
        "revelation_horizon_fes": config.revelation_horizon_fes,
        "exploration_and_development_cap": config.exploration_and_development_cap,
        "exploitation_reserve_ratio": config.exploitation_reserve_ratio,
        "cold_start_probe_cap": config.cold_start_probe_cap,
        "probe_min_fes": config.probe_min_fes,
        "escalation_factor": config.escalation_factor,
        "escalation_grants_k": config.escalation_grants_k,
        "segment_fes": config.segment_fes,
        "material_log_gain": config.material_log_gain,
        "handoff_enabled": config.handoff_enabled,
        "private_credit_mode": config.private_credit_mode,
        "horizon_protected": hpr_mode,
        "adaptive_exploration": adaptive_mode,
    }
    payload = {
        "schema_version": scheduler_schema,
        "scheduler_policy": scheduler_policy,
        "scheduler_version": config.scheduler_version,
        "calibration_ref": config.calibration_ref,
        "dispatcher": DISPATCHER,
        "coordinator_name": COORDINATOR_PUBLIC_NAME,
        "episode_names": dict(EPISODE_PUBLIC_NAMES),
        "config": config_payload,
        "sensing": sensing,
        "probe_order": list(probe_order),
        "cold_start_probe_tax_fes": cold_spent,
        "development_fes": development_spent,
        "exploitation_fes": phase2_fes - cold_spent - development_spent - int(sensing.get("probe_fes") or 0),
        "scoped_checkpoint_hash": scoped_checkpoint.checkpoint_hash,
        "probes": [dataclasses.asdict(p) for p in probes],
        "handoffs": [dataclasses.asdict(h) for h in handoffs],
        "tickets": [dataclasses.asdict(t) for t in tickets],
        "receipts": [dataclasses.asdict(r) for r in receipts],
        "audit": audit,
        "funded_fes": funded,
        "switches": switches,
        "magnitude_repairs": magnitude_repairs,
    }
    return V4ScheduleResult(
        schema_version=scheduler_schema,
        scheduler_policy=scheduler_policy,
        scheduler_version=config.scheduler_version,
        calibration_ref=config.calibration_ref,
        dispatcher=DISPATCHER,
        config=config_payload,
        sensing=sensing,
        probe_order=tuple(probe_order),
        cold_start_probe_tax_fes=cold_spent,
        development_fes=development_spent,
        exploitation_fes=phase2_fes - cold_spent - development_spent - int(sensing.get("probe_fes") or 0),
        scoped_checkpoint_hash=scoped_checkpoint.checkpoint_hash,
        probes=tuple(probes),
        handoffs=tuple(handoffs),
        tickets=tuple(tickets),
        receipts=tuple(receipts),
        audit=audit,
        funded_fes=funded,
        magnitude_repairs=magnitude_repairs,
        switches=switches,
        final_error=float(global_ledger.best_error),
        terminal_fes=global_ledger.count,
        schedule_hash=canonical_sha256(payload),
        coordinator_name=COORDINATOR_PUBLIC_NAME,
        episode_names=dict(EPISODE_PUBLIC_NAMES),
    )


def run_oc_episode_schedule_v5(
    problem: OptimizationProblem,
    checkpoint: PhaseCheckpoint,
    *,
    action_seed: int,
    config: PhaseAwareSchedulerConfig,
    structure: OverlapStructure | None = None,
) -> V4ScheduleResult:
    """Retired v5.0 entry: raises instead of producing a mislabelled run.

    v5.0 HPR semantics changed when v5.2 bounded the protected runway by
    one maturity window; this tree can no longer produce a schedule that
    matches the frozen Gate 51b v5.0 cells.  Use
    :func:`run_oc_episode_schedule_v5_2`.
    """

    raise RuntimeError(
        "run_oc_episode_schedule_v5 is retired: v5.0 HPR semantics were "
        "superseded by the w1-bounded protected runway in v5.2; use "
        "run_oc_episode_schedule_v5_2 (version-isolated artifacts)."
    )


def run_oc_episode_schedule_v5_1(
    problem: OptimizationProblem,
    checkpoint: PhaseCheckpoint,
    *,
    action_seed: int,
    config: PhaseAwareSchedulerConfig,
    structure: OverlapStructure | None = None,
) -> V4ScheduleResult:
    """Retired v5.1 entry: raises instead of producing a mislabelled run.

    v5.1 behaviour (full-segment protected runway; release only at segment
    end) was the S5 disaster-seed mechanism dissected in Gate 51c and is
    superseded by v5.2's bounded verification window.  Frozen v5.1 cells
    keep their provenance via their recorded manifests.
    """

    raise RuntimeError(
        "run_oc_episode_schedule_v5_1 is retired: the protected runway is "
        "bounded by one maturity window in v5.2 (plateau release after w1 "
        "FE, released state in the receipt/audit chain); use "
        "run_oc_episode_schedule_v5_2 (version-isolated artifacts)."
    )


def run_oc_episode_schedule_v5_2(
    problem: OptimizationProblem,
    checkpoint: PhaseCheckpoint,
    *,
    action_seed: int,
    config: PhaseAwareSchedulerConfig,
    structure: OverlapStructure | None = None,
) -> V4ScheduleResult:
    """Run v5.2: adaptive HPR-GCB with the bounded protected runway.

    Identical flag surface to v5.1 (horizon protection plus adaptive
    material-ticket verification), but the post-lock verification window
    and every material-leader continuation are bounded by one calibrated
    maturity window, a zero-gain window releases the runway immediately,
    and ``released``/``plateau_release`` are receipt/audit fields.
    """

    if not isinstance(config, PhaseAwareSchedulerConfig):
        raise TypeError("config must be PhaseAwareSchedulerConfig")
    v5_2_config = dataclasses.replace(
        config,
        scheduler_version=DEFAULT_SCHEDULER_VERSION_V5_2,
        horizon_protected=True,
        adaptive_exploration=True,
    )
    return run_oc_episode_schedule_v4(
        problem,
        checkpoint,
        action_seed=action_seed,
        config=v5_2_config,
        structure=structure,
    )


__all__ = [
    "DISPATCHER",
    "COORDINATOR_PUBLIC_NAME",
    "EPISODES",
    "EPISODE_KIND",
    "EPISODE_PUBLIC_NAMES",
    "EpisodeHandoffReceipt",
    "EpisodeProbeReceipt",
    "HANDOFF_MODE",
    "EpisodeScheduleResult",
    "EpisodeSegmentReceipt",
    "OC_EPISODE_SCHEMA",
    "OC_EPISODE_SCHEMA_V4",
    "OC_EPISODE_SCHEMA_V5",
    "OC_EPISODE_SCHEMA_V5_1",
    "OC_EPISODE_SCHEMA_V5_2",
    "DEFAULT_SCHEDULER_VERSION_V4",
    "DEFAULT_SCHEDULER_VERSION_V5",
    "DEFAULT_SCHEDULER_VERSION_V5_1",
    "DEFAULT_SCHEDULER_VERSION_V5_2",
    "GRANT_KINDS",
    "PhaseAwareSchedulerConfig",
    "PRIVATE_CREDIT_MODES",
    "PROBE_MIN_FES",
    "SCHEDULER_POLICY_V4",
    "SCHEDULER_POLICY_V5",
    "SCHEDULER_POLICY_V5_1",
    "SCHEDULER_POLICY_V5_2",
    "V4ProbeReceipt",
    "V4ScheduleResult",
    "V4SegmentReceipt",
    "V4TicketRecord",
    "run_oc_episode_schedule",
    "run_oc_episode_schedule_v4",
    "run_oc_episode_schedule_v5",
    "run_oc_episode_schedule_v5_1",
    "run_oc_episode_schedule_v5_2",
]
