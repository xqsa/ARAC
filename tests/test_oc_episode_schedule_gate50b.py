"""Gate 50b targeted tests for the episode scheduler's new semantics.

Pinned behaviours (pre-registered):

1. global-archive materiality: a privately improving episode that does
   not move the global archive is non-material and loses scheduling
   priority (Gate 50's false stickiness);
2. forced probes: all four episodes receive one executable probe window
   before any exploitation segment;
3. loud starvation: when the remaining budget cannot fund a minimum probe
   for every unprobed episode, the protocol raises instead of letting an
   episode decay to a 0/1-FE formality (Gate 50's R2/R6 AOR failure).
"""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.episodes import run_oc_episode_schedule
from arac.runtime.contracts import PhaseCheckpoint
from arac.runtime.phase2 import Phase2StateError

DIMENSION = 24
BLOCKS = tuple(tuple(range(start, start + 6)) for start in range(0, DIMENSION, 6))
ACTION_SEED = 20260853


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        for block in BLOCKS:
            inner = batch[:, list(block)]
            result += 0.25 * np.sum(inner**2, axis=1) ** 2 / len(block)
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _checkpoint(total: int = 6_000, phase1: int = 300) -> PhaseCheckpoint:
    problem = _problem()
    incumbent = tuple(0.5 for _ in range(DIMENSION))
    return PhaseCheckpoint(
        protocol="gate50b-unit",
        run_seed=3,
        total_budget_fes=total,
        phase1_fes=phase1,
        incumbent=incumbent,
        incumbent_error=float(problem.objective(np.asarray(incumbent))),
        feature_names=("log10_center_error", "line_high_frequency_fraction_median"),
        feature_values=(1.0, 0.4),
        blocks=BLOCKS,
        relations=(),
    )


def _run(**overrides):
    kwargs = dict(
        action_seed=ACTION_SEED,
        segment_fes=400,
        probe_min_fes=200,
        probe_share=0.05,
    )
    kwargs.update(overrides)
    return run_oc_episode_schedule(_problem(), _checkpoint(), **kwargs)


def test_all_four_episodes_receive_executable_probes_before_exploitation() -> None:
    result = _run()
    probe_phases = [r for r in result.receipts if r.phase == "probe"]
    exploit_phases = [r for r in result.receipts if r.phase == "exploit"]
    assert len(probe_phases) == 4
    assert {r.episode for r in probe_phases} == {"ctp", "gcb", "smp", "aor"}
    assert all(r.budget_fes >= 200 for r in result.probes)
    if exploit_phases:
        first_exploit_index = exploit_phases[0].segment_index
        assert all(r.segment_index < first_exploit_index for r in probe_phases)
    assert result.terminal_fes == 6_000


def test_materiality_is_owned_by_global_archive_not_private_gain() -> None:
    result = _run()
    for receipt in result.receipts:
        global_gain = receipt.global_gain
        local_gain = receipt.local_gain
        # A receipt whose private gain exceeds its global gain must never be
        # marked material on the private evidence alone.
        if receipt.material:
            assert global_gain > 0.0
        # The dual record exists for every segment (diagnostics preserved).
        assert receipt.local_error_before >= receipt.local_error_after - 1e-12
        assert local_gain >= 0.0


def test_probe_starvation_fails_loudly() -> None:
    # Phase-II leaves 5700 FE; four minimum probes of 1500 FE need 6000,
    # so the protocol must fail before any episode silently starves.
    with pytest.raises(Phase2StateError, match=r"probe (tax|protocol)"):
        _run(
            segment_fes=100,
            probe_min_fes=1_500,
            probe_share=0.25,
        )


def test_receipts_carry_dispatcher_and_episode_kind_fields() -> None:
    result = _run()
    assert result.dispatcher == "gcb_coordinator"
    kinds = {r.episode: r.episode_kind for r in result.receipts}
    assert kinds["gcb"] == "gss"
    assert kinds["ctp"] == "ctp"


def test_probe_order_rubric_responds_to_sensing() -> None:
    from arac.coordination.episodes import _probe_order

    # Wide relative response ranks AOR first (restart value); narrow
    # response ranks SMP first (refinement); bias-dominant sensing with
    # moderate width ranks GSS first (graph balancing).  CTP's midpoint
    # score is structurally middle-ranked under this v1 rubric.
    wide = _probe_order({"mean_bias": 0.05, "mean_relative_width": 0.9})
    assert wide[0] == "aor"
    narrow = _probe_order({"mean_bias": 0.1, "mean_relative_width": 0.05})
    assert narrow[0] == "smp"
    biased = _probe_order({"mean_bias": 0.9, "mean_relative_width": 0.5})
    assert biased[0] == "gcb"


def test_evidence_block_order_ranks_scored_blocks_first() -> None:
    from arac.coordination.episodes import evidence_block_order

    blocks = ((0, 1), (2, 3), (4, 5), (6, 7))
    ordered = evidence_block_order(blocks, [0.0, 0.9, 0.0, 0.4])
    assert ordered[0] == (2, 3)
    assert ordered[1] == (6, 7)
    # unscored blocks keep their relative order after scored ones
    assert ordered[2:] == ((0, 1), (4, 5))
    with pytest.raises(ValueError, match="block score count"):
        evidence_block_order(blocks, [0.1, 0.2])


def test_archive_adoption_is_strictly_monotone_and_fe_preserving() -> None:
    from arac.coordination.episodes import _EpisodeLedger

    problem = _problem()
    global_ledger = type("G", (), {"count": 300})()
    incumbent = tuple(0.5 for _ in range(DIMENSION))
    ledger = _EpisodeLedger(
        problem,
        total_budget_fes=6_000,
        initial_count=300,
        initial_incumbent=incumbent,
        initial_error=100.0,
        global_ledger=global_ledger,
    )
    worse = np.asarray(incumbent) + 1.0
    global_ledger.best_x = worse
    global_ledger.best_error = 50.0
    adopted, refusal = ledger.adopt_global_archive(
        source=global_ledger, accept_out_of_bounds=True
    )
    assert adopted is True and refusal == "none"
    assert ledger.best_error == 50.0
    assert ledger.count == 300
    global_ledger.best_error = 80.0  # worse than the adopted baseline
    adopted, refusal = ledger.adopt_global_archive(
        source=global_ledger, accept_out_of_bounds=True
    )
    assert adopted is False and refusal == "not_better"
    assert ledger.best_error == 50.0
    assert ledger.count == 300
    # out-of-bounds baton: accepted only under the unbounded policy
    global_ledger.best_x = np.asarray(incumbent) + 100.0
    global_ledger.best_error = 10.0
    adopted, refusal = ledger.adopt_global_archive(
        source=global_ledger, accept_out_of_bounds=False
    )
    assert adopted is False and refusal == "oob_incumbent"
    adopted, refusal = ledger.adopt_global_archive(
        source=global_ledger, accept_out_of_bounds=True
    )
    assert adopted is True and refusal == "none"


def test_handoff_receipts_record_the_relay_chain() -> None:
    result = _run()
    assert result.handoffs
    for handoff in result.handoffs:
        assert handoff.handoff_from != handoff.handoff_to
        assert handoff.handoff_mode in {
            "reanchor_next_segment", "reanchor_next_visit", "fresh_by_design", "disabled",
        }
        assert len(handoff.to_snapshot_hash) == 64
        if handoff.handoff_from:
            assert len(handoff.from_snapshot_hash) == 64
    # the aor baton is archive-only by design
    aor_handoffs = [h for h in result.handoffs if h.handoff_to == "aor"]
    if aor_handoffs:
        assert all(h.handoff_mode == "fresh_by_design" for h in aor_handoffs)


def test_disabled_handoff_records_disabled_mode() -> None:
    result = _run(handoff_enabled=False)
    assert result.handoffs
    assert all(h.handoff_mode == "disabled" and h.adopted is False for h in result.handoffs)


def test_aor_search_state_is_untouched_by_archive_adoption() -> None:
    # Adoption changes only the ledger baseline; the AOR session payload
    # (mean/covariance/candidates) must stay bit-identical.
    from arac.actions.recovered import RecoveredAorExecutor
    from arac.coordination.episodes import _EpisodeLedger
    from arac.runtime.contracts import ActionContext

    problem = _problem()
    incumbent = tuple(0.5 for _ in range(DIMENSION))
    holder = type("G", (), {"count": 300})()
    holder.best_x = np.asarray(incumbent) + 0.25
    holder.best_error = 0.5
    ledger = _EpisodeLedger(
        problem,
        total_budget_fes=6_000,
        initial_count=300,
        initial_incumbent=incumbent,
        initial_error=100.0,
        global_ledger=holder,
    )
    context = ActionContext("aor", _checkpoint(), problem, ledger, action_seed=ACTION_SEED)
    state = RecoveredAorExecutor().initialize(context)
    before = state.snapshot().state_hash
    adopted, _ = ledger.adopt_global_archive(source=holder, accept_out_of_bounds=True)
    assert adopted is True
    after = state.snapshot().state_hash
    assert before == after
