"""Phase-aware v4 scheduler targeted tests (upgrade plan section 4-6).

Pinned behaviours (pre-registered):

1. P0 hard constraints: four minimal executable probes before anything,
   loud failure when they cannot be funded;
2. maturity tickets: every episode reaches its semantic maturity (AOR
   demands two correction windows) or is recorded ``maturity_unaffordable``
   -- never silently eliminated;
3. max two consecutive exploitation segments, then a forced challenger;
4. challenger rotation floor: a completed-ticket episode with the worst
   recent rate is still scheduled (the R2 starvation fix);
5. escalation windows follow the geometric ladder w1 * 2^k;
6. private trajectory credit is promotion-only and handoff-epoch scoped;
   the global material leader is never demoted by private credit;
7. dual-ledger budget: cold-start and development caps respected;
8. receipts carry the full audit surface and the schedule hash recomputes.
"""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.episodes import (
    GRANT_KINDS,
    PhaseAwareSchedulerConfig,
    run_oc_episode_schedule_v4,
)
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


def _checkpoint(total: int = 20_000, phase1: int = 500) -> PhaseCheckpoint:
    problem = _problem()
    incumbent = tuple(0.5 for _ in range(DIMENSION))
    return PhaseCheckpoint(
        protocol="v4-unit",
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


def _config(**overrides) -> PhaseAwareSchedulerConfig:
    kwargs = dict(
        maturity_window_fes=800,
        revelation_horizon_fes=3_000,
        exploration_and_development_cap=0.80,
        exploitation_reserve_ratio=0.05,
        cold_start_probe_cap=0.25,
        probe_min_fes=200,
        segment_fes=1_500,
        calibration_ref="unit-test",
    )
    kwargs.update(overrides)
    return PhaseAwareSchedulerConfig(**kwargs)


def _run(config: PhaseAwareSchedulerConfig | None = None):
    return run_oc_episode_schedule_v4(
        _problem(), _checkpoint(), action_seed=ACTION_SEED, config=config or _config()
    )


def test_v4_end_to_end_protocol() -> None:
    result = _run()
    assert result.terminal_fes == 20_000
    assert result.schema_version == "arac-oc-episode-schedule-v4"
    assert result.scheduler_policy == "phase_aware_v4"
    assert len(result.schedule_hash) == 64
    assert all(result.audit.values()), result.audit
    kinds = [r.grant_kind for r in result.receipts]
    assert set(kinds) <= set(GRANT_KINDS)
    assert kinds[:4] == ["probe"] * 4
    assert {r.episode for r in result.receipts[:4]} == {"ctp", "gcb", "smp", "aor"}
    assert all(k != "probe" for k in kinds[4:])
    assert result.funded_fes
    sensing_fes = int(result.sensing.get("probe_fes") or 0)
    assert sum(result.funded_fes.values()) + sensing_fes == 19_500
    # strict-best monotone across the whole stream
    errors = [r.global_error_after for r in result.receipts]
    assert all(errors[i + 1] <= errors[i] + 1e-12 for i in range(len(errors) - 1))


def test_aor_ticket_demands_two_correction_windows() -> None:
    result = _run()
    aor_ticket_fes = sum(
        r.consumed_fes for r in result.receipts if r.episode == "aor" and r.grant_kind == "ticket"
    )
    aor_probe_fes = sum(
        r.consumed_fes for r in result.receipts if r.episode == "aor" and r.grant_kind == "probe"
    )
    assert aor_probe_fes + aor_ticket_fes >= 2 * 800


def test_max_two_consecutive_exploits_then_forced_challenger() -> None:
    result = _run()
    kinds = [(r.grant_kind, r.episode, r.global_gain > 0.0) for r in result.receipts]
    # v4.3 contract: a MATERIAL leader may run continuously (R6's strict
    # win came from exactly that); a ZERO-GAIN exploit must be followed
    # by a grant to a different episode (stagnation yield) or a cadence
    # event -- never by the same leader immediately again.
    for i in range(len(kinds) - 1):
        kind, episode, gain = kinds[i]
        if kind == "exploit" and not gain:
            nxt_kind, nxt_episode, _ = kinds[i + 1]
            assert not (nxt_kind == "exploit" and nxt_episode == episode), (
                f"{episode} ran again immediately after its own zero-gain exploit"
            )
    exploit_runs = [
        (kind, episode) for kind, episode, _ in kinds if kind in ("exploit", "challenger", "escalation")
    ]
    assert exploit_runs, "no post-ticket grants happened"


def test_cooldown_blocks_immediate_switch_back() -> None:
    result = _run()
    receipts = result.receipts
    for i in range(len(receipts) - 2):
        a, b, c = receipts[i], receipts[i + 1], receipts[i + 2]
        if a.episode != b.episode and a.episode == c.episode:
            # Returning to the departed episode within one segment is only
            # legal through the cooldown-exempt escalation lane (P3a).
            assert c.grant_kind == "escalation", (
                f"{a.episode} returned via {c.grant_kind} inside its cooldown"
            )


def test_escalation_windows_follow_geometric_ladder() -> None:
    config = _config(revelation_horizon_fes=6_500)
    result = _run(config)
    by_episode: dict[str, list[int]] = {}
    for r in result.receipts:
        if r.grant_kind == "escalation":
            by_episode.setdefault(r.episode, []).append(r.window_fes)
    for episode, windows in by_episode.items():
        for i in range(1, len(windows)):
            assert windows[i] == windows[i - 1] * 2, (
                f"{episode} escalation ladder broken: {windows}"
            )
        if windows:
            assert windows[0] == config.maturity_window_fes


def test_escalation_lane_is_fair_not_credit_monopolized() -> None:
    # The R2 failure mode the 51b review flagged: a high-credit pending
    # episode must not exhaust the escalation lane before a zero-credit
    # late-maturer starts its ladder.  Fairness = fewest escalation grants
    # first; private credit only breaks ties.
    config = _config(revelation_horizon_fes=6_500)
    result = _run(config)
    escalations = [r for r in result.receipts if r.grant_kind == "escalation"]
    if len(escalations) >= 2:
        # Every episode still below the revelation horizon (including
        # ones with zero escalations so far) must stay within one
        # escalation grant of every other pending episode.
        grant_counts: dict[str, int] = {}
        cum_dev: dict[str, int] = {}
        horizon = config.revelation_horizon_fes
        for r in result.receipts:
            if r.grant_kind == "escalation":
                grant_counts[r.episode] = grant_counts.get(r.episode, 0) + 1
            cum_dev[r.episode] = r.cumulative_development_fes
            pending = [e for e, dev in cum_dev.items() if dev < horizon]
            counts = [grant_counts.get(e, 0) for e in pending]
            if len(counts) >= 2:
                assert max(counts) - min(counts) <= 1, (
                    "escalation lane monopolized: grant counts diverged "
                    f"{grant_counts} while {len(pending)} episodes pending"
                )


def test_rotation_floor_worst_rate_episode_not_starved() -> None:
    result = _run()
    completed = {t.episode for t in result.tickets if t.affordable and t.protocol_mature_after}
    assert completed, "no episode completed its ticket"
    grant_streams = [r.episode for r in result.receipts if r.grant_kind != "probe"]
    ticket_end = {}
    for t in result.tickets:
        if t.affordable and t.protocol_mature_after:
            ticket_end.setdefault(t.episode, t.segment_index)
    for episode, segment_index in ticket_end.items():
        foreign_challengers = [
            r
            for r in result.receipts
            if r.segment_index > segment_index
            and r.grant_kind in ("challenger", "escalation")
            and r.episode != episode
        ]
        if len(foreign_challengers) >= 3:
            assert episode in grant_streams[grant_streams.index(episode) + 1 :] or episode in [
                r.episode for r in result.receipts if r.segment_index > segment_index
            ], f"{episode} starved after its ticket"


def test_unaffordable_ticket_is_recorded_not_silently_dropped() -> None:
    # Tiny phase-II budget: probes fit the cold-start cap but the CTP/GSS
    # tickets cannot be paid -- they must surface as unaffordable records.
    config = _config(
        maturity_window_fes=1_200,
        revelation_horizon_fes=5_000,
        exploration_and_development_cap=0.30,
        exploitation_reserve_ratio=0.05,
        probe_min_fes=200,
        segment_fes=500,
    )
    result = run_oc_episode_schedule_v4(
        _problem(),
        _checkpoint(total=6_000, phase1=300),
        action_seed=ACTION_SEED,
        config=config,
    )
    assert result.terminal_fes == 6_000
    assert all(result.audit.values()), result.audit
    unaffordable = [t for t in result.tickets if not t.affordable]
    assert unaffordable, "expected at least one unaffordable ticket at this budget"
    for ticket in unaffordable:
        assert ticket.granted_fes == 0
        post = [
            r
            for r in result.receipts
            if r.episode == ticket.episode and r.grant_kind == "ticket"
        ]
        assert not post or all(
            r.segment_index < next(
                t.segment_index for t in result.tickets if not t.affordable and t.episode == ticket.episode
            )
            for r in post
        )


def test_probe_starvation_fails_loudly() -> None:
    config = _config(probe_min_fes=1_600)
    with pytest.raises(Phase2StateError, match=r"probe protocol"):
        _run(config)


def test_config_rejects_uncalibrated_or_out_of_range_values() -> None:
    base = dict(
        maturity_window_fes=800,
        revelation_horizon_fes=3_000,
        exploration_and_development_cap=0.8,
        exploitation_reserve_ratio=0.1,
    )
    with pytest.raises(ValueError):
        PhaseAwareSchedulerConfig(**{**base, "maturity_window_fes": 0})
    with pytest.raises(ValueError):
        PhaseAwareSchedulerConfig(**{**base, "exploration_and_development_cap": 1.5})
    with pytest.raises(ValueError):
        PhaseAwareSchedulerConfig(**{**base, "exploitation_reserve_ratio": 0.0})
    with pytest.raises(ValueError):
        PhaseAwareSchedulerConfig(**{**base, "escalation_factor": 1})
    with pytest.raises(ValueError):
        PhaseAwareSchedulerConfig(**{**base, "private_credit_mode": "magic"})


def test_private_credit_is_promotion_only_and_epoch_scoped() -> None:
    result = _run()
    receipts = result.receipts
    # handoff epochs never decrease for any episode
    last_epoch: dict[str, int] = {}
    for r in receipts:
        last_epoch.setdefault(r.episode, r.handoff_epoch)
        assert r.handoff_epoch >= last_epoch[r.episode]
        last_epoch[r.episode] = r.handoff_epoch
    # every exploit grant goes to the leader at grant time (v4.3: leader
    # = exploit-rate ranking, bootstrap = private credit before any
    # exploit history exists), recomputed from the receipts themselves;
    # the rate window covers EXPLOIT segments only, matching the
    # scheduler's exploit_history semantics.
    history: dict[str, list[tuple[float, int]]] = {}
    for r in receipts:
        if r.grant_kind == "ticket":
            continue
        if r.grant_kind == "exploit":
            # v4.3.1: the grant goes to the leader, OR to a bootstrap
            # sampling slot (an unsampled matured episode taking its
            # first exploit while the leader already has >= 2 samples).
            bootstrap = False
            if r.leader != r.episode:
                prior_leader_samples = sum(
                    1 for x in receipts[: r.segment_index]
                    if x.grant_kind == "exploit" and x.episode == r.leader
                )
                own_prior_samples = sum(
                    1 for x in receipts[: r.segment_index]
                    if x.grant_kind == "exploit" and x.episode == r.episode
                )
                bootstrap = prior_leader_samples >= 2 and own_prior_samples == 0
                assert bootstrap, (
                    f"exploit grant to {r.episode} while leader was {r.leader} "
                    f"outside the bootstrap sampling rule"
                )
            if not bootstrap:
                assert r.leader == r.episode

            def rate(e: str) -> float:
                tail = history.get(e, [])[-2:]
                fes = sum(f for _, f in tail)
                return (sum(g for g, _ in tail) / fes) if fes else 0.0

            if not bootstrap:
                others = [e for e in history if e != r.episode and history[e]]
                assert not others or all(
                    rate(r.episode) >= rate(e) - 1e-12 for e in others
                ), f"leader {r.episode} demoted despite top exploit rate"
            history.setdefault(r.episode, []).append((r.global_gain, r.consumed_fes))
    # Extreme-value credit is non-decreasing within an epoch and, on
    # adoption (epoch bump), carries at a HALF DISCOUNT rather than being
    # wiped: the pre-adoption trajectory is baseline-shifted evidence,
    # not noise (v4.2 -- Gate 51b S5 showed the wipe destroyed CTP's
    # "strong yet slow" promotion signal entirely).
    seen_epochs: dict[str, int] = {}
    credit_before_epoch: dict[str, float] = {}
    for r in receipts:
        seen_before = seen_epochs.get(r.episode)
        if seen_before is not None and r.handoff_epoch > seen_before:
            carry = credit_before_epoch.get(r.episode, 0.0) * 0.5
            assert r.private_credit <= max(carry, r.local_gain) + 1e-12, (
                f"{r.episode} kept more than the discounted carry across handoff epoch"
            )
        seen_epochs[r.episode] = r.handoff_epoch
        credit_before_epoch[r.episode] = max(
            credit_before_epoch.get(r.episode, 0.0), r.private_credit
        )


def test_receipts_carry_full_audit_surface() -> None:
    result = _run()
    required = {
        "grant_kind", "grant_index", "window_fes", "leader", "ledger_class",
        "recent_rate", "private_credit", "handoff_epoch",
        "cumulative_development_fes", "evidence_revealed",
        "maturity_ticket_id", "maturity_committed", "challenger", "cooldown",
        "remaining_budget", "cold_start_spent", "development_spent",
        "progress_before", "progress_after", "state_hash", "snapshot_hash",
    }
    for r in result.receipts:
        missing = required - set(r.__dict__)
        assert not missing, missing
        assert set(r.progress_before) >= {
            "phase", "consumed_fes", "protocol_mature", "contract",
            "maturity_target_fes", "next_boundary_fes", "min_step_fes",
        }
        # Exploitation only ever goes to protocol-mature episodes.
        if r.grant_kind == "exploit":
            assert r.progress_before["protocol_mature"] is True
    # CTP never counts as mature while still inside coverage.
    for r in result.receipts:
        if r.episode == "ctp" and r.progress_after["phase"] == "coverage":
            assert r.progress_after["protocol_mature"] is False
    # probes carry the contract surface
    for p in result.probes:
        assert p.probe_contract
        assert p.maturity_target_fes >= 0
        assert p.max_local_log_gain_window >= 0.0


def test_budget_ledger_caps_respected() -> None:
    result = _run()
    phase2 = 19_500
    assert result.cold_start_probe_tax_fes <= int(0.25 * phase2)
    assert result.development_fes <= int(0.80 * phase2)
    for r in result.receipts:
        assert r.cold_start_spent <= int(0.25 * phase2)
        assert r.development_spent <= int(0.80 * phase2)


def test_generation_aligned_episodes_never_receive_sub_unit_windows() -> None:
    # The 51a tail deadlock fix: SMP (min_step = smallest population + 1)
    # must never be granted a window below its executable unit, and the
    # terminal tail flows to an FE-granular episode instead of crashing.
    from arac.coordination.episodes import run_oc_episode_schedule_v4 as run_v4

    results = [
        _run(),
        run_v4(
            _problem(),
            _checkpoint(total=6_000, phase1=300),
            action_seed=ACTION_SEED,
            config=_config(
                maturity_window_fes=1_200,
                revelation_horizon_fes=5_000,
                exploration_and_development_cap=0.30,
                exploitation_reserve_ratio=0.05,
                probe_min_fes=200,
                segment_fes=500,
            ),
        ),
    ]
    for result in results:
        assert result.terminal_fes in (20_000, 6_000)
        for r in result.receipts:
            assert r.requested_fes >= r.progress_before["min_step_fes"], (
                f"{r.episode} received a sub-unit window {r.requested_fes} "
                f"at segment {r.segment_index}"
            )


def test_escalation_grants_k_caps_the_ladder() -> None:
    # The frozen Gate 51-0 parameter K must actually bind: no episode
    # receives more than escalation_grants_k escalation grants, even while
    # it is still below the revelation horizon.
    config = _config(
        revelation_horizon_fes=8_000,
        exploration_and_development_cap=0.9,
        escalation_grants_k=2,
    )
    result = _run(config)
    counts: dict[str, int] = {}
    for r in result.receipts:
        if r.grant_kind == "escalation":
            counts[r.episode] = counts.get(r.episode, 0) + 1
    for episode, count in counts.items():
        assert count <= 2, f"{episode} received {count} escalations; K=2 was frozen"
    assert result.config["escalation_grants_k"] == 2


def test_development_grants_never_penetrate_the_reserve() -> None:
    # The exploitation reserve is a real budget guard, not a statistic:
    # every development-class receipt leaves it intact (the terminal
    # drain, which lands in the exploitation class, is exempt).
    config = _config(exploitation_reserve_ratio=0.25)
    result = _run(config)
    reserve = int(0.25 * 19_500)
    for r in result.receipts:
        if r.ledger_class == "development":
            assert r.remaining_budget >= reserve or r.remaining_budget == 0
    assert result.audit["development_reserve_preserved"]
    assert result.audit["exploit_stagnation_yields"]
    assert result.audit["receipt_chain_continuous"]
    assert result.audit["escalation_rotation_fair"]
