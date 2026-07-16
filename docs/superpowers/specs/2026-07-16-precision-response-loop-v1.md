# Precision Response Loop v1 Preregistration

Date: 2026-07-16
Executor: Codex
Status: frozen before new optimizer FE

## Estimand And Boundary

`precision-response-loop-v1` tests whether a current-trajectory, same-direction
precision response can safely authorize one short precision group block. It
does not train the former 16-feature utility model, revive v41, authorize a
repeat lease, or change writeback/resource channels.

Runtime identity and outcome boundaries remain strict. Case, seed, function
family, graph/group/component identifiers, raw objective values, paper-best,
historical results, and terminal outcomes cannot enter the gate. Structural
topology and FE state may establish feasibility. Only derived paired-probe
responses enter the release decision.

## Opportunity

The first safe overlap revisit is eligible when the current group has already
completed a visit, `outer_iter >= 1`, its non-dense overlap component has shared
variables, no component lease or trajectory/search confirmation is pending,
and the trajectory has not consumed its one probe. The deterministic scheduler
must remain able to close the current group block and the next same-group
revisit after reserving 32 probe FE. The opportunity does not require the old
16-feature history or phase-rescue retirement.

## Probe And Gate

Two fresh one-generation vendor CMA instances use the same checkpoint mean,
population 16, counter-based seed, standardized directions and `legacy_none`
boundary behavior. Normal uses the v37 sigma; precision uses the v38 sigma.
Each arm consumes 16 real objective evaluations. Neither arm updates CMA,
group, controller, or formal-block state. The best of all 32 candidates may
update only the guarded global incumbent archive in both A1 and A2.

For each pair, with minimization error values `f_b` and `f_p`:

```text
r_i = clip((f_b - f_p) / max(abs(f_b), abs(f_p), 1e-300), -1, 1)
win = f_p < f_b
large_loss = f_p >= 1.2 * max(f_b, 1e-300)
```

The fixed gate releases only when every check passes: 16 direction pairs match;
the one-sided 95% Wilson win LCB is greater than 0.55; the large-loss Wilson UCB
is at most 0.20; median `r_i` is positive; standardized diversity ratio is at
least 0.95; precision boundary-hit rate is at most 25% and no greater than the
normal rate; and the best precision candidate strictly improves the checkpoint.
Non-finite or negative errors fail closed. The canonical numeric configuration
is `configs/precision_response_loop_v1.json` and may not be tuned after smoke.

## Arms And Delayed Review

- A0: unmodified v37 plus side-effect-free opportunity logging.
- A1: paired probe and identical guarded-archive update, then a normal v37 block.
- A2: the same probe/archive update; one precision block only when the gate passes,
  otherwise the same normal block as A1.

All arms use the same absolute terminal FE. Probe FE is charged to the 3M
ledger. A released lease is reviewed at the next canonical visit of the same
group using existing component gain, neighbor gain, overwrite/survival,
restored normal sigma, standardized diversity recovery, and prior same-group
normal progress. v1 records the review but never renews the lease.

## Pilot Matrix And Hard Stop

Before the pilot, run E1/E2 seeds 1/2 at 5k in all three arms, then A4 seed 1
at 100k. If A4 has no opportunity, permit one trace-only escalation to 3M.

Run A0 opportunity coverage for A4/A5/E1/E2/E3/E4/S2/S5, seeds 60-64, strict
3M FE. Continue to A1/A2 only when at least 30/40 trajectories are applicable,
covering at least six cases and all five seeds.

The pilot passes only with complete integrity, at least ten releases across
four cases and three seeds, positive one-sided 95% two-way case-by-seed
bootstrap LCBs for A2-A0 and released A2-A1, non-negative medians, positive
seed-stratified means for at least three seeds, released wins exceeding losses,
at least five positive lease effects of 1% or more, zero 20% catastrophic
losses for every contrast, and 100% delayed-credit closure.

Any failed gate is a completed no-go result. No runtime model, full-24 run, or
performance claim is authorized by this preregistration.
