# Cross-Sweep CMA Sigma Continuation Design

Date: 2026-07-14
Executor: Codex
Status: pre-registered engineering candidate

## Purpose

Add an opt-in v39 controller that inherits v38 and changes one canonical
group-search parameter: the starting CMA-ES sigma. Instead of discarding the
optimizer's terminal step-size after every group block, v39 carries a bounded
terminal-sigma factor to the next outer sweep of the same Phase-I group.

The change targets the v38 evidence-coverage failure. Retirement-triggered
precision reached A4/A5 and S2, but did not reach the remaining unstable
E6/R1/R2/S6 runs. Every protected run, including no-overlap and non-retirement
runs, executes canonical group CMA-ES blocks, so optimizer-state continuation
has complete route coverage without a case or family dispatch table.

This is an engineering stability ablation. It is not a new optimizer and is
not a standalone novelty claim.

## Source And Runtime Evidence

Both the read-only HCC source and the vendored backend expose the terminal
CMA-ES state in `ES._collect()`:

```text
results["sigma"] = self.sigma
```

`CMAES.optimize()` returns that value together with the final mean, best point,
FE count, generation count, and restart count. The ARAC executor currently
uses the best point and FE count but discards terminal sigma before the next
outer sweep.

The v38 current-winning-13 audit has 39/39 runs with positive canonical CC FE.
The remaining losses span both runtime routes:

- non-retirement losses in E6, R1, R2, and S6;
- retirement-covered but unflipped losses in E2, S5, A4, and A5.

The labels above are offline diagnosis only. They establish coverage needs and
must never enter runtime dispatch.

## Runtime Boundary

v39 may use only:

- the immutable Phase-I grouping and each group's variable-index tuple;
- the current run's v38 reference sigma for the current group block;
- the terminal `results["sigma"]` returned by that CMA-ES block;
- the current run's FE ledger and optimizer restart count for audit.

v39 must not use case identity, function family, paper values, historical or
prior-run outcomes, final error, relative gain, seed-win labels, mean/worst
labels, or catastrophic labels.

## Single-Variable Rule

The state key is the Phase-I group variable tuple. It is stable within one run
and contains no problem or case identifier. Let `reference_sigma` be exactly
the sigma that v38 would use for the current block:

```text
dense route                 -> base sigma
non-dense route             -> base sigma * 0.5
post-retirement non-dense   -> base sigma * 0.25
```

For the first visit to a group:

```text
applied_factor = 1.0
cc_sigma = reference_sigma
```

After the group optimizer returns:

```text
raw_factor = terminal_sigma / reference_sigma
next_factor = clip(raw_factor, 0.5, 1.5)
```

On the next outer-sweep visit to the same group:

```text
cc_sigma = reference_sigma * stored_next_factor
```

The lower bound is the existing deep-refine/refine ratio (`0.25 / 0.5`), and
the upper bound is the existing phase-rescue sigma multiplier (`1.5`). No
outcome-fitted threshold is introduced. The factor is relative to the current
v38 reference, so a later retirement transition still changes the underlying
reference scale. The state is scoped to one run and is never transferred
between seeds or cases.

Terminal sigma must be finite and positive. Missing or invalid backend state
is a hard runtime error; v39 must not silently fabricate a continuation value.

## Audit Surface

v39 emits one `cross_sweep_cma_sigma_continuation` trace row for every
canonical group CMA-ES block. The v39-only trace fields are:

- `cma_sigma_reference`;
- `cma_sigma_applied_factor`;
- `cma_sigma_terminal`;
- `cma_sigma_next_factor`;
- `cma_sigma_route` (`cold_start` or `continued`);
- `cma_restart_count`.

The row also records group position, actual group FE, incumbent before/after,
and whether the group incumbent improved. v1-v38 trace schemas remain
unchanged.

## Failure Policy

- v33-v38 behavior and trace schemas must remain unchanged.
- The first visit to every group must use the exact v38 reference sigma.
- v39 may change only canonical group starting sigma; mean, population,
  budget, optimizer seed, relation policy, rescue, retirement, and writeback
  behavior remain unchanged.
- v39 may not add an optimizer call, objective evaluation, or hidden fallback.
- A group may consume only its own current-run terminal-sigma state.
- Any forbidden dispatch source, FE overspend, AOB change, or trace mismatch
  rejects the candidate.

## Verification Ladder

1. Pure tests cover cold start, per-group isolation, lower/upper clipping,
   invalid terminal state, and v38 isolation.
2. Matched-FE tests prove v39 changes only the second visit's sigma and adds no
   objective call.
3. A real-HCC 5k smoke verifies CLI, fresh execution, FE, AOB, leakage, and
   v39 trace reconstruction across dense, non-dense, and no-overlap routes.
4. A 3M route probe on one non-retirement run must show at least two visits to
   one group and a non-unit continued factor.
5. Run the current-winning-13, seeds 1/2/3, strict 3M FE.

The strengthened development release gate is fixed before execution:

- best-of-three `13/13`;
- mean wins at least `6/13`;
- worst-seed wins at least `5/13`;
- seed wins at least `26/39`;
- catastrophic seeds at most `7/39`;
- S2/S3 retain `6/6` seed wins;
- `39/39` fresh, zero FE overspend, unchanged AOB inputs, anti-leakage pass,
  reconstructable continuation rows, and no case-specific dispatch.

Only a passing candidate may proceed to held-out seeds and full-24. The final
full-24 target remains best at least `13/24`, mean at least `6/24`, worst at
least `4/24`, and catastrophic at most `27/72`. Three development seeds remain
pilot evidence and must not be described as statistical significance.

## Candidate Critique

The mechanism uses optimizer-internal evidence with universal route coverage,
but terminal sigma is not a causal estimate of long-horizon group utility.
Carrying only sigma also discards covariance and evolution paths, so this is a
minimal continuity test rather than full CMA-ES state continuation. Reuse of
seeds 1/2/3 creates development-set overfitting risk; a passing result requires
held-out seed validation before a robust claim.

Verdict: conditionally acceptable as a single-variable, reference-blind
stability ablation.

`[CONTRACT-ACKNOWLEDGED]`
