# Post-Retirement Precision Reanchor Design

Date: 2026-07-14
Executor: Codex
Status: pre-registered engineering candidate

## Purpose

Add an opt-in v38 controller that inherits v37 exactly and changes one search
starting-state parameter after v37 has already retired zero-yield phase rescue.
For every later canonical group optimizer, v38 keeps the current group
incumbent as the mean and contracts sigma from the existing non-dense refine
level to the repository's existing deep-refine level.

The v37 result is the direct ablation control. The only v38-v37 difference is
the post-retirement canonical group search scale.

## Evidence And Root Cause

v37 reduced phase-rescue rows from `204` to `140` in the current-winning-13
gate but left aggregate performance at best/mean/worst `13/5/4`. Relative to
v36, only seven of 39 final errors changed and the changes were small. Returning
FE to canonical CC at the same search scale therefore did not remove the local
platform.

The post-retirement trace is strongly asymmetric without using case labels at
runtime:

- each A4/A5 seed retained 30 to 35 complete outer sweeps after retirement;
- S2 retained 6 to 7 sweeps;
- retired E/R/S5 runs usually retained only 1 to 4 sweeps;
- runs with an accepted rescue before the boundary never retired.

This supplies a current-run search-start surface: repeated broad rescue failed,
rescue was retired, and substantial budget remains. Continuing the same CC
search scale is a weak response to that evidence.

## Runtime Boundary

v38 may use only:

- v37's immutable `phase_rescue_retired` state;
- the current group incumbent already used as the canonical CMAES mean;
- the configured base sigma and existing fixed refine multipliers;
- the current run's FE ledger and group trajectory.

v38 must not use case identity, function family, paper values, historical
outcomes, final error, relative gain, win labels, or catastrophic labels.

## Search-Start Rule

Before retirement, v38 equals v37:

```text
cc_mean = current group incumbent
cc_sigma = base_sigma * REPAIR_PROTECT_REFINE_SIGMA_MULTIPLIER
```

After v37 retirement latches:

```text
cc_mean = current group incumbent
cc_sigma = base_sigma * REPAIR_PROTECT_DEEP_REFINE_SIGMA_MULTIPLIER
```

The multipliers are existing executor constants (`0.5` and `0.25`). No new
outcome-fitted threshold is introduced. The optimizer seed, population, group
budget, relation policy, v36 maturity, and v37 rescue retirement remain
unchanged.

The runtime trace emits one `post_retirement_precision_reanchor` trajectory row
for each affected group optimizer. It records outer/group position, normal and
contracted sigma, actual CC block FE, incumbent before/after, and whether the
group improved.

## Failure Policy

- v33-v37 must remain unchanged.
- No precision row may appear before rescue retirement.
- Productive-rescue maturity must continue to block retirement and precision
  reanchoring.
- v38 may not add FE, restart an optimizer, or change AOB inputs.
- Missing state keeps v37 behavior.
- Any forbidden runtime dispatch input rejects the candidate.

The critical unverified hypothesis is that post-retirement stagnation is local
enough for a smaller search scale to help. If contraction merely reduces
exploration without improving the incumbent, v38 must fail its protected gate.

## Verification Ladder

1. Pure tests prove v38 equals v37 before retirement and uses the existing
   deep-refine multiplier only after retirement.
2. Runtime tests prove precision rows are post-retirement, matched-FE, and
   absent from v37.
3. A 5k real-HCC smoke verifies CLI, FE, AOB, leakage, and protected routes;
   precision may be unreachable at 5k.
4. An A4 seed-1 3M probe must record precision rows after one retirement row.
5. Run the current-winning-13, seeds 1/2/3, strict 3M FE.

The full-24 release gate is unchanged:

- best-of-three `13/13`;
- mean wins at least `6/13`;
- worst-seed wins at least `4/13`;
- seed wins at least `24/39`;
- catastrophic seeds at most `9/39`;
- S2/S3 retain `6/6` seed wins;
- `39/39` fresh, no FE overspend, AOB unchanged, anti-leakage pass, and no
  case-specific dispatch.

Only a passing candidate may run full-24. The final project target remains
best at least `13/24`, mean at least `6/24`, worst at least `4/24`, and
catastrophic at most `27/72`.

## Candidate Critique

Blind hard blocks are outcome-tuned sigma, early-rescue changes, extra FE,
v36/v37 drift, or a weaker gate. The strongest attribution evidence is the v37
control: retirement alone changed resource allocation but not aggregate
performance, so v38-v37 isolates the search-scale contraction. The main warning
is reuse of the same three development seeds and the lack of a theoretical
guarantee that smaller sigma matches the post-retirement landscape.

Verdict: conditionally acceptable as an engineering ablation, not as a
standalone novelty or final-performance claim.

`[CONTRACT-ACKNOWLEDGED]`
