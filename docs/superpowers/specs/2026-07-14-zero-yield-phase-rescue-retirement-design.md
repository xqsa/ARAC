# Zero-Yield Phase-Rescue Retirement Design

Date: 2026-07-14
Executor: Codex
Status: pre-registered engineering candidate

## Purpose

Build an opt-in v37 controller on top of the corrected v36 candidate. v37 keeps
all v36 relation-writeback and maturity behavior, but adds one independently
attributable resource-allocation action: stop spending FE on phase-rescue
multistart after repeated current-run evidence shows that rescue has not yet
produced an accepted candidate.

The candidate does not change the rescue optimizer, rescue search starts,
acceptance rule, relation policy, or total FE budget. Retired rescue FE remains
available to the canonical cooperative-coevolution loop.

This is an engineering stability ablation, not a publication-level novelty
claim.

## Evidence And Motivation

The sigma-corrected v36 current-winning-13 run produced:

- best-of-three `13/13`;
- mean wins `5/13`;
- worst-seed wins `4/13`;
- seed wins `24/39`;
- catastrophic seeds `9/39`.

v36 is bit-equivalent to v33.8 on all 33 case-seed rows without a maturity
route. Only S2 and S3 changed, and they reached six seed wins out of six. This
isolates the remaining stability gap from relation writeback.

The phase-rescue trace exposes a separate resource surface. Each rescue block
already runs three independent starts and is accepted only when its best
candidate improves the current incumbent. In the fresh full-24 v33.8 trace,
57 runs attempted phase rescue. A pure replay of the rule below would latch in
32 runs and suppress at least one later rescue in 21 runs, returning about
1.05M FE to canonical CC across 72 runs. The effectful trigger covers A1-A5 and
small subsets of R3/R4/R5/S1/S2. This is an offline trigger audit, not a case
route table and not evidence that the final result will improve.

## Alternatives

### Tune v36 maturity thresholds

Rejected. v36 passed its registered gate after the sigma fix. Changing its
thresholds after observing final outcomes would destroy attribution.

### Disable phase rescue globally

Rejected. E4, R1, and R2 contain accepted rescue blocks with material
current-run improvement. A global disable would ignore direct utility evidence.

### Retire after cumulative paper-relative loss or a case-specific pattern

Rejected. Paper values, case identity, family labels, relative gain, and final
outcomes are offline-only evidence and illegal runtime inputs.

### Pre-maturity zero-yield retirement

Selected. It consumes only the existing accepted/rejected rescue result from
the current trajectory. It changes resource allocation without changing the
relation-writeback mechanism or adding FE.

## Runtime Boundary

v37 may use only:

- the current run's phase-rescue attempts;
- whether each completed rescue block produced an accepted incumbent update;
- the existing fixed three-start rescue contract;
- the v36 per-run state, topology, trust, and maturity evidence;
- the current FE ledger and remaining budget.

v37 must not use:

- case or problem identifiers;
- function-family labels;
- paper-best or reported baseline values;
- historical or prior-run outcomes;
- final error, relative gain, seed-win, mean, worst, or catastrophic labels.

Problem identifiers remain execution and offline-audit keys only.

## Runtime State And Rule

Extend the per-run controller state with:

```text
v37_enabled
phase_rescue_rejected_before_maturity
phase_rescue_productive_mature
phase_rescue_retired
phase_rescue_resource_reason
```

The retirement threshold is the existing `PHASE_RESCUE_START_COUNT` (`3`). A
completed rescue block already evaluates three independently perturbed starts,
so three rejected blocks represent nine failed rescue starts. The threshold is
bound to the executor contract rather than fitted to a final objective value.

After each completed phase-rescue block:

```text
if v37 is disabled:
    preserve v36 exactly
elif a rescue candidate is accepted before retirement:
    latch productive maturity for the run
elif productive maturity is not latched:
    increment the rejected-block count
    if rejected-block count >= PHASE_RESCUE_START_COUNT:
        retire phase rescue for the rest of the run
```

Once an accepted rescue latches productive maturity, later rejected blocks do
not retire rescue. Once retirement latches, later phase-rescue optimizer calls
are skipped. Both decisions are immutable and scoped to one run.

## Protected Defaults

- v33-v36 behavior remains unchanged.
- v37 inherits v36 writeback maturity exactly.
- Dense overlap and repair-lock phase-rescue exclusions remain unchanged.
- A run with an accepted rescue before the rejection boundary keeps the v36
  phase-rescue behavior for the rest of the run.
- Missing or incomplete evidence cannot retire rescue.
- No objective evaluation is added, replayed, or hidden.

## Audit Surface

v37 action traces add:

- `phase_rescue_resource_route`;
- `phase_rescue_rejected_before_maturity`;
- `phase_rescue_productive_mature`;
- `phase_rescue_retired`.

The third rejected rescue row records `zero_yield_phase_rescue_retired`. The
first accepted rescue row records `productive_phase_rescue_mature`. Other rows
leave the transition route empty while retaining reconstructable state fields.

## Failure Policy

Reject v37 if it:

- changes v36 relation decisions or non-dense refine sigma;
- retires rescue after an earlier accepted rescue;
- adds objective evaluations or overspends FE;
- changes AOB inputs or fails anti-leakage;
- uses a forbidden runtime field;
- cannot reconstruct the retirement transition from action trace rows.

The key unverified hypothesis is that FE reclaimed from repeatedly rejected
rescue blocks has better long-horizon utility in canonical CC. A rejected
rescue is only local evidence; it does not prove every future rescue would
fail. The protected gate must decide this empirically.

## Verification Ladder

1. Pure tests cover three rejected blocks, productive maturity, immutability,
   v36 isolation, and trace formatting.
2. Matched-FE tests prove v37 adds no objective call and preserves v36 before
   the retirement boundary.
3. Real-HCC 5k smoke covers CLI, provenance, FE, AOB, leakage, and unchanged
   relation routes. Retirement need not be reachable at 5k.
4. One A4 seed-1 3M route probe must record the retirement transition and fewer
   rescue rows than v36 without an FE overspend.
5. Run the 13 current winning cases at seeds 1/2/3 and strict 3M FE.

The v37 full-24 release gate is fixed before execution:

- best-of-three `13/13`;
- mean wins at least `6/13`;
- worst-seed wins at least `4/13`;
- seed wins at least `24/39`;
- catastrophic seeds at most `9/39`;
- S2/S3 retain `6/6` seed wins;
- `39/39` fresh, zero FE overspend, unchanged AOB inputs, anti-leakage pass,
  and no case-specific dispatch.

Only if this gate passes may v37 run the full 24-case protocol. The final
project target remains best at least `13/24`, mean at least `6/24`, worst at
least `4/24`, and catastrophic at most `27/72`.

## Candidate Critique

Blind acceptance criteria were runtime legality, a named resource mechanism,
same-budget attribution, v36 isolation, protected successful-rescue controls,
and a measurable mean-win gain. Hard blocks were outcome dispatch, global
rescue disable, extra FE, or weakening the final gate. Warnings are reuse of
the same three development seeds and the possibility that later canonical CC
cannot exploit the returned FE.

Verdict: conditionally acceptable as an engineering ablation. It is not
accepted as a standalone novelty claim, and any positive result remains
pilot-level until broader seeds or an external validation protocol exists.

`[CONTRACT-ACKNOWLEDGED]`
