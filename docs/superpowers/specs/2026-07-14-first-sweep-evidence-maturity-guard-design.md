# First-Sweep Evidence Maturity Guard Design

Date: 2026-07-14
Executor: Codex
Status: approved for implementation under the user's standing execution authority

## Purpose

Improve the cross-seed stability of the canonical v33.8 controller on the 13
current best-of-three winning cases without using case identity, function
family, paper values, historical outcomes, or final results at runtime.

The opt-in v36 candidate replaces two weakly identified exact-key trust
decisions with run-local evidence that has a direct action interpretation:

1. a v31 non-dense repair lock is treated as mature evidence for its forced
   repair action immediately; and
2. a completed first sweep may mature sparse, unanimous coordinate actions
   for later sweeps when their confidence-weighted rank support is sufficient.

All other active actions retain the v33.8 trust guard. The topology-scoped
fallback behavior remains unchanged. v36 adds no optimizer, restart, resource
block, or objective evaluation.

This is an engineering stability candidate. It is not a publication-level
novelty claim, and passing its first gate is not sufficient to claim the final
`mean >= 6/24` stability target.

## Stability Target And Current Evidence

The fresh v33.8 full-24 result is:

- best-of-three: `13/24`;
- mean wins: `4/24`;
- worst-seed wins: `2/24`;
- seed wins: `21/72`;
- catastrophic seeds: `31/72`.

The 13 current best-of-three winning cases are `A4`, `A5`, `E1`, `E2`, `E3`,
`E4`, `E6`, `R1`, `R2`, `S2`, `S3`, `S5`, and `S6`.

v35 showed that unconditional transparent active writeback is not safe. It
restored all six S2/S3 seed wins, but lost the protected A4 and R2 cases. The
same full writeback had mixed long-horizon effects across R2 seeds, so final
outcomes cannot justify a per-seed threshold.

The v33.8/v35 trace audit provides a narrower identifiable surface:

- the existing v31 repair-lock trigger occurs only on S3 in the fresh full-24
  trace, for all three seeds, and forces repair for the rest of the run;
- S3 repair-lock rows are capped or damped by v33.8 even though the upstream
  controller has already committed to repair;
- S2 v33.8 and v35 outer-sweep 0 are bit-equivalent across all 57 relation
  rows for previous delta, current delta, and writeback norm;
- after that common sweep, all S2 seeds expose five active coordinate rows out
  of 19, one active family, and confidence-weighted mean rank support between
  `0.5181` and `0.5333`;
- E2 has the same active-count range but support only `0.4207` to `0.4658`;
  R2 reaches at most `0.4782`; A4 has mixed active families;
- across all 72 v33.8 runs, the registered sparse-coordinate rule below
  selects only the three S2 runs. The repair-lock rule selects only the three
  S3 runs. This is an offline trigger audit, not a runtime case table.

The additional S5 v35 ablation was clean but did not justify another branch.
Its errors improved from v33.8 on all seeds (`7829 -> 5884`, `40157 -> 36438`,
`62872 -> 59586`), while its mean remained about `33969`, far above paper-best
`9230`, and two seeds remained catastrophic. Its first-sweep action evidence
also overlaps A4. v36 therefore does not dispatch on an S5-specific pattern.

## Alternatives

### One-step or next-group utility gate

Rejected. v34 demonstrated that one-group credit mixes shared-variable
writeback with the following optimizer block. It is not a reliable estimate of
long-horizon action utility.

### Cumulative norm or exposure budget

Rejected for v36. S2, E2, and R2 have overlapping early exposure ranges, and a
support-only rule would also admit A3, R3, or A5 runs. Exposure alone does not
identify the protected behavior.

### First-sweep action maturity

Selected. It uses an upstream repair commitment directly and admits repeated
coordinate writeback only after a complete common sweep supplies coverage,
family consensus, and normalized support. It leaves ambiguous runs on v33.8.

## Runtime Boundary

v36 may use only evidence from the current optimizer run:

- relation topology, shared-variable indices, and the existing dense/non-dense
  run state;
- the existing v31 repair-lock state and trigger;
- relation action family, confidence, and normalized rank signal;
- relation counts and active-action counts from a completed outer sweep;
- the existing v31 value guard and v33.8 topology fallback route.

v36 must not use or derive runtime decisions from:

- case or problem identifiers;
- function-family labels;
- paper-best or reported baseline values;
- prior-run or historical outcomes;
- final error, relative gain, seed-win, mean, worst, or catastrophic labels.

Problem identifiers remain execution and offline-audit keys only.

## Runtime State

Extend the per-run v31 controller state with a v36-only sweep evidence state.
It records one current sweep and one immutable maturity decision:

```text
current_outer_iter
relation_count
active_count
active_families
confidence_rank_support_sum
coordinate_maturity_latched
coordinate_maturity_reason
```

The state is scoped to one run and is never serialized or reused across seeds.
When the next outer iteration begins, the previous sweep is finalized before
the new relation executes. Only outer sweep 0 may latch coordinate maturity;
later trajectory changes cannot retroactively select a mode.

## Maturity Rules

### Repair-lock maturity

After the existing v31 relation executor has selected a forced repair:

```text
non_dense_repair_locked is true
and canonical action is repair_shared_variable_binding
```

the v31 guarded repair writeback is committed transparently. Exact-key trust,
its `0.5` active-action cap, quarantine, and exposure state are not consulted
for that row. The trace route is `repair_lock_transparent`.

This is not a new repair detector. v36 consumes the existing v31 commitment
that already controls every subsequent relation action in that run.

### Sparse-coordinate maturity

At the end of outer sweep 0, latch coordinate maturity only when all conditions
hold:

```text
relation_count > 0
active_count >= 4
0.20 <= active_count / relation_count <= 0.30
all active actions are coordinate
mean(confidence * rank_signal over active actions) >= 0.50
repair lock is not active
```

The thresholds are normalized evidence boundaries, not fitted objective
values. `0.20..0.30` defines a minority intervention surface, while `0.50`
requires above-midpoint confidence-weighted rank support. Their observed
three-seed margins remain small, so the candidate must be described as pilot
evidence and must pass the registered controls.

When latched, later coordinate actions commit the v31 guarded writeback
transparently. Repair, isolate, and fallback actions do not inherit coordinate
maturity. The trace route is `first_sweep_sparse_coordinate_mature`.

### Protected default

Every row not covered by the two maturity rules follows v33.8 exactly:

- active actions use exact-key trust damping/quarantine;
- dense fallback preserves v31;
- non-dense fallback is bounded to norm `0.5`;
- search-state and resource-allocation behavior are unchanged.

## Implementation Boundary

- `scripts/hcc_smoke_runner.py`: add a pure sweep-evidence accumulator and an
  explicit v36 relation executor. Extract the post-v31 v33 trust application
  only if necessary to avoid running the stateful v31 executor twice.
- `src/arac/actions/contracts.py`: register one opt-in v36 trajectory action.
- `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`: expose one
  v36 lane profile.
- Tests: cover repair-lock transparency, coordinate latch boundaries, v33/v35
  isolation, trace fields, CLI wiring, and matched FE.

No second experiment runner, case route table, optimizer dependency, or result
schema truth source is introduced.

## Audit Surface

v36 action trace rows add empty-by-default fields for:

- `active_maturity_route`;
- `sweep_evidence_relation_count`;
- `sweep_evidence_active_count`;
- `sweep_evidence_active_fraction`;
- `sweep_evidence_support`;
- `sweep_evidence_reason`.

The values used to latch maturity must be reconstructable from
`action_decision.csv`. Other routes leave these fields empty. Trust fields stay
auditable on protected rows and are empty on transparent maturity rows.

## Failure Policy

- Missing, non-finite, or out-of-range confidence/rank evidence blocks maturity
  and keeps v33.8 behavior.
- An incomplete first sweep cannot latch coordinate maturity.
- A zero-norm proposal remains a no-op and creates no synthetic trust state.
- A repair lock without a repair action does not receive transparency.
- Any v33/v35 behavior regression, extra objective evaluation, FE overspend, or
  forbidden runtime field rejects the candidate.

## Verification Ladder

### Code and CLI

1. Pure tests establish every maturity boundary and fail closed on incomplete
   evidence.
2. v36 repair-lock rows equal v35/v31 active semantics.
3. v36 matured coordinate rows equal v35/v31 active semantics.
4. Non-mature v36 rows equal v33.8, including both fallback topology routes.
5. v33.8 and v35 focused suites remain unchanged.
6. Matched runner tests prove no additional objective call or FE.

### Real-HCC 5k smoke

Run S2, S3, A4, and R2 at seeds 1/2/3 with strict 5k FE. Require:

- `12/12` fresh completion and zero FE violations/overspends;
- unchanged AOB inputs and passing anti-leakage audit;
- repair-lock transparency on S3;
- sparse-coordinate maturity on S2 when a complete sweep is available;
- no maturity route on A4 or R2;
- unchanged topology fallback behavior.

### Current-winning-13 gate

Run `A4/A5/E1/E2/E3/E4/E6/R1/R2/S2/S3/S5/S6`, seeds 1/2/3, strict
3M FE. Join paper-best only after all runtime artifacts complete. The v36 stage
gate requires:

- best-of-three `13/13` on the current winning set;
- mean wins at least `5/13` and strictly above v33.8's `4/13` contribution;
- worst-seed wins at least `4/13` and strictly above v33.8's `2/13`;
- at least `24/39` seed wins;
- at most `9/39` catastrophic seeds;
- `39/39` fresh runs, zero FE overspend, unchanged AOB inputs,
  anti-leakage pass, and no case-specific dispatch.

The expected new stable cases are S2 and S3, but those labels are offline
predictions only and never enter the runtime rule.

### Full-24 and final target

Do not spend a full-24 run merely because the intermediate v36 gate passes.
The project-level adoption target remains:

- best-of-three at least `13/24`;
- mean wins at least `6/24`;
- worst-seed wins at least `4/24`;
- catastrophic seeds at most `27/72`;
- complete FE, AOB, anti-leakage, and dispatch audits.

If v36 reaches only its expected `mean=5` and `worst=4`, retain it as a
diagnostic candidate and continue with a separately attributable stability
mechanism before full-24. Do not weaken the final gate or claim completion.

## Candidate Critique

Blind acceptance criteria were fixed before selecting the implementation:
runtime legality, a named mechanism, reconstructable normalized evidence,
matched-FE attribution, protected controls, and measurable mean/worst gains.
Hard blocks were case/family/outcome dispatch, extra FE, or a signal already
confounded by the v34 result. Warnings were the three-seed sample and the small
support margin around `0.50`.

`[CONTRACT-ACKNOWLEDGED]`

Open review verdict: conditionally acceptable as an engineering ablation, not
as a novelty claim. Its strongest evidence is the matched v33.8/v35 causal
surface and the all-72 trigger audit. Its unresolved risk is that the sparse
coordinate boundary may identify only one observed topology and may not
generalize. The current-winning-13 gate and later full-24 audit are therefore
mandatory; a threshold change after seeing v36 final outcomes requires a new
candidate and cannot be folded into this route.
