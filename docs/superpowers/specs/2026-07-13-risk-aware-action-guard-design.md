# Risk-Aware Action Guard Design

Date: 2026-07-13
Executor: Codex
Status: approved for implementation

## Purpose

Improve cross-seed robustness of the evidence-to-action controller without
changing the frozen canonical v32 route. The new opt-in v33 route tracks
whether a runtime relation/action has earned trust, limits early intervention,
and quarantines actions that repeatedly produce weak or unstable downstream
evidence.

The change targets the actual failure mode observed in the full-24 audit:
some high-variance cases are harmed by a small number of aggressive shared
variable interventions. It does not claim that a relation action's local
fitness delta is a causal effect; the trace records the signal as a runtime
credit proxy and keeps the policy conservative.

## Non-goals and leakage boundary

- v32 remains the default and its outputs must be unchanged for the same seed.
- No new restart, optimizer, or action portfolio is introduced.
- Runtime dispatch cannot read case labels, function families, paper values,
  historical outcomes, final error, relative gain, or offline win tables.
- The controller never runs multiple complete action lanes and chooses the
  best final result.
- The policy does not add objective evaluations. All evidence comes from
  already evaluated CC blocks and existing writeback context.

## Runtime chain

```text
Phase-I grouping and overlap evidence
  -> v31 relation decision and value-delta guard
  -> v33 trust lookup for stable relation/action key
  -> probation/trusted/quarantined action decision
  -> bounded robust writeback
  -> downstream CC signal credits the pending action
  -> trace trust phase, reason, exposure, and credit
```

The stable key is built from group positions, shared-variable identity, and
canonical action name. It deliberately excludes `outer_iter` so evidence can
accumulate across sweeps, and it is scoped to one run so no historical state
is reused.

## Trust state machine

Each key starts in `probation`. The state stores only bounded runtime state:

- attempts and cumulative intervention exposure;
- EWMA gain and gain dispersion;
- positive/no-gain and harmful streaks;
- instability count and cooldown remaining;
- phase: `probation`, `trusted`, or `quarantined`.

Policy behavior:

1. A previously unseen probation action shadows the existing v32 writeback.
   After one weak or harmful objective credit, later probation attempts use a
   bounded blend strength of 0.20.
2. Two consecutive material positive credits promote the key to trusted.
3. Two consecutive weak/harmful or unstable credits quarantine the key.
4. Quarantine returns the current values unchanged and starts a bounded
   cooldown; after cooldown the key returns to probation with its exposure
   retained.
5. A hard exposure cap forces the same protected fallback even if the key is
   trusted. This prevents one relation from consuming the whole intervention
   budget.

The default trust score is `ewma_gain - 0.5 * ewma_std`. It is audit data and
does not directly reallocate FE in v33. A later resource policy may consume
it only after a separate ablation.

## Robust writeback

The existing action first produces a proposal from the relation evidence.
v33 then applies two bounds:

1. convex damping toward the current CC candidate, using the phase-dependent
   trust strength; and
2. Euclidean clipping of the writeback step to the existing relation guard
   threshold.

Quarantine and exposure-cap decisions return the current candidate exactly.
The operation is deterministic, finite-value checked, and does not evaluate
the objective. Dense-overlap v31 protected fallback writebacks remain outside
the v33 trust guard; non-dense fallback writebacks use the explicitly bounded
topology route specified below.

## Audit surface

Existing action trace rows gain v33 fields for:

- stable trust key;
- trust phase and guard reason;
- trust score;
- action exposure and cooldown;
- downstream credit and instability flag;
- objective values immediately before and after the shared-variable writeback;
- the v33-only topology fallback route: `dense_preserve_v31` or
  `non_dense_bounded_0_5`.

The v33 fields are emitted only in the v33 action trace; the legacy v1-v32
trace schema remains unchanged. `action_decision.csv` remains the original
relation-policy decision log; v33 execution state is recorded only in the
action trace, so the original policy audit is not rewritten.

## Verification gates

Code gates:

1. Pure trust policy tests fail before implementation and pass after it.
2. Probation, promotion, quarantine, cooldown, and exposure cap are
   deterministic and bounded.
3. Robust writeback never exceeds its configured norm and rejects non-finite
   proposals.
4. v32 helper behavior and v32 CLI parsing remain unchanged.
5. v33 is explicit in the runner and experiment profile; no v32 route silently
   changes.

Runtime gates:

1. A real-HCC 5k smoke shows trust transitions and clean trace fields.
2. The first 3M pilot uses E2/E4/E6/S6/R1/R2/A4/A5, seeds 1/2/3.
3. It must retain at least the v32 12/24 best-of-three baseline when later
   compared on the protected set, and must retain E1/E3/S2/S3 controls when
   the full protected run is expanded.
4. No FE overrun, AOB-input change, anti-leakage failure, or catastrophic
   protected-case loss is admissible.

## First 3M Pilot And Credit Correction

The first v33 pilot completed all 24 trajectories for E2/E4/E6/S6/R1/R2/A4/A5
at seeds 1/2/3 with zero same-budget violations, unchanged AOB inputs, and no
anti-leakage failure. It retained only 5/8 protected best-of-three wins,
losing E2, A4, and S6. It improved the E4/E6 worst-to-best ratios by about
41%/57%, but the preservation gate failed and the route was not adopted.

Trace analysis found a semantic defect in the first credit proxy. It compared
the next group's fitness delta with the source group's delta, even though the
two groups are different subproblems. E2/E4/E6/S6 therefore recorded mostly
non-positive credits and almost never promoted an action to trusted. The
uniform 0.20 first-action damping also changed useful early v32 behavior before
any risk evidence existed.

The first correction hypothesis stored the global protected objective before
writeback. The next already-evaluated CC group supplied its post-optimization
objective value, and credit was the bounded relative minimization improvement:

```text
credit = clip((protected_before - downstream_after) /
              max(abs(protected_before), abs(downstream_after)), -1, 1)
```

No FE was added. The first action shadowed v32; one weak/harmful credit enabled
0.20 damping, and a second consecutive weak/harmful credit quarantined it.
The 7/8 result below showed that this hypothesis improved preservation but did
not yet isolate the action outcome correctly.

## Objective-Paired Credit Correction

The second 8-case pilot completed all 24 fresh 3M-FE trajectories with zero
same-budget violations, 237/237 unchanged AOB inputs, and all anti-leakage
checks passing. It recovered E2 and S6 and retained 7/8 protected
best-of-three wins. A4 remained a strict loss by only 4.01 objective units
(`78304.01` versus paper-best `78300`), so the route still failed the
preservation gate.

The remaining defect was another temporal mismatch. The implementation used
the global guarded incumbent as the baseline and credited the action only
after the next group optimizer had completed. That signal mixed three things:
the current CC context's gap from the global incumbent, the shared-variable
writeback, and the next group's optimizer gain. It therefore could damp a
useful v32 action even when the writeback itself was harmless.

The corrected credit uses two objective values for the same CC candidate:

```text
pre_writeback = current_group_original_fitness - current_group_delta
apply shared-variable writeback
post_writeback = next_group_original_fitness
credit = clip((pre_writeback - post_writeback) /
              max(abs(pre_writeback), abs(post_writeback)), -1, 1)
```

`next_group_original_fitness` is the objective evaluation that already occurs
at the start of the next CC group, before that optimizer runs. The correction
therefore isolates the writeback effect without adding FE. Both objective
values are recorded as `trust_pre_writeback_fitness` and
`trust_post_writeback_fitness`, so every trust credit can be recomputed from
the trace. A proposal with writeback norm at or below `1e-12` is a no-op: it
remains visible as a relation decision but does not create trust state,
consume exposure, or receive credit. The case label, function family, paper
value, historical result, and final outcome remain unavailable to runtime
dispatch.

## Topology-Scoped Protected Fallback

The fallback preservation pilot completed the protected 8-case set at 3M FE
with 24 fresh trajectories. It restored S6 (`12026.00` at seed 2) but lost
the earlier R2 win: the best-of-three result was `7/8`, not the required
`8/8`. The paired comparison isolates the cause:

- v33.6 clipped every protected fallback to a `0.5` norm and retained R2;
- v33.7 preserved every v31 fallback and restored S6;
- R2 exposes a non-dense runtime topology (`adaptive_v26`, one shared
  variable, support ratio `0.02`), while S6 exposes dense relation-first
  topology (`adaptive_v24`, ten shared variables, support ratio `0.20`).

The next route therefore scopes the fallback behavior to runtime topology:

```text
active coordinate/repair/isolate -> v33 trust guard
dense overlap protected fallback  -> preserve v31 writeback exactly
non-dense protected fallback      -> retain v33.6 bounded fallback (norm 0.5)
```

This is a runtime evidence rule, not a case-specific dispatch rule. It uses
only the already computed overlap degree and relation topology; it cannot
read a case name, function family, paper value, historical outcome, or final
fitness. Existing v31 non-dense repair-lock behavior remains authoritative.

### Acceptance And Audit Gates

The implementation must add unit tests for both topology branches and must
keep v32 unchanged. A 5k real-HCC smoke must show the dense S6 branch keeps
the original fallback norm while the non-dense R2 branch clips the proposal.
The fresh 3M protocol then reruns E2/E4/E6/S6/R1/R2/A4/A5 with seeds 1/2/3.
Acceptance requires:

1. `8/8` protected best-of-three wins against the frozen paper-best values;
2. `24/24` fresh optimizer executions and zero same-budget violations;
3. unchanged AOB input hashes and a passing anti-leakage audit;
4. no v32 regression, action portfolio, or additional FE consumption;
5. trace evidence identifies the topology branch and its writeback norm.

## Topology-Scoped 3M Result

The topology-scoped route passed the protected gate on 2026-07-13. All 24
case/seed trajectories were fresh, same-budget violations were `0/24`, AOB
inputs were `237/237` unchanged, and all `16/16` anti-leakage checks passed.
The best-of-three comparison against the frozen paper-best table was:

| Case | v33 best | Paper best | Relative gain |
|---|---:|---:|---:|
| E2 | 3,885,968.0 | 6,870,000.0 | 43.44% |
| E4 | 12,913,830.0 | 19,000,000.0 | 32.03% |
| E6 | 21,431,810.0 | 26,200,000.0 | 18.20% |
| S6 | 12,026.0 | 13,300.0 | 9.58% |
| R1 | 169,214.2 | 174,000.0 | 2.75% |
| R2 | 227,665.2 | 248,000.0 | 8.20% |
| A4 | 78,299.33 | 78,300.0 | <0.01% |
| A5 | 78,149.10 | 78,200.0 | 0.07% |

The runtime trace confirms the intended split. R2 recorded 685 non-dense
fallback rows with maximum norm `0.5`; S6 recorded 300 dense-run fallback rows,
including 111 above `0.5` with a maximum norm of `84.06751`. This is an 8-case
three-seed pilot result, not a 25-run statistical performance claim.

## Full 24-Case 3M Replay Result

The S02 replay expanded the same reference-blind v33 selector to all 24 AOB
cases without changing its runtime inputs or action mapping. All `72/72`
case/seed trajectories were fresh, same-budget violations were `0/72`, AOB
inputs were `708/708` unchanged, and all `16/16` anti-leakage checks passed.
Every run used the same tracked code at commit `8cf16ae` and the same strict
3M-FE protocol; paper-best values were joined only after runtime completion.

The offline comparison reached the requested best-of-three pilot threshold:

- best-of-three wins: `13/24` (`E1`, `E2`, `E3`, `E4`, `E6`, `S2`, `S3`,
  `S5`, `S6`, `R1`, `R2`, `A4`, `A5`);
- three-seed mean wins: `4/24` (`E1`, `E3`, `E4`, `S2`);
- worst-seed wins: `2/24` (`E1`, `E3`);
- per-seed wins: `21/72`;
- catastrophic seeds: `31/72`, using the existing relative-gain threshold
  `<= -20%` against the offline paper-best value.

The runtime trace explains why the best-of-three count is not a robust success
claim. Of 1,238 objective-paired credit rows, 1,080 were negative and only 158
were positive. Across 1,478 trust-state rows, only 7 reached `trusted`, while
241 were quarantined and 469 were probation-limited. The protected fallback
remained topology-scoped: 1,352 non-dense rows were bounded at maximum norm
`0.5`, while 327 dense rows preserved v31 behavior with maximum norm
`457.486`. No trust or writeback instability flags were raised. The no-overlap
controls `A1`, `E1`, `R1`, and `S1` correctly had no trust rows, so their mixed
performance reflects the underlying optimizer path rather than a hidden
case-specific action branch.

These trace statistics are runtime credit proxies, not causal effect estimates.
The result supports claim level 6 (full configured evaluation completed) and
meets the registered `13/24` best-of-three pilot target, but the mean, worst
seed, and catastrophic-loss gates block a robust final-success or SOTA claim.
The auditable derived reports are:

- `results/controller_v338_full24_seed123_3m_20260713/offline_paper_best_comparison.csv`;
- `results/controller_v338_full24_seed123_3m_20260713/offline_paper_best_comparison.md`;
- `results/controller_v338_full24_seed123_3m_20260713/runtime_evidence_case_summary.csv`.
