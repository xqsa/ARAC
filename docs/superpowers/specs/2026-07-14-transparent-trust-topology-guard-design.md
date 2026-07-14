# Transparent-Trust Topology Guard Design

Date: 2026-07-14
Executor: Codex
Status: implemented and rejected at the protected gate

## Purpose

Improve the cross-seed stability of the v33.8 grouping-to-action controller
without using case identity, function family, paper values, historical
outcomes, or final results at runtime.

The opt-in v35 candidate removes the exact-key trust damping and quarantine
from active relation actions while retaining the v33.8 topology-scoped fallback
guard. It tests one hypothesis: the objective-paired trust credit is too weakly
identified to suppress coordinate, repair, and isolate actions, while the
dense/non-dense topology split remains useful for bounding fallback writeback.

This is an engineering stability ablation. It is not yet a publication-level
novelty claim.

## Evidence

The fresh v33.8 full-24 run reached best-of-three `13/24`, but mean only
`4/24`, worst-seed only `2/24`, and catastrophic seeds `31/72`.

The paired v32/v33.8 evidence isolates a stability regression on the common
13-case target:

- v32: best `12/13`, mean `4/13`, worst `4/13`, seed wins `21/39`;
- v33.8: best `13/13`, mean `4/13`, worst `2/13`, seed wins `21/39`;
- S2 and S3 were `3/3` seed wins under v32, but fell to `2/3` and `1/3`
  under v33.8.

The trust proxy does not justify that loss. In the full v33.8 trace, `1,080`
of `1,238` objective-paired credits were negative, including almost 90% of
credits in winning overlap runs. Only `7` rows reached `trusted`; `241` were
quarantined and `469` were probation-limited. A seed-wise identifiability audit
found no early topology, norm, exposure, or credit feature with enough
within-case consistency to dispatch a bad-seed classifier.

The source HCC loop in `E:\HCC-main\2025_HCC_GECCO-main\HCC_SRC\HCC-ES.py`
updates each subproblem and then performs the native overlap blend. The v32
active relation path preserves that source meaning with the existing v31
guards. v35 restores that active path instead of adding another optimizer.

## Alternatives

### Elite archive

Rejected. Across the 72 v33.8 runs, only one trace contained an earlier known
fitness below the final value, and the maximum difference was `1.28e-5%`.
HCC already retains the effective global incumbent.

### Delayed trust maturity

Rejected. Requiring an arbitrary number of negative credits before damping
would tune a threshold on a proxy already shown not to identify long-horizon
utility.

### Within-budget multi-start

Rejected for this candidate. Historical phase-rescue runs on E5/R3/R4 were
mixed or harmful and consumed CC opportunity. It changes optimizer execution
and resource allocation together, preventing a clean attribution.

### Transparent active actions plus topology fallback guard

Selected. It removes the unsupported trust intervention, preserves the one
v33.8 component with a concrete topology semantics, adds no FE, and can be
falsified on S2/S3 plus the original protected cases.

## Runtime Boundary

v35 may use only current-run evidence already available to v31/v33.8:

- relation topology and shared-variable indices;
- overlap degree and the existing dense/non-dense run state;
- current and previous group values and contribution deltas;
- the existing v31 relation decision and value guard.

It must not use or derive decisions from:

- case or problem identifiers;
- function-family labels;
- paper-best or reported baselines;
- prior-run or historical outcomes;
- final error, relative gain, seed-win, catastrophic, mean, or worst labels.

Problem identifiers remain execution and offline-audit keys only.

## Version And Compatibility

The route is explicit and opt-in as `arac_evidence_action_controller_v35`.
No existing v1-v34 selector changes behavior. v35 uses the same v33 trace
surface so the topology fallback route remains auditable; trust fields are
present but empty because no trust state is created. v34 recovery fields are
not present.

## Runtime Semantics

For every adjacent overlap relation:

1. Run the existing v31 relation decision, repair lock, and value guard.
2. If the canonical action is coordinate, repair, or isolate, commit the v31
   adjusted values exactly. Do not create, consult, or credit an exact-key
   trust state.
3. If the action is fallback and the run topology is dense, preserve the v31
   fallback writeback exactly and trace `dense_preserve_v31`.
4. If the action is fallback and the topology is non-dense, clip the writeback
   step to norm `0.5` and trace `non_dense_bounded_0_5`.
5. Continue the same CC trajectory and FE ledger. No objective evaluation,
   optimizer restart, search-state block, or paper comparison is added.

The dense/non-dense rule is inherited unchanged from v33.8. The only behavioral
delta from v33.8 is that active relation actions no longer pass through the
exact-key trust policy.

## Components

- `scripts/hcc_smoke_runner.py`: add the explicit v35 action name, route
  classification, and a topology-only relation executor shared with v33 where
  practical.
- `src/arac/actions/contracts.py`: register v35 as the existing trajectory/core
  intervention family.
- `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`: expose one
  v35 lane and reuse the v33 trace schema without recovery fields.
- Focused tests: prove active transparency, both fallback branches, version
  isolation, CLI/profile wiring, no extra FE, and forbidden-field exclusion.

## Failure Policy

- Missing or non-finite relation values fail through the existing explicit
  guards; v35 adds no fallback default.
- A no-op writeback remains a no-op and does not create synthetic trust data.
- If topology state is unavailable, the route fails the focused contract test;
  it must not infer topology from case identity.
- Any v33/v34 behavior or trace-schema regression rejects the implementation.

## Verification Ladder

### Code and CLI

1. Pure tests show v35 active coordinate/repair/isolate output equals v31/v32.
2. Dense fallback is bit-equivalent to v31; non-dense fallback is bounded at
   norm `0.5`.
3. v33 trust and v34 recovery tests remain unchanged and pass.
4. v35 CLI and exp003 profile expose exactly one runtime lane.
5. Focused runner tests prove no extra objective evaluation or FE.

### Real-HCC smoke

Run S3, R2, and S6 at seeds 1/2/3 with strict 5k FE. Require 9/9 fresh
completion, zero FE violations, unchanged AOB hashes, anti-leakage pass,
active v35 rows with empty trust fields, and both topology fallback routes.

### Protected 10-case gate

Run E2/E4/E6/S6/R1/R2/A4/A5/S2/S3 at seeds 1/2/3 and strict 3M FE.
Paper-best is joined only after all runtime artifacts complete. Continue to
full-24 only if:

1. the original protected eight remain strict best-of-three `8/8`;
2. S2 and S3 are strict seed wins in all `6/6` runs;
3. aggregate best is `10/10`, mean at least `2/10`, worst at least `2/10`,
   seed wins at least `15/30`, and catastrophic seeds at most `8/30`;
4. all 30 runs are fresh, FE violations are zero, AOB inputs are unchanged,
   anti-leakage passes, and no case-specific dispatch exists.

### Full-24 stability gate

Only after the protected gate, run all 24 cases at seeds 1/2/3 and strict 3M
FE. Adoption requires:

- best-of-three at least `13/24`;
- mean wins at least `6/24` and strictly above the v33.8 `4/24`;
- worst-seed wins at least `4/24` and strictly above the v33.8 `2/24`;
- catastrophic seeds at most `27/72` and below v33.8 `31/72`;
- `72/72` fresh runs, zero FE overspend, unchanged AOB inputs, anti-leakage
  pass, and zero forbidden runtime-dispatch fields.

Three seeds remain pilot-level evidence. Passing these gates does not justify
a 25-run, robust-final, or SOTA claim.

## Execution Result

The opt-in v35 implementation was completed in four commits:

- `8567a17`: pure transparent active-action and topology fallback semantics;
- `e422719`: runner, trace-schema, and action-contract registration;
- `5b582c0`: single exp003 v35 lane;
- `933368d`: matched-FE integration coverage.

Focused verification passed with `268 passed, 1 skipped`; all Git-tracked tests
passed with `602 passed, 1 skipped`. The real-HCC 5k smoke at
`results/controller_v35_transparent_trust_5k_20260714` completed `9/9` fresh
runs with zero FE violations or overspends, AOB inputs `90/90` unchanged,
anti-leakage `16/16`, no non-empty trust or recovery values, and both fallback
routes (`dense_preserve_v31`: 45 rows; `non_dense_bounded_0_5`: 99 rows).

The protected 10-case run at
`results/controller_v35_transparent_trust_10case_seed123_3m_20260714` completed
all `30/30` fresh trajectories. FE violations and overspends were zero, AOB
inputs were `297/297` unchanged, anti-leakage was `16/16`, and every run had
its raw action trace, decision, mismatch audit, overlap relations, budget
summary, AOB manifest, and evaluation record. Paper-best was joined only after
runtime completion, with `runtime_dispatch_used=0`.

The protected performance gate failed:

- original protected eight: best-of-three `6/8`, not the required `8/8`;
- S2/S3: seed wins `6/6`, meeting the restoration target;
- aggregate: best `8/10`, mean `3/10`, worst `2/10`, seed wins `15/30`,
  catastrophic seeds `7/30`;
- failed protected cases: A4 and R2.

Therefore the conditional v35 full-24 run was not started and v35 is not
adopted.

## Runtime-Evidence Diagnosis

Removing trust damping produced the intended S2/S3 recovery but increased
active-action exposure elsewhere. At the first matched v33.8/v35 divergence,
the same coordinate action and relation received a v35 writeback norm exactly
five times the v33.8 probation-limited norm for every A4 and R2 seed. This is
the direct semantic effect of replacing v33.8's `0.2` probation blend with the
transparent v31 writeback.

The accumulated exposure was substantial in A4. Seed 2 coordinate rows rose
from 19 to 80 and active norm sum from `6.11` to `36.93`; seed 3 rose from 14
to 66 and from `6.26` to `30.64`. Yet final effects remained small and mixed:
v35 changed A4 versus v33.8 by `-0.0041%`, `+0.0021%`, and `-0.0078%` across
seeds 1/2/3, missing paper-best by only `2.51` in its best seed.

R2 shows the stronger counterexample. The same transparent policy changed
v35 versus v33.8 by `-15.78%`, `+4.36%`, and `-0.022%` across seeds 1/2/3.
Its best error was `263580.7`, so it lost all three seeds to paper-best
`248000`; v33.8 seed 1 had previously reached `227665.2`. Early local trust
credits around the first divergence do not separate these opposite long-term
directions.

The evidence rejects unconditional active transparency as a stability fix.
It does not validate the v33.8 exact-key trust proxy either: both the earlier
trust audit and this ablation show local one-step evidence with inconsistent
long-horizon utility. A future candidate needs an independently identifiable,
reference-blind long-horizon signal; it must not tune on A4/R2 labels or final
outcomes.
