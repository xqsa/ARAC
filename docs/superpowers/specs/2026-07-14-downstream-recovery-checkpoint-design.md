# Downstream Recovery Checkpoint Design

Date: 2026-07-14
Executor: Codex
Status: protected candidate failed; not adopted

## Purpose

Improve the cross-seed stability of the v33.8 evidence-to-action controller
without using case identity, function family, paper values, historical results,
or final outcomes at runtime.

The new opt-in controller route keeps the existing relation decision, trust
guard, topology-scoped fallback, FE allocation, and optimizer behavior. It adds
one trajectory safety contract: a relation writeback may influence the next CC
group, but it is committed to the continuing candidate only if that downstream
group recovers past the candidate fitness known immediately before writeback.

The design targets mean and worst-seed stability. It does not claim that an
immediate writeback credit is a causal estimate of final optimizer utility.

## Evidence And Root Cause

The fresh v33.8 full-24 run completed 72/72 trajectories and reached 13/24
best-of-three wins, but only 4/24 three-seed mean wins and 2/24 worst-seed wins,
with 31/72 catastrophic seeds.

The runtime trace shows that the current per-key trust mechanism cannot protect
high-cardinality relation streams early enough:

- 747 exact trust keys were observed across the 60 overlap-applicable runs;
- 330/747 keys appeared only once;
- 685/747 first credits were negative;
- 762 `probation_shadow` writebacks contributed total norm 489.3, while all
  later limited/quarantined/trusted decisions contributed total norm 50.6;
- only 7 rows reached `trusted`.

The immediate credit also does not separate final winners from failures. Nearly
90% of credit rows in winning overlap runs were negative. Therefore a policy
that globally damps or rejects every new action after negative credit would
overfit a weak proxy and repeat the already observed failure of uniform early
damping.

The instability is not exclusively a relation-writeback problem. Four
catastrophic seeds occurred in no-overlap controls, and 18 catastrophic seeds
came from six cases that failed in all three seeds. The checkpoint is therefore
a no-harm trajectory intervention, not a promise to repair every systematic
optimizer gap.

## Alternatives Considered

### 1. Run- or family-level trust inheritance

New exact keys would inherit risk learned from earlier keys in the same run or
action family. This directly addresses key churn, but the observed immediate
credit has weak alignment with final utility. It could suppress useful actions
after one noisy negative observation. This option is not selected for the first
candidate.

### 2. Immediate writeback rollback

The controller could restore the pre-writeback candidate as soon as the next
scheduled objective evaluation reports negative immediate credit. This is
simple and same-budget, but almost all successful runs also contain negative
immediate credits. It would remove the downstream search opportunity that the
action is intended to create. This option is rejected.

### 3. Downstream recovery checkpoint

The selected design treats one relation writeback plus the next CC group as a
bounded trajectory transaction. The next group receives the acted candidate
and its existing optimizer budget. The transaction commits only if the best
candidate after that group is strictly better than the known pre-writeback
checkpoint. Otherwise the controller restores the checkpoint before producing
the next relation decision. This preserves exploration while preventing an
unrecovered writeback loss from propagating through later groups.

### 4. Enable resumable Phase-I search-state allocation in v33

The current `phase_i_mmes` v33 route does not capture or resume Phase-I state;
its active search-start hook is the existing phase-rescue multistart path. A
state-resume resource policy is a larger, independent intervention. Historical
v31 evidence is materially weaker than the v32/v33 target route, so it must not
be bundled into this stability fix. It remains a later ablation if the
checkpoint cannot meet the registered gates.

## Runtime Boundary

The checkpoint may use only values already available in the current optimizer
run:

- the candidate vector immediately before relation writeback;
- its already-known objective fitness;
- the objective evaluation already performed at the start of the next group;
- the best candidate and fitness already produced by that next group;
- current relation identity, action identity, topology, and FE state for audit.

It must not read or derive runtime decisions from:

- case or problem identifiers;
- function-family labels;
- paper-best or reported baseline values;
- prior-run or historical outcomes;
- final error, relative gain, win labels, or catastrophic labels.

Problem identifiers remain execution and offline-audit keys only.

## Version And Compatibility

The new route is explicit and opt-in as
`arac_evidence_action_controller_v34`. The v33 route and all v1-v33 trace
schemas and behavior remain unchanged.

The experiment runner receives an explicit v34 lane profile. No existing
profile silently changes. The action registry identifies v34 as the same
trajectory/core-intervention class as v32 and v33.

## Checkpoint State

Each run owns at most one pending checkpoint because relation writebacks are
consumed sequentially by the CC loop. A checkpoint stores:

- a copy of the candidate immediately before writeback;
- the known pre-writeback fitness;
- the writeback norm and fallback route;
- the action trace row that initiated the checkpoint;
- the post-writeback fitness once the next group evaluates the acted candidate.

The checkpoint is scoped to one run. It is never persisted across seeds or
reused by another problem execution.

## Runtime Sequence

```text
current CC group finishes
  -> save candidate and known fitness
  -> execute existing v33 relation writeback
  -> if writeback is a no-op, do not open a checkpoint
  -> next CC group evaluates the acted candidate using its scheduled evaluation
  -> record immediate post-writeback fitness and existing trust credit
  -> run the next group optimizer and existing phase-rescue logic unchanged
  -> compare downstream best with the pre-writeback checkpoint
     -> strictly better: commit downstream candidate
     -> not strictly better: restore checkpoint candidate
  -> derive the current group delta from the committed trajectory
  -> only then build the next relation decision
```

No objective evaluation is added. The next-group evaluation and optimizer FE
are already present in v33.8 and remain charged to the same strict 3M ledger.

## Commit And Restore Semantics

A transaction commits when:

```text
downstream_fitness < checkpoint_fitness
```

On commit:

- the downstream candidate remains active;
- recovery credit remains measured against the checkpoint fitness;
- the next group's original candidate, original fitness, and local optimizer
  delta remain unchanged;
- subsequent relation evidence therefore keeps the same local CC meaning as
  v33.8 instead of consuming a checkpoint-to-downstream composite delta.

On restore:

- the candidate is restored exactly to the checkpoint vector;
- the effective group delta becomes zero;
- subsequent relation evidence cannot consume a discarded optimizer delta;
- the objective record remains untouched, so all evaluated evidence and FE stay
  auditable.

If a scheduled search-state action, outer-loop termination, or budget boundary
preempts an unresolved checkpoint, the controller restores it before the
intervening trajectory action or run exit. An unevaluated writeback is never
silently committed.

## Trust Interaction

The existing objective-paired trust credit remains unchanged and continues to
describe the immediate writeback effect. The recovery checkpoint does not
rewrite that credit into a causal claim.

The checkpoint adds a separate downstream recovery signal. Trust damping and
quarantine still control future writeback exposure exactly as in v33.8; the
checkpoint controls whether the current acted trajectory may propagate. This
keeps the two meanings explicit:

- `trust_credit`: immediate writeback proxy;
- `trajectory_guard_recovery_credit`: writeback-plus-next-group recovery.

## Audit Fields

Only the v34 action trace adds:

- `trajectory_guard_status`:
  `pending`, `committed`, `restored`, or `preempted_restored`;
- `trajectory_guard_pre_fitness`;
- `trajectory_guard_post_writeback_fitness`;
- `trajectory_guard_downstream_fitness`;
- `trajectory_guard_recovery_credit`;
- `trajectory_guard_restored` as `0` or `1`.

The recovery credit is the existing bounded minimization credit formula applied
to checkpoint and downstream fitness. Audit summaries must report commit,
restore, and preempted-restore counts per case and seed. Offline paper joins are
performed only after all runtime artifacts are complete.

## Error Handling

- Candidate dimensions must match the checkpoint vector.
- Candidate and fitness values must be finite.
- Opening a second checkpoint while one is unresolved is an explicit runtime
  error.
- Finalizing without a post-writeback observation is an explicit runtime error.
- Non-finite downstream values restore the checkpoint and fail the run visibly;
  they are not converted into a silent success path.

## Test Strategy

Pure unit tests must prove:

1. strict downstream improvement commits;
2. equality or degradation restores the exact checkpoint candidate;
3. commit preserves the next group's local CC evidence while restore returns
   the checkpoint candidate and zeroes the discarded group delta;
4. no-op writebacks do not open checkpoints;
5. preemption restores unresolved state;
6. non-finite and shape-mismatched values fail explicitly;
7. v33 behavior and trace schema remain unchanged;
8. v34 CLI/profile/action registration is explicit;
9. forbidden runtime fields are absent from the checkpoint state and policy.

Integration tests must prove that checkpoint finalization occurs before the
next relation decision and does not add FE.

## First Protected Result And Commit-Evidence Correction

The first v34 protected protocol completed on 2026-07-14 at commit `13569ac`.
All 24 case/seed trajectories were fresh, FE violations were `0/24`, AOB rows
were `237/237` unchanged, and anti-leakage checks were `16/16` pass. The
candidate failed the preservation gate:

- best-of-three wins: `5/8` (`E4`, `E6`, `R1`, `R2`, `A5`);
- three-seed mean wins: `1/8` (`E4`);
- worst-seed wins: `1/8` (`E4`);
- seed wins: `8/24`;
- catastrophic seeds: `8/24` at relative gain `<= -20%`;
- failed best-of-three cases: `E2`, `S6`, and `A4`.

The failed candidate is preserved under
`results/controller_v34_recovery_8case_seed123_3m_20260714/`. Paper values
were joined only after all runtime artifacts completed.

Runtime evidence identified a semantic confound in the CC integration. The
pure checkpoint decision was correct, but the runner also rewrote
`original_best`, `original_fitness`, and `current_delta` after every commit.
That replaced the next group's local optimizer delta with a composite
checkpoint-to-downstream delta. Of the committed recovery rows, most had an
immediately harmful writeback followed by downstream recovery, including
`94/111` for E2 and `74/81` for S6. The composite delta therefore
systematically changed the scale and meaning of subsequent relation evidence.

The effect was observable without paper or case labels. Relative to the v33.8
trace, all overlap-bearing runs changed relation decisions, while the
no-recovery R1 control matched all `705/705` decisions and reproduced the same
three final errors exactly. The correction is therefore not a case-specific
exception: a committed checkpoint must be transparent to the existing local
CC evidence. Only a restore may replace the candidate, restore the checkpoint
baseline, and zero the discarded delta. Recovery credit remains separately
auditable against the checkpoint.

## Corrected Protected Result And Disposition

The local-evidence correction at commit `52a3b0d` passed the focused contract
tests and the real-HCC 5k smoke before a second protected run was started. The
fresh protected artifacts are preserved under
`results/controller_v34_recovery_local_evidence_8case_seed123_3m_20260714/`.
The runtime protocol was complete and clean:

- `24/24` case/seed runs completed as fresh optimizer executions;
- FE violations and overspends were `0/24`, with maximum charged FE exactly
  `3,000,000`;
- AOB input hashes were `237/237` unchanged;
- anti-leakage checks were `16/16` pass with no forbidden field found;
- raw evaluation, budget, action-trace, action-decision, and AOB artifacts were
  present for all `24/24` runs;
- the trajectory audit covered `24/24` runs with zero pending checkpoints,
  `467` commits, `226` restores, and `1` preempted restore.

The static `backend_semantics_diff.csv` surface changed in `22/24` runs. The
two zero rows, E6 seed 1 and S6 seed 3, still resolved respectively `25` and
`35` trajectory checkpoints. This is an audit-schema boundary: the static
table records ownership, relation handling, coordination, allocation, update
order, and acceptance-rule fields, while v34 recovery is recorded in
`trajectory_guard_summary.csv` and the action trace.

The frozen paper-best table was joined only after runtime completion. The
corrected candidate still failed the mandatory protected gate:

- best-of-three wins: `6/8` (`E2`, `E4`, `E6`, `R1`, `A4`, and `A5`);
- three-seed mean wins: `1/8` (`E4`);
- worst-seed wins: `0/8`;
- seed wins: `9/24`;
- catastrophic seeds: `5/24` at relative gain `<= -20%`;
- failed best-of-three cases: `S6` and `R2`.

The correction improved the first v34 result from `5/8` to `6/8` and reduced
catastrophic seeds from `8/24` to `5/24`, but it did not preserve the v33.8
protected result of `8/8`. Full-24 execution was therefore not authorized by
the registered ladder and was not started.

The runtime traces explain why threshold tuning is not justified. The R1
no-recovery control reproduced all `705/705` relation decisions and all three
final errors exactly. Across the overlap-bearing protected runs, recovery
changed `364` actions among `6,465` relation ids shared with v33.8, with another
`140` new-only and `65` old-only ids. On S6, the first restore and first action
divergence both occurred in outer iteration zero; relative to v33.8, seed 2
degraded by `200.33%` while seed 3 improved by `22.76%`. On R2, restored rows
had nearly identical tiny mean recovery credits for seed 1 (`-9.7e-5`) and
seed 3 (`-1.01e-4`), yet their final changes relative to v33.8 had opposite
signs: `-63.82%` and `+18.07%` respectively.

Thus the one-downstream-group credit is locally auditable but not identifiable
as long-horizon optimizer utility. A restore can prevent propagation of an
unrecovered local loss while also changing all later candidate and relation
evidence. The available runtime signal does not separate the harmful and
helpful long-horizon branches. Choosing a tolerance from S6/R2 final outcomes
would use offline labels indirectly and violate the reference-blind boundary.

v34 is consequently a failed candidate. The existing v33.8 full-24 result
remains the canonical fresh evidence: best-of-three `13/24`, mean `4/24`,
worst-seed `2/24`, seed wins `21/72`, and catastrophic seeds `31/72`. Those
stability limitations remain explicit and still block a robust performance or
SOTA claim.

## Experimental Gates

The user approved these adoption gates for the current three-seed pilot:

1. protected E2/E4/E6/S6/R1/R2/A4/A5 best-of-three remains 8/8;
2. full-24 best-of-three remains at least 13/24;
3. full-24 three-seed mean wins reach at least 6/24;
4. full-24 worst-seed wins reach at least 4/24;
5. catastrophic seeds fall to at most 27/72;
6. fresh execution is complete, FE violations are zero, AOB inputs are
   unchanged, anti-leakage passes, and no case-specific dispatch exists.

The execution ladder is:

1. focused unit and CLI tests;
2. real-HCC 5k smoke with commit and restore trace examples;
3. protected 8-case, seeds 1/2/3, strict 3M FE;
4. full 24-case, seeds 1/2/3, strict 3M FE only if the protected gate passes.

Failure to meet any performance gate is reported as a failed candidate, not
hidden by best-seed selection. The next candidate, if needed, requires a new
single-hypothesis design rather than stacking unrelated changes.

## Claim Boundary

Passing unit tests and smoke proves only that the runtime action is connected
and auditable. Passing the protected set permits full-24 evaluation. Only the
fresh full-24 metrics above can establish that this candidate improved the
requested stability. Three seeds remain pilot-level evidence and do not support
a 25-run, robust-final, or SOTA claim.
