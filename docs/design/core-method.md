# Core Method: ARAC-CAR Within-Run Causal Action Racing

Date: 2026-07-15
Status: implementation freeze for CAR-W; not a performance claim
Executor: Codex

## One-Line Definition

`ARAC-CAR` (Counterfactual Action-Risk Controller) treats overlap-derived
actions as hypotheses. It generates a candidate from current-run graph evidence,
probes the candidate and the canonical fallback from the same optimizer
checkpoint with equal function evaluations and coupled random streams, and
commits only when a conservative lower-tail gate passes. Otherwise it abstains.

The method is therefore:

```text
dynamic overlap evidence
  -> candidate action proposal
  -> same-checkpoint paired probe
  -> normalized short-horizon contrast
  -> conservative runtime safety score + lower-tail risk gate
  -> atomic commit or fallback/abstain
  -> frozen Phase-II action plan
```

The contribution is the **within-run causal calibration protocol**, not a new
low-level optimizer, a new decomposition heuristic, or a case-specific action
table.

## Why v36 Is Replaced

The held-out artifact
`results/controller_paired_v33_v36_13case_seed45678_3m_20260714_retry` contains
13 cases x 5 seeds x 4 lanes = 260 fresh trajectories. Integrity passed, but
the utility gate was blocked: 59/65 candidate/fallback pairs were unchanged,
only six pairs changed, two were meaningful wins, and one was a catastrophic
loss. The v36 first-sweep binary latch consequently has both low coverage and
poor long-horizon risk identification. Adding another scalar threshold would
reuse the same unidentifiable signal.

The literature audit in `docs/literature_review.md` also shows that the broad
“overlap evidence -> runtime action” claim is already adjacent to OCC, dynamic
CC, UCB-CC, contribution-based resource allocation, learning-based CC,
probing trajectories, and CRN racing. CAR therefore narrows the claim to one
combination: an overlap/shared-variable backend intervention is contrasted
with the native fallback from an identical full checkpoint, both arms consume
one same-budget CC ledger, and a fallback-relative downside gate can abstain.

## Runtime Boundary

The policy input is a capability-limited `DispatchEvidence` value. CAR-W passes
only current-run overlap evidence and the pre-registered writeback contract:

- immutable overlap graph/component fingerprints;
- candidate action name/family and shared-variable geometry;
- overlap strength, shared-variable count, complete-sweep count and coverage;
- bounded writeback norm.

Optimizer state fingerprints, remaining FE, and probe exposure are checked by
the executor and ledger, not supplied as identity-bearing dispatch features.

Run, case, seed, function family, paper-best, historical outcome, final error,
relative gain, and win/catastrophic labels live only in a separate
`AuditEnvelope`. The runtime policy type cannot receive that envelope. The
final evaluator is outside the dispatch call graph. Static AST checks and a
runtime forbidden-field check are both required.

## Phase Boundary

**Phase I** consists of the canonical global stage, at least two complete
overlap-component evidence sweeps, and the CAR calibration probes. A probe
endpoint is named `phase1_probe_fitness_*` and is not a final outcome.

After the candidate plan is frozen, **Phase II** runs the remaining canonical
optimizer trajectory. No Phase-II final result, paper table, or offline
comparison can update the plan. A topology/graph fingerprint change invalidates
the plan and forces fallback.

## Candidate Generation

Candidate generation is deliberately conservative and reuses existing action
contracts rather than inventing another threshold family.

### Writeback channel (first and mandatory experiment)

Use the existing v31 relation proposal as `a`, then apply the fixed bounded
dose (`alpha = 0.20`) and the existing v33 norm guard. The probe horizon remains
the complete overlap connected component. Inside that component, only edges
whose latest two complete sweeps agree on the same non-fallback action name and
family enter the **stable support subgraph**. Candidate writeback touches only
shared variables on that support; every unsupported or unstable edge behaves as
native fallback in both arms. Evidence must be finite and fully covered, and a
complete component horizon must fit in the reserved probe budget. If the active
support mixes multiple non-fallback families, the component abstains.

The fallback `a0` is the unchanged v33.8 behavior. The probe tests `a` versus
`a0`; it does not infer action utility from support or rank thresholds alone.
CAR-W is registered as a v33-derived runtime route: outside the leased probe
horizon every guarded controller, sigma, rescue and fallback decision is the
same as v33.8. If the overlap graph has no edges, CAR bypasses probe reservation
entirely, emits `no_overlap_component_candidate`, and must be bit-equivalent to
v33 for the same seed.

### Resource channel (only after writeback passes)

Construct a zero-sum group budget proposal from current centrality, conflict,
stagnation and confidence, then project it onto:

```text
sum(group_budget) = canonical_sum
|candidate - canonical| <= 0.10 * canonical per group
group_budget >= population_minimum
```

The resource channel is a separate ablation. Its probe inherits a committed
writeback plan and changes no other channel.

### Search-start channel (last)

Use only an already captured, valid resumable state. A search-start candidate
may change the existing mean/sigma/continuation state, but not the optimizer
family, population, or FE count. If the state fingerprint is unavailable or
invalid, abstain. This channel is not a new restart optimizer.

The channels are tested in the fixed order `W -> R -> S`; they are never stacked
in a single first experiment. Each later channel inherits only the frozen plan
from the prior channel.

## Paired Probe and Risk Gate

Probes start only at a complete-component barrier. Let `S_k` be the main
checkpoint, `F_k` the canonical fallback branch, and `C_k` the candidate branch.
Both branches receive the same component horizon `H`, population-rounded FE,
and a counter-based coupled random stream. The global ledger charges both
branches.

Use `K = 3` sequential pairs. For pair `k`, run both branches from a snapshot of
`S_k`. For `k < K`, advance the main trajectory with `F_k` while retaining the
candidate endpoint only as probe evidence. On the final pair, adopt `C_K` only
if the gate passes; otherwise adopt `F_K`. A successful candidate receives one
component-horizon lease only; all later Phase-II sweeps return to the canonical
v33.8 fallback controller. This gives persistence checks across checkpoints
without merging mutable branch state or turning a short probe into unbounded
action exposure.

For minimization, define the probe contrast:

```text
d_k = (f_F,k - f_C,k)
      / max(abs(f_start,k), abs(f_F,k), abs(f_C,k), epsilon)
```

Positive `d_k` means the candidate is better. Let `m` and `s` be the mean and
sample standard deviation of the three contrasts:

```text
Safety = m - kappa * s / sqrt(K)    (kappa = 1, fixed; trace field `lcb`)
Tail = mean of the lowest ceil(K/3) d_k values
```

Commit iff all of the following hold:

1. every pair has equal actual arm FE, finite values, valid state fingerprints,
   and no ledger overrun;
2. the graph/component fingerprint and candidate action family remain stable;
3. `Safety > epsilon` and `Tail >= 0`;
4. the final candidate endpoint is not worse than its own start;
5. the candidate has not exhausted its channel exposure/risk budget.

Otherwise the controller **abstains** and adopts the paired fallback state.
With `K = 3`, neither `Safety` nor `Tail` is a nominal confidence interval.
`Tail` is the worst replicate and `Safety` is only a conservative runtime
dispatch score. The retained `lcb` trace name is a schema label, not a
statistical confidence claim. Final inference uses case-clustered paired
summaries, not relation rows as independent samples.

The deployment index is fixed before probing (the final pair); the controller
must never choose the numerically best probe endpoint after seeing all probes.

## State and FE Isolation

The implementation must add a `CARProbeExecutor` and a single `BudgetLedger`
without modifying the canonical v33/v32 route. At a barrier it explicitly
clones candidate vectors, group caches, controller state, RNG descriptors and
the evaluator's branch-local record. It must not deep-copy a live AOB function
or share its mutable `fitness_record`/evaluation counter.

Each branch has its own best state and trace. The parent ledger is the only FE
authority:

```text
total_FE = canonical_FE + fallback_probe_FE + candidate_probe_FE
probe_FE <= 0.06 * max_FE
```

Suggested reserved caps are W=3%, R=2%, S=1% of the configured budget. If a
complete horizon cannot fit, the channel abstains; no silent shorter probe or
unbudgeted fallback is allowed. Discarded branch records cannot contaminate the
primary committed-state result. If an evaluated-elite sensitivity is reported,
it is a separate, pre-registered column.

## Required Trace

Add one CAR registry entry and these auditable artifacts:

- `car_probe_trace.csv`: pair id, channel, component fingerprint, arm FE,
  counter-based seed descriptor, `phase1_probe_fitness_before/after`, `d_k`,
  LCB/Tail, gate result and abstain reason;
- `car_state_ledger.csv`: checkpoint/state fingerprints, adopted branch,
  committed plan and exposure/risk budget;
- `car_branch_manifest.csv`: branch-local evaluator and record hashes;
- `car_dispatch_boundary_audit.csv`: a type-level inventory proving
  `DispatchEvidence` is disjoint from `AuditEnvelope` and forbidden
  identity/outcome fields, alongside the existing payload/runtime audit.

## Falsifiable Hypotheses

- **H1:** graph-conditioned probes predict the sign of the later 3M paired
  contrast better than a shuffled-graph probe.
- **H2:** CAR-W versus v33 has negative mean and median paired log-error delta,
  non-positive upper-tail CVaR, and zero catastrophic losses after probe cost.
- **H3:** R and S provide positive incremental value only after W passes, in
  separate ablations.
- **H4:** the net improvement repays the 6% probe opportunity cost.
- **H5:** no-overlap controls abstain and remain bit-equivalent to v33.

Failure of any hypothesis is a method result, not a reason to tune a threshold
against the final outcome. Low commit coverage means the evidence is
non-identifiable; a positive short probe with a wrong long-horizon sign means
the horizon/evidence contract is wrong and must be redesigned and
re-registered.

## Staged Verification

1. Implement and test CAR-W only; keep v33 and v32 untouched.
2. Run CLI/5k smoke with snapshot round-trip, branch-order swap, CRN replay,
   equal-FE, AOB-hash, payload and discarded-branch isolation checks.
3. Freeze parameters, then use seeds 9-11 on the pre-registered diagnostic
   suite `E1/E2/S3/R4/A5/E6`. Its AOB overlap degrees are respectively
   `0/.019/.057/.095/.133/.190`, covering no-overlap plus every overlap stratum
   while balancing the four offline objective transforms. Each case includes
   v33, CAR-W, shuffled-graph, no-action and paired-fallback-probe controls.
   Suite identities are experiment indices only and never runtime features.
4. Require at least six commits across at least three cases and two topology
   strata, probe sign agreement >=60%, mean <0, median <=0, zero catastrophic
   losses, and probe overhead <=6%. If this gate fails, stop before R/S.
5. If CAR-W passes, run the held-out 13-case set with seeds 12-16. Require
   integrity pass, mean-case wins >=7/13, worst-seed wins >=5/13, zero
   catastrophic losses, non-positive upper-tail CVaR, and shuffled control not
   better than CAR in a majority of paired cases.
6. Only after the held-out gate passes, evaluate the full 24 cases with seeds
   17-21. Paper-best comparisons are offline secondary reporting; they are not
   runtime inputs and `13/24` best-of-three is not sufficient by itself for a
   success claim.

## Claim Boundary

The strongest claim allowed before the new gates pass is:

> ARAC-CAR defines a reference-blind, same-budget, within-run causal action
> racing protocol for testing overlap-conditioned interventions with explicit
> abstention and branch-level auditability.

## CAR-W2 zero-regret lease

CAR-W2 is a separately registered implementation of the W channel after the
CAR-W v1 utility failure. Its defining constraint is that discovery is part of
the native v33.8 route: before a stable two-sweep component plan exists, W2
does not reserve probe FE, reshape sweep budgets, evaluate a checkpoint solely
for CAR, or invalidate pending v33 trust state. A missing or structurally
futile plan therefore returns the untouched v33 state with zero probe FE.

Only a surviving plan can request the fixed 3% probe cap. The lease is charged
from the remaining ledger at the barrier; if a complete paired horizon cannot
fit, W2 abstains without consuming a shortened probe. The positive LCB,
lower-tail, endpoint, equal-FE, branch-isolation, and atomic final-pair gates
are unchanged from W1. The zero-regret prefix is a protocol invariant, not an
offline outcome check.

CAR-W3 retains that native prefix and adds a logically safe sequential futility
stop. Pair 0 is executed before pairs 1-2 are reserved. A non-positive pair-0
candidate contrast or a candidate endpoint worse than its checkpoint adopts
the fallback immediately. Such an observation cannot satisfy the frozen
`Tail >= 0` and positive-LCB gate, so the early stop changes probe cost but not
which candidates are eligible to commit. A positive, endpoint-safe pair 0
continues to the unchanged three-pair gate.

Performance, generalization, and any paper-best win count remain empirical
claims gated by the staged protocol above.

## State-conditioned delayed credit: audited candidate, not current runtime

The 2026-07-15 offline state-sufficiency audit found that a single generic
local credit cannot govern all actions. One-group downstream writeback credit
did not order later utility, resource retirement was approximately neutral,
and only post-action precision-search progress showed a useful descriptive
association with later incremental utility. The evidence and limitations are
recorded in `docs/design/state-evidence-sufficiency-audit-20260715.md`.

Any future state-conditioned controller must lock shared-variable actions at
the overlap-connected component, allow at most one pending action per mutable
component, and resolve credit on an action-specific semantic horizon.
`group_id -> action` tables, common credit windows across W/R/S, and thresholds
fit to the three-seed terminal outcomes are prohibited. The first eligible
pilot is search-start exposure control after a capped initial lease; writeback
and resource revisions remain blocked until component persistence, neighbour
spillover and overwrite/survival are explicitly traced.

### v40 trace-only instrumentation boundary

The v40 profile is the first implementation of that evidence boundary. It
inherits v38's runtime route and adds only `ComponentDelayedCreditTrace` fields:
component topology, proposal disagreement, decision FE and remaining budget,
pending/lock state, action-specific revisit resolution, local/component/neighbor
gain, and shared-variable overwrite/survival. The tracker writes trace rows
only; it has no action-selection API and cannot alter dispatch, candidates,
optimizer state, RNG, or FE accounting. A 5k E2/seed1 parity run produced
identical v38/v40 final error, FE, and all common trace fields, with 19 relation
observations. It did not exercise precision reanchor, so no delayed-credit
utility or stability claim follows from this smoke.

### S20 coverage result: credit needs a mutex and a completable horizon

The held-out A4/S2/E2 seed31-33 trace probe preserved v38 behavior exactly on
three 3M parity anchors and passed all FE, AOB, and anti-leakage checks. Its
coverage gate nevertheless failed because E2 seed33 started 15 precision
actions near the budget endpoint and none reached the next canonical revisit.
Across all seven precision-bearing runs, 1,796/1,803 actions started while the
same component already had pending credit, and the pending depth reached 19.

This observation sharpens the method boundary. A group-local action may not
open a new lease while its mutable overlap component has pending credit, and a
new lease must be feasible to resolve before the remaining FE budget expires.
The required state is therefore `(component_pending_count,
projected_revisit_fe, remaining_fe)`, derived only from the current run. It is
not a `group_id -> action` table. The next eligible work is an offline replay
of these two eligibility conditions; v40 remains trace-only until that replay
is preregistered and passes.

### S21 feasibility replay: mutex works, prior-cycle horizon does not release

The exploratory S21 replay applied the two eligibility conditions to the
frozen S20 traces without optimizer execution. A lease was selected only when
the component had no replay-selected pending lease and the remaining ledger
could cover the maximum of the latest completed per-relation-group cycle
intervals observed before the action sweep. Current-action credit status,
resolution, gains, case and seed identity, and terminal outcomes were excluded
from eligibility.

The mutex removed all overlap and every selected lease closed: 71/71 selected
actions resolved, with zero overlap violations. This did not pass the coverage
gate. Selection occurred in only five runs and two cases, below the frozen six
run/three case minimum; every E2 action abstained on the horizon check. E2
seed32 illustrates the estimator failure: its first precision action had
12,433 FE remaining, the prior-cycle component maximum projected 342,185 FE,
and the observed revisit took 11,503 FE. The scheduler's per-group work shrinks
strongly near the budget endpoint, so a prior full-cycle duration is not a
stationary estimate of the next cycle. This is explicit in both the upstream
HCC source and the ARAC runner: each sweep recomputes the uniform group budget
as `ceil(remaining_fes / group_count)`.

The projection was also not a strict upper bound: 13/71 selected actions had
observed resolution delay above the estimate, although all happened to close
in this sample. The replay therefore establishes the component mutex as a
necessary serialization rule but rejects prior-cycle extrapolation as the
horizon contract. No runtime lease controller is authorized. The next eligible
horizon must be computed from action-time committed scheduler/ledger state as
a deterministic revisit cap, or the controller must abstain; it may not be a
threshold fitted to these observed resolutions.
