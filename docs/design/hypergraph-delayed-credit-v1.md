# Hypergraph Delayed Credit v1

Date: 2026-07-17
Executor: Codex
Status: frozen before new optimizer FE

## Decision And Scientific Boundary

`hypergraph-delayed-credit-v1` replaces the failed precision line with a
falsifiable two-stage program:

1. observe unmodified v37 and test whether pre-action hypergraph state predicts
   the next complete normal sweep across held-out cases and seeds;
2. only after both predictive gates pass, test one fixed sweep-end
   shared-variable coordinated commit.

The `0.5 * sigma` precision action is permanently retired for new optimizer
runs. This includes v38 precision dose, v41 leases, causal-v2, response-loop-v1,
and component-precision-action-validity-v1. Their code and artifacts remain
available only for audit or explicit frozen replay. Sigma, thresholds,
opportunities, and schedulers on that route must not be tuned again.

The baseline is `arac_evidence_action_controller_v37`. Its CMA kernel, sigma,
CSA, population, restart behavior, requested group budgets, optimizer seeds,
native group writeback, and native sweep-end handlers remain unchanged. The
observer is opt-in and trace-only. Resource reallocation, visit-frequency
changes, early-stop changes, repeated actions, and a scheduler are out of scope.

Observer parity requires exact equality of optimizer results, FE breakdown,
action trace, AOB hashes, budget summary, and the behavioral bytes of
`evaluation_record.txt`. The sole excluded field is its `Run Time:` wall-clock
line, which is nondeterministic across fresh subprocesses and is not optimizer
state. Observer-only trace/audit files are outside the common parity surface.

Formal screen and full runs require a clean tracked Git tree. Their aggregate
manifest binds the Git commit and a canonical source bundle containing the pure
policy, observer, HCC binding, runner, exp003 entry point, and auditor. The full
stage must use the same commit and source bundle as a recomputed passing screen
gate; a standalone full matrix cannot authorize action implementation.

## Hypergraph And Pre-Action State

The original decomposition is a hypergraph. Decision variables are vertices
and every raw group is one hyperedge. A shared variable keeps every original
group membership. Union-find, connected-component transitive closure, and
neighbor-of-neighbor expansion are forbidden in the new observer and action
call graph.

For a focal raw hyperedge `H`, its shared set `S` contains exactly the variables
in `H` owned by at least two raw groups. The direct-owner set of a variable is
its one-hop star. The direct-neighbor set `N` is the union of the owners of
variables in `S`, excluding `H`. Group and variable identifiers may route and
audit a selected action, but they cannot be score inputs, fingerprints, or
utility tie-breakers.

All errors used below must be finite and non-negative. Let `eps = 1e-300`. For
group `g` in complete native sweep `t`, define:

```text
gain[g,t] = log((pre_error[g,t] + eps) /
                (min(pre_error[g,t], best_error[g,t]) + eps))
u[g,t] = 1000 / actual_fe[g,t] * gain[g,t]
success[g,t] = 1 if best_error[g,t] < pre_error[g,t], else 0
```

`actual_fe` must be the positive full native group interval from immediately
before v37's existing precheck evaluation through the end of every group-local
rescue/recovery path. It is not the requested cap and not only the primary CMA
block. Raw errors are discarded from the policy snapshot after these derived
values are computed; they remain only in the audit surface.

The first complete policy state uses exactly three complete sweeps ending at
`t`. `success_ratio_3` is computed only to derive difficulty; it is not a
second policy feature because `success_ratio_3 = 1 - difficulty` exactly:

```text
ewma_u[g,t] = EWMA(u[g,t-2], u[g,t-1], u[g,t]), alpha = 0.5
success_ratio_3[g,t] = mean(success[g,t-2:t])
difficulty[g,t] = 1 - success_ratio_3[g,t]
stagnation[g,t] = min(consecutive sweeps ending at t with u <= 0, 3) / 3
```

For shared variable `v`, let `p[g,v,t]` be the proposal captured after direct
owner `g` completed every native group-local rescue/recovery path and before
any relation writeback. This is the final owner proposal, not an earlier raw
CMA result. With finite domain width `w[v] = upper[v] - lower[v] > 0`:

```text
disagreement[v,t] = clip((max_g p[g,v,t] - min_g p[g,v,t]) / w[v], 0, 1)
disagreement[H,t] = mean(disagreement[v,t] for v in S)
```

An empty `S`, an incomplete group visit, missing proposal, missing three-sweep
history, or missing previously closed next-sweep survival/overwrite makes the
state inapplicable. Every source watermark is at or before the decision FE and
the snapshot is immutable before any action evaluation.

The prior owner outcome at state `t` is computed from the proposal issued in
complete sweep `t-1`. Let `a[g,v,t-1]` be the shared-variable context immediately
before that owner block, `p[g,v,t-1]` its captured proposal, and `x[v,t]` the
state after every native handler at the end of sweep `t`. Per-variable
retention is the clipped directional projection of `x-a` onto `p-a`, weighted
by `abs(p-a)`. If every shared displacement is exactly zero, survival and
overwrite are both the neutral value `0.5`. The policy snapshot retains only
prior overwrite; survival remains an audit/outcome value because
`survival = 1 - overwrite` exactly. Thus no future or entry-time label is
backfilled into the snapshot, and no complementary signal is double-weighted
by support distance or reliability.

The six independent policy features are current and EWMA unit-FE contribution,
difficulty, stagnation, direct-owner proposal disagreement, and prior
next-sweep overwrite. Success ratio and prior survival are audit-only
complements.

All scalar ranks are ascending within the same complete sweep and population
of eligible raw hyperedges or direct owners. Ties use midrank percentiles; a
one-element population has percentile `0.5`. Define:

```text
C[H,t] = mean(rank(u[H,t]), rank(ewma_u[H,t]))
Q[H,t] = mean(rank(difficulty[H,t]),
              rank(stagnation[H,t]),
              rank(disagreement[H,t]))
priority[H,t] = 0                         if C + Q == 0
                2 * C * Q / (C + Q)       otherwise

reliability[g,t] = mean(rank(u[g,t]),
                        rank(ewma_u[g,t]),
                        rank(1 - prior_overwrite[g,t]))
```

Each raw group is one hyperedge and contributes one observation per complete
sweep. The unique highest-priority hyperedge is selected. A complete numeric
tie abstains; neither index order nor identity may break a utility tie.

## Predictive Gate Before Runtime

The predictive estimand uses exactly one decision snapshot per trajectory: the
first complete sweep end with three complete history sweeps and a legally
closed `t-1 -> t` owner outcome. That snapshot contains one row for every
eligible raw hyperedge. Later sweeps may resolve its `t+1` labels but cannot
create another decision cohort. A trajectory is decision-eligible using only
pre-label evidence: the locked snapshot is complete, contains at least two raw
hyperedges with shared variables, and has one unique highest-priority focal
hyperedge. Hyperedge rows are repeated measurements inside one case-by-seed
trajectory, never independent observations or bootstrap units.

A no-overlap state, a complete priority tie, or an incomplete first opportunity
is explicitly inapplicable and cannot be replaced by a later, more favorable
snapshot. An incomplete opportunity writes one audit row for every raw group;
groups never visited because the native sweep ended use empty group-level
fields and an explicit natural-censor placeholder. A complete decision snapshot
whose next sweep cannot finish at the terminal budget remains valid integrity
evidence and remains in the decision-eligible denominator, but its label is
terminal-censored. It therefore forces the required label-closure gate to fail;
the auditor cannot delete it and estimate on a more favorable complete subset.

The required-state missing fraction is fixed before state or label filtering.
Its denominator is every trajectory that reaches its first locked opportunity
and whose raw topology contains overlap; its numerator is any such trajectory
for which one or more of the six required state values cannot be reconstructed.
A zero denominator is undefined and fails closed. This prevents missing states
from disappearing by first being classified as inapplicable.

The trace label is the complete next-normal-sweep unit-FE contribution
`u[g,t+1]`. Owner labels are directional survival and overwrite observed only
after the next sweep and all native sweep-end handlers finish. Entry-time or
next-group closure is invalid.

Let trajectory `i=(case, seed)` contain eligible hyperedges `G_i`, unique focal
`g*`, priority `p_ig`, next-sweep contribution `y_ig`, owner reliability `q_ig`,
and next-sweep survival `s_ig`. The fixed predictive checks are computed inside
each trajectory before any cross-trajectory aggregation:

```text
rho_priority_i = Spearman(p_i, y_i)
focal_rank_i = midrank_percentile(y_i)[g*] - 0.5
rho_owner_i = Spearman(q_i, s_i)
diagnostic_delta_i = y_i,g* - mean(y_i,g for g != g*)
```

If an otherwise decision-eligible trajectory has a constant outcome or an
undefined correlation, its corresponding rank statistic is preregistered as
zero: it supplies no predictive evidence and is not removed based on its
outcome. `diagnostic_delta_i` is reported but is not a cross-case hard gate
because raw contribution magnitudes vary by problem. The three bounded/rank
statistics give every case-by-seed trajectory exactly one equal-weight value,
irrespective of its number of groups.

Overwrite is `overwrite > 0.5`. Leave-one-case-out (LCO) and leave-one-seed-out
(LSO) folds fit the reliability threshold on their training trajectories using
row weight `1 / |G_i|`; the threshold is the trajectory-weighted median and the
prediction is `q_ig < threshold`. The weighted median is the lower weighted
quantile: the infimum observed reliability whose cumulative normalized weight
is at least `0.5`. Test-fold confusion contributions use the same
within-trajectory weights before balanced accuracy is computed. A held-out fold
may contain one class provided the complete pooled cross-fitted LCO or LSO route
contains both classes; a single-class pooled route is undefined and fails
closed. In a bootstrap replicate, a route missing either class remains in the
bootstrap distribution with balanced accuracy `0`, rather than being deleted.
Case and seed are fold keys only and never enter a score.

There is no OOD support filter in the hard gate. The score is a fixed
within-trajectory rank formula, not a learned cross-case runtime model, and the
conditional action would run on every decision-eligible state. Support may be
added as a diagnostic in a future learned scheduler but cannot filter rows or
rescue a negative all-state result here.

All confidence bounds use 2,000 case-by-seed two-way pigeonhole bootstrap
resamples and the 5th percentile as a one-sided 95% lower bound. Cases and seeds
are sampled independently; each trajectory scalar receives the product of its
case and seed multiplicities. Balanced-accuracy bounds apply the same weights
to fixed cross-fitted prediction/confusion rows and are explicitly conditional
on those fitted thresholds. Non-finite statistics, single-class balanced
accuracy, incomplete folds, or empty bootstrap replicates fail closed. A
diagnostic tree or post-hoc threshold cannot rescue the fixed score.

The auditor does not trust either derived CSV as a second source of truth. For
each trajectory it first derives the earliest lock opportunity from the raw
complete/incomplete sweep sequence and requires the manifest and every cohort
flag to name that same sweep. Every complete sweep must cover all raw groups in
canonical order with non-overlapping FE intervals, one shared decision FE, one
fitness-prefix hash, and one native endpoint. For every variable, a proposal
row's `t -> t+1` value and FE must equal the canonical endpoint and decision FE
recorded by the raw `t+1` sweep; a self-consistent backfill is not trusted.

The auditor then reconstructs the six state values from the `t-2:t` raw group
audits, raw shared proposals, topology, bounds, and FE watermarks; reconstructs
within-snapshot ranks, scores, and the unique focal; formats every reconstructed
state/score value with the runtime `.17e` representation and hashes that exact
payload; and reconstructs `t+1` unit-FE contribution, directional survival, and
overwrite from the next raw sweep. Pending, no-overlap, tie, incomplete, closed,
and terminal-censored manifest status/closure are also derived from raw evidence
rather than trusted. Any mismatch is an integrity failure, not a missing label
or an alternative estimand.

The same exact-string rule applies to outcomes. A reported contribution,
survival, or overwrite must equal the raw reconstruction in runtime `.17e`
format; a numeric tolerance is not accepted, and only the reconstructed value
enters any rank or gate statistic. Each per-run manifest path is stored relative
to and contained by the aggregate result root. Its four raw CSV hashes and
headers are rechecked, and those per-run rows must reconstruct the four aggregate
raw CSVs exactly. Editing a root CSV and its root hash therefore cannot create a
new result while the per-run artifacts remain unchanged.

Every formal run must finish with HCC status `completed` and observed FE in
`[target - tolerance, target]`, where `target` is the matrix terminal FE and
`tolerance` is the maximum native group CMA population size recorded by the
runner. The unique `(case, seed, observer lane)` ledger row must agree with the
per-run target, observed FE, and tolerance. Early termination, overspend,
duplicate routes, or a missing route fail the stage.

The trace screen is the eight cases `A4/A5/E1/E2/E3/E4/S2/S5`, seeds `91-95`,
v37 observer only, strict 3M FE (40 trajectories). It passes only with:

- 100% fresh, FE, AOB, RNG, hash, source-watermark, and observer integrity;
- at least 30 applicable trajectories, six cases, and all five seeds;
- complete next-sweep labels for 100% of pre-label decision-eligible
  trajectories and at most 5% missingness in required state;
- positive equal-trajectory means for `rho_priority_i` and `focal_rank_i`;
- trajectory-weighted overwrite balanced accuracy above `0.55` separately for
  LCO and LSO cross-fitted predictions, without support filtering;
- positive `focal_rank_i` direction in at least three seeds and five overlap
  cases.

The full trace matrix is all 24 cases, seeds `96-103`, v37 observer only,
strict 3M FE (192 trajectories). It passes only with:

- a supplied screen gate that is re-audited as `screen_pass`, with unchanged
  input hashes, Git commit, config/spec hashes, and source bundle;
- the same 100% integrity requirements;
- at least 120 applicable trajectories, 16 cases, and all eight seeds;
- complete next-sweep labels for 100% of pre-label decision-eligible
  trajectories;
- positive two-way-cluster bootstrap lower bounds over all equal-weight
  decision-eligible trajectories for `rho_priority_i`, `focal_rank_i`, and
  `rho_owner_i`;
- conditional LCO and LSO overwrite balanced-accuracy lower bounds above
  `0.5`;
- positive `focal_rank_i` direction in at least 75% of case means and six of
  eight seed means;
- no case mean contributing over 50% of the summed absolute focal-rank
  advantage.

The final concentration statistic is exactly
`max_c |mean_i-in-c focal_rank_i| / sum_c |mean_i-in-c focal_rank_i|`. A zero
denominator is undefined and fails closed.

Failure of either trace gate is the completed scientific result: do not create
an action runtime profile, model bundle, fresh action run, or scheduler.

Static leakage is not delegated to the generic runtime-payload CSV. The formal
auditor independently parses the pure policy state/score/build call graph,
requires the relevant dataclasses to remain frozen, and rejects identity,
fingerprint, raw-objective, future-outcome, historical-result, or paper-best
inputs. Problem and seed remain audit/fold identities outside that policy call
graph.

## Conditional One-Hop Coordinated Commit

`one-hop-shared-commit-v1` may be implemented only when a canonical passing
`hypergraph_identifiability_gate.json` is bound by protocol, config, source,
and data hashes. The first applicable decision occurs once per trajectory at
the first complete sweep end with three complete history sweeps, a unique focal
hyperedge among at least two shared hyperedges, no auxiliary context
replacement, and enough remaining FE for the two action evaluations, the next
complete normal sweep, and the absolute terminal target. It inherits the trace
decision-eligibility definition; a singleton cannot become action-eligible.

For each `v` in `S`, direct-owner raw weights are `1 + reliability[g,t]`. After
normalization, weights use the unique Euclidean projection onto:

```text
sum_g weight[g,v] = 1
0 <= weight[g,v] <= 0.65
```

Let `z[v]` be the weighted direct-owner proposal and `x_A` the immutable native
sweep-end anchor. Define domain-normalized Euclidean lengths:

```text
D = norm2((max_g p[g,S,t] - min_g p[g,S,t]) / w[S])
L = norm2((z[S] - x_A[S]) / w[S])
R = max(|S| / |H|, |N| / (|N| + 1))
lambda = min(1, (1 - R) * D / (L + eps))
x_C[S] = x_A[S] + lambda * (z[S] - x_A[S])
x_C[not S] = x_A[not S]
```

The candidate must be finite, inside the original domain, and have a non-zero
shared-coordinate displacement. The trajectory has exactly one decision
attempt: once the first applicable snapshot is sealed, an invalid candidate is
a permanent abstention and cannot be retried later.

The three frozen arms are:

- `c0_v37`: plain v37 plus side-effect-free opportunity shadow logging;
- `c1_shadow_coordinate`: evaluate anchor then candidate with exactly two
  counted FE, apply the same archive bookkeeping as c2, and retain the native
  cooperative context;
- `c2_commit_if_improved`: take the identical c1 path and replace only the
  shared coordinates in the cooperative context when `f(x_C) < f(x_A)`.

The two evaluations do not consume optimizer RNG or update any CMA kernel.
c1/c2 evaluation order, hashes, FE, and archive bookkeeping are identical. A
rejected c2 must be bit-equivalent to c1 at terminal and on the common trace.
After the one decision, all optimization remains unmodified v37.

Delayed credit closes only after the next complete normal sweep and every
native sweep-end handler. For non-zero `delta[v] = x_C[v] - x_A[v]`:

```text
retention[v] = clip((x_D[v] - x_A[v]) / delta[v], 0, 1)
survival = sum_v abs(delta[v]) * retention[v] / sum_v abs(delta[v])
overwrite = 1 - survival
credit = log((error_anchor + eps) / (error_delayed + eps))
         - log(1.01) * overwrite
```

The paired survival and credit effects are c2 minus c1. They are audit outcomes
and cannot enter the v1 commit. A first action cannot repeat or renew.

## Fresh Action Gates

All action runs use the same prefix through the decision, strict absolute
terminal FE, frozen topology/state/plan/candidate hashes, and 100% delayed
closure. Define:

```text
tau_A = log(error_c0_terminal / error_c2_terminal)
tau_C = log(error_c1_terminal / error_c2_terminal)
tau_D = log(error_c1_delayed / error_c2_delayed)
delta_credit = credit_c2 - credit_c1
delta_survival = survival_c2 - survival_c1
```

Errors use the `1e-300` floor. A material total effect is
`tau_A >= log(1.01)`. A catastrophic event is a candidate-side error at least
`1.2` times its registered control for c1/c0, c2/c0, c2/c1, or delayed c2/c1.
ITT contains every registered case-seed and assigns a zero action effect to a
bit-equivalent inapplicable pair. ATT is fixed by the immutable pre-action
applicability state. The commit-eligible stratum is fixed by the two immediate
evaluations (`f(x_C) < f(x_A)`) before any delayed or terminal outcome exists.

The action screen is the eight screen cases, seeds `104-108`, all three arms,
strict 3M FE (120 trajectories). It passes only with:

- 100% fresh, FE, AOB, anti-leakage, prefix, topology, state, plan, candidate,
  two-FE, terminal, and closure integrity;
- at least 30/40 applicable pairs over six cases and five seeds;
- at least ten commit-eligible pairs over four cases and three seeds;
- positive one-sided 95% lower bounds and non-negative medians for tau_A ITT,
  tau_A ATT, commit-eligible tau_C, tau_D, and delta_credit;
- positive tau_A and tau_C seed means in at least three of five seeds;
- at least five material tau_A pairs over three cases and three seeds;
- positive lower bounds and non-negative medians for delta_survival and
  delta_credit;
- zero terminal or delayed catastrophes, 100% delayed closure, and 100%
  rejected-c2/c1 bit parity.

The confirmation matrix is all 24 cases, seeds `109-116`, all three arms,
strict 3M FE (576 trajectories). It passes only with:

- the same 100% integrity, closure, and rejected-c2/c1 parity;
- at least 120/192 applicable pairs over 16 cases and all eight seeds;
- at least 59 commit-eligible pairs over 12 cases and all eight seeds;
- positive lower bounds and non-negative medians for tau_A ITT, tau_A ATT,
  commit-eligible tau_C, tau_D, and delta_credit;
- positive case means for tau_A in at least 13/24 cases and tau_C in at least
  12/24 cases;
- non-negative tau_A and tau_C means in every seed, with at least six of eight
  strictly positive for each contrast;
- non-negative worst-10% CVaR for applicable tau_A and commit-eligible tau_C;
- zero terminal and delayed catastrophes, with the zero-event one-sided 95%
  Clopper-Pearson upper bound at most 5%;
- material tau_A in at least 20% of applicable pairs over eight cases and six
  seeds;
- positive delta-survival lower bound and median, with strictly positive
  retention in at least half of commit-eligible pairs;
- no case contributing over 50% of absolute tau_A ATT advantage and a positive
  tau_A mean over the 16 cases absent from the screen.

Passing confirmation establishes only a stable main effect for this exact
one-shot action. It authorizes a separate scheduler preregistration, not a
scheduler implementation. Any gate failure retires this exact coordinated
commit without changing scores, weights, risk, matrices, or thresholds.

## Artifacts And Leakage Boundary

Trace-only stages emit:

- `hypergraph_manifest.json`;
- `hyperedge_cycle_features.csv` (decision id plus derived whitelist only);
- `hyperedge_cycle_audit.csv`;
- `shared_proposal_audit.csv`;
- `hyperedge_cycle_outcomes.csv`;
- `hypergraph_fold_assignments.csv`;
- `hypergraph_crossfit_predictions.csv`;
- `hypergraph_predictive_summary.csv`;
- `hypergraph_trace_manifest.json`;
- `hypergraph_identifiability_gate.json`.

The conditional action stage additionally emits branch, candidate, pair,
delayed-credit, FE-ledger, manifest, and gate artifacts named in the canonical
JSON config. Raw objective values, case/seed identity, raw group/variable
indices, and hashes are audit-only.

Policy scores and commit selection must reject case/problem/seed/family/run/lane
identity; group, variable, relation, graph, component, or fingerprint identity;
raw objective/incumbent/target values; paper-best or historical results; current
action gain; delayed/terminal outcome; win, catastrophic, survival, overwrite,
or resolution labels that were not closed before the immutable snapshot.

## Literature Claim Boundary

Recent work already covers overlapping multi-membership CC, dynamic
variable-importance adaptation, contribution-based resource allocation, UCB
subproblem selection, and learned CC. In particular, the 2024 OCC work
([DOI 10.1145/3638529.3654171](https://doi.org/10.1145/3638529.3654171)),
2025 dynamic variable-importance CC
([DOI 10.1016/j.asoc.2025.113363](https://doi.org/10.1016/j.asoc.2025.113363)),
and 2025 CC resource-allocation work
([DOI 10.1109/TEVC.2025.3629151](https://doi.org/10.1109/TEVC.2025.3629151))
rule out claims that hypergraph overlap, contribution scores, or adaptive
resource selection are independently novel.

The defensible hypothesis is narrower: preserve raw overlap multiplicity,
require held-out next-sweep predictiveness before runtime, and evaluate a
risk-bounded one-hop coordinated commit with delayed overwrite-penalized
credit. These ingredients do not establish novelty or effectiveness by
themselves. Only the preregistered fresh gates can establish empirical support.
