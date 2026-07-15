# State-Evidence Sufficiency Audit: 2026-07-15

Status: partial support for action-specific delayed credit; insufficient
evidence for a new runtime controller. The original offline audit consumed no
FE; its S19 trace-only follow-up consumed only the paired 5k parity smoke
recorded below. No performance claim is made.

Executor: Codex. Source HEAD before the audit: `6e5adc3`.

## Question and boundary

This audit asks whether runtime state can repair the observed failure:
overlap/grouping evidence can trigger an action, but does not stably predict
the action's long-horizon utility across seeds.

All terminal values are joined offline after execution. Case identity,
function family, paper-best values, historical/final outcomes, oracle-selected
arms, and seed labels are not proposed as runtime inputs. Rows are aggregated
at run/case-seed level; thousands of relation/group rows are not treated as
independent samples.

The reproducible offline entry is `scripts/audit_state_evidence.py`. Generated
tables are under `results/state_evidence_sufficiency_audit_20260715`:

- `state_field_coverage.csv`;
- `candidate_run_state_features.csv`;
- `candidate_state_associations.csv`;
- `car_applied_pair_state_features.csv`;
- `car_state_associations.csv`.

The audit reads the frozen v33.8, v34, v35, v36, v37, v38 and CAR v2 raw
directories. It changes neither their artifacts nor the runtime implementation.

## S19 trace-only follow-up

S19 added `arac_evidence_action_controller_v40` as an observation-only profile.
It inherits v38 runtime behavior and records overlap-connected component id and
topology, relation proposal disagreement, search-start decision FE and budget,
pending/lock state, revisit resolution, local/component/neighbor gain, and
shared-variable overwrite/survival. The tracker has no dispatch return path and
does not mutate candidates, optimizer state, RNG, or the FE ledger. It records a
same-component pending conflict for diagnosis but does not block the existing
v38 action in this trace-only stage.

The paired E2/seed1/5k smoke artifacts are:

- `results/controller_v40_component_credit_5k_20260715_final_v38`;
- `results/controller_v40_component_credit_5k_20260715_final_v40`.

Both lanes were fresh and used strict accounting. Final error was
`6.101156e+12` in both lanes, FE was `4996` in both lanes, all 101 common trace
fields matched across 20 rows, AOB input rows were `10/10` unchanged, and the
anti-leakage audit was `16/16` pass. v40 added 19 relation-observation rows with
component ids and disagreement values. No precision reanchor row occurred in
this 5k run, so delayed-credit resolution, overwrite/survival coverage, and
unresolved run-end behavior remain unvalidated by this smoke and cannot be
treated as zero incidence.

## S20 held-out coverage preregistration

This section is frozen before any S20 optimizer run by the Git commit that
contains it. S20 is a trace-coverage experiment, not a performance comparison
or a threshold-fitting dataset.

### Question and selection boundary

The confirmatory question is whether v40 can record valid, action-specific
search-start credit through its semantic resolution horizon without changing
v38 behavior. Cases are selected only from historical v38 action coverage:
A4 had 2,147 precision rows across 3/3 seeds, S2 had 282 across 3/3 seeds, and
E2 had 46 across 2/3 seeds. Historical final error, paper-best, family labels,
and seed outcomes were not used to choose the matrix or any runtime action.

### Frozen matrix

- v40 coverage arm: A4/S2/E2, seeds 31/32/33, one lane, strict 3,000,000 FE;
  9 fresh trajectories in
  `results/controller_v40_component_credit_heldout_seed31_33_3m_20260715`.
- v38 parity arm: A4/S2/E2, seed 31, strict 3,000,000 FE; 3 fresh trajectories
  in `results/controller_v40_component_credit_parity_v38_seed31_3m_20260715`.
- Runtime environment: pinned Python/NumPy/SciPy and single-thread
  `PYTHONHASHSEED`, OMP, OpenBLAS, MKL, and NumExpr settings already enforced by
  exp003. The manifests must identify the preregistration commit.
- Offline audit output:
  `results/controller_v40_component_credit_heldout_audit_20260715`, generated
  only by `scripts/audit_component_credit_coverage.py`.

The v40 seeds are held out from the frozen v38 seed1/2/3 coverage artifact.
Nine trajectories are sufficient only for a wiring/coverage gate; they are
not powered for a performance effect and no significance claim is permitted.

### Frozen integrity and coverage gate

All 12 runs must be fresh, completed, at or below 3M FE, AOB-unchanged, and
anti-leakage clean. The three seed31 parity pairs must match exactly on final
error, FE, AOB hashes, action-trace row count, and every common non-lane trace
field. v40 action plans must retain `trace_affects_dispatch=false` and exclude
v39 sigma continuation.

The coverage gate passes only if all of the following hold:

1. At least 6/9 v40 runs contain a precision action.
2. At least two cases contain precision actions in at least two seeds each.
3. Every v40 run contains component relation observations, and every
   precision-bearing run contains at least one resolved action.
4. At least one `unresolved_run_end` row is observed and no serialized row
   remains `pending`.
5. Aggregate resolved/precision coverage is at least 0.90.
6. Overwrite/survival is observed in at least three runs and two cases; every
   emitted pair is finite, within [0,1], and sums to one.
7. At least one same-component lock conflict is observed. All decision and
   resolution FE values are monotonic and stay within the run ledger.

If integrity fails, the affected trajectories are invalid and must be rerun in
a new directory after a root-cause fix. If coverage fails with valid runs, S20
stops: no credit-gated lease rule, threshold tuning, 3M expansion, or full-24
run is authorized. A coverage pass permits only the next preregistered design
step; it is not evidence that a controller improves optimization performance.

## Field sufficiency

The common trace schema is wider than the values actually recorded. Eight
action-trace fields are empty in every audited artifact:

- `remaining_budget_ratio` and `decision_point`;
- `cc_utility`;
- `search_state_conflict_fraction`;
- `search_state_writeback_unstable`;
- `search_state_relative_writeback_max`;
- `search_state_block_fe` and `search_state_utility`.

The existing non-empty evidence is sparse or action-specific:

| Evidence | Representative coverage | Interpretation |
|---|---:|---|
| v33 trust credit | 1,238/21,696 rows (5.71%) | immediate writeback-local credit |
| v34 downstream recovery credit | 693/6,770 (10.24%) | one subsequent group, not component persistence |
| v36/v37 sweep support | 97.62% / 98.39% | broad structural maturity, not action payoff |
| v38 precision `best_before/after` | 4,537/13,184 (34.41%) | usable search-start local progress |
| v38 `cc_block_fe` | 4,397/13,184 (33.35%) | action dose exists, but `cc_utility` is empty |
| CAR v2 trust credit | 658/7,546 (8.72%) | not aligned to the CAR checkpoint horizon |

There is no explicit proposal-disagreement, shared-variable survival/overwrite,
incident-neighbour spillover, pending-action id, or action-resolution FE. The
full component delayed-credit expression therefore cannot be reconstructed
from current raw artifacts.

## Existing local credit versus later utility

For v34-v36, terminal utility is paired offline against v33.8. To isolate the
new action in later versions, v37 is paired against v36 and v38 against v37.
Positive terminal log advantage means the candidate is better.

| Channel/evidence | Paired result | State-credit association |
|---|---|---|
| v34 writeback recovery | 24 runs; mean log advantage -0.008808; 10 wins, 11 losses, 2 catastrophic | mean recovery credit Spearman -0.234; within-case concordance 10/21 |
| v37 resource retirement | 39 runs; 7 changed; mean -0.0000586; 3 wins, 4 losses, 0 catastrophic | retirement exposure Spearman -0.076; no within-case ordering because exposure is mostly constant |
| v38 precision reanchor | 39 runs; 12 changed; mean +0.007932; 10 wins, 2 losses, 0 catastrophic | precision local log gain Spearman +0.741; within-case concordance 9/11 |

The v34 result rejects the idea that one-group local recovery credit can stand
in for long-horizon writeback credit. Resource retirement is safe in this
sample but did not provide positive incremental utility. Search-start is the
only channel with a useful descriptive delayed-credit signal.

That search-start signal is not release evidence. It is available in only 17
runs across eight cases, and S2 contributes 91.5% of the absolute v38-v37
terminal difference. The Spearman value remains +0.531 when S2 is removed,
and leave-one-case-out values range from +0.531 to +0.833, which makes it worth
a narrow held-out pilot. It does not justify a fitted threshold: there are
only three seeds, the same traces generated the hypothesis, and local gain is
observed after the first action rather than before initial dispatch.

## CAR pre-action state

CAR v2 supplies 10 applied case-seed pairs over four cases. The two structural
sweeps and 38 pre-checkpoint relation rows are constant for every applied pair,
so they cannot rank utility. The strongest post-hoc structural association is
mean rank gap:

| State feature | n | Spearman to terminal | Within-case concordance |
|---|---:|---:|---:|
| checkpoint FE ratio | 10 | +0.079 | 3/8 |
| mean absolute delta gap | 10 | +0.091 | 4/8 |
| mean rank gap | 10 | +0.492 | 6/8 |
| fallback-margin proxy | 10 | -0.467 | 2/8 |

S3 contributes 92.6% of the absolute terminal contrast, so these values remain
descriptive. Mean rank gap stays positive when S3 is removed (Spearman +0.575),
but only two repair-action samples exist. A threshold learned from them would
be a post-hoc case/seed surrogate.

The direct S3 comparison shows why structure alone is insufficient:

| Seed | Checkpoint FE ratio | Mean rank gap | Fallback margin | Closure log advantage | 9x log advantage | Terminal |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.738 | 3.526 | 0.931955 | -0.001316 | -0.048288 | -0.732213 |
| 3 | 0.678 | 3.921 | 0.928983 | -0.030603 | -0.014044 | +0.026262 |

Both seeds selected the same repair action with the same two-sweep maturity.
The ultimately winning seed looked worse at closure. Longer observation did
not solve the general ranking problem: Spearman to terminal was +0.158 at
closure, +0.207 at 3x and -0.067 at 9x; within-case concordance was respectively
1/7, 3/7 and 2/7.

## Method decision

Adding state evidence is necessary, but the correct unit is not
`group_id -> action`. The next defensible method candidate is a
**component-locked, state-conditioned delayed-credit controller**:

1. Structure proposes an eligible action and its scope; it does not certify
   long-horizon benefit.
2. Shared-variable writeback is locked at an overlap-connected component.
   Adjacent groups cannot independently write the same shared variable.
3. Resource and search-start actions may be group-local only when they do not
   share mutable state; otherwise they inherit the component lock.
4. Each component has at most one pending action. A second action cannot be
   credited until the first reaches its action-specific resolution window.
5. Writeback credit waits for a complete component cycle and all incident
   groups to revisit; resource credit waits for its allocation block plus the
   next canonical revisit; search-start credit waits for its search block plus
   the next CC revisit.
6. Credit combines local gain, component gain, neighbour spillover and
   shared-variable overwrite/survival. Unresolved or harmful credit reduces
   dose, enters cooldown or abstains.

The first implementation candidate should be search-start exposure control,
not a simultaneous W/R/S controller. Existing v38 evidence supports using
post-action local progress to decide whether another precision lease is
allowed. It does not support changing the initial dispatch or tuning a repair
threshold.

## Release decision and next gate

- Do not implement a learned group policy, tune thresholds, restart CAR R/S,
  or run a new full-24 matrix from these three-seed artifacts.
- First add trace-only, zero-behaviour-change instrumentation for decision FE,
  remaining budget, component/action id, pending/resolution state, component
  and neighbour gain, proposal disagreement, overwrite rate and survival.
- Then preregister one search-start controller: the existing structural route
  grants the first capped lease; subsequent leases require positive resolved
  component credit and no neighbour/overwrite harm.
- Validate on held-out seeds with same-budget paired references. Report action
  coverage, abstention, mean, worst seed and catastrophic loss. Writeback and
  resource channels remain frozen until their own credit is identifiable.

For the July paper, v33.8 full-24 remains the performance result. This audit is
a mechanism result and an evidence-based roadmap, not a replacement score.
