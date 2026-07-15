# Causal-Risk Precision Scheduler Design

Date: 2026-07-15
Executor: Codex
Status: frozen before implementation commit and fresh optimizer FE
Source HEAD: `a739997`

## Decision

The v41 component-lease controller remains permanently dispatch-blocked. Its
component mutex and deterministic scheduler cap solved attribution and credit
closure, but did not identify terminal action utility. No v41 threshold,
cooldown, margin, case, seed, or fallback revision is permitted.

The next falsifiable method adds an independent utility-identification layer.
Its first and only action channel is the existing post-retirement precision
reanchor. The frozen fallback is v37 normal-refine sigma; the candidate applies
the existing v38 precision sigma to one group optimizer with exactly the same
FE, then both arms continue under v37 semantics. Each trajectory contributes
at most its first feasible treatment opportunity and a deployed policy may
release at most one treatment per trajectory.

Repeat precision leases, writeback, resource allocation, and search-start
portfolio stacking are outside this protocol.

## Three-Layer Contract

### Feasibility

The candidate exists only when the inherited post-retirement precision route
is structurally eligible, its overlap component is unlocked, the deterministic
scheduler revisit cap is reachable, and all pre-action history windows are
complete. Feasibility answers whether the action can be executed and observed;
it does not claim utility.

### Pre-Action Utility State

The immutable model payload contains exactly these sixteen finite fields:

1. `remaining_fe_ratio`
2. `revisit_cap_remaining_ratio`
3. `component_group_fraction`
4. `component_shared_variable_ratio`
5. `component_mean_overlap_ratio`
6. `proposal_disagreement_mean_2`
7. `candidate_dose_ratio`
8. `phase_i_tail_progress_rate`
9. `cc_progress_rate_last`
10. `cc_progress_rate_slope_4`
11. `cc_progress_rate_std_4`
12. `cc_stagnation_streak`
13. `terminal_sigma_ratio_last`
14. `log_sigma_slope_3`
15. `success_generation_ratio_last`
16. `offspring_diversity_ratio_last`

Every source watermark must satisfy `source_end_fe < decision_fe`. Current
action outcomes may never mutate this snapshot. Case/problem/seed/function
family, run/lane identity, graph/component/group/relation fingerprints or
indices, raw objective magnitude, incumbent or target vectors, RNG seeds,
paper/history/oracle data, and all current/final outcome fields are forbidden.

### Safe Release

The runtime decision order is fixed:

```text
candidate feasible
  -> component unlocked
  -> immutable state complete
  -> audited model/schema/hash valid
  -> state in support
  -> calibrated LCB(tau) > 0
  -> catastrophic-risk UCB <= 0.05
  -> no earlier release in this trajectory
  -> release the existing precision dose
otherwise -> explicit v37 fallback with an auditable reason
```

A missing or invalid model is a profile startup failure. A valid model that
rejects one state produces an explicit baseline decision; there is no silent
fallback model.

## Causal Logging Contract

Long-horizon labels reuse two independent fresh HCC subprocess lanes. They
deterministically replay the same run to the selected action-time barrier,
then execute fallback or action. This is not represented as a complete
in-memory HCC checkpoint clone.

The paired integrity gate requires identical prefix state and fitness-record
hashes, immutable feature hash, controller-state manifest, candidate contract,
counter-based random descriptor, actual intervention FE, AOB inputs,
environment, and absolute terminal target. Both lanes disable further
experimental precision actions after the selected opportunity and continue
with v37.

Before outcomes exist, a fixed SHA-256 assignment designates one arm as the
randomized logged observation with known propensity 0.5. The other arm remains
sealed from a held-out fold until predictions are frozen; it is then used only
for exact-pair and sign auditing.

For non-negative errors with floor `1e-300`:

```text
Y_a = log(checkpoint_error) - log(terminal_error_a)
tau = Y_action - Y_fallback
    = log(terminal_error_fallback) - log(terminal_error_action)
catastrophic = terminal_error_action >= 1.20 * terminal_error_fallback
```

Only terminal `tau` is a training target. Component closure, 3x, and 9x
horizons are diagnostic.

## Estimator And Support

- Treatment and nuisance models use 1,000 case-cluster bootstrap depth-2
  regression trees with `min_samples_leaf=max(8, ceil(0.1*n_train))`.
- Propensity is the registered 0.5 and is not estimated.
- The CATE lower bound is the bootstrap 5th percentile adjusted by a
  cross-fitted one-sided residual conformal quantile.
- Catastrophic risk uses Beta-smoothed leaf counts; the runtime upper bound is
  the more conservative bootstrap and exact-binomial bound.
- OOD preprocessing is fitted inside each training fold: median/IQR scaling,
  per-feature raw training range, and `k=5` nearest-neighbour distance. The
  distance limit is the training leave-one-out 95th percentile.
- Primary inference is leave-case-out; leave-seed-out is a required
  sensitivity analysis. Policy value uses cross-fitted doubly robust scores
  and 2,000 case-by-seed multiway cluster bootstrap replicates.
- Offline training may use a pinned scikit-learn extra. Runtime consumes only
  the exported, hash-bound JSON tree bundle.

## Required Artifacts

Raw facts:

- `causal_decision_features.csv` (decision id plus the exact model payload);
- `causal_decision_audit.csv` (identity, scope, cap, assignment, hashes and
  source watermarks, none of which are model columns);
- `causal_branch_manifest.csv` and `causal_outcomes.csv`;
- `randomized_log.csv`, `feature_manifest.json`, and
  `causal_logging_manifest.json`;
- the existing strict FE, AOB input, environment and anti-leakage artifacts.

Derived validation:

- `fold_assignments.csv`, `crossfit_predictions.csv`, and
  `policy_value_summary.csv`;
- `causal_identifiability_gate.json` with an explicit
  `runtime_scheduler_authorized` boolean;
- `causal_risk_precision_model.json` only when every full gate passes.

The trainer reads an explicit feature allowlist. It may not infer features by
selecting numeric CSV columns.

## Frozen Execution Ladder

1. CLI/5k: E1/E2, seeds 1/2, fallback/action. If no action snapshot is
   exercised, run one A4 seed1 100k trace smoke.
2. Pilot: A4/A5/E1/E2/E3/E4/S2/S5, seeds 40-44, two arms, strict 3M FE.
3. Full logging, only after the pilot passes: all 24 AOB cases, seeds 40-51,
   two arms, strict 3M FE.
4. Shadow, only after full authorization: all 24 cases, seeds 52/53, v37 with
   model decisions unable to affect execution.
5. Live pilot: the eight pilot cases, seeds 54-58, scheduler versus v37.
6. Full evaluation, only after the live pilot passes: all 24 cases, seeds
   54-58, scheduler/v37/v33.8. Paper-best is joined offline afterward.

Every stage uses the pinned numerical environment, strict FE accounting,
unchanged AOB inputs, fresh provenance, and a new non-overwriting output root.

## Hard Stops

The pilot may continue only with 100% integrity, at least 30 applicable pairs
from six cases and five seeds, at most 5% critical-feature missingness, at
least 15 one-percent material effects, positive LCO and LSO DR point values,
sign balanced accuracy above 0.55, at least 50% in-support coverage, and no
policy-selected catastrophe.

Runtime is authorized only when the expanded data also has at least 80
applicable pairs from eight cases and eight seeds, LCO support at least 60%,
positive one-sided 95% DR lower bounds under both LCO and LSO, exact-pair and
DR direction agreement, no case contributing more than half of absolute
gain, sign-accuracy lower bound above 0.5 on at least 30 material pairs,
positive point value in at least 75% of case folds and 60% of seed folds, and
at least 59 released held-out decisions across six cases/five seeds with zero
catastrophes and a one-sided exact-binomial risk bound no greater than 5%.

Failure of any gate is a completed no-go result. It prohibits model export,
runtime registration, shadow/live/full extension, and threshold or matrix
revision against the observed outcomes.

`[CONTRACT-ACKNOWLEDGED]`
