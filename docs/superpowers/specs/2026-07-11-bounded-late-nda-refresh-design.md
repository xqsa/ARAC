# Bounded Late NDA Refresh Design

Date: 2026-07-11
Executor: Codex
Status: approved design, pending implementation plan

## Objective

Extend the single-lane `canonical_evidence_controller_v1` runtime with an
auditable late-search action that can recover from low-yield, conflict-heavy CC
states without consuming the entire remaining budget. The change targets the
remaining R3 reproducibility gap while preserving the current 12/13 best-of-three
paper-best wins.

The runtime must not use case identifiers, function-family labels, paper values,
historical final errors, relative gains, or prior run outcomes for dispatch.

## Evidence Behind The Design

- The current pinned canonical protocol reaches 12/13 paper-best wins by
  best-of-three. R3 is the only miss, with a best value of `3.974568e5` against
  the paper-best value `3.28e5`.
- An older CC-harm run reached `3.214242e5`, but used Python 3.12.7, NumPy 2.1.3,
  SciPy 1.17.1, Torch 2.12.0, and OpenBLAS 0.3.27.
- Re-running the same fixed CC-harm action in the pinned final environment
  produced `4.316439e5`. The old paper-best win is therefore not a reproducible
  implementation anchor.
- The pinned CC-harm action still improved its R3 seed-3 pre-refresh incumbent
  from `1.375505e6` to `4.316439e5`, showing that a late NDA intervention has
  directional value even though a full takeover is insufficient.
- Historical negative controls triggered full refresh much earlier on A4 and R2,
  with roughly 36%-70% of the budget remaining. S3 is now protected by the
  non-dense repair lock. These observations support a narrower late-window action
  instead of generic CC-harm takeover.

## Considered Approaches

### A. Bounded late NDA refresh inside canonical v31 (selected)

Run a limited NDA refresh from the protected incumbent only after runtime
evidence indicates late CC stagnation or conflict. Reserve budget for canonical
CC continuation after the probe. Accept only an improving refresh candidate.

This preserves one optimizer trajectory and maps runtime evidence to a search
state action without selecting among final outcomes from multiple lanes.

### B. Full remaining-budget CC-harm takeover

This reuses the existing action directly, but the pinned candidate does not beat
paper-best and historical preservation runs regress R2 and S3. It is rejected.

### C. Additional coordinate or repair thresholds

R3's strongest evidence appears late in the search and concerns low CC yield and
relation conflict, not a stable shared-variable ownership error. More relation
thresholds would add selector complexity without addressing the observed failure
mode. It is rejected for this iteration.

## Runtime Trigger

The action is eligible only when every condition below is true:

1. The active controller is canonical v31 and its overlap degree is below the
   existing dense threshold.
2. At least the existing minimum number of group updates has completed, and
   every non-empty overlap relation observed so far in the current outer
   iteration uses exactly three shared variables.
3. The non-dense repair lock is not active and search-state intervention remains
   enabled.
4. The existing CC-harm detector reports low CC gain together with high relation
   conflict or severe group stagnation.
5. The remaining-budget ratio is within the late window `[0.08, 0.30]`.
6. The action has not already been consumed during the run.

All fields are produced by the current run. The trigger cannot access offline
comparison tables or final outcomes.

## Action Semantics

The action name is `bounded_late_nda_refresh`.

1. Preserve the best current or Phase-I incumbent as the guard state.
2. Compute a continuation reserve of 5% of total FE.
3. Set the refresh budget to the smaller of 15% of total FE and the remaining
   FE after subtracting that reserve. Trigger only if the resulting refresh
   budget can fund at least one full-space optimizer population.
4. Warm-start the existing MMES/NDA continuation from the guard state with the
   existing CC-harm refresh sigma multiplier and deterministic derived seed.
5. Accept the refresh candidate only when it strictly improves the guard fitness.
6. Resume canonical CC from the better of the guard and refresh candidate using
   the reserved budget.

The action does not run parallel optimizer lanes and does not choose a result by
consulting final performance. It is one auditable intervention inside one run.

## State And Budget Boundaries

The v31 run state gains one consumption flag for the bounded refresh. Existing
dense locks, non-dense repair locks, relation policy selection, and phase-rescue
behavior remain unchanged.

Refresh FE, resumed CC FE, total FE, and unused FE remain visible in the existing
same-budget ledger. Budget allocation must be clipped before optimizer calls; an
overrun or stage-total mismatch invalidates the run.

## Audit Surface

The existing action-trace schema is extended to record:

- trigger reason and remaining-budget ratio;
- shared-variable count and repair-lock state;
- refresh budget and reserved continuation budget;
- guard source and guard fitness;
- refresh candidate fitness and acceptance result;
- best fitness before refresh, after refresh, and after resumed CC;
- deterministic optimizer seed and FE consumed by each stage.

The trigger row is emitted when the refresh starts. A completion row is emitted
after canonical CC resumes so the final post-continuation fitness is auditable
without mutating an already written trace record.

The policy source must identify `bounded_late_nda_refresh` without embedding a
case or function label.

## Failure Handling

- Insufficient budget for both the minimum refresh population and the continuation
  reserve: do not trigger.
- Refresh optimizer exception, malformed output, or non-finite candidate: fail
  the run explicitly; do not silently convert an execution error into a fallback.
- Non-improving candidate: retain the guard and resume CC.
- Existing repair lock active: do not trigger.
- Any FE, input-hash, environment, or anti-leakage gate failure: invalidate the
  run and withhold performance claims.

## Verification

### Unit tests

- Trigger on an R3-like runtime state with three shared variables, late remaining
  budget, low gain, and high conflict.
- Reject E3-like productive relations, A4-like five-variable overlap, R2-like
  one-variable overlap, and S3-like repair-locked state.
- Verify refresh cap, continuation reserve, deterministic seed, single-use flag,
  strict acceptance, and exact FE reconciliation.
- Verify that no forbidden offline field can enter the action decision.

### Runtime gates

1. Run R3 seed 3 at 3M FE in the pinned environment.
2. If the action triggers and improves the pinned canonical value, run R3 seeds
   1-3.
3. Run E3, S3, R2, and A4 seeds 1-3 as preservation controls.
4. Continue only if all runs are fresh, same-budget clean, input-hash stable, and
   anti-leakage clean, with no loss of an existing best-of-three paper-best win.
5. Run the complete 13-case, three-seed canonical protocol only after the focused
   preservation gate passes.

The target is 13/13 best-of-three paper-best wins while retaining the existing
12 wins. Three-seed means and seed-level wins remain separately reported; no
25-run or SOTA mean claim is implied.

## Non-Goals

- Reproducing the old-environment `3.214242e5` value by changing dependencies.
- Dispatching by R3, Rastrigin, paper-best thresholds, or historical outcomes.
- Running multiple action lanes and selecting the best final result.
- Retuning dense overlap or the non-dense repair lock in this change.
- Claiming a 25-run mean from a three-seed pilot.
