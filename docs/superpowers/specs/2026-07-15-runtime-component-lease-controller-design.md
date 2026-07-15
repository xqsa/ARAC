# Runtime Component-Lease Controller Design

Date: 2026-07-15
Author: Codex
Status: frozen before v41 implementation and optimizer FE

## Decision

S22 passed the preregistered scheduler-feasibility gate on fresh A4/S2/E2
seeds 34/35/36. Offline replay selected 111 precision actions in six runs and
three cases. All 111 reached the next canonical group revisit, with zero
component overlap, zero deterministic-cap underprediction, and zero cap
contract drift. A4 and S2 each covered two or more selected seeds. The input
matrix also passed 9/9 integrity, 3/3 v38 parity, AOB immutability, strict FE,
and anti-leakage checks.

This result authorizes exactly one runtime pilot. v41 may serialize the
existing post-retirement precision action with a component mutex and the S22
deterministic scheduler revisit cap. It does not authorize a new action dose,
an outcome-trained threshold, or a full-24 run.

## Runtime Contract

The v41 precision action is eligible only when all conditions hold before the
current group optimizer starts:

1. the inherited v38 post-retirement precision route is active;
2. the deterministic scheduler revisit cap is valid and reachable; and
3. no earlier v41 precision lease is pending in the same overlap-connected
   component.

An eligible action opens one component lease. The lease closes only at the
next canonical revisit of the action's own group. Other groups in the same
component must abstain while the lease is open. An unreachable cap or an open
component lease selects the inherited conservative v38 fallback sigma, not a
new optimizer state.

The decision function may consume only current-run grouping topology, pending
lease identity, sweep-start FE, decision FE, strict CC budget limit, current
group index, current uniform group budget, current optimizer budget, and
ordered group population sizes. It must not consume problem, case, seed,
function family, paper-best, historical/final outcome, current action FE,
resolution, gain, overwrite, survival, win, or catastrophic labels.

The runtime and offline audit must call the same pure lease-eligibility
function. v40 remains trace-only and behavior-identical to v38.

## Trace Contract

Every v41 post-retirement eligibility attempt records:

- component identity and pending lease identity before dispatch;
- exact scheduler-cap inputs and result;
- `component_lease_decision` as `selected` or `abstained`;
- a closed abstention reason from `abstain_scheduler_unreachable` and
  `abstain_component_mutex`;
- whether the precision action was consumed by the optimizer.

Only selected attempts register delayed credit. Abstained attempts must not
create pending actions, consume the precision sigma, change candidates, alter
RNG, or consume additional FE.

## Staged Verification

Before fresh 3M FE:

1. TDD the shared eligibility function, component release/mutex behavior,
   runtime sigma fallback, and trace fields.
2. Run focused and full tests, direct CLI help, and diff/compile checks.
3. Run matched v38/v41 E2 seed1 at 5k. Final error, FE, AOB inputs, and every
   common trace field must be identical; anti-leakage and strict FE must pass.
4. Commit and push the implementation and audit before the pilot.

## Frozen Pilot

Run matched fresh v38 and v41 arms on A4/S2/E2, seeds 37/38/39, strict
3,000,000 FE, with pinned hash and numerical-library thread settings. The raw
directories are:

- `results/controller_v41_runtime_lease_heldout_seed37_39_3m_20260715`;
- `results/controller_v41_runtime_lease_parity_v38_seed37_39_3m_20260715`.

Offline paired performance uses
`log(max(v38_error, 1e-300) / max(v41_error, 1e-300))`; positive values favor
v41. A catastrophic loss is the existing relative gain `<= -20%`. Final
errors and these labels are offline audit fields only.

The pilot passes only if all conditions hold:

- 9/9 fresh runs per arm complete with no FE overspend, unchanged AOB inputs,
  and anti-leakage pass;
- every serialized cap exactly recomputes from action-time inputs;
- no eligibility result changes when current-action outcome fields are
  mutated offline;
- at least six v41 runs and all three cases select a lease, with at least two
  cases selecting leases in two seeds;
- every selected lease resolves, with zero component overlap and zero cap
  underprediction;
- at least three paired runs change, catastrophic losses are zero, mean paired
  log advantage is strictly positive, median paired log advantage is
  non-negative, and changed-run wins are not fewer than changed-run losses.

Any failure permanently stops the component-lease controller route. The gate,
thresholds, cases, seeds, and baseline may not be retuned. Passing permits a
separate full-24 preregistration; it is not itself a final performance claim.
