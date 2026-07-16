# Component Precision Action Validity v1

Date: 2026-07-16
Executor: Codex
Status: frozen before optimizer execution

## Question And Boundary

`component-precision-action-validity-v1` asks one question before any further
scheduler work:

> Does one complete component-wide precision action have a positive component
> endpoint and terminal effect relative to v37?

This is a strict two-arm action-validity experiment:

- `a0_v37`: unmodified v37;
- `a1_precision_component_once`: at the first safe pre-action opportunity,
  every group in the selected overlap-connected component uses the existing
  v38 precision sigma (`0.5 * v37 sigma`) for one complete component horizon.

There is no paired point probe, response gate, group mask, learned model,
repeat lease, or runtime selector in this protocol. The old
`precision-response-loop-v1` remains frozen and is not reused. Case, seed,
function family, paper values, historical/final outcomes, and survival cannot
enter the opportunity rule.

## Component-Horizon Action

The opportunity is the first safe overlap-component opportunity defined using
only pre-action topology, lock, scheduler, and remaining-FE state. At that
point the component, canonical group order, group population sizes, native
sigmas, FE caps, RNG descriptor, and complete checkpoint are frozen.

A component horizon visits every group in the component once in canonical
order. The A1 dose applies to all component groups, not only the triggering
group. At component start, both arms freeze the component, group order,
population sizes, requested group budgets, and CRN descriptor. A1 additionally
freezes the component-wide precision dose. Its fresh-subprocess trajectory
then executes the component groups sequentially, with the normal
within-trajectory incumbent and optimizer-state updates. Precision cannot be
reconsidered or redispatched between groups. Exactly one `H` endpoint is
recorded after the final component group, then execution returns permanently
to v37. A trajectory may act once.

Natural CMA early stopping is a treatment-path mediator. Every group must
consume a positive integer number of complete populations, but its actual FE
may be smaller than its frozen requested FE and may differ between A0 and A1.
The audit records this difference without padding either arm. Component
endpoint sequence identity is required; equal component-endpoint FE is not.
Both arms must still reach the same registered absolute terminal FE.

Here `atomic` describes the scheduling unit, not a transactional full-state
clone or commit. A0 remains the unmodified v37 main loop. A pair is invalid if
the component plan drifts mid-horizon, either arm misses a group in the frozen
component sequence, an actual group budget is not generation-complete, more
than one `H` endpoint is emitted, or the arms do not reach the same terminal
absolute-FE endpoint.

## Population And Estimands

Each registered case-seed contributes one pair. ITT contains the complete
registered matrix; a trajectory with no pre-action opportunity contributes
`tau_H=tau_T=0` and must be bit-equivalent across arms. ATT is fixed before
outcomes and contains exactly the pre-action applicable pairs.

The audit recomputes:

```text
tau_H = log(max(error_A0_component, 1e-300)
            / max(error_A1_component, 1e-300))
tau_T = log(max(error_A0_terminal, 1e-300)
            / max(error_A1_terminal, 1e-300)).
```

Positive values favor the component precision action. A catastrophic event is
A1 error at least `1.2 * A0 error`. A terminal material effect is
`tau_T >= log(1.01)`.

`S_H` is shared-variable survival at the component endpoint and `S_D` is
survival at the registered delayed closure. Their paired effects are
`delta_S_H=S_H_A1-S_H_A0` and `delta_S_D=S_D_A1-S_D_A0`. Survival is an
offline outcome only. `strict_survival` means that A1 retains a non-zero
shared-variable displacement at `H`; confirm requires it in at least 50% of
ATT pairs. It cannot affect applicability or action execution.

All confidence bounds use 2,000 case-by-seed two-way cluster bootstrap
resamples. Catastrophic risk uses a one-sided 95% exact Clopper-Pearson upper
bound. Rows are never treated as independent group/relation samples.

## Frozen Matrices

Screen:

- cases: `A4/A5/E1/E2/E3/E4/S2/S5`;
- seeds: `65/66/67/68/69`;
- two arms, strict 3M FE, 40 registered pairs and 80 fresh trajectories.

Confirm, only after screen passes:

- all `E1-E6/S1-S6/R1-R6/A1-A6`;
- seeds `70-77`;
- two arms, strict 3M FE, 192 registered pairs and 384 fresh trajectories.

No case or seed may be substituted after outcomes are observed.

## Five Input CSVs

The offline audit consumes exactly:

1. `component_action_branch_manifest.csv` - two fresh branches, provenance,
   prefix/checkpoint/action-plan hashes, frozen requested and observed actual
   component FE, endpoint evidence, and errors;
2. `component_endpoint_outcomes.csv` - paired component endpoint and `tau_H`;
3. `component_shared_survival.csv` - paired `S_H`, `S_D`, closure, and deltas;
4. `component_action_pairs.csv` - registered applicability, pair integrity,
   terminal outcome, `tau_T`, catastrophe, and no-op parity;
5. `component_budget_ledger.csv` - strict FE, AOB, leakage, component action FE,
   and endpoint closure for both arms.

Every branch row binds the Git commit plus the SHA-256 of this spec and
`configs/component_precision_action_validity_v1.json`. The auditor recomputes
all effects, deltas, catastrophe flags, and matrix membership. The planned
output is `component_action_gate.json`.

## Screen Gate

All conditions are mandatory:

- 100% fresh, FE, AOB, leakage, provenance, prefix, pair, frozen action-plan,
  no-mid-horizon-redispatch, unique component-endpoint, and delayed-closure
  integrity;
- at least 30 applicable pairs over at least six cases and all five seeds;
- ITT and ATT point means for both `tau_T` and `tau_H` are positive, and all
  four medians are non-negative;
- at least three of five seed-level ITT `tau_T` means are positive;
- at least ten pairs have terminal `tau_T >= log(1.01)`;
- zero component or terminal catastrophic events;
- component and delayed closure are 100%;
- paired `delta_S_H` and `delta_S_D` means and medians are non-negative.

Screen failure is a no-go. Confirm must not start.

## Confirm Gate

All conditions are mandatory:

- the same 100% integrity requirements;
- at least 59 applicable pairs over at least 12 cases and all eight seeds;
- terminal ITT LCB, terminal ATT LCB, and component ATT LCB are positive;
- terminal ITT/ATT and component ATT medians are non-negative;
- zero component or terminal catastrophic events, with ATT one-sided 95%
  Clopper-Pearson UCB at most 5%;
- at least 13 of 24 case-level ITT terminal means are positive;
- all eight seed-level ITT terminal means are non-negative and at least six
  are strictly positive;
- worst-10% ATT terminal CVaR is non-negative;
- material terminal effects cover at least 20% of applicable pairs, eight
  cases, and six seeds;
- `delta_S_H` and `delta_S_D` LCBs and medians are non-negative, and each has
  a strictly positive fraction of at least 50%;
- no case contributes more than 50% of absolute ATT terminal effect;
- the mean ITT terminal direction over the 16 non-screen cases is positive.

Passing confirm establishes only action validity. This protocol and its audit
always emit `runtime_scheduler_authorized=false` and
`full_24_authorized=false`. A runtime policy would require a new protocol.
