# Evidence-Guided Diagonal Search-State Action Design

Date: 2026-07-12
Executor: Codex
Status: approved for implementation

## Purpose

Add a mature diagonal CMA-ES executor to the existing grouping-to-action
controller. The controller remains responsible for deciding whether a
`trajectory` intervention is justified; diagonal CMA-ES is only the executor
of that action.

This closes the current implementation gap: the repository's direct
separable-CMA path is a hand-written diagonal proxy, while the canonical
search-state action only resumes Phase-I MMES state.

## Method Boundary

The runtime chain remains:

```text
Phase-I grouping and overlap evidence
  -> complete CC sweep evidence snapshot
  -> reference-blind trajectory-action decision
  -> bounded diagonal CMA-ES block
  -> strict incumbent acceptance and FE audit
  -> continue the same optimizer trajectory
```

The action must not use case identifiers, function-family labels, paper-best
values, historical outcomes, final error, or relative gain. It must not run a
second complete lane and select by final result.

## Backend

Use `cma==4.4.4` with:

- `CMA_diagonal=True` for linear-memory diagonal covariance adaptation;
- `ask()` / `tell()` blocks containing complete populations;
- explicit bounds and a deterministic local NumPy generator;
- a protected incumbent supplied by the controller;
- no internal paper or historical comparison.

The backend state is in-memory and resumable within one canonical execution.
It records the strategy, local RNG, protected best candidate, FE count, and a
fingerprint over mean, sigma, diagonal scaling, evolution paths, iteration/FE
counters, incumbent, and RNG state. Malformed dimensions, non-finite values,
or FE overrun fail explicitly.

## Controller Integration

The existing `SearchStateEvidence` and scheduler remain the policy boundary.
`plan_search_state_action` accepts an allowed trajectory action name so the
same reference-blind state machine can dispatch either:

- `resume_phase_i_search_state`; or
- `continue_diagonal_search_state`.

The experiment config selects which executor is available; runtime evidence
decides whether it executes. This is an optimizer-backend ablation, not a
case-specific route.

For the diagonal executor:

1. The first accepted plan initializes state at the current protected
   incumbent.
2. Probe and confirmation blocks continue the same diagonal state.
3. A CC improvement between blocks updates the protected incumbent but does
   not silently re-anchor the diagonal distribution.
4. Only a strict candidate improvement replaces the global incumbent.
5. A rejected block still consumes and reports actual FE.

The existing policy caps remain unchanged: 1% FE per block, 15% cumulative
state-action FE, 10% canonical CC reserve, and the 1.5x/2.0x utility gate.
Because one complete CC sweep otherwise consumes nearly all remaining FE, the
runner reserves the 15% intervention cap plus the 10% CC reserve before the
first diagonal decision. A failed or ineligible probe blocks further diagonal
spending and immediately returns the unused reserve to canonical CC.

An active shared-variable repair lock continues to block MMES state resumption.
It does not block the full-space diagonal action when the same current-run
conflict evidence supports the action: diagonal search does not write a group
owner directly, starts from a bounded projection of the protected incumbent,
and can update the incumbent only by strict global improvement.

## Configuration And Audit

Add `search_state_backend` with exactly two values:

```text
phase_i_mmes
diagonal_cma
```

The default remains `phase_i_mmes` so the frozen v32 baseline is reproducible.
The diagonal candidate uses the same canonical experiment entry with an
explicit backend option.

Each action trace records backend name, optimizer seed, block FE, utility,
state fingerprint before/after, incumbent before/candidate/after, acceptance,
CC reserve, and cumulative intervention FE. The run manifest records the
`cma` version and module hash.

## Verification Gates

Code gates:

1. Same seed and inputs produce identical diagonal blocks.
2. Every block uses complete populations and never exceeds requested FE.
3. State fingerprints change after an evaluated block and remain unchanged
   after a zero-budget block.
4. Worse candidates cannot replace the protected incumbent.
5. Existing MMES search-state behavior is unchanged when the default backend
   is used.
6. Runtime policy dataclasses remain free of forbidden offline fields.

Runtime gates:

1. Run a small deterministic backend smoke test.
2. Run R3 at 3M FE for seeds 1, 2, and 3 using one diagonal-enabled canonical
   trajectory per seed.
3. If R3 shows useful action reachability, run E6, S6, R2, and A4 preservation
   controls.
4. Only then rerun all existing twelve winning cases.

Adoption requires preserving the existing 12/12 best-of-three wins, no FE
overrun, clean anti-leakage, and no catastrophic loss. R3 improvement is
reported separately and cannot compensate for a lost protected win.
