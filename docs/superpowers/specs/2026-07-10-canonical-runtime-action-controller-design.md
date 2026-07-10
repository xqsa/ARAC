# Canonical Runtime Action Controller Design

Date: 2026-07-10
Executor: Codex
Status: proposed

## Objective

Provide one auditable 3M-FE experiment entry that runs one optimizer trajectory per
case and maps Phase-I runtime evidence to Phase-II actions. The entry must preserve
the currently reproducible paper-best wins before attempting to recover the full
13-case historical win set.

The runtime must not use case labels, function-family labels, paper values,
historical final errors, oracle lanes, or relative gains for dispatch.

## Evidence Behind The Design

- E6 seed 3 with adaptive v26 reproducibly reaches `2.509370e7`, below the
  paper-best value `2.62e7`.
- S6 seed 2 with adaptive v24 exactly reproduces `7.954037e3`, below the
  paper-best value `1.33e4`.
- Both runs use at most 3M FE and have `same_budget_violation=0`.
- The older E6 value `2.143181e7` cannot be reproduced from the frozen runner and
  policy hashes alone, so it is not a valid implementation anchor.
- The current runner resolves AOB data through `Path.cwd()`. The F6 design under
  `E:\HCC-main` differs from the ARAC/canonical F6 design, proving that cwd is an
  untracked experimental input.

## Considered Approaches

### A. Canonical runtime profile in the current runner (recommended)

Add an explicit data-root boundary and a single evidence-controller runtime
profile to the current runner. Reuse the existing optimizer, v24/v26 relation
policies, incumbent protection, FE ledger, and audit artifacts.

This keeps one maintained implementation and makes all runtime inputs explicit.

### B. Execute a frozen historical runner as a permanent backend

This would reproduce selected historical paths quickly, but it would create a
second optimizer implementation and preserve a large legacy snapshot as production
code. It conflicts with ARAC's clean-package and single-fact-source goals.

### C. Continue tuning the current v31 selector

This is premature while cwd, data hashes, numerical environment, and runtime
profile are not frozen. Further threshold tuning would mix protocol drift with
controller quality.

## Runtime Architecture

### Explicit benchmark binding

`HccAobExecutionRequest` gains an explicit `aob_data_root`. The backend passes it
to the smoke runner through `--aob-data-root`. The runner validates the required
`F1`-`F6` metadata files and never derives benchmark data from cwd.

The canonical default is the ARAC-owned AOB data directory. A different root is
allowed only when explicitly supplied and is recorded in the manifest with file
hashes.

### Single evidence-driven controller

The final entry runs one lane per case/seed. Phase I produces the incumbent and
runtime evidence. The first implementation reuses the existing v31 runtime-prefix
lock (`select_evidence_action_controller_v31_dense_lock_mode`) without retuning
its thresholds. The controller early-locks one relation behavior:

- adaptive v24 for relation-first coordination and repair;
- adaptive v26 when the runtime prefix supports its broader intervention mode.

The decision may use overlap degree, relation-prefix density, shared-variable
support, contribution imbalance, conflict signals, action confidence, stagnation,
and remaining FE. It may not inspect the case id, family, paper table, historical
outcome, or final error.

The controller retains the Phase-I incumbent. Phase-II candidates are accepted
only when they improve the incumbent, so controller intervention cannot replace a
better known state with a worse one.

### Reproducibility manifest

Each run records:

- runner, policy, experiment-entry, optimizer-module, and AOB data hashes;
- Python, NumPy, SciPy, Torch, and BLAS versions;
- BLAS/OMP/MKL thread variables and worker count;
- explicit data root and actual cwd;
- per-stage FE totals and overhead FE;
- selected runtime mode, lock evidence, action trace, and incumbent before/after.

Paper values remain offline comparison inputs written only after optimizer output
exists.

## Stable Interfaces

- `HccAobExecutionRequest.aob_data_root: Path`
- runner CLI `--aob-data-root <path>`
- final protocol profile `canonical_evidence_controller_v1`
- one selected result row per case/seed
- existing `action_trace.csv`, `same_budget_ledger.csv`,
  `anti_leakage_audit.csv`, and `run_manifest.md` remain the primary artifacts

No new parallel final runner or second result schema is introduced.

## Failure Handling

- Missing or incomplete data root: fail before optimizer execution.
- Data hash changes within one run: fail the protocol audit.
- FE requests are bounded before optimizer calls. Any remaining overrun or
  unexplained stage-total mismatch marks the run invalid.
- Forbidden selector field detected: fail anti-leakage audit and withhold the
  selected result.
- Phase-II candidate is worse: retain the incumbent and record a rejected action.

## Verification

### Unit and integration tests

- Command construction includes the explicit data root.
- Changing cwd does not change the resolved AOB inputs.
- Invalid data roots fail before subprocess execution.
- Controller decisions depend only on allowed runtime evidence.
- Paper, case, family, historical, and final-outcome fields cannot enter dispatch.
- No-harm acceptance preserves a better incumbent.
- FE stage totals reconcile with the fitness-record total.

### Runtime gates

1. Small-FE smoke from both `E:\ARAC` and `E:\HCC-main` produces identical input
   hashes and deterministic trace prefixes.
2. E6 seed 3 must remain below `2.62e7` at 3M FE.
3. S6 seed 2 must remain below `1.33e4` at 3M FE.
4. Both anchors must have `same_budget_violation=0` and pass anti-leakage audit.
5. Only after these gates pass, run the 13 target cases at three seeds and compare
   against the current reproducible baseline. Catastrophic loss remains a hard
   blocker.

## Non-Goals

- Reproducing the unfrozen E6 value `2.143181e7`.
- Running several action lanes and choosing the best final outcome.
- Claiming a 25-run mean from historical or pilot results.
- Tuning dispatch with paper-best or case-specific rules.

## Implementation Order

1. Add failing tests for explicit data-root propagation and cwd independence.
2. Implement the data-root boundary and hash audit.
3. Add failing tests for the single evidence-controller profile and no-harm gate.
4. Wire the existing v24/v26 behaviors into the single controller path.
5. Run focused tests, small-FE cwd equivalence, then the two 3M-FE anchor gates.
6. Proceed to the 13-case three-seed regression only if both anchors pass.
