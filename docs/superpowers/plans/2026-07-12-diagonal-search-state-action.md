# Diagonal Search-State Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, resumable `pycma` diagonal CMA-ES executor to the existing single-track evidence-guided trajectory action.

**Architecture:** Keep the existing search-state scheduler and canonical experiment entry. Add one focused backend wrapper, let the scheduler emit an explicitly configured trajectory action name, and pass the backend choice through the existing HCC request and runner. Preserve the MMES path as the default and audit both executors through the same FE ledger and trace schema.

**Tech Stack:** Python 3.12, NumPy 2.3.5, cma 4.4.4, pytest, existing HCC/AOB runner.

**Implementation status (2026-07-13):** Tasks 1-3 are implemented. Task 4
found that the 25% initial hold altered canonical CC allocation and failed the
E6/S6/R2 preservation gate. Task 5 corrects the hold before repeating the
runtime gates.

---

## File Map

- Create `src/arac/backends/diagonal_cma.py`: pycma state, deterministic RNG,
  fingerprint, initialization, and complete-population block execution.
- Create `tests/test_diagonal_cma.py`: backend determinism, limits, state, and
  incumbent protection tests.
- Modify `pyproject.toml`: pin `cma==4.4.4` in the HCC optional dependency set.
- Modify `src/arac/policy/search_state_policy.py`: allow a validated trajectory
  action name without adding offline evidence.
- Modify `src/arac/actions/contracts.py`: register
  `continue_diagonal_search_state` as a trajectory core intervention.
- Modify `scripts/hcc_smoke_runner.py`: initialize/continue diagonal state at
  existing complete-sweep decision points and extend trace fields.
- Modify `src/arac/backends/hcc.py`: pass `search_state_backend` through the
  subprocess boundary.
- Modify `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py` and
  `experiments/final/exp_005_hcc_final_protocol_pilot/run.py`: expose the
  backend option without adding lanes.
- Modify focused tests for policy, runner, adapter, exp003, and exp005.

## Task 1: Build The Diagonal Backend With TDD

- [ ] Add failing tests proving that `DiagonalCMAState` and
  `run_diagonal_cma_block` do not yet exist.
- [ ] Run `E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_diagonal_cma.py -q`
  and verify the missing-module failure.
- [ ] Implement state initialization with `CMA_diagonal=True`, explicit bounds,
  deterministic local RNG, protected incumbent, and a stable fingerprint.
- [ ] Add failing tests for complete-population rounding, zero-budget no-op,
  deterministic same-seed execution, malformed dimensions, and non-finite
  objective output.
- [ ] Implement minimal `ask()` / batch objective / `tell()` block execution;
  report requested FE, actual FE, unused FE, best before/after, candidate best,
  acceptance, sigma, population, and fingerprints.
- [ ] Run the focused backend tests and commit.

## Task 2: Extend The Pure Policy Without Leakage

- [ ] Add a failing policy test that requests
  `continue_diagonal_search_state` and verifies the emitted action name.
- [ ] Add a failing test that an unknown action name raises `ValueError`.
- [ ] Update `plan_search_state_action` with a keyword-only
  `trajectory_action_name`, defaulting to `resume_phase_i_search_state`.
- [ ] Register the diagonal action as `ActionFamily.TRAJECTORY` with
  `core_intervention` backend role.
- [ ] Run `tests/test_search_state_policy.py` and package-boundary tests.

## Task 3: Integrate The Executor Into The Single Runner Trajectory

- [ ] Add failing CLI and command-builder tests for
  `--search-state-backend diagonal_cma` and default `phase_i_mmes`.
- [ ] Add `search_state_backend` to `SmokeConfig` and
  `HccAobExecutionRequest`, validating exactly the two approved values.
- [ ] Add diagonal state to `EvidenceActionControllerV31RunState` without
  removing the existing Phase-I MMES state.
- [ ] At the existing complete-sweep scheduler boundary, choose the configured
  action name, initialize diagonal state from the protected incumbent on first
  use, and continue it on later blocks.
- [ ] Keep strict incumbent acceptance, policy utility transitions, CC reserve,
  and cumulative FE accounting shared across both backends.
- [ ] Add `search_state_backend` to action trace rows and verify that the
  fingerprint and FE fields are populated for diagonal blocks.
- [ ] Replace the direct separable proxy implementation with the same mature
  backend wrapper so no second diagonal implementation remains.
- [ ] Run runner, adapter, exp003, and exp005 focused tests.

## Task 4: Verification And Pilot

- [ ] Run Ruff or syntax compilation for modified modules.
- [ ] Run all focused policy/backend/runner/experiment tests.
- [ ] Run the full pytest suite and `git diff --check`.
- [ ] Run a small-budget R3 smoke execution and verify action reachability,
  exact FE reconciliation, trace fields, and anti-leakage.
- [ ] If the smoke gate passes, run R3 seeds 1/2/3 at 3M FE with the explicit
  diagonal backend and bounded parallelism.
- [ ] Compare the fresh three-seed result to the frozen v32 R3 baseline and
  paper-best offline; do not feed either value into runtime dispatch.
- [ ] Run preservation controls only if R3 action execution is valid and has no
  catastrophic loss.

## Task 5: Correct Initial Search-State Budget Hold

**Files:**

- Modify: `tests/test_hcc_smoke_runner_cli.py`
- Modify: `scripts/hcc_smoke_runner.py`
- Modify: `docs/superpowers/specs/2026-07-12-diagonal-search-state-action-design.md`

- [x] Change the existing hold test to require `330_000` FE at a 3M budget:
  one `30_000` FE executable block plus the `300_000` FE CC reserve.
- [x] Run the focused test and verify that it fails with the current `750_000`
  FE result.
- [x] Change `scheduled_search_state_hold_fes` to use
  `FIRST_PROBE_FRACTION` for every ready scheduler phase. Keep the 15%
  cumulative cap in `plan_search_state_action`; do not use it as a hold.
- [x] Run the focused runner and policy tests, compile the modified modules,
  run the full test suite, and run `git diff --check`.
- [x] Run R3 seeds 1/2/3 at 3M FE with `diagonal_cma`. Compare only offline
  against the frozen v32 baseline and paper-best.
- [x] Run E6/S6/R2/A4 seeds 1/2/3 after the R3 pilot. The adoption gate
  failed: A4 and S6 retained their best-of-three wins, while E6 and R2 did
  not. Do not expand this backend to the twelve protected winners.

**Task 5 runtime result:** R3 improved to `3.340394e5`, which is 1.84% above
the offline paper-best `3.28e5`. Strict FE accounting and anti-leakage passed,
but the backend remains experimental because the preservation gate was only
2/4.

## Task 6: Add Pre-Hold Evidence Audit

**Files:**

- Modify: `src/arac/policy/search_state_policy.py`
- Modify: `scripts/hcc_smoke_runner.py`
- Modify: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`
- Modify: `tests/test_search_state_policy.py`
- Modify: `tests/test_hcc_smoke_runner_cli.py`
- Modify: `tests/test_exp_003_hcc_runtime_consumer_smoke.py`

- [x] Add failing tests for a pure `PreHoldEvidence` snapshot and verify that
  its dataclass excludes all forbidden offline fields.
- [x] Add failing trace tests proving that only the first scheduler decision
  carries the snapshot and that serialization is deterministic.
- [x] Implement scale-free topology and budget-compression calculations
  without changing the scheduler decision or FE ledger.
- [x] Aggregate populated trace snapshots into `pre_hold_evidence.csv` and add
  a focused experiment artifact test.
- [x] Run focused tests, compile modified modules, run the full suite, and
  run `git diff --check`.
- [x] Run the protected 13-case set for one seed at 3M FE with parallel jobs,
  then inspect the audit fields offline before proposing an admission rule.

**Task 6 runtime result:** all 13 seed1 lanes produced pre-hold evidence with
clean FE and anti-leakage audits. E1/R1 had zero overlap edges but still held
330k before this structural gate was added; positive-overlap cases remain
unresolved and no fitted admission threshold is adopted.

## Task 7: Gate Structurally Unreachable Holds

**Files:**

- Modify: `scripts/hcc_smoke_runner.py`
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [x] Add a failing test that zero overlap edges produce zero scheduled hold.
- [x] Pass the Phase-I overlap-edge count into the hold boundary and preserve
  the existing 330k hold for positive-overlap cases.
- [x] Run the full test suite and verify E1/R1 at 3M FE with one seed.

**Task 7 runtime result:** E1/R1 recorded zero hold, zero search-state FE, and
budget-retention ratio 1.0 with clean FE and anti-leakage audits. E1 improved
from `3.106604e6` under the unreachable hold to `2.207863e6`, matching the
canonical v32 seed1 value `2.208219e6` within normal numerical variation. R1
was unchanged at `1.704316e5` in this audit run.

## Task 8: Terminal Diagonal Probe Without CC Reserve

**Files:**

- Modify: `src/arac/policy/search_state_policy.py`
- Modify: `scripts/hcc_smoke_runner.py`
- Modify: `tests/test_search_state_policy.py`
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [x] Add failing tests proving the diagonal runtime holds only 1% FE, emits
  zero CC reserve, and cannot schedule confirmation or expansion.
- [x] Preserve the existing 11% staged hold and state machine for the default
  `phase_i_mmes` backend.
- [x] Execute one diagonal probe only after canonical CC reaches the terminal
  1% window; keep strict incumbent acceptance and complete-population FE
  accounting.
- [x] Run focused tests, the full suite, compilation, and `git diff --check`.
- [x] Run a small-budget R3 smoke and verify one probe, zero CC reserve, exact
  FE reconciliation, populated fingerprints, and clean anti-leakage.
- [x] If the smoke gate passes, run R3 seeds 1/2/3 at 3M FE.
- [x] Run E6/S6/R2/A4 preservation controls because the R3 execution was
  runtime-valid and non-catastrophic relative to frozen canonical v32.

**Adoption gate:** diagonal remains opt-in unless all four preservation
controls retain their frozen best-of-three wins. R3 improvement cannot offset
a lost protected win.

**Task 8 R3 result:** the 5k smoke executed exactly one 50-FE probe with zero
CC reserve. The 3M-FE run executed exactly one 30k-FE probe for each seed with
clean FE and anti-leakage audits. Final R3 errors were `4.505813e5`,
`5.835945e5`, and `3.552121e5`; the best remained above the offline paper-best
`3.28e5`. This is better than the frozen canonical v32 R3 best
`3.974568e5`, but worse than the previous 11% hold pilot best `3.340394e5`.
The protocol is runtime-valid but has not passed the performance gate.
E6/S6/R2/A4 preservation controls remain required before adoption.

**Task 8 preservation result:** all twelve control trajectories used one
terminal probe, zero CC reserve, clean FE accounting, unchanged AOB inputs,
and clean anti-leakage. Best-of-three retained A4 (`7.829752e4 < 7.83e4`)
and E6 (`2.318760e7 < 2.62e7`), but lost R2
(`2.927030e5 > 2.48e5`) and S6 (`1.448072e4 > 1.33e4`). The adoption gate was
only 2/4. Do not expand this backend to the twelve protected winners. The next
design must improve runtime admission evidence rather than reserve another
budget variant.

## Task 9: Corrected Search-State Evidence Shadow Audit

**Files:**

- Modify: `src/arac/policy/search_state_policy.py`
- Modify: `scripts/hcc_smoke_runner.py`
- Modify: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [x] Prove that fallback relations must not count as active interventions.
- [x] Add a scale-free writeback norm based on bounded shared-subspace span.
- [x] Record legacy and corrected evidence together without changing the
  current selector.
- [x] Run a 5k real-HCC smoke and verify populated evidence with clean FE.
- [x] Run E6/S6/R2/R3/A4 seeds 1/2/3 at 3M FE as a shadow audit.

**Task 9 result:** legacy absolute writeback instability was true for 15/15
lanes. Corrected relative instability was true for 8/15 and no longer
degenerate, but its range overlapped useful and harmful cases. Active
intervention fraction excluded R2/R3 but overlapped A4/E6/S6. No corrected
single threshold safely separated the cases, so the selector is unchanged.

The paired trace exposed a stronger mechanism. A4/E6 mostly retained their
results when the diagonal candidate was rejected or tiny, while R2/S6/R3
usually accepted large immediate improvements before later diverging. The
next design should protect an accepted diagonal incumbent while isolating it
from immediate replacement of the canonical CC context.
