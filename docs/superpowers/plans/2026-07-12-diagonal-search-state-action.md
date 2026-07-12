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
