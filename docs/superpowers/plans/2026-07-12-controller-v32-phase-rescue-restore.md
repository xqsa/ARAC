# Controller V3.2 Phase Rescue Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the runtime-evidence-triggered group-local phase rescue that previously produced the R2 canonical win, while preventing the incompatible stale Phase-I state resume from consuming canonical FE.

**Architecture:** Keep relation dispatch, repair lock, incumbent protection, and FE accounting unchanged. Add a v3.2 controller profile whose search-state action is the existing group-local `phase_rescue_multistart`; complete-sweep Phase-I resume remains implemented for audit/ablation but is not bound to the canonical v3.2 route.

**Tech Stack:** Python, pytest, HCC AOB runner, CMA-ES/MMES backends.

---

### Task 1: Lock The V3.2 Action Contract

**Files:**
- Modify: `tests/test_hcc_smoke_runner_cli.py`
- Modify: `tests/test_hcc_action_execution_plan.py`
- Modify: `tests/test_exp_003_hcc_runtime_consumer_smoke.py`

- [x] Add a failing test proving v3.2 enables group-local phase rescue when runtime evidence allows it.
- [x] Add a failing test proving v3.2 does not bind `resume_phase_i_search_state` as its canonical search-state hook.
- [x] Add a failing test proving the canonical lane maps to one v3.2 trajectory rather than an outcome-selected portfolio.
- [x] Run the three focused tests and confirm they fail because v3.2 is not implemented.

### Task 2: Implement The Minimal V3.2 Route

**Files:**
- Modify: `src/arac/action_space.py`
- Modify: `src/arac/backends/hcc.py`
- Modify: `HCC_SRC/arac_hcc_smoke_runner.py`
- Modify: `experiments/exp_003_hcc_runtime_consumer_smoke/run.py`
- Modify: `experiments/exp_005_hcc_final_protocol_pilot/run.py`
- Modify: `experiments/exp_005_hcc_final_protocol_pilot/README.md`

- [x] Register `arac_evidence_action_controller_v32` as a trajectory action.
- [x] Bind v3.2 to relation-first/repair-lock semantics plus `phase_rescue_multistart` only.
- [x] Reuse the existing group stagnation and rescue acceptance logic; do not add case labels, paper values, or historical outcomes to runtime inputs.
- [x] Point the canonical experiment profile to the single v3.2 trajectory.
- [x] Run the focused tests and confirm they pass.

### Task 3: Verify Regression Gates

**Files:**
- Modify only if a test exposes a contract mismatch.

- [x] Run search-state, runner CLI, backend adapter, execution-plan, exp003, and exp005 focused tests.
- [x] Run the full pytest suite.
- [x] Run `git diff --check` and inspect `git status --short`.

### Task 4: Run The Preservation Pilot

**Files:**
- Generate: `results/exp_009_controller_v32_controls_seed123_3m/`

- [x] Run `R2 seed3` at 3M FE as the first reproduction gate.
- [x] If R2 seed3 is below `2.48e5`, run `R2/E6/S6/A4`, seeds 1-3, at 3M FE.
- [x] Require best-of-three wins for all four cases before expanding the protocol.
- [x] Audit fresh execution, same-budget FE, unchanged AOB inputs, anti-leakage, and action traces.

### Task 5: Commit The Verified Version

**Files:**
- Stage only source, tests, and protocol documentation; exclude `results/`.

- [x] Review the final diff for duplicate routes, hidden fallback, and leakage.
- [ ] Commit with a message describing the v3.2 runtime action restoration.
- [ ] Report the commit and pilot evidence; request confirmation before any push.
