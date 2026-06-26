# AOB 1-Run Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin `exp_002_aob_1run_pilot` experiment that runs one AOB pilot pass over 24 cases, emits pilot truth tables, and keeps paper-reported baselines evaluation-only.

**Architecture:** Reuse the existing ARAC runtime boundary, HCC backbone adapter, policy, audit, and same-budget ledger helpers. Add one focused experiment module with its own tests, README, and expected outputs so the AOB pilot stays separate from `exp_001_schema_smoke`. The runner should write pilot artifacts under `results/exp_002_aob_1run_pilot/` and never route paper-reported baseline values into runtime dispatch.

**Tech Stack:** Python 3.11, `pytest`, standard library CSV/Path utilities, existing `arac` package modules.

---

### Task 1: Add a failing contract test for the AOB pilot experiment

**Files:**
- Create: `tests/test_exp_002_aob_1run_pilot.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from experiments.exp_002_aob_1run_pilot.run import run_aob_1run_pilot


def test_aob_pilot_writes_one_run_truth_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / "pilot"
    run_aob_1run_pilot(output_dir)

    expected = {
        "pilot_run_manifest.md",
        "our_result_by_case.csv",
        "same_budget_ledger.csv",
        "backend_semantics_diff.csv",
        "anti_leakage_audit.csv",
        "paper_reported_comparison.csv",
        "negative_control_audit.csv",
        "catastrophic_loss_audit.csv",
    }
    assert expected == {path.name for path in output_dir.iterdir()}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_exp_002_aob_1run_pilot.py -q`

Expected: FAIL with `ModuleNotFoundError` because the new experiment module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def run_aob_1run_pilot(output_dir: Path) -> Path:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_exp_002_aob_1run_pilot.py -q`

Expected: PASS and the required files appear in the temp directory.

- [ ] **Step 5: Commit**

```bash
git add tests/test_exp_002_aob_1run_pilot.py experiments/exp_002_aob_1run_pilot
git commit -m "feat: add AOB 1-run pilot experiment"
```

### Task 2: Implement the thin AOB pilot runner

**Files:**
- Create: `experiments/exp_002_aob_1run_pilot/__init__.py`
- Create: `experiments/exp_002_aob_1run_pilot/run.py`
- Create: `experiments/exp_002_aob_1run_pilot/README.md`
- Create: `experiments/exp_002_aob_1run_pilot/expected_outputs.md`
- Modify: `experiments/README.md`

- [ ] **Step 1: Write the failing test**

```python
from experiments.exp_002_aob_1run_pilot.run import run_aob_1run_pilot


def test_aob_pilot_marks_oracle_and_reported_baselines_offline_only(tmp_path):
    output_dir = run_aob_1run_pilot(tmp_path / "pilot")
    comparison = (output_dir / "paper_reported_comparison.csv").read_text(encoding="utf-8")
    manifest = (output_dir / "pilot_run_manifest.md").read_text(encoding="utf-8")

    assert "paper-reported evaluation-only baselines" in comparison
    assert "must not enter runtime dispatch" in manifest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_exp_002_aob_1run_pilot.py -q`

Expected: FAIL because the module and files are still missing.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import csv
from pathlib import Path

from arac.backends.hcc import HccBackboneSnapshot, HccGroupSignal, build_hcc_evidence_profile, hcc_backend_semantics_for
from arac.evaluation import SameBudgetLedger, classify_utility, relative_gain
from arac.audit import claim_gate
from arac.policy import ActionDecision, decide_action


def run_aob_1run_pilot(output_dir: Path) -> Path:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_exp_002_aob_1run_pilot.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp_002_aob_1run_pilot tests/test_exp_002_aob_1run_pilot.py experiments/README.md
git commit -m "feat: implement AOB 1-run pilot runner"
```

### Task 3: Verify the full local smoke surface and clean generated artifacts

**Files:**
- Modify: `README.md` only if run instructions need a brief mention of the new experiment

- [ ] **Step 1: Run the focused tests**

Run: `py -m pytest tests/test_exp_001_schema_smoke.py tests/test_aob_pilot_protocol.py tests/test_exp_002_aob_1run_pilot.py -q`

Expected: PASS.

- [ ] **Step 2: Inspect generated outputs**

Confirm the `results/exp_002_aob_1run_pilot/` directory contains the eight pilot truth artifacts and that the CSV rows cover all 24 AOB cases.

- [ ] **Step 3: Remove transient caches if created**

Delete `__pycache__` and `.pytest_cache` entries generated by the verification run.

- [ ] **Step 4: Final review and handoff**

Summarize the new experiment, the pilot claim boundary, and the remaining gap to the 25-run final protocol.
