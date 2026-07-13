# Binary LSGO Focused 3-Seed Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent `exp_010` entry that runs the five fixed binary LSGO cases over three fixed optimizer seeds and evaluates the pre-registered promotion gates.

**Architecture:** The experiment imports the existing deterministic benchmark and native binary ARAC backend, executes a fixed 45-run case/seed/lane matrix, and performs all gain and gate calculations offline after execution. It writes one per-run truth table, one case summary, a structured promotion gate, and a reproducibility manifest without changing `exp_009`, the backend, or policy.

**Tech Stack:** Python 3.11 standard library (`argparse`, `csv`, `dataclasses`, `hashlib`, `json`, `statistics`), existing ARAC binary benchmark/backend/evaluation modules, pytest.

---

## File map

- Create `experiments/exp_010_binary_lsgo_focused_3seed/__init__.py`: experiment package marker.
- Create `experiments/exp_010_binary_lsgo_focused_3seed/run.py`: fixed protocol, 45-run execution, aggregation, gates, artifacts, and CLI.
- Create `tests/test_exp_010_binary_lsgo_focused_3seed.py`: protocol, aggregation, gate, determinism, and CLI tests.
- Modify `experiments/README.md`: add the canonical command and claim boundary.
- Modify `README.md`: add the focused pilot to the experiment inventory.
- Modify `docs/superpowers/specs/2026-07-13-binary-lsgo-focused-3seed-pilot-design.md`: mark the approved design as implemented after verification.

### Task 1: Establish the fixed 45-run protocol contract

**Files:**
- Create: `experiments/exp_010_binary_lsgo_focused_3seed/__init__.py`
- Create: `experiments/exp_010_binary_lsgo_focused_3seed/run.py`
- Create: `tests/test_exp_010_binary_lsgo_focused_3seed.py`

- [ ] **Step 1: Write the failing fixed-protocol test**

Create `tests/test_exp_010_binary_lsgo_focused_3seed.py`:

```python
import csv
import json
from pathlib import Path

from experiments.exp_010_binary_lsgo_focused_3seed.run import (
    FOCUSED_PROBLEM_IDS,
    LANES,
    OPTIMIZER_SEEDS,
    run_focused_pilot,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_fixed_protocol_writes_45_same_budget_rows(tmp_path: Path):
    output = run_focused_pilot(tmp_path / "pilot", total_fes=40)
    rows = read_csv(output / "run_results.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == 45
    assert tuple(FOCUSED_PROBLEM_IDS) == (
        "BLSGO-F07", "BLSGO-F08", "BLSGO-F09", "BLSGO-F14", "BLSGO-F15"
    )
    assert tuple(OPTIMIZER_SEEDS) == (20260713, 20260714, 20260715)
    assert tuple(LANES) == (
        "native_baseline", "arac_policy", "shuffled_evidence_negative_control"
    )
    assert {(row["problem_id"], int(row["optimizer_seed"])) for row in rows} == {
        (problem_id, seed)
        for problem_id in FOCUSED_PROBLEM_IDS
        for seed in OPTIMIZER_SEEDS
    }
    assert {row["total_fe"] for row in rows} == {"40"}
    assert {row["same_budget_violation"] for row in rows} == {"0"}
    assert manifest["execution_count"] == 45
```

- [ ] **Step 2: Run the test and verify the missing package failure**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_010_binary_lsgo_focused_3seed.py -q
```

Expected: collection fails with `ModuleNotFoundError` for
`experiments.exp_010_binary_lsgo_focused_3seed`.

- [ ] **Step 3: Add fixed constants and one typed execution record**

Create `run.py` with these public constants and record contract:

```python
RUN_ID = "exp_010_binary_lsgo_focused_3seed"
FOCUSED_PROBLEM_IDS = (
    "BLSGO-F07", "BLSGO-F08", "BLSGO-F09", "BLSGO-F14", "BLSGO-F15"
)
TARGET_PROBLEM_IDS = ("BLSGO-F08", "BLSGO-F15")
OPTIMIZER_SEEDS = (20_260_713, 20_260_714, 20_260_715)
LANES = (
    "native_baseline", "arac_policy", "shuffled_evidence_negative_control"
)
CANONICAL_TOTAL_FES = 2_000
PHASE_ONE_FRACTION = 0.20


@dataclass(frozen=True)
class FocusedRunRecord:
    result: BinaryLsgoExecutionResult
    input_hash: str
    offline_gain_vs_native: float
    utility_label_vs_native: str
    forbidden_runtime_fields: tuple[str, ...]
    negative_evidence_changed: bool
    claim_allowed: bool = False
```

Validate that every fixed problem ID exists in `standard_binary_lsgo_specs()` before execution.

- [ ] **Step 4: Implement the fixed execution matrix**

Implement:

```python
def execute_focused_matrix(total_fes: int) -> tuple[list[FocusedRunRecord], dict[str, str]]:
```

For each fixed problem, generate it once; for each fixed seed, execute all three lanes with the same
problem and seed. Assert the three results share `initial_vector_hash` and `phase_one_objective`.
Calculate native-relative gain only after all three results return. For the negative-control record,
set `negative_evidence_changed` from a comparison of `priority_spread` against the matching
`arac_policy` evidence. Scan `asdict(result.evidence)` against `FORBIDDEN_RUNTIME_FIELDS`; construct
every record with `claim_allowed=False`.

- [ ] **Step 5: Serialize the complete per-run row**

Implement `run_record_to_row(record)`. Include the fixed schema:

```text
run_id, problem_id, optimizer_seed, lane_id, input_hash, initial_vector_hash,
phase_one_objective, final_objective, selected_action_name, selected_action_family,
decision, trigger_reason, optimizer_consumed, variable_owner_changed,
relation_handling_changed, coordination_mode_changed, budget_allocation_changed,
feature_coverage, overlap_degree, shared_var_support_ratio, direction_disagreement,
harmful_coord_score, group_gain_asymmetry, priority_spread, rank_stability,
budget_remaining_ratio, fallback_margin_proxy, phase_i_fe, phase_ii_fe, total_fe,
budget_limit, same_budget_violation, offline_gain_vs_native, utility_label_vs_native,
catastrophic_loss, forbidden_runtime_fields, runtime_dispatch_allowed,
negative_evidence_changed, claim_allowed
```

`claim_allowed` is always 0 at row level. `catastrophic_loss` is 1 only when the lane is
`arac_policy` and gain is `<= -0.20`.

- [ ] **Step 6: Write the initial runner and manifest**

Implement `run_focused_pilot(output_dir, total_fes=CANONICAL_TOTAL_FES) -> Path`. Validate
`total_fes` is an integer >= 2, write `run_results.csv`, and write `manifest.json` containing the
fixed case list, seed list, lane list, execution count, FE split, benchmark/backend/runner hashes,
input hashes, artifact names, date, executor, and focused-pilot claim level.

- [ ] **Step 7: Run the fixed protocol test**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_010_binary_lsgo_focused_3seed.py::test_fixed_protocol_writes_45_same_budget_rows -q
```

Expected: `1 passed`.

- [ ] **Step 8: Commit the fixed execution matrix**

```powershell
git add -- experiments/exp_010_binary_lsgo_focused_3seed tests/test_exp_010_binary_lsgo_focused_3seed.py
git commit -m "feat: add focused binary LSGO three-seed matrix"
```

### Task 2: Implement case aggregation and pre-registered promotion gates

**Files:**
- Modify: `experiments/exp_010_binary_lsgo_focused_3seed/run.py`
- Modify: `tests/test_exp_010_binary_lsgo_focused_3seed.py`

- [ ] **Step 1: Add failing summary and noncanonical-budget tests**

Append:

```python
from statistics import median


def test_case_summary_matches_run_rows(tmp_path: Path):
    output = run_focused_pilot(tmp_path / "pilot", total_fes=40)
    rows = read_csv(output / "run_results.csv")
    summaries = read_csv(output / "case_summary.csv")
    for summary in summaries:
        gains = [
            float(row["offline_gain_vs_native"])
            for row in rows
            if row["problem_id"] == summary["problem_id"]
            and row["lane_id"] == "arac_policy"
        ]
        assert float(summary["median_relative_gain"]) == pytest.approx(median(gains))
        assert float(summary["minimum_relative_gain"]) == pytest.approx(min(gains))


def test_noncanonical_budget_cannot_pass_promotion_gate(tmp_path: Path):
    output = run_focused_pilot(tmp_path / "pilot", total_fes=40)
    gate = json.loads((output / "promotion_gate.json").read_text(encoding="utf-8"))
    assert gate["canonical_budget"]["passed"] is False
    assert gate["canonical_budget"]["reason"] == "configured_total_fes_is_not_2000"
    assert gate["overall_pass"] is False
```

Add `import pytest` to the test module.

- [ ] **Step 2: Run the tests and verify missing summary artifacts**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_010_binary_lsgo_focused_3seed.py -q
```

Expected: failures because `case_summary.csv` and `promotion_gate.json` do not exist.

- [ ] **Step 3: Implement `build_case_summaries`**

Implement:

```python
def build_case_summaries(records: list[FocusedRunRecord]) -> list[dict[str, object]]:
```

For each fixed case, select exactly three `arac_policy` records and three negative-control records.
Raise `ValueError` when rows are missing or duplicated. Calculate action-consumed count, mean/median/
minimum ARAC gain, median action-consumed gain (empty string when count is 0), catastrophic count,
negative evidence changed count, all-same-budget flag, and runtime-boundary flag. Return rows in fixed
case order.

- [ ] **Step 4: Implement `build_promotion_gate`**

Implement:

```python
def build_promotion_gate(
    records: list[FocusedRunRecord],
    summaries: list[dict[str, object]],
    *,
    configured_total_fes: int,
) -> dict[str, object]:
```

Return seven named gate objects plus `overall_pass`. Use these exact checks:

- `canonical_budget`: configured budget is 2000;
- `target_action_frequency`: F08 and F15 each have action-consumed count >= 2;
- `target_action_median_gain`: each target has a nonempty action-consumed median >= 0;
- `no_catastrophic_loss`: all 15 policy rows have gain > -0.20;
- `same_budget`: 45 ledgers total exactly the configured budget with no violation;
- `runtime_boundary`: no record has forbidden fields;
- `negative_control`: exactly 15 negative rows, all row-level claim flags remain 0, and every negative
  record has `negative_evidence_changed=True`.

Each object contains `passed`, `observed`, `threshold`, and `reason`; `overall_pass` is `all()` of
the seven booleans.

- [ ] **Step 5: Add pure gate tests for pass and independent failures**

Create a small test helper that builds `FocusedRunRecord` instances from real 40-FE backend results,
then uses `dataclasses.replace` to set gains, decisions, semantics, ledger, forbidden fields, and
negative-control flags. Verify a constructed canonical record set passes all gates, then parameterize
mutations that independently fail action frequency, target median, catastrophic loss, same budget,
runtime boundary, and negative control. This test calls `build_promotion_gate` directly and does not
run the 45-run matrix repeatedly.

- [ ] **Step 6: Write summary and gate artifacts**

Update `run_focused_pilot` to write `case_summary.csv` with a fixed field order and
`promotion_gate.json` with `sort_keys=True`, `indent=2`, ASCII-safe output, and a trailing newline.
Add both filenames to the manifest.

- [ ] **Step 7: Run all exp_010 tests**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_010_binary_lsgo_focused_3seed.py -q
```

Expected: all exp_010 tests pass.

- [ ] **Step 8: Commit aggregation and gates**

```powershell
git add -- experiments/exp_010_binary_lsgo_focused_3seed/run.py tests/test_exp_010_binary_lsgo_focused_3seed.py
git commit -m "feat: gate focused binary LSGO pilot results"
```

### Task 3: Add deterministic CLI coverage, documentation, and canonical execution

**Files:**
- Modify: `experiments/exp_010_binary_lsgo_focused_3seed/run.py`
- Modify: `tests/test_exp_010_binary_lsgo_focused_3seed.py`
- Modify: `README.md`
- Modify: `experiments/README.md`
- Modify: `docs/superpowers/specs/2026-07-13-binary-lsgo-focused-3seed-pilot-design.md`

- [ ] **Step 1: Add failing deterministic and CLI tests**

Append tests that run `run_focused_pilot` twice at 40 FE and assert byte equality for all four
artifacts. Test `parse_args(["--output-dir", str(path), "--total-fes", "40"])`, call `main(...)`,
and assert the output exists. Assert invalid budgets 0, 1, `True`, and non-integers raise
`ValueError` through the public runner.

- [ ] **Step 2: Implement `parse_args` and `main`**

Use only:

```text
--output-dir  default results/exp_010_binary_lsgo_focused_3seed
--total-fes   integer, default 2000
```

Do not expose case, seed, lane, policy-threshold, or gate-threshold overrides.

- [ ] **Step 3: Run deterministic and CLI tests**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_010_binary_lsgo_focused_3seed.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Update project documentation**

Add `exp_010_binary_lsgo_focused_3seed` to `README.md` and `experiments/README.md`. Document the
fixed five cases, three seeds, three lanes, 2000-FE budget, four output artifacts, and the claim level
`focused 3-seed pilot`. Include this exact command:

```powershell
$env:PYTHONPATH='src'
& 'E:\ARAC\.venv\Scripts\python.exe' -m experiments.exp_010_binary_lsgo_focused_3seed.run `
  --output-dir results/exp_010_binary_lsgo_focused_3seed `
  --total-fes 2000
```

Change the design document status from `已确认`/`待用户复核` to `已实现并验证` only after the
canonical run and all verification commands succeed.

- [ ] **Step 5: Run the canonical 45-run protocol**

Run the documented command. Expected: exit 0; `run_results.csv` has 45 rows;
`case_summary.csv` has 5 rows; every run uses 2000 FE with no budget violation; gate JSON contains
all seven checks. Do not stage `results/`.

- [ ] **Step 6: Run focused and repository verification**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest `
  tests/test_binary_lsgo_benchmark.py `
  tests/test_binary_lsgo_backend.py `
  tests/test_exp_009_binary_lsgo_arac_pilot.py `
  tests/test_exp_010_binary_lsgo_focused_3seed.py -q
E:\ARAC\.venv\Scripts\python.exe -m pytest -q
E:\ARAC\.venv\Scripts\python.exe -m compileall -q src experiments/exp_010_binary_lsgo_focused_3seed
git diff --check
git status --short
```

Expected: all tests pass except documented skips; compileall exits 0; diff check has no output;
`results/`, caches, inherited source, and unrelated user files are not staged.

- [ ] **Step 7: Commit documentation and verified corrections**

```powershell
git add -- README.md experiments/README.md \
  docs/superpowers/specs/2026-07-13-binary-lsgo-focused-3seed-pilot-design.md \
  experiments/exp_010_binary_lsgo_focused_3seed tests/test_exp_010_binary_lsgo_focused_3seed.py
git diff --cached --check
git commit -m "feat: complete focused binary LSGO three-seed pilot"
```

- [ ] **Step 8: Report the evidence without overclaiming**

Report per-target action frequency, target action-consumed median gains, catastrophic count,
same-budget status, negative-control status, and `overall_pass`. State explicitly that passing the
gate permits a larger experiment but is not final performance evidence. Do not push until the user
explicitly confirms remote reconciliation and push.
