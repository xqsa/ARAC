# Binary LSGO ARAC Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the deterministic binary overlapping LSGO suite to ARAC through a real binary cooperative-coevolution backend, runtime evidence, optimizer-consumed actions, and a same-budget 18-case pilot.

**Architecture:** Keep `BinaryLsgoProblem` as the problem source and add a standard-library-only backend that owns optimization state, Phase-I trace collection, evidence conversion, action binding, and exact FE accounting. A thin `exp_009` runner executes native, ARAC-policy, and shuffled-evidence lanes from identical seeds, then writes offline results and audit artifacts without importing HCC.

**Tech Stack:** Python 3.11 dataclasses, `random.Random`, `hashlib`, `csv`, `json`, existing ARAC evidence/policy/evaluation modules, pytest.

---

## File map

- Create `src/arac/backends/binary_lsgo.py`: request/result contracts, deterministic grouped bit-flip search, trace snapshot, evidence conversion, action binding, and exact FE ledger.
- Create `tests/test_binary_lsgo_backend.py`: focused backend, evidence, action-semantics, reproducibility, and budget tests.
- Create `experiments/exp_009_binary_lsgo_arac_pilot/__init__.py`: experiment package marker.
- Create `experiments/exp_009_binary_lsgo_arac_pilot/run.py`: 18-case three-lane orchestration and CSV/JSON artifact writing.
- Create `tests/test_exp_009_binary_lsgo_arac_pilot.py`: artifact schema, lane identity, same-budget, anti-leakage, and deterministic-output tests.
- Modify `README.md`: replace the “independent only” inventory wording with the new binary backend and pilot entry.
- Modify `experiments/README.md`: add the exact pilot command and claim boundary.

### Task 1: Define the binary backend contracts and runtime evidence conversion

**Files:**
- Create: `src/arac/backends/binary_lsgo.py`
- Test: `tests/test_binary_lsgo_backend.py`

- [ ] **Step 1: Write failing validation and evidence tests**

Create `tests/test_binary_lsgo_backend.py` with a small deterministic fixture and these initial tests:

```python
from dataclasses import asdict

import pytest

from arac.backends.binary_lsgo import (
    BinaryLsgoExecutionRequest,
    BinaryLsgoGroupStats,
    BinaryLsgoSnapshot,
    build_binary_lsgo_evidence_profile,
)
from arac.benchmarks.binary_lsgo import BinaryLsgoSpec, generate_binary_lsgo
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS


def small_problem():
    return generate_binary_lsgo(
        BinaryLsgoSpec("small", 40, 8, 2, 5, True, 0.5, 0.5, 0.5, 0.5, 11)
    )


def test_request_rejects_invalid_budget_or_seed():
    problem = small_problem()
    with pytest.raises(ValueError, match="total_fes"):
        BinaryLsgoExecutionRequest(problem, optimizer_seed=1, total_fes=1)
    with pytest.raises(ValueError, match="phase_one_fraction"):
        BinaryLsgoExecutionRequest(problem, optimizer_seed=1, phase_one_fraction=1.0)
    with pytest.raises(ValueError, match="optimizer_seed"):
        BinaryLsgoExecutionRequest(problem, optimizer_seed=-1)


def test_snapshot_converts_to_runtime_legal_evidence():
    problem = small_problem()
    stats = tuple(
        BinaryLsgoGroupStats(group_index=index, proposed=2, accepted=index % 2, gain=float(index))
        for index in range(len(problem.topology.groups))
    )
    snapshot = BinaryLsgoSnapshot(
        run_id="test",
        lane_id="arac_policy",
        problem_id=problem.spec.problem_id,
        optimizer_seed=9,
        consumed_fes=8,
        total_fes=40,
        group_stats=stats,
        shared_proposals=4,
        rejected_shared_proposals=1,
        conflicting_shared_variables=2,
        rank_stability=0.75,
        topology=problem.topology,
    )
    evidence = build_binary_lsgo_evidence_profile(snapshot)
    assert evidence.problem_id == "small"
    assert evidence.budget_remaining_ratio == pytest.approx(0.8)
    assert evidence.harmful_coord_score == pytest.approx(0.25)
    assert set(asdict(evidence)).isdisjoint(FORBIDDEN_RUNTIME_FIELDS)
```

- [ ] **Step 2: Run the focused tests and verify the missing module failure**

Run: `python -m pytest tests/test_binary_lsgo_backend.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'arac.backends.binary_lsgo'`.

- [ ] **Step 3: Add immutable contracts and evidence formulas**

Create `src/arac/backends/binary_lsgo.py` with these public contracts and guards:

```python
from __future__ import annotations

from dataclasses import dataclass

from arac.benchmarks.binary_lsgo import BinaryLsgoProblem, BinaryLsgoTopology
from arac.evidence import EvidenceProfile, validate_runtime_payload


@dataclass(frozen=True)
class BinaryLsgoGroupStats:
    group_index: int
    proposed: int = 0
    accepted: int = 0
    gain: float = 0.0
    early_gain: float = 0.0
    late_gain: float = 0.0


@dataclass(frozen=True)
class BinaryLsgoSnapshot:
    run_id: str
    lane_id: str
    problem_id: str
    optimizer_seed: int
    consumed_fes: int
    total_fes: int
    group_stats: tuple[BinaryLsgoGroupStats, ...]
    shared_proposals: int
    rejected_shared_proposals: int
    conflicting_shared_variables: int
    rank_stability: float
    topology: BinaryLsgoTopology


@dataclass(frozen=True)
class BinaryLsgoExecutionRequest:
    problem: BinaryLsgoProblem
    optimizer_seed: int
    total_fes: int = 2_000
    phase_one_fraction: float = 0.20
    run_id: str = "binary_lsgo_arac"
    lane_id: str = "arac_policy"

    def __post_init__(self) -> None:
        if self.optimizer_seed < 0:
            raise ValueError("optimizer_seed must be non-negative")
        if self.total_fes < 2:
            raise ValueError("total_fes must be at least 2")
        if not 0.0 < self.phase_one_fraction < 1.0:
            raise ValueError("phase_one_fraction must be in (0, 1)")

    @property
    def phase_one_fes(self) -> int:
        return max(1, min(self.total_fes - 1, round(self.total_fes * self.phase_one_fraction)))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def build_binary_lsgo_evidence_profile(snapshot: BinaryLsgoSnapshot) -> EvidenceProfile:
    payload = {
        "run_id": snapshot.run_id,
        "lane_id": snapshot.lane_id,
        "problem_id": snapshot.problem_id,
        "optimizer_seed": snapshot.optimizer_seed,
        "consumed_fes": snapshot.consumed_fes,
        "total_fes": snapshot.total_fes,
    }
    validate_runtime_payload(payload)
    gains = [max(0.0, item.gain) for item in snapshot.group_stats]
    maximum_gain = max(gains, default=0.0)
    gain_asymmetry = _ratio(maximum_gain - min(gains, default=0.0), maximum_gain + 1e-12)
    group_count = len(snapshot.topology.groups)
    possible_pairs = group_count * (group_count - 1) / 2
    overlap_degree = _ratio(len(snapshot.topology.adjacency_pairs), possible_pairs)
    shared_support = _ratio(
        snapshot.topology.shared_variable_count,
        snapshot.topology.decision_dimension,
    )
    harmful = _ratio(snapshot.rejected_shared_proposals, snapshot.shared_proposals)
    conflict = _ratio(
        snapshot.conflicting_shared_variables,
        snapshot.topology.shared_variable_count,
    )
    covered_groups = sum(item.proposed > 0 for item in snapshot.group_stats)
    return EvidenceProfile(
        run_id=snapshot.run_id,
        problem_id=snapshot.problem_id,
        seed=snapshot.optimizer_seed,
        unit_type="problem",
        unit_id=f"binary_lsgo_backend:{snapshot.problem_id}",
        feature_coverage=_ratio(covered_groups, group_count),
        overlap_degree=overlap_degree,
        shared_var_support_ratio=shared_support,
        direction_disagreement=conflict,
        harmful_coord_score=harmful,
        group_gain_asymmetry=gain_asymmetry,
        priority_spread=gain_asymmetry,
        rank_stability=_ratio(snapshot.rank_stability, 1.0),
        budget_remaining_ratio=_ratio(
            snapshot.total_fes - snapshot.consumed_fes,
            snapshot.total_fes,
        ),
        fallback_margin_proxy=1.0 - harmful,
    )
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_binary_lsgo_backend.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the contracts**

```powershell
git add -- src/arac/backends/binary_lsgo.py tests/test_binary_lsgo_backend.py
git commit -m "feat: define binary LSGO backend contracts"
```

### Task 2: Implement two-phase search and optimizer-consumed ARAC actions

**Files:**
- Modify: `src/arac/backends/binary_lsgo.py`
- Modify: `tests/test_binary_lsgo_backend.py`

- [ ] **Step 1: Add failing execution, budget, reproducibility, and action tests**

Append tests that use `total_fes=80` and explicit `ActionDecision` values:

```python
from arac.action_space import ActionFamily
from arac.backends.binary_lsgo import run_binary_lsgo
from arac.policy import ActionDecision


def decision(family, name):
    return ActionDecision(family, name, "allow", "test_override", 1.0)


def test_execution_is_reproducible_and_exact_budget():
    request = BinaryLsgoExecutionRequest(small_problem(), optimizer_seed=17, total_fes=80)
    first = run_binary_lsgo(request)
    second = run_binary_lsgo(request)
    assert first == second
    assert first.ledger.phase_i_fe == 16
    assert first.ledger.phase_ii_fe == 64
    assert first.ledger.total_fe == 80
    assert not first.ledger.violation


@pytest.mark.parametrize(
    ("action", "field"),
    [
        (decision(ActionFamily.COORDINATE, "allow_beneficial_coordination"), "coordination_mode_changed"),
        (decision(ActionFamily.ISOLATE, "isolate_conflicting_relation"), "relation_handling_changed"),
        (decision(ActionFamily.REASSIGN_REPAIR, "repair_shared_variable_binding"), "variable_owner_changed"),
        (decision(ActionFamily.PROTECT, "protect_high_margin_group"), "budget_allocation_changed"),
    ],
)
def test_actions_change_optimizer_consumed_semantics(action, field):
    result = run_binary_lsgo(
        BinaryLsgoExecutionRequest(small_problem(), optimizer_seed=17, total_fes=80),
        decision_override=action,
    )
    assert result.optimizer_consumed
    assert getattr(result.semantics, field)
    assert result.action_trace.consumed_fe == result.ledger.phase_ii_fe


def test_unsupported_action_fails_loudly():
    unsupported = decision(ActionFamily.TRAJECTORY, "budget_shift_mean_blend")
    with pytest.raises(ValueError, match="unsupported binary LSGO action"):
        run_binary_lsgo(
            BinaryLsgoExecutionRequest(small_problem(), optimizer_seed=17, total_fes=80),
            decision_override=unsupported,
        )
```

- [ ] **Step 2: Run tests and verify missing execution symbols**

Run: `python -m pytest tests/test_binary_lsgo_backend.py -q`

Expected: collection fails because `run_binary_lsgo` is not defined.

- [ ] **Step 3: Implement execution result and semantic contracts**

Add immutable `BinaryBackendSemanticsDiff`, `BinaryLsgoActionTrace`, and
`BinaryLsgoExecutionResult` dataclasses. The result must include `initial_vector_hash`,
`phase_one_objective`, `final_objective`, `final_vector`, `evidence`, `decision`,
`semantics`, `ledger`, `action_trace`, and `optimizer_consumed`. Define the exact map:

```python
SUPPORTED_ACTIONS = {
    "conservative_no_action": "native_round_robin",
    "allow_beneficial_coordination": "prioritize_related_groups_after_shared_accept",
    "isolate_conflicting_relation": "owner_only_shared_write",
    "repair_shared_variable_binding": "gain_ranked_owner_only_shared_write",
    "protect_high_margin_group": "gain_weighted_group_schedule",
}
```

Use these exact field contracts so the runner and tests have one source of truth:

```python
@dataclass(frozen=True)
class BinaryBackendSemanticsDiff:
    variable_owner_changed: bool = False
    relation_handling_changed: bool = False
    coordination_mode_changed: bool = False
    budget_allocation_changed: bool = False

    @property
    def changed(self) -> bool:
        return any((
            self.variable_owner_changed,
            self.relation_handling_changed,
            self.coordination_mode_changed,
            self.budget_allocation_changed,
        ))


@dataclass(frozen=True)
class BinaryLsgoActionTrace:
    action_name: str
    decision: str
    trigger_reason: str
    phase: str
    affected_group_count: int
    affected_shared_variable_count: int
    allocated_fe: int
    consumed_fe: int


@dataclass(frozen=True)
class BinaryLsgoExecutionResult:
    run_id: str
    lane_id: str
    problem_id: str
    optimizer_seed: int
    initial_vector_hash: str
    phase_one_objective: float
    final_objective: float
    final_vector: tuple[int, ...]
    evidence: EvidenceProfile
    decision: ActionDecision
    semantics: BinaryBackendSemanticsDiff
    ledger: SameBudgetLedger
    action_trace: BinaryLsgoActionTrace
    optimizer_consumed: bool


def run_binary_lsgo(
    request: BinaryLsgoExecutionRequest,
    *,
    decision_override: ActionDecision | None = None,
) -> BinaryLsgoExecutionResult:
    """Run the exact two-phase binary backend for one lane."""
```

Use `SameBudgetLedger` for the returned ledger. Unsupported action names must raise
`ValueError` before Phase II starts.

- [ ] **Step 4: Implement the deterministic Phase-I loop**

Initialize `random.Random(request.optimizer_seed)`, generate the initial vector with
`rng.randrange(2)`, hash it with SHA-256, and count its first objective call inside Phase I.
For each remaining Phase-I FE, visit groups round-robin, choose a bit from that group with the
same local RNG, evaluate one flipped candidate, and accept only strict global improvement.
Update per-group proposed/accepted/gain and early/late gains. Derive rank stability from pairwise
agreement between early-gain and late-gain orderings. Store shared proposal rejection counts and
variables whose containing groups have different acceptance outcomes.

- [ ] **Step 5: Build evidence, choose the lane decision, and bind action state**

After exactly `phase_one_fes`, build `BinaryLsgoSnapshot` and `EvidenceProfile`. Decision order is:

```python
if decision_override is not None:
    selected = decision_override
elif request.lane_id == "native_baseline":
    selected = ActionDecision(
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "fallback",
        "native_baseline_lane",
        0.0,
    )
else:
    selected = decide_action(evidence)
```

For `shuffled_evidence_negative_control`, deterministically permute only the group statistics with
`random.Random(request.optimizer_seed + 1_000_003)` before evidence conversion. Do not permute
topology, IDs, budget, objective values, or final outputs.

- [ ] **Step 6: Implement Phase-II schedules**

Consume every remaining FE with one candidate evaluation per loop:

- fallback: native round-robin over every group and every listed variable;
- coordinate: after accepting a shared-variable flip, enqueue its other containing groups before
  continuing round-robin;
- isolate: create first-containing-group owners and disallow non-owner proposals for shared bits;
- repair: select the highest Phase-I-gain containing group as each shared bit owner, then enforce it;
- protect: build a deterministic weighted schedule where the top gain quartile appears twice and all
  other groups once.

Ineligible shared bits are skipped when selecting a bit; if a group has no eligible bit, advance to
the next group without consuming FE. Reject an impossible schedule instead of looping forever.

- [ ] **Step 7: Run backend tests and the existing benchmark tests**

Run: `python -m pytest tests/test_binary_lsgo_backend.py tests/test_binary_lsgo_benchmark.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit the executable backend**

```powershell
git add -- src/arac/backends/binary_lsgo.py tests/test_binary_lsgo_backend.py
git commit -m "feat: execute ARAC actions on binary LSGO"
```

### Task 3: Add the 18-case, three-lane pilot and audit artifacts

**Files:**
- Create: `experiments/exp_009_binary_lsgo_arac_pilot/__init__.py`
- Create: `experiments/exp_009_binary_lsgo_arac_pilot/run.py`
- Create: `tests/test_exp_009_binary_lsgo_arac_pilot.py`

- [ ] **Step 1: Write the failing pilot contract test**

```python
import csv
import json

from experiments.exp_009_binary_lsgo_arac_pilot.run import run_pilot


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_pilot_writes_three_same_budget_lanes_for_all_cases(tmp_path):
    output = run_pilot(tmp_path / "pilot", total_fes=40)
    results = read_csv(output / "execution_results.csv")
    ledger = read_csv(output / "same_budget_ledger.csv")
    evidence = read_csv(output / "runtime_evidence.csv")
    trace = read_csv(output / "action_trace.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert len(results) == len(ledger) == len(evidence) == len(trace) == 54
    assert {row["lane_id"] for row in results} == {
        "native_baseline", "arac_policy", "shuffled_evidence_negative_control"
    }
    assert {row["problem_id"] for row in results} == {
        f"BLSGO-F{index:02d}" for index in range(1, 19)
    }
    assert {row["total_fe"] for row in ledger} == {"40"}
    assert {row["same_budget_violation"] for row in ledger} == {"0"}
    assert all(row["claim_allowed"] == "0" for row in results)
    assert manifest["benchmark_case_count"] == 18
    assert manifest["lane_count"] == 3
```

- [ ] **Step 2: Run the test and verify the missing experiment package failure**

Run: `python -m pytest tests/test_exp_009_binary_lsgo_arac_pilot.py -q`

Expected: collection fails with `ModuleNotFoundError` for `exp_009_binary_lsgo_arac_pilot`.

- [ ] **Step 3: Implement the pilot runner**

Set constants `RUN_ID="exp_009_binary_lsgo_arac_pilot"`, `DEFAULT_TOTAL_FES=2000`,
`PHASE_ONE_FRACTION=0.20`, `OPTIMIZER_SEED_BASE=20260713`, and the three lane names. For every
`standard_binary_lsgo_specs()` entry, generate the problem once and execute all lanes with seed
`OPTIMIZER_SEED_BASE + case_index`. Assert all lane results have the same initial-vector hash and
Phase-I objective before writing output.

Implement `run_pilot(output_dir, total_fes=DEFAULT_TOTAL_FES) -> Path`, `parse_args()`, and `main()`.
The CLI must accept `--output-dir` and `--total-fes`.

- [ ] **Step 4: Write fixed artifact schemas**

Write UTF-8/newline-safe CSVs with `csv.DictWriter`:

- `execution_results.csv`: run/lane/problem/seed, initial hash, phase-one/final objective, selected
  action, optimizer-consumed flag, offline gain vs native, utility label, catastrophic-loss flag,
  `claim_allowed=0`, and `pilot_single_seed_not_final_claim` blocker;
- `action_trace.csv`: selected action, trigger, affected groups/shared variables, allocated/consumed
  Phase-II FE and all semantics-diff booleans;
- `same_budget_ledger.csv`: Phase-I/II/total/limit/violation/fresh execution;
- `runtime_evidence.csv`: every `EvidenceProfile` field plus a semicolon-joined forbidden-field scan;
- `manifest.json`: date, executor, code source, benchmark/lane counts, FE split, seeds, input hashes,
  result role, and generated artifact names.

Use `relative_gain()` and `classify_utility()` only after all lane executions. Never feed those
values back into evidence, decisions, schedules, or ownership.

- [ ] **Step 5: Add deterministic-output and anti-leakage tests**

Run the pilot twice at `total_fes=40`; assert all CSV and JSON bytes match. Assert each
`(problem_id, seed)` group has one shared initial hash and Phase-I objective, no runtime evidence
header intersects `FORBIDDEN_RUNTIME_FIELDS`, negative-control rows have `claim_allowed=0`, and
`manifest.json` contains 18 distinct problem input hashes.

- [ ] **Step 6: Run pilot tests**

Run: `python -m pytest tests/test_exp_009_binary_lsgo_arac_pilot.py -q`

Expected: all tests pass.

- [ ] **Step 7: Run a real default-budget pilot smoke outside Git artifacts**

Run:

```powershell
$env:PYTHONPATH='src'
& 'E:\ARAC\.venv\Scripts\python.exe' -m experiments.exp_009_binary_lsgo_arac_pilot.run `
  --output-dir results/exp_009_binary_lsgo_arac_pilot `
  --total-fes 2000
```

Expected: exit 0; 54 result rows; every ledger row has `total_fe=2000` and
`same_budget_violation=0`. Do not stage `results/`.

- [ ] **Step 8: Commit the pilot**

```powershell
git add -- experiments/exp_009_binary_lsgo_arac_pilot tests/test_exp_009_binary_lsgo_arac_pilot.py
git commit -m "feat: add binary LSGO ARAC pilot"
```

### Task 4: Update entry-point documentation and run release verification

**Files:**
- Modify: `README.md`
- Modify: `experiments/README.md`

- [ ] **Step 1: Update the project inventory**

Change the binary benchmark entry in `README.md` to state that it remains independent of continuous
HCC but now has a native binary backend at `src/arac/backends/binary_lsgo.py`. Add the
`exp_009_binary_lsgo_arac_pilot` directory and explicitly call it a single-seed same-budget pilot,
not final performance evidence.

- [ ] **Step 2: Add the exact experiment command**

Append this command and artifact summary to `experiments/README.md`:

```powershell
$env:PYTHONPATH='src'
& 'E:\ARAC\.venv\Scripts\python.exe' -m experiments.exp_009_binary_lsgo_arac_pilot.run `
  --output-dir results/exp_009_binary_lsgo_arac_pilot `
  --total-fes 2000
```

State that it executes 18 cases x 3 lanes, uses Phase I = 20%, and leaves generated results out of
Git.

- [ ] **Step 3: Run focused verification**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py tests/test_binary_lsgo_backend.py tests/test_exp_009_binary_lsgo_arac_pilot.py -q
```

Expected: all focused tests pass.

- [ ] **Step 4: Run repository verification**

Run:

```powershell
python -m pytest -q
python -m compileall -q src experiments/exp_009_binary_lsgo_arac_pilot
git diff --check
git status --short
```

Expected: pytest passes except documented skips; compileall exits 0; diff check has no output;
generated `results/`, caches, inherited sources, and unrelated untracked files are not staged.

- [ ] **Step 5: Commit documentation and any final verified corrections**

```powershell
git add -- README.md experiments/README.md
git commit -m "docs: document binary LSGO ARAC pilot"
```

- [ ] **Step 6: Record the final local commit without pushing**

Run: `git log -4 --oneline` and report the implementation commits, test counts, pilot row counts,
and any residual risk. `git push` requires explicit user confirmation.
