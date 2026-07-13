# Binary LSGO Mechanism Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a same-budget F08/F15 diagnostic that separates ARAC policy-trigger limitations from the binary backend's inability to cross deception valleys with single-bit proposals.

**Architecture:** Extend the existing binary backend with an explicit Phase-II proposal operator and an immutable proposal trace while preserving `single_bit` as the default. Add a thin `exp_011` runner that executes four controlled lanes from identical Phase-I state, performs offline-only signal classification, and writes deterministic CSV/JSON artifacts.

**Tech Stack:** Python 3.11 dataclasses, `random.Random`, `csv`, `json`, existing ARAC benchmark/policy/evaluation modules, pytest.

---

## File map

- Modify `src/arac/backends/binary_lsgo.py`: validate the Phase-II operator, generate single-bit or whole-group proposals, and expose proposal statistics.
- Modify `tests/test_binary_lsgo_backend.py`: cover validation, backward compatibility, block proposal semantics, FE accounting, and trace determinism.
- Create `experiments/exp_011_binary_lsgo_diagnostic/__init__.py`: experiment package marker.
- Create `experiments/exp_011_binary_lsgo_diagnostic/run.py`: fixed 40-run matrix, offline classification, artifacts, manifest, and CLI.
- Create `tests/test_exp_011_binary_lsgo_diagnostic.py`: protocol, classifications, artifact determinism, CLI, and invalid-budget tests.
- Modify `experiments/README.md`: document the exact command and diagnostic-only claim boundary.

## Task 1: Add proposal-operator and trace contracts

**Files:**
- Modify: `src/arac/backends/binary_lsgo.py`
- Modify: `tests/test_binary_lsgo_backend.py`

- [ ] **Step 1: Write failing request-validation tests**

Add these tests and import `BinaryLsgoProposalTrace` only after the first RED run:

```python
@pytest.mark.parametrize("operator", ["", "bit", "group", 1, True, None])
def test_request_rejects_unsupported_phase_two_operator(operator):
    with pytest.raises(ValueError, match="phase_two_operator"):
        BinaryLsgoExecutionRequest(
            small_problem(),
            optimizer_seed=1,
            total_fes=40,
            phase_two_operator=operator,
        )


def test_request_defaults_to_single_bit_operator():
    request = BinaryLsgoExecutionRequest(small_problem(), optimizer_seed=1, total_fes=40)
    assert request.phase_two_operator == "single_bit"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_binary_lsgo_backend.py::test_request_rejects_unsupported_phase_two_operator tests/test_binary_lsgo_backend.py::test_request_defaults_to_single_bit_operator -q
```

Expected: failure because `BinaryLsgoExecutionRequest` does not accept or expose
`phase_two_operator`.

- [ ] **Step 3: Add the minimal contracts**

Add the request field and validation:

```python
PROPOSAL_OPERATORS = frozenset({"single_bit", "group_block"})


@dataclass(frozen=True)
class BinaryLsgoExecutionRequest:
    problem: BinaryLsgoProblem
    optimizer_seed: int
    total_fes: int = 2_000
    phase_one_fraction: float = 0.20
    run_id: str = "binary_lsgo_arac"
    lane_id: str = "arac_policy"
    phase_two_operator: str = "single_bit"

    def __post_init__(self) -> None:
        # Keep all existing guards unchanged before this new guard.
        if (
            not isinstance(self.phase_two_operator, str)
            or self.phase_two_operator not in PROPOSAL_OPERATORS
        ):
            raise ValueError(
                "phase_two_operator must be one of: "
                + ", ".join(sorted(PROPOSAL_OPERATORS))
            )
```

Add the immutable trace and result field:

```python
@dataclass(frozen=True)
class BinaryLsgoProposalTrace:
    operator: str
    proposed_count: int
    accepted_count: int
    multi_bit_proposed_count: int
    multi_bit_accepted_count: int
    maximum_accepted_flip_width: int


@dataclass(frozen=True)
class BinaryLsgoExecutionResult:
    # Keep existing fields in their current order.
    proposal_trace: BinaryLsgoProposalTrace
```

Temporarily construct a zeroed trace at the existing return site so contract tests can turn
green before Task 2 implements counting:

```python
proposal_trace=BinaryLsgoProposalTrace(
    operator=request.phase_two_operator,
    proposed_count=0,
    accepted_count=0,
    multi_bit_proposed_count=0,
    multi_bit_accepted_count=0,
    maximum_accepted_flip_width=0,
),
```

Export `BinaryLsgoProposalTrace` and `PROPOSAL_OPERATORS` from `__all__`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_binary_lsgo_backend.py -q
```

Expected: all backend tests pass.

- [ ] **Step 5: Commit the contracts**

```powershell
git add -- src/arac/backends/binary_lsgo.py tests/test_binary_lsgo_backend.py
git diff --cached --check
git commit -m "feat: define binary LSGO proposal operators"
```

## Task 2: Implement block proposals and exact proposal traces

**Files:**
- Modify: `src/arac/backends/binary_lsgo.py`
- Modify: `tests/test_binary_lsgo_backend.py`

- [ ] **Step 1: Write failing behavior tests**

Add:

```python
def fixed_group_problem():
    return generate_binary_lsgo(
        BinaryLsgoSpec("fixed", 40, 8, 5, 5, True, 0.8, 0.5, 0.5, 0.5, 19)
    )


def fallback_decision() -> ActionDecision:
    return ActionDecision(
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "fallback",
        "test_native_lane",
        0.0,
    )


def test_explicit_single_bit_preserves_default_execution():
    default = run_binary_lsgo(
        BinaryLsgoExecutionRequest(fixed_group_problem(), 23, total_fes=80),
        decision_override=fallback_decision(),
    )
    explicit = run_binary_lsgo(
        BinaryLsgoExecutionRequest(
            fixed_group_problem(),
            23,
            total_fes=80,
            phase_two_operator="single_bit",
        ),
        decision_override=fallback_decision(),
    )
    assert default == explicit
    assert explicit.proposal_trace.proposed_count == 64
    assert explicit.proposal_trace.multi_bit_proposed_count == 0
    assert explicit.proposal_trace.maximum_accepted_flip_width in {0, 1}


def test_group_block_uses_one_fe_per_multi_bit_proposal():
    result = run_binary_lsgo(
        BinaryLsgoExecutionRequest(
            fixed_group_problem(),
            23,
            total_fes=80,
            phase_two_operator="group_block",
        ),
        decision_override=fallback_decision(),
    )
    trace = result.proposal_trace
    assert result.ledger.phase_ii_fe == 64
    assert trace.operator == "group_block"
    assert trace.proposed_count == 64
    assert trace.multi_bit_proposed_count == 64
    assert trace.accepted_count == trace.multi_bit_accepted_count
    assert trace.maximum_accepted_flip_width in {0, 5}
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_binary_lsgo_backend.py::test_explicit_single_bit_preserves_default_execution tests/test_binary_lsgo_backend.py::test_group_block_uses_one_fe_per_multi_bit_proposal -q
```

Expected: trace counts remain zero and the block lane still produces single-bit candidates.

- [ ] **Step 3: Centralize eligibility and implement both proposal modes**

Replace `_choose_group_variable` with these two helpers:

```python
def _eligible_group_variables(
    group: tuple[int, ...],
    *,
    action_name: str,
    group_index: int,
    owners: dict[int, int],
    shared_variables: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    if action_name not in {"isolate_conflicting_relation", "repair_shared_variable_binding"}:
        return group
    return tuple(
        variable
        for variable in group
        if variable not in shared_variables or owners[variable] == group_index
    )


def _proposal_variables(
    eligible: tuple[int, ...],
    *,
    operator: str,
    rng: random.Random,
) -> tuple[int, ...]:
    if not eligible and operator == "group_block":
        raise ValueError("group_block proposal has no eligible variables")
    if not eligible:
        return ()
    if operator == "single_bit":
        return (rng.choice(eligible),)
    if operator == "group_block":
        return eligible
    raise ValueError(f"unsupported phase_two_operator: {operator}")
```

Before the Phase-II loop initialize counters:

```python
proposed_count = 0
accepted_count = 0
multi_bit_proposed_count = 0
multi_bit_accepted_count = 0
maximum_accepted_flip_width = 0
```

Inside the loop, replace the single-variable candidate construction with:

```python
eligible = _eligible_group_variables(
    group,
    action_name=decision.action_name,
    group_index=group_index,
    owners=owners,
    shared_variables=shared_variables,
)
proposal_variables = _proposal_variables(
    eligible,
    operator=request.phase_two_operator,
    rng=rng,
)
if not proposal_variables:
    empty_scans += 1
    if empty_scans >= len(group_schedule) and not coordinated_queue:
        raise ValueError("binary LSGO action left no eligible variables")
    continue
empty_scans = 0
candidate = list(vector)
for variable in proposal_variables:
    candidate[variable] = 1 - candidate[variable]
candidate_objective = problem.evaluate(tuple(candidate))
consumed_fes += 1
proposed_count += 1
proposal_width = len(proposal_variables)
if proposal_width > 1:
    multi_bit_proposed_count += 1
if candidate_objective < current_objective:
    vector = candidate
    current_objective = candidate_objective
    accepted_count += 1
    maximum_accepted_flip_width = max(maximum_accepted_flip_width, proposal_width)
    if proposal_width > 1:
        multi_bit_accepted_count += 1
    if decision.action_name == "allow_beneficial_coordination":
        coordinated_variables = [
            variable for variable in proposal_variables if variable in shared_variables
        ]
        for variable in coordinated_variables:
            coordinated_queue.extend(
                other for other in shared_variables[variable] if other != group_index
            )
```

Replace the temporary trace at the return site with:

```python
proposal_trace=BinaryLsgoProposalTrace(
    operator=request.phase_two_operator,
    proposed_count=proposed_count,
    accepted_count=accepted_count,
    multi_bit_proposed_count=multi_bit_proposed_count,
    multi_bit_accepted_count=multi_bit_accepted_count,
    maximum_accepted_flip_width=maximum_accepted_flip_width,
),
```

- [ ] **Step 4: Run backend and existing experiment tests**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_binary_lsgo_backend.py tests/test_exp_009_binary_lsgo_arac_pilot.py tests/test_exp_010_binary_lsgo_focused_3seed.py -q
```

Expected: all tests pass; existing experiments retain single-bit behavior.

- [ ] **Step 5: Commit the operator implementation**

```powershell
git add -- src/arac/backends/binary_lsgo.py tests/test_binary_lsgo_backend.py
git diff --cached --check
git commit -m "feat: add group-block binary LSGO proposals"
```

## Task 3: Add the fixed exp_011 diagnostic runner

**Files:**
- Create: `experiments/exp_011_binary_lsgo_diagnostic/__init__.py`
- Create: `experiments/exp_011_binary_lsgo_diagnostic/run.py`
- Create: `tests/test_exp_011_binary_lsgo_diagnostic.py`

- [ ] **Step 1: Write failing protocol and classification tests**

Create the test file with these core tests:

```python
import csv
import json
from pathlib import Path

import pytest

from experiments.exp_011_binary_lsgo_diagnostic.run import (
    DIAGNOSTIC_PROBLEM_IDS,
    LANES,
    OPTIMIZER_SEEDS,
    classify_diagnostic_signals,
    main,
    parse_args,
    run_diagnostic,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((3, 3, 2, 5), "optimizer_limited"),
        ((2, 3, 3, 3), "policy_limited"),
        ((3, 3, 3, 3), "mixed"),
        ((2, 2, 2, 2), "inconclusive"),
    ],
)
def test_classification_covers_all_reachable_labels(counts, expected):
    assert classify_diagnostic_signals(*counts).label == expected


def test_fixed_protocol_writes_40_same_budget_rows(tmp_path: Path):
    output = run_diagnostic(tmp_path / "diagnostic", total_fes=40)
    rows = read_csv(output / "run_results.csv")
    summaries = read_csv(output / "case_summary.csv")
    diagnosis = json.loads((output / "diagnosis.json").read_text(encoding="utf-8"))
    assert tuple(DIAGNOSTIC_PROBLEM_IDS) == ("BLSGO-F08", "BLSGO-F15")
    assert tuple(OPTIMIZER_SEEDS) == (
        20260713,
        20260714,
        20260715,
        20260716,
        20260717,
    )
    assert tuple(LANES) == (
        "native_single_bit",
        "native_group_block",
        "forced_isolate",
        "arac_policy",
    )
    assert len(rows) == 40
    assert len(summaries) == 2
    assert {row["total_fe"] for row in rows} == {"40"}
    assert {row["same_budget_violation"] for row in rows} == {"0"}
    assert {row["claim_allowed"] for row in rows} == {"0"}
    assert len(diagnosis["case_diagnoses"]) == 2


def test_lanes_share_initial_state_and_phase_one_objective(tmp_path: Path):
    rows = read_csv(run_diagnostic(tmp_path / "diagnostic", total_fes=40) / "run_results.csv")
    for problem_id in DIAGNOSTIC_PROBLEM_IDS:
        for seed in OPTIMIZER_SEEDS:
            matched = [
                row
                for row in rows
                if row["problem_id"] == problem_id
                and int(row["optimizer_seed"]) == seed
            ]
            assert len({row["initial_vector_hash"] for row in matched}) == 1
            assert len({row["phase_one_objective"] for row in matched}) == 1


def test_operator_and_action_identities_remain_separate(tmp_path: Path):
    rows = read_csv(run_diagnostic(tmp_path / "diagnostic", total_fes=40) / "run_results.csv")
    by_lane = {lane: [row for row in rows if row["lane_id"] == lane] for lane in LANES}
    assert {row["proposal_operator"] for row in by_lane["native_group_block"]} == {
        "group_block"
    }
    assert {row["selected_action_name"] for row in by_lane["native_group_block"]} == {
        "conservative_no_action"
    }
    assert {row["proposal_operator"] for row in by_lane["forced_isolate"]} == {
        "single_bit"
    }
    assert {row["selected_action_name"] for row in by_lane["forced_isolate"]} == {
        "isolate_conflicting_relation"
    }


def test_artifacts_are_byte_deterministic(tmp_path: Path):
    first = run_diagnostic(tmp_path / "first", total_fes=40)
    second = run_diagnostic(tmp_path / "second", total_fes=40)
    for filename in ("run_results.csv", "case_summary.csv", "diagnosis.json", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_cli_and_invalid_budget(tmp_path: Path):
    output = tmp_path / "cli"
    args = parse_args(["--output-dir", str(output), "--total-fes", "40"])
    assert args.total_fes == 40
    assert main(["--output-dir", str(output), "--total-fes", "40"]) == output
    with pytest.raises(ValueError, match="total_fes"):
        run_diagnostic(tmp_path / "invalid", total_fes=1)
```

- [ ] **Step 2: Run tests and verify import RED**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_011_binary_lsgo_diagnostic.py -q
```

Expected: collection fails because the experiment package does not exist.

- [ ] **Step 3: Implement the fixed matrix and offline classifier**

Create `__init__.py` with the package docstring, then implement `run.py` with these exact public
contracts and constants:

```python
RUN_ID = "exp_011_binary_lsgo_diagnostic"
DIAGNOSTIC_PROBLEM_IDS = ("BLSGO-F08", "BLSGO-F15")
OPTIMIZER_SEEDS = (20_260_713, 20_260_714, 20_260_715, 20_260_716, 20_260_717)
LANES = ("native_single_bit", "native_group_block", "forced_isolate", "arac_policy")
CANONICAL_TOTAL_FES = 2_000
PHASE_ONE_FRACTION = 0.20
SIGNAL_SEED_THRESHOLD = 3


@dataclass(frozen=True)
class DiagnosticSignals:
    optimizer_signal: bool
    policy_signal: bool
    label: str


@dataclass(frozen=True)
class DiagnosticRunRecord:
    result: BinaryLsgoExecutionResult
    input_hash: str
    offline_gain_vs_native: float
    utility_label_vs_native: str
    forbidden_runtime_fields: tuple[str, ...]
    claim_allowed: bool = False
```

Use an explicit classifier:

```python
def classify_diagnostic_signals(
    block_improved_seed_count: int,
    block_accepted_seed_count: int,
    forced_isolate_improved_seed_count: int,
    policy_isolate_not_consumed_seed_count: int,
) -> DiagnosticSignals:
    optimizer_signal = (
        block_improved_seed_count >= SIGNAL_SEED_THRESHOLD
        and block_accepted_seed_count >= SIGNAL_SEED_THRESHOLD
    )
    policy_signal = (
        forced_isolate_improved_seed_count >= SIGNAL_SEED_THRESHOLD
        and policy_isolate_not_consumed_seed_count >= SIGNAL_SEED_THRESHOLD
    )
    if optimizer_signal and policy_signal:
        label = "mixed"
    elif optimizer_signal:
        label = "optimizer_limited"
    elif policy_signal:
        label = "policy_limited"
    else:
        label = "inconclusive"
    return DiagnosticSignals(optimizer_signal, policy_signal, label)
```

Configure lanes without changing the policy or action space:

```python
def _native_decision() -> ActionDecision:
    return ActionDecision(
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "fallback",
        "diagnostic_native_lane",
        0.0,
    )


def _forced_isolate_decision() -> ActionDecision:
    return ActionDecision(
        ActionFamily.ISOLATE,
        "isolate_conflicting_relation",
        "allow",
        "diagnostic_forced_isolate",
        1.0,
    )


def _lane_configuration(lane_id: str) -> tuple[str, ActionDecision | None]:
    if lane_id == "native_single_bit":
        return "single_bit", _native_decision()
    if lane_id == "native_group_block":
        return "group_block", _native_decision()
    if lane_id == "forced_isolate":
        return "single_bit", _forced_isolate_decision()
    if lane_id == "arac_policy":
        return "single_bit", None
    raise ValueError(f"unsupported diagnostic lane: {lane_id}")
```

`execute_diagnostic_matrix(total_fes)` must generate each problem once, run all four lanes for
each seed, pass the operator through `BinaryLsgoExecutionRequest`, pass the optional decision to
`run_binary_lsgo`, and assert one initial hash and one Phase-I objective per problem/seed. Compute
all gains only after the four runs exist, using `native_single_bit` as the offline baseline.

`build_case_summaries(records)` must require 20 records per case and five records per lane, then
calculate:

```python
block_improved_seed_count = sum(record.offline_gain_vs_native > 0.0 for record in block)
block_accepted_seed_count = sum(
    record.result.proposal_trace.multi_bit_accepted_count > 0 for record in block
)
forced_isolate_improved_seed_count = sum(
    record.offline_gain_vs_native > 0.0 for record in forced
)
policy_isolate_not_consumed_seed_count = sum(
    record.result.decision.action_name != "isolate_conflicting_relation"
    or not record.result.optimizer_consumed
    for record in policy
)
```

Each summary must include those four counts, median gain for block/forced/policy, both boolean
signals, `diagnosis_label`, same-budget status, and runtime-boundary status. `diagnosis.json` must
contain `claim_level: "mechanism_diagnosis_only"`, `claim_allowed: false`, the threshold, and the
two case summaries under `case_diagnoses`.

`run_results.csv` must include lane/operator/action identity, initial and Phase-I values, final
objective, six proposal-trace fields, selected online evidence fields, exact Phase-I/II/total FE,
offline gain, utility label, forbidden fields, and `claim_allowed=0`. `manifest.json` must use a
fixed date, sorted JSON keys, input/code hashes, protocol constants, artifact names, and the same
claim boundary. Do not write output-directory paths into deterministic artifacts.

- [ ] **Step 4: Run exp_011 tests and verify GREEN**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_011_binary_lsgo_diagnostic.py -q
```

Expected: all new experiment tests pass.

- [ ] **Step 5: Commit the runner**

```powershell
git add -- experiments/exp_011_binary_lsgo_diagnostic tests/test_exp_011_binary_lsgo_diagnostic.py
git diff --cached --check
git commit -m "feat: add binary LSGO mechanism diagnostic"
```

## Task 4: Document and run the canonical diagnosis

**Files:**
- Modify: `experiments/README.md`
- Generated, not committed: `results/exp_011_binary_lsgo_diagnostic/*`

- [ ] **Step 1: Update experiment inventory and command**

Add `exp_011_binary_lsgo_diagnostic/` to the layout and this command:

```powershell
$env:PYTHONPATH='src'
& 'E:\ARAC\.venv\Scripts\python.exe' -m experiments.exp_011_binary_lsgo_diagnostic.run `
  --output-dir results/exp_011_binary_lsgo_diagnostic `
  --total-fes 2000
```

State that the 40 executions cover F08/F15, five fixed seeds, and four causal lanes; output is
offline mechanism diagnosis only and cannot promote ARAC or treat group-block as an ARAC action.

- [ ] **Step 2: Run focused regression tests**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_binary_lsgo_backend.py tests/test_exp_009_binary_lsgo_arac_pilot.py tests/test_exp_010_binary_lsgo_focused_3seed.py tests/test_exp_011_binary_lsgo_diagnostic.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run the canonical 40-execution matrix**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m experiments.exp_011_binary_lsgo_diagnostic.run --output-dir results/exp_011_binary_lsgo_diagnostic --total-fes 2000
```

Expected: four artifacts are created; `run_results.csv` has 40 rows, no budget violation, no
forbidden runtime field, and no claim-allowed row.

- [ ] **Step 4: Run full verification**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest -q
E:\ARAC\.venv\Scripts\python.exe -m compileall src experiments/exp_011_binary_lsgo_diagnostic tests
git diff --check
git status --short
```

Expected: tracked tests pass, compileall exits 0, diff check exits 0, and generated results remain
ignored.

- [ ] **Step 5: Review results and commit documentation**

Read `case_summary.csv` and `diagnosis.json`; verify every label follows the registered counts.
Then inspect the exact diff and commit:

```powershell
git add -- experiments/README.md
git diff --cached --check
git commit -m "docs: document binary LSGO mechanism diagnosis"
```

Do not push until the user explicitly confirms remote synchronization.
