# Binary Overlapping LSGO Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the inherited binary overlapping LSGO test-function generator into ARAC as a deterministic Python benchmark with topology metadata, objective evaluation, and 18 standard cases.

**Architecture:** Keep the benchmark self-contained in `src/arac/benchmarks/binary_lsgo.py`. `BinaryLsgoSpec` validates immutable generation inputs, a seeded generator builds a `BinaryLsgoTopology` and template, and `BinaryLsgoProblem` evaluates one or many binary vectors. Tests exercise the generator, topology invariants, objective formula, and standard suite without touching HCC/AOB runtime code.

**Tech Stack:** Python 3.11 standard library, dataclasses, `random.Random`, pytest, existing `src` package layout.

---

## File Map

- Create: `src/arac/benchmarks/binary_lsgo.py` — public spec, topology, problem, generator, evaluator, and standard 18-case factory.
- Modify: `src/arac/benchmarks/__init__.py` — expose benchmark module symbols without importing HCC/AOB.
- Create: `tests/test_binary_lsgo_benchmark.py` — focused test-first coverage for validation, deterministic generation, topology, objective, batch evaluation, and standard cases.
- Modify: `README.md` — add the benchmark to the clean pipeline inventory and state the binary/HCC boundary.

## Task 1: Establish the public contract with failing tests

**Files:**
- Create: `tests/test_binary_lsgo_benchmark.py`
- Create: `src/arac/benchmarks/binary_lsgo.py` only after the red test run

- [ ] **Step 1: Write validation and object-shape tests.** Add tests that import `BinaryLsgoSpec`, `BinaryLsgoProblem`, `BinaryLsgoTopology`, and `generate_binary_lsgo`, then assert:

```python
def test_spec_rejects_invalid_generation_inputs() -> None:
    with pytest.raises(ValueError, match="nominal_dimension"):
        BinaryLsgoSpec("bad", 4, 4, 1, 2, True, 0.1, 0.5, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="alpha"):
        BinaryLsgoSpec("bad", 10, 1, 2, 5, True, 0.9, 0.5, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="ratio"):
        BinaryLsgoSpec("bad", 10, 1, 2, 5, True, 0.1, 0.0, 0.5, 0.5, 1)


def test_generated_problem_exposes_explicit_dimension_semantics() -> None:
    spec = BinaryLsgoSpec("small", 20, 4, 2, 5, True, 0.1, 0.5, 0.5, 0.5, 7)
    problem = generate_binary_lsgo(spec)

    assert problem.decision_dimension == 16
    assert problem.topology.nominal_dimension == 20
    assert problem.topology.membership_count == 20
    assert len(problem.template) == 16
```

- [ ] **Step 2: Run only the new tests and verify the expected red state.**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py -q
```

Expected: collection fails because `arac.benchmarks.binary_lsgo` does not yet exist. Fix only test syntax/import mistakes if needed; do not add production code before the missing-module failure is observed.

- [ ] **Step 3: Add the minimum module declarations.** Create the module with typed frozen dataclasses and function declarations that raise `NotImplementedError`; this makes later test failures occur at behavior boundaries rather than import errors.

- [ ] **Step 4: Re-run the two contract tests.**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py::test_spec_rejects_invalid_generation_inputs tests/test_binary_lsgo_benchmark.py::test_generated_problem_exposes_explicit_dimension_semantics -q
```

Expected: validation test fails on missing validation and shape test fails on unimplemented generation. This confirms the tests are red for the intended reasons.

## Task 2: Implement deterministic group and overlap generation

**Files:**
- Modify: `src/arac/benchmarks/binary_lsgo.py`
- Test: `tests/test_binary_lsgo_benchmark.py`

- [ ] **Step 1: Add topology invariant tests before implementation.** Use a small spec (`nominal_dimension=40`, `overlap_count=8`, bounds `2..5`) and assert:

```python
def test_topology_has_valid_groups_and_exact_overlap_slots() -> None:
    problem = generate_binary_lsgo(BinaryLsgoSpec("topology", 40, 8, 2, 5, True, 0.5, 0.5, 0.5, 0.5, 11))
    topology = problem.topology
    assert sum(topology.group_sizes) == 40
    assert all(2 <= size <= 5 for size in topology.group_sizes)
    assert topology.membership_count == 40
    assert topology.overlap_slot_count == 8
    assert all(0 <= index < 32 for group in topology.groups for index in group)
    assert all(len(group) == len(set(group)) for group in topology.groups)
    assert all(count >= 1 for count in topology.variable_occurrence_counts.values())
    assert topology.max_variable_occurrence_count >= 2
```

Add separate tests for equal bounds, shuffled order, and invalid impossible configurations.

- [ ] **Step 2: Run the topology tests and confirm they fail because generation is still unimplemented.**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py -k 'topology or group or shuffle' -q
```

Expected: failures point to the generator, not malformed assertions.

- [ ] **Step 3: Implement spec validation and seeded base partitioning.** Use `random.Random(spec.seed)`. Validate positive dimensions, `0 <= overlap_count < nominal_dimension`, `1 <= min_group_size <= max_group_size`, feasible group count, `0 <= alpha < 0.9`, ratios in `(0, 1]`, and nonnegative integer seed. Calculate `group_count = ceil(2*n/(min+max))`; generate bounded group sizes summing to `n`; create `decision_dimension = n - C` unique indices; preserve or shuffle the index order according to `continuous_groups`.

- [ ] **Step 4: Implement bounded repeated-variable assignment.** Reserve exactly `C` slots across eligible groups using `overlap_distribution_ratio`; fill every non-reserved slot with each unique variable exactly once; fill reserved slots from other groups. Use `related_group_ratio` to determine source-group count and `max_repeat_ratio` to derive a bounded maximum occurrence count. Raise `ValueError` when the requested topology cannot be constructed instead of retrying indefinitely.

- [ ] **Step 5: Compute topology metadata.** Populate ordered groups, group sizes, variable occurrence counts, shared-variable groups, adjacency pairs, realized shared-group count, realized maximum occurrence count, and the source semantics identifier. Assert internal invariants in the constructor or generator before returning.

- [ ] **Step 6: Run the topology tests and make them green.**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py -k 'topology or group or shuffle' -q
```

Expected: all selected tests pass with deterministic output.

## Task 3: Implement the deceptive binary objective

**Files:**
- Modify: `src/arac/benchmarks/binary_lsgo.py`
- Test: `tests/test_binary_lsgo_benchmark.py`

- [ ] **Step 1: Add hand-calculated objective tests.** Generate a small seeded problem, then evaluate its template, complement, and a one-bit mutation. Define the reference contribution in the test itself so expected values do not call production code. This covers the no-deception case and both branches of the piecewise formula. Also assert minimization sign and binary/length validation:

```python
def _reference_contribution(group_size: int, matching: int, alpha: float) -> float:
    if alpha == 0:
        return float(matching)
    local_optimum = 0.9 * group_size
    deception_point = 10 * alpha * group_size / 9
    if matching < deception_point:
        return -(local_optimum / deception_point) * matching + local_optimum
    return (
        (group_size / (group_size - deception_point)) * matching
        - (group_size * deception_point / (group_size - deception_point))
    )


def test_objective_matches_piecewise_deception_formula() -> None:
    problem = generate_binary_lsgo(
        BinaryLsgoSpec("objective", 20, 4, 2, 5, True, 0.5, 0.5, 0.5, 0.5, 3)
    )
    assert problem.evaluate(problem.template) == pytest.approx(-20.0)

    complement = tuple(1 - bit for bit in problem.template)
    assert problem.evaluate(complement) == pytest.approx(-18.0)

    mutated = list(problem.template)
    mutated[0] = 1 - mutated[0]
    expected = 0.0
    for group in problem.topology.groups:
        matching = len(group) - (1 if 0 in group else 0)
        expected += _reference_contribution(len(group), matching, 0.5)
    assert problem.evaluate(mutated) == pytest.approx(-expected)


def test_objective_rejects_non_binary_or_wrong_length_vectors() -> None:
    problem = generate_binary_lsgo(
        BinaryLsgoSpec("validation", 20, 4, 2, 5, True, 0.1, 0.5, 0.5, 0.5, 4)
    )
    with pytest.raises(ValueError, match="length"):
        problem.evaluate((0, 1))
    with pytest.raises(ValueError, match="binary"):
        problem.evaluate((2,) + problem.template[1:])
```

- [ ] **Step 2: Run the objective tests and verify the intended red state.**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py -k 'objective or binary' -q
```

Expected: failures show the evaluator is not implemented or returns the wrong value.

- [ ] **Step 3: Implement scalar evaluation.** Validate the candidate length and each bit. For each group compute `m`, mismatch count, matching count `u`, and the inherited piecewise contribution. Use `contribution=u` for `alpha=0` to make the no-deception case explicit. Return the negative sum as `float`.

- [ ] **Step 4: Implement batch evaluation.** Evaluate each vector through the scalar path, return a tuple, and return `()` for an empty iterable. Do not mutate the problem, template, topology, or caller-owned vectors.

- [ ] **Step 5: Run the objective tests and make them green.**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py -k 'objective or binary' -q
```

Expected: all selected tests pass.

## Task 4: Add deterministic standard cases and package exports

**Files:**
- Modify: `src/arac/benchmarks/binary_lsgo.py`
- Modify: `src/arac/benchmarks/__init__.py`
- Test: `tests/test_binary_lsgo_benchmark.py`

- [ ] **Step 1: Add standard-suite tests before the factory implementation.** Assert exactly 18 unique IDs, alpha sequence `0.1/0.5/0.8`, equal and unequal bounds in the correct rows, overlap counts `100/200/300`, fixed seeds, and generated decision dimensions `900/800/700`.

- [ ] **Step 2: Run the standard-suite tests and verify red.**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py -k standard -q
```

Expected: failure because `standard_binary_lsgo_specs()` is not implemented.

- [ ] **Step 3: Implement `standard_binary_lsgo_specs()`.** Construct the 18 immutable specs in the table order `F01` through `F18`, use per-case deterministic seeds derived from a fixed tuple of integer seeds, and set all three overlap ratios to `0.5`. Do not encode any optimizer result or HCC/AOB family label in the specs.

- [ ] **Step 4: Export the public symbols.** Update `src/arac/benchmarks/__init__.py` with explicit imports and `__all__`, without importing optional HCC dependencies.

- [ ] **Step 5: Add reproducibility and metadata-boundary tests.** Assert identical specs produce identical `template`, `groups`, and metadata; different seeds alter at least one structural field; metadata does not contain `final_error`, `relative_gain`, `oracle`, `reported_baseline`, or prior outcome fields.

- [ ] **Step 6: Run all focused tests.**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py -q
```

Expected: all focused tests pass.

## Task 5: Document the benchmark boundary

**Files:**
- Modify: `README.md`
- Modify: `src/arac/benchmarks/binary_lsgo.py` module docstring if needed

- [ ] **Step 1: Add a short README inventory entry.** State that the benchmark is an independent binary-domain problem source, exposes seeded topology and objective evaluation, and is not connected to the continuous HCC runner.

- [ ] **Step 2: Add provenance comments only where needed.** Keep the source paths and the source-semantics distinction in the module docstring; do not copy the historical MATLAB tree into ARAC.

- [ ] **Step 3: Run focused tests after documentation changes.**

Run:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py -q
```

Expected: same focused pass count as Task 4.

## Task 6: Full verification and scoped commit

**Files:**
- Inspect only: all changed benchmark files and focused tests

- [ ] **Step 1: Run the full local test suite.**

Run:

```powershell
python -m pytest -q
```

Expected: exit code 0 and no test failures. Any failure must be diagnosed before proceeding.

- [ ] **Step 2: Run repository hygiene checks.**

Run:

```powershell
git diff --check
git status --short
```

Confirm that no `results/`, cache, MATLAB source, large artifact, or unrelated user file is staged.

- [ ] **Step 3: Review the final diff.**

Run:

```powershell
git diff -- src/arac/benchmarks/binary_lsgo.py src/arac/benchmarks/__init__.py tests/test_binary_lsgo_benchmark.py README.md
```

Check that runtime code does not import HCC/AOB, no forbidden outcome field enters metadata, and all public names match the tests and plan.

- [ ] **Step 4: Stage only the benchmark port.**

Run:

```powershell
git add src/arac/benchmarks/binary_lsgo.py src/arac/benchmarks/__init__.py tests/test_binary_lsgo_benchmark.py README.md
git diff --cached --check
```

- [ ] **Step 5: Commit the implementation.**

Run:

```powershell
git commit -m "feat: add deterministic binary LSGO benchmark"
```

- [ ] **Step 6: Report verification evidence and push status.** Include the exact focused/full pytest results, commit ID, changed files, and any remaining limitation. Do not push `origin/main` without the required user confirmation for external Git state changes.
