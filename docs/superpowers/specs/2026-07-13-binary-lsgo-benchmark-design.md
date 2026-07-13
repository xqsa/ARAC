# Binary Overlapping LSGO Benchmark Design

- Date: 2026-07-13
- Executor: Codex
- Status: approved design
- Scope: native Python benchmark only; no HCC optimizer integration

## Goal

Port the inherited binary large-scale global optimization (LSGO) test-function
construction method into ARAC as an independent, deterministic benchmark. The
benchmark must provide a problem generator, objective evaluator, fixed-seed
topology generation, auditable topology metadata, and the 18 standard parameter
combinations described in the inherited thesis.

The benchmark has a role similar to AOB as an experimental problem source, but
it remains a binary-domain benchmark. It must not be routed through the current
continuous CMA-ES/MMES HCC execution path.

## Source Evidence

The implementation is grounded in these read-only files:

- `C:\Users\83718\Desktop\继承\WYQ\测试集代码\test_function\test_function.m`
- `C:\Users\83718\Desktop\继承\WYQ\测试集代码\test_function\deception_map.m`
- `C:\Users\83718\Desktop\继承\WYQ\测试集代码\test_function\grouping.m`
- `C:\Users\83718\Desktop\继承\WYQ\测试集代码\test1\generateGroups.m`
- `C:\Users\83718\Desktop\继承\WYQ\测试集代码\test1\repeat.m`
- `C:\Users\83718\Desktop\继承\WYQ\5.28大论文.docx`, Chapter 4 and Table 4-1
- `C:\Users\83718\Desktop\继承\汪雨琪.pptx`, slides 17-23

The earlier 2018 files establish the binary template, grouping, and deceptive
mapping. The later WYQ files add repeated variables and complex overlap. The
thesis defines the intended controllable properties and the 18 standard cases.

## Selected Approach

Use a native Python implementation under `src/arac/benchmarks/`. It will use the
Python standard library only and will not require MATLAB, Octave, NumPy, or an
external process.

Rejected alternatives:

- Calling MATLAB from Python preserves the old environment dependency and makes
  automated tests and deployment fragile.
- Introducing a general benchmark framework before this port adds abstractions
  that are not required by the current task.

## Public API

The implementation will live in:

```text
src/arac/benchmarks/binary_lsgo.py
```

It will expose:

```python
BinaryLsgoSpec
BinaryLsgoTopology
BinaryLsgoProblem
generate_binary_lsgo(spec)
standard_binary_lsgo_specs()
```

`BinaryLsgoSpec` is the complete generation contract. It contains:

- `problem_id`
- `nominal_dimension`
- `overlap_count`
- `min_group_size`
- `max_group_size`
- `continuous_groups`
- `alpha`
- `overlap_distribution_ratio`
- `related_group_ratio`
- `max_repeat_ratio`
- `seed`

All generation inputs are explicit. A problem instance does not read global
random state, environment variables, historical results, or benchmark labels.

`BinaryLsgoProblem` contains an immutable spec, template, and topology and
provides:

```python
evaluate(vector) -> float
evaluate_batch(vectors) -> tuple[float, ...]
```

## Dimension Semantics

The inherited MATLAB implementation uses `n` for the total number of group
membership slots and `zn = n - C` for the actual binary vector length. This is
easy to confuse with conventional benchmark dimension terminology, so the port
will preserve the behavior while naming each quantity explicitly:

- `nominal_dimension`: inherited `n`; total group membership slots.
- `overlap_count`: inherited `C`; membership slots replaced by repeated
  variables.
- `decision_dimension`: `nominal_dimension - overlap_count`; required input
  vector length.
- `membership_count`: sum of all generated group sizes; equal to
  `nominal_dimension`.
- `shared_variable_count`: number of distinct variables occurring in more than
  one group.

The generator rejects `overlap_count >= nominal_dimension`. The topology
metadata records all five values so downstream experiments cannot silently
confuse repeated memberships with independent decision variables.

## Deterministic Group Generation

Each call creates a local `random.Random(seed)` instance. Repeating generation
with the same spec must return identical group sizes, variable ordering,
template, duplicate assignments, and topology metadata.

Generation proceeds as follows:

1. Calculate the group count as
   `ceil(2 * nominal_dimension / (min_group_size + max_group_size))`.
2. For equal groups, fill groups with `min_group_size` memberships and allow a
   final residual group when necessary.
3. For unequal groups, sample sizes within the inclusive bounds and adjust only
   eligible groups until the sizes sum to `nominal_dimension`.
4. Create exactly `decision_dimension` unique zero-based variable indices.
   Preserve their order when `continuous_groups=True`; otherwise shuffle them
   once with the local random generator.
5. Reserve exactly `overlap_count` group slots for repeated variables. The
   `overlap_distribution_ratio` controls whether those slots are concentrated
   in fewer groups or spread over more groups.
6. Fill all non-reserved slots so every decision variable occurs at least once.
7. Fill reserved slots with variables from other groups. The
   `related_group_ratio` controls how many source groups contribute to a target
   group. `max_repeat_ratio` bounds how many groups may contain one variable.
8. Reject impossible specifications before generation rather than retrying
   without a bound. Candidate sets are computed explicitly, so generation
   cannot enter an unbounded MATLAB-style `while` loop.

The three overlap ratios are constrained to `(0, 1]`. Their integer effects are
recorded in topology metadata. The default for each ratio is `0.5`, matching the
middle of the inherited random parameter range while making the standard suite
reproducible.

No group may contain the same variable twice. A shared variable may occur in
multiple different groups.

## Objective Function

For each group of size `m`:

1. Select the candidate and template bits indexed by that group.
2. Count matching positions `u = m - hamming_distance`.
3. Set `a = 0.9 * m` and `z = 10 * alpha * m / 9`.
4. Calculate the group contribution:

```text
alpha = 0:        contribution = u
0 < alpha < 0.9:
  u < z:          contribution = -(a / z) * u + a
  otherwise:      contribution = (m / (m - z)) * u - (m * z / (m - z))
```

The full objective is the negative sum of group contributions, preserving the
inherited minimization convention.

Candidate and template bits must be exact integers or booleans equal to `0` or
`1`. Wrong lengths and non-binary values raise `ValueError`. Empty batches
return an empty tuple.

## Standard 18-Case Suite

`standard_binary_lsgo_specs()` returns immutable specs named `BLSGO-F01` through
`BLSGO-F18`. All use `nominal_dimension=1000`, `continuous_groups=True`, fixed
per-case seeds, and overlap-control ratios of `0.5`.

The cases follow the inherited Table 4-1 matrix:

| Cases | Alpha | Group bounds | Overlap |
| --- | ---: | --- | ---: |
| F01-F03 | 0.1 | 5, 5 | 10%, 20%, 30% |
| F04-F06 | 0.1 | 2, 5 | 10%, 20%, 30% |
| F07-F09 | 0.5 | 5, 5 | 10%, 20%, 30% |
| F10-F12 | 0.5 | 2, 5 | 10%, 20%, 30% |
| F13-F15 | 0.8 | 5, 5 | 10%, 20%, 30% |
| F16-F18 | 0.8 | 2, 5 | 10%, 20%, 30% |

The resulting decision dimensions are 900, 800, and 700 for the 10%, 20%, and
30% cases respectively. This is intentional source compatibility and is exposed
in metadata rather than described as a fixed 1000-dimensional input suite.

## Topology Metadata

`BinaryLsgoTopology` records:

- ordered groups and group sizes
- decision and membership dimensions
- overlap slot count
- distinct shared-variable indices and count
- occurrence count for every variable
- groups containing each shared variable
- group adjacency induced by shared variables
- realized number of groups containing shared variables
- realized maximum variable occurrence count
- source semantics identifier

This metadata is structural evidence. It does not include optimizer outcomes,
oracle actions, paper-reported values, final errors, or prior run results.

## Integration Boundary

This version will update `src/arac/benchmarks/__init__.py` only to make the
benchmark module discoverable. It will not modify:

- the HCC subprocess runner
- AOB data files or AOB adapters
- CMA-ES/MMES objective handling
- ARAC runtime policy inputs
- reported baseline tables

A future binary optimizer adapter must be designed and validated separately.

## Tests

Tests will be added in `tests/test_binary_lsgo_benchmark.py` using test-first
development. Required behaviors are:

1. Invalid dimensions, bounds, ratios, alpha, vector lengths, and bit values are
   rejected with clear errors.
2. Equal and unequal group sizes satisfy their contracts and sum to the nominal
   dimension.
3. Every decision variable appears at least once and no group contains a
   duplicate index.
4. Generated indices stay within `[0, decision_dimension)`.
5. The requested overlap slot count is realized exactly.
6. Identical specs generate identical templates, groups, and metadata.
7. Different seeds change at least one generated structural field.
8. Continuous and shuffled grouping modes behave as documented.
9. Hand-calculated objective examples match the inherited piecewise formula for
   `alpha=0`, `0.1`, and `0.8`.
10. Single and batch evaluation agree.
11. The standard suite contains exactly 18 unique, correctly parameterized
    cases.
12. Topology metadata is internally consistent and contains no runtime-forbidden
    outcome fields.

The focused test file must pass before the full local test suite is run.

## Verification And Delivery

Implementation completion requires fresh evidence from:

```powershell
python -m pytest tests/test_binary_lsgo_benchmark.py -q
python -m pytest -q
git diff --check
git status --short
```

Before commit, the diff will be checked for accidental changes to `results/`,
cache files, source documents, HCC/AOB assets, and unrelated user work. Only
files belonging to this benchmark port will be staged.

## Known Limits

- Fixed seeds guarantee Python-port reproducibility, not bit-for-bit equality
  with MATLAB's random-number stream.
- The inherited source contains several exploratory variants and potentially
  unbounded retry loops. This design preserves the documented semantics while
  replacing those loops with bounded, validated construction.
- Historical MATLAB performance results are not imported as new ARAC results.
- This version evaluates binary vectors but does not provide an optimizer.
