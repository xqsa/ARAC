# AOB Final Evaluation Protocol

Date: 2026-06-25
Author: Codex

## Purpose

This protocol fixes the benchmark contract for evaluating ARAC on the
Two-Phase CC paper's Auto Overlapping Benchmark (AOB). It also separates the
current pilot from the final evaluation target.

Current pilot: 1 independent run.

Final protocol: 25 independent runs.

The 1-run pilot is only a first-look utility and failure-surface check. It must
not be described as final performance evidence.

## Benchmark

AOB contains 24 cases:

```text
E1-E6 = Elliptic with overlap levels 1-6
S1-S6 = Schwefel with overlap levels 1-6
R1-R6 = Rastrigin with overlap levels 1-6
A1-A6 = Ackley with overlap levels 1-6
```

The benchmark dimension is `D = 1000`, with search range `[-100, 100]^D`.

The overlap levels are:

| AOB id | Gamma per adjacent subspace |
| --- | ---: |
| 1 | 0 |
| 2 | 1 |
| 3 | 3 |
| 4 | 5 |
| 5 | 7 |
| 6 | 10 |

## Budget

Each pilot row uses:

```text
total_fe = 3,000,000
run_count = 1
seed = 1
```

The final protocol keeps the paper's `25` independent runs, but that expansion
is gated on the 1-run pilot showing enough evidence to justify the cost.

For ARAC-HCC, Phase-I evidence collection and Phase-II action execution must be
accounted together under the same `3,000,000` FE budget. Decomposition FE follows
the paper setting and is not counted as optimizer TFEs.

## Comparison Baselines

The Table 2 baselines are paper-reported evaluation-only baselines. They are
not rerun in this project during the pilot.

Current committed anchor:

```text
references/paper_reported_table2_hcc_es.csv
```

This file currently records the paper-reported HCC-ES mean, standard deviation,
and time for all 24 AOB cases. Additional Table 2 methods can be added later,
but HCC-ES is the first required anchor because it is the direct HCC backbone
comparison.

Paper-reported values must not enter runtime dispatch. They may be used only
after our run finishes, inside final or pilot evaluation tables.

## Runtime Boundary

Runtime policy must remain reference-blind. The action policy may use only
runtime-observable trace, grouping, overlap, shared-variable, and budget
features.

Forbidden runtime inputs include:

- final error
- relative gain
- oracle labels
- user reference best
- reported baseline values
- problem-family labels
- prior final outcomes
- prior pilot outcomes

`problem_id` may be used for execution identity, artifact naming, and final
table joining. It must not be used as a policy shortcut.

## Pilot Completion Surface

The 1-run pilot should produce:

- our final result per AOB case
- same-budget ledger
- backend semantics diff
- anti-leakage audit
- paper-reported HCC-ES comparison
- catastrophic-loss audit
- negative-control status

Allowed pilot claim:

```text
1-run AOB pilot signal under paper-matched budget
```

Forbidden pilot claims:

- final success
- SOTA
- 25-run statistical significance
- reproduced Table 2 baselines
- runtime policy used paper-reported values
