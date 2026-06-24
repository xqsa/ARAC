# Method Boundaries


## Positive Scope

`ARAC` is about:

- uncertain overlap structures
- shared-variable conflict
- relation-level and group-level dynamic evidence
- reference-blind action selection
- backend intervention semantics
- same-budget utility validation
- negative control and catastrophic-risk auditing

## Negative Scope

`ARAC` is not:

- a new low-level optimizer by itself
- a decomposition-only method
- a wrapper that chooses the best result after seeing final outcomes
- a problem-id or family-label special-case table
- a copy of the historical HCC Mxx milestone chain
- a claim that the old Route A branch is final-success ready

## Runtime Legality

Runtime policy may use:

- Phase-I trace features
- relation and group behavior features
- shared-variable evidence features
- budget state features
- feature-coverage and reliability indicators
- pre-registered action contracts

Runtime policy must not use:

- final error
- relative gain
- oracle best action
- user reference best
- reported baseline values
- final success labels
- problem-family labels
- problem-id special cases
- prior final or pilot outcomes

Problem identifiers may appear only for execution identity, file grouping, and
offline audit joins. They must not be used as policy triggers.

## Backend Boundary

Core actions are:

- `coordinate`
- `isolate`
- `protect`
- `reassign_repair`
- `fallback`

Backend optimizers and support components are not core actions:

- selector
- baseline optimizer
- executor
- cache materializer
- final evaluator

They may execute or support an action, but they are not the innovation by
themselves.

## Claim Ladder

Use the following claim ladder exactly:

| Level | Name | Meaning |
| --- | --- | --- |
| 1 | schema complete | Tables and fields are defined. |
| 2 | preflight valid | Files, hashes, anti-leakage, and contract checks pass. |
| 3 | runtime connected | Policy is connected to backend semantics. |
| 4 | fresh same-budget smoke | Real optimizer run, fresh provenance, FE ledger clear. |
| 5 | pilot-level evidence | Limited problem set supports local utility. |
| 6 | final evaluation completed | Full configured evaluation ran to completion. |
| 7 | final success claim | Meaningful wins and all safety gates pass. |

Never report a lower level as a higher-level success.

## Failure Honesty

Separate these claims:

- concept correctness
- protocol correctness
- runtime legality
- local utility
- final performance

A green light at one layer does not imply success at another layer.
