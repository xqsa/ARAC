# ARAC

`ARAC` is a clean extraction package for the core idea pulled out of the
original `E:\HCC-main` / MI-ARAC-Action workspace.

The package keeps the method, boundaries, schemas, and minimum code skeleton.
It intentionally does not copy the historical Mxx milestone chain, failed
repair branches, or HCC-specific artifact clutter.

## Core Idea

The core innovation is not a new low-level optimizer and not decomposition
alone. It is a reference-blind utility mapping from uncertain overlapping
structure evidence to executable backend intervention actions.

In one sentence:

```text
Map Phase-I trace-derived evidence about shared variables, overlap relations,
group behavior, and resource state into coordinate, isolate, protect,
reassign/repair, or fallback backend interventions, then verify the actions
under same-budget execution with leakage, semantics, negative-control, and
catastrophic-loss audits.
```

## What This Folder Keeps

- `docs/core-method.md`: clean research framing and contribution statements.
- `docs/boundaries.md`: hard boundaries for runtime legality and claim levels.
- `docs/schemas.md`: truth-table schemas for evidence, decisions, execution,
  utility, and audits.
- `docs/hcc-backbone-extraction.md`: first extraction contract between the HCC
  grouping/optimizer backbone and the ARAC evidence/action layer.
- `src/arac/`: minimum Python skeleton for evidence extraction, action space,
  policy mapping, backend adapter, evaluation, and audit.
- `configs/default.yaml`: minimal experiment contract.
- `experiments/README.md`: how to structure future runs without returning to
  the old Mxx clutter.
- `references/source-index.md`: pointers to the original source evidence inside
  `E:\HCC-main`.

## Clean Pipeline

```text
trace data
  -> evidence_profile.csv
  -> action_decision.csv
  -> backend execution / semantic diff
  -> same_budget_ledger.csv
  -> action_utility_audit.csv
  -> final claim gate
```

## Runtime Inputs

Allowed runtime inputs are reference-blind features derived from optimization
process traces, such as:

- overlap degree
- shared-variable support
- direction disagreement
- harmful coordination score
- group gain asymmetry
- priority spread
- rank stability
- feature coverage
- budget remaining ratio
- fallback margin proxy

Forbidden runtime inputs include final answers or identity shortcuts:

- final error
- relative gain
- oracle labels
- user reference best
- reported baseline values
- problem family labels
- problem-id special cases
- prior final/pilot outcomes

## Minimum Action Taxonomy

- `coordinate`: allow or strengthen beneficial cooperation.
- `isolate`: isolate harmful overlap or conflicting relations.
- `protect`: protect important groups, shared variables, or budget lanes.
- `reassign_repair`: repair ownership, grouping, or shared-variable binding.
- `fallback`: abstain or use a conservative backend when evidence is unsafe.

Backend optimizers, selectors, and executors are not core actions by themselves.
They are support surfaces.

## Claim Boundary

This package is a method scaffold. It does not claim final performance success.

Valid claim ladder:

1. schema complete
2. preflight valid
3. runtime connected
4. fresh same-budget smoke
5. pilot-level evidence
6. final evaluation completed
7. final success claim

Only the last level may be called final success, and only if meaningful wins,
catastrophic-loss gates, same-budget accounting, anti-leakage, and backend
semantics audits all pass.
