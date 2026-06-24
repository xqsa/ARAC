# ARAC Project Plan

## Goal

Build a clean, portable implementation of reference-blind
evidence-to-intervention utility mapping for uncertain overlapping structures.

## Current Status

Created as a scaffold extracted from `E:\HCC-main` on 2026-06-24.

This folder currently contains:

- core method framing
- runtime and claim boundaries
- minimum truth-table schemas
- minimal Python package skeleton
- smoke tests for policy and leakage gate behavior

It does not contain a real optimizer backend yet.

## Stage Gates

### Stage 1: Schema Smoke

Acceptance:

- `evidence_profile.csv`, `action_decision.csv`, `backend_semantics_diff.csv`,
  `same_budget_ledger.csv`, and `action_utility_audit.csv` can be generated
  from a tiny synthetic trace.
- Forbidden runtime fields are rejected.

### Stage 2: Backend Binding Smoke

Acceptance:

- At least one active action changes optimizer-consumed backend semantics.
- `backend_semantics_diff.csv` records the change.
- Fallback action remains allowed without active semantics.

### Stage 3: Fresh Same-Budget Smoke

Acceptance:

- Phase-I and Phase-II FE are both counted.
- Fresh execution is distinguishable from cache/materialization.
- Same-budget violation is checked automatically.

### Stage 4: Utility and Risk Pilot

Acceptance:

- Actions are compared against fallback or no-action lanes.
- `meaningful_win` requires relative gain >= 0.05.
- `catastrophic_loss` blocks escalation.
- Negative controls pass.

## Next Action

Implement `experiments/exp_001_schema_smoke/` to generate the five truth tables
from a tiny synthetic trace and run `tests/test_policy_smoke.py`.
