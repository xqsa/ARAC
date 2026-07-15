---
session_no: S18
suggested_title: "[ARAC] S19 trace component delayed credit"
parent_session: S17
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

The existing v33-v38 and CAR v2 raw artifacts have been audited for runtime
state sufficiency without consuming new FE. Adding state evidence is supported
in principle, but a new controller is blocked until component persistence and
interference state are explicitly logged. CAR full-24 and threshold tuning
remain stopped.

## Completed

- Added the offline-only `scripts/audit_state_evidence.py` and focused tests.
- Generated `results/state_evidence_sufficiency_audit_20260715` with field
  coverage, run-level paired state features, CAR pre-action features, and
  delayed-horizon associations.
- Confirmed eight declared action-trace state fields are empty across every
  audited artifact, including budget/decision, `cc_utility`, and search-state
  conflict/writeback/utility fields.
- Repaired the offline comparison boundary: v37 is compared with v36 and v38
  with v37, so the final action is not credited with cumulative earlier-version
  changes.
- Found that v34 writeback recovery credit does not order terminal utility
  (Spearman -0.234; within-case 10/21), while v37 resource retirement is nearly
  neutral (3 wins, 4 losses, mean log advantage -0.0000586).
- Found a narrow search-start signal: v38 precision local log gain versus the
  v38-v37 terminal increment has Spearman +0.741 and within-case concordance
  9/11. S2 contributes 91.5% of absolute terminal effect; excluding it leaves
  Spearman +0.531. This is descriptive pilot evidence, not a threshold.
- CAR pre-action mean rank gap is a hypothesis only (Spearman +0.492, 6/8
  within-case pairs). S3 supplies 92.6% of absolute CAR terminal contrast and
  there are only two repair-action samples.
- Recorded the component-locked, action-specific delayed-credit candidate and
  the release block in the core method and formal audit report.

## Decision

Do not build `group_id -> action`, do not fit a repair threshold from S3 seeds
1/3, and do not combine W/R/S under one credit. The first eligible candidate is
search-start exposure control: an existing structural route may grant one
capped lease, then another lease is allowed only after resolved component
credit is positive and neighbour/overwrite harm is absent.

## Next steps

1. Add trace-only, zero-behaviour-change component state: decision FE and
   remaining budget, pending action id, resolution FE, proposal disagreement,
   component/neighbor gain, shared-variable overwrite and survival.
2. Verify trace parity on CLI/5k before changing any action semantics.
3. Freeze one search-start delayed-credit rule and use held-out seeds with a
   same-budget paired reference; report coverage, abstention, worst seed and
   catastrophic loss before any larger matrix.
4. Keep the July paper on v33.8 full-24 performance and use CAR/state audits as
   mechanism limitations and future-work evidence.

## Required reading

1. `.light/handoff/S18-arac-state-evidence-sufficiency-audit.md`
2. `docs/design/state-evidence-sufficiency-audit-20260715.md`
3. `docs/design/car-actionability-audit-result-20260715.md`
4. `docs/design/core-method.md`
5. `.light/passport.yaml`

## Prohibited

- No case/function/seed/paper-best/history/final-outcome runtime dispatch.
- No relation/group rows treated as independent samples.
- No new full-24, CAR R/S expansion, or threshold tuning from this audit.
- No shared-variable writeback by adjacent groups without a component lock.
