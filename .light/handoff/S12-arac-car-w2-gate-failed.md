---
session_no: S12
suggested_title: "[ARAC] S13 CAR-W3 first-pair futility redesign"
parent_session: S11
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

CAR-W2 3M diagnostic completed with integrity pass but utility failure. The
lazy lease fixed no-plan overhead, yet 11 stable graph plans all spent the full
three-pair probe and all abstained. The pipeline is stopped before R/S and
full-24.

## Evidence

- Results: `results/car_w2_diagnostic_6case_seed22_24_3m_20260715`
- Report: `docs/design/car-w2-diagnostic-result-20260715.md`
- 90/90 fresh; FE violations 0/90; AOB 885/885; anti-leakage 16/16;
  CAR boundary 28/28; branch actual/requested FE 198/198.
- W2 graph: 11 stable plans, 0 commits, 33 probe observations.
- No-plan parity: 7/7 W2 graph runs use zero probe FE and match v33.
- Overlap W2-v33 mean log delta `+0.101125`; catastrophic losses `2/15`;
  maximum probe overhead `2.9986%`.

## Root cause

W2's structural futility screen is too weak. It filters missing/zero-norm plans,
but once a coordinate plan exists it still pays all three paired horizons before
the risk gate can abstain. The failure is lease timing, not an LCB threshold.

## Next step

1. Freeze CAR-W3 with the same native prefix, lazy no-plan lease, and unchanged
   W1/W2 safety gates.
2. Make pair 0 a fixed futility stage. If its candidate contrast is non-positive
   or its endpoint is worse than the checkpoint, stop immediately after pair 0;
   adopt fallback and emit an explicit `futility_pair_not_positive` reason.
3. Only a positive pair 0 may run pairs 1 and 2. Keep equal-FE arms, CRN,
   branch isolation, and final-pair-only deployment.
4. Run CLI/5k parity, then a new preregistered diagnostic before any R/S/full24.

## Forbidden

- Do not retune LCB, tail, endpoint, or catastrophic thresholds.
- Do not reuse W2 final outcomes as runtime dispatch inputs.
- Do not start R/S or full-24.
- Do not overwrite W1/W2 raw artifacts or user-owned untracked files.
