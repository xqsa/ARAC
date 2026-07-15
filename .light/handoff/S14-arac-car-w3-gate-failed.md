---
session_no: S14
suggested_title: "[ARAC] S15 CAR-W3 utility gate failed; July paper stop-loss"
parent_session: S13
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

CAR-W3 fresh 3M diagnostic completed on frozen commit `e2a65da`. Integrity is
valid, but the pre-registered utility gate failed. This is the final algorithm
iteration for the July paper; do not start R/S, held-out, or CAR full-24.

## Frozen evidence

- Raw directory: `results/car_w3_diagnostic_6case_seed26_28_3m_20260715`
- Protocol: E1/E2/S3/R4/A5/E6, seeds 26/27/28, five lanes, strict 3M FE.
- Integrity: 90/90 fresh; 0/90 FE violations; 885/885 AOB unchanged; 16/16
  anti-leakage; 28/28 CAR boundary; 68/68 branch FE; 10/10 no-plan parity.
- Utility: 8/18 stable graph plans, 1/8 commits, mean log delta +0.019356626,
  overlap mean +0.023227951, median 0, numeric wins 2/18, meaningful wins 0/18,
  catastrophic 0/18, worst seed 28.
- Probe cost: 5 first-pair aborts, 14 probe rows, total 419,772 FE, mean
  overhead 0.777356% over all graph runs, maximum 2.998600%.

## Root cause and decision

W3 successfully shortens probes that are already unable to pass the frozen gate,
but it does not improve candidate coverage or identify positive long-horizon
utility. The only commit is an effectively tied A5 seed 26 result. The method
is therefore safe/auditable but ineffective as a performance intervention on
this diagnostic.

Stop threshold tuning and new action channels. Finish the July manuscript as a
protocol plus failure-analysis paper, reporting v33 full-24 instability, W1/W2
utility failures, and W3 probe-cost reduction without claiming stable
performance improvement.

## Key files

- `docs/design/car-w3-diagnostic-result-20260715.md`
- `docs/design/2026-07-15-car-w3-diagnostic-preregistration.md`
- `docs/design/core-method.md`
- `docs/literature_review.md`
- `docs/arac_action_guided_cc_manuscript_draft_zh.md` (user-owned untracked draft;
  update only after reviewing the new result)
