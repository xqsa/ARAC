---
session_no: S23
suggested_title: "[ARAC] S24 finish July paper from v33.8 and causal no-go evidence"
parent_session: S22
project: arac
date: 2026-07-16
author: Codex
---

## Current stage

The precision-first three-layer causal-risk scheduler has been implemented and
falsified at its preregistered 40-pair pilot gate. Raw integrity passed, but
long-horizon utility was not identifiable. Full logging, model export, runtime
registration, shadow, live pilot, and full-24 are prohibited.

## Completed

- v41 remains frozen and dispatch-blocked.
- Implemented immutable 16-feature pre-action state, paired fresh-subprocess
  causal logging, SHA-256 randomization, common-terminal labels, LCO/LSO
  cross-fitting, DR value, OOD, LCB/risk, and fail-closed JSON contracts.
- Fixed the inactive v37 CC-history source with an independent trace-only
  history in commit `f184b84`.
- Preregistered the common-endpoint FE amendment in `650d491`; implemented it
  in `852a268`. Both are on `origin/main`.
- Validation: 421 focused/adjacent tests passed with one skip; tracked full
  suite passed 868 tests with one skip.
- v2 5k smoke passed: 8/8 fresh, 4/4 pairs, zero FE overrun, 76/76 AOB
  unchanged, 16/16 anti-leakage.
- v2 A4 seed1 3M trace passed: 1/1 applicable, one action-only release,
  common terminal outcome valid, zero integrity failure.
- Frozen pilot completed all 80 branches. Integrity: 40/40 pairs, zero FE
  overrun, 790/790 AOB unchanged, 16/16 anti-leakage, zero missing features.

## Frozen failure

- Applicable coverage: 16/40 pairs from four cases; required 30 pairs from six
  cases.
- Material effects: 4; required 15.
- LCO support: 0%; LSO support: 31.25%; required at least 50% in both.
- LCO/LSO candidate DR value: 0/0; sign balanced accuracy: 0.5.
- Safe policy released 0 actions. `runtime_scheduler_authorized=false`.
- No model bundle exists.

The 16 raw effects had mean tau `-7.7897e-03`, median `+5.5738e-07`, eight
wins, seven losses, one tie, and zero 20% catastrophic events. A4/A5 were
near-zero; all four material pairs were S2, which had two wins and three
losses with worst tau `-9.1729e-02`.

## Root cause

The first feasible precision opportunity is structurally sparse and does not
transport across cases. E1 has no overlap; E3/E4/S5 generally never reach the
eligible retirement/cap/history state; E2 is mostly cap- or timing-blocked.
Among observable states, material utility is concentrated in S2 and changes
sign across seeds. The sixteen identity-free state features therefore cannot
provide a positive held-out utility lower bound or calibrated release support.

This is a causal identifiability failure, not another mutex, threshold,
cooldown, FE, AOB, leakage, or implementation bug. The stable policy is v37
fallback.

## Next step

1. Stop causal scheduler experimentation for the July paper.
2. Keep v33.8 as the main full-24 performance evidence.
3. Use the causal pilot as a negative mechanism result: structure/state can
   expose an action opportunity but do not identify its terminal benefit.
4. Finish paper tables, figures, reproducibility packaging, limitations, and
   submission text. Do not spend the remaining July window on a new
   opportunity/channel estimand.

## Required reading

1. `.light/handoff/S23-arac-causal-risk-precision-pilot-failed.md`
2. `docs/design/causal-risk-precision-pilot-result-20260716.md`
3. `docs/superpowers/specs/2026-07-15-causal-risk-precision-scheduler-design.md`
4. `results/causal_precision_logging_v2_pilot_8case_seed40_44_3m_jobs24_852a268_20260716T025226/causal_logging_manifest.json`
5. `results/causal_precision_logging_v2_pilot_8case_seed40_44_3m_jobs24_852a268_20260716T025226_audit/causal_identifiability_gate.json`
6. `.light/handoff/S02-arac-v338-full24-result.md`

## Prohibited

- No full logging, model export, runtime profile, shadow, live pilot, or
  scheduler full-24 from this failed gate.
- No post-hoc relaxation of applicability, OOD, LCB, risk, materiality,
  coverage, case, seed, or opportunity thresholds.
- No reuse of the precision model for repeat lease, writeback, resource, or
  search-start actions.
- No case, seed, family, fingerprint, paper-best, history, final outcome, or
  action-resolution field in runtime dispatch.
