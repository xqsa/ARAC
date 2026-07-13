# Runtime Module Split Baseline

Date: 2026-07-12
Executor: Codex
Stable behavior source: v3.2 restructuring branch at `8568719`

## Pre-Refactor Regression

Command:

```powershell
python -m pytest tests/test_relation_policy.py `
  tests/test_hcc_action_execution_plan.py `
  tests/test_hcc_backbone_adapter.py `
  tests/test_hcc_evaluation_record_parser.py `
  tests/test_hcc_execution_adapter.py `
  tests/test_policy_smoke.py -q
```

Observed result: `112 passed`.

## Policy Boundary

The first extraction moves the relation-level decision contracts to
`src/arac/policy/evidence_model.py` and the existing v3.2 action computation to
`src/arac/policy/action_policy.py`. `relation_policy.py` remains a re-export
compatibility path. The historical logger name is retained because experiment
audits filter `arac.policy.relation_policy` explicitly.

## HCC Injection Constraint

The HCC adapter cannot be converted to a wildcard re-export shim without a
behavior change. Existing focused tests and audit tools monkeypatch private
parsers and trace helpers on `arac.backends.hcc`; functions defined in another
module retain that module's globals and would bypass those patches.

Therefore HCC extraction must use one of these explicit boundaries:

1. Move pure parser/plan functions and have `hcc.py` call them through explicit
   module attributes that tests can inject.
2. Add dependency parameters to execution functions before moving their
   implementations, while preserving current defaults.

A plain copy or wildcard re-export is rejected because it creates either a
second implementation or a hidden monkeypatch regression.
