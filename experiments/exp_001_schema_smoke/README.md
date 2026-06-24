# exp_001_schema_smoke

## Purpose

Generate the first ARAC truth-table surface from a tiny synthetic trace. This
experiment proves schema completeness, toy backend semantic binding,
same-budget accounting, anti-leakage scanning, and claim blocking for failed
negative controls.

It does not claim final optimizer performance.

## Run

```powershell
$env:PYTHONPATH = "E:\ARAC\src;E:\ARAC"
py -m experiments.exp_001_schema_smoke.run
```

Default outputs are written to:

```text
results/exp_001_schema_smoke/
```

## Runtime Boundary

Runtime lanes:

- `policy_action`
- `no_action`
- `fallback`
- `random_action`
- `shuffled_evidence_action`

Offline-only lane:

- `oracle_action_eval_only`

The oracle lane may be used for evaluation rows only. It must not appear with
`used_for_runtime=1` and must not be eligible for runtime dispatch.

Comparison-only lanes:

- `no_action`
- `fallback`

These lanes are used as controls and baselines. They are not eligible for
action-utility claims.

## Claim Level

Allowed claim:

```text
schema smoke with toy backend semantics
```

Forbidden claims:

- pilot utility on real benchmarks
- full evaluation completed
- final success
- SOTA or reported-baseline improvement
