# exp_003_hcc_runtime_consumer_smoke

This experiment is a runtime-connected utility smoke test, not a performance
claim.

It runs one AOB case (`E2`, seed `1`, 2000 FE budget) through three HCC smoke
lanes:

- `fallback`: fixed `conservative_no_action`
- `fixed_repair`: fixed `repair_shared_variable_binding`
- `relation_dispatch_rule`: per-overlap-relation rule dispatch

The purpose is to prove that runtime actions reach the ARAC-owned HCC smoke
runner, relation dispatch emits joinable `relation_id` artifacts, and
`action_utility_audit.csv` reports utility failures plainly instead of turning
runtime connection into a performance claim.

Run:

```powershell
py -3 experiments\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\exp_003_hcc_runtime_consumer_smoke
```

The source HCC project remains read-only. The subprocess executes the
ARAC-owned wrapper in `E:\ARAC\HCC_SRC\arac_hcc_smoke_runner.py` with
`cwd=E:\HCC-main` so HCC imports and AOB data files come from the source
project.
