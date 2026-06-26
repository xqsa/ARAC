# exp_003_hcc_runtime_consumer_smoke

This experiment is a runtime-connected smoke test, not a performance claim.

It runs one AOB case (`E2`, seed `1`) through two HCC smoke lanes:

- `conservative_no_action`
- `repair_shared_variable_binding`

The purpose is to prove that an ARAC selected action is passed into the
ARAC-owned HCC smoke runner, consumed by the overlap repair hook, traced in
`action_trace.csv`, and returned as a fresh HCC execution result while the
same-budget ledger and anti-leakage audit remain explicit.

Run:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\83718\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m experiments.exp_003_hcc_runtime_consumer_smoke.run --output-dir results\exp_003_hcc_runtime_consumer_smoke
```

The source HCC project remains read-only. The subprocess executes the
ARAC-owned wrapper in `E:\ARAC\HCC_SRC\arac_hcc_smoke_runner.py` with
`cwd=E:\HCC-main` so HCC imports and AOB data files come from the source
project.
