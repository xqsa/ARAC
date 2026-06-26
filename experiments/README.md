# Experiments

Future experiments should be small, resumable, and table-backed.

## Recommended Layout

```text
experiments/
  exp_001_schema_smoke/
  exp_002_aob_1run_pilot/
  exp_003_backend_binding_smoke/
  exp_004_same_budget_smoke/
```

Each experiment should contain:

- `README.md`: purpose, scope, and claim level.
- `config.yaml`: frozen parameters.
- `run.ps1` or `run.py`: executable entrypoint.
- `expected_outputs.md`: required truth tables.

Results should go under `results/<experiment_id>/`, not in the project root.

For AOB work, start from `configs/aob_pilot.yaml`. The current pilot is fixed to
one independent run so ARAC-HCC can expose utility and failure signals before
expanding to the paper's 25-run protocol.

Run the current AOB pilot topology probe with:

```powershell
$env:PYTHONPATH='src'; & 'C:\Users\83718\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m experiments.exp_002_aob_1run_pilot.run
```

`exp_002_aob_1run_pilot` writes eight table-backed artifacts, keeps
paper-reported Table 2 values offline-only, and labels its own result rows as
`hcc_source_grounded_grouping_probe`. It reads source AOB topology from
`E:\HCC-main` but does not yet run MMES/CMAES optimizer execution.

## Minimum Gates

An experiment is not complete unless it states:

- claim level
- allowed runtime inputs
- forbidden runtime inputs checked
- same-budget FE status
- backend semantics status
- negative-control status
- catastrophic-loss status
