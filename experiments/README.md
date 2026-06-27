# Experiments

Future experiments should be small, resumable, and table-backed.

## Recommended Layout

```text
experiments/
  exp_001_schema_smoke/
  exp_002_aob_1run_pilot/
  exp_003_hcc_runtime_consumer_smoke/
  exp_004_hcc_main_historical_result_recovery/
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

`exp_003_hcc_runtime_consumer_smoke` runs HCC smoke through fallback, fixed
repair, fixed coordinate, per-overlap-relation dispatch, and shuffled negative
control lanes. Its claim is runtime connection plus explicit utility auditing,
not performance: relation artifacts must join by `relation_id`, candidate
action mismatches stay in `action_mismatch_audit.csv`, utility failures stay in
`action_utility_audit.csv`, and negative-control failures stay in
`negative_control_comparison.csv`. In AOB multi-problem summaries, level 1
cases are no-overlap controls and levels 2-6 are the overlap-applicable utility
scope. SOTA escalation is gated by `policy_evidence_diagnosis.csv`.

`exp_004_hcc_main_historical_result_recovery` scans historical
`E:\HCC-main\HCC_SRC\result\**\evaluation_record.txt` files, writes an
inventory, and joins detected AOB cases to the paper-reported Table 2 anchors.
These recovered results are preserved as offline evidence only and must not
enter runtime dispatch.

## Minimum Gates

An experiment is not complete unless it states:

- claim level
- allowed runtime inputs
- forbidden runtime inputs checked
- same-budget FE status
- backend semantics status
- negative-control status
- catastrophic-loss status
