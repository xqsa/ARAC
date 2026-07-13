# Experiments

Experiments are organized by research stage so that smoke tests, offline
recovery, and final protocols cannot be mistaken for the same claim level.

## Layout

```text
experiments/
  pilots/
    exp_001_schema_smoke/
    exp_002_aob_1run_pilot/
    exp_003_hcc_runtime_consumer_smoke/
  infrastructure/
  recovery/
    exp_004_hcc_main_historical_result_recovery/
  ablations/
  final/
    exp_005_hcc_final_protocol_pilot/
  exp_009_binary_lsgo_arac_pilot/
  exp_010_binary_lsgo_focused_3seed/
  exp_011_binary_lsgo_diagnostic/
```

Each experiment should contain:

- `README.md`: purpose, scope, and claim level.
- `config.yaml`: frozen parameters.
- `run.ps1` or `run.py`: executable entrypoint.
- `expected_outputs.md`: required truth tables.

Results should go under `results/<experiment_id>/`, not in the experiment
source directory. Canonical default output directories are resolved against
the repository root by `experiments.paths`; explicitly supplied paths retain
each runner's existing semantics.

For AOB work, start from `configs/aob_pilot.yaml`. The current pilot is fixed to
one independent run so ARAC-HCC can expose utility and failure signals before
expanding to the paper's 25-run protocol.

Run the current AOB pilot topology probe with:

```powershell
$env:PYTHONPATH='src'; & 'C:\Users\83718\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m experiments.pilots.exp_002_aob_1run_pilot.run
```

`exp_002_aob_1run_pilot` writes eight table-backed artifacts, keeps
paper-reported Table 2 values offline-only, and labels its own result rows as
`hcc_source_grounded_grouping_probe`. It reads source AOB topology from the
canonical `vendor/hcc/` snapshot but does not yet run MMES/CMAES optimizer execution.

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

Run the native binary LSGO ARAC pilot with:

```powershell
$env:PYTHONPATH='src'
& 'E:\ARAC\.venv\Scripts\python.exe' -m experiments.exp_009_binary_lsgo_arac_pilot.run `
  --output-dir results/exp_009_binary_lsgo_arac_pilot `
  --total-fes 2000
```

`exp_009_binary_lsgo_arac_pilot` executes 18 cases across native baseline,
ARAC policy, and shuffled-evidence negative-control lanes. Each lane uses 20%
of its 2000-FE budget for Phase I and the remainder for action execution. Its
CSV and JSON outputs are generated under `results/` and stay out of Git. This
single-seed run verifies runtime connection, action semantics, reproducibility,
same-budget accounting, and leakage boundaries; it is not final performance
evidence.

Run the focused binary LSGO three-seed pilot with:

```powershell
$env:PYTHONPATH='src'
& 'E:\ARAC\.venv\Scripts\python.exe' -m experiments.exp_010_binary_lsgo_focused_3seed.run `
  --output-dir results/exp_010_binary_lsgo_focused_3seed `
  --total-fes 2000
```

`exp_010_binary_lsgo_focused_3seed` is fixed to BLSGO-F07, F08, F09, F14,
and F15; optimizer seeds 20260713-20260715; and native baseline, ARAC policy,
and shuffled-evidence negative-control lanes. The canonical protocol performs
45 executions at 2000 FE each and writes `run_results.csv`, `case_summary.csv`,
`promotion_gate.json`, and `manifest.json`. Passing its pre-registered gate
only permits escalation to a larger experiment; it is not final performance
evidence. Generated artifacts stay under `results/` and out of Git.

Run the binary LSGO mechanism diagnosis with:

```powershell
$env:PYTHONPATH='src'
& 'E:\ARAC\.venv\Scripts\python.exe' -m experiments.exp_011_binary_lsgo_diagnostic.run `
  --output-dir results/exp_011_binary_lsgo_diagnostic `
  --total-fes 2000
```

`exp_011_binary_lsgo_diagnostic` runs BLSGO-F08 and F15 with optimizer seeds
20260713-20260717 across native single-bit, native group-block, forced isolate,
and current ARAC-policy lanes. The canonical protocol performs 40 executions
at 2000 FE each. It separates proposal-operator capacity from policy triggering
under identical initial and Phase-I states. Its outputs are offline mechanism
diagnosis only: group-block is not an ARAC action, and no lane is eligible for
promotion or performance claims.

## Canonical Scope

The staged HCC layout contains `exp_001` through `exp_005`; the independently
tracked binary-LSGO pilots remain at `exp_009` through `exp_011`. Later untracked
FlyKi and guarded-portfolio materials are reviewed separately; they are not
imported into the stable runtime or silently added to this tree.

- `exp_006` and `exp_007` remain deferred because their FlyKi adapter, build
  scripts, tests, and source checkout are all untracked external work.
- `exp_008` remains quarantined because it is a post-v3.2 guarded portfolio
  candidate with offline paper/historical comparison surfaces; it has not
  passed the v3.2 stable-runtime promotion gate.

## Minimum Gates

An experiment is not complete unless it states:

- claim level
- allowed runtime inputs
- forbidden runtime inputs checked
- same-budget FE status
- backend semantics status
- negative-control status
- catastrophic-loss status
