# exp_003_hcc_runtime_consumer_smoke

This experiment is a runtime-connected utility smoke test, not a performance
claim.

By default it runs one AOB case (`E2`, seeds `1 2 3`, 2000 FE budget) through
five HCC smoke lanes:

- `fallback`: fixed `conservative_no_action`
- `fixed_repair`: fixed `repair_shared_variable_binding`
- `fixed_coordinate`: fixed `allow_beneficial_coordination`
- `relation_dispatch_rule`: per-overlap-relation rule dispatch
- `shuffled_relation_dispatch`: deterministic shuffled relation-dispatch
  negative control

The purpose is to prove that runtime actions reach the ARAC-owned HCC smoke
runner, relation dispatch emits joinable `relation_id` artifacts, and
`action_utility_audit.csv` reports utility failures plainly instead of turning
runtime connection into a performance claim. `negative_control_comparison.csv`
blocks escalation if the shuffled control stably outperforms real relation
dispatch. `policy_evidence_diagnosis.csv` records the stop reason and next
step when utility evidence does not support SOTA escalation.
`claim_evidence_table.md` maps each claim/gate to its observed evidence and
source artifact. `run_manifest.md` preserves the command shape, configured
problems/seeds, key gates, artifact list, parallel job count, and the
anti-leakage boundary for runtime dispatch.

Run:

```powershell
py -3 experiments\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\exp_003_hcc_runtime_consumer_smoke
```

For wider smoke runs, pass explicit problems/seeds and parallel jobs:

```powershell
py -3 experiments\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\exp_003_hcc_runtime_consumer_smoke --seeds 1 2 3 4 5 --problems E1 E2 S1 S2 R1 R2 A1 A2 --jobs 8
```

The source HCC project remains read-only. The subprocess executes the
ARAC-owned wrapper in `E:\ARAC\HCC_SRC\arac_hcc_smoke_runner.py` with
`cwd=E:\HCC-main` so HCC imports and AOB data files come from the source
project.
