# Expected Outputs

`run_aob_1run_pilot(output_dir)` writes exactly these pilot artifacts:

- `pilot_run_manifest.md`
- `our_result_by_case.csv`
- `same_budget_ledger.csv`
- `backend_semantics_diff.csv`
- `anti_leakage_audit.csv`
- `paper_reported_comparison.csv`
- `negative_control_audit.csv`
- `catastrophic_loss_audit.csv`

Required invariants:

- all 24 AOB cases are covered once for seed `1`
- `phase_i_fe + phase_ii_fe = 3000000`
- `paper_reported_comparison.csv` is offline evaluation only
- `runtime_dispatch_allowed=0` for paper-reported comparison rows
- runtime payloads exclude oracle, final error, relative gain, reported baseline,
  problem-family label, and prior outcome fields
- negative controls and catastrophic-loss checks are visible audit surfaces
