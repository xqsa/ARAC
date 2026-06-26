# Expected Outputs

`run_aob_1run_pilot(output_dir)` writes exactly these pilot artifacts:

- `pilot_run_manifest.md`
- `our_result_by_case.csv`
- `same_budget_ledger.csv`
- `backend_semantics_diff.csv`
- `action_execution_plan.csv`
- `anti_leakage_audit.csv`
- `paper_reported_comparison.csv`
- `negative_control_audit.csv`
- `catastrophic_loss_audit.csv`

Required invariants:

- all 24 AOB cases are covered once for seed `1`
- result rows use `source_level=hcc_source_topology`
- result rows use `pilot_result_source=hcc_source_grounded_grouping_probe`
- optional HCC smoke overlays use `pilot_result_source=hcc_subprocess_smoke_execution`
  and expose `hcc_smoke_final_error`, `hcc_smoke_fe_used`, `hcc_smoke_status`,
  and `fresh_optimizer_execution`
- AOB topology fields include `dimension_real`, `group_count`,
  `overlap_group_count`, `overlapping_element_count`, `degree_of_overlap`, and
  `global_fes`
- `phase_i_fe + phase_ii_fe = 3000000`
- `paper_reported_comparison.csv` is offline evaluation only
- `runtime_dispatch_allowed=0` for paper-reported comparison rows
- runtime payloads exclude oracle, final error, relative gain, reported baseline,
  problem-family label, and prior outcome fields
- `action_execution_plan.csv` records whether selected ARAC actions are
  optimizer-consumed by HCC; unwired active actions must be explicit blockers
- negative controls and catastrophic-loss checks are visible audit surfaces
- the default stage does not run MMES/CMAES and does not claim optimizer
  performance
- explicit HCC smoke execution requires the `hcc` optional dependency set from
  `pyproject.toml`; smoke final errors are offline-only and cannot be copied
  into runtime dispatch
- CLI smoke mode writes HCC subprocess scratch outputs below
  `<output-dir>/_hcc_smoke/` while keeping the top-level artifact contract at
  the eight files listed above
