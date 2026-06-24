# Expected Outputs

Running `py -m experiments.exp_001_schema_smoke.run` produces:

- `evidence_profile.csv`
- `action_decision.csv`
- `backend_semantics_diff.csv`
- `same_budget_ledger.csv`
- `action_utility_audit.csv`
- `anti_leakage_audit.csv`
- `run_manifest.md`

Required checks:

- `isolate_conflicting_relation` changes relation handling.
- `protect_high_margin_group` changes budget allocation.
- `repair_shared_variable_binding` changes variable ownership.
- `allow_beneficial_coordination` changes coordination mode.
- `conservative_no_action` does not report active backend semantics.
- Every lane records `phase_i_fe + phase_ii_fe = total_fe`.
- Every lane in this smoke uses the same `total_fe=100`.
- `policy_action` may be claim-eligible only when semantics, budget,
  anti-leakage, utility, and negative-control gates pass.
- `no_action` and `fallback` are comparison-only and not claim-eligible.
- `oracle_action_eval_only` has `used_for_runtime=0`.
- Runtime payloads contain no forbidden final/oracle/reference fields.
- Failed negative controls are blocked from claims.
