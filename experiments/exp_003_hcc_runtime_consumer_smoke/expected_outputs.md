# exp_003 Expected Outputs

The experiment writes seven CSV artifacts:

- `our_result_by_case.csv`
- `same_budget_ledger.csv`
- `backend_semantics_diff.csv`
- `action_execution_plan.csv`
- `action_trace.csv`
- `anti_leakage_audit.csv`
- `claim_gate.csv`

The required smoke evidence is:

- `repair_shared_variable_binding` has `optimizer_consumed=1` in
  `action_execution_plan.csv`.
- `repair_shared_variable_binding` has `variable_owner_changed=1` in
  `backend_semantics_diff.csv`.
- `action_trace.csv` contains rows with
  `semantic_surface=shared_variable_owner_rebinding` and
  `optimizer_consumed=1`.
- `claim_gate.csv` for the repair lane does not contain
  `active_action_not_consumed_by_hcc_runtime`.
- Final errors are offline-only smoke outputs and must not enter runtime
  dispatch.
