# exp_003 Expected Outputs

The experiment writes fourteen CSV artifacts and two Markdown audit files:

- `our_result_by_case.csv`
- `same_budget_ledger.csv`
- `backend_semantics_diff.csv`
- `action_execution_plan.csv`
- `action_trace.csv`
- `action_decision.csv`
- `action_mismatch_audit.csv`
- `overlap_relations.csv`
- `relation_join_audit.csv`
- `action_utility_audit.csv`
- `negative_control_comparison.csv`
- `policy_evidence_diagnosis.csv`
- `anti_leakage_audit.csv`
- `claim_gate.csv`
- `run_manifest.md`
- `claim_evidence_table.md`

The required smoke evidence is:

- `repair_shared_variable_binding` has `optimizer_consumed=1` in
  `action_execution_plan.csv`.
- `repair_shared_variable_binding` has `variable_owner_changed=1` in
  `backend_semantics_diff.csv`.
- `action_trace.csv` contains rows with
  `semantic_surface=shared_variable_owner_rebinding` and
  `optimizer_consumed=1`.
- The opt-in `evidence_action_controller_v33` profile also records
  `trust_key`, `trust_phase`, `trust_reason`, `trust_score`, `trust_exposure`,
  `trust_cooldown`, `trust_credit`, `trust_unstable`,
  `trust_pre_writeback_fitness`, and `trust_post_writeback_fitness`. Legacy
  controller profiles omit these columns and retain their original schema.
  `fallback_route` identifies `dense_preserve_v31` or
  `non_dense_bounded_0_5` for protected fallback rows.
- For `landscape_escape`, `action_trace.csv` also records BIPOP search-state
  audit fields such as `search_state_action_type`, `bipop_restart_mode`,
  `sigma_before`, `sigma_after`, `population_before`, `population_after`,
  `escape_budget`, `restart_triggered`, and `restart_accepted`.
- `claim_gate.csv` for the repair lane does not contain
  `active_action_not_consumed_by_hcc_runtime`.
- `relation_dispatch_rule` has matching `relation_id` rows across
  `action_decision.csv`, `action_trace.csv`, and `overlap_relations.csv`.
- `action_mismatch_audit.csv` records candidate action scores, selected action,
  second-best action, score margin, and abstain reason for each relation.
- `action_utility_audit.csv` contains `final_error`, `fe_used`,
  `same_budget_violation`, `relative_gain_vs_fallback`, `utility_label`,
  `action_mix`, `claim_allowed`, and `claim_blockers`.
- `negative_control_comparison.csv` reports whether shuffled relation dispatch
  stably outperforms real relation dispatch across the configured seeds.
- `policy_evidence_diagnosis.csv` gives the stop/continue decision for
  same-budget, utility, catastrophic-loss, shuffled-control, and SOTA
  escalation gates.
- `run_manifest.md` records the command shape, problem/seed set, lanes, key
  gates, artifact list, parallel job count, wrapper/backend Python executables,
  git commit, code/config hashes, and the rule that final/reported/oracle values
  must not enter runtime dispatch.
- `claim_evidence_table.md` maps each diagnosis claim/gate to status,
  observed evidence, blockers, and source artifact.
- Final errors are offline-only smoke outputs and must not enter runtime
  dispatch.
