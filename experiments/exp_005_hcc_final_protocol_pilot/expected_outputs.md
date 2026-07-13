# exp_005_hcc_final_protocol_pilot Expected Outputs

The final-protocol pilot writes the same core audit tables as `exp_003`:

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
- `claim_evidence_table.md`
- `aob_input_manifest.csv`
- `final_protocol_environment.json`
- `run_manifest.md`

Protocol scope:

- `3,000,000` FE per case
- 3 seeds by default
- `canonical_evidence_controller_v1` single-lane profile
- explicit AOB data root and before/after SHA256 audit
- fail-fast pinned backend environment gate before optimizer execution
- manifest records runner/policy/optimizer hashes, wrapper/backend Python
  executables, Python/NumPy/SciPy/Torch/BLAS, thread variables, cwd, and FE
  accounting
- optional `landscape_escape` lane profile for fallback/repair/coordinate/BIPOP
  search-state comparison
