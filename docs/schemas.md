# Truth Table Schemas

These schemas define the minimum artifact surface for `ARAC`. A clean run should
produce these tables instead of many milestone-specific one-off files.

## `evidence_profile.csv`

One row per problem, seed, phase window, and evidence unit.

| Field | Meaning |
| --- | --- |
| `run_id` | Stable run identifier. |
| `problem_id` | Execution identity only; not a policy feature. |
| `seed` | Random seed. |
| `window_id` | Trace window. |
| `unit_type` | `problem`, `group`, `relation`, or `shared_variable`. |
| `unit_id` | Local unit id. |
| `feature_source` | Source trace or extractor. |
| `feature_coverage` | Fraction of required features available. |
| `overlap_degree` | Degree of overlap for this unit. |
| `shared_var_support_ratio` | Support ratio for shared-variable evidence. |
| `direction_disagreement` | Direction conflict signal. |
| `harmful_coord_score` | Harmful coordination signal. |
| `group_gain_asymmetry` | Asymmetry of gains across related groups. |
| `priority_spread` | Resource or contribution priority spread. |
| `rank_stability` | Stability of ranking across trace windows. |
| `budget_remaining_ratio` | Remaining budget at decision time. |
| `fallback_margin_proxy` | Reference-blind proxy for fallback safety. |
| `used_for_runtime` | Must be `1` only for legal runtime features. |

## `action_decision.csv`

One row per evidence unit and selected policy decision.

| Field | Meaning |
| --- | --- |
| `run_id` | Stable run identifier. |
| `problem_id` | Execution identity only. |
| `seed` | Random seed. |
| `window_id` | Trace window. |
| `unit_type` | Evidence unit type. |
| `unit_id` | Evidence unit id. |
| `selected_action_family` | `coordinate`, `isolate`, `protect`, `reassign_repair`, or `fallback`. |
| `selected_action_name` | Concrete action primitive. |
| `decision` | `allow`, `block`, `fallback`, or `shadow_only`. |
| `trigger_features` | Semicolon-separated legal features. |
| `trigger_reason` | Human-readable reason. |
| `expected_gain_proxy` | Reference-blind gain proxy. |
| `action_cost_proxy` | Cost proxy. |
| `risk_penalty_proxy` | Risk proxy. |
| `utility_proxy` | Expected gain minus cost and risk. |
| `fallback_action` | Fallback when unsafe. |
| `negative_control_status` | `pass`, `fail`, or `not_run`. |
| `used_for_runtime` | `1` for runtime decisions. |

## `backend_semantics_diff.csv`

One row per executed action and backend semantic effect.

| Field | Meaning |
| --- | --- |
| `run_id` | Stable run identifier. |
| `problem_id` | Execution identity. |
| `seed` | Random seed. |
| `selected_action_name` | Concrete action primitive. |
| `backend_name` | Backend adapter or optimizer surface. |
| `variable_owner_changed` | `1` if variable ownership changed. |
| `coordination_mode_changed` | `1` if coordination behavior changed. |
| `budget_allocation_changed` | `1` if budget allocation changed. |
| `update_order_changed` | `1` if update order changed. |
| `acceptance_rule_changed` | `1` if acceptance rule changed. |
| `semantics_hash_before` | Hash before action. |
| `semantics_hash_after` | Hash after action. |
| `backend_semantics_changed` | `1` if action had executable backend effect. |

## `same_budget_ledger.csv`

One row per problem, seed, and plan.

| Field | Meaning |
| --- | --- |
| `run_id` | Stable run identifier. |
| `problem_id` | Execution identity. |
| `seed` | Random seed. |
| `plan_name` | Action lane, fallback lane, or baseline lane. |
| `phase_i_fe` | FE spent collecting evidence. |
| `phase_ii_fe` | FE spent executing backend intervention. |
| `total_fe` | `phase_i_fe + phase_ii_fe`. |
| `budget_limit` | Configured total FE. |
| `same_budget_violation` | `1` if total FE exceeds the limit. |
| `fresh_execution` | `1` if not cache/materialization only. |

## `action_utility_audit.csv`

One row per evaluated action lane.

| Field | Meaning |
| --- | --- |
| `run_id` | Stable run identifier. |
| `problem_id` | Execution identity. |
| `seed` | Random seed. |
| `plan_name` | Action lane. |
| `fallback_plan` | Fallback or comparison lane. |
| `action_final_error` | Final error of action lane. |
| `fallback_final_error` | Final error of fallback lane. |
| `relative_gain_vs_fallback` | Relative gain against fallback. |
| `meaningful_win` | `1` only if gain passes the threshold. |
| `catastrophic_loss` | `1` if loss exceeds catastrophic threshold. |
| `negative_control_pass` | `1` if negative controls pass. |
| `backend_semantics_changed` | Joined from semantic diff. |
| `claim_eligible` | `1` only if all gates pass. |

## `anti_leakage_audit.csv`

One row per runtime table and field scan.

| Field | Meaning |
| --- | --- |
| `run_id` | Stable run identifier. |
| `artifact_path` | Scanned artifact. |
| `forbidden_field` | Forbidden field name or pattern. |
| `found_in_runtime_payload` | `1` if present in runtime payload. |
| `audit_status` | `pass` or `fail`. |
| `note` | Short explanation. |

