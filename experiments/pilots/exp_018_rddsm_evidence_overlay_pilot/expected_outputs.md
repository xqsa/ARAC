# Expected outputs

Each stage directory contains exactly these aggregate contracts (the `_runs/`
tree retains per-trajectory raw evidence):

- `run_manifest.md`: human-readable protocol and gate status.
- `manifest.json`: frozen matrix, config hash, source bundle, and raw manifest bindings.
- `same_budget_ledger.csv`: all native stages, overlay FE, overhead, actual FE, and closure.
- `probe_plan.csv`: all scored relations with a `selected` marker; four are selected.
- `probe_evidence.csv`: candidate-level four-point evidence (16 rows per probe bundle).
- `delayed_outcomes.csv`: owner-level next-sweep labels (8 rows per probe bundle).
- `shadow_decisions.csv`: observer-only repair/coordinate/fallback decisions.
- `runtime_actions.csv`: explicit runtime authorization, consumption, and invalidation records.
- `run_results.csv`: native terminal error and all-evaluation best error kept separate.
- `lane_summary.csv`: completion, fresh execution, applicability, FE, and terminal summaries.
- `promotion_gate.json`: mechanical or mechanism checks and all blockers.
- `aob_input_manifest.csv`: before/after hashes proving benchmark inputs were unchanged.
- `anti_leakage_audit.csv`: runtime-field, AOB-truth, and authorization audit.

For every trajectory the shared runner must write a case-prefixed overlay
manifest plus checkpoint, plan, probe, delayed-outcome, shadow-decision, and
`*_evidence_overlay_runtime_actions.csv` artifacts. The manifest supplies the
six relative paths and a separate SHA-256 mapping. Absolute paths, paths
escaping the trajectory directory, renamed case artifacts, and hash mismatches
are rejected.

`E1` still writes its one-row three-sweep checkpoint in all three lanes. Its
plan/probe/delayed/shadow files may be header-only and its probe FE is zero;
missing checkpoint evidence is not treated as a no-overlap shortcut and fails
the triplet parity gate.

AOB `Pvector/subgroups` are opened only after all optimizer executions finish.
Their shared-variable precision/recall appears inside `promotion_gate.json`
and `manifest.json` as `offline_reference_topology_evaluation` with both
`used_for_runtime=false` and `used_for_gate=false`.

Generated outputs belong under `results/` and are not committed.
