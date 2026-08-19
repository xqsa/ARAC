# ARAC historical recovery gate

This gate answers one question before selector evaluation:

> Do the four fixed expert lanes reproduce the historical ARAC table under one
> common protocol?

The reference is the repository table at
`output/pdf/aob_arac_method_comparison_corrected.csv`. Its ARAC column is the
machine-readable source of the 24 target means and sample standard deviations.

The table itself does not record seeds, dimension, or FE budget. The candidate
protocol in `config.json` is therefore bound by the existing A/S/R evidence:
all 18 visible aggregates match after the table's `.2E` rounding, and all three
result sets use AOB dimension 1000, seeds 117 through 141, and exactly 3,000,000
FE per run. This is strong repository-internal provenance, but it is not a
substitute for an external paper protocol if one exists.

The fixed expert mapping is:

```text
A -> AOR
E -> SMP
S -> CTP
R -> GCB
```

Gate order:

1. Reproduce all four fixed experts.
2. Test whether Phase-I evidence selects the appropriate or low-regret expert.
3. Test ARAC-Core end to end against the historical ARAC column.

Existing result directories are read-only. A missing E/SMP campaign must use a
new output directory and a frozen protocol; it must not resume or overwrite any
historical campaign.

Two evidence layers are kept separate:

- The historical expert artifacts reproduce 18 A/S/R rows exactly at the
  table's displayed precision; their E/SMP lane is incomplete.
- The frozen v5 independent-action matrix covers all 24 cases after a common
  180,000-FE Phase-I checkpoint, but it uses a different seed namespace and its
  recorded source hashes no longer match most current action/runtime files.

Neither layer alone proves that the current ARAC-Core code recovers the table.

The representative replay now has a separate frozen-source control. The current
legacy execution path and the manifest-bound frozen source produce identical
terminal FE, final error, result hash, and receipt hash on all four representative
contexts. Only A1/AOR reproduces its stored v5 arm exactly; E1/SMP, R1/GCB, and
S1/CTP do not. The v5 block-action receipts are therefore useful aggregate
evidence, but not a valid bit-exact recovery oracle.

The aggregate recovery campaign is frozen in `fixed_expert_config.json`. It
creates one Phase-I checkpoint for each of the 24 cases and 25 historical seeds,
then executes only the family-mapped expert from that checkpoint. Checkpoints and
arms run with 24 workers and write resumable progress under
`artifacts/historical_recovery_fixed_expert_v1`.

That campaign completed all 600 arms at the exact 3,000,000-FE terminal budget,
but it is not a historical-table replay oracle. The current v9 protocol spends
180,000 FE in Phase-I and gives the mapped action 2,820,000 FE; the historical
AOR representative is a fresh 3,000,000-FE vendor HCC Sep-CMA run from
`initial_mean=0`, while historical CTP/GCB use separate HCC action-routing
protocols. The completed v9 campaign therefore has a closed aggregate recovery
gate (6/24 displayed-precision mean matches, 0/24 sample-standard-deviation
matches, 0/24 cases recovered) and is retained as current-protocol evidence.
See `fixed_expert_drift.md` for the read-only provenance diagnosis. Selector
correctness and ARAC-Core end-to-end claims stay deferred until the historical
action protocol is reconstructed or the target is explicitly rebound to v9.
