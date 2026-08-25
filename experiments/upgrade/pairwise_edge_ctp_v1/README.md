# pairwise_edge_ctp_v1 upgrade candidate (v5.1)

Semantics fix after the v5.0 H0 post-mortem: shared-variable certificates
are explicit ``PairwiseSharedEdge(j, region_a, region_b)`` objects (exactly
two regions, bilateral evidence, no unexplained third-region evidence);
three-region outputs are rejected outright and never split post-hoc;
generator truth is offline-only.

Run pattern (Git Bash, project interpreter):

```text
PYTHONPATH='.;src' ./.venv/Scripts/python -m experiments.upgrade.pairwise_edge_ctp_v1.p0_pairwise_evidence
PYTHONPATH='.;src' ./.venv/Scripts/python -m experiments.upgrade.pairwise_edge_ctp_v1.chain_pair_isolation_diagnostic
```

Stages:

- `p0_pairwise_evidence.py` - pairwise evidence gate on ``pairs3-strong``
  x five fresh discovery seeds (20270401-05) with the preregistered
  DSM/RDG budget cap of 100,000 FE (`artifacts/upgrade_pairwise_edge_ctp_v1_p0_v1`).
  **GATE FAILED 2026-08-23**: 3/5 seeds certify perfectly (precision 1.0,
  recall 1.0, zero merges); 2/5 seeds lose exactly one pair link because the
  frozen coarse RDG cover merges two planted blocks into one ~200-variable
  region whenever a shared variable seeds its component first (iteration
  order artifact, probability ~8% per pair per seed - not budget-related;
  the 100k cap changed nothing).  All 15 issued certificates across the
  five seeds are precision-1.0.  Per the preregistered rules the patch
  stage is NOT authorized and no performance comparison was run.
- `chain_pair_isolation_diagnostic.py` - offline diagnostic on
  ``chain4-strong`` x 3 seeds
  (`artifacts/upgrade_pairwise_edge_ctp_v1_chain_diagnostic_v1`):
  every shared variable of every chain link interacts with both of its
  owners' residuals and with no far-side residual (289 FE per seed).
  **Conclusion: chain overlap IS separable by pair-specific residual
  evidence** - the v5.0 chain failure came from the transitive
  resolved-hyperedge construction, not from the evidence model.  A future
  v5.2 needs per-link resolution plus protection against the seeding-order
  region merge; thresholds are not relaxed post-hoc.
