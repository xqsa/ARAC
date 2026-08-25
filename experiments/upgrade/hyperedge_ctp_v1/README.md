# hyperedge_ctp_v1 upgrade candidate

v5.0 plan: revalidate the CTP long-tail causal signal (line A, T0), then
build the minimal loop "real Phase-I hyperedge evidence -> overlapping
owner scopes -> boundary strict-best coordination" inside CTP only (line B,
H0-H2, AOB preservation), against the frozen recovered baseline
`arac-recovered-baseline-20260823-v1`.

Run pattern (Git Bash, always the project interpreter):

```text
PYTHONPATH='.;src' ./.venv/Scripts/python -m experiments.upgrade.hyperedge_ctp_v1.t0_tail_causal
PYTHONPATH='.;src' ./.venv/Scripts/python -m experiments.upgrade.hyperedge_ctp_v1.h0_sidecar
```

Stages:

- `t0_tail_causal.py` - T0 fresh matched-tail causal revalidation
  (`artifacts/upgrade_hyperedge_ctp_v1_t0_v1`): S3/S6 x seeds 20270111-15,
  frozen CTP (`tail_20pct`) vs the reserve-free copy (`no_reserved_tail`)
  on one shared fresh Phase-I checkpoint per pair.  The variant differs
  from the frozen executor in the positive-relation tail reserve only and
  is bitwise identical on zero-relation checkpoints.  **PASSED 2026-08-23**:
  R(tail/no_tail)=0.178 (S3) and 0.419 (S6) with CI uppers 0.585/0.518 -
  the frozen 20% relation tail has a strong fresh-seed causal benefit;
  tail-ratio tuning branches stay permanently closed.  (The first summary
  computation inverted the pair orientation; raw receipts were always
  correct and the orientation fix is documented in the progress log.)
- `h0_sidecar.py` - H0 explicit-evidence sidecar certification
  (`artifacts/upgrade_hyperedge_ctp_v1_h0_v1`): soft-RDDSM discovery with
  the frozen plan configuration + exact-180k MMES top-up on the six
  generator-v3 cells x three discovery seeds; hyperedge truth audit is
  offline-only.  **GATE FAILED 2026-08-23** - chain topologies produce
  3-region/merged-region hyperedges that the certification criteria
  correctly reject, and pairs3-strong misses recall >= 0.75 on one of its
  three seeds (16/24).  Per the plan stop-loss line B stops here: H1, H2
  and AOB preservation were never executed and no performance comparison
  was made.

Seed registry note: 20270101-20270125 are registered by the never-executed
`current_selector_fresh_e2e` protocol; the overlap is recorded in the root
protocol rather than silently ignored.
