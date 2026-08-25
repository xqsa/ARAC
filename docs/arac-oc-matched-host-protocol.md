# ARAC-OC Matched-Host Protocol

`overlap_shared_patch_matched_host_gate.py` uses a known conflicting overlap
generator with fixed checkpoint, proposals, seed and strict-best ledger. The
host is explicitly labelled `forced_ctp` or `forced_gss`; this label is local to
the experiment and does not alter the production planner or selector.

## Gates

- M0 requires nonzero A1–A4 patch receipts, candidate traces, A3 state trace,
  A4 radius trace, exact FE, strict-best, state hashes and no arbitration-only
  route.
- M1 uses the nested A0–A4 matrix: A2−A1 is candidate increment, A3−A2 is
  persistent-state increment, and A4−A3 is adaptive-radius increment. It is an
  attribution gate, not a superiority claim.
- M2 uses fresh seeds and is allowed only after an M1 artifact passes.

Soft routing is a post-M1 utility only. `soft_routing.py` computes continuous
activity from rank-normalized disagreement and progress residual; its only
permitted outputs are patch scope ordering, candidate strength and radius upper
bound. It cannot change Phase-I, outer action selection, FE reservations or
component deactivation.
