# Archived v3.3 Late-Stagnation NDA Takeover

Status: failed experiment, archived for negative-result provenance only.

## Scope

This record refers to the uncommitted v3.3 `late_stagnation_nda_takeover`
candidate and its 3-seed R3 pilot output:

```text
E:\ARAC\results\exp_017_controller_v33_r3_seed123_3m_jobs16
```

The run did not pass the pre-registered `3.28e5` upgrade gate. Its observed
behavior is therefore evidence against promoting the candidate as the stable
runtime controller, not evidence for a performance claim.

## Boundary

- Canonical stable algorithm: v3.2 at the restructuring baseline.
- v3.3 implementation status: 641 lines of uncommitted experimental code;
  the implementation is intentionally not copied into this repository.
- Allowed use: offline failure analysis and future design review only.
- Forbidden use: runtime dispatch, final-result selection, or performance
  claims presented as the canonical ARAC method.

## Promotion Rule

Any future resurrection of this idea must first be reimplemented behind the
current action contracts, pass the v3.2 regression suite, and pass a fresh
same-budget pilot with explicit anti-leakage and catastrophic-loss gates.
