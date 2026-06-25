# HCC Backbone Extraction

Date: 2026-06-25
Author: Codex

## Purpose

ARAC is a clean extraction from the original `E:\HCC-main` workspace. It is not
a from-zero optimizer project and it is not a direct copy of the historical
milestone runners.

The intended split is:

```text
HCC = grouping + cooperative coevolution optimization backbone
ARAC = reference-blind evidence-to-intervention decision and audit layer
HCC + ARAC = one concrete method instance
```

This document records the first extraction boundary: ARAC can consume a small
HCC backbone snapshot and convert it into an `EvidenceProfile`, then map ARAC
action families back to HCC-consumed semantic effects.

## Kept From HCC

The first adapter keeps only reference-blind backbone signals:

- grouping identity and run identity
- dimension, group count, overlap group count, and overlapping element count
- per-group trace signals such as fitness delta, priority rank, and shared
  variable count
- remaining budget ratio

These are method inputs for ARAC's Phase-I evidence surface. They are not final
evaluation outcomes.

## Not Extracted

This step deliberately does not copy:

- historical Mxx runners
- Route A repair branches
- cached artifacts or old milestone directories
- reported baseline values
- final error, oracle labels, paper-relative results, or user reference best
- HCC optimizer implementation details beyond the minimal semantic contract

The old workspace remains useful as source evidence and design history, but the
clean ARAC repository should grow from small contracts, tests, and audited
schemas.

## Runtime Boundary

Runtime dispatch must remain reference-blind. The HCC snapshot is validated
through the same forbidden-field guard used by other ARAC evidence surfaces.

Allowed runtime signals describe current optimization state. Forbidden runtime
signals include final answers, oracle labels, reported baselines, problem-family
shortcuts, and previous outcome fields.

Oracle or reference lanes may exist only as offline evaluation controls. They
must not enter runtime action dispatch.

## First Adapter Contract

`src/arac/backends/hcc.py` defines:

- `HccGroupSignal`: one HCC group-level, reference-blind trace row.
- `HccBackboneSnapshot`: the minimal HCC state ARAC may consume.
- `build_hcc_evidence_profile(...)`: converts the snapshot into
  `EvidenceProfile`.
- `hcc_backend_semantics_for(...)`: maps ARAC action families to HCC backend
  semantic effects.

Current action semantics are:

- `isolate` changes relation handling.
- `protect` changes budget allocation.
- `reassign_repair` changes variable ownership.
- `coordinate` changes coordination mode.
- `fallback` does not claim an active HCC semantic change.

## Claim Boundary

This extraction proves only a schema and adapter-level step:

- HCC-style grouping signals can be represented as ARAC evidence.
- Forbidden outcome fields are rejected before runtime use.
- ARAC actions can be mapped to named HCC backend semantic surfaces.

It does not prove pilot utility, benchmark improvement, or final performance.
Those require fresh same-budget execution, negative controls, anti-leakage
audits, backend semantics diffs, and catastrophic-loss checks.
