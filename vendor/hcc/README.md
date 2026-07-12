# HCC Vendor Boundary

`vendor/hcc/` is reserved for the reviewed HCC/AOB source snapshot used by ARAC's backend
adapter. During Task 2 this directory intentionally contains only this contract; tracked source
remains under `HCC_SRC/` until the separately tested Task 3 migration.

## Provenance, Inputs, And Outputs

- Provenance input: the read-only source/evidence repository `E:\HCC-main`, reviewed against the
  v3.2 canonical ARAC baseline rooted at commit `b88a4d9`.
- Runtime input after migration: explicit backend root, AOB benchmark data, ARAC action plan,
  seed, and FE budget.
- Output: optimizer traces and same-budget records written outside the vendor tree to
  `results/` through the adapter boundary.

## Git And Modification Policy

Approved source, provenance notes, and `result/README.md` are tracked. Generated files under
`result/`, `__pycache__`, and `.pyc` files are ignored and must not be tracked. Vendor code is
read-only application code: ARAC evidence, policy, claim gates, and paper comparisons belong
outside this directory. Required fixes must be explicit, reviewed, tested for source
equivalence, and documented as patches rather than silently mixed into the snapshot.

The current `HCC_SRC/` compatibility path is a nonfatal, explicit Task 3 warning. It exists to
avoid changing runtime behavior in Task 2 and must not weaken cache, results-ignore, or
reference-blind audit failures.
