# Independent action semantic parity audit

Historical HCC-backed results are numerical golden references only. Production
ARAC remains independent and must recover useful action semantics without importing
or renaming HCC code.

- Production HCC runtime imports: `0`
- Selector evaluation authorized: `False`
- Next gate: `fixed_checkpoint_single_case_mechanism_screen`

| Action | Frozen independent v3 | Historical HCC | Selector ready |
|---|---|---|---|
| AOR | compatible | not_equivalent | no |
| CTP | different | not_equivalent | no |
| SMP | different | unresolved | no |
| GCB | different | not_equivalent | no |

## Mechanism differences

### AOR

- Frozen v3: Phase-I evidence routes one full-space Sep-CMA or MMES continuation.
- Current: The same evidence route is retained inside the independent runtime.
- Historical boundary: Historical AOR is a fresh 3,000,000-FE vendor Sep-CMA run from mean 0; current AOR starts from a Phase-I checkpoint and receives only the remaining budget.

### CTP

- Frozen v3: Use 4 zero-relation or 2 positive-relation coverage sweeps, then 8 block polish sweeps, then terminal full-space MMES.
- Current: Spend about 20% on persistent coverage and then allocate nearly all remaining budget to sequential block or relation-cover polish.
- Historical boundary: Historical CTP binds four coverage sweeps at a separate HCC decision boundary before full-space polish; current scheduling and checkpoint lifecycle differ.

### SMP

- Frozen v3: Persist one block-CMA state per block for the available Phase-II budget.
- Current: Use stateful visits, stale-state restarts, directional rescue, and conditional full-space polish.
- Historical boundary: The complete historical 25-seed SMP lane and its exact action lifecycle are absent.

### GCB

- Frozen v3: Use full-space Sep-CMA immediately for zero relations; otherwise spend about 10% on graph-ordered persistent blocks before global coordination.
- Current: Run up to three cold block sweeps, one short full-space coordination burst, then restart cold block sweeps to the terminal budget.
- Historical boundary: Historical GCB uses a native phase-boundary burst followed by persistent native resume sweeps; current cold-start sessions do not preserve that lifecycle.

## Decision

Do not restore an HCC production runner and do not evaluate the selector yet.
First screen one representative fixed checkpoint per action, then promote only
mechanisms that preserve the independent runtime and pass the action-level gate.
