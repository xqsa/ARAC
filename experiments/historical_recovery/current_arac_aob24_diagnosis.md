# Current ARAC-Core AOB-24 diagnosis

Date: 2026-08-11  
Executor: Codex  
Scope: read-only diagnosis of `artifacts/current_arac_aob24_recovery_v1`.

## Integrity result

- `600/600` current receipts are present and verified.
- All terminal FE counts are exactly `3,000,000`.
- All Phase-I FE counts are exactly `180,000`.
- All action results are bound to their generated Phase-I checkpoint.
- All terminal values are finite; there are no failed runs.
- The 600 current checkpoint hashes are identical to the corresponding
  checkpoints in `artifacts/historical_recovery_fixed_expert_v1`.

The current experiment is therefore a valid measurement of the frozen current
production chain. Its scientific recovery gate is still closed: `3/24` cases
meet the displayed reference mean under the strict rule `current <= reference`.
The passing cases are `S1`, `R2`, and `A1`.

## Routing diagnosis

The production rule in `src/arac/core.py::select_core_action` is a three-branch
structural rule:

1. incomplete structure -> `AOR`;
2. complete structure with zero relations -> `SMP`;
3. complete structure with relations -> `CTP` only when the largest component
   is disconnected, otherwise `GCB`.

It does not use the numerical landscape or progress features after they are
computed. In this campaign the resulting action counts are:

| Action | Count |
|---|---:|
| AOR | 0 |
| CTP | 50 |
| SMP | 76 |
| GCB | 474 |

Thus GCB concentration is a deterministic consequence of the rule, not a
selector-training or seed-sampling artifact. All E2-E6 and S2-S6 runs are
routed to GCB; all six of those cases fail their reference means.

## Counterfactual fixed-expert check

The prior fixed-expert campaign uses the same current v9 Phase-I checkpoint
protocol and the same 25 seeds. Replacing only the route with the historical
family mapping `A->AOR`, `E->SMP`, `R->GCB`, `S->CTP` gives these strict mean
passes:

| Case group | Fixed action | Strict passes |
|---|---|---:|
| A1-A6 | AOR | 1/6 |
| E1-E6 | SMP | 5/6 |
| R1-R6 | GCB | 1/6 |
| S1-S6 | CTP | 1/6 |
| **Total** |  | **8/24** |

The current fixed-expert summary records `A4`, `E2-E6`, `R2`, and `S2` as
strictly below the displayed reference mean, which is 8 cases when counting
the exact values in the frozen arm receipts; the displayed-precision summary
has a different match criterion and must not be substituted for the strict
gate. Among the 150 current/fixture arms where the selected action and the
family-mapped action coincide (`E1` and `R2-R6`), all 150 terminal errors are
bitwise identical. This confirms that route correction alone cannot close the
24/24 gate.

## Root-cause classification

### 1. Selector collapse

The current rule overweights graph connectivity and has no evidence branch for
the numerical regimes that distinguish the A, E, S, and R families. This
explains the `474/600` GCB count and the improvement from the S1 zero-relation
route, but it is not sufficient for recovery.

### 2. Action semantic drift

The current independent actions are not the historical action contracts at the
Phase-I boundary:

- CTP allocates almost all positive-relation budget to sequential block polish,
  while the historical S lane has a distinct relation-triggered full-space
  MMES tail.
- GCB uses current cold-start sweeps and a short coordination burst, while the
  historical R lane uses a phase-boundary burst followed by native resume
  sweeps.
- SMP's exact historical lockstep port is proven only for the zero-start,
  `phase1_fes=0` contract. The production two-stage path starts from a
  Phase-I incumbent, uses different block ordering/seed lifecycle, and splits
  budget across rescue and terminal alignment.
- AOR is a full-space continuation from the current checkpoint with only
  `2,820,000` action FE, whereas the historical A lane is a fresh full-space
  `3,000,000`-FE run from its own initial state.

### 3. Numerical warnings are not the primary failure

The stderr log contains 119 known PyPop7 overflow warnings (`94` scalar
multiply, `14` multiply, `11` exp), but all receipts are finite and all runs
completed. These warnings should be audited during action repair, but treating
them as the cause would not explain the deterministic routing and protocol
differences above.

## Decision

The AOB-24 current ARAC-Core recovery gate remains **failed**. Do not start
fairness, selector, probe, racing, or cross-suite experiments. The next
authorized work is action-contract recovery: port the evidenced CTP/GCB/SMP/AOR
state and budget semantics into the independent runtime, validate each action
on its own fixed checkpoints, then re-run the two-stage ARAC-Core gate with a
fresh output root. Production routing must remain identity-blind; no case-ID or
family-ID branch is permitted.
