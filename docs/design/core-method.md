# Core Method: RDDSM Evidence Overlay

Date: 2026-07-18
Status: frozen observer-only pilot
Executor: Codex

## Positioning

The active ARAC method is an evidence-acquisition layer over RDDSM, not a new
decomposition algorithm. RDDSM remains the structural partition and Phase-II
topology. The overlay asks a narrower question: can a small, same-checkpoint
counterfactual probe make owner/action evidence more identifiable before the
existing ARAC decision layer is allowed to act?

```text
AOB design matrix
  -> reference-blind RDDSM partition and path order
  -> three complete Phase-I evidence sweeps
  -> immutable checkpoint and topology boundary
  -> owner/bridge counterfactual probes
  -> delayed survival and overwrite labels
  -> offline shadow decisions and promotion gate
  -> unchanged Phase-II optimization
```

## Structural Partition

RDDSM membership is reconstructed from the AOB design matrix. Runtime code must
not read `Pvector`, `subgroups`, oracle overlap labels, final outcomes, or
problem-family shortcuts.

Groups are ordered from their observed intersections. A valid overlap instance
must form one reference-blind path; non-path topology fails closed. When no
groups overlap, canonical member ordering is used and the overlay consumes zero
probe FE.

Only relations with exactly two direct owners are eligible for probing.

## Phase Boundary

The boundary is created at the first sweep end after:

- three complete sweeps have finished;
- the previous sweep's survival labels are closed; and
- every raw RDDSM group in the current sweep is complete.

The checkpoint FE, fitness-prefix hash, incumbent hash, RDDSM topology hash,
and group-order hash are immutable lane-pairing keys. After the boundary, the
RDDSM topology remains frozen through Phase II.

## Probe Selection

For relation `e`:

```text
D_e = shared-variable proposal disagreement
P_e = max(priority_left, priority_right)
VOI_e = harmonic_mean(midrank(D_e), midrank(P_e))
```

Ranks are computed within one trajectory. The evidence lane probes only when a
unique top-four boundary exists; a tie at that boundary abstains for the whole
trajectory.

The negative-control lane applies a deterministic, no-fixed-point cyclic
permutation using salt `arac-evidence-overlay-shuffled-v1`. If its selected set
still equals the evidence lane's top four, the shift advances until the sets
differ or the lane fails closed.

## Four-Point Counterfactual

Every selected relation evaluates four vectors from the same anchor:

- `x0`: unchanged anchor;
- `xL`: left owner's proposal written to the shared variables;
- `xR`: right owner's proposal written to the shared variables;
- `xB`: reliability-weighted bridge between the two owner proposals.

Bridge weights are proportional to `1 + owner_reliability`, with either owner
capped at `0.65`. The four direct objective calls consume exactly `4 FE`; four
relations consume at most `16 FE`. Repeated `x0` evaluations across relations
also act as a determinism check.

The overlay starts no additional CMA instance. It consumes no native RNG and
must not change the incumbent, cooperative context, CMA/MMES state, RDDSM
membership, group order, controller state, or native trace prefix.

## Budget Contract

Before probing, the strict ledger must prove that remaining budget covers:

```text
16 overlay FE + one complete native sweep + terminal tolerance
```

Insufficient budget yields a zero-FE abstention. Overlay FE is reported
separately as `evidence_overlay_fe`, while all native stages plus overlay and
overhead must equal the evaluation-record FE count.

Two terminal metrics are kept separate:

- `native_terminal_error`: excludes observer-only probe candidates;
- `all_evaluation_best_error`: includes every objective call.

## Shadow Decision

For candidate `a`:

```text
u_a = log((f(x0) + eps) / (f(xa) + eps))
```

A unique best owner with `u_a >= log(1.01)` maps to a shadow repair; a unique
best bridge maps to a shadow coordinate action. Every other outcome maps to
fallback. `runtime_authorized` is always `0`.

The next complete native sweep supplies delayed owner survival and overwrite
labels. Those labels are evaluation targets only and cannot affect the frozen
Phase-II route.

## Experiment Contract

The only active experiment is `exp_018`:

- lane A: native RDDSM audit;
- lane B: evidence-ranked owner/bridge probes;
- lane C: shuffled negative control;
- cases: `E1/E3/A4/S5`;
- observer-only in every lane;
- no Dynamic-CC lane and no action writeback in v1.

The mechanical and mechanism gates are implemented in
`experiments/pilots/exp_018_rddsm_evidence_overlay_pilot/protocol.py`. Any
integrity, identifiability, negative-control, delayed-closure, or catastrophic-
risk failure produces `pilot_no_go` without threshold retuning.

## Claim Boundary

The strongest permitted claim before the frozen gate passes is:

> ARAC defines a reference-blind, same-budget owner/bridge evidence overlay for
> testing action identifiability on a frozen RDDSM partition.

The pilot does not claim a new grouping algorithm, runtime performance
improvement, or authorization to change Phase-II actions.
