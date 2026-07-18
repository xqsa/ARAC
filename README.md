# ARAC

This repository contains one active method line: the observer-only RDDSM
evidence overlay implemented by `exp_018`.

The overlay does not replace RDDSM and does not alter Phase-II optimization.
It freezes the RDDSM partition, collects same-checkpoint counterfactual evidence
for a small number of two-owner overlap relations, and records shadow actions
for offline identifiability tests. Every probe evaluation is charged to the
strict FE ledger.

## Active Method

1. RDDSM builds the structural partition from the AOB design matrix.
2. After three complete Phase-I sweeps, the topology and checkpoint are frozen.
3. The overlay ranks eligible two-owner relations using proposal disagreement
   and owner priority.
4. The active lane evaluates `x0`, the two owner proposals, and a reliability-
   weighted bridge for at most four relations (`16 FE` total).
5. Probe results never update the incumbent, optimizer state, native RNG,
   cooperative context, or Phase-II topology. Only an offline shadow decision
   is emitted.

The frozen lanes are:

- `a_rddsm_original_order`: native audit, zero probe FE.
- `b_rddsm_evidence_overlay`: evidence-ranked owner probes.
- `c_rddsm_shuffled_overlay`: deterministic shuffled negative control.

The active AOB cases are `E1`, `E3`, `A4`, and `S5`. AOB truth is used only
after execution for offline topology precision/recall; it is forbidden at
runtime.

## Repository Layout

- `src/arac/`: overlay policy, HCC adapter, execution contracts, and required
  reference-blind policy dependencies.
- `scripts/hcc_smoke_runner.py`: canonical real HCC/AOB execution runner.
- `experiments/pilots/exp_018_rddsm_evidence_overlay_pilot/`: the only active
  experiment and promotion gate.
- `configs/rddsm_evidence_overlay_pilot_v1.json`: the only active config.
- `vendor/hcc/`: the minimal HCC/AOB runtime plus F1/F3/F4/F5 inputs.
- `tests/`: focused unit, adapter, protocol, budget, and runtime tests.
- `docs/`: current method boundary and literature positioning only.

Historical experiments, generated results, binary LSGO branches, failed
controller variants, and copied source trees are intentionally absent. The
original evidence workspace at `E:\HCC-main` is read-only.

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[hcc]" pytest
```

## Verify

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m experiments.pilots.exp_018_rddsm_evidence_overlay_pilot.run --help
```

See `experiments/pilots/exp_018_rddsm_evidence_overlay_pilot/README.md` for the
frozen smoke and mechanism commands. Generated outputs belong under `results/`
and are not tracked by Git.

## Claim Boundary

`exp_018` is an observer-only identifiability pilot. Passing its gate may
authorize the design of a separate action-v2 experiment; it does not authorize
runtime writeback, dynamic regrouping, or a performance claim.
