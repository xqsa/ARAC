# exp_018 RDDSM Evidence Overlay Pilot

`exp_018` tests an observer-only active evidence layer over the frozen RDDSM
partition. It is not a replacement grouping algorithm. The overlay observes a
single Phase-I checkpoint, evaluates at most four owner relations with four
direct objective calls each, and records shadow decisions. It never writes a
probe result into the incumbent, CMA state, cooperative context, controller,
or Phase-II topology.

The three frozen lanes are:

- `a_rddsm_original_order` (`native_audit`)
- `b_rddsm_evidence_overlay` (`paired_owner`)
- `c_rddsm_shuffled_overlay` (`shuffled_owner`)

The mechanical smoke contains 15 fresh trajectories: `E1/E3/A4/S5`, seed 1,
three lanes at 100k FE, followed by `A4`, seed 1, three lanes at 3M FE. Both A4
cohorts require exactly 16 probe FE in B and C. The mechanism pilot contains 60
fresh 3M-FE trajectories over seeds 117--121 and runs with `jobs=24` only after
the bound smoke gate passes.

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m experiments.pilots.exp_018_rddsm_evidence_overlay_pilot.run `
  --stage smoke `
  --output-dir results\exp_018_rddsm_evidence_overlay_pilot

.venv\Scripts\python.exe -m experiments.pilots.exp_018_rddsm_evidence_overlay_pilot.run `
  --stage mechanism `
  --output-dir results\exp_018_rddsm_evidence_overlay_pilot
```

The mechanism command validates the smoke gate, frozen config hash, and source
bundle before consuming optimizer FE. Every invocation is fresh-only; a
non-empty stage directory or a source mode other than `fresh_runtime_probe` is
rejected before execution.

The promotion decision is intentionally fail-closed. Any missing raw artifact,
identity mismatch, non-fresh result, FE mismatch, changed AOB input, runtime
truth leakage, incomplete delayed label, or failed statistical threshold yields
`pilot_no_go`. A full pass authorizes only the design of a separate action-v2;
it does not authorize a runtime action or Dynamic-CC lane.
