# Exp 020: beneficial coordination pilot

This experiment isolates one existing Phase-II overlap action. It compares native
HCC overlap writeback (`conservative_no_action`) with the existing clipped blend
(`allow_beneficial_coordination`) on E3, A4, and S5.

Both lanes use seed 1 and 100,000 FE with strict accounting. Relation dispatch and
the evidence overlay remain disabled, so the only intended difference is the
overlap writeback action. The clipped blend implementation and its `[0.35, 0.65]`
weight bound are not changed by this experiment.

Run from the repository root:

```powershell
python -m experiments.pilots.exp_020_beneficial_coordination_pilot.run
```

Regenerable output is written under
`results/exp_020_beneficial_coordination_pilot/`. `decision.json` reports a
positive pilot effect only when at least two of three pairs improve, median log
gain is positive, no action run exceeds 1.2 times its paired baseline error, and
all paired FE ledgers close without overrun and within the preregistered 200-FE
terminal batch tolerance. Exact actual-FE differences remain in the paired audit.
This is a single-seed pilot, not a final statistical claim.
