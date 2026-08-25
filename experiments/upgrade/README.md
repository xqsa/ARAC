# ARAC-OC Upgrade Lane

The recovered baseline is frozen by
`experiments/historical_recovery/recovered_baseline_freeze_protocol_v1.json`.

Run the baseline check before touching an upgrade candidate:

```text
$env:PYTHONPATH='.;src'; python -m experiments.historical_recovery.verify_recovered_baseline_freeze verify
```

Candidate code belongs in a versioned subdirectory here. Do not edit or overwrite
the frozen source, protocol, or evidence artifacts. Every candidate must register
U0-U4 and retain the baseline as the rollback target.
