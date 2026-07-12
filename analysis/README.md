# Analysis Contract

`analysis/` contains reusable offline aggregation, statistical testing, and figure/table build
code. It runs only after runtime execution has produced immutable run artifacts.

## Inputs And Outputs

- Inputs: completed run metadata and traces under `results/`, versioned experiment configs,
  paper evidence under `references/paper/`, and historical evidence under
  `references/historical/`.
- Outputs: deterministic summaries, confidence/statistical tables, gate reports, and plotting
  data under `analysis/generated/`; paper-ready sources may be promoted to `paper/` only with
  their upstream run IDs and claim level recorded.

## Git And Claims

Reusable scripts, schemas, and this contract are tracked. `analysis/generated/` is ignored
because it must be reproducible from inputs, code, and parameters. Analysis may compare against
paper or historical evidence but cannot rewrite runtime decisions or raise a scaffold, smoke, or
pilot beyond the claim level supported by its protocol and catastrophic-loss gates.
