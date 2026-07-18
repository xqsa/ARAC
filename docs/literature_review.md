# Evidence-Overlay and Conflict-Resolution Literature Boundary

Date: 2026-07-18
Scope: positioning for exp_018 and exp_019; not a priority claim
Executor: Codex

## Research Question

Does prior cooperative-coevolution work already provide the complete exp_018
combination: a frozen RDDSM partition, a same-checkpoint owner/bridge
counterfactual probe, strict in-run FE accounting, observer-only shadow action,
and a frozen Phase-II topology?

## Closest Prior Work

| Work | Covered capability | Boundary for exp_018 |
| --- | --- | --- |
| HCC, *A Novel Two-Phase Cooperative Co-evolution Framework for Large-Scale Global Optimization with Complex Overlapping*, DOI [10.1145/3712255.3726560](https://doi.org/10.1145/3712255.3726560) | RDDSM-style structure discovery and a two-phase HCC optimization backbone. | RDDSM remains the structural method and control. exp_018 does not claim a new partition algorithm. |
| Dynamic CC, DOI [10.1109/TEVC.2019.2895860](https://doi.org/10.1109/TEVC.2019.2895860) | Online decomposition/regrouping during optimization. | exp_018 freezes topology at the Phase-I boundary to preserve CMA/MMES state and attribution. |
| Contribution-based CC, DOI [10.1109/TCYB.2020.3025577](https://doi.org/10.1109/TCYB.2020.3025577) | Contribution-aware subcomponent treatment and resource decisions. | Proposal priority and owner reliability are evidence inputs, not sufficient action labels. |
| OCC, *Overlapping Cooperative Co-Evolution for Overlapping Large-Scale Global Optimization Problems*, DOI [10.1145/3638529.3654171](https://doi.org/10.1145/3638529.3654171) | Explicit treatment of shared variables and multiple assignments. | exp_018 probes competing owner proposals and a bridge from one checkpoint, without adopting any assignment. |
| OEDG, *An Enhanced Differential Grouping Method for Large-Scale Overlapping Problems*, DOI [10.1109/TEVC.2024.3390719](https://doi.org/10.1109/TEVC.2024.3390719) | Overlap-aware decomposition and the conforming/conflicting distinction. | It establishes that grouping overlapping variables is an active line; exp_019 does not claim a new grouping algorithm. |
| Blanchard et al., *Investigating Overlapped Strategies to Solve Overlapping Problems in a Cooperative Co-evolutionary Framework*, DOI [10.1007/978-3-030-85672-4_19](https://doi.org/10.1007/978-3-030-85672-4_19) | Direct comparison of strategies for overlapping variables, including the distinction between compatible and conflicting local optima. | It is prior evidence that overlap handling affects optimization; exp_019 isolates one synthetic value-resolution question rather than claiming the topic is untouched. |
| PACE, *Anytime-Valid Acceptance Tests for Self-Evolving Agents*, [arXiv:2606.08106](https://arxiv.org/abs/2606.08106) | Paired e-process acceptance under optional stopping in self-evolving agents. | This is a cross-domain acceptance-gate precedent, not a cooperative-coevolution or shared-variable value-resolution method. e-process is outside exp_019 v1. |
| LH-CC, *A Learning-Based Cooperative Coevolution Framework for Heterogeneous Large-Scale Global Optimization*, [arXiv:2604.01241](https://arxiv.org/abs/2604.01241) | Reinforcement-learning-based optimizer selection across heterogeneous subproblems. | It adapts optimizer choice rather than resolving conflicting proposals for one shared variable; it also prevents a broad priority claim over adaptive CC. |

Enhanced differential grouping and related decomposition methods further show
that structural detection accuracy is an established research objective. That
is why exp_018 optimizes for downstream action identifiability rather than
claiming that higher grouping precision alone is the contribution.

## Defensible Positioning

The broad idea "use grouping evidence to adapt cooperative coevolution" is not
new. The repository's prior search did not identify the complete exp_018
protocol as one existing method, but this is negative search evidence, not proof
of global novelty.

The narrow, testable positioning is:

> On a frozen RDDSM partition, collect a bounded set of same-checkpoint
> owner/bridge counterfactual evaluations so that later action evidence can be
> tested for identifiability, while charging every probe FE and leaving the
> native optimizer state untouched.

The potentially distinctive part is the protocol combination:

1. structural partition remains RDDSM;
2. eligible overlap relations have exactly two direct owners;
3. evidence and shuffled-control probes share an immutable checkpoint;
4. four direct candidates consume a fixed, explicit FE budget;
5. probe outcomes remain observer-only;
6. delayed overwrite/survival labels close on the next native sweep; and
7. Phase-II topology is frozen.

Removing these constraints reduces the method to established decomposition,
dynamic regrouping, contribution scheduling, or shared-variable assignment.

## Experimental Consequence

The first comparison is deliberately A/B/C:

- A: RDDSM native audit;
- B: evidence-ranked overlay;
- C: deterministic shuffled overlay.

Dynamic CC is deferred until B beats C on the preregistered mechanism gate.
Observer-only v1 does not require terminal improvement over A. It requires
exact FE accounting, owner identifiability, delayed-label agreement, positive
value over the shuffled control, and no catastrophic trajectory.

Any failed gate yields `pilot_no_go`; it does not authorize threshold tuning,
runtime action, dynamic regrouping, or a new performance claim.

## exp_019 Conflict-Resolution Boundary

exp_018 returned a real `pilot_no_go` on conforming AOB E3/A4/S5 cases. exp_019
therefore asks a narrower question: when adjacent owners have deliberately
conflicting local optima, does a frozen reliability-weighted bridge improve the
same-checkpoint objective over a preregistered owner baseline?

The controlled benchmark changes only the two local optimum copies of each
shared variable. It retains vendor topology, rotations, design, weights,
Pvector, bounds, and objective transformations. The resulting variants are
explicitly synthetic and non-official; they provide a mechanism diagnostic,
not evidence of terminal optimizer improvement.

The defensible positioning is limited to the following observation:

> In the sources identified by the current finite search, overlapping-CC work
> primarily addresses decomposition, assignment, overlap strategy, resource
> allocation, or optimizer selection. This search did not identify the same
> observer-only, same-checkpoint reliability-weighted value bridge tested with
> paired conforming and controlled-conflicting twins.

This is negative search evidence, not a global novelty proof. The project must
not say “no one has done this,” “first,” or equivalent absolute language.

## Search and Verification Limits

The four DOI records above were checked through the official CrossRef API; the
PACE and LH-CC identifiers and titles were checked through the official arXiv
API. OpenAlex discovery returned HTTP 429 during this search, so coverage is
necessarily limited. DOV is not cited because its bibliographic metadata has
not been verified. Those limits are part of the claim boundary, not evidence
supporting priority.
