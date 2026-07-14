# ARAC Literature Review and Novelty Audit

Date: 2026-07-15
Scope: 2024-07-15 through 2026-07-15 (with older papers retained only as boundary precedents)
Executor: Codex

## Question

Does the proposed ARAC mechanism already exist in recent cooperative
coevolution (CC), large-scale global optimization (LSGO), or online algorithm
selection work, and what claim can still be defended after the 260-trajectory
held-out failure?

The audited claim was deliberately decomposed into four atoms:

1. overlap/group evidence is converted into a runtime action;
2. the action can alter shared-variable writeback, subproblem resources, or a
   search state;
3. a candidate action is evaluated against a fallback from the same optimizer
   checkpoint and the same FE budget;
4. an uncertainty/lower-tail risk rule can commit or abstain, with all probe FE
   charged to the run budget.

Atoms 1-2 are established prior art. The proposed differentiator is the
combination of atoms 3-4 in an in-run CC intervention protocol, with atomic
branch adoption and a hard FE ledger.

## Search Record

| Source | Query / endpoint | Window | Result |
|---|---|---|---|
| Crossref | `cooperative coevolution overlapping large scale global optimization` | date filter | HTTP 200; 685,703 broad records; screened top relevant records |
| Crossref | `cooperative coevolution subproblem selection resource allocation` | date filter | HTTP 200; 164,358 records; UCB-CC and TEVC resource allocation surfaced |
| Crossref | `dynamic algorithm selection black-box optimization` | date filter | HTTP 200; 546,818 records; restart, model training, GBDT selection and GNN prediction surfaced |
| Crossref | `risk-aware black-box optimization` | date filter | HTTP 200; 612,979 records; VaR-CMA-ES and constrained risk work surfaced |
| Crossref | `counterfactual intervention optimization` | date filter | HTTP 200; 278,759 records; no CC-specific paired-probe paper in the screened results |
| Crossref DOI lookups | 12 DOI endpoints listed below | n/a | HTTP 200 for every lookup; abstracts were not exposed for several ACM/IEEE records |
| DBLP | exact-title searches for GECCO/CoRR records | n/a | HTTP 200 for *How to Train...*, *Greedy Restart...*, and LH-CC; some later requests rate-limited |
| arXiv API | LCC, LH-CC, and Greedy Restart IDs | n/a | HTTP 200; abstracts available |
| arXiv API | CC + counterfactual/risk/paired/common-random-numbers combinations, submitted 2024-07-15..2026-07-15 | windowed | HTTP 200; 0 direct matches for each of the three exact combinations |
| Semantic Scholar | single DOI recheck | n/a | HTTP 429 in the final single-item retry; not used as sole evidence |
| Semantic Scholar | batch DOI lookup for the four specified records | n/a | HTTP 200; all four records parsed; GNN record has an abstract, the other three are publisher-elided |
| OpenAlex | DOI/work lookup | n/a | unavailable/timeout; not counted as coverage |

The zero-hit arXiv queries are negative evidence only: they do not prove that
no unpublished or non-indexed work exists. They do support the narrower claim
that no directly indexed preprint matching all of the paired-probe terms was
found in the stated window.

## Most Relevant Works

| Work (verified identifier) | What it covers | Collision with original ARAC claim | Defensible CAR delta |
|---|---|---|---|
| **Overlapping Cooperative Co-Evolution for Overlapping LSGO Problems**, GECCO 2024, DOI [`10.1145/3638529.3654171`](https://doi.org/10.1145/3638529.3654171) | Shared variables receive multiple assignments and overlap structure affects cooperative treatment. | Directly overlaps atom 1 and part of atom 2. | CAR must not claim that overlap-to-writeback/resource mapping is new; it adds same-checkpoint action calibration and abstention. |
| **Dynamic cooperative coevolution based on variable importance for non-separable LSGO**, Applied Soft Computing 2025, DOI [`10.1016/j.asoc.2025.113363`](https://doi.org/10.1016/j.asoc.2025.113363) | Dynamic CC driven by variable-importance evidence. | Covers runtime structure-driven adaptation. | CAR tests whether an intervention is useful before carrying it forward; it does not infer utility from importance alone. |
| **A Novel Two-Phase Cooperative Co-evolution Framework ... Complex Overlapping**, GECCO Companion 2025, DOI [`10.1145/3712255.3726560`](https://doi.org/10.1145/3712255.3726560), arXiv [`2503.21797`](https://arxiv.org/abs/2503.21797) | Two-phase treatment of complex overlap. | Overlaps the phase-based decomposition/action framing. | CAR's phase boundary includes an explicit paired counterfactual probe and a risk gate, rather than a fixed phase rule. |
| **Utilization of Upper Confidence Bound Algorithms for Effective Subproblem Selection in CC Frameworks**, Mathematics 2025, DOI [`10.3390/math13183052`](https://doi.org/10.3390/math13183052) | UCB/non-stationary UCB selects subproblems and balances exploration/exploitation. Crossref supplied an abstract. | Directly covers adaptive resource/subproblem selection (atom 2). | CAR does not learn a cross-run arm value; it compares candidate and canonical fallback at the same in-run checkpoint and can abstain. |
| **Evolutionary Contribution and Problem Heuristic Information Ensemble-Based Resource Allocation for CC**, IEEE TEVC 2025, DOI [`10.1109/TEVC.2025.3629151`](https://doi.org/10.1109/TEVC.2025.3629151) | Contribution/heuristic information used for CC resource allocation (title and Crossref metadata verified; abstract unavailable in the endpoint). | Directly overlaps contribution-driven resource allocation. | Resource allocation is a later CAR channel and is admitted only after a paired risk gate; no claim of a new allocation heuristic. |
| **How to Train Algorithm Selection Models: Insights from Black-box Continuous Optimization**, GECCO Companion 2025, DOI [`10.1145/3712255.3734311`](https://doi.org/10.1145/3712255.3734311) | Training/evaluating algorithm-selection models for black-box optimization. Crossref and DBLP metadata agree. | Covers learned selection from optimization features. | CAR is run-local and reference-blind; it does not train a selector on historical final outcomes. |
| **Geometric Learning in Black-Box Optimization: A GNN Framework for Algorithm Performance Prediction**, GECCO Companion 2025, DOI [`10.1145/3712255.3726696`](https://doi.org/10.1145/3712255.3726696), arXiv [`2506.16144`](https://arxiv.org/abs/2506.16144) | Heterogeneous graph + GNN predicts algorithm performance; the verified abstract reports BBOB/modCMA-ES/modDE experiments and up to 36.6% MSE improvement over a tabular model. | Covers graph-to-performance prediction. | CAR uses the graph to generate a candidate, then obtains local causal evidence by paired execution; it does not predict a final winner from a learned model. |
| **Automated algorithm selection for black-box optimization using light gradient boosting machine**, Swarm and Evolutionary Computation 2025, DOI [`10.1016/j.swevo.2025.102071`](https://doi.org/10.1016/j.swevo.2025.102071) | Feature-based automated algorithm selection. Crossref metadata verified. | Covers feature-to-algorithm dispatch. | CAR's primary evidence is an equal-budget intervention contrast, not a historical supervised label. |
| **Greedy Restart Schedules: A Baseline for Dynamic Algorithm Selection ...**, GECCO 2025, DOI [`10.1145/3712256.3726408`](https://doi.org/10.1145/3712256.3726408), arXiv [`2504.11440`](https://arxiv.org/abs/2504.11440) | Dynamic schedule selection on the distribution of unsolved problems. arXiv abstract verified. | Covers dynamic scheduling and restart selection. | CAR races actions inside one trajectory and charges probe opportunity cost; it does not choose a schedule from a cross-instance training distribution. |
| **Algorithm Selection with Probing Trajectories**, arXiv [`2501.11414`](https://arxiv.org/abs/2501.11414) | Short objective-per-FE trajectories feed offline algorithm-selection classifiers. arXiv abstract verified. | Closest probing precedent, but the probe describes an instance for a classifier rather than contrasting two actions from one optimizer checkpoint. | CAR must add native-fallback pairing, same-FE branch accounting, and deployment-time risk abstention; “short probe” alone is not new. |
| **Configuration Tuning for ISAC: Cost-Efficient Adaptation via RACE-CMA**, arXiv [`2604.05792`](https://arxiv.org/abs/2604.05792) | Two-stage racing, common random numbers, noise-aware ranking and constraints. arXiv abstract verified. | Directly covers racing/CRN ingredients, but in stochastic configuration tuning rather than overlapping CC state intervention. | CAR's delta is the complete optimizer checkpoint plus fallback-relative downside gate in one CC trajectory. |
| **Pareto-Optimal Anytime Algorithms via Bayesian Racing**, arXiv [`2603.08493`](https://arxiv.org/abs/2603.08493) | Bayesian racing and calibrated uncertainty for anytime algorithm comparison. arXiv abstract verified. | Covers uncertainty-aware pairwise algorithm comparison across instances. | CAR uses a fixed local horizon and commits one backend state; it is not a cross-instance Pareto-ranking framework. |
| **Distributed Bandit-Based Cooperative Coevolution for Large-Scale Multi-Objective Data Publishing**, IEEE TSC 2025, DOI [`10.1109/TSC.2025.3609875`](https://doi.org/10.1109/TSC.2025.3609875) | Bandit-based CC resource allocation (title and Crossref metadata verified). | Reinforces that bandit/resource dispatch is not a new ARAC claim. | CAR's distinct unit is a shared-variable intervention versus native fallback, with a same-run downside gate. |
| **CMA-ES with Individual Adaptive Reevaluation for Black-Box Value-at-Risk Optimization**, GECCO Companion 2025, DOI [`10.1145/3712255.3726637`](https://doi.org/10.1145/3712255.3726637) | Risk-sensitive evaluation in black-box optimization. Crossref metadata verified. | Supports the relevance of tail-risk objectives, not the CC intervention mechanism. | CAR uses a lower-tail gate to prevent harmful intervention; it does not redefine the benchmark objective as VaR. |
| **Advancing CMA-ES with Learning-Based CC for Scalable Optimization**, arXiv [`2504.17578`](https://arxiv.org/abs/2504.17578), GECCO 2026 DOI [`10.1145/3795095.3805053`](https://doi.org/10.1145/3795095.3805053) | PPO schedules decomposition strategies from optimization-status features. arXiv abstract verified. | Covers state-driven dynamic decomposition selection. | CAR does not train a policy; it obtains a short, same-state action contrast and can refuse to deploy it. |
| **A Learning-Based CC Framework for Heterogeneous LSGO**, arXiv [`2604.01241`](https://arxiv.org/abs/2604.01241), GECCO 2026 DOI [`10.1145/3795095.3805054`](https://doi.org/10.1145/3795095.3805054) | MDP/meta-agent chooses an optimizer per heterogeneous subproblem. arXiv abstract and DBLP metadata verified. | Covers adaptive optimizer/resource dispatch. | CAR's novelty is not optimizer selection; it is causal calibration and risk-bounded commitment of an existing action contract. |

Older boundary precedents remain relevant: contribution-based CC for
overlapping subcomponents (DOI [`10.1109/TCYB.2020.3025577`](https://doi.org/10.1109/TCYB.2020.3025577)) and sustainable CC with a multi-armed bandit (DOI [`10.1145/2463372.2463556`](https://doi.org/10.1145/2463372.2463556)) already rule out a novelty claim based only on contribution-driven action or bandit-style subproblem selection.

The racing ingredients also have established simulation-optimization
precedents, including constrained ranking and selection under common random
numbers (DOI [`10.1080/0740817X.2015.1009198`](https://doi.org/10.1080/0740817X.2015.1009198))
and Bayesian simulation optimization with CRN (DOI [`10.1109/WSC40007.2019.9004680`](https://doi.org/10.1109/WSC40007.2019.9004680)). They are boundary
references, not recent CC papers.

## Novelty Verdict

The broad statement “convert coevolutionary grouping and overlap into runtime
actions” is **not novel enough** on its own. It is an extension/combination of
OCC, dynamic CC, contribution-based CC, UCB-CC, and learning-based CC.

The defensible method-level claim is narrower, but the novelty audit classifies
it as an **incremental extension with a potentially distinctive protocol**, not
as an established “first” result:

> In overlapping CC, a graph-conditioned action is treated as a hypothesis,
> not as a dispatch decision. At a complete-component checkpoint, ARAC-CAR
> performs a same-FE, common-random-number, within-run paired probe against the
> canonical fallback, estimates a normalized short-horizon contrast, and
> commits the candidate only when a pre-registered conservative lower-tail gate
> passes; otherwise it atomically abstains.

This exact combination was not found in the screened 2024-07-15..2026-07-15
CC/LSGO records, but that is negative search evidence rather than a proof of
global priority. Confidence is **moderate**, not high, because ACM/IEEE full
abstracts were partly inaccessible and OpenAlex could not be used during the
final recheck. The safe paper framing is therefore “a same-budget causal
calibration and risk-commit extension of dynamic CC/action selection,” with an
explicit comparison against OCC, UCB-CC, learned CC, probing-trajectory,
racing, and algorithm-selection baselines. The five required differences are:

1. overlap/shared-variable backend intervention, not a solver arm alone;
2. an identical full optimizer checkpoint;
3. a candidate-versus-native-fallback paired contrast;
4. one same-budget CC trajectory that charges both probe arms; and
5. a fallback-relative downside gate with abstention.

Removing any one of these reduces CAR to an established ingredient (dynamic
CC, probing trajectories, CRN racing, UCB/LCB selection, or bandit resource
allocation).

## Implication for ARAC

The current v36 binary latch should be retired as the proposed contribution.
The held-out artifact shows why: 59/65 candidate/fallback pairs were identical,
only six changed, and one changed pair was a catastrophic loss. Static
support/maturity thresholds therefore neither provide coverage nor identify
long-horizon utility. CAR must be evaluated as a new protocol with fresh seeds,
not as another threshold revision.
