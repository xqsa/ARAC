# Core Method: Evidence-to-Intervention Utility Mapping

## One-Line Definition

`ARAC` studies how to map uncertain overlapping structure evidence into
backend intervention actions under a reference-blind and same-budget protocol.

The method moves cooperative coevolution from:

```text
detect interactions -> decompose variables -> optimize subcomponents
```

to:

```text
observe dynamic structure evidence -> choose or block backend intervention
actions -> verify action utility under same-budget execution
```

## Research Problem

Large-scale overlapping optimization is difficult because shared variables and
overlap relations are not always beneficial cooperation signals. The same
overlap pattern may require coordination, isolation, protection, repair, or no
action, depending on dynamic optimization evidence.

The central gap is:

```text
structure evidence is not action utility.
```

A method that only detects variable interactions can still fail if it maps the
detected structure to the wrong backend behavior.

## Core Hypothesis

Reference-blind Phase-I trace features can provide enough evidence to select
safe and useful backend interventions, provided that the policy explicitly
models gain, cost, fallback safety, and catastrophic risk.

## Main Pipeline

1. Phase-I evidence collection

   Collect trace-derived features from an initial optimization stage. These
   features describe shared variables, overlap relations, group behavior,
   resource state, and uncertainty.

2. Structure feature modeling

   Convert trace rows into evidence profiles. Each profile must be independent
   of final outcomes, oracle labels, reported baselines, and problem-family
   shortcuts.

3. Evidence-to-intervention mapping

   Map an evidence profile to one of the action families:

   - `coordinate`
   - `isolate`
   - `protect`
   - `reassign_repair`
   - `fallback`

   The policy must be allowed to abstain.

4. Backend intervention execution

   Bind the selected action to optimizer-consumed backend semantics, such as
   variable ownership, coordination mode, relation handling, budget allocation,
   update ordering, or conservative fallback.

5. Same-budget utility evaluation

   Count Phase-I and Phase-II function evaluations together. Compare the action
   lane against fallback, no-action, uniform, shuffled, or external final-only
   references without using final-only data during runtime dispatch.

6. Risk and audit gates

   Verify anti-leakage, backend semantics change, action effect attribution,
   negative controls, same-budget accounting, and catastrophic-loss gates.

## Utility View

The policy is not merely a classifier. It is a utility mapping:

```text
pi(a | e): evidence profile e -> backend intervention action a

U(a, e) = expected_gain(a, e) - action_cost(a, e) - risk_penalty(a, e)
```

An action should be admitted only when:

```text
feature coverage is sufficient
and trigger evidence is stable
and negative controls pass
and expected utility is positive
and fallback gap is safe
and catastrophic risk is low
```

Otherwise the policy should choose `fallback`.

## Contribution Statements

Contribution 1:

```text
We formulate uncertain overlapping optimization as an evidence-to-intervention
problem, where dynamic overlap and shared-variable evidence must be mapped to
backend actions rather than only to decomposition decisions.
```

Contribution 2:

```text
We design a reference-blind action policy that uses only Phase-I trace-derived
features to select, block, or fallback among intervention actions, preventing
leakage from final outcomes, oracle labels, or reported baselines.
```

Contribution 3:

```text
We introduce a same-budget utility evaluation protocol with negative controls,
backend-semantics auditing, action-effect attribution, and catastrophic-loss
gates to verify whether an intervention action provides real optimization
utility.
```

## Chinese Short Version

核心创新不是提出新的底层优化器，也不是单纯改进变量分解，而是提出一套
面向大规模重叠优化的 reference-blind evidence-to-intervention mapping。
该方法从 Phase-I 优化 trace 中提取 shared-variable、overlap relation、
group behavior 和 resource state 等动态证据，在不使用 final error、oracle、
reported baseline 或 problem-specific label 的前提下，将不确定结构证据映射为
coordinate、isolate、protect、reassign/repair 或 fallback 等后端干预动作。
每个动作必须通过同预算执行、负控审计、后端语义差异审计和 catastrophic-loss
gate 验证其真实效用。

