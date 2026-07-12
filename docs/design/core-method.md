# Core Method: Evidence-to-Intervention Utility Mapping

## One-Line Definition

`ARAC` studies how to turn grouping and overlap evidence from the first stage
of cooperative coevolution into runtime intervention actions in the second
stage, under a reference-blind and same-budget protocol.

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

## Canonical Innovation Chain

```text
Phase I: structure recognition and evidence collection
  -> groups, shared variables, overlap relations, group contribution,
     conflict, budget state, and search state
  -> Phase II: action decision and optimization intervention
  -> coordinate / isolate / repair / protect / fallback / trajectory
  -> change shared-variable writeback, inter-group coordination,
     subproblem resource allocation, and search starting state
  -> change the subsequent optimization trajectory
  -> change the final optimization result
```

Grouping is therefore not the endpoint of the method. It is runtime decision
evidence. The contribution is the closed loop from grouping evidence to an
executed and audited optimization action. A new optimizer, by itself, is not
the contribution; an optimizer is an action executor only when the controller
selects and configures it from admissible current-run evidence.

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
   - `trajectory`

   The policy must be allowed to abstain.

4. Backend intervention execution

   Bind the selected action to optimizer-consumed backend semantics, such as
   variable ownership, coordination mode, relation handling, budget allocation,
   update ordering, search starting state, trajectory continuation, or
   conservative fallback.

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

核心创新不是提出新的底层优化器，也不是单纯改进变量分解，而是在两阶段协同
进化中，把第一阶段得到的分组、共享变量、重叠关系、组贡献、冲突和预算状态等
运行时证据，映射为第二阶段的 coordinate、isolate、repair、protect、fallback
或 trajectory 动作。动作通过改变共享变量写回、组间协调、子问题资源分配和搜索
起点来干预后续优化轨迹。分组不是方法的终点，而是动作决策的依据；优化器不是
贡献本身，而是动作的执行器。整个映射必须 reference-blind，不得使用 final error、
oracle、reported baseline、历史结果或 problem-specific label，并通过同预算执行、
负控审计、后端语义差异审计和 catastrophic-loss gate 验证真实效用。

