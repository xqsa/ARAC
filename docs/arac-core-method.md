# ARAC-Core 历史方法规范

日期：2026-08-09  
执行者：Codex  
状态：历史记录，不是当前生产方法。当前唯一主线是 ARAC-OC，规范见
`docs/arac-oc-design.md`，生产入口是 `arac.run_arac_oc`。本文保留旧 ARAC-Core
实验的来源、边界和结果，不能作为新实验的入口、动作选择器或方法定义。

## 1. 研究问题

ARAC-Core 研究以下受限问题：

> 仅根据 Phase-I 身份盲黑盒证据，能否选择一个机制匹配的 Phase-II 动作，并在相同总
> 预算下比固定使用单一动作更稳健？

该问题不要求每条运行都选中事后四动作 oracle，也不在运行时试跑多个动作。

## 2. 方法链路

```text
black-box objective
  -> one EvaluationLedger
  -> Phase-I evidence and immutable checkpoint
  -> one structural decision
  -> CTP | SMP | GCB | AOR
  -> selected action only
  -> exact terminal FE and strict-best result
```

Phase-I 当前使用 `arac-identity-blind-evidence-v9`。checkpoint 包含 incumbent、40 个
数值特征、变量块、块间关系和预算边界，不包含函数编号、family、benchmark 名称或人工
动作标签。

## 3. 重叠协调链

ARAC-Core 的正式 overlap-focused 路径不再把 GCB 仅定义为 sweep 顺序：

```text
Phase-I overlap evidence (through the fail-closed adapter)
  -> shared-variable local proposals
  -> residual B/W/C and persistent conflict
  -> CTP shared-core repair
  -> post-CTP refreshed proposals
  -> GCB counted two-sided objective probes
  -> selected component CTP dispatch
  -> if refreshed C remains high: bounded full-space AOR correction
  -> strict-best archive and feedback

历史入口为 `arac.overlap_core.run_overlap_arac`。它在 Phase-I 结束后只消费
`Phase1OverlapAdapter` 产出的 evidence interaction clique cover，不读取 benchmark truth。
每个 refresh cycle 从同一个当前 incumbent 为每个 evidence group 生成完整 proposal，随后对
每个 overlap component 依次执行：

1. 完整 candidate arbitration；
2. 每个 owner 一次 proposal/reflection context 写回；
3. 固定 32 FE 的 proposal-conditioned group neighborhood search。

局部候选只修改该 group 的全部坐标（共享变量和独立变量），但每次都写入同一个完整全局
context 并评价真实黑盒目标；`EvaluationLedger` 的 strict-best 负责接受/拒绝。这样
Phase-I 的 overlap evidence 直接改变“哪些变量一起被搜索、如何处理共享变量冲突”，而不只是
改变 sweep 顺序。
```

Oracle validation uses a known overlap structure only to isolate this Phase-II
mechanism. It does not claim that Phase-I already discovers the structure
correctly.

### 3.1 当前 Phase-I 接口门

当前 `PhaseCheckpoint` 只包含互斥变量分区 `blocks` 和块间
`RelationEvidence(left_block, right_block, strength, disagreement)`。每个变量在
checkpoint 中恰好属于一个 block，因此 relation edge 只能说明两个块的响应存在交互，
不能说明某个变量同时属于两个组。

`Phase1OverlapAdapter` 对这类 checkpoint 返回
`status="inference_incomplete"`、`reason="checkpoint_contains_partition_only"`，
并且不构造 `OverlapStructure`。只有未来 Phase-I 显式输出以下变量级证据时，才允许
进入共享变量协调链：

- 每个组的变量集合；
- 每个变量的多重 owner membership；
- 每个 `(variable, group)` 成员关系的置信度；
- 证据完整性标记，并且 groups 与 memberships 逐项一致。

这是一道 fail-closed 接口门，不是对 relation strength 的隐式重解释。当前新增的
`discover_overlap` 探针在 Gate 5 中完成了小维度 oracle-vs-inferred 验证：5 个 fresh
seed、5 种拓扑、5 种 base function、conforming/conflicting 两种模式，共 250 例；
200 个可辨识拓扑精确恢复 groups 和 shared variables，50 个 disjoint control 没有
共享变量误报，FE、确定性和 adapter readiness 全部通过。

Gate 5 使用多锚点二阶混合差分，单个 anchor 的成本为
`1 + d + d(d-1)/2`，因此当前实现是 `O(d^2)` FE 探针。它证明了接口和可辨识模型类，
但不适合作为大维度正式 Phase-I 的直接实现。正式接入前仍需设计稀疏候选筛选或分层探针，
并在真实 AOB/HCC 维度上重新通过 FE 和恢复率门控。

## 4. 历史选择规则（不属于 ARAC-OC）

ARAC-Core 使用无训练、确定性的结构规则：

```text
structural inference incomplete  -> AOR
complete and zero relations      -> SMP
disconnected relation graph      -> CTP
fully connected relation graph   -> GCB
```

这是一条机制路由规则。它主张动作与当前证据描述的结构相匹配，不主张提前知道哪一个动作
会成为某个随机 seed 下的终局冠军。

## 5. 历史执行契约（不属于 ARAC-OC）

- `run_arac(problem, total_budget_fes, run_seed, action_seed)` 运行完整两阶段方法。
- `run_arac_core(checkpoint, problem, ledger, action_seed)` 从冻结 Phase-I 边界继续。
- 正式路径使用 `ActionRegistry.execute()`，只构造被选动作的 `ActionContext`。
- ledger 必须恰好位于 Phase-I 边界；偏离边界时在动作执行前失败。
- 动作必须返回同一 checkpoint hash 并精确到达 terminal FE。
- 动作异常直接暴露；当前方法不尝试第二动作，也不做隐藏 fallback。

实现位于 `src/arac/core.py`，契约测试位于 `tests/test_core_policy.py`。

## 6. 贡献与非声明

当前拟议贡献：

1. 将有限黑盒评价转换为身份盲的结构 checkpoint。
2. 将结构完成度和关系拓扑映射到四种专用优化动作。
3. 在一个 FE 账本中完成一次决策、一个动作和可审计终值。

不宣称原创或当前已证明的内容：

- CMA-ES、Sep-CMA-ES、MMES 和底层数值优化器不是原创。
- AOB、IOH/BBOB 是测试来源，不是算法贡献。
- RF v3 结果不是跨函数或跨 suite 泛化证据。
- 当前规则不保证逐运行 oracle 最优，也尚未证明优于最佳固定动作。

## 7. 已完成的端到端验证

Gate 28 在 Gate 25 同一 1000-D shifted sparse-overlap objective、seed `20260825` 上运行：

- Phase-I：`180000 FE`；Phase-II：`2820000 FE`；terminal：`3000000 FE`；
- Phase-I checkpoint error：`664.7601031854064`；final error：`114.542357712886`；
- 16 个 refresh cycle，proposal budget `44051 FE/group/cycle`，每个 component 每 cycle
  固定 `32 FE` neighborhood；tail `39 FE` 精确补齐；
- proposal-only gain：`255.5694519614`；coordination extra gain：`294.64829351112`；
- 证据发现、adapter、shared variables、proposal coverage、FE reconciliation、strict-best
  和确定性重放全部通过。

这证明当前主线已经完成真实 Phase-I -> Phase-II overlap-aware 闭环。它不是跨 benchmark
泛化证明；Gate 25 的旧 `427.9864` 数值只作为历史参考，因其 checkpoint 标签包含
proposal FE，不能作为同协议公平对照。

## 8. 历史路线（已被 ARAC-OC 取代）

下一步不是固定四动作公平对照，也不是 RF selector。应先完成唯一缺失的接口：

1. 将 Gate 5 的二阶探针降为适合大维度的稀疏/分层探针，并在真实维度复验；
2. 在不读取 benchmark identity 的前提下生成 owner-local proposals；
3. 将同一个 checkpoint、ledger 和 proposal 流接入已验证的 residual、CTP、value-probe
   GCB 和条件 AOR 链；
4. 先在小规模 conforming/conflicting oracle-vs-inferred paired test 上测结构发现误差的影响；
5. 该接口门通过后，才进入 AOB-24 或独立 suite 的端到端实验。

最终实验仍需包含固定动作对照，但那是完整方法接通后的性能验证，不是当前下一步。

## 9. 历史与未来工作

- `docs/final-method.md` 记录历史 RF v3 proof-of-concept。
- `docs/phase2-v2-validation.md` 记录 common-anchor probe 的工程验证与负结果。
- RF selector、trajectory forecasting、common-anchor probe、delayed commit、survivor racing 和
  risk-aware online allocation 均属于未来工作，不进入 ARAC-Core 当前声明。
