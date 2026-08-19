# Phase-II v2 验证协议

> **未来工作状态（2026-08-09）**：本协议及冻结结果继续保留，但 common-anchor probe、
> delayed commit 和 racing 已退出当前主线。当前方法为 `docs/arac-core-method.md` 定义的
> Phase-I 证据到单一 Phase-II 动作的 ARAC-Core。

日期：2026-08-09  
执行者：Codex  
状态：协议、本地状态机及 AOB+IOH 预注册对照验证已完成；当前 probe policy 不晋级。

## 目标

验证“同一 Phase-I checkpoint -> 少量等额分支预算 -> 延迟提交一个动作 ->
提交后只运行该动作”的在线协议。该验证不训练 Random Forest，也不把四个
动作的终值标签输入运行时。

## 适配器边界

每个 benchmark suite 必须提供一个薄适配器，输出统一的
`OptimizationProblem`：

- `objective` 接受单个向量或二维候选批次，并返回对应有限标量值；
- `dimension`、逐坐标 `lower_bounds`、`upper_bounds` 和可选 `optimum`；
- 不向 `PhaseCheckpoint`、动作、轨迹或策略传入函数 ID、family、实例名、
  benchmark 标签或人工动作映射。

AOB 继续使用 `src/arac/benchmarks/aob.py`。外部套件必须使用独立的函数/实例
命名空间和未参与 AOB 设计的 seed 集；适配器测试只验证数值表面和边界，不验证
动作优劣。

## 在线协议

1. 在 Phase-I 终点创建一个 `CommonAnchorProbe`，四个分支使用独立
   `EvaluationLedger`、相同 checkpoint 和等额 FE。
2. 分支以固定的 `initialize/step/snapshot/resume` 状态机推进。推荐初始探针预算
   为 Phase-II 剩余预算的 5%（同时设置绝对上限），但预算参数必须写进实验清单，
   不能由测试集身份决定。
3. 从 `probe.trajectories` 读取四条 best-so-far 前缀，在保护下限之后调用
   `decide_delayed_commit`。该规则只使用当前观测轨迹、相对 margin 和 leader
   边际收益稳定性。
4. 规则返回 `None` 且未到预注册 cap 时，只允许扩展同一 probe；到 cap 后按当前
   leader 确定性提交，并把 `probe_cap_*` 原因写入收据。不得重建分支或重放 FE。
5. 规则返回一个动作后调用 `probe.commit(action_name)`。从该点到终点只暴露被选
   分支，其余分支永远不再消耗 FE。
6. 四分支 probe 均计入 global FE；被选状态在 global FE 耗尽处返回可恢复前缀，
   不伪造完整 `ActionResult`。每条运行必须精确预算、单一提交且失败显式记录。

## 审计指标

正式运行报告：终值误差、FE、失败数、commit 时间和 abstain/extend 次数。只有
在离线审计中才计算四动作 oracle regret、crossover rate 和 horizon rank；这些
指标不能反向改变运行时阈值。

在 AOB 与独立外部套件上分别报告：

- 无训练的 deterministic mechanism-score baseline；
- v2 delayed-commit 的终值 regret 和样本标准差；
- 每个 suite 的函数级结果和跨实例聚合结果；
- 适配器、配置、源码和 vendor tree 哈希。

## 当前结果与停止条件

IOH `0.3.22` 已冻结。预算中性 6-run pilot 位于
`artifacts/phase2_v2_pilot_ioh_v3`：6/6、零失败、每条精确 `4096 FE`，Phase-I
和 action schedule 都绑定同一 global budget。manifest 为
`1842f37b1a847058bad0ddd4489fb1957ba8fc328639e86cbd94ecfa31cd0da5`。

预注册对照验证位于 `artifacts/phase2_v2_validation_ioh_v2`：AOB
`A1/E1/R1/S1 x 2 seeds` 和 IOH BBOB `F1/F8/F15/F21 x 3 instances x 2 seeds`，
共 32 contexts、64 方法运行。全部精确 `40000 FE`、零失败、单一提交且方法间
checkpoint hash 相同。运行 manifest 为
`29ae0964098c4ff2737b72c97db1fcf150b69c4a36384fe55d76937a07369dbc`。

独立 shifted-log 分析位于
`artifacts/phase2_v2_validation_ioh_v2_analysis_v1`，manifest 为
`a911bb9a7cbb2ae8e3ec05723842d66acf5a3fdbf9ab5a0d249b47f367d583f2`：

- AOB：probe `0` 胜、`6` 负、`2` 平，mean shifted log ratio `+0.0193`；
- IOH：probe `7` 胜、`7` 负、`10` 平，mean shifted log ratio `-0.0358`；
- 全部：probe `7` 胜、`13` 负、`12` 平，中位 shifted log ratio `0.0`；
- `24/32` probe 到 cap 仍 margin 不足，只 `8/32` 达到 stable margin；
- 4 条 probe 运行触发 3443 次 MMES sigma-floor 事件，机制基线为 0。

第一轮冻结验证曾暴露 MMES sigma 下溢为 0；通用机器精度 floor 修复后，同配置
v2 为 64/64。修复次数进入快照和收据，没有静默丢 FE。当前结果证明协议可执行，
但不支持 probe policy 优于机制基线：AOB 明显更差，IOH 胜负相抵且 fallback 率
过高。因此不得启动 calibration、holdout 或 E2E，也不得恢复 v6。下一步若继续
研究，应先解释短前缀 margin 不足和动作 crossover，而不是增加训练选择器。

## AOB 保真分解

后续只读审计位于 `artifacts/phase2_v2_aob_preservation_audit_v1`，其 manifest
为 `72f4b4dc3ac8531f8a44f09b80d38d2b6493cb41d9800504aed36a4485867de3`。v2
对照的总预算为 `40000 FE`，所以 Phase-I 只有 `2500 FE`；A1/E1/R1/S1 的
8 个 context 全部 `structural_inference_complete=0`，probe 和 mechanism
均路由 AOR。probe 的 selected ledger 是 `38464 FE`，比无 probe 的机制运行
少 `1536 FE`。

原 v3 的同四个 case 参考收据使用 `3000000/180000 FE`，100 条记录全部完成
结构推断，动作分布为 CTP `24`、GCB `5`、SMP `71`、AOR `0`。因此 v2 的
AOB 负结果不能直接解释为原 AOB 优势消失；它首先是 Phase-I 边界不一致和
probe tax 的诊断结果。下一步只允许先做恢复 `180000 FE` Phase-I 的 AOB-only
pilot，并加入 overhead-matched control。
