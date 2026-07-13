# 二进制重叠 LSGO 接入 ARAC 设计

日期：2026-07-13  
执行者：Codex  
状态：待用户复核

## 1. 目标与边界

将已移植的 `BLSGO-F01` 至 `BLSGO-F18` 二进制重叠 LSGO 测试集接入 ARAC 当前的
“证据 -> 动作决策 -> backend 执行 -> 同预算评估”主链路。接入后，ARAC 的动作必须
改变二进制搜索器的实际行为，而不是只生成接口演示数据。

本次实现范围：

- 新增独立的二进制协同进化 backend，不调用连续型 HCC、CMA-ES 或 MMES；
- 复用现有 `EvidenceProfile`、`decide_action()` 和 `ActionDecision`；
- 在固定测试集种子和优化器种子下提供可重复执行；
- 对 baseline、ARAC action lane 使用相同问题、相同总 FE 预算和相同初始状态；
- 输出 final objective、实际 FE、动作轨迹、语义变化和同预算账本；
- 先提供 1-run pilot，支持后续扩展到多 seed。

明确不在本次范围内：

- 不修改 `E:\HCC-main`；
- 不把二进制向量编码后送入连续 HCC 优化器；
- 不把 final error、历史结果、论文 baseline 或 problem family label 注入 runtime
  evidence；
- 不在第一版复刻师姐原始搜索器的全部优化细节。原始搜索器后续可以作为同一
  backend 接口下的可替换 optimizer，但不能改变本测试集和 ARAC 动作契约。

## 2. 推荐架构

新增 `src/arac/backends/binary_lsgo.py`，保持 benchmark 生成和优化执行分离：

```text
BinaryLsgoSpec
    -> generate_binary_lsgo()
    -> BinaryLsgoProblem + BinaryLsgoTopology
    -> BinaryLsgoBackend.phase_one()
    -> BinaryLsgoEvidenceSnapshot
    -> EvidenceProfile -> decide_action()
    -> BinaryLsgoBackend.phase_two(decision)
    -> BinaryLsgoExecutionResult + SameBudgetLedger
```

backend 只依赖标准库、`BinaryLsgoProblem`、ARAC policy/evidence/action/evaluation
模块，不依赖 HCC 源码或可选的连续优化依赖。实验入口放在
`experiments/exp_009_binary_lsgo_arac_pilot/`，只负责读取配置、运行 lane 和写 CSV
及 manifest，不把优化逻辑复制到实验脚本中。

### 2.1 优化器状态

backend 维护一个全局二进制向量、当前目标值、每组统计量和共享变量所有权。基础搜索
器是确定性的分组 bit-flip 局部搜索：每次从当前组选择一个候选位翻转，只有目标值改善
时才接受；组按固定顺序轮询，候选位顺序由独立的 optimizer seed 派生。每次目标函数调用
计为 1 FE，候选评估不得绕过 `BinaryLsgoProblem.evaluate()`。

这样做的理由是：算法简单、无外部依赖、二进制语义明确，并能直接暴露共享变量的写回
行为。该优化器不是对论文 baseline 的复刻，也不以一次 pilot 的绝对数值宣称优越性。

### 2.2 两阶段预算

`BinaryLsgoExecutionRequest` 显式接收 `total_fes` 和 `phase_one_fraction`。默认 pilot
使用 `total_fes=2000`、`phase_one_fraction=0.20`。优化器 seed 为
`20260713 + case_index`，同一 case 的所有 lane 共用同一个 seed 和初始向量。backend
将预算拆成：

- Phase I：400 FE（包含初始向量的首次目标函数评估），用原生轮询搜索收集组级增益、
  共享变量冲突和排名稳定性；
- Phase II：剩余 FE，先构造 runtime 合法 `EvidenceProfile`，调用 `decide_action()`，
  再执行对应动作；
- 所有 lane 的 `phase_one_fe + phase_two_fe == total_fes`，超预算立即报错，不静默截断。

baseline lane 使用同一 Phase I 和同一初始向量，Phase II 选择
`conservative_no_action`。ARAC lane 使用同一初始向量和同一 FE 上限，仅改变动作语义。

## 3. ARAC 动作到二进制 backend 的绑定

动作绑定必须返回结构化的 `BinaryBackendSemanticsDiff`，并在动作轨迹中记录。第一版
支持现有策略实际可能输出的动作：

| Action | 二进制执行语义 |
| --- | --- |
| `conservative_no_action` | 原生组轮询，默认所有权，接受改善的 bit flip |
| `allow_beneficial_coordination` | 共享变量的候选翻转允许在相关组间协调写回 |
| `isolate_conflicting_relation` | 冲突共享变量只由当前所有者写回，非所有者候选不落地 |
| `repair_shared_variable_binding` | 根据 Phase I 组级改进，将共享变量所有权重分配给贡献更高的组 |
| `protect_high_margin_group` | 按稳定排名给高优先级组增加 Phase II 轮次，同时保持总 FE 不变 |
| 未支持的 action | 返回明确的 `unsupported_action` 失败，不静默当作 fallback |

动作改变的必须是优化器消费的状态：候选可写变量集合、共享变量 owner、组访问预算或
协调模式。每条 action trace 至少包含 action name、decision、触发原因、phase、受影响组、
受影响共享变量数、语义 diff、allocated FE 和 consumed FE。trace 中不写入 final error
或任何历史结果字段。

## 4. EvidenceProfile 构造

新增一个二进制 backend 内部的转换函数，将在线快照转换为现有 `EvidenceProfile`：

- `overlap_degree`：共享变量涉及的组关系占比；
- `shared_var_support_ratio`：共享变量数除以 decision dimension；
- `direction_disagreement`：相关组对共享变量候选改动方向的不一致比例；
- `harmful_coord_score`：被拒绝的协调候选与共享支持信号的组合值；
- `group_gain_asymmetry`：组级在线改善量的不对称程度；
- `priority_spread` 与 `rank_stability`：组级在线排名的离散和稳定性；
- `budget_remaining_ratio`：只根据当前请求预算和已消费 FE 计算；
- `fallback_margin_proxy`：由在线冲突/稳定性信号计算，不能引用最终目标值。

构造前调用 `validate_runtime_payload()`。`problem_id` 仅作为问题实例标识，不额外推导
family label；`BinaryLsgoTopology` 继续只存生成元数据，不存优化结果。

## 5. 实验输出与审计

`exp_009_binary_lsgo_arac_pilot` 对标准 18 个 case 至少提供以下文件：

- `execution_results.csv`：lane、case、seed、final objective、FE、状态；
- `action_trace.csv`：每次 ARAC 动作及 backend 语义变化；
- `same_budget_ledger.csv`：phase I/II FE、总 FE、预算上限和违规标志；
- `runtime_evidence.csv`：EvidenceProfile 字段和 anti-leakage 检查结果；
- `manifest.json`：配置、代码/输入哈希、固定种子和运行状态。

每个 case 固定运行三条同预算 lane：`native_baseline`、`arac_policy` 和
`shuffled_evidence_negative_control`。negative control 只对 Phase I 的组级在线统计做固定
seed 排列，再调用同一个 `decide_action()`；它不得进入主方法结果或被标记为正向收益。

结果比较只允许作为 offline evaluation：记录 ARAC lane 相对 baseline 的变化、是否触发
catastrophic-loss gate 和是否存在预算违规。pilot 不得因为单次 win 就生成“方法有效”的
结论。

## 6. 错误处理与安全边界

- problem dimension、向量长度、bit 值、FE 预算和 seed 非法时立即抛出 `ValueError`；
- 动作未绑定、共享变量 owner 不存在、FE 超预算时抛出可定位异常，并让实验结果标记
  `failed`；
- 不捕获宽泛异常后继续运行；
- 所有路径由实验入口显式传入，结果写入 `results/exp_009_binary_lsgo_arac_pilot/`；
- 不修改 benchmark 生成器的随机状态，不使用全局随机数。

## 7. 测试策略

先写失败测试，再实现代码。测试覆盖：

1. backend 在固定问题和 seed 下完全可复现；
2. 每个 lane 恰好消费 `total_fes`，预算违规为 0；
3. baseline 与 ARAC lane 初始状态一致；
4. coordinate、isolate、reassign 和 protect 分别改变预期的 backend 语义；
5. 未支持动作明确失败；
6. evidence 通过 runtime 边界校验，不包含禁止字段；
7. pilot 输出覆盖 18 个标准 case，CSV 列和 manifest 固定；
8. negative control（固定随机动作或打乱 evidence）不能被标记为 ARAC 正向结果；
9. 完整现有测试集不回归。

## 8. 版本交付顺序

1. 提交本设计文档并由用户复核；
2. 用 `writing-plans` 生成实现计划；
3. 先添加 backend 合同测试，再实现 backend；
4. 添加 pilot 和输出测试；
5. 运行聚焦测试、完整 pytest、`compileall`、`git diff --check`；
6. 查看精确 diff 后提交实现版本。推送远程仓库前另行取得确认。
