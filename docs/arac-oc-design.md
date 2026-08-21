# ARAC-OC 运行设计契约

日期：2026-08-19  
执行者：Codex  
状态：ARAC-OC 主线设计已接入当前生产入口；阈值参数仍标记为未校准，不能据此宣称性能优越性

## 1. 目标与责任边界

ARAC-OC 的顶层责任是根据 Phase-I 的重叠证据，持续决定：

- 哪个共享变量或重叠 component 当前最值得处理；
- 应覆盖多大的变量范围；
- 预留多少 FE；
- 调用 CTP、GSS、SMP 还是 AOR。

GCB 是 coordinator，只产生调度计划，不是第五个搜索动作；CTP、GSS、SMP、AOR
是被调度的四个 search episode。历史 receipt 为兼容保留 `episode="gcb"`，
`episode_kind="gss"` 是公开名称映射。所有候选仍由真实目标函数评价，最终由同一个 strict-best ledger
维护全局 archive。

```text
Phase-I evidence
  -> immutable OverlapCheckpoint
  -> runtime CoordinatorState
  -> proposal sensing + counted conflict probes
  -> GCB coordinator dispatch plan
  -> one selected search episode for the plan
  -> strict-best arbitration
  -> feedback state update
```

## 2. 失败、升级与预算语义

### 2.1 AOR 不是异常 fallback

AOR 是一种预注册的正常升级动作。当 GCB 根据固定规则判断拓扑复杂、冲突
持续或局部修复价值不足时，调度计划显式写入 `action="aor"`、作用范围和
预留 FE。它与 CTP/SMP 具有相同的 operator contract，不代表前一个动作
发生了异常。

### 2.2 operator 异常必须 fail-closed

算子异常、契约校验失败、越界候选或状态 hash 不匹配都属于运行失败：

1. 已经发生的每一次目标函数评价保留在 `EvaluationLedger`；
2. 收据记录 `operator_failed`、动作名、已消耗 FE 和剩余 FE；
3. 不重试该动作，不尝试第二动作，不伪造 terminal `ActionResult`；
4. 本次运行向上抛出显式异常，不能把剩余 FE 静默交给 AOR。

因此“动作失败后转 AOR”不是异常处理路径。若需要 AOR，必须在动作调用
前由 GCB 产生 AOR 计划。

### 2.3 无收益不是异常

算子正常完成但没有 strict-best 改善时，记录 `no_gain`，更新停滞状态并
进入 cooldown。它可以在下一次预注册窗口再次被调度，但每个 component 在
一个 escalation window 内最多调用一次 AOR。一个 escalation window 定义为
连续 `k_window` 个协调器 cycle（版本化配置；默认值等于一次覆盖全部 active
component 的完整重探测 sweep 所用的 cycle 数）。连续无收益达到固定
`stall_cap` 后，该 component 停止新的 CTP/SMP/AOR 派发，剩余的 owner-local
SMP sense lane 不被关闭。sense/probe 仍按统一循环的预留预算执行；只有没有
可负担的 sense 窗口时，剩余预算才进入预注册 terminal tail，不执行隐藏重启。

成功运行必须精确到达总 FE。失败运行不宣称 terminal result，而是报告 ledger
实际计数和剩余预算。

## 3. checkpoint 与反馈状态

Phase-I 输出的 `OverlapCheckpoint` 永远不可变，包含：

- `S_g`：每个 group 的变量集合；
- `M(j)`：变量 `j` 的 owner 集合；
- `q_jg`：Phase-I 成员置信度；
- incumbent、Phase-I FE、协议版本和 checkpoint hash。

协调器不修改 checkpoint。每次运行另建可恢复的 `CoordinatorState`，至少包含：

- `qhat_jg`：运行时自适应信任值，初值为 `q_jg`；
- 原始冲突分数的 EMA；
- enter/exit streak、cooldown、stall count；
- 当前 budget pulse、已用/未用动作预算；
- 最近 dispatch 事件和 state hash。

运行时信任值使用固定 EMA 更新，不回写 Phase-I：

```text
qhat <- clip((1 - alpha) * qhat + alpha * realized_credit, 0, 1)
```

`realized_credit` 定义为本次算子实际 strict-best gain 相对调度前预测 gain
的比值，截断到 `[0, 1]`：

```text
realized_credit = clip(realized_gain / max(predicted_gain, gain_floor), 0, 1)
```

`gain_floor` 是版本化配置中的正常数下限，用于预测 gain 接近零时的数值保护。
算子达到预测收益时 credit 趋近 1，无收益时为 0。

credit 只分回本次 dispatch plan 成员覆盖的 `(j, g)` 对；多变量 scope 内
均匀分摊同一个 scope 级 `realized_credit`。单变量 scope（共识类动作）的
归因无歧义。多变量均匀分摊是 v1 的已声明粗糙性，按变量精细归因推迟到
v2 消融，不进入当前冻结语义。`CoordinatorState` 的每次更新都必须进入
snapshot/receipt，因此确定性重放依赖 `(checkpoint_hash, state_hash, seed,
config_hash)`，而不是依赖可变的 checkpoint。

## 4. SMP 的两个接口必须分开

SMP 有两个不同语义，不能继续使用一个隐含接口同时表达：

1. `sense()`：从持久 SMP 状态产生 owner-local proposal，提供候选值、
   不确定性和搜索范围；
2. `execute()`：当 GCB 选择 SMP 时，按 operator contract 执行被调度的
   状态记忆搜索。

冲突等级不由 SMP 的新旧状态直接决定。SMP proposal 只用于：

- 构造候选值；
- 选择需要探测的共享变量；
- 提供探针中心和尺度。

proposal residual 的持续性是 dispatch gate；SMP 状态只通过 proposal 的残差
和后续 qhat 反馈影响路由，不直接决定等级。这样一次偶然的 proposal 分歧不会
开火，只有连续高残差才会进入 CTP/SMP/AOR 路径。

## 5. B/W/C 与 counted probe

对 GCB 选定的共享变量集合，探针从当前 strict-best incumbent 构造完整上下文
的 `x_plus` 和 `x_minus`，并评价 `x0, x_plus, x_minus`。由这些真实函数值
计算：

- `B_j`：两侧响应的方向性偏差；
- `W_j`：两侧相对 incumbent 的局部响应宽度；
- `C_j`：归一化冲突强度，用于 scope 排序、探针尺度和诊断收据，**不再作为
  动作升级阈值**。

Gate 47 证明 C_j 幅值在 chain/star 拓扑之间不存在可用的阈值间隙；因此它不能
回答“冲突是否值得修复”。探针有两种粒度，用途不同：

- 逐变量探针（scope 排序用）：对 scope 内每个共享变量 `j` 单独构造
  `x_plus`/`x_minus`（其余坐标保持 incumbent），每变量 2 FE，一次 scope
  探针共 `2|S|` FE；`f(x0)` 复用 ledger 已有计数。它给出每个变量的
  `B_j/W_j/C_j`，是“先探测哪个变量”和审计解释的依据。
- 组件联合探针（已有 oracle 机制，`value_probe`）：整个共享集合同时
  两侧扰动，2 FE/组件，只给组件级修复价值估计，不做逐变量定级。

若剩余预算不足以完成当前粒度的探针，GCB 先尝试把 scope 收缩到可负担
的最大前缀；不存在可负担的非空 scope 时，记录 `probe_budget_unavailable`
收据、跳过本次 dispatch，剩余预算直接进入预注册 terminal tail。禁止把该
状态静默降级为 LOW 或升级到 CTP/AOR。

低冲突时的 weighted-mean、owner、median 和 incumbent 仍放入同一候选仲裁；
weighted-mean 永远不能自动提交。生产 unified loop 复用 strict-best archive
中 incumbent 的既有目标值，不为它重复支付 FE；因此每个冲突事件最多评价
三个新增完整候选。历史 oracle/对照入口保留旧的四候选计费口径，避免改变已冻结
实验的预算定义。

v6.0-a 还为仲裁胜出的完整候选登记一个冻结反事实收据：将选定 shared scope
恢复为仲裁前 incumbent、保留候选的私有坐标，再支付 1 次完整目标评价。定义

```text
G_full     = f(x_before) - f(x_candidate)
G_frozen   = f(x_before) - f(x_candidate with shared scope frozen)
G_coupled  = G_full - G_frozen
```

该评价只用于 shadow diagnosis，不改变 strict-best archive、不扩大四个 episode
的调度臂空间；收据包含 component、scope、候选名、三种 gain、FE 数和 archive
保持标志，可由 hash 复核。

### v6.0-a coupling diagnostic gate

在转入 scope 级调度前，先用固定 oracle gate 验证这个信号是否具有可解释性。
Gate 覆盖 chain/star 两种重叠拓扑，以及 `none`、`synergy`、`conflict`、
`neutral` 四种预注册 regime；每个 cell 使用独立 ledger、固定 incumbent、
固定 proposal，并另外运行相同 FE 的 sequential joint patch 对照。首轮 16-cell
结果（2026-08-20）为：

```text
median G_coupled: none=0, neutral=0, synergy=0.578125, conflict=-1.734375
correlation with joint-patch gain: Pearson=0.2582, Spearman=0.2294
```

所有 counterfactual 均恰好消耗 1 FE、保持 archive，仲裁和 patch 均保持
strict-best 单调且预算精确。该结果只证明 `G_coupled` 能在受控 regime 中区分
共享作用方向；相关性仍属中等偏弱，因此 v6.0-a 继续把它保留为 shadow receipt，
不直接扩大调度臂或改变 scope 优先级。

随后执行了独立的 200-cell fresh gate（2 topology × 4 regime × 25 seed，固定
bootstrap 2,000 次）。方向性结果仍稳定：synergy 的 `G_coupled` 95% CI 为
`[0.5111, 0.7355]`，conflict 为 `[-2.2065, -1.5333]`，none/neutral 的 CI
均包含 0，且 chain/star 分层方向一致。

但它与 8-FE sequential joint-patch gain 的预测关系不稳定：Spearman 中位数为
`0.0457`，95% bootstrap CI 为 `[-0.1168, 0.1612]`；预注册 authority threshold
为 `0.30`，因此 `promotion_recommended=false`。当前结论是：`G_coupled` 可以
作为“共享作用方向”的可审计证据，但不能作为“是否值得调度修复”的可靠代理。
v6.0-b 的 component-level EMA、scope priority 和 ticket 升级在此 gate 下保持
冻结，避免把诊断相关性误写成调度因果性。

### Two-baseline interaction diagnostic

为排除 shared-only 主效应，新增 v2 收据并行记录 private-only 与 shared-only
两个基线，定义纯交互收益：

```text
G_int = G_full - G_private - G_shared
```

该收据额外消耗 2 FE，评价后恢复 archive，但仍不进入 unified loop。对同一
200-cell fresh 协议的 paired gate 显示，方向性更清晰：synergy 的 `G_int` 95%
CI 为 `[1.0222, 1.4396]`，conflict 为 `[-1.4710, -1.0256]`，none/neutral
均为 0。然而 `abs(G_int)` 与 joint-patch gain 的 Spearman 中位数为 `-0.1109`，
95% CI 为 `[-0.2491, 0.0188]`，仍未达到 `0.30` authority threshold。

因此 v2 估计量解决了归因解释问题，但没有解决调度预测问题。当前协议冻结
v1/v2 收据为 shadow diagnostics，不实现任何 v6.0-b 的 EMA、scope priority
或 ticket 接管。

### Direct action-value gate

为避免继续堆叠耦合公式，新增一个匹配预算的端到端诊断：在相同
checkpoint、proposal、probe、component 和 seed 下，比较 `owner_control`、
`shared_sequential` 与 `shared_joint` 三个动作；每个动作支付 32 FE，随后三臂
都执行相同的 32-FE `full_context_writeback` handoff checkpoint。该 gate 的目的
是直接检验真实动作收益能否预测下一 checkpoint 的全局改善，仍不修改生产 GCB。

2026-08-20 的 60-context fresh matrix 完整性全部通过（proposal/probe/FE
parity、strict-best 和 handoff trace 均通过）。shared 动作相对 owner 的最佳
端到端 excess gain 中位数为 `258.7041`，win-or-tie 为 `0.6833`，说明共享动作
本身有组合价值；但即时 action excess 对 handoff 后续增益的 Spearman 为
`-0.8524`，95% bootstrap CI 为 `[-0.7249, -0.4524]`，即时收益对端到端收益的
Spearman CI 为 `[0.1905, 0.5547]`，动作收益选出的 arm 命中端到端 oracle 的比例
仅为 `0.55`。这表明即时收益会受到“先获得收益后剩余改进空间变小”的反作用，
不能直接充当未来 ticket/scope/action 的价值估计。

因此 direct action value 与 `G_coupled/G_int` 一样暂存为离线证据，但原因更具体：
共享动作可以保留在组合协议中，单窗口 material gain 不得直接升级为未来收益预测
或调度权威。只有重新注册带 maturity/revelation horizon 的多窗口证据并达到
`0.30` authority threshold，才可考虑接管调度；当前 v6.0-b 仍冻结。

### Multi-window maturity/revelation horizon gate

随后将同一 paired protocol 扩展为 3 个连续窗口。每个窗口固定支付
`4-FE arbitration + 32-FE action + 32-FE full-context handoff`，窗口间只接力
strict-best archive；owner、shared-sequential、shared-joint 仍复用同一 proposal，
不重置状态，也不读取未来窗口结果。rung-1/rung-2 信号分别表示前 1/2 个窗口
累计的 shared-vs-owner excess，future signal 表示其后窗口的 excess。

2026-08-20 的 60-context fresh matrix 完整性全部通过，但没有形成可授权的
maturity-aware value：rung-1 对后续收益的 Spearman 95% CI 为
`[-0.5369, -0.1686]`，rung-2 的 CI 为 `[-0.3710, 0.0277]`，均未达到 `0.30`
authority threshold。rung-2 的最终 horizon 选择命中率为 `0.70`，但这不抵消
shared action 相对 owner 的 horizon win-or-tie 仅 `0.2667`、median excess
`-499.8302` 的结果。

该 gate 的边界是受控的：它只说明在“固定 proposal、重复共享修复、archive 接力”
协议下，给动作更多窗口并不会自动揭示隐藏价值，甚至可能让 shared repair 落后
owner control；它不宣称所有可更新 proposal 的多窗口策略都失败。当前不得把
rung-1/rung-2 直接接入 maturity ticket、revelation horizon、scope priority 或
action selector。后续若继续，必须先注册带新 proposal sensing/状态更新的协议，
并区分动作成熟度与重复同一修复 kernel 的收益递减。

## 6. 固定阈值与分级策略

低/中/高/复杂的动作映射是固定设计，不在运行时根据 benchmark 身份或最终
结果调参。proposal residual 的持续次数、relative hub、qhat 信任和 AOR
升级次数进入版本化配置和收据。`tau_enter/tau_exit/k_enter/k_exit` 仍保留在
配置中用于 hash/兼容审计，但 Gate47-R 起不再是动作开关。

任何常数只允许在独立的离线 calibration/ablation 中确定。至少报告：

- 阈值网格对 dispatch 频率、FE、收益和失败率的影响；
- conforming/conflicting 两类数据上的相同配置结果；
- 固定阈值与不分级基线的配对比较。

仲裁后价值门 `arbitration_value_ratio` 遵循同一版本化纪律。它比较本周期完整
候选仲裁在 `error_after_sense` 上的相对改进：

```text
arbitration_ratio =
    max(0, error_after_sense - error_after_arbitration)
    / max(abs(error_after_sense), 1)
```

当该比例达到门槛时，GCB 将本周期原计划的 operator 改为
`arbitration_value_gate`（零 operator FE）。这不是按 benchmark 或拓扑的特判，
也不撤销已经由 strict-best 接受的仲裁候选；它只避免在同一 incumbent 已有实质
改进后继续支付低价值 pulse，保留下一周期的 SMP sensing 和完整 terminal tail。
该门槛在新 fresh-seed gate 中单独确认，未确认前不作性能优越性声明。

operator 窗口还有同尺度的 `operator_value_ratio` 后验门。operator 仍必须完整消耗
其 reservation；若实际 strict-best 改进低于门槛，只恢复 archive incumbent，FE
不会退还，receipt 以 `no_gain` 反馈给 stall/pulse。这样低价值 operator 不能仅凭
浮点级微小改进改变 terminal tail 的起点，同时真实 operator 路径仍被执行和计费。

对 shared-core CTP，GCB 还执行预算可行性检查：选定 scope 中每个共享变量至少
预留 2 FE（proposal 与 contrasting move）。当前 pulse 或可用 operator pool 不足
时直接产生 `shared_core_budget_unavailable` 的 arbitration-only receipt，不支付一个
结构上不足以运行联合修复的 pulse。

运行时映射为：

```text
LOW       -> 共识/owner/median 候选仲裁
MEDIUM    -> 受限 CTP 局部联合修复
HIGH      -> shared-core CTP 联合优化
COMPLEX   -> 预留预算内的一次 AOR 全局校正
```

候选组装在每个等级都执行：`incumbent + owner + consensus + median` 统一进入
strict-best 仲裁，不得无条件写回；生产 unified loop 中 incumbent 复用 archive
值，owner/consensus/median 等新增候选各消耗 1 FE，随后最多支付 1 FE 生成冻结
反事实收据。`LOW` 是唯一不额外 dispatch 算子的等级，其收益完全来自这套
候选仲裁。

## 7. 防振荡反馈回路

原始 `C_j` 先经过固定 EMA，仅用于诊断和 scope 优先级。动作升级使用
proposal residual persistence，不使用 C_j 幅值：

```text
dispatch gate: high proposal residual for `persistent_streak` consecutive cycles
escalation:    persistent streak >= `escalation_streak` and AOR not yet used
topology:      `relative_hub < complex_hub_ratio` -> restricted CTP
               `relative_hub >= complex_hub_ratio` -> shared-core CTP
trust:         `qhat_mean < smp_trust_floor` -> SMP state rebuild
```

每次真实 operator 调度后进入固定 cooldown，cooldown 内不得立即重新选择同一
component 的 operator。cooldown/stall 不影响该 component 的 `SMP.sense` 和
计数探针预算；所有 component 仍可提供 proposal，冲突转移可以改变 operator
优先队列，但不能绕过 cooldown。

预算采用有界 pulse：

```text
pulse_next = clamp(
    pulse * gamma_up   if realized_gain > 0 else pulse * gamma_down,
    pulse_min,
    pulse_max,
)
```

每个 pulse 的 FE 上限、增长/衰减系数和总预算预留在配置中固定。连续
`stall_cap` 次无收益后 component 停用；没有隐式重启或重复试错。

### 7.1 预算挤占的结构性不可达

Gate 36 的 `conflicting/star/overlap=6` 回归通道是隐式预算扣减：CTP 消耗
直接从 proposal-conditioned neighborhood 预算中减去。ARAC-OC 语义下该通道
必须结构性不可达，并以回归测试固定：算子只消费 dispatch plan 显式预留的
FE；neighborhood、sense、probe 和算子预算互不隐式侵占。测试断言任一
cycle 内各类预算的实际消耗不超过自身预留，且 dispatch 后其他预算类别的
可用量不因算子消耗而减少。

## 8. 目标运行循环

```text
Phase-I -> OverlapCheckpoint
state <- CoordinatorState(checkpoint)
while ledger.remaining > reserved_tail:
    sensing <- state.sensing_components()        # dispatch stall/cooldown 不关闭 sense
    proposals <- SMP.sense(incumbent, sensing groups)
    active <- state.active_components()          # 仅用于 operator priority
    scope <- GCB.select_scope(proposals, state)       # EMA(C_j) ranking only
    if 2*|scope| > affordable probe budget:
        shrink scope to largest affordable prefix
        if empty: receipt probe_budget_unavailable; break to terminal tail
    probe <- counted_probe(scope)                # 2*|scope| FE, f(x0) reused
    residuals <- B/W/C(probe)
    arbitration <- strict-best proposal arbitration
    state.observe_proposal_conflict(arbitration.conflict_level == HIGH)
    plan <- GCB.make_plan(residuals, topology, state)   # residual streak + hub + qhat
    if arbitration_ratio >= arbitration_value_ratio:
        plan <- arbitration_value_gate              # skip same-cycle operator pulse
    candidates <- incumbent + owner + consensus + median
    strict-best arbitrate(candidates)
    operator <- plan.action                      # smp | ctp | shared-core ctp | aor
    operator.execute(plan.reserved_budget)       # never implicitly reduces SMP base lane
    if operator_ratio < operator_value_ratio:
        restore_archive_without_refunding_FE()   # receipt=no_gain
    SMP proposal-neighborhood writeback           # fixed 32-FE/component base lane
    update state, EMA, cooldown, pulse, qhat     # scope members only, uniform credit
    on operator exception: emit failed receipt and abort
terminal tail <- pre-registered terminal policy
```

## 9. 当前实现边界

`run_arac_oc` 是唯一的 ARAC-OC 生产入口，执行完整的。入口有两个显式调度模式：

- `scheduler_mode="legacy_unified"`：保留旧 `oc_unified` coordinator loop，供兼容和历史对照；
- `scheduler_mode="v5_1"`：运行带 ticket/challenger/escalation/exploit、handoff 和审计收据的四-episode GCB coordinator。

v5.1 的命名边界为：

```text
GCB coordinator
  -> CTP | GSS(legacy receipt name: gcb) | SMP | AOR
```

两种模式都执行完整的：

```text
Phase-I pilot -> evidence probe -> GCB coordinator plan
-> CTP/GSS/SMP/AOR episode -> strict-best archive -> handoff/feedback
```

当前实现已具备：

- 字段级 `OperatorPlan` / `OperatorReceipt` 契约和 exact-FE/fail-closed 语义；
- GCB 的 component 优先级、scope 收缩、proposal persistence 分级、预算 pulse
  和动作计划；
- SMP sense/execute、受限 CTP、shared-core CTP 和有界全空间 AOR；
- 每个可负担 cycle 的固定 SMP proposal-neighborhood writeback 基线 lane；
- writeback 在 arbitration/operator 之后执行，使下一周期的 SMP sense 从完整的
  post-coordination incumbent 继续；该 lane 与 GCB operator reservation 分离；
- 一次无收益 operator 后的 stall guard 只做一次 arbitration 退避并关闭该组件的
  operator/probe priority，不关闭 `sensing_components()` 的持续 SMP sense；
- `C_j` 诊断 EMA、proposal persistence、cooldown、stall、qhat、snapshot/state hash；
- scope 真实传递到 repair operator，停用或 cooldown component 仍可消耗预留 sense FE；
- Gate 5/6 的显式重叠结构校验。

`run_overlap_from_pilot` 和旧 `gcb_coordinated` 仅作为历史/控制实验臂保留，
不会被 `run_arac_oc` 调用，也不定义 ARAC-OC 方法语义。默认配置中的
`UNCALIBRATED_FIELDS` 仍需通过独立校准门后才能支持比较性结论；Gate47-R
只验证调度路径和契约，AOB 端到端性能仍需另行注册并完成后续门。
