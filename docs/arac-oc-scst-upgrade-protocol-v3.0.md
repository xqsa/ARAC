# ARAC-OC 升级协议 v3.0：结构认证的共享变量事务层（SCST）

日期：2026-08-23
基线：`arac-recovered-baseline-20260823-v1`
状态：**最终升级方案，待用户裁决后冻结**；冻结前不得运行任何新实验
上游资产：
- SCST 概念提案（`docs/proposal-not-shared-block.txt`）及三点收紧修正；
- v2.1 通用契约（`arac-oc-stepwise-upgrade-plan-v2.1.md`：零误差守卫、
  lane 预算纪律、统计口径）；
- v2.2 修订记录（`arac-oc-stepwise-upgrade-plan-v2.2.md`：S1/S2 关闭、
  宿主表达性规则、generator v3 稀疏拓扑规约）；
- U0/U1/S1 artifacts 与 Gate 53/54a 历史证据。

本文件是**唯一主线协议**。v2.x 的重排阶梯（S1/S2）已关闭；S3a/S3b/S4/S5
中的有效资产按下文 §8 映射迁移，其余不再作为升级候选。

---

## 1. 研究主张边界

只主张验证：

> 在相同 Phase-I checkpoint、相同外层动作、相同总 FE、相同 strict-best
> ledger 下，一个无状态、固定 FE、边界挂载的共享变量事务层，能否在
> conflicting sparse host 上产生可归因的**终值**增量，同时在 ov0 与
> AOB conforming 上保持逐位静默或终值非劣。

不主张：

- AOB conforming 上有终值改善（Gate 53 的预注册预测恰好相反，见 §6-T4）；
- 任何一致性分类、收敛性保证或与 ADMM 的等价性；
- 事务层改变 Phase-I 或 selector 的行为；
- 无状态 kernel 之外的一切（半径/方向/权重）——它们属于 T5 独立 rung。

## 2. 冻结边界与四动作宿主映射

不可修改：freeze manifest、recovered 四动作实现、Phase-I 180k FE 协议、
selector 输入输出、patch-off baseline 的 receipt/FE/hash contract。

| 动作 | SCST 角色 | 首版处理 |
|---|---|---|
| AOR | 无 owner block、无协调边界，proposal 集恒为空 | 完全关闭，合法静默，作为可信负对照 |
| SMP | stateful visits → rescue → global polish | 首选宿主：仅在 stateful-visits → rescue/global-polish 的首个合格边界执行一次事务 |
| CTP | coverage → relation-cover polish → MMES tail | 首选宿主：仅在 coverage → relation-cover polish 边界；绝不插入 coverage 内部 |
| GCB | source sweeps → full-space coordination → native sweeps | 只记录 proposal 与边界，不作为开发期 active host；待 T3 成功后作为冻结的迁移验证 host |

四动作共享同一个可审计接口；只有具备真实 owner 写回与下游重锚边界的
动作允许激活。

## 3. 核心定义契约

### 3.1 Proposal：写回工件，不是优化器内部猜测

统一定义：

```text
Proposal(g, j, phase) =
  block g 在当前 episode 的当前 phase 内，
  最后一次 strict-best 写回事件中，
  留在共享坐标 j 上的 committed incumbent value
```

每条 proposal receipt 必须记录：

```text
coordinate_id
owner_block_id
phase_id
commit_fe               # ledger FE 序号，不用墙钟时间
value_j
incumbent_hash_after
anchor_hash_at_visit_start
writeback_event_id
```

**没有 strict-best 写回就没有 proposal。** 禁止从 CMA 均值、局部 best、
最后一个未接受样本或历史缓存补造。

逐宿主冻结：

- CTP：从 coverage phase 内每个 block/session 的最后一次 strict-best 写回
  读取；只在 coverage → relation-cover polish 边界使用；
- SMP：从当前 stateful-visits phase 内每个 block 的最后一次 strict-best
  写回读取；首版只在 stateful visits → rescue/global-polish 的首个合格
  边界使用；
- GCB：只记录，不消费；
- AOR：proposal 集恒为空。

### 3.2 新鲜性

proposal 仅在**同一 run、同一 action episode、当前 source phase** 内有效。
跨 phase、跨 episode、跨 checkpoint 一律无效，不缓存、不接力。

### 3.3 结构认证（T0 的判定对象）

从 `checkpoint.blocks` 构建变量—owner incidence，确认 `M(j)`：

```text
|M(j)| >= 2 且结构证据可信，才允许 j 进入事务候选集
```

generator 侧前置：Phase-I 发现图必须稀疏且度数有差异；
**完全图或 max leverage == min leverage → fail**，不进入任何性能实验
（沿用 v2.2 §5 的 generator v3 规约：chain/pairs 优先、hub 度数 ≤3、
preflight 硬判据 leverage 方差 > 0）。

### 3.4 合格边界（qualified boundary）

一个边界只有同时满足以下三项才计数：

1. 前一 phase 有**至少两个新鲜 owner proposal**（§3.1/§3.2）；
2. 该边界允许执行事务（在 §2 宿主映射的白名单内）；
3. 下一原生 phase 会从 ledger 当前 incumbent 重建/读取锚点
   （即事务写回**会被下游消费**——P1 零触发教训的条款化）。

## 4. T2 机制契约：固定 FE、无状态事务内核

在每个合格边界：

1. 从事务候选集中按 `(-|M(j)|, coordinate_id)` 选**最多 4 个**坐标；
2. 每坐标生成两个**无权威**候选（不相信任何 owner 的局部收益方向）：

   ```text
   candidate-1: x[j] <- median(P_j)
   candidate-2: x[j] <- mean(P_j)
   ```

3. 每候选一次完整向量评价（其余坐标保持当前 incumbent），
   `consumed_fes = 2 × min(4, |候选坐标|) ≤ 8`；
4. `best_error_before` 从 ledger 读取，禁止重复评价；
5. 全部过 bounds / finite / strict-best；只有严格优于 before 才提交写回；
6. lane FE 从该动作预注册 operator reservation 划出；不借 sense/probe/tail
   或其他动作预算；未用额度明确返还并记录 `returned_fes`；
7. **无状态**：不维护 z/r/u/权重；不接受、不生产任何跨边界记忆。

禁止项（显式）：不使用 classifier、不使用 `u_j`、不使用分歧方向、
不使用 owner 权重、不新增任何 probe FE。

## 5. 阶梯：T0 → T5

### T0 — structure certificate

内容：§3.3 的 incidence 构建与认证 + generator v3 preflight。  
通过条件：认证规则可重放；generator 稀疏异质硬判据全过；完全图直接 fail。  
不评价性能。

### T1 — proposal & boundary audit（不加事务，纯仪器）

逐 action × case × seed 输出普查表：

```text
native phase boundary count
qualified boundary count
proposal-bearing boundary count
downstream re-anchor count
max patch FE = 8 × qualified boundary count
```

通过条件：

- proposal 定义（§3.1）在 CTP/SMP 上可观测、可重放；
- **传播证据**：若构造一次事务接受，下游阶段确实以新 incumbent 初始化
  （CTP 必须证明 relation-cover polish 读到更新后的 incumbent；SMP 必须
  证明 rescue/global stage 可见）；
- CTP/SMP 的边界数以实测为准，不预先假设很多。

**T1 完成后冻结**：每动作的最大 lane 预算（= 8 × 实测合格边界数）与
T3 的 seed 数。看结果后再调即视为协议违反。

### T2 — stateless kernel contract gate

§4 内核的契约验证（toy + matched host + snapshot/restore）：候选在界内、
exact FE、strict-best 单调、lane 只消费自身 reservation、无状态断言、
fail-closed、proposal 缺失时静默。不通过则不进入性能实验。

### T3 — matched-host attribution（conflicting 列，主判定）

数据源：generator v3 稀疏 conflicting cells。对照：

```text
A0: patch off（冻结 baseline）
A1: owner proposal 事务（median/mean，固定 ≤8 FE/边界）
```

统计口径（按真实效应尺度设定——一次性派发下每 run 仅个位数边界、
8–16 FE，AUC 不是合法主指标）：

- **主指标**：paired final-error log 差值 / 几何均值比
  （沿用 v2.1 §4.2 零误差守卫：`err_safe = max(error, eps_ref)`，
  eps_ref 逐 case 预注册）；
- **次指标**：patch acceptance、每 run 触发数、FE 组成；
- **探索性**：anytime AUC。

通过条件（screen → confirmation 两段，seed 集合 T1 后冻结）：

1. 全部 contract 审计绿；
2. 至少一个预注册 cell：geometric mean R < 0.95 且 95% paired bootstrap
   CI 上界 < 1.0；
3. 不得有任何 cell 的 CI 下界 > 1.05；
4. reachability：非零 patch receipt、候选 trace 非空、下游消费证据在链。

### T4 — AOB preservation（Gate 53 作为预注册预测）

预注册预测（写入协议，跑前冻结）：

```text
AOB/conforming 上 owner/consensus/median 事务近零接受；
若 proposal spread 不满足激活条件，lane 零消耗、route/FE/hash 精确保持；
即使发生合法候选评价，严格写回预期为零（Gate 53 的复现形式）。
```

通过条件：**不要求触发、不要求改善**；只要求 paired final error 非劣、
terminal FE / selector / action route 合规。

术语分开，不得混写：

- **零接受**：`accepted_patch_count = 0`（Gate 53 直接支持的是这项）；
- **零执行税**：`patch_consumed_fes = 0` 且 route/hash 逐位相同
  （能否声称由按 §3.1 新定义重跑的 T1/T4 receipt 决定）；
- **终值非劣**：有少量被拒绝候选但无预注册意义的回归。

### T5 — 独立 rung（每档单独协议，不可跳级）

仅 T3 通过后才启动，逐档比较：

```text
T5a - A1：持久 z,r（信任域半径；契约沿用 v2.1 §S4，含局部 context hash、
          self-accept 不 reset、lane 预算纪律 §4.3）
T5b - T5a：disagreement-direction candidate（契约沿用 v2.1 §S5a 的
          v2.1 修订方向定义：g* 对其余加权均值，平局 coordinate_id 字典序）
T5c - T5b：soft owner weight（契约沿用 v2.1 §S5b：α ≤ 0.2、单纯形投影、
          秩归一化、权重只影响候选中心）
```

任一档失败：保留上一档，不允许合并宣称。

## 6. 统计与记录协议

- common-random-number 配对；paired bootstrap 10,000 次为主，
  paired permutation 10,000 次为辅；
- 全部阈值、seed、case、配置运行前冻结；seed registry 冲突预检直接失败；
- 不允许按 final result 调整任何参数、边界定义或 gate 阈值；
- 每 cell 记录：final error、log10 anytime AUC（探索性）、checkpoint 误差、
  win/tie/loss、patch acceptance、触发数、FE 组成（lane/sense/probe/tail/
  operator）、exact-FE、strict-best、receipt/state hash 链。

## 7. 归因结构

```text
T3:  A1 - A0   = 无状态事务内核（median/mean 候选 + 边界挂载）
T5a: A2 - A1   = 持久信任域状态
T5b: A3 - A2   = 分歧方向候选
T5c: A4 - A3   = soft owner 权重
```

每次只动一列。A0（无新增 FE）不得与 T2 以后的 FE 机制混合解释。

## 8. 与 v2.x 资产的关系（显式映射，防双梯并存）

| v2.x 资产 | 去向 |
|---|---|
| S1/S2（重排/接续） | 已关闭（v2.2 §1），永久 |
| S3a 共享核 scope 重构 | **保留为后续结构级候选**，不进本主线。理由：SCST 不解决"共享变量被各 owner 重复扫、轮流拉扯"；S3a 是唯一消除拉据、且唯一在 conforming 侧有收益假设（早期方差压缩）的机制。作用于不同通道（扫掠结构 vs 边界候选），启动前提 = T3 通过 + 单独协议 |
| S3b 微补丁候选 | 由 T2 取代（边界挂载 + proposal 候选是更小的第一步） |
| S4 / S5a / S5b | 迁移为 T5a / T5b / T5c，契约沿用 |
| generator v3 稀疏拓扑规约 | 并入 §3.3，是 T0 前置 |
| v2.1 §4.2/§4.3（零误差守卫、lane 纪律） | 并入 §6 与 T5a，原样有效 |

## 9. 失败与止损线

- T0 失败：无可信共享结构，停止；记录为测试仪器/结构限制；
- T1 失败（无 proposal 或无传播边界）：精确说明失败于"没有可信共享结构"
  还是"没有可传播的执行边界"，停止性能实验；
- T2 失败：内核契约问题，修复重验，不进入性能实验；
- T3 失败：保留 A0；kernel 作为负结果记录；T5 不启动；
- T4 失败（conforming 回归）：事务层默认关闭，回退 A0；
- T5 任一档失败：保留上一档；
- 不允许通过修改 Phase-I、selector、冻结源、宿主生命周期或预算语义
  挽救任何一级。

## 10. 执行顺序

```text
T0 structure certificate（含 generator v3 preflight）
  -> T1 proposal & boundary audit（产出普查表 → 冻结 lane 预算与 T3 seed 数）
  -> T2 kernel contract gate
  -> T3 conflicting screen -> fresh-seed confirmation
  -> T4 AOB preservation
  -> （T3 通过后）T5a -> T5b -> T5c
  -> production E2E（全部前置通过后）
```

## 11. 论文逻辑闭环

```text
Gate 53：conforming 下共享候选不被全局目标接受（历史证据）
T1：    共享 proposal 与可传播边界真实存在（存在性）
T3：    conflicting 下小预算事务是否带来终值增量（有效性）
T4：    conforming 下机制保持静默或至少非劣（安全性/预测复现）
```

Gate 54a（不可辨识性）+ 本阶梯 = "不分类、不指定唯一 owner、不相信任何
耦合信号的方向权威；只认证结构、收集写回工件、在天然边界做一次
strict-best 裁决"。

## 12. 冻结参数表（待用户裁决后写入协议）

| 参数 | 建议值 | 来源 |
|---|---|---|
| 每边界候选坐标上限 | 4（≤8 FE/边界） | SCST 提案 |
| lane 预算 | 8 × T1 实测合格边界数 | T1 冻结 |
| T3 seed 数与集合 | T1 后冻结（screen/confirmation 两段） | T1 冻结 |
| eps_ref | 逐 case 预注册（基线全 seed 最小正终值 1/10） | v2.1 §4.2 |
| K_core / hub 度数上界 / leverage 方差阈值 | 50 / ≤3 / >0 | v2.2 §5 |
| T5 各档参数 | 沿用 v2.1 对应契约 | v2.1 |
