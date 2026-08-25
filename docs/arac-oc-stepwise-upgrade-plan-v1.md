# ARAC-OC 阶梯式升级方案 v1（基于 recovered baseline）

日期：2026-08-23
基线：`arac-recovered-baseline-20260823-v1`（冻结协议
`experiments/historical_recovery/recovered_baseline_freeze_protocol_v1.json`）。
状态：提案，待用户裁决后逐级预注册。

## 0. 设计公理

1. **恢复基线不可侵犯**：每级升级都是可选 feature；任一级 gate 失败 = 该级关闭，
   生产回退到上一级，基线永远可复现。
2. **每级只加一个机制**：收益必须可归因到该级；不允许捆绑。
3. **conforming 的诚实预期**：Blanchard 阴性结果 + 54a 不可辨识性 →
   conforming 列的收益形式是 **anytime 效率（AUC）与早期方差压缩**，
   终值判据一律非劣；终值改善的期望只放在发生器 conflicting 列。
4. **FE 开销逐级递增**：前两级是纯重排（零 FE 风险），后级才引入有界 FE lane。
5. **挂载点必须在流量路径上**（P1 零触发教训）：机制挂的位置必须是
   生产循环真实经过的代码路径，gate 判据含"机制实际触发次数 > 0"。

## 1. 创新论点（论文叙事）

conforming 重叠上文献只留下三个被观察过、但未被系统利用的事实：

- **杠杆**：共享变量一单位改善同时改善多个 owner 子函数（CBCCO/HCC 的结构事实）；
- **传播**：OCC 在 conforming 函数上的胜因是"相邻组件交替产生新最优"——
  改善通过共享变量传给邻居（GECCO'24）；
- **早期方差**：HCC 观察到 CC 在重叠问题上早期高方差（共享值被反复拉扯）。

阶梯的故事：**把"重叠结构"从 benchmark 标签变成扫掠顺序、传播拓扑和
共享坐标搜索空间——逐级加码，每级独立证明价值。**

## 2. 阶梯定义

### S0：基线护栏（无机制）

- 冻结 recovered baseline 的 receipt/账本/审计作为全部后续 gate 的对照锚；
- 定义每级通用判据（全部同时满足，预注册后不调）：
  1. ov0 零税（机制不改变任何行为，开销有界可忽略）；
  2. conforming 列 median anytime AUC 相对基线非劣（预注册阈值）；
  3. 终值相对基线非劣（ratio ≤ 1.05，paired CRN）；
  4. 契约审计全绿（exact FE / strict-best / receipt 链）；
  5. **机制触发计数 > 0**（防空虚通过）；
- 失败语义：关闭该级 feature，如实记录，不允许 gate 内调参。

### S1：杠杆优先扫掠序（纯重排，零 FE）

- CTP/GSS 的块扫掠中，含共享变量的 scope 获得优先序位；
  权重 ∝ 该 scope 含共享变量数 × Phase-I 成员置信度（全部来自既有 checkpoint，
  无运行时新信号）；
- 早期份额有界（预注册，如每次扫掠前 20% 槽位），之后恢复原顺序；
- ov0 时顺序不变（零税由构造保证）；
- 文献依据：杠杆论点 + CBCCO/OCC 在 conforming 上的顺序敏感性证据。

### S2：传播接续规则（纯重排，零 FE）

- 某 scope 产生 strict-best 接受后，与其共享变量的相邻 scope 获得下一个
  扫掠槽位（OCC 式传播规则的扫掠序版本）；
- 只加一个指针状态（last-improved scope），无候选、无新评价；
- 与 S1 的关系：S1 是静态优先，S2 是事件驱动接续；S2 的 gate 对照是 S1。

### S3：共享坐标微补丁（首个真正动共享变量的机制，有界 FE）

- 每次 CTP/GSS scope 访问，固定 lane（预注册 8 FE，从该动作自身预算划出，
  不借别处预算）在该 scope 的共享坐标上试 ±r 候选；
- 全部完整目标评价 + strict-best 过滤；无状态、无分类、无 proposal 依赖；
- 这是方法第一次**直接搜索共享变量坐标**（而非仅在它们周围排序）；
- gate 对照为 S2；额外记录 lane 接受率。

### S4：状态化信任域（z_j, r_j + 上下文局部哈希）

- S3 的半径变为逐共享变量持久状态：成功扩、失败缩；
- context_hash **只覆盖该变量的上下文坐标**（排除 patch scope 自身，
  修复"自致重置"缺陷）；上下文变化 → r 回 base；
- 不设 u_j、不设强度调节（留给 S5）；gate 对照为 S3，
  隔离"持久状态"本身的增量。

### S5（conflicting 列扩展，不进 AOB 主线）

- 加入 disagreement 方向候选 + 软 owner 权重 w_jg（单纯形乘法更新，
  秩归一化、变化率有界）；
- 评估列 = 自有发生器 conflict 模式（真值可取，相对口径判据）；
- AOB 上预期结构性静默（分歧≈0 → 候选退化）——此级在 conforming 列的
  判据仅为非劣。

## 3. 评测矩阵

```text
              判据
ov0 列        每级：零税（行为不变）
AOB 列        S1–S5：anytime AUC 非劣/改善 + 终值非劣（S5 仅非劣）
发生器冲突列   S3–S5：相对口径终值改善 ≥1 处（S5 主战场）
```

全部 paired CRN；主统计 paired bootstrap 10,000 次；seed 与既有 registry
冲突预检直接失败。

## 4. 文献映射

| 级 | 机制 | 依据 |
|---|---|---|
| S1 | 杠杆优先序 | CBCCO/OCC 顺序敏感性；杠杆结构事实 |
| S2 | 传播接续 | OCC（GECCO'24）conforming 胜因 |
| S3 | 共享坐标搜索 | FEA 竞争写回的严格-best 化；DFO 坐标搜索 |
| S4 | 逐变量信任域 | FT-DNPSO（TFS 2014）；trust-region 经典 |
| S5 | 软权重 + 分歧候选 | Soft Grouping（CEC 2019）；RB/ACADMM 强度思想 |

## 5. 与既往资产的边界

- 54a/54b 双支路、完整目标仲裁主路径、G_coupled 调度信号：维持冻结/负结果；
- shared-patch v2 内核（P0 已过）：其状态/receipt 基建在 S4 复用，
  但挂载点从"仅 CTP"改为"CTP/GSS 扫掠 + 仲裁路径双挂载"，
  且必须先过 S1–S3 才轮到 S4——不允许跳级；
- v5.x 调度器：不恢复。
