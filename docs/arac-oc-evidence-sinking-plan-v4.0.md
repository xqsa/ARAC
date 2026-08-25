# ARAC-OC 升级方案 v4.0：证据下沉（Evidence Sinking）

日期：2026-08-23
状态：提案，待用户裁决
前序文档：`arac-oc-scst-upgrade-protocol-v3.0.md`（conflicting 侧事务层，保留）、
`arac-oc-stepwise-upgrade-plan-v2.2.md`（S1/S2 已按止损线关闭）
基线：tag `arac-recovered-baseline-20260823-v1`（冻结，不动）

---

## 0. 本方案回答的三个问题

1. 历史升级里哪里**真的有收益**，收益由什么导致？（§1 数据挖掘）
2. Phase-I 证据有哪些**已发现但没被 Phase-2 使用**的通道？（§2 通道盘点）
3. 基于 1+2，升级阶梯怎么设计？（§3-§5）

本方案与之前所有失败升级的根本区别：**不新增任何在线反馈机制，只做"证据下沉"——
把 Phase-I 已经发现（或可从已有收据离线重算）的静态结构证据，沿契约下沉到四个
动作的预算分配与生命周期结构层。**

---

## 1. 历史收益目录与因果归因

### 1.1 有真实收益的升级（按收益幅度排序）

| # | 来源 | 收益 | 因果机制（收据级证据） | 机制类别 |
|---|---|---|---|---|
| G1 | SMP lifecycle repair（2026-08-23） | E2-E6 geo ratio **1.789 / 4.559 / 8.687 / 11.756 / 10.663**，24/25 pair 胜 | 恢复 rescue + ~1.41M MMES global polish，消除 no-op 尾部 | **预算所有权** |
| G2 | v4.3 S5 突破 | 43× → **5.05×**（66,820），ON/OFF=0.043 | CTP 票据信用引导后获得 1.59M 连续长程打磨——正是它需要的预算尺度 | **预算所有权** |
| G3 | CTP S6 tail ablation | 3/3 candidate wins，geo **0.2897**（未配对，仅机制证据） | reserved tail 段的存在与否 | **预算所有权** |
| G4 | Gate 46 统一循环 | 24/24 不劣，**22 严格胜 + 2 平 + 0 负** | 省下内核强制 32 FE/组件 envelope，预算全给终端 MMES | **预算所有权** |
| G5 | E1 zero-relation preservation（2026-08-23） | E1 均值恢复至 **7.75e-06** | 零关系拓扑走 hybrid lifecycle，正关系走 historical-compatible——**按拓扑位路由生命周期** | **拓扑条件化** |
| G6 | v5.2 杠杆 2 horizon promotion | R6 **1.126→1.049** 双条款达标（唯一达标 case） | challenger 通道的发现获得进入 exploit 的显式通道（10 次 promotion、6 次 material，因果链全收据） | **发现→利用通道** |
| G7 | Gate 50c R6 | **×0.870** 严格胜全部四条 standalone | handoff 接力链 + 全局 materiality 过滤（ON<OFF 因果成立） | **发现→利用通道** |
| G8 | Gate 41a/b 证据派发 | 对 HCC-ES **20 胜 4 负**，geo mean **0.244** | tail_log10_gain + relation_density 两特征一次性选动作 | **证据驱动选择**（原始核心） |

### 1.2 失败或零收益的升级（同样重要）

| # | 来源 | 结果 | 失败的因果机制 |
|---|---|---|---|
| F1 | S1 leverage 重排（2026-08-23） | CTP 30/30 对精确 1.0000；GCB CI 上界 1.26 破非劣 | **顺序扰动**：CTP 块会话一次性创建不重锚，顺序原理性不可表达；GCB 顺序与早停耦合，无稳定收益方向 |
| F2 | Gate 47b rastrigin 层 | 0/6，终值差 +29~+170 | **微窗口干预**：20-FE 共享核补丁在 1000-D rastrigin 上价值≈0，锚点扰动方差 >> 窗口价值 |
| F3 | v4.0-v4.4 / v5.x 调度器迭代 | 跷跷板（S5 5.05×↔25.9×、R2/R6 互斥） | **反馈驱动预算重分配**：短窗 gain/FE 排序系统性误判晚熟动作 |
| F4 | sense 探针开销 | 曾吃 93.7% 预算 | 在线 sensing 的成本失控 |
| F5 | Gate 47 τ 校准 | `tau_gap_absent`，不定阈值 | counted C 幅值对"可修复性"无判别力——**可修复性由拓扑决定**（chain 可逐对修、star hub 修复无收益） |
| F6 | Gate 54a | 分析性证明 + 三次 pilot 失败 | conforming/conflicting 在聚合黑盒目标上不可辨识，硬分类不可行 |
| F7 | G_coupled / G_int / EMA 调度信号 | 全部未过阈 | 在线反馈信号作为调度 authority 的第三次独立失败 |

### 1.3 元规律（本方案的设计公理）

把 §1.1 与 §1.2 并排放，一条可证伪的元规律浮现：

> **公理 1（尺度公理）**：本项目全部真实收益都作用于**预算所有权 / 生命周期尺度**
> （≥100k FE 段的归属：rescue 有没有、polish 尾有多长、CTP 能不能连跑 1.59M）；
> 全部失败都作用于**微窗口（≤75k）或顺序扰动**尺度。
>
> **公理 2（信号源公理）**：全部真实收益由**静态结构证据**条件化（关系有无、拓扑类、
> checkpoint 特征——Phase-I 一次性产出、零运行时成本）；全部**在线反馈信号**
> （C 幅值、耦合 EMA、短窗速率）作为调度 authority 均失败或被证伪。
>
> **公理 3（通道公理）**：收益需要显式通道——发现的东西必须能进入利用
> （G6 promotion 是正例；50c 之前 challenger 锁死是反例）。Phase-I 证据亦然：
> 发现了但没路径进入动作内部，等于没发现。
>
> **公理 4（保守默认公理）**：机制不触发时必须逐位恒等（Gate 46 的"正确的保守"
> 22 胜正是省下预算的收益；ov0 控制是每级 gate 的硬条件）。

这条元规律同时解释了：为什么"回到第一版"后一切升级都失败——
v2.x/SCST 之前的候选全部违反公理 1 或 2（重排=顺序扰动、patch=微窗口、
仲裁=在线反馈）。**升级的正确方向不是更聪明的在线机制，而是证据下沉。**

---

## 2. Phase-I 证据通道盘点（代码事实）

### 2.1 三段数据流

**Phase-I v9 实际产出**（`evidence/phase1.py`，协议 `arac-identity-blind-evidence-v9`，180k FE）：

- landscape 探针特征（240 FE，10 + line 特征）
- `infer_structure`（counted interaction）→ `StructuralEvidence(blocks, relations[strength, disagreement])`
- incumbent + 进度特征（phase1_log10_improvement 等）

**checkpoint 持久化**（`runtime/contracts.py: PhaseCheckpoint`）：

- ✅ blocks（划分）、relations（块级，含 strength / disagreement 数值）
- ✅ incumbent、feature_names/values
- ❌ **逐变量共享身份不在契约里**：soft-RDDSM 分支能发现 `shared_candidates`
  （AOB 召回 0.908 / 精确率 1.0，2026-08-15 验证），但该发现分支当前
  **不在生产 Phase-I 数据流内**（`discover_hierarchical_soft` 仅被实验脚本调用，
  `phase1.py` 不引用）
- ❌ 拓扑摘要：`summarize_relation_topology` 计算度数集中度/熵/最大组件，
  但 `core.py:88` 只取第三个返回值（largest_component），**前两个被丢弃**

**四动作消费**（`actions/*.py` 逐行核实）：

| 证据字段 | dispatch | CTP | GCB | SMP | AOR |
|---|---|---|---|---|---|
| blocks | ✓ 完备性 | ✓ coverage/polish | ✓ 扫掠 | ✓ visits | — |
| relation **count** | ✓ 二值 | ✓ 二值（cover/tail） | ✓ 二值分支 | ✓ 二值（rescue） | — |
| relation **strength/disagreement 数值** | ✗ | **✗** | ✓ 仅初始静态排序一次 | **✗** | ✗ |
| 拓扑类（chain/star/hub、集中度、熵） | ✗（丢弃） | ✗ | ✗ | ✗ | ✗ |
| landscape/progress 特征 | ✓（41a 规则） | ✗ | ✗ | ✗ | ✓ |
| 逐变量共享身份 | — | ✗ | ✗ | ✗ | ✗ |

### 2.2 结论

Phase-I 证据在 Phase-2 的使用形态是"**一个计数器 + 一次静态排序**"。
三个已发现/可发现的维度——**关系数值、拓扑类、逐变量共享身份**——全部闲置。
这就是"Phase-I 证据没怎么用在 Phase-2"的精确含义，也是升级空间所在。

---

## 3. 升级阶梯：E0 → E4

设计规则（继承项目纪律）：

- 每级独立 gate，失败按预注册止损线关闭，不回调已判规则
- 每级第 0 步是**可表达性预检**（U1 式 reachability/identity），证明机制
  在该宿主可表达且仪器化零税，再跑机制
- 零关系臂（S1/R1/E1 对照）全程逐位恒等（公理 4）
- 全部机制只读 checkpoint 静态证据，**零在线反馈信号**（公理 2）
- 代码全部在 `experiments/upgrade/evidence_sinking_v1/`，不触碰冻结源；
  动作层修改走 registry/wrapper 层（SMP lifecycle repair 已有先例）

### E0 — 契约扩展：让证据先"存在"

**动机**：公理 3——证据必须先进入契约，动作才能合法消费。这是所有后续级的前提。

**内容**：

- E0a（可行性，零运行风险）：验证共享身份与拓扑摘要能否从**冻结 Phase-I 收据
  离线重算**（`Phase1Probe.fitness_values` 已持久化全部评价记录）。可行 → E0b；
  不可行（soft-RDDSM 探针需要特定扰动结构）→ 重新评估旁路探针的轨迹影响，
  并在协议中明示代价，**不允许静默改变 Phase-I 轨迹**。
- E0b：新协议版本（v10，与 v9 并存），PhaseCheckpoint 增加三个只读字段：
  `shared_candidates`（逐变量）、`relation_topology_summary`
  （度数集中度 / 熵 / 最大组件占比 / hub 度）、`per_block_leverage`。

**Gate**：扩展 checkpoint 的重跑与冻结 screen 收据**逐位一致**
（blocks/relations/incumbent/final_error/route 全同——扩展是只读附加）。
identity gate fail-closed。

**止损**：E0a 不可行且旁路探针影响轨迹 → 只做拓扑摘要（可从 relations 纯计算，
零风险），shared 身份降级为离线诊断，不阻断 E1-E3（它们不依赖逐变量身份）。

### E1 — CTP：tail reserve 的证据条件化（S 族）

**动机**：**全部历史资产中最大的单一未捕获收益**。G3：matched checkpoint tail
ablation 3/3 胜、geo 0.2897（≈3.45× 改善潜力），但 seeds 未配对；
进度日志已把"fresh matched S6 screen-seed tail attribution"列为下一 gate。
同时 Gate 41b 的 S6 失败（历史列 7.79×）部分源于 CTP 在 S6 的尾部不足。

**机制**：tail reserve 的大小由 checkpoint 证据条件化（当前是固定比例二值）。
候选自变量：relation strength 总量、拓扑类、tail_log10_gain。
**注意**：不改会话内行为（CTP 块会话不重锚是已证结构事实），只在会话创建前
决定预算分割——这在 wrapper 层可表达。

**Gate**：S6 screen seeds 全新配对（5 seed × {当前, 证据条件化}），
判据 = paired final-error 非劣 + geo ratio ≤0.98；S1-S5 非劣保护（CI 上界 <1.05）；
S1（零关系）逐位恒等。

**止损**：S6 geo ≥0.98 或任一 S2-S5 破非劣 → 关闭，tail 保持冻结值。
G3 的 0.2897 若配对后不兑现，如实记录为"未配对机制证据不可迁移"的边界发现。

### E2 — SMP：rescue/polish 预算的连续化（E 族）

**动机**：G1 证明 E 族是对生命周期/预算结构**最敏感**的宿主
（1.8×-11.8×！），而当前的证据使用是最粗的二值（relation_count>0 → rescue 有无）。
最敏感宿主 × 最粗证据粒度 = 最大期望升级空间。

**机制**：rescue 预算与 global-polish 预算从二值改为 checkpoint 证据的连续函数
（候选：relation 密度、strength 总量、拓扑集中度）。函数形式**预注册为单调有界
分段线性**，不允许 gate 内调参；零关系端点必须精确退化为当前冻结值
（保护 G5 的 E1 修复成果）。

**Gate**：E2-E6 × 5 seed paired；判据 = 逐 case 非劣 + ≥1 case geo ≤0.98；
E1 逐位恒等（硬条件）。

**止损**： pooled geo >1.0 或 E1 不恒等 → 关闭，回到二值。

### E3 — GCB：source window 预算的证据分配（R 族）

**动机**：R 族是 S1 中唯一"机制可激活、可观测"的 lane（route 全部改变），
只是顺序这个杠杆无稳定收益。**预注册禁令：E3 不动顺序**（S1 已杀死顺序）；
动的是每个块的 **source window 大小分配**（当前均分），按块级 leverage/
strength 加权。

**注意**：早停时点随预算分配移动，route 漂移必然发生（S1 已示）——
判据不罚 route 变化本身，只看 final/anytime。

**Gate**：R2-R6 × 5 seed paired；判据 = final 非劣（CI 上界 <1.05，逐 case）
+ pooled geo ≤1.0；R1 逐位恒等。

**止损**：任一 case CI 上界 ≥1.05 → 关闭。

### E4 — 拓扑修饰的动作内路由（最后，风险最高）

**动机**：F5 的设计层发现（可修复性由拓扑决定）+ Gate 47-R 在稀疏域已验证
"relative hub 选 restricted CTP vs shared-core CTP"这条路径可开火。
把拓扑类作为**动作内变体选择**的修饰信号（不改动作选择本身——dispatch 不动）。

**内容**：hub 度超阈 → CTP polish 用 restricted 变体；chain 拓扑 → 允许
shared-core 变体。**前置条件**：E0 的拓扑摘要已入契约且 E1 已通过
（E4 与 E1 共享 CTP 宿主，串行防混淆）。

**Gate / 止损**：同 E1 结构；失败即关闭，不影响 E1-E3 已冻结成果。

---

## 4. 与 SCST v3.0 的关系（显式分工，防双梯混淆)

| | 本方案（证据下沉） | SCST v3.0 |
|---|---|---|
| 作用面 | conforming 侧 + 预算/生命周期层 | conflicting 侧 + 边界事务层 |
| 证据基础 | 8 个历史收益点（G1-G8） | Gate 53 预测 + T1 待审计 |
| 风险 | 低（静态证据、wrapper 层、逐位回退） | 中（新机制、需 T1 先证边界存在） |
| 依赖 | E0 契约扩展 | T0/T1 的 proposal 审计 |

**执行顺序**：本方案先行。理由：(a) 风险更低、历史证据更直接；(b) **E0b 的
`shared_candidates` 入契约同时是 v3.0 T1 proposal 审计的前提**——proposal receipt
需要逐变量 coordinate_id，当前契约里没有。本方案为 v3.0 铺路而非取代。
若 E1-E3 全部失败，v3.0 仍是 conflicting 侧的正解，两梯互不阻塞。

---

## 5. 阶梯总表

| 级 | 宿主 | 杠杆 | 证据通道 | 主判据 | 止损 |
|---|---|---|---|---|---|
| E0 | 契约 | 只读扩展 | shared 身份 + 拓扑 + leverage | 逐位恒等 | 降级为拓扑-only |
| E1 | CTP/S 族 | tail reserve 大小 | strength 总量、拓扑、tail_log10_gain | S6 geo ≤0.98 + S2-S5 非劣 | tail 回冻结值 |
| E2 | SMP/E 族 | rescue/polish 预算连续化 | 密度、strength、集中度 | ≥1 case geo ≤0.98 + 逐 case 非劣 | 回二值 |
| E3 | GCB/R 族 | source window 预算分配 | 块级 leverage/strength | pooled geo ≤1.0 + 逐 case 非劣 | 回均分 |
| E4 | CTP 变体 | 拓扑修饰路由 | hub 度、拓扑类 | 同 E1 | 关闭变体 |

不变的硬条件（每级）：零关系臂逐位恒等；冻结校验器全绿；零在线反馈信号；
失败默认回退；不允许 gate 内调参。

---

## 6. 论文逻辑（这条线的叙事价值)

本方案即使部分失败，也产出完整的论文资产：

1. **元规律本身是贡献**：在 15 个独立 gate（8 正 + 7 负）上归纳出的
  "结构证据必须作用于预算所有权尺度；在线反馈作用于微窗口必然失败"
  是一条可证伪的设计原则，正面有 G1-G8 收据链，反面有 F1-F7 判定书。
2. **与 Gate 54a 拼成边界地图**：什么不可知（conforming/conflicting 硬分类）、
  什么已知但没用（证据通道盘点表）、用了之后发生什么（E1-E4 判定）。
3. **E0-E4 的阶梯判定**（无论成败）是"证据价值分层"的实证研究：
  计数器证据（已用）→ 数值证据（E2/E3）→ 拓扑证据（E1/E4）→ 身份证据（E0/v3.0），
  每一层一个 paired 判定，这本身就是"面向重叠问题的证据经济学"。

---

## 7. 待用户裁决的参数

| 参数 | 建议值 |
|---|---|
| E0a 离线重算的最大允许额外探针 FE | 0（优先纯收据重算） |
| E1 tail reserve 候选函数 | 冻结基线比例 × (1 + min(1, strength 总量归一化))，预注册后不改 |
| E2 预算连续化函数族 | 单调有界分段线性，端点精确等于冻结二值 |
| E3 加权指数 | source window ∝ block score^α，α∈{0(=冻结均分), 1} 两档预注册 |
| 每级 seed 数 | 5（与 screen 对齐） |
| 每级配对容差 | final 非劣 CI 上界 <1.05；严格胜 geo ≤0.98 |
