# ARAC-OC 双重叠升级设计：一致性分类 + 两类支路

日期：2026-08-22
状态：**设计提案，待用户裁决后预注册**。本文档是 `arac-oc-completion-plan.md`
主线（发现 + 派发 + strict-best 底座，无调度器）上的升级，不恢复任何
在线 episode 调度机制。
上游证据：Gate 53 判定（仲裁在派发路径零接受）、vendor 代码事实
（AOB 四底函数全走 `rotateVectorConform`，conflict 路径未被使用）、
文献对照（CBCCO / OCC / RDG3 / Blanchard ORDG / HCC）。

---

## 1. 设计公理

1. **一致性决定机制**：重叠问题分 conforming（共享变量各 owner 最优值
   相同）与 conflicting（最优值不同）。两类需要的处理**相反**——
   一致型要"利用杠杆"，冲突型要"裁决归属"。用错方向就是负优化
   （Gate 53 的仲裁零接受发生在全 conforming 的 AOB 上，符合文献预测）。
2. **无调度器**：本升级不含任何在线 episode/预算调度决策。全部机制是
   派发动作**内部**的固定协议，由结构证据条件触发。
3. **零税条款**：无重叠（ov0）或全部 conforming 时，机制必须静默，
   开销有界且可忽略（沿用 Gate 53 已验证的 S1 条款形式）。
4. **底座不动**：strict-best archive、exact FE、fail-closed、receipt/hash
   链全部保留；每次写回都过 archive。

## 2. 总体结构

```text
Phase-I（不动）：soft-RDDSM v3 → 共享变量 + owner 集合 + 置信度
        ↓
证据驱动派发（不动）：Gate 41a 规则选动作
        ↓
┌─ 动作执行内部的重叠协议（本升级）─────────────────────┐
│ L1 一致性分类：有界 counted probe（复用 B_j/W_j 仪器）  │
│    逐共享变量判 conforming / conflicting + 置信度        │
│        ↓                                              │
│ L2a conforming 支路：杠杆优先（leverage-priority）       │
│ L2b conflicting 支路：贡献所有权 + 动态复核              │
│    （CBCCO 式归属，运行中有界复核）                      │
└──────────────────────────────────────────────────────┘
        ↓
strict-best archive 写回（不动）
```

## 3. L1：一致性分类器

**判据**：对共享变量 j，在其各 owner 子函数的上下文里做方向性探测
（现有 counted probe 原语，2 FE/变量/上下文）：

- 各 owner 对 j 的局部最优方向**一致**（B_j 同号或响应可忽略）
  → conforming；
- 方向**冲突**（B_j 异号且幅值超噪声地板）→ conflicting。

**协议要点**：

- 探测只在动作启动后早期的有限上下文点进行，每变量预算有界
  （建议 ≤ 8 FE/变量，预注册）；总量 = O(|shared| × owners)，
  相对 3M 预算可忽略；
- 输出逐变量标签 + 置信度入 receipt；
- **低频复核**（可选，预注册开关）：每 K 个窗口重判一次，
  处理"一致性随上下文漂移"的情况；默认关，先证静态版。

**已知的正确性锚点**（Gate 54a 的判定材料）：

- 自有函数发生器的 conforming/conflicting 实例真值直接可取
  （`OverlapObjective` 的 optima/groups）→ 分类器先在真值已知的
  实例上校准；
- AOB 原版应全部判 conforming（vendor 代码事实的反推验证）；
- 发生器 conflicting 模式应全部判 conflicting。

## 4. 两条支路

### L2a conforming 支路：杠杆优先

**机理**：一致型共享变量是**杠杆点**——改善它同时改善多个 owner 子函数，
单位 FE 收益是私有变量的倍数。文献佐证：HCC 观察到 CC 在重叠问题早期
高方差（共享值被反复拉扯）；OCC 在 conforming 函数上靠"改善传播"
获胜；Blanchard 证明此时无需值裁决。

**机制**（作用于块扫掠型动作 CTP/GSS；SMP/AOR 为全局动作不变）：

1. **扫掠序杠杆加权**：现有 block_scores 排序中给含共享变量的 scope
   加杠杆权重 ∝ Σ_owners |B_j|（重叠 scope 先修）；
2. **早期脉冲份额**：运行早期给重叠 scope 一个有界的优先份额
   （预注册比例，如首 20% 预算窗口内重叠 scope 优先）；
3. **传播规则（OCC 式）**：某 scope 产生 material 改善后，与其共享变量
   的相邻 scope 获得下一个扫掠槽位——让改善后的共享值传播进邻居
   上下文，而不是按固定顺序轮转。

**预期收益形式（诚实设定）**：一致型上的收益主要是**早期效率**
（anytime 曲线下移 / AUC 改善），终值差距可能小——Blanchard 的阴性
结果提示终值上"不处理也差不太多"。Gate 判据必须含 anytime 指标，
否则会把成功误判为失败。

### L2b conflicting 支路：贡献所有权 + 动态复核

**机理**：冲突型共享变量在各 owner 处最优值不同，必须裁决归属。
CBCCO 已证明"归贡献最大者"优于随机归属；FEA 式竞争在有限预算下
不如串行（CBCCO 论文、Gate 53 双重佐证——故不恢复仲裁为主路径）。

**机制**：

1. **校准窗**：动作启动早期，每个 owner scope 各跑一个有界校准窗
   （CBCCO 用 N=100 代；按本架构折算为预注册 FE 额），测每个 owner
   子问题的**贡献**（全局 archive 差，沿用 materiality 口径）；
2. **归属**：每个 conflicting 共享变量归贡献最大的 owner；
   归属结果写入 scoped checkpoint——非 owner 的 scope 不再修改该变量
   （结构性排除冲突写）；
3. **动态复核（相对 CBCCO 的增量）**：归属不终身锁定。当贡献排名
   发生翻转且持续 m 窗（预注册），归属迁移一次；复核次数有界、
   全部入 receipt。**这是与 CBCCO 的差异化点**，也是审计设计
   的重点（迁移必须可追溯）；
4. 归属后的执行仍是原动作，无新增调度决策。

**边界**：复核不是调度器——它不选动作、不定预算，只在贡献证据
翻转时迁移归属。如复核被证明无收益（Gate 54b 消融臂），退化为
一次性归属（= CBCCO 语义），仍然成立。

## 5. 基准分工：AOB 管一致型，自有函数发生器管冲突型

- **一致型测试集 = AOB 原版**（F1–F6，重叠度 0/1/3/5/7/10；vendor 四
  底函数全走 `rotateVectorConform`，全为 conforming）。它同时是
  对 HCC 的对比场，不动。
- **冲突型测试集 = 自有函数发生器**
  （`src/arac/benchmarks/overlap_objective.py` + `overlap_groups.py`）：
  - `conflict_mode="conflicting"`：每组独立采样自身最优向量，
    共享变量被各 owner 拉向不同值——冲突的构造定义；
  - 旋钮齐全：拓扑（random/chain/star）、重叠预算、组数、底函数
    （sphere/ackley/elliptic/rastrigin/schwefel）、旋转、变换、
    可选四次耦合项 `interaction_strength`；
  - **自带真值与诊断**：groups/optima/weights 可取，
    `per_group_contribution` 支持冲突诊断——这是它比任何外部
    冲突基准都强的地方；
  - 变换链与 AOB/CEC'2013 谱系逐位一致，conforming 模式可复现
    AOB 数值面——两套基准之间的可比性由构造保证；
  - 已在 Gate 37–40 的 conflicting 配置中实战过，收据与真值
    基础设施现成。
- **评估口径注意**：conflicting 实例的零下界一般不可达（最优是
  加权妥协），绝对误差含不可约冲突项——判定一律用**相对口径**
  （臂间 ratio / anytime AUC），不用绝对误差；`optimum_point()`
  只作诊断。

二维基准设计：

```text
              overlap degree 0/1/3/5/7/10（或发生器等效梯度）
             ┌──────────────┬──────────────┐
 conforming  │ AOB 原版      │  预测：L2a    │
             │              │  anytime 收益 │
             ├──────────────┼──────────────┤
 conflicting │ 自有发生器     │  预测：L2b    │
             │ conflict 模式 │  终值收益     │
             └──────────────┴──────────────┘
 ov0 整列：零税条款（机制静默）
```

## 6. Gate 序列（预注册骨架）

**Gate 54a：分类器正确性**（小预算，判定先行）
- 自有发生器 conforming 实例（预测全 conforming）与 conflicting 实例
  （预测全 conflicting），真值由 `OverlapObjective` 直接供给；
  外加 AOB 原版（预测全 conforming，vendor 代码事实的反推验证）；
- 通过标准：三类来源的标签与真值/构造一致 ≥ 预注册比例；探测 FE 在界内。

**Gate 54b：二维性能网格**（主判定）
- 臂：派发+双支路协议 vs 裸派发 vs 两个消融臂
  （无条件全归属 / 全不处理——证明**条件化**本身的价值）；
- 判定（全部同时，预注册后不调参）：
  1. ov0 零税（沿用 S1 形式）；
  2. conforming 列：anytime AUC 改善（预注册指标与显著性）；
  3. conflicting 列：终值严格改善（至少 1 处）；
  4. 条件化臂 ≥ 两个消融臂（证明分类器不是摆设）；
  5. 契约审计全绿（exact FE / strict-best / receipt 链）。
- 失败语义：按 case 关闭对应支路，如实记录；不允许 gate 内调参。

## 7. 与既有资产的映射

| 已有资产 | 在本设计中的角色 |
|---|---|
| soft-RDDSM v3 发现 | 不动，供给共享变量与 owner |
| Gate 41a 派发规则 | 不动，选动作 |
| counted probe B_j/W_j | 复用为 L1 分类器仪器 |
| block_scores / scope 排序 | L2a 杠杆加权的挂载点 |
| materiality / 全局 archive 口径 | L2b 贡献测量的口径 |
| strict-best / exact FE / fail-closed | 不动，全部写回的守卫 |
| 完整目标仲裁 | 降级为可选候选源（非主路径），不进本设计 |
| v4–v5.3 调度机制（已删，tag 中） | 不恢复 |

## 8. 风险与诚实边界

1. **conforming 支路收益可能很小**（Blanchard 阴性结果）——所以
   判据走 anytime 指标，并在论文里把"一致型的正确动作是轻触"
   作为发现陈述，而不是硬凑终值差；
2. **一致性可能随上下文漂移**——L1 默认静态判定，复核开关留作
   消融；若漂移普遍，54a 会暴露；
3. **两测试集的尺度对齐**：AOB（1000-D、20 组、vendor 数据文件）与
   自有发生器实例的维度/组数/底函数需配置对齐（发生器支持
   `num_groups=20` 与重叠预算梯度），否则"二维曲面"的行列不可比；
   conflicting 列的绝对误差含不可约冲突项，跨行比较只用相对口径；
4. **L2b 复核与调度器的边界**——复核只管归属迁移，引入任何
   动作/预算决策即视为越界，按纪律回退；
5. 文献机制均为机制级借鉴（CBCCO/OCC 的实验在 DG2 已知结构、
   CEC2013 上），数字不可横比。

## 9. 一句话定位

> 文献分别摸到了重叠的两半——CBCCO 的归属（冲突型）与 OCC/Blanchard
> 的传播与阴性结果（一致型）——但没有人把"运行时识别一致性"作为
> 机制的触发条件。本升级的核心主张：**先分类，再对症；一致型用杠杆，
> 冲突型用归属；无冲突不动作**。Gate 53 的零接受从失败变为该主张的
> 第一个证据。
