# ARAC-OC 阶梯式升级方案 v2.2（修订协议）

日期：2026-08-23  
基线：`arac-recovered-baseline-20260823-v1`  
状态：**修订协议，待用户裁决后冻结**；冻结前不得运行任何新实验  
上游：v2.1（`arac-oc-stepwise-upgrade-plan-v2.1.md`）+ U0/U1/S1 执行结果  
（`artifacts/upgrade_u0_baseline_guard_v1`、`artifacts/upgrade_u1_host_reachability_v1`、
`artifacts/upgrade_s1_leverage_sweep_v1`）

---

## 0. 修订动因（对预注册止损线的正式修订，非绕行）

v2.1 §9 预注册止损线"S1/S2 失败：关闭重排，不接入 patch"已于 S1 gate
判定后触发。本修订按该止损线的**立法意图**处理：止损线的目的是
"防止在基质不健康时向上堆风险机制"。S1 的执行把基质缺陷精确诊断出来了，
诊断结果改变了后续级别的设计前提，因此以显式修订协议重新进入阶梯，
而不是静默绕过止损线。

S1 的三个结构发现（全部有 receipt 级证据）：

1. **CTP 宿主会话惰性**：S2-S6 全部 30 对 ratio 精确 1.0000——coverage
   交错段的块会话一次性创建、从不重锚。**推论：任何依赖"重访同一 scope"
   的机制（顺序、接续、状态累积）在 CTP 宿主上原理性不可表达。**
2. **generator v2 退化**：6 cell 全部发现为 8 块完全图（leverage 全等 →
   平局 → 恒等重排）。**推论：conflicting 列本次未构成对 S1 的有效测试；
   任何依赖 leverage 异质性的级别都需要稀疏拓扑 generator。**
3. **GCB lane 顺序↔早停耦合**：R2-R6 route 全部改变（顺序移动早停时点），
   pooled geo≈0.998 但 R3=1.056、R5=1.054 破非劣界、CI 上界至 1.26。
   **推论：顺序类杠杆通过早停伪影通道起作用，不是机制通道。**

结论（"排序非杠杆"三重证据闭环）：在本宿主族上，扫掠顺序作为创新杠杆
被三重独立证据证伪（不可表达 / 退化 / 噪声放大）。

## 1. 关闭记录（永久，非暂定）

- **S1（杠杆优先扫掠序）：关闭。** 依据：上述三重发现。
- **S2（传播接续）：关闭。** 依据：发现 1——接续需要"下一个槽位"语义，
  CTP 宿主不存在；其传播意图由 S3a 以结构方式实现（见 §3 机理 3）。
- 两级关闭记录进入 progress log 与论文分析节素材；不删除任何 artifacts。

## 2. 机制类型与宿主表达性规则（新增，贯穿后续所有级别）

任何机制挂载前必须声明其**表达性类型**，并按类型匹配宿主：

| 类型 | 定义 | 可挂载宿主 |
|---|---|---|
| 候选级 | 单次访问即可表达（生成候选→完整目标评价→strict-best） | 任何有 scope 访问的宿主（CTP + GCB） |
| 结构级 | 改变 scope/块的定义，单次会话即可表达 | 任何有 scope 访问的宿主 |
| 状态级 | 依赖跨访问持久状态生效 | 仅 U1/S1 诊断证明有重访的宿主（当前：GCB lane） |
| 顺序级 | 改变既有工作的排列 | **全部宿主禁用**（S1 三重证据） |

## 3. S3a：共享核 scope 重构（结构级，FE 中性）

### 3.1 机制定义

利用 Phase-I checkpoint 的共享变量身份与 owner 隶属（**不使用 confidence**），
把每个 overlap component 的扫掠结构从"owner scope 含共享变量"重构为：

```text
owner scope g  →  只含 g 的私有变量（原 scope 减去全部共享变量）
核 scope c     →  该 component 的全体共享变量，联合上下文扫掠
```

每轮扫掠周期内，每个变量**恰好被访问一次**（共享变量从 |M(j)| 次降为
核 scope 中的 1 次）；节省的 FE 明确返还该动作自身 continuation，
receipt 记录 `fe_saved_by_dedup` 与去向。

所有写回照旧过完整目标 + strict-best；候选生成规则不变（用宿主动作
自身的扫掠原语，不新增候选族）。

### 3.2 机理（预期收益的三个通道，写入论文假设）

1. **拉据消除**：共享值不再被各 owner 在各自上下文轮流覆写——结构性
   关闭 HCC 观察到的重叠早期高方差通道；
2. **杠杆直达**：核 scope 联合上下文中移动共享变量，一次改善对全部
   owner 同时生效（conforming 上文献唯一证实过的收益形式）；
3. **传播结构化**：核 scope 改善共享值后，后续 owner 会话的上下文自然
   携带新值——无需任何接续规则。

### 3.3 构造契约

```text
输入：Phase-I checkpoint（groups, shared_variables, owner_membership）
输出：每 component 一个 scope 列表 = [私有 scope 集] + [核 scope]

核 scope 尺寸上界（预注册）：
  |核 scope| > K_core 时，按 owner 数降序拆分为多个子核 scope，
  每个子核 ≤ K_core；K_core = 50（运行前冻结，冻结后不调）

确定性：
  scope 内坐标按 coordinate_id 升序；
  component 按最小 group id 升序；
  拆分平局按 coordinate_id 字典序

ov0 / 无共享变量的 component：
  scope 列表与冻结 baseline 逐位一致（零税由构造保证）
```

### 3.4 FE 中性对账（Gate 硬条款）

- 每轮扫掠周期内，逐变量访问计数表入 receipt：`visits_j == 1` 对全部
  j 成立（私有 1 次、共享 1 次）；
- 周期总 FE ≤ baseline 同段总 FE；差额入 `fe_saved_by_dedup` 且去向
  只能是本动作 continuation；
- 不允许从 sense/probe/tail 或其他动作借预算（沿用 v2.1 §4-3）。

### 3.5 Gate S3a

对照臂：A0 = 冻结 baseline（patch-off）。

- **ov0**：route、FE 分类、terminal FE、receipt hash、final error 逐位一致；
- **AOB/conforming（非劣 + 方差指标）**：
  - paired geometric mean R ≤ 1.05 且 95% paired bootstrap CI 上界 ≤ 1.05
    （沿用 v2.1 §4.2 零误差守卫与 log 差值口径）；
  - anytime AUC paired CI 下界不低于 −0.05 相对 margin；
  - **早期方差指标（预注册）**：best-so-far 曲线前 20% FE 窗口的跨 seed
    方差，paired 比较不显著高于 baseline（机理 1 的可观测签名；
    显著降低是支持性证据，不显著不构成失败——方差指标只做非劣）；
- **conflicting（sparse generator v3，观察级）**：记录 acceptance 与
  相对口径收益，不作 superiority 判定（判定留给 S3b）；
- **reachability**：scope 访问 trace 非空；核 scope 出现且尺寸契约满足；
- 失败语义：关闭 S3a，回退 A0；不允许调 K_core 或任何阈值。

### 3.6 与 S1 死因的免疫声明（写入协议）

- 不改变任何顺序（顺序级禁用条款不适用）；
- 不要求重访（结构级，单次会话可表达，CTP 惰性不影响）；
- ov0 构造静默；
- 依赖 leverage 异质性 × —— 但依赖**共享变量非空**：
  完全图退化（发现 2）下核 scope = 全体变量，机制退化为全局扫掠，
  因此 generator v3 稀疏拓扑是前置依赖（见 §5）。

## 4. 重锚消融链（S1/S2 关闭后的归因结构）

```text
A0    = 冻结 baseline
A3a   = A0 + 共享核 scope 重构          （结构级）
A3b   = A3a + 共享坐标微补丁（v2.1 S3）  （候选级，挂在核 scope 上）
A4    = A3b + 持久信任域状态（v2.1 S4）  （状态级，仅 GCB lane）
A5    = A4 + disagreement 候选（v2.1 S5a，conflicting only）
A6    = A5 + soft owner weighting（v2.1 S5b）
```

归因链每次只动一列：

```text
A3a-A0 ：结构重构（去重+联合）
A3b-A3a：候选注入（注意：patch 候选空间随核 scope 改变，对照必须锚在 A3a 上）
A4-A3b ：持久状态
A5-A4  ：disagreement 候选
A6-A5  ：soft owner weight
```

v2.1 §4.2 判据、§4.3 lane 预算纪律（5% 总量上界 + rej≥4 退避）对
S3b 及以后各级原样适用。S3a 无 lane（无新增 FE），lane 纪律从 S3b 起生效。

## 5. generator v3：稀疏拓扑规约（前置工作项，非性能级别）

- 拓扑族：**chain / pairs 优先**；hub 允许但度数有界（预注册 ≤ 3）；
  禁止完全图；
- preflight 硬判据（不满足则 preflight 失败，不静默继续）：
  1. Phase-I 发现的 leverage（= scope 内共享变量计数）**方差 > 0**；
  2. 每 cell 发现关系数 > 0 且核 scope 非空、尺寸 ≤ K_core；
  3. 构造真值（groups/optima/per_group_contribution）可取并入 receipt；
- 6-cell 矩阵（chain/pairs × mild/strong + 2 个补充 cell）用 v3 重建，
  重新通过 Phase-I 发现后方可进入 S3a 的 conflicting 观察列；
- generator v3 是测试仪器修复，不构成算法性能主张；其变更记录单独归档。

## 6. 执行顺序（修订后）

```text
G-gen   generator v3 构建 + preflight（leverage 方差 > 0 等）
  -> S3a 契约门（FE 中性对账 + 确定性 + ov0 逐位）
  -> S3a screen（AOB 24 × seeds 117/123/129/135/141 + conflicting v3 6 cells）
  -> S3a fresh-seed confirmation（≥10 paired seeds，冲突预检）
  -> S3b 契约门 + screen（判据沿用 v2.1）
  -> S4/S5a/S5b 按 v2.1 阶梯
  -> production E2E（全部前置通过后）
```

每级独立 manifest / raw receipts / summary / protocol hash / result hash。

## 7. 止损线（修订版，替换 v2.1 §9 对应条目）

- G-gen preflight 失败：停止 conflicting 列，S3a 仅凭 AOB/ov0 判定
  （conflicting 缺口如实记录为测试仪器限制）；
- S3a 失败（含 FE 中性对账失败）：关闭结构重构，回退 A0，阶梯终止于
  负结果（S1/S2/S3a 全部关闭 → 论文按边界结果形态撰写）；
- S3b 失败：保留 S3a，放弃候选注入；
- S4/S5a/S5b 失败语义同 v2.1；
- AOB preservation 任何一级失败：该级默认关闭；
- 不允许通过修改 Phase-I、selector、冻结源或预算语义挽救任何一级。

## 8. 不变得的部分（显式重申）

v2.1 的冻结边界（§3）、通用 gate 契约（§4）、统计口径（§4.2 含零误差
守卫）、lane 预算纪律（§4.3）、S4/S5a/S5b 机制契约、论文表述边界（§8
可以声称/不能声称）全部原样有效。本文件只变更：S1/S2 状态、阶梯结构
（插入 S3a）、消融链锚点、generator 版本、执行顺序。
