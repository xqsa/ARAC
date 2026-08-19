# Phase-I v10.2 设计：分层软置信度重叠证据发现

日期：2026-08-15（v10.2 修订）
状态：设计草案第三版；修复三项阻塞点，预注册三个前置 Gate；实现以
Gate 42/43/44 通过为前提
前置：`docs/arac-oc-design.md`、`docs/arac-oc-progress-log.md`

## 修订记录（v10.1 → v10.2）

1. **变量签名机制落实**：v10.1 的"签名零额外 FE"不成立——粗筛只有
   bucket 聚合响应。v10.2 引入可计费的变量级交互签名（固定上下文
   batch 池 + 每变量混合差分，§3.2）；
2. **三层证据模型**：`RegionRelation` / `VariableRegionInteraction` /
   `ResolvedOverlapHyperedge`，"区域交互 ≠ 变量成员"写入 schema 约束；
3. **HIERARCHICAL 独立接口层**：`RegionProposal` / `RegionConflictProbe`
   / `RegionCoordinator`，废除"复用现有协调器结构接口"的说法；
4. 条件探针多随机方向；
5. Jeffreys 下界明确统计单位；
6. `DENSE` 更名 `EVIDENCE_DENSE`（证据密集，非真实结构声明）；
7. `S_max/C_max/深度/组件配额`给出具体预注册整数。

## 1. 三层证据模型（核心 schema）

```text
RegionRelation
    (left_leaf, right_leaf, score, stability, depth)
    语义：区域 r 与区域 g 之间存在交互。仅此而已。

VariableRegionInteraction
    (variable, source_region, target_region, q_lb, support, sign_stability)
    语义：位于 source_region 的变量 j 与 target_region 发生条件交互。
    这不是"j 属于 target_region"——叶子互斥，成员语义非法。

ResolvedOverlapHyperedge
    (variables, regions, dual_evidence)
    语义：经过条件探针双重确认的真实重叠超边（变量 j 对 ≥2 个区域
    的交互证据同时成立且符号稳定）。只有此层可以转换为
    OverlapStructure 的 groups/memberships。
```

```text
Phase1Evidence（不可变）
├─ region_tree                    # 二分树，叶子互斥
├─ region_relations               # 第一层
├─ variable_region_interactions   # 第二层
├─ resolved_hyperedges            # 第三层（可为空）
├─ variable_status                # observed_separable | member_candidate | not_yet_resolved
├─ per_component_mode             # SPARSE | HIERARCHICAL | EVIDENCE_DENSE（按 component）
└─ level_budgets                  # 逐阶段 FE 收据
```

**schema 硬约束**：不存在"变量属于两个互斥叶子"的字段路径；
`OverlapStructure` 构造函数只接受 `resolved_hyperedges`，直接传入
区域叶子抛 `TypeError`。

## 2. 统计规格

### 2.1 条件探针（多方向）

对候选 (j, g)：K_dir = 3 个随机符号方向 Δ_g^(1..3)，A_cond = 2 个
anchor：

```text
I(j,g)^(a,k) = f(x_a + Δ_j + Δ_g^(k)) − f(x_a + Δ_j) − f(x_a + Δ_g^(k)) + f(x_a)
```

单侧项缓存（Δ_j 每变量一次；每方向 Δ_g 每区域一次）。

### 2.2 置信下界的统计单位

- **置信水平**：95%（Jeffreys 非参数下界）；
- **有效样本单位**：独立的 (anchor, 方向) 探针实例，A_cond × K_dir
  = 6；(j,g) 的 support = 实际完成的实例数；
- **相关性处理**：同一树路径上的粗筛证据只用于候选排序，不进入
  下界分子；anchor 视为独立块（不同起点）；
- **sign_stability 定义**：方向对齐后交互符号一致的实例比例（≥ 5/6
  才可成为 member_candidate 的必要条件）；
- anchor 间不一致（某 anchor 全不显著）→ 降级 not_yet_resolved，
  不合并进下界。

### 2.3 三态判定（按 component）

`EVIDENCE_DENSE` 命名规则：只声明"当前预算与探针下无法解析出稀疏
结构"，真实结构是否稠密由离线构造审计讨论，运行时不声明。

## 3. 发现流程与预算契约

### 3.1 流水线

```text
Stage A  粗筛（随机 hash 分区，32 区域）          → RegionRelation 候选
Stage A' 变量交互签名（固定 batch 池，可计费）     → 每变量签名向量
Stage B  签名谱排序（纯计算）                      → 变量新顺序
Stage C  递归细分（重排序上二分，S_max 内）        → region_tree
Stage D  条件探针（C_max 内，多方向）              → VariableRegionInteraction
                                                   → ResolvedOverlapHyperedge
Stage E  incumbent（下限保护）
```

### 3.2 变量签名的真实获取（替代 v10.1 的空步骤）

**机制（v10.2.1 修订，Gate 43 实现后冻结）**：P = 12 个固定随机探针
batch（每 batch 16 变量）构成**共享测量基**——全变量对同一基测量，
签名互相可比较（v10.2 草案的"每变量 hash 指派 4 batch"方案被否决：
不同基上的向量不可比）。签名：

```text
s_j = [ I(j, B_p) 归一化 ]_{p=1..P}，A_sig = 1 个 anchor，step 0.25
I(j,B) 的联合项与 batch 单侧项在 j ∈ B 时使用修正构造
（j 只扰动一次；batch 单侧用 f(x + Δ_{B∖{j}})），消除伪自交互。
```

**FE 契约**（精确公式，已由 Gate 43 逐 cell 验证）：

```text
FE_sig = 1 + d + P + d×P + P×probe_size = 1 + 1000 + 12 + 12000 + 192 = 13,205
```

每项对应 ledger 可审计评价；修正单侧恰好 P×probe_size 个（确定性）。

### 3.3 预注册整数（运行时禁止反解）

| 参数 | 值 |
|---|---|
| anchors A / 粗筛轮 L0 / 粗筛区域 R0 | 5 / 6 / 32 |
| 粗筛 FE 上界 | 5×6×529 = 15,870 |
| 签名 anchor 数 / 共享探针 P / 探针尺寸 | 1 / 12 / 16 |
| 签名 FE | 13,205 |
| 细分：每 split FE（anchor×轮×7） | 5×2×7 = 70 |
| **S_max（split 总数）** | **900** |
| 细分 FE 上界 | 63,000 |
| 条件探针方向 K_dir / anchors A_cond | 3 / 2 |
| **C_max（(j,g) 对总数）** | **1000** |
| 每 component 条件探针配额 | 250 |
| 最大树深度 / 最小区域尺寸 | 7 / 8 |
| edge_threshold（校准 §7.3 第一轮：噪声地板 ~1e-13，网格信号 ~1e-4，两侧各一个量级余量） | 1e-10 |
| 条件 FE 上界 | 1000×6×2.5 = 15,000 |
| incumbent 硬下限 | 50,000 |

最坏情况合计 = 15,870 + 13,205 + 63,000 + 15,000 = 107,075 ≤ 130k；
incumbent ≥ 72,925 ≥ 50k。terminal 精确 180k，逐级收据。
**Phase-I 质量门**：v10.2 incumbent 同 seed 配对不劣于 v9（容差预注册）。

## 4. HIERARCHICAL 独立接口层

```text
RegionStructure
    ↓ RegionProposal          # 区域级局部提案（新接口，不复用 produce_local_proposal）
    ↓ RegionConflictProbe      # 区域级 counted B/W/C（新接口）
    ↓ RegionCoordinator        # 区域 patch 调度 + GCB 脉冲
    ↓ 条件变量 refinement      # overlay 内追加，产出局部 ResolvedHyperedge
    ↓ 局部 OverlapStructure    # 仅由确认超边构造
    ↓ 变量级 CTP               # 现有机制
```

**可复用**：strict-best ledger、算子内部的候选评价、sequential patch
的搜索原语、收据/信封纪律。
**不可复用**：`OverlapStructure`/`produce_local_proposal`/`counted_probe`
的结构接口（它们假设共享变量与 owner 集合存在）。

## 5. EVIDENCE_DENSE 分支

区域图高密度或签名排序无增益的 component：区域冲突调度 + AOR，
或显式转 ARAC-Core 动作派发 fallback（`dispatch_policy.py`，命名与
声明严格分离）。DENSE 对比 gate 预注册：区域协调 ≥ fallback 的
21/3/0 才切换默认。

## 6. 三个前置 Gate（实现顺序与通过判据）

### Gate 42：证据语义 Gate（零 FE，property 测试）✅ 已通过（2026-08-15）

实现：`src/arac/evidence/hierarchical.py` +
`tests/test_hierarchical_evidence_gate42.py`（9/9 通过）。判据逐条对应：

1. 互斥叶子直接构造 `OverlapStructure` ⇒ `shared_variables = ∅`
   （25 组随机划分 property 验证）；`to_overlap_structure` 在无确认
   超边时 fail-closed 抛错；
2. schema 拒绝一切成员语义：source≠target、home-leaf 一致性、超边
   证据完整性（⊆ 已记录交互）、sign-stability ≥ 5/6、证据引用
   审计；
3. 超边转换审计：共享变量 = 超边变量、membership ≥2、组覆盖全体、
   `RegionStructure` 无任何转换路径。

1. 任意只含互斥叶子 + RegionRelation + VariableRegionInteraction 的
   `Phase1Evidence`，构造 `OverlapStructure` 必然失败或
   `shared_variables = ∅`——除非存在 `ResolvedOverlapHyperedge`；
2. schema 拒绝"变量属于两个互斥叶子"的任何字段路径；
3. 超边→`OverlapStructure` 转换的审计追踪（dual_evidence 完整性）。

### Gate 43：变量签名 Gate ✅ 已通过（2026-08-15）

实现：`src/arac/evidence/variable_signature.py` +
`tests/test_variable_signature_gate43.py`（6/6）+
`experiments/overlap_signature_gate43.py`（12 cell × 2 变体，1000-D）。

1. **可计费性**：FE = 1+d+P+d×P+P×probe_size，全部 24 次运行逐 cell
   与 ledger 相等；
2. **信号质量**：1000-D 稀疏网格上同组签名近邻命中提升 = 基线的
   30–90×（均值 50.6×）；
3. **置换不变性**：块内置换（活跃/哑变量分别置换，破坏索引连续性）
   后均值提升 36.0×，保持恒等情形的 71%。
   实现中修复两个真实缺陷：j∈自身探针 batch 时的双重扰动伪影、
   及其修正单侧的精确 FE（P×probe_size）。

### Gate 44：HIERARCHICAL 接口 Gate ✅ 已通过（2026-08-15）

实现：`src/arac/coordination/region.py` +
`tests/test_region_coordinator_gate44.py`（5/5 通过）。

- `produce_region_proposal`：区域坐标上的锚定优化（复用
  `_MirroredLedger` 私有归档 + 全局账本计费），预算精确；
- `region_conflict_probe`：候选变量两侧计数探针，每变量精确 2 FE，
  f(x0) 复用，B/W/C 公式与 `counted_probe` 一致；
- `RegionCoordinator.run_cycle`：组件选择 → 区域提案 → 冲突探针 →
  贪心 strict-best patch，逐级 FE 收据且总和与 ledger 增量相等；
- **全路径不构造 `OverlapStructure`**（monkeypatch 断言验证）；
- 唯一的变量级桥梁仍是 `to_overlap_structure` 的超边门（Gate 42）。

## 7. 后续序列（三 Gate 通过后）

可识别性审计（重叠度量族 + 三层真实区分）→ 退化性语义等价测试 →
阈值校准 → 预算契约测试 → HIERARCHICAL 协调 gate → EVIDENCE_DENSE
对比 gate → 完整 AOB 正式实验（25 seed × 24 case）。

## 7.5 v10.3 双侧证据门（2026-08-16 实施）

用户纠错后的核心修正：hyperedge 声明必须通过**三元组双侧证据**——

```text
j 与目标侧成员 t 单对耦合（t 逐成员测试选出，杜绝叶污染）
j 在其序邻域（半径 8 扫描）内与某 h 耦合（≥1 anchor 触发）
h 与 t 在全部 anchor 上可分（同组弱对不得假静默）
⇒ j 跨越两个可分领地 ⇒ 真共享变量
```

预注册：`neighbour_scan=8`、`two_sided_budget=30k`。分层审计
（R1/R2/R4/R6 = 重叠 0/1/5/10）：

| case | hyperedge | 命中 | FP | 召回 | 精确率 |
|---|---|---|---|---|---|
| R1 | 1 | 0 | 1 | — | 0 |
| R2 | 34 | 0 | 34 | 0.00 | 0.00（异常，待查） |
| R4 | 10 | 5 | 5 | 0.05 | 0.50 |
| R6 | 15 | 13 | 2 | 0.07 | 0.87 |

对比 v10.2（无门）：R1 98 全假阳性 → 现在 1。**无重叠函数不再产生
重叠声明**（用户要求达成）。

已知边界（开放项）：
- R2 全假阳性异常（机制未明，怀疑低重叠下 t/h 选择的测试误差）；
- 召回上限由靶向覆盖（~0.43）与转换率（~0.27）共同压住——瓶颈在
  签名排序质量与边界候选定位，属 Stage B 结构改进而非阈值问题；
- 逐 anchor 方向重采样可再提特异度但预算超支（已试，回退）。

## 8. 声明边界

同 v10.1，另加：签名与排序是"证据驱动的启发式定位"，其质量由
Gate 43 量化声明，不宣称最优；所有探针 FE 由统一 ledger 计费，
任何"零成本推断"都是设计错误（v10.1 的教训已写入修订记录）。
