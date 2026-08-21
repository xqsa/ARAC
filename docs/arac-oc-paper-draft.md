# ARAC-OC 论文初稿（骨架 + 实质内容 v1）

日期：2026-08-22 凌晨
素材源：`arac-oc-paper-materials.md`（主张/证据映射）、
`arac-oc-related-work-gaps.md`（引用）、`arac-oc-gate54a-judgment.md`（F1）、
全部 artifacts/。目标刊物口径：演化计算方向（TEVC/TCYB/CEC）。

---

## 题目（候选）

1. Black-Box Overlap Discovery and Evidence-Dispatched Optimization for
   Large-Scale Problems with Overlapping Subcomponents
2. Knowing Which Variables Are Shared: Discovery, Dispatch, and the Limits
   of Overlap-Aware Coordination

## 摘要（草）

大规模黑盒优化中，子分量间的变量重叠（shared variables）既是结构信息也是
冲突来源。我们提出 ARAC-OC：一个三段式协议——(i) 带预算契约的黑盒重叠发现
（soft-RDDSM：召回 0.79–1.00、精度 1.00、180k FE、fail-closed）；
(ii) 由结构证据驱动的动作派发（两特征规则，对 oracle-上界方法 HCC-ES
24 case × 25 seed 20 胜 4 负、几何均值比 0.244）；(iii) 循环语境下的共享
变量完整目标仲裁（对 proposal-only 36/36 全胜）。我们进一步给出两条
机制边界：(a) 在线组合调度的跷跷板定律与信号不可调度性的系统刻画
（四代调度器、消融、oracle-gap、学习式排序器的天花板 ~1.6%）；
(b) **一致性的聚合目标不可辨识性**——在可加同基目标族上，owner 间
optima 一致与否在聚合剖面中精确同形，故运行时一致性分类需要分解内部
信道。全部主张以预注册 gate 与收据链背书。

## 1. Introduction

- 问题：overlapping LSGO（CEC'2010/'2013 谱系 + AOB-24）；
- 三个问题分层：发现 / 表示 / 值处理（调度作为可选第四层）；
- 贡献清单（对应 §5 实验）：
  C1 黑盒发现协议；C2 发现谱曲线；C3 证据派发；C4 循环语境仲裁；
  C5 组合收益存在性；C6 工程契约；F1 不可辨识性边界。

## 2. Related Work

- CC 资源分配：CCFR/CCFR3、DCCC（难度×贡献）、SACC；
- 重叠处理：FEA（factor 注入 + 全局适应度裁决——我们的仲裁先例）、
  CBCC-overlapping（贡献归属）、OCC（GECCO'24）；
- AOS：rank/相对归一 credit、extreme credit（strict-best archive 的
  lineage）、离线+在线混合（HF）；
- 学习路线：LCC/LH-CC/RLDO（展望）；
- **定位**：HCC-ES 从构造文件直读 oracle 结构（源码级证据，
  `docs/sota-oracle-confirmation.md`）；HCC 自留开放问题
  （"更智能的 exploration/exploitation 平衡"）由本文分析节回应。
  ARAC-OC 测的是**黑盒可达能力**。

## 3. Method

### 3.1 Phase-I：soft-RDDSM v3 发现（冻结）
signature → soft DSM → 块 + 超边证据 → OverlapCheckpoint
（owner 集 + 置信度 + incumbent，不可变）。FE 契约 180k 逐笔对账。

### 3.2 Phase-II：证据驱动派发（冻结）
两特征规则（tail_log10_gain × structural_relation_density）→
{ctp, smp, gcb, aor}。规则离线校准、零运行时调参。

### 3.3 循环语境的共享变量仲裁
候选 = incumbent / owner / weighted consensus / weighted median，
真实完整目标 tournament，strict-best 写回（FEA 式）；
counted probe 提供 B_j/W_j/C_j 证据（C_j 仅诊断）。

### 3.4 工程契约（贯穿）
exact FE / fail-closed / receipt+state hash 链 / archive 单调 /
预算车道互不侵占（两个 gate 失败案例作为契约自执行的展示）。

## 4. 识别性分析（F1，新结果）

命题（可加同基目标的聚合不可辨识）：见
`arac-oc-gate54a-judgment.md` §F1——共享变量 v 的条件剖面
Σ_{g∋v} w_g·b(v−o_{g,v}) 对 owner optima 的"一致性"不可辨识
（sphere 基下两类精确同形）。三次仪器 pilot 的否定证据链
（cross-context / scale-instability / owner-calibration）作为
经验佐证。推论：冲突处理若要黑盒化，必须引入分解内部信道
（owner 提案源、或 per-owner 条件化轨迹的差分结构）。

## 5. Experiments（全部映射到冻结 artifact）

| 实验 | artifact | 关键数字 |
|---|---|---|
| 发现质量 | soft_rddsm_aob_baseline_v3 | 召回 0.789–1.000 / 精度 1.000 / ov0 FP=0 |
| 发现谱 | oc_gate53_arbitration_wire/cells | shared 0/18/51/80/119/130 ↔ ov 0..10，跨 seed 稳定 |
| 派发 vs HCC | overlap_action_dispatch_gate41_online | 20/4，几何均值 0.244，E1→8e-6 |
| 仲裁（循环语境） | proposal_neighborhood_gate27 等 | 36/36，中位增益 37.9514 |
| 组合存在性 | oc_action_episode_gate50c | R6 0.870× 严格胜 4 standalone |
| 调度消融 | oc_mechanism_ablation_screening | v4 12/12 审计稳 vs v5.3 6/12；S5 5× vs 73× |
| 双轴象限 | oc_two_axis_replay_gate | S5-CTP protect 3/3；R2/A3 不稳定 |
| 信号链 | oc_lagged_coupling_normalized_gate 等 | 阴性链 + G8/G9 突破（0.383/0.433） |
| 仲裁移植边界 | oc_gate53_arbitration_wire | 24/24 契约过、零接受 |
| sense 预算 | oc_sense_overhead_ablation | 30% vs 45%：S5 −42% / R6 +50% |

## 6. Analysis：在线组合调度的边界知识

1. 跷跷板定律（v4→v5.3 收据级刻画：单轴规则表达不了双轴状态）；
2. 象限稳定性 ↔ 调度价值（S5 稳定可赢、R2/A3 不稳定谁都赢不了）；
3. 信号不可调度链：C_j / G_coupled / G_int / lagged EMA 全阴性
  （尺度归一后仍阴性）；oracle-gap 仅 ~1.6%，但简单事前特征可提取
  ~38%（G8/G9，hold-out 超过 production 与 best-fixed）→ challenger
  lane 是唯一有证据的学习入口；
4. 预算的结构性事实：sense 的减税流向 tail 而非算子（operator 恒 8–16 FE）。

## 7. Honest Boundaries（必须保留的边界段）

dense-overlap Phase-I 曾 fail-closed；仲裁仅在提案源存在的循环语境有效
（Gate 53 零接受 + F1 不可辨识性）；在线调度不宣称优越（存在性 R6 除外）；
两测试集相对口径；conforming 收益的 anytime 预期（Blanchard 先例）；
G8/G9 天花板 ~1.6%。

## 8. Conclusion & Future Work

承重墙 = 发现、派发、（语境化的）仲裁。后续：Gate 53b（提案源接入派发
路径）、双轴 + CCFR3 调度线（Gate 52 预注册已冻结）、G8/G9 晋级 gate。

---

### 写作待办（明晚继续）

- [ ] 图 1：架构图（三段式 + 契约层）；
- [ ] 图 2：发现谱曲线（shared vs ov，含 4-seed 误差带）；
- [ ] 图 3：Gate 41b 对 HCC 的 per-case 比值图；
- [ ] 图 4：双轴象限热图 + 跷跷板时间线（v4→v5.3）；
- [ ] 表 1：主张-证据-编号对照（直接扩自 §5）；
- [ ] F1 命题的正式化（附录：构造性证明）；
- [ ] 数字逐一对回 artifact（写作期禁止新跑实验）。

## 附录 A：命题 F1 的构造性表述（草）

**命题**（聚合不可辨识性）. 设 b: R→R 为任意基函数，w_g > 0，共享变量 v
属于组集合 O(v)。定义两族实例：一致型（∀g∈O(v): o_g = o*）与冲突型
（o_g 不全相等）。则对任意上下文 c（其余变量取值固定），两族中存在参数
选择使 v 的条件剖面 f(v|c) 逐点相等；特别地当 b 为二次时，冲突型的
条件剖面恒为 (Σw_g)(v−m)² + K（m 为加权均值、K 为常数），与一致型
（(Σw_g)(v−o*)²）在 m = o* 时**精确相同**。

**证明思路**. f(v|c) − f(v'|c) 只依赖 {w_g b(v−o_g)}_{g∈O(v)} 的和；
取冲突型参数使加权二阶展开与一致型重合（二次情形直接配方）。∎

**推论**. 任何仅查询聚合目标 f 的（随机化）算法无法以高于 1/2 的优势
区分上述两族；一致性信息存在于分解参数 {o_g} 中，其黑盒 manifestation
需要分解内部信道（owner 条件化轨迹的差分、或提案源）。

**经验佐证**（三次声明的仪器 pilot，`artifacts/oc_gate54a_pilot/`）：
cross-context 敏感度（可分目标按构造为零、方向反向）、尺度不稳定性
（被 incumbent 位置效应淹没）、owner 校准（同一限制函数 ⇒ 端点恒同）。
