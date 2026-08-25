# ARAC-OC 推进日志

用途：夜间自主推进的连续性锚点 + 用户晨读摘要。所有 gate 结论以对应
protocol 文档和 artifacts 为准，本文件只做索引和决策记录。

## 2026-08-22 Recovery-first 状态

- P1 实质判定：**不可评估**。12/12 cell 为 `arbitration_only`，patch
  receipt=0，A0-A4 逐位相同；旧 P2/P4 停止。
- **B0**：recovery-first preflight 已通过（4 case × 1 seed 的只读
  checkpoint/current-receipt provenance）。
- **B1**：完整 24×25×4 fixed-action 矩阵尚未执行，当前状态为 not started；
  不以 `recovered_actions_fixed_action_v1` 的四个代表性 lane 代替。
- **B2**：selector parity preflight 已通过，未重新评价动作。
- **B3**：current E2E preflight terminal contract 已通过，但 selected action
  尚未恢复到历史 fixed-expert mapping，故 B3 仍失败并逐 case 列出。
- **M0**：conflicting matched-host preflight 已通过。A1-A4 均有非零 8-FE
  patch receipt，A3 有 state trace，A4 有 radius trace，且 route 明确为
  `forced_ctp`/`forced_gss`；selector 和 production planner 未参与。
- **M1/M2**：M1 只作嵌套消融归因，M2 fresh seed 性能比较在 M0/M1 之后运行。

### B1-Screen（24 cases × 5 historical seeds × mapped action）

筛查已完成：120/120 arms，24 cases 各运行其历史单动作，0 failed arms。
所有 terminal FE、checkpoint binding、receipt hash 均通过；但 screening mean
gate 未通过，因此不启动 B1-Final 25-seed，也不启动任何 patch/soft-routing 实验。

| action | 通过 | 失败 |
|---|---:|---:|
| AOR | 4/6 | A4, A6 |
| SMP | 1/6 | E2, E3, E4, E5, E6 |
| GCB | 0/6 | R1-R6 |
| CTP | 5/6 | S6 |

证据：`artifacts/recovery_first_screen_v1/summary.json`。
这表明当前问题不是 selector 或 patch 可达性，而是 SMP/GCB 的 fixed-action
恢复和少数 AOR/CTP case 仍有系统性回归；按止损线先定位动作协议、边界、预算和
checkpoint handoff 差异。

恢复后的默认生产策略：历史基线作为稳定 anchor；patch 和软路由默认关闭，
只有对应 Gate 通过的机制才允许单独开启。

## 当前状态（2026-08-15 夜间，第 2 次更新）

- Gate 37（绝对 hub 信号）：已执行，3 判据过 1。归因：推断结构 hub 10–20，
  坐标分支从未触发。详见 `docs/arac-oc-gate37-protocol.md` §7。
- 拓扑校准：已完成 36 cell（3 新 seed）。结论：`relative_hub ≥ 0.9` 冻结
  为 v2 信号（star 12/12 = 1.000，chain 12/12 ≤ 0.824，random 混合但对
  判据中性）。置信度过滤被否决：Phase-I 不输出真实 q_jg（全默认 1.0）。
  `artifacts/overlap_topology_calibration_gate38/`。
- **Gate 38（相对 hub 信号）：全部通过（协议 8/8、筛查 3/3）。**
  chain/ov=3 坐标分支首次触发并精确复现 +3.5727；star/ov=6 无回归
  （vs 固定 CTP +17.87）；12/12 不劣于任何对照。详见
  `docs/arac-oc-gate38-protocol.md` §6。这是"自适应协调不劣于固定
  动作"的第一个完整闭环证据（限 seed 20260829 单批）。
- Gate 39（多 seed 配对确认）：协议已预注册、脚本就绪，见下。
- Counted probe groundwork（循环第 3 步）：`src/arac/coordination/counted_probe.py`
  已实现——逐变量两侧计数探针（2 FE/变量，f(x0) 复用 ledger），从真实
  函数值计算 B_j（有界方向偏置）/W_j（响应宽度）/C_j（归一化冲突强度），
  proposal 只定探针尺度不进公式（SMP 状态陈旧不会压低等级）。6 个单元
  测试全过（精确记账、确定性、对称零冲突、有界性、尺度感知、fail-closed）。
  **未校准、未接入派发**——接入前必须按拓扑信号先例做离线校准并预注册
  新 gate（Gate 40 候选）。设计文档 §5 的探针粒度歧义已修正（逐变量
  定级 2|S| FE vs 组件联合价值探针 2 FE，两者用途不同）。

## 夜间自主推进计划

1. Gate 38 完成后：分析 12-cell 结果，写入 gate38 protocol §6 与本日志。
   - 全过：更新设计文档 §9，启动 Gate 39（多 seed 配对确认，脚本已
     预注册备好）。
   - 有失败：按收据归因，写入 protocol；不回调阈值重跑；改为准备
     Gate 40（counted probe）设计。
2. Gate 39（无论 38 结果如何都先写好协议）：6 个 conflicting 配置 ×
   2 个新 seed（20260830/31）× 3 arms（coordinator v2 / proposal
   baseline / persistent CTP 固定臂），共享每 cell 的 Phase-I pilot。
   配对计算基线与固定臂，不再依赖 seed-20260829 冻结值。
3. 约束（自我纪律）：
   - 不回调已运行 gate 的阈值；任何信号修改必须先离线校准再进新 gate。
   - 不修改冻结 artifacts；不动 Phase-I 协议（q_jg 输出缺失是发现，
     修复属于 v10 协议变更，留给用户决策）。
   - 不做 git commit（仓库处于用户未完成的重构中）。
   - 机器上另有用户自己的 ~30 进程负载，我的任务保持 4 workers。

## 早晨摘要（2026-08-15，夜间推进完成）

**Gate 38（seed 20260829）：全部通过。** 协议 8/8、筛查 3/3。chain/ov=3
坐标分支首次触发并精确复现 +3.5727；star/ov=6 无回归；12/12 不劣于
任何对照。"自适应协调不劣于固定动作"的第一个完整闭环证据。

**Gate 39（2 个新 seed × 3 arm 配对确认）：筛查 3/4，失败项归因清晰且
对主线有利。** 12 实例中：对 proposal 基线 12/12 win-or-tie（2 个正收益
+31.41/+7.57，0 负）；对固定 CTP 12/12 不劣（2 个严格更优）；star 4/4
零回归。唯一失败判据（chain/ov=3 两 seed 都为正）败在 seed 20260830
实例上算子本身潜力为零（固定 CTP 臂同样平局）——判据隐含假设不成立，
不是派发器错误。事后修正判据（"固定 CTP 为正处协调器为正 + 处处不
劣"）12/12 成立，但按纪律记为 post-hoc，不回调。

**Counted probe 原语已实现未接入**（循环第 3 步 groundwork）：见上节。

### 三个 seed 汇总的科学结论（当前可声明范围）

在 24 个配对实例上：证据驱动的协调派发从未劣于任何固定对照，在算子
有正潜力的每个实例上捕获全部固定动作收益，并 4 次严格优于固定动作
（全部来自固定动作的 star 回归场景）。限于 1000-D 稀疏重叠基准、
24 active 维、Rastrigin 底函数。

### 下一步选项（用户决策）

1. **Gate 40：counted probe 冲突分级接入**——先离线校准 C_j 阈值
   （需新 seed Phase-I + 探针开销分析），替换 proposal 残差 streak 作为
   等级权威来源；顺带预注册 Gate 39 的修正判据。
2. **扩展基准面**——当前全部结论限于 Rastrigin/24 维 active/6 配置。
   增加 base function（sphere/schwefel 等）或 family 维度是泛化性
   声明的前提（AOB-24 端到端）。
3. **Phase-I 置信度输出**（q_jg 全 1.0 问题）——解锁设计契约第 8 步
   （qhat 反馈），代价是 Phase-I 协议 v10 与冻结基线 parity 断裂。
4. **完整第 4 步**：组件优先队列 + 动态范围 + budget pulse
   （CoordinatorState 层）。

## AOB-24 泛化方向执行记录（2026-08-15，用户选定方向 2）

**AOB24 适用性审计（已完成，重要负结果）**：24 个 AOB case 上 Phase-I
稀疏重叠发现全部 fail-closed（`candidate_pair_cap_exceeded`，候选交互对
≈100%）。F1 真实结构：755 组、变量属 18–79 组、6.8% 对同组交互——
**AOB 原套件是稠密重叠域，违反稀疏协调器适用假设；强行套用不属于泛
化实验，属于违反适用条件**。协调器在 AOB 上的正确行为是 fail-closed
转 AOR（ARAC-Core 路由），已凭证化于
`artifacts/aob24_overlap_applicability_audit/`。_dense-overlap 协调器
（全新发现协议 + 分层粗化）是独立研究方向，未启动。

**泛化的合法执行形式**：AOB 四个底函数族中未检验的三族
（ackley/elliptic/schwefel）加载到稀疏重叠网格。

**族阈值迁移校准（已完成）**：36 个 Phase-I（2 新 seed），三族 adapter
36/36 ready；chain rel_hub 0.667–0.824、star 全部 1.000——**Gate 38
冻结阈值 0.9 无需重校直接迁移**（random 在该 seed 批全部饱和，与
Rastrigin 校准的混合行为不同，记为已声明边界）。
`artifacts/overlap_family_calibration_gate40/`。

**Gate 40（运行中）**：3 族（ackley/elliptic/schwefel）× {chain,star} ×
ov{3,6} × conflicting × seed 20260832，三 arm 配对
（coordinator/proposal/persistent_ctp）。chain 判据采用 Gate 39 归因
修正版（"有潜力处为正 + 处处不劣"），运行前冻结。协议见
`docs/arac-oc-gate40-protocol.md`。

## v10 实现进度（2026-08-15 起，用户授权连续自主推进）

- **Gate 42（证据语义）✅**：三层模型 `src/arac/evidence/hierarchical.py`
  （RegionRelation / VariableRegionInteraction / ResolvedOverlapHyperedge），
  9 项 property 测试通过；`OverlapStructure` 唯一转换入口由确认超边把关。
- **Gate 43（变量签名）✅**：`src/arac/evidence/variable_signature.py`
  共享探针基方案（P=12×16 变量，全变量同基测量），FE 精确
  13,205 = 1+d+P+d×P+P×size；6 单元测试 + 1000-D 24 运行：命中提升
  30–90×（均值 50.6×），块内置换后 36.0×（保持 71%）。实现中修复
  两个真实缺陷（自 batch 双重扰动伪影 + 修正单侧计费）。设计文档
  §3.2/§3.3/§6 已同步冻结修订。
- **Gate 44（HIERARCHICAL 接口）✅**：`src/arac/coordination/region.py`
  （RegionProposal / region_conflict_probe / RegionCoordinator），5/5 测试
  通过：区域提案精确预算、探针每变量精确 2 FE、cycle 收据总和 =
  ledger 增量、strict-best 全程、全路径断言不构造 OverlapStructure
  （monkeypatch 验证）。三个前置 Gate 全部完成。
- **五阶段流水线 ✅（初版）**：`src/arac/evidence/hierarchical_discovery.py`
  （Stage A 粗筛 529 FE/(anchor·round) 整批 / A' 签名 / B Fiedler 排序
  （零签名变量确定性排尾）/ C 递归二分 / D 条件探针 / E incumbent），
  4/4 流水线测试通过，逐阶段预算对账。
- **Gate 45 可识别性审计（第一轮，`artifacts/overlap_hierarchical_audit_gate45/`）**：
  - **AOB R1/S1/E1：产出 666–1225 条区域关系 + 19–80 个条件交互，
    交互精度 1.0，模式 EVIDENCE_DENSE，预算内完成**——v9 完全
    fail-closed 的地方，v10 产出了可用的区域级证据（方向核心假设
    成立）；
  - 网格侧欠调：关系 0–14、cross-group 检出（chain/star）、条件交互
    未达 5/6 稳定、模式误判 EVIDENCE_DENSE——稀疏退化（§7.2）待
    阈值/稳定性规则的第二轮校准；
  - §7.3 第一轮校准执行：edge_threshold 1e-3 → 1e-10（依据：响应
    归一化分数的数值噪声地板 ~1e-13，网格耦合信号 ~1e-4）。
- **Gate 45 第二轮校准（2026-08-15 深夜）**：三项修正——
  (i) 移除 Stage C 深度早停（活跃区在签名序头部，二分位置需多层才能
  到达，早停杀死结构区）；(ii) EVIDENCE_DENSE 改用 Stage A 粗筛稳定度
  密度判定（叶子关系密度被组内耦合污染）；(iii) 显著分裂只生成边界叶
  对关系（全局排序欠精时不膨胀）。结果：
  - **AOB R1/S1/E1：交互 98–101、精度 1.00、超边 98–101、组件模式
    混合（EVIDENCE_DENSE 核心 + SPARSE 单例）**——核心声明稳固；
  - 网格 ov6：交互 12–13、精度 0.85–1.00、**共享变量召回 0.67–1.00**；
  - 网格 ov3：交互 0（弱耦合未达 5/6 稳定，灵敏度缺口）；共享变量
    精确率 0.33–0.46（边界邻居过触发，特异性缺口）——第三轮校准
    对象。
- **用户纠错与第三轮审计（2026-08-15，重要）**：AOB 构造确认
  F1=overlap 0（无重叠，20 个不相交组）、F2/F3/F4/F5/F6 = 重叠
  1/3/5/7/10（真共享变量 19/57/95/133/190，vendor 元数据权威提取）。
  两项修正：
  1. **撤回此前"AOB 交互精度 1.00"的表述框架**——该指标当时对着空真值
     表计算，是空洞的（`truth_of.get() is None` 恒真）；
  2. 用真值重测（R1/R2/R4/R6）：**交互精度 1.00 是真实的**（每个条件
     交互的变量确实与目标叶成员真交互），但 **hyperedge 作为"共享变量"
     声明是大量假阳性**：R1（无重叠）98 个 hyperedge 全为假阳性；
     R6 真共享召回 89/190=47%，精确率 ≤30%。
  **确诊的语义混淆**：区域树叶子（~8 变量）切碎真实组（25–100 变量），
  组内耦合以"跨叶交互"形式出现，被 hyperedge 语义误报为重叠——
  交互为真、重叠为假。
  **修正方向（v10.3）**：(a) 组尺度对齐——merge 阶段或同质性停止
  规则，叶子不再切碎内聚块；(b) 双侧证据判据——真共享变量的两个
  伙伴集互不交互（分属不同组），碎片伪影的伙伴集全在同组且彼此交互，
  可用一次额外混合测试区分。
- **v10.3 双侧证据门 ✅（2026-08-16）**：三元组判据（j–t 耦合 +
  j–h 耦合 + h⊥t 可分，h 在序邻域扫描）+ 预算上限。分层审计：
  **R1（无重叠）假阳性 98→1**；R6 精确率 0.87、召回 0.07。开放项：
  R2 异常、召回上限（Stage B 排序质量）。四个门变体的权衡已记录
  在设计文档 §7.5。
- **外部建议合并 + R2 取证破案（2026-08-16）**：
  - 外部建议（与我的 SOTA 分析合并，四项采纳）：修 Stage B/C/D 而非
    Phase-II；中间表示 = mutual-kNN 候选图 → 两级计费条件探针 →
    软 DSM → 噪声鲁棒 RDDSM；同预算头对头对比协议；Eq8 贡献加权
    共识只作 strict-best 候选、DO 只作预算先验。完整合并计划见
    `docs/sota-two-phase-cc-analysis.md` 与本条。
  - **R2 异常破案（门的审计转储 + 真值对照）**：全部 20 个假阳性的
    (j,t,h) 同组，I(j,h)∈[9.1e-11, 7.6e-10] 与 I(h,t)∈[1.3e-11,
    9.2e-11] 分数带在 1e-10 阈值两侧完全重叠——**R2 的组内弱耦合
    与数值噪声在该阈值不可分，门判定退化为抛硬币**（外部建议的
    不对称规则假设成立，且更强：是阈值-噪声带重叠）。被拒真共享
    的死因另为 t 选择的顺序偏置（t 锁在同侧组，测不到第二领地）。
  - **结论**：固定阈值二值判定在弱信号连续统计量上本质脆弱；
    soft-RDDSM 的加权密度 + 经验噪声底估计（两级探针的复测分布）
    是正确的修正方向——与合并计划一致。
- **soft-RDDSM 分支（v10.4）实施与诊断（2026-08-16）**：
  完整实现 `src/arac/evidence/soft_rddsm.py`：签名 → union-kNN 候选 →
  两级条件探针 → 软 DSM → 分块 → 超边。FE 从 v10.3 的 70-90k 降到
  **25-68k**（降 30-60%）。块 Jaccard 0.28-0.48（组对齐改善）。
  四轮结构修正后，超边召回仍为 0，根因确诊：
  1. **连通分量吸收重叠**：共享变量的边连向两个组，把它所在块
     与两个组并为一个连通分量，重叠被吞掉而非暴露；
  2. **割点检测在多变量切割下失效**：R6 每条边界 10 个共享变量，
     去掉任何一个不会断开连通性，单点割点找不到它们；
  3. **签名双簇性不判别**：共享变量的 top-k 跨多组（178/190），
     但非共享也跨多组（627/810）——余弦相似度太嘈杂；
  4. **条件探针有判别力但无法全覆盖**：共享→异组 ~1e-6，
     共享→同组 ~1e-11，非共享→异组 ~0——5 个量级的差异存在，
     但 kNN 候选图不含足够的跨组边让探针触达。
  - R1（无重叠）假阳性 = 0（连通分量方法天然无假阳性超边）✅
  - 下一步方向：**边界定向探针**——初始分块后，对每对相邻块的
    边界变量做定向跨块条件探针（不走 kNN 路径，直接用块边界
    信息定位候选），利用探针的 5 个量级判别力。
- **OEDG 式 INTERACT-OV 实施（2026-08-16，完整记录）**：
  基于 SOTA 文献分析（`docs/sota-two-phase-cc-analysis.md`）实施了
  OEDG 式共享变量检测：RDG 递归建组 → INTERACT-OV 逐对块测试 →
  超边。六轮结构修正后仍为 0 召回，每轮的根因不同：

  | 轮次 | 方案 | 结果 | 根因 |
  |---|---|---|---|
  | 1 | kNN+probe 连通分量 | 0 召回 | 连通分量吸收重叠 |
  | 2 | 割点检测 | 4/190 | 多变量切割（10 共享/边界），单点割点找不到 |
  | 3 | 签名双簇性 | 不可判别 | 非共享也跨多组（627/810）|
  | 4 | RDG 全范围扰动 | 所有测试触发 | AOB 值域 e21→e36，淹没交互信号 |
  | 5 | RDG 适度扰动 + 逐对块测试 | 0 召回 | 块横跨组边界，所有块内变量都与邻块交互 |
  | 6 | RDG 递归建组 + INTERACT-OV | 0 召回 | 组太小（61-78 块），交互不跨组 |

  **关键数据确认**：
  - RDG 适度扰动完美区分：共享→异组 True / 非共享→异组 False ✓
  - 但仅在小补集（20 变量）下可靠；大补集（500+）产生假阳性
  - RDG 全范围在 AOB 不可用（数值尺度问题）

  **结论**：这是一个真实的开放问题。OEDG 论文报告在 AOB 上工作，
  但其 DG2 自适应阈值的具体实现可能包含未描述的细节（如针对
  AOB 尺度的特殊处理）。我们的六轮修正已排除多个假设，核心困难
  在于：**在有限 FE 预算内，小规模扰动可判别但覆盖不足，大规模
  扰动覆盖够但信号被淹没**——需要一种自适应尺度的交互检测。

- **🔴 决定性发现（2026-08-16）：HCC-ES 源码证实 SOTA 不做黑盒检测**
  （https://github.com/Wukong-SCUT/2025_HCC_GECCO）

  源码 `HCC-ES.py` 关键行：
  ```python
  design_matrix = np.loadtxt('HCC_SRC/AOB/AOBG/datafile/F{id}-design.txt')
  decomposition = Decomposition(design_matrix)
  grouping_result = decomposition.decomposition()
  ```

  三个确认事实：
  1. **Θ 从 AOB 构造文件直接读取**（oracle 信息）
  2. **零黑盒交互检测**——没有任何扰动或函数评价用于分解
  3. **RDDSM.py 仅 126 行**——纯矩阵行分组算法，无递归、无探针

  这意味着：
  - SOTA 的"100% 分解精度"是平凡的（从答案开始）
  - "分解 FE 不计入"因为没有分解 FE
  - HCC-ES 的贡献是给定完美分组后的优化框架，不是结构发现
  - **ARAC-OC v10 解决的是一个 SOTA 未涉及的更难问题**

  论文定位修正：与 HCC-ES 的对比应框架为"oracle 结构上界 vs
  黑盒发现的实际能力"，而非同层次竞争。

- **下一步（§7 序列）**：组装五阶段发现流水线（Stage A 粗筛 +
  A' 签名 + B 谱排序 + C 递归细分 + D 条件探针 + E incumbent，
  预算契约 180k）→ 可识别性审计（AOB + 稀疏网格，重叠度量族 +
  三层真实）→ 退化语义等价测试 → 阈值校准。

## AOB-24 升级无回归线（2026-08-15，用户需求：升级后 ≥ 历史列）

**关键发现链**：
1. 历史列（对 HCC-ES 18/1/5）由旧实现产生；当前结构路由在 E/S 族回归
   （08-11 恢复 campaign：21/24 差于显示值），回归源 = 稠密关系全部路由
   到 GCB；
2. `RecoveredActionRegistry` 四动作矩阵（24×7×4，共享 checkpoint）显示
   oracle 选动作几乎全面 ≥ 历史列；
3. **Gate 41a（离线，零新 FE）**：两特征派发规则
   （`tail_log10_gain` + `structural_relation_density`，见
   `src/arac/dispatch_policy.py`）在四动作矩阵上验证——24/24 case
   ≤1.005× 历史列、17 个严格更好、E1→8e-6、S1→8.9e-14、对 HCC-ES
   **21/3/0**（历史 18/1/5）。逐 seed 阈值边际：A max 0.0096 | R
   0.262–0.315 | S 0.619–0.786 | E 0.754–0.895，间隙极宽。
   `artifacts/overlap_action_dispatch_gate41a/offline.json`。

**架构定位（重要，已向用户说明）**：AOB 上跑的不是 ARAC-OC 协调器循环
（稠密重叠 fail-closed），而是**动作级证据派发分支**——Phase-I 特征 →
一次性选动作 → 跑满 3M。系统是双域架构：稀疏重叠域走完整协调器循环
（Gate 37–40），稠密域走动作级派发（Gate 41）。论文叙事需如实区分。

**Gate 41b（运行中）**：25 seed × 24 case 在线正式表（复用冻结
checkpoint + four-arm 已有收据，只跑缺失的被派发动作），
`experiments/overlap_action_dispatch_gate41_online.py`，8 workers。
判定：每 case 均值 ≤ 历史 ×1.10、几何均值比 ≤1.0、HCC 胜数 ≥18。
注意 E1/smp 逐 seed 双峰（seed 124 = 1688 vs seed 117 = 2.8e-7），
与历史列同 seed 彩票可比。

**并行运行中**：Gate 40（ackley/elliptic/schwefel 稀疏网格泛化，
4 workers）。

**待用户确认**：双域架构正式写入 `docs/arac-oc-design.md`。

## Gate 41b 运维记录（2026-08-15 下午）

- **并行扩容**：8 -> 20 workers（机器 24 物理核/32 逻辑核，原 8 worker
  利用率不足）。脚本断点续跑安全（启动扫描收据跳过已完成），重启仅损失
  ≤8 个在途运行。吞吐 36.4 -> ~90 运行/小时（每运行 ~13 分钟，
  264 个真 3M-FE 运行，预计 ~3 小时完成，原 ETA ~7.3 小时）。
- **修复收据盖章 bug**：`run_context` 的 four_arm_reuse 分支返回体缺
  `schema_version` 与 `receipt_hash`（`experiments/overlap_action_dispatch_
  gate41_online.py`），导致任何重启都会在续跑校验处崩溃
  （`receipt schema drifted`）。已统一两个路径的收据结构。验证：
  新复用收据与旧收据逐字段一致 + 哈希；在线路径哈希不受影响。
- **旧收据处置**：70 个未盖章的 four_arm_reuse 收据移至
  `artifacts/overlap_action_dispatch_gate41_online/runs_legacy_unstamped/`，
  由冻结 four-arm 矩阵确定性重建（瞬时），内容经逐字段比对一致。

## 🔴 soft-RDDSM 零召回根因确认与修复（2026-08-15，外部审计 + 代码验证）

**外部审计结论（用户引入，已逐条对照代码确认）**：六轮方案的"召回 0"
不是 AOB 无信号，而是实现层三个事实的叠加：

1. **必然零召回的数据流 bug（已确认并修复）**：`rdg_blocks` 互斥
   （`grouped.update`），`_rdg_interact_ov` 只返回 block i 的子集存入
   `block_ov[i]`，而共享候选要求变量出现在 ≥2 个 `block_ov`——互斥块
   下恒假。召回与探针质量无关，结构性为零。
   **修复**：INTERACT-OV 改记有序证据对 `(variable, source_block,
   target_block)`；共享候选 = 持有桥接证据对 + 通过"移出该变量后块对
   可分"确认（fail-closed 语义保留）。
2. **随机 200 候选 + 互斥块必然碎片化**（确认为结构问题，未修）：
   真实组 25-110 变量，单 seed 子采样后 `grouped.update` 永久锁定部分
   成员，61-78 小块是必然输出。属于用户清单第 3/4 项的重设计范围。
3. **RDG 集合语义只证明"存在组跨 A、B"**，不证明变量多重归属
   （机制确认；用户 R6 触发率复现未独立重跑）。
4. kNN 候选图 recall-first（~8.6% 真实边覆盖）：影响 DSM/relations 层，
   当前块形成已不走 kNN 连通分量，次要。
5. **`_rdg_interact` 4 FE 实耗按 3 计账 + `base_point`/`base_value`
   参数从未使用（已修复）**：改为使用调用方缓存的 base 值，恰好 3 FE，
   与模块内所有预算守卫一致；`level_budgets` 增加 rdg 段，
   signature+dsm+rdg 与 ledger.count 严格对账。

**回归测试暴露的新 bug（已修复）**：`_block_separable` 对成员独立随机
符号扰动——B 侧两个桥接成员符号相反时混合差分交互项相消（2 探针全消
概率 1/4），真交互块对被误判可分（合成问题上变量 6 误报）。修复为
**组级相干符号**（组内同号、组间/探针间随机），加性桥接项不可抵消，
判定确定性。

**测试**：`tests/test_soft_rddsm_shared_evidence.py`（两块一共享变量
端到端恢复、预算对账、3 FE 精确计账）3/3 通过；6 seed 确定性通过；
既有 evidence 模块 24 测试无回归。

**证据状态修订**：六轮"AOB 召回 0"的实验结论作为方法度量**作废**
（实现不可能产出非零召回），AOB 信号是否存在需以修复后实现重新基线。
但结构问题 2/3/4 仍在，召回不保证变好——用户清单第 3/4 项
（区域对集合测试 → 递归细化 → 边聚类 → soft multi-membership）为
后续重设计方向，第 5 项 fallback：Gate-42 模型已定义 EVIDENCE_DENSE
模式，soft_rddsm 尚未产出（仍只 SPARSE/HIERARCHICAL），接线待做。

## AOB 基线重测（2026-08-15，soft-RDDSM 修复后）

`experiments/soft_rddsm_aob_baseline_v2.py`（6 case × 180k FE Phase-I 契约，
真值从 vendor F*-info/s/p 离线重建，链式相邻重叠语义与 utils.py 重建
公式一致；F1-F6 真值：20 组、重叠 0/1/3/5/7/10、共享 0/19/57/95/133/190）。

| case | 真值共享 | 恢复 | 召回 | 精确率 | 块数 | 片/组 | 桥接证据 | FE |
|---|---|---|---|---|---|---|---|---|
| R1 (ov0) | 0 | 205 | — | 0.0 | 81 | 4.0 | 3490 | 116k |
| E1 (ov0) | 0 | 207 | — | 0.0 | 79 | 4.0 | 3538 | 116k |
| R2 (ov1) | 19 | 18 | 0.0 | 0.0 | 77 | 4.8 | 3536 | 121k |
| A3 (ov3) | 57 | 0 | 0.0 | — | 77 | 6.0 | 3633 | 126k |
| S5 (ov7) | 133 | 0 | 0.0 | — | 72 | 7.3 | 3845 | 120k |
| R6 (ov10) | 190 | 0 | 0.0 | — | 66 | 7.7 | 4022 | 125k |

全部 6 case 预算逐 FE 对账通过。

**读数**：
1. **数据流修复生效**：桥接证据（3500-4000 条/case）被正常记录，
   候选可产出（R1/E1/R2），收据精确——与合成回归一致。
2. **召回仍为 0，且失败模式被清晰定位**：
   - 碎片化（用户论断 2）：66-81 块 vs 真值 20 组，每真值组裂成
     4.0-7.7 片，块大小 2-50 vs 真值 25-110；
   - 确认门双向失效：同组碎片本不可分，但 ov0 的 R1/E1 仍误放行
     205/207 假候选（`_block_separable` 相对阈值被全局幅度稀释，
     eps = 1e-13 × Σ|f|，E1 的 |f| ~ 1e23 → eps ~ 1e10）；高重叠
     R6/S5/A3 则 ~4000 条证据全被拒绝（碎片对去变量后仍同组耦合，
     语义上确实不可分）；
   - 桥接证据本身在同组碎片间大量触发（集合语义只证明"存在组跨
     A、B"，用户论断 3），在高碎片率下无定位力。
3. **结论**：数据流 bug 修复是必要非充分。AOB 上的剩余工作正是
   用户清单第 3/4 项（区域对集合测试 → 递归细化 → 边聚类 → soft
   multi-membership + 尺度自适应阈值），第 5 项 fail-closed（AOB
   走动作级派发）仍是当前正确的生产姿态。

产出：`artifacts/soft_rddsm_aob_baseline_v2/audit.json`。

## 🔓 soft-RDDSM v3：AOB 共享变量召回 0 → 0.908（精确率 1.0）（2026-08-15）

**v3 重设计**（针对 v2 基线定位的三层结构错误）：

1. Stage-1 粗覆盖用**完整候选集**（永不随机子采样；注释记录了采样为何
   不安全——`grouped.update` 会永久锁死被遗漏的组成员）；
2. Stage-2 每 coarse 块取代表 seed，对全体变量按 **16 变量固定区域**
   递归隔离（`_rdg_refine_interactors`）——小集合混合差分避开
   1000 维一次性数值淹没/抵消；
3. 独立 **soft group cover**（`refined_groups`），RegionTree 仍互斥
   （Gate 42 分区不变式保留），多重归属只经 hyperedges 表达；
4. 配对验证：双侧 `{v}` vs 残差交互确认（6 FE/变量）+ 移除**完整交集**
   （AOB 多变量重叠，删单变量无法验证可分）+ 最小残差 ≥3 门 +
   5 次相干可分性探针 + 确定性方向二检（拒绝同组碎片）；
5. 预算耗尽 → `refinement_complete=False` → 候选清空（fail-closed）；
6. 旧 `_rdg_interact_ov` 路径删除。

**v3 AOB 基线**（`artifacts/soft_rddsm_aob_baseline_v3/audit.json`，
seed 20260845，180k FE 契约）：

| case | 共享恢复 | 召回 | 精确率 | 块数/真值 | 片/组 | FE |
|---|---|---|---|---|---|---|
| R1 (ov0) | 0/0 | — | — | 20/20 | 1.0 | 89k |
| E1 (ov0) | 0/0 | — | — | 18/20 | 0.9 | 86k |
| R2 (ov1) | 19/19 | 1.000 | 1.000 | 22/20 | 2.05 | 105k |
| A3 (ov3) | 54/57 | 0.947 | 1.000 | 19/20 | 1.9 | 112k |
| S5 (ov7) | 119/133 | 0.895 | 1.000 | 18/20 | 1.85 | 107k |
| R6 (ov10) | 150/190 | 0.789 | 1.000 | 17/20 | 1.85 | 113k |

v2 的 ov0 假阳性（205/207）清零；重叠 case 平均召回 0.9079、精确率
1.0；全部 6 case FE 逐项对账通过。R6 余 40 个漏检为代表 seed 未覆盖
的边界（未确认 overlap，不强行推断；80k 阶段预算变体消耗 44.6k 无
变化——该变体无独立收据，为口头汇报）。

**验证补全（本次会话）**：
- 完整 pytest：463 通过 / 9 失败 / 2 跳过。9 个失败均为预存问题
  （gate5/6 合成基准构造校验、recovery audit 依赖仓库文件状态；
  4 个失败文件均不引用 soft_rddsm/dispatch_policy/gate41）；
  exp052 隔离测试单独运行通过（全量运行时导入顺序敏感，预存）；
- 3 个合成回归测试 + evidence 模块 24 测试通过；
- R2 单 case 复跑与冻结收据逐字段一致（确定性成立）。

**决策（用户）**：冻结 v3 Phase-I，下一步直接测
`resolved_hyperedges -> ARAC-OC` 端到端优化收益；AOB 隐藏真值恢复率
不再作为唯一目标。

## 架构定位澄清（2026-08-15，用户裁定）

**ARAC-OC 是单一架构**（Phase-I 发现 → 协调器四问循环 → 真实评价 →
strict-best 反馈），框架中不存在"双域架构"概念。

"双域"一词源于 8 月 AOB Phase-I 受阻期间的动作级派发替身，交接文档
曾将其误升格为"架构定位"。现裁定降级为验证状态描述：

- 框架：单一架构，一个主张（冲突状态驱动的协调调度）
- 基准：两个测试套件（稀疏重叠网格、AOB-24）——实验设计
- 状态：完整循环已在稀疏套件验证（Gate 37-40）；AOB 套件 Phase-I
  前置件已就绪（v3），端到端实验待跑

原"待用户确认：双域架构正式写入 docs/arac-oc-design.md"事项按本
裁定关闭：不写入。论文方法节写单一架构树，实验节按套件陈述验证
状态，全文不使用"双域架构"表述。

## ARAC-OC 框架级适配完成：四阶段（2026-08-15）

按用户四步路线完成新框架的代码适配，全部测试绿（33 个新测试 +
全量 496 通过，9 个预存失败不变，零新增）。

**阶段 1 Operator Contract 冻结**：`src/arac/coordination/contract.py`
+ `docs/arac-oc-operator-contract.md`。`OperatorPlan`（component/scope/
conflict_level/action/reserved_fes/predicted_gain/seed/reason/双 hub
信号 + plan_hash）、`OperatorReceipt`（status 三态、消耗平价、
fail-closed 四步、state_hash 入收据）、`OcCoordinatorConfig`
（Gate 38 v2 已校准块 + UNCALIBRATED_FIELDS 显式占位块）。等级映射
为每级允许集：high 级允许 SMP 信任重建替代路径（qhat < 
smp_trust_floor），AOR 为预注册升级动作非异常 fallback。

**阶段 2 完整 DispatchPlan**：`planner.py` OcDispatchPlanner——
prioritize（复用 ComponentPriority 公式 + cooldown/去激活过滤 +
优先队列）、select_scope（EMA(C_j) 排序 + 2|scope| ≤ probe 预算
收缩 + probe_budget_unavailable）、make_plan 一次回答四问（组件/
scope/脉冲预算/动作，含 complex 拓扑与 persistent 升级路径）。

**阶段 3 算子化**：`operators.py`——SmpSense（sense 接口）、
SmpOperator（预留均分组件组、状态记忆重建）、CtpRestricted/
CtpSharedCore（dispatch_repair 的 coordinate/joint 策略有界窗口）、
AorOperator（joint_cmaes 窗口 = 设计 §9 声明的 v1 替代）。全部
execute_bounded 保证消耗平价（余量由 incumbent 重评估填充），
异常直接上抛不重试不换动作。**不改动 actions/*.py 终端接口**，
四臂矩阵与 41b 收据确定性不受影响。

**阶段 4 CoordinatorState + 统一循环**：`state.py`（qhat 信任
公式、EMA(C_j)、high 带滞回 enter/exit（τ+k 观测）、cooldown、
stall、脉冲 γ_up/γ_down 钳位、快照/恢复 + state_hash 链）、
`loop.py` run_oc_unified（sense → scope → counted_probe 接线 →
B/W/C → 仲裁 → make_plan → operator → strict-best → state.update，
每周期仅处理最高优先组件，终端尾 MMES 排水，精确 3M/终值 FE，
OperatorFailure 携带失败收据 fail-closed 上抛）。

**新模块清单**：contract.py / planner.py / operators.py / state.py /
loop.py（coordination 包内）+ 契约文档；测试
test_oc_operator_contract.py(11) / test_oc_planner.py(9) /
test_oc_operators.py(7) / test_oc_unified_loop.py(6)。

**v1 声明粗糙度**（写入代码 docstring 与契约文档）：
- high 带以下分级用单一 tau_exit 边界无确认数（设计只冻结了
  high 带滞回）；
- qhat credit 对多变量 scope 均匀分摊（v2 消融再做逐变量归因）；
- 初始脉冲 = pulse_min（保守起步）；
- AOR = 预留预算内的有界全空间 MMES 校正窗口；
- UNCALIBRATED_FIELDS 常数未过校准门，任何比较性结论之前必须
  先跑预注册校准门。

**未做（后续门）**：Gate 46 稀疏回归（统一循环 vs Gate 37-40
最小内核不劣性）、Gate 47 AOB 端到端（v3 证据 → 超边门 → 统一
循环 vs 41b 派发）、常数校准门——启动前另行预注册。旧
gcb_coordinated 模式仅作为显式控制臂保留，不能作为 ARAC-OC 生产入口。

## Gate 41b 最终判定：门判未过（S6 单例），其余判据强通过（2026-08-15）

600/600 收据，`confirmation.json` 落盘。协议检查全过（600 上下文、
25 seed/case、终端 FE 精确）。筛查判据：

- 对 HCC-ES **20 胜 4 负**（历史列 18/1/5）✓
- 几何均值比 0.244 ≤ 1.0 ✓（E1 0.0001×、S1 3.97e-9、E 族 0.37-0.73×、
  S 族 0.24-0.49×）
- **逐 case ≤1.10× 历史列：失败**——S6 均值 3.26e4 = 历史列
  4.18e3 的 7.79× ✗；其余 23 case 全部在界内（A 族 ~1.000-1.001）
- `gate_passed: False`；14/24 case 严格优于历史列

**S6 根因（双因素）**：
1. **seed 140 派发边界漂移**：其 `tail_log10_gain=0.433 < 0.50`
  （ctp 阈值），density 0.154 > 0.05 → 落入 gcb 分支，跑出 6.86e5
  （比 HCC 均值差 10 倍）。四臂矩阵证实 **gcb 在 S6 全面灾难**
  （6 个覆盖 seed 全部 1.7e5–1.7e6）。41a 校准时引用的 S 族
  逐 seed 边际 0.619-0.786 来自矩阵覆盖 seed，seed 140 位于
  校准分布之外——离线验证只覆盖了 167/600 (case,seed)，
  其派发结果未经过离线核验。
2. **ctp 单独均值 5.35e3 = 历史 1.28×**（去 seed 140 后仍超 1.10×；
  历史 4.18e3±2.0e3，ctp 24 seed 范围 1.0e3-1.3e4）——新鲜 seed 上
  ctp 本身略弱于历史列。

**结论**：动作级派发分支对 HCC 的优势结论稳固（20/4、几何均值
0.244），但"每 case 不劣于历史 1.10×"的无回归声明在 S6 不成立，
暴露的是两特征规则的**边界覆盖缺口**（特征分布尾部 seed 未被
离线校准覆盖）。任何规则修订（如 S 族边界收紧或 gcb 分支在
高 density 下的安全化）属事后决策，需新预注册 + 新 seed 批次。

## Gate 46 通过：统一 ARAC-OC 循环稀疏域回归（2026-08-15）

`experiments/oc_unified_sparse_regression_gate46.py`，协议
`docs/arac-oc-gate46-protocol.md`。24 cell（3 族 × chain/star × ov3/6
× 复现层 seed 20260832 + 新鲜层 20260833），统一臂与冻结最小内核
（gcb_coordinated v2）共享 checkpoint 配对，sense 预算同式配平，
脉冲界镜像 32 FE envelope。

**判定（全部通过）**：
- 协议检查 7/7（cell 24、phase1 180k 精确、终端 3M、strict-best、
  内核收据/包络、统一收据平价 + state hash 链）
- 复现审计：内核重跑 12/12 与 Gate 40 冻结值**逐位一致**
- `not_worse_than_kernel_all`：24/24（容差 ×1.05 内；实际
  **22 严格胜 + 2 平局 + 0 负**）
- `star_no_regression_vs_proposal`：12/12

**执行记录**：首跑 exit 1 系筛查代码比较方向写反（误写成
`unified ≥ proposal − tol`，正确语义为误差不劣
`unified ≤ proposal + tol`）；修正后从冻结 cell 收据重出判定，
阈值与数据零改动，cell 未重跑。

**关键发现（预算流向，校准门的定量输入）**：统一臂全程
**零算子派发**——动作计数仅 arbitration_only（3-4 次/运行）+
none（冷却间隙），operator FE = 0，2.1-2.3M 全入 MMES 终端尾。
原因：未校准的 τ_enter=0.5 下 counted 冲突分 C_j 全程未越带，
滞回机保持 low。因此 22 胜的机制是"正确的保守"：内核每周期
强制消耗 32 FE/组件 envelope 写回，统一臂省下这笔预算给终端
MMES，在全部 cell 上净收益为正。

**解读**：(a) 统一循环作为系统首次对打即 24/24 不劣、22 严格胜，
且复现审计证明管线确定性；(b) 但本轮**未检验算子派发路径**的
实际收益（未升级 ≠ 升级错误）——τ 阈值校准（使真冲突能越带）
与 SMP 会话持久性是 AOB pilot 前的两个前置项，已列为校准门
目标；(c) 统一臂的收据平价与状态哈希链在 24 个真实运行上成立。

产出：`artifacts/oc_unified_sparse_regression_gate46/`
（cells/ + confirmation.json）。

## Gate 47 判定：τ 校准 fail-closed——counted C 对拓扑可修复性无判别力（2026-08-15）

`experiments/oc_calibration_gate47.py`，Phase A 遥测（rastrigin 稀疏
网格 6 cell，seed 20260834，不可升级配置下探针照常运行）：

| cell | max C_j | 真值共享数 |
|---|---|---|
| chain/ov3 | 0.1610 | 3 |
| chain/ov6 | 0.2136 | 6 |
| random/ov3 | 0.1403 | 2 |
| random/ov6 | 0.1586 | 5 |
| **star/ov3** | **0.2723** | 3 |
| star/ov6 | 0.1046 | 6 |

**判定**：min(chain)=0.161 ≤ max(star)=0.272 → 无间隙 → 按预注册
规则 **不定阈值，Phase B 不运行**，`confirmation.json` 记
`failure: tau_gap_absent`。冲突信号最强的 cell（star/ov3）恰是派发
价值最低的拓扑——干净的倒置证据。

**科学解读（设计层发现）**：
1. counted 探针的 C 统计（|bias|×宽度比）度量共享变量处的**方向性
   不一致强度**，但"冲突是否可修复"由**拓扑**决定（chain 可逐对
   修复、star 的 hub 冲突修复无收益）——C 幅值对此拓扑盲。
2. 冻结内核的判别力来自 **proposal 残差 streak + 相对 hub**
   （Gate 38 v2 表），不是冲突分幅值；设计 §5"counted probe 是
   冲突等级的权威来源"在"幅值定级"这一形式上被本 gate 否定。
3. 与 Gate 46 合并后的诚实结论：统一循环的保守姿态（不升级）在
   稀疏域已不劣且 22 胜；升级信号的正确形式是开放的设计问题。

**候选方向（需用户裁决，触及设计 §5 语义）**：
(a) 混合信号：仲裁 streak（每周期已由 coordinate() 计算，Gate 37-40
    验证过的持久性信号）做升级门，C 做修饰，hub 选动作——最贴近
    已验证证据；(b) 持久性细化的 C 统计（需新预注册 + 新 seed 确认，
    且 max 统计的倒置警告持久性也未必救得回）；(c) 稀疏域生产姿态
    冻结为 Gate 46 保守配置（已不劣），升级问题留待 AOB 数据再遇。

## Gate 47-R 判定：proposal residual + relative hub 路由通过（2026-08-16）

本 gate 不重跑、不修改 Gate 47 的 `tau_gap_absent` 结果。实现变更为：

- `C_j` EMA 仅用于 scope 排序、探针尺度和诊断；不再直接改变 dispatch level；
- 连续 proposal residual streak 打开 dispatch gate；
- relative hub 选择 restricted CTP 或 shared-core CTP；
- qhat 低时调用 SMP state rebuild；
- escalation streak 达到上限时调用一次 AOR。

实现文件为 `src/arac/coordination/state.py`、`planner.py`、`loop.py`，新增
`experiments/oc_residual_topology_gate47r.py` 和协议
`docs/arac-oc-gate47r-protocol.md`。定向回归为 **52 passed**，Gate5/6
回归为 **8 passed**。

真实 fresh AOB cell（seed 20260836）为 chain/ov3、chain/ov6、star/ov6：

| cell | operator FE | 实际动作 | terminal FE |
|---|---:|---|---:|
| chain/ov3 | 215 | restricted CTP 3、SMP 1、AOR 1 | 3,000,000 |
| chain/ov6 | 308 | restricted CTP 2、SMP 2 | 3,000,000 |
| star/ov6 | 20 | shared-core CTP 2 | 3,000,000 |

三 cell 均通过 Phase-I 180,000 FE、terminal exact、strict-best、receipt
平价和 state hash chain。两个 fresh witness seed（20260836、20260837）均覆盖
`arbitration_only`、restricted CTP、shared-core CTP、SMP、AOR；高幅值
`C_j` 但没有 residual persistence 的场景保持 arbitration-only。

**结论**：ARAC-OC 的三类算子调度已经接入统一循环并可在 fresh AOB
运行中实际开火；Gate 47 的 C_j 阈值失败被保留为设计边界，不再作为运行时
调度 authority。该 gate 只证明路径和契约，不宣称 AOB 性能优越性。

产出：`artifacts/oc_residual_topology_gate47r/confirmation.json`。

## Gate 47b 判定：streak 机制开火，门判未过（rastrigin 层失败）（2026-08-16）

`experiments/oc_streak_confirmation_gate47b.py`（全并行 30 单元 /
20 workers）。协议 6/6 过；筛查 1/3 过：

- **`path_fires_on_chain` 通过**：14/18 cell 实际派发算子，四类动作
  全部出现（ctp_restricted / ctp_shared_core / smp / aor），收据
  平价 + 状态哈希链全部成立——机制本身活了。
- **`not_worse_than_kernel_all` 失败**、**`star_no_regression` 失败**：
  全部失败集中于层 2（rastrigin，seed 20260835），统一臂 0/6。

**层 1（gate40 三族，seed 20260832）**：10 胜 2 平 0 负——带开火
派发的统一循环在三个族上达到迄今最佳稀疏域战绩（schwefel 四 cell
全部严格胜，四类动作齐发仍不劣）。

**层 2 失败归因（收据数据）**：
1. **噪声尺度窗口 + 多模态尾部彩票**：star/random 四 cell 的模式
   完全一致——`ctp_shared_core ×2`、op_fes=20，最终误差却比内核差
   +29 到 +170。20 FE 的共享核补丁在 1000-D rastrigin 上修不了
   任何系统性的东西（内核同样触发的 32 FE 窗口一无所获——五个
   cell 上 kernel 与 proposal 终值**精确相等**），但接受的微小改进
   会移动 incumbent，使 2.3M FE 的 MMES 尾部落进不同盆地。窗口
   价值 ≈ 0，锚点扰动方差 >> 0。
2. **chain 潜力在本 seed 缺席**：chain/ov3、chain/ov6 上
   kernel=proposal 精确相等——固定 CTP 在 seed 20260835 无净收益
   潜力（与 Gate 39 的 seed 依赖性发现一致），收益捕获判据本轮
   无从检验，只能观察到伤害。
3. 层 1 家族（ackley/elliptic/schwefel）对锚点移动不敏感
   （elliptic 单模态、ackley/schwefel 盆地宽），同样的小窗口
   无害且常有小胜。

**结论与开放决策**：脉冲尺度（pulse_max=32 镜像内核 envelope）在
多模态族上是错误的仪器尺度——要么校准到能系统性修复的量级
（历史 CTP 车道是千级 FE），要么加接受实质性门（微小改进不入
锚）。层 1 证据支持 streak 配置作为 gate40 三族的生产姿态；
rastrigin 族回归如实记录为当前边界。

## Gate 47b 后续修复：dispatch 生命周期与 SMP sense 解耦（2026-08-16）

Gate47b 的失败结果和收据保持不变，但其“主要由微小 incumbent 移动导致”的
解释被同 seed 预算审计修正为更强的结构性根因：`arbitration_only` 被错误地
计入 stall/cooldown/qhat feedback；达到 `stall_cap` 后整个 21-group component
停止 sense，导致约 2.29M FE 从 owner-local proposal lane 转入 MMES tail。

修复内容：

- `arbitration_only` 不再更新 dispatch feedback；
- 新增 `CoordinatorState.sensing_components()`；
- `run_oc_unified` 在 cooldown/stall 期间继续为所有 component 分配 SMP sense，
  但仍禁止其产生新的 operator plan；
- 设计文档同步声明 sense 生命周期与 dispatch 生命周期分离。

同一 `rastrigin/star/ov3/seed=20260835` 因果确认：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| sense FE | 528,507 | 2,818,704 |
| operator FE | 20 | 16 |
| MMES tail FE | 2,291,443 | 1,250 |
| final error | 301.723 | 132.569 |
| frozen kernel | 131.595 | 131.595 |

修复后通过 terminal exact、strict-best、receipt parity、state hash chain，且
统一结果进入 kernel ×1.05 容差。旧 Gate47b 的 18-cell 性能门仍需在新鲜 seed
上另行重跑，不能用这个单 cell 因果确认替代完整 Gate47c。

产出：`artifacts/oc_budget_lifecycle_repair/star_ov3_seed20260835.json`。

## Gate 47g-47i：低价值 pulse 的两级价值门与预算可行性（2026-08-16）

Gate47f 的唯一回归是 `star/ov6` 上 8-FE shared-core CTP 改变 terminal tail
起点。逐周期诊断显示，Gate47f 的 operator 只带来约 `0.005123` 即时改进；
Gate47h fresh seed 上 operator 实际改进为 0，后验 archive 回滚正确，但已消耗的
8 FE 仍形成尾部机会成本。因此仅靠 strict-best 或事后回滚不能解决预算价值问题。

最终修复分三层：

- `arbitration_value_ratio=0.01`：完整候选仲裁已有实质相对改进时，不再同周期派发；
- `operator_value_ratio=0.01`：低价值 operator 完整计费但不写回 archive，反馈 `no_gain`；
- shared-core CTP 最小窗口为 `2 * |scope|` FE；pulse/operator pool 不足时在预留前
  返回 `shared_core_budget_unavailable`，避免结构上不可行的联合修复窗口。

Gate47g/h 的失败产物保持不变。Gate47i 使用新鲜 Layer-2 seed 20260843，协议
6/6、`path_fires_on_chain`、`not_worse_than_kernel_all`、
`star_no_regression_vs_proposal` 全部通过；13/18 个 unified cell 仍有真实 operator，
Layer-2 1 胜 5 平。`random/ov6` unified 为 459.690，对两个 474.744 对照严格胜；
`star/ov6` 与 proposal 在 1e-9 预注册容差内持平。

产出：`artifacts/oc_streak_confirmation_gate47i/confirmation.json`。

## Gate 47i 通过：价值门修复后 streak 配置全绿（2026-08-16）

针对 47b 的"噪声窗口锚点漂移"失败，用户会话实施关键修复：
仲裁后价值门 + operator 后验价值门（contract.py）、低价值 operator
可恢复 archive 但 FE 不退款（ledger.py）、shared-core CTP 最小
2×scope 变量数 FE 预算守卫（planner.py，不足不产生无效 reservation）、
收据新增价值比例与门控状态（loop.py）。

**Gate 47i 判定（全部通过，`oc_streak_confirmation_gate47i/`）**：
- 协议 6/6；`path_fires` = true（13/18 cell 真实派发）；
  `not_worse_than_kernel_all` 18/18；`star_no_regression` 8/8。
- 层 1：12/12 数值平局（价值门把 47b 的噪声微胜也归零——
  非劣性以最保守形式成立）。
- 层 2：5 平 + **1 个真实胜绩——random/ov6 统一臂 459.7 vs
  内核/基线 474.7（−15.05）**：kernel=proposal 精确相等处，统一臂
  通过一次 AOR 升级窗口拿到了两个对照都没拿到的收益——这是统一
  循环首个"超出内核"的实例捕获记录。

**稀疏域状态**：统一循环（streak + 价值门配置）现为已验证生产
姿态——18/18 不劣、零回归、四类动作实际运转、且在对照无潜力的
cell 上展现独立收益。稀疏域收口，主线转向 AOB（Gate 48）。

## Gate 48a 完成：AOB 接线首次端到端跑通（适用性 4/4），两个关键发现（2026-08-16）

`experiments/oc_aob_wiring_pilot_gate48a.py`（R2/A3/S5/R6，seed
20260845，v3 发现 → 超边门 → `run_oc_unified_from_structure`）。

**适用性判定全过**：接线 4/4（超边门转换 + 统一循环无 fail-closed）、
Phase-I 精确 180k / 终端精确 3M、strict-best、收据平价 + 状态哈希链。
发现侧召回与 v3 冻结基线逐 case 一致（1.000/0.947/0.895/0.789），
确定性成立。**统一循环在 AOB 上第一次作为系统运行。**

**遥测结果**（vs 41b case 均值，未配对参照）：
| case | 最终误差 | vs 41b | op_fes |
|---|---|---|---|
| A3 | 7.85e4 | **1.006×** | 0 |
| R6 | 4.96e5 | 1.298× | 0 |
| R2 | 4.10e5 | 1.885× | 0 |
| S5 | 1.14e6 | **284×** | 0 |

**发现 1：sense 吞掉了 93.7% 的 Phase-II 预算，算子被预算饿死**。
`_proposal_budget` 公式从稀疏网格迁移（组均分全部可用 FE），四个
case 的流向完全一致：sense 2.64M（93.7%）+ MMES 尾 ~6%，probe
0.1-0.2%，算子 0——动作计数只有 1-3 次 arbitration_only 和 12-14
次 none（预算池耗尽）。op_fes=0 是预算架构结果，不是信号缺失；
该分配假设不迁移到 AOB。

**发现 2：41b 参照未配对，S5 的 284× 无法归因**。41b 各臂从历史
恢复管线 checkpoint（强 incumbent）起跑；本 pilot 的 Phase-I 边界
是 v3 发现 + 通用 MMES 填充。A3 的 1.006× 说明 sense+tail 架构
本身能接近 ctp 水平；S5 的差距是"循环架构"还是"Phase-I 边界
质量"，必须用**同 checkpoint 配对对照**（本 pilot 边界 + 41b 派发
动作的终端执行器）才能分离。

**Gate 48 设计修正（两项，进预注册）**：
1. 预算重分配：sense 份额设上限（如 50-60%），给算子池留出真实
   预算（脉冲从剩余份额起算）；
2. 配对对照：同 Phase-I 边界跑派发动作执行器，隔离
   循环-vs-动作的差异。

## Gate 48b 完成：预算饿死修复 + 同 checkpoint 配对实验（2026-08-16）

用户会话实施：`capped_proposal_budget`（sense 总预算上限 = Phase-II
的 45%，`overlap_core.py`）+ 配对实验脚本（`oc_aob_paired_gate48b.py`，
对照 = 同一 Phase-I checkpoint 的纯 MMES 终端）。

**配对结果**（同 checkpoint hash，全部契约检查通过）：

| case | Phase-I incumbent | 统一循环 | MMES 对照 | Δ(对照−循环) |
|---|---|---|---|---|
| A3 | 7.92e4 | 78,531 | 78,335 | −196（平局偏负） |
| R2 | 1.20e7 | 555,439 | 2,135,831 | **+1.58e6（3.8× 胜）** |
| R6 | 1.56e7 | 603,613 | 1,319,886 | **+7.2e5（2.2× 胜）** |
| S5 | 1.60e8 | 47,746 | 14,836 | −3.3e4（3.2× 负） |

预算结构已修复：sense 45.0% / 尾 ~55%，四 case 各实际触发 1 次
算子（R2=aor，其余 ctp_restricted），收据平价 + 哈希链成立。

**解读**：
1. **R 族两场大胜是配对证据**：同一起跑线（incumbent ~1.2-1.6e7），
   sense+协调+尾架构把终值带到 5.6e5/6.0e5，纯 MMES 只有
   2.1e6/1.3e6——协调架构在 rastrigin（真实重叠冲突 + v3 最高
   召回）上净贡献 2-4 倍。
2. **S5 是真实的架构性损失**：schwefel 上 45% 的 sense 会话
   主动劣于纯 MMES（14.8k vs 47.7k）——S 族想要的是
   coverage+polish 结构，owner-local 提案会话从该 incumbent 出发
   反而拖累尾部。
3. **算子仍是象征性规模**：每 case 恰好 1 次 8-FE 最小脉冲
   （stall_cap=2 + 无收益即去激活）。预算池现在有钱了，但脉冲
   尺度还没校准——47b 的教训（过小=噪声）在 AOB 上尚未被检验，
   而这里有真实修复目标（R6 有 150 个确认共享变量）。
4. Phase-I 边界质量确认为 48a S5 之谜的主要成分（纯 MMES 也只有
   14.8k，41b 的 4000 来自强 checkpoint + ctp）。

**下一步（Gate 48c 建议）**：脉冲尺度阶梯实验——同配对设计，
pulse_max ∈ {8, 128, 1024}，4 case × 3 配置（对照可复用），量化
算子预算份额与终值关系；S5 的 sense 份额适应性问题同批诊断。

## 48b 差距归因修正：边界因素小，机制因素占主导（2026-08-16）

用户问：为何 48b 统一循环与历史四动作表（24-case 对比）差距大。
读取历史恢复 checkpoint 的 180k 边界 incumbent 与 48b v3 边界对比：

| case | 历史 checkpoint | v3+MMES 边界 | 边界差 | 历史动作终值 | 48b 循环终值 |
|---|---|---|---|---|---|
| A3 | 7.89e4 | 7.92e4 | ~1× | 7.81e4 | 7.85e4（平） |
| R2 | 7.39e6 | 1.20e7 | 1.6× | 2.17e5 | 5.55e5（2.6×） |
| R6 | 1.04e7 | 1.56e7 | 1.5× | 3.82e5 | 6.04e5（1.6×） |
| S5 | 4.86e7 | 1.60e8 | 3.3× | 8.38e3 | 4.77e4（5.7×） |

**结论**：起跑线差距只有 1-3×，终值差距 1-6×——**主导因素是
Phase-II 机制，不是 Phase-I 边界**（修正 48b 日志条目里对边界
因素的权重估计）。历史四动作是从恢复战役中精调的整体作战方案
（ctp = 20% 覆盖 + 关系块逐块打磨 + 交错 MMES 尾；smp = 状态化
访问 + 救援调度；全程 2.82M 交错结构化扫掠），而统一循环的当前
车道是通用原语：45% 默认参数 owner-local sense 会话 + 55% 单次
MMES 尾 + 8-FE 象征性算子脉冲。同起跑线下循环对纯 MMES 净胜
2-4×（R 族），说明协调骨架价值为真；差的是车道还没吸收动作
内部结构。追平路径 = 车道升级（sense → smp 式状态化访问、尾 →
ctp 式交错扫掠、算子 → 真实尺度窗口），或维持 41b 派发为生产
姿态、循环作为架构层。

## Gate 50a 通过：四动作真实规模分段等价性 + 修复两个 v2 恢复 bug（2026-08-16）

收编四动作为 `ActionEpisode` 的正确性前提验证
（`experiments/oc_action_episode_equivalence_gate50a.py`：1000 维合成
问题、20 块 + 邻接关系、总预算 600k（smp 救援段/gcb 原生窗口/ctp
覆盖-打磨-尾换相全部实际运转）、不规则分段模式 + **逐段快照恢复**）。

**判定**：四动作 one-shot vs segmented 的终值/incumbent/消耗
**逐位一致**（ctp 9.12e-27 / smp 9.38e-27 / gcb 1.22e-20 /
aor 8.98e-27 双模式相同），22 个既有 v2 测试无回归。

**修复两个真实 bug（34-FE 玩具测试不可达的路径）**：
1. **全空间段的维度比较**（`phase2_v2.py`）：段 payload 存
   `dimensions=None`，会话 payload 存展开元组，恢复时
   `(0..999) != None` 恒假——凡在全空间段（MMES 尾等）中途快照
   必报 "coordinates drifted"。修为展开形式比较。
2. **全空间段的 population_size 重建**：`_restore_private` 用
   payload 维度判 population（全空间段被误判为块范围 → 12），
   与 `_start_segment` 的 24 不符，报（误导性文案的）
   "budget drifted"。修为与创建侧同判据（segment.dimensions）。

**注册表缺口盘点（Gate 50 范围决策项）**：
- ctp / gcb：生产执行器即历史执行器，v2 齐备 ✓
- smp / aor：**历史四臂矩阵与 41b 的列来自 Recovered 变体**
  （无裁剪 offspring / 零均值 AOR），`recovered.py` 只有 execute、
  无 initialize/resume——收编这两条需要补 v2 包装
  （recovered-aor = 单个可续会话，简单；recovered-smp = 状态化
  访问需要会话级续行）。

## recovered-AOR v2 包装完成 + 关键实证：v2 与 legacy 非逐位（2026-08-16）

**RecoveredAorPhase2State**（`phase2_v2.py`）落地：零均值锚、
DEFAULT_SIGMA、原始 action_seed、24 population——镜像 recovered
executor 的单发语义，只新增 step/snapshot/restore 生命周期；
`RecoveredAorExecutor` 接线 initialize/resume，legacy execute 原样
保留（历史收据路径不动）。

**等价性测试**（`test_recovered_aor_v2_equivalence.py`，3/3 过）：
v2 单发 vs 不规则分段 + 逐段快照恢复——终值/incumbent/state_hash
**逐位一致**（60.1k FE，100 维）。v2/历史回归 32 测试无回归。

**关键实证（影响 Gate 50 设计）**：v2 会话路径与 legacy
`PypopOptimizerPort.run` 单发**不是逐位等价**（同种子同预算下
legacy 1.07e-52 vs v2 9.88e-36——会话手工驱动代际循环并禁用上游
early stopping，轨迹合法但不同）。结论：

1. **Gate 50 配对两侧必须同为 v2 episode**（ARAC-OC 交错 vs 同
   checkpoint 的 v2 episode 单独跑）——机器对等，配对公平；
2. 历史 legacy 列降级为参照线，不作配对控制；
3. 需要一个补充校准检查：v2 单独跑 vs 历史列的实距（决定"≥ 历史"
   叙事还能否直接引用历史数字）。

**剩余缺口**：recovered-smp 的 v2（状态化块访问的会话级续行，
中等工程量）——四动作中最后一条。

## recovered-SMP v2 完成：四动作收编全部就绪（2026-08-16）

**关键实证（vendor 目标函数批形状敏感）**：AOB vendor objective 对
同输入行，批式调用与单行调用返回值差 ~1e-14（约半数行位不同；≥2 行
子批与全批位同）。推论：任何"拆开一代逐行求值"的设计都不可能保持
与批式 legacy 的逐位等价。

**设计定案（代对齐步进契约）**：episode 永不拆分一代——所有求值都
是整代批调用（与 legacy 完全同路径）；`step(budget)` 消耗不超过
budget 的最大对齐量并在 `step_fes` 如实上报（部分消耗是契约不是错
误）；低于下一单元的预算显式报错、调用方加大重试。

**实现**：`_PersistentBlockSession` 增加 prepare/evaluate_batch/
finalize 三段拆分（advance 重组为三段组合，行为不变——锁步测试
19/19 恢复通过）+ state_dict/restore_state_dict（x/steps/fitness
为草稿区：仅在有意义时序列化，未初始化内存不进入快照）；
`RecoveredSmpPhase2State`（访问循环重构为可暂停状态机：precheck/
整代/noop 单元、streak/cold_start/restart/扫描位置全量入快照）；
`from_phase2_snapshot` 增加 allow_out_of_bounds 透传。

**等价性测试**（`test_recovered_smp_v2_equivalence.py`，2/2）：
- v2 单发 vs legacy execute **逐位一致**（含 route 字符串——
  口径修正：legacy 的 visits FE 含 precheck）；
- 任意分段 + 逐段快照恢复 vs 单发**逐位一致**。

**收编进度板（全部就绪）**：ctp ✓ / gcb ✓ / aor(recovered) ✓ /
smp(recovered) ✓（smp 与 legacy 双逐位，最强）。全量回归 99 测试
过（动作层 + 协调层 + 历史锁步）。

下一步：Gate 50 正式协议——ActionEpisode 调度层 + 四条带 ε 配对
"ARAC-OC ≥ episode"判据 + 交错上行项。

## Gate 50 判定：未过（R2/R6 探索失败），机制与收据全绿（2026-08-16）

20 臂全并行（Phase-I 缓存复用、逐臂容错），零失败、协议 5/5、
收据/终端/strict-best 全部成立。运行期修复：sigma→+inf 在
ResumableOptimizerSession 中改为收敛至下限 + 修复计数（legacy
车道的同类数值事件一直以裁剪存活；v2 首次在 AOB 3M 规模暴露）。

**配对结果**（oc_schedule vs 四 standalone episode，同 checkpoint）：

| case | ctp | gcb | smp | aor | oc | oc/best | 判定 |
|---|---|---|---|---|---|---|---|
| A3 | 7.83e4 | 7.84e4 | 7.87e4 | **7.82e4** | 7.87e4 | 1.007 | 过 |
| R2 | 5.73e5 | 3.46e5 | 3.94e5 | **2.21e5** | 4.22e5 | 1.907 | **败** |
| R6 | 1.40e6 | 4.34e5 | 1.03e6 | **3.17e5** | 4.68e5 | 1.476 | **败** |
| S5 | **1.32e4** | 7.07e5 | 9.90e5 | 7.95e5 | 1.32e4 | **1.000** | 过 |

**归因（调度轨迹收据）**：
1. **S5 = 完美样例**：初始选择对（ctp），粘性规则让 ctp 连跑全部
   10 段全 material，oc 与最优 episode **精确相等**（零开销性质
   实证）。
2. **R2/R6 失败 = 探索失败，不是协调机制失败**：两 case 的最优
   episode 都是 aor（v3 checkpoint 上的排名：aor > gcb > smp > ctp
   ——与 41b 历史checkpoint 上的 gcb 优选**不同**，初始选择按 41b
   表冻结于是选错）；切换目标按"最低 stall + 固定顺序"选取，
   aor 排序最后 + 粘性规则阻止后期探索 → **aor 全程只分到 1-4 FE**，
   最优路线从未被探索。
3. A3 四 episode 收敛接近（ackley 平台），+0.7% 在容差内。

**设计结论（Gate 50b 输入）**：
- 非回归下界的可行路径 = 探索保证（portfolio bandit）：先给每个
  episode 一个强制探针段（4 段轮转），再按单位 FE 对数增益粘住
  最优——R2/R6 上 aor 的首段增益可见，该规则可发现它；
- 零开销性质已证（S5），机制层（收据/切换/镜像账本/stuck 处理）
  无需改动。

**运行纪律沉淀**：开发迭代用降规模预算（~600k 总量，速度 ×5），
3M 只留最终登记表；Phase-I 按 case 落盘缓存已成惯例。

## Gate 50b 600k screening 通过：探索保证 + 全局 materiality 双双实证（2026-08-16）

按用户五项约束实现（`episodes.py` v2）：双增益（global archive 差
独占 materiality/调度权，local 仅诊断入收据）、counted B/W/C 感知
只做探针排序与 scope（无阈值门）、四 episode 强制可执行探针
（不足即响亮失败）、收据新增 `dispatcher="gcb_coordinator"` /
`episode_kind`（gcb→gss，历史收据名冻结）。三个定向测试 4/4 +
等价性回归全绿。600k screening（4 case，~10 分钟）**20/20 检查
通过**。

**三条成功判据的实证**：
1. **AOR 真窗口 4/4**：每 case 42k FE 探针（Gate 50 是 1-4 FE）；
2. **假粘性被杀死（真实数据物证）**：R2/ctp 探针 local_gain=0.746
   但 global_gain=0.0000——私有进步、全局零贡献，被正确判为
   零收益并失去调度权（Gate 50 的失败模式），R2/gcb local 1.73/
   global 1.52 双正 → 正当选为 exploitation 对象；
3. **预算集中**：R2/S5/R6 switches=0，最优探针 episode 拿走
   exploitation 的 ~70%；A3 平坦地形 switches=5 分布式。

**诚实发现（3M 前必须知道）**：AOR 在四个 case 的 42k 探针上
global_gain 全部为 0——fresh 零均值重启是晚熟策略，短窗口
gain/FE 排序系统性低估它。探索已保证，但"AOR 能否被**选中**"
要看 3M 的 282k 探针窗口（0.1×2.82M）。同时声明：4×0.1 探针 =
40% 探索开销是发现保证的价格；若 3M 配对 ε 判据因此失败，
probe_share 校准是预注册的下一杠杆（不回调其他规则）。

术语落地：收据已带 dispatcher/episode_kind 字段，GCB=协调器、
GSS(gcb)=episode 的区分进入全部新协议文档。

## Gate 50b 审查修复轮：scope 机制 + 审计强化 + 复筛通过（2026-08-16）

用户审查五项（P1×4 + P2×2 + P3）逐项处置：

1. **scope 进入 episode 执行（P1）**：`evidence_block_order`（块按
   感知分降序、稳定平局）→ `dataclasses.replace` 生成 scoped
   checkpoint 供 CTP/GSS（SMP/AOR 为全局组合，声明不受 scope
   约束）；`block_scores`/`scoped_checkpoint_hash` 入收据。
   当前语义 = "证据 → 块序 → 扫掠先搜哪里"，尚非变量级范围。
2. **B/W vs B/W/C（P1）**：定为 **B/W 排序、C 仅诊断**（用户选项
   一），docstring 明示；C 幅值不参与任何门或排序（Gate 47）。
3. **顺序探针语义（P1）**：声明为 **sequential contextual
   scheduler**——四探针相对同一演化中的全局 archive 计"上下文
   边际贡献"，非同基线能力估计；调度相关量即边际贡献，双增益
   字段保持可审计。
4. **生产入口（P1，按用户排序留最后）**：`run_arac_oc` 未切换，
   双轨入口如实声明；50c 验证后再接。
5. **探针税（P2）**：`probe_share ∈ (0, 0.25]` 校验 + 前端可负担
   性检查（税后须余可执行 exploitation 预算）；
   `probe_tax_fes`/`exploitation_fes` 入结果与 cell。
6. **审计强化（P2）**：`_audit_receipts` 重算 schedule_hash、
   段索引连续、探针先于 exploit、全局误差单调、FE 对账
   （phase1 + sensing + funded == terminal）、state hash 链。
7. **Ruff（P3）**：三个未用 import 清除，`ruff check` 全绿。

单测扩至 6 个（新增 `_probe_order` 三政体响应 + 块排序/长度
校验；饥饿测试改为接受前端税检查或探针协议两类响亮失败）。

**600k 复筛（新代码）20/20 通过**，含哈希重算审计。数值（筛选
规模，非 3M 性能）：A3 79,154 / R2 492,973（较上轮 -14%）/
R6 1.98e6 / S5 4.20e7——scope 排序改变了 CTP/GSS 扫掠起点，
R2/R6 受益。AOR 探针仍 42k×4 真窗口；无假粘性。

## handoff 机制闭环 + 600k 冒烟/消融完成（2026-08-16）

按用户规格实现（`episodes.py` v3）：

- **双概念分离**：archive handoff（`adopt_global_archive`，严格单调
  + 私有 FE 不变断言，返回 (adopted, refusal)）与 search-state
  handoff（CTP/GSS 下一 segment、SMP 下一 visit 经私有账本锚点
  自然生效，episode 内部零改动）。
- **AOR = fresh_by_design**：接收 archive baton（local_gain 获得
  相对 x* 的真实边际基线）但搜索分布保持零均值；单测证明注入
  前后 AOR 会话 state_hash 逐位不变。
- **越界边界策略（预警第 3 条，实际触发）**：全局最优可为 SMP
  无裁剪 offspring 的越界点；有界锚机制（CTP/GSS）拒绝越界
  baton（refusal="oob_incumbent"，等下一个界内 x*），SMP/AOR
  接受。600k 实测：R2 拒 2 次、R6 拒 3 次、S5 拒 2 次——
  预警完全命中，A3 无拒绝。
- **收据**：EpisodeHandoffReceipt（from/to/mode/adopted/refusal/
  incumbent_error/from/to snapshot_hash），segment 收据补
  snapshot_hash（覆盖 archive 状态）；哈希域升 v3（含 handoffs/
  探针税/scoped hash），审计同步。
- **消融开关**：handoff_enabled=False 时全链记 disabled/adopted=
  False。
- 单测扩至 10 个（注入单调性/FE 保持/OOB 拒绝、接力链、disabled、
  AOR 独立性）。

**600k 配对结果（同 seed）**：

| case | handoff ON | OFF（消融） | Δ |
|---|---|---|---|
| A3 | 79,152.4 | 79,153.9 | 平 |
| R2 | 847,551 | **492,973** | ON 差（接力误配 smp>gcb） |
| R6 | **1,484,460** | 1,980,210 | ON 好 −25% |
| S5 | **3.382e7** | 4.199e7 | ON 好 −20% |

两臂各 20/20 检查通过。机制结论：handoff 真实改变调度与轨迹
（R6/S5 的 exploitation 对象因此改变），600k 规模下 2 好 1 差
1 平——符合"600k 验证机制非性能"的定位；R2 的回退由收据归因
（baton 使 smp 探针边际增益变正 → 误配），3M 上 aor 才是 R2
最优，此误配与 3M 判定无关但记录在案。

## Gate 50c 3M 六臂判定：未整体通过，但 R6 拿下首个完整的"组合 > 全部单臂"因果证据（2026-08-16 夜）

`oc_action_episode_gate50c`：4 case × 6 臂（4 standalone 复用 Gate 50
收据 + OC handoff ON/OFF），同 checkpoint（哈希核验一致）、同
action seed、精确 3M。协议层全过、零臂失败。

**最终表**：

| case | 最优 standalone | OC-ON | OC-OFF | 判定 |
|---|---|---|---|---|
| A3 | aor 78,180 | 78,517 (×1.004) | 78,244 | 不劣过 |
| R2 | aor 221,157 | 422,018 (×1.91) | 434,065 | **不劣败** |
| S5 | ctp 13,243 | 1.67e6 (×126) | 1.27e6 | **不劣败** |
| R6 | aor 317,086 | **276,031 (×0.870)** | 318,413 | **严格胜** |

gate_passed = false（性能层 2/4）。

**R6 = 首个完整组合证据（互补层全过）**：
- 严格优于全部四条 standalone（0.870×）；
- ≥2 episode material：smp（探针 2.05 + 6 个 exploit 段全 material
  累计 0.84 对数增益）、ctp（0.13）、aor（1.01）——三路贡献；
- ON 276k < OFF 318k（关 handoff 优势消失，因果成立）；
- 接力链收据可见：smp 探针大幅领先 → gcb/ctp 交接被 OOB 拒绝
  （smp 无裁剪 offspring 的越界最优，边界策略正确工作）→
  ctp→aor baton 采纳（fresh 探针拿 1.01 增益）→ aor→smp 回棒
  → smp 六段连续 material 放大收益。
- 假粘性对照：gcb 探针 local 0.85 / global 0.00——被正确判非
  material 并失去调度权。

**R2/S5 失败归因（同一根因，比 R2 条款更普遍）**：
短窗口 gain/FE 排序系统性低估晚熟 episode——R2 的 aor（fresh
重启晚熟，42k/282k 窗口都探不出）与 S5 的 ctp（全长打磨才兑现，
探针期让 gcb 的即时增益胜出，exploitation 锁死 gcb）。S5 上 ON/OFF
同错（都选 gcb）证明这是探索排序问题而非 handoff 问题。未来方向
（预注册，不回调现有规则）：晚熟动作保护（预留 correction tail /
多时间尺度收益估计）。

**夜间数值健壮性修复（三次溢出调查）**：
1. sigma→+inf：会话层收敛至下限（已有）；
2. sigma 巨大但有限（~1e200+）：新增 sigma 上限 = 活跃跨度
  （计数修复）——不够；
3. 根因（实测）：AOB vendor 变换在 |x|≥5e4 即 NaN（R2/Tasy 指数
  含 sqrt(z)，幂次爆炸），远早于 float 上限。`_EpisodeLedger` 加
  幅值护栏 |x|≤1e4（公共界 ±100 的 100 倍，四 case 实测有限；
  NaN/inf 先 nan_to_num 再裁剪；修复计数入收据——R2/off 实测
  smp 14 次修复后正常运行至终端）。1-D 转发 bug（裁剪后转发了
  未裁剪原数组）在第二次调查中修复。

**生产入口**：未接入（预注册门控：50c 通过才接；性能层未过，
等待用户裁决——R6 证据 + R2/S5 归因已足够支撑方向决策）。

## v4 升级计划定稿 + Gate 51-0 启动 + 调度器 v4 全量落地（2026-08-17）

**v4 升级计划定稿**（`docs/arac-oc-v4-upgrade-plan.md`，两轮评审并入
12 条修正）：Gate 51-0 显现 horizon 测量前置（先测后冻结）；R2 饿死
修复四件套（ticket 防淘汰 + 递增开发窗防看不见 + 轮转下限防饿死 +
私有轨迹信用仅升权）；R6 回归预注册归因 + challenger 非对称杠杆；
预算双账本（cold_start 25% / exploration_and_development cap）；
protocol_mature 与 evidence_revealed 概念分离；调度裁决 P0-P5 唯一
答案；版本与确认隔离（确认后改动 → v4.1，fresh seed 才能确认）；
判据消歧（严格胜 ≤0.98×、median ≤1.05 + 最差 seed ≤1.10、新增
Gate 51d 泛化层）；定位表述改为 block-order scope 协作超启发式。

**Gate 51-0 注册并启动**（`docs/arac-oc-gate51-0-protocol.md` +
`experiments/oc_horizon_gate51_0.py`）：aor/R2 与 ctp/S5 两个 3M
分段 instrumented 重跑（150k × ~19 段，每段误差轨迹 + state hash），
终值与 Gate 50 standalone 收据逐位对账；参考水平取 50c ON cell
其它臂探针期末误差。启动前冒烟确认：R2 参照水平全部收敛于
512,640（smp 探针后 archive 位）——正是 aor 轨迹要穿越的目标。
两个 3M 跑后台运行中。

**§3 进度契约落地**：`runtime/phase2.py` 新增公共 `EpisodeProgress`
（episode/phase/consumed_fes/next_boundary_fes/min_step_fes/
maturity_target_fes/protocol_mature/contract）+ 基类默认 progress()；
四个可调度 state 各自覆写——CTP coverage→polish（边界未成熟语义）、
GSS warmup→coordination/continuation、SMP 代对齐（min_step = 最小
种群+1，visit 边界）、AOR 单 regime 窗口。调度器零第二套映射。

**§4-6 调度器 v4 落地**（`episodes.py`，v3 原样保留为历史路径）：
`PhaseAwareSchedulerConfig`（w1/h*/开发上限/保底四参数无默认值，
强制来自 51-0 定标表）；P0-P5 裁决（探针硬约束→可负担 ticket→
leader exploit（max-2 连续）→强制 challenger（(a) evidence 待显现
递增窗、私有信用排序、cooldown 豁免；(b) 非 leader 轮转，cooldown
过滤，最早到期兜底）→排序信号→fallback）；双账本计账（授予时刻
leader 身份归类，dev 满时强制 challenger 如实溢出记 exploitation
账本）；私有信用（极值/速率两模式，仅升权，handoff epoch 换纪
重置）；AOR ticket 双窗保护；13 项收据审计（递增几何阶梯、垄断、
ticket 后饥饿、双账本上限、FE 对账、哈希可重算等）。

**实现期修复的真实 bug（toy 对账发现）**：switched 标记在
previous_episode 更新后计算恒 False；cooldown 先置 1 又被同次
授予减回 0（回避期失效）；AOR ticket 只给 1 窗（_ticket_size 误用
progress 单窗目标）+ protocol_mature 单窗即 True 导致双窗条款
被跳过（引入 _ticket_complete）；challenger 授予绕过账本检查击穿
dev cap；leader 存在但无 ticket 完成集合时角落兜底按 rate 连选
同一臂（Gate 50 饿死模式复活，垄断审计正确拦截）——改为 leader
存在时全非 leader 轮转，真 600k 角落才 rate 排序 + 自排除。

**测试**：新增 18 个定向测试（12 调度器 + 6 进度契约）全过；全量
回归 564 过 / 4 败——4 个失败为预存的隔离类测试互染（要求进程内
零 arac 预加载，与本次改动无关，基线核实）。Ruff 全仓绿。

**Gate 51a 脚本就绪**（`experiments/oc_phase_aware_gate51a.py`）：
600k × 4 case × handoff ON/OFF 机制筛选；配置强制从
`artifacts/oc_horizon_gate51_0/calibration.json`（status=frozen）
读取，缺表/未冻结响亮失败——参数纪律的代码级执行。机制检查含
progress 表面、ticket/不可负担记录、递增窗、垄断、exploit 只发给
leader、CTP 覆盖段不成熟、handoff 拒绝语义、manifest 一致。

**等待**：51-0 两个 3M 跑完成后做 horizon 定标分析 → 冻结
calibration.json → 跑 51a。

## Gate 51-0 通过并冻结定标（2026-08-17）

两 run 终值与 Gate 50 standalone 收据**逐位相等**（aor/R2 221,181.93、
ctp/S5 13,244.70），terminal 精确、轨迹单调、FE 对账精确，
gate_passed = true。

**核心测量**：两晚熟轨迹都在 (150k, 300k] 累计区间穿越探针期
archive 参照线，且 150k 时都差于参照（aor 2.14e6 vs 5.13e5；
ctp 6.9e7 vs 4.4e7）——v3 的 282k 探针失败由数据完全解释：
累计窗口不足，非估计器问题。

**manifest 竞态（诚实记录）**：两 run 执行期间 v4 调度器同仓实现，
执行时点 manifest 不可复原；cell 统一盖跑结束时戳，数值有效性由
逐位对账锚定（强于 manifest 戳）；脚本改为 main 单点计算传
worker，判定改为"各 cell 戳一致"必须 + "匹配当前"信息字段。

**冻结定标**（calibration.json，ref=gate51-0-20260817）：h*=450k
（300k 实测 × 1.5 安全系数，覆盖粒度与 AOR contextual 偏移）、
w1=75k、r=2、K=3（累计 525k）、开发上限 0.55（R2 全修复路径
1.54M ≤ 1.55M 算术验证）、保底 0.10。推导与局限入协议附录。

## v4 首轮 51a 暴露并修复的两个实现缺陷（2026-08-17）

1. **终端尾部死锁**：剩余 FE 低于 SMP 代对齐最小步长（~13）时
   裁决器仍选 smp → Phase2StateError 炸场（R2/on、R6/on 复现）。
   修复：裁决各层（P2/P3(a)/P3(b) 两池/角落/P5）加 min_step_fes
   可执行性过滤；_execute 对不可执行授予标记 stuck、无收据返回、
   重新裁决（v3 验证过的模式）。回归测试：代对齐臂不得收到
   低于执行单元的窗口（两预算场景）。
2. **收据 leader 字段时序**：原实现在本段收益入账后计算 leader，
   本段自己的增益可能改变排名 → A3/on 的"exploit 只发给 leader"
   检查误报。规格语义是"授予时刻"：_execute 在执行前捕获
   leader_at_grant 传入收据。修复改变收据与 schedule_hash →
   8 cell 全部按版本纪律重新生成。

51a 门控同步修正：aor_two_window_ticket 降为信息字段（计划明确
600k 只保证最小探针；AOR 双窗验收在单测与 51b）。

## Gate 51a 通过：v4 机制筛选全绿（2026-08-17 夜）

8/8 cell（4 case × handoff ON/OFF）零失败；80 项机制检查全过；
三个门控标志全部行使——escalation 递增窗、不可负担 ticket 记录、
强制 challenger；aor 双窗为信息字段（600k 结构性不可负担，符合
计划"600k 只保证最小探针"）。

R2/on 机制故事（600k）：smp ticket 两段完成（54,989+16，代对齐
部分消耗→P1 续给，maturity 机制按设计工作）；ctp ticket 97k
完成；gcb（159k）/aor（150k）ticket 不可负担如实入收据；随后
exploit/smp + escalation/ctp + challenger/aor 交错——四条机制
车道（ticket/exploit/escalation/challenger）在同一 run 全部出现。
handoff 8 次（4 采纳，oob_incumbent/not_better 拒绝语义合法——
50c 预警的 OOB 边界策略在 v4 下继续正确工作）。

筛选规模数值（不构成性能结论，仅记录）：ON 在 R2/S5/R6 均优于
OFF（1.26e6 vs 1.84e6；8.38e6 vs 9.64e6；1.80e6 vs 1.97e6），
A3 持平。同尺度 v3 对照（50b 600k）：S5 8.38e6 vs v3 的 3.38e7。

产出：`artifacts/oc_phase_aware_gate51a/`（cells + confirmation.json，
manifest 全一致）。

**下一步（51b，预注册）**：重新生成四个 standalone（含分段轨迹
收据）+ v4 ON/OFF，同 checkpoint、同 seed、精确 3M；判据 4/4
非劣（≤1.05×）+ ≥1 严格胜（≤0.98×）+ 严格胜 case 双 material +
ON/OFF 条款 + R6 型归因清单（challenger 段 material 分解）。

## Gate 51b（v4.0）失败归因与 v4.1 修复（2026-08-17 深夜）

**v4.0 结果**：24 cell 中 16 个 standalone 全部逐位对账通过；OC 侧
R2/on、R6/on、A3/on 完成但性能失败，A3/off、S5/on、S5/off 被
`no_post_ticket_starvation` 审计响亮拦截（三 cell 未落盘）。
R2 433,870（1.96×）；R6 401,342（1.27×，v3 曾 0.870 严格胜）。

**归因（收据证据）——两个真实设计缺陷**：

1. **强制 challenger 被绕过（全局 streak 缺失）**：max-2 按"同一
   leader 连续"计数，rate 重选让 leader 每 2 段换人（R2 实测流：
   smp,ctp,ctp,smp,smp,gcb 全 exploit）→ P3 从未触发 → aor 的
   递增窗从未运行（R2 上 aor 仅 ticket 130k，距 300k 穿越线甚远）。
   S5 的审计失败正是同一饿死模式——审计正确，裁决梯实现错误。
2. **ticket 收益污染 leader 选举**：ctp 337k ticket 贡献 gain 2.27
   进入 recent_rate → ctp 当 leader 连跑 2×300k exploit 全零增益
   （R2 白烧 600k）。ticket 是协议单元，不是速率样本。

**v4.1（版本隔离流程）**：
- exploit_streak 全局计数：任意混合的两连续 exploit 段强制开放
  challenger（计划规则 3 的本意）；
- rate 历史排除 ticket 段（probe/exploit/challenger/escalation 计入）；
- 新增全局 streak 定向断言；19/19 过；
- 重过 51a 级 600k 复筛后重跑 51b OC 8 cell（16 个 standalone
  cell 逐位有效，保留复用）。

**v4.0 的教训入档**：机制筛选（51a）未覆盖"leader 轮换绕过
max-2"的路径——600k 预算下 ticket 占主导、exploit 轮次少，该
缺陷只在 3M 的充足 exploit 阶段显形。审计链（no_post_ticket_
starvation）按设计拦截了它。

**v4.1 第三处修订（评审预判采纳）**：51b 复盘指出私有信用排序可能
让高信用臂耗尽开发上限、低信用 AOR 永不开梯。P3(a) 改为最少递增
授予优先的公平轮转（credit 仅平局打破）——Successive Rejects 式
公平，显现即退出候选集。公平性定向断言（待显现臂含零授予者的
授予数差 ≤1）20/20 过。审计判据同步修正：被拒 ticket 允许更小
余量的合法重试，禁止更大尺寸绕账。

**P0 拦截（评审）**：51b 脚本原会按"存在即跳过"把 v4.0 的 5 个
OC cell 混入 v4.1 判定。修复：输出目录按 scheduler 版本隔离派生；
cell 复用强制校验（OC 臂查版本+调度树 manifest；standalone 臂查
standalone 依赖 manifest——调度器代码不在其执行路径）。v4.1 目录
24 cell 全量重跑。

## 静态审查 P0-P2 修复轮（2026-08-17 深夜二）

评审拦截 8 项，全部处置（51b v4.1 首次启动被主动终止——两项修复
改变调度行为，跑出的结果不可作确认）：

1. **[P0] 51a 版本隔离**：输出目录随 scheduler 版本派生；cell 复用
   强制校验（版本 + manifest + 定标文件哈希三方一致）。
2. **[P1] reserve 真实化**：所有非 exploit 授予执行前经
   `_challenger_reservation` 预约（开发余量 + 保底双重钳制，小窗
   开道而非穿透）；终端排水/无 leader/车道耗尽态的保底释放为
   **裁决期声明**（exploitation 账本），非事后改记；记账三分支
   （cold_start / development / exploitation）+ 双防禦断言。顺带
   修复 v3 遗留的 step 倍增重试（会消费超预留窗口）——改为暂态
   阻塞（min_window_needed），仅请求达剩余上限仍败才永久 stuck；
   以及 remaining 双减算术错。
3. **[P1] material 归因收紧**：无采纳 handoff → 空集；排除 probe；
   要求 handoff_epoch > 0。
4. **[P1] escalation_grants_k 接入**：config 字段 + 校验 + P3(a)
   边界 + config_payload（schedule hash 覆盖）+ 51a/51b 加载。
5. **[P1] 两处恒真断言删除**，改为真断言（funded+sensing 对账；
   exploit 只发给 protocol_mature）。
6. **[P2] 审计补四项**：receipt 链连续性（精确等）、全局 streak
   （含两个压力阀例外、收据可重建）、escalation 公平（含 leader
   豁免）、development 保底保持；垄断审计加强制态豁免（终端/车道
   耗尽时的被迫连跑合法）。
7. **[P2] 文档版本清理**：51b 协议与升级计划的 v4.0 字样更正。

测试 22 个调度器/契约定向全过；全量回归 568 过 / 4 败（预存隔离
类，零新回归）。流水线重启：51a（版本隔离新目录）→ 51b（守卫会
自动拒绝被杀运行留下的旧 manifest cell）。

## 51a 复筛卡死：P1 ticket 空转循环（2026-08-17 深夜三）

症状：600k 复筛 6 个 worker 各烧 ~59 CPU 分钟只完成 2/8 cell
（正常 ~20 分钟/cell）；完成 cell 仅 9 授予，排除授予爆炸。

根因（代码机制确证）：上一轮把 v3 的 step 倍增重试改为暂态阻塞
（min_window_needed）后，P1 ticket 授予没查该阻塞——SMP 的 ticket
余量窗口低于其当前 visit 单元时：step 失败 → 阻塞记录 → P1 以同
尺寸重试 → 永久空转（无收据、预算不动、CPU 烧在裁决循环）。
1000 维 AOB 的块种群大于 toy，toy 测试未覆盖该路径。

修复三处：_ticket_size 纳入有效执行下限（min_step 与暂态阻塞的
max）；P1 授予前加 _executable 闸（不可执行则留待更大窗口）；
主循环加 200 次连续执行失败熔断（未来任何空转响亮化）。22/22
测试过，cell 清空重跑。

## 差分实验反转 + 速度问题最终归因 + 4 cell 全预算复现模式（2026-08-17 夜）

**差分实验**：gcb/ctp @ R2 在 300k 预算下整预算 = 75k = 37.5k 分块
**逐位一致**——"数值混沌分块分岔"假说在 300k 尺度被否证；全尺度
（2.82M）的分岔机制待后续定位（可能深藏于 sigma 溢出恢复段）。
处置：R2/R6 的 ctp/gcb 四 cell 改用 **Gate 50 同款整预算单步模式**
（FULL_REPRO_CELLS，同代码路径确定性保证逐位；轨迹收据降为单段，
协议声明此限制）。分块校验按 cell 感知 expected_segments；
standalone manifest allow-list 扩为全部冻结历史值（有效性由逐位
对账锚定，manifest 只记生成代）。

**速度问题最终归因**（修正早先的 E 核单一归因）：主因是 51a 空转
run 的 8 个僵尸 worker 与 51b 重叠 2h13m（已修 + 熔断）；次因是
20-24 worker 过饱和。P 核钉定 + 8 worker 实测恢复：44-58 分/OC cell
（优于 Gate 50 的 ~60 分）。

**早期读数（4/12 OC cell）**：S5/on 572,420（v3 的 1.67e6 好 2.9×，
ON 较 OFF 好 2.7×）但仍为 ctp standalone 的 43×——指向 max-2 交错
打断 ctp 全长打磨（预注册 v4.2 杠杆的用武之地）；R2/off 331,666
（1.50×，v3 1.96×）；A3/off 1.003× 达标。全部已完 cell 13 项审计绿。

## 会话暂停（2026-08-17 深夜，用户关机）

**中断原因**：用户离开关机。全部后台任务已停、worker 已清；cell 逐个
原子落盘，已完成的全部保留。

**51b 磁盘状态**：
- OC（v4_1 目录）：R2_off、S5_on、S5_off、A3_off 已完成且审计绿；
  **缺 R2_on、R6_on、R6_off、A3_on**（管道事故两次丢失在途进度，
  教训入档：长跑后台命令禁止套 head/grep 管道——head 退出触发
  BrokenPipeError 杀死父进程，池瘫痪只剩孤儿 worker）。
- standalone 共享根：14 cell（12 逐位有效 + R2_ctp/R2_gcb 为分岔值，
  将被守卫拒绝）；缺 R6_ctp/R6_gcb。
- 下次续跑一条命令（复用守卫自动只跑缺失 8 cell；R2/R6 ctp/gcb
  自动以全预算单步模式重生成）：

  .venv\Scripts\python.exe -m experiments.oc_phase_aware_gate51b --workers 8 --pin-p-cores

  预计 ~1-1.5 小时出全部数据与三层判定。

**今晚早期读数**（4 OC cell）：S5/on 572,420（v3 好 2.9×，ON 较 OFF
好 2.7×，但仍 43× 未达非劣——指向 v4.2 challenger 非对称杠杆）；
R2/off 1.50×（v3 1.96×）；A3 1.003× 达标；审计全绿。

## 性能优化批次落地：vendor 2.4× + 快照/轨迹轻量化（2026-08-17）

**#3 vendor 热循环预计算**（Benchmarks.py）：transform_asy 系数缓存
（消除每 FE 的 arange+repeat+掩码索引）、elliptic 指数权重缓存、
Lambda 指数缓存、rotateVectorConform 索引缓存。运算顺序不变。
**验收**：4 case × 1/8/24 行 + 单行，与补丁前黄金参照逐位一致；
永久守卫测试 tests/test_vendor_perf_patch_bitwise.py 锁定；
金样与 README 存 artifacts/vendor_perf_patch_20260817/。
**提速实测：2,713 → 6,491 FE/s（2.4×）**。

**#1 fitness_record 默认关闭**（record_fitness 开关，四个函数文件）：
省 ~100MB/worker；AOB 相关 24 测试过。

**#2 retain_trajectory 接线**：机制（ActionContext 字段 + 快照/restore
适配 + v4 调度器关闭）由用户并行完成；本轮接上 51b standalone 入口。

**#4 snapshot_hash 缓存**（contracts.py）：property 改为计算一次缓存
（frozen dataclass 经 object.__setattr__），值不变。

**campaign manifest 有记录再冻结**：vendor 树哈希变更触发历史重放的
防篡改守卫（test_current_replay_plan_binds…）——按规范修订
outcome_calibration_v5/train/campaign_manifest.json：新树哈希 +
amendments 记录（原因/原哈希/证据指针）+ 自哈希重算。守卫恢复绿。

回归：568+3 过 / 4 败（预存隔离类零变化）。#5（trusted scalar 入口）
按测量否决（ledger 自身开销 0.015s/42s）。

## R6/on 饥饿审计拦截 → 轮转下限缺席计数实现（2026-08-17 深夜四）

51b 优化后重跑中 R6/on 被 `no_post_ticket_starvation` 响亮拦截。
诊断重跑（审计旁路）定位：**gcb 被 P3(a) 结构性饿死**——其 ticket
有 material 增益（0.567/619k），但 ctp/aor 长期待显现使每个强制
challenger 事件全走递增道，P3(b) 轮转从未触发；且预算恰在第三个
外来 challenger 段后耗尽，审计按定义判饿死。ON 终值 374,998 还差于
OFF 的 324,825——组合被饿死直接伤害。

**修复**：计划规则 4b 的缺席计数实现——`challenger_absence` 追踪每
臂自上次 challenger 类授予以来的连续缺席；缺席 ≥ N-1（=2）的非
leader 臂在下一 challenger 事件**强制插队**（优先于 P3(a)），触发
阈值取 N-1 保证返回落在 N 窗口内（预算尾部恰差一拍的场景）。
22/22 测试过；51b 全量重启（OC 8 cell 全部重跑——调度器 manifest
变更，v4.1 仍为预确认迭代）。

## v4.2：三项语义修正 + 尾部护栏（2026-08-17 深夜五，用户关机后自主推进）

51b v4.1.1 结果（5 OC cell）+ 用户深度归因确认：S5 43×（CTP 长程
被饿）、R2 1.50×（AOR 显现即终局）、R6/off 1.024×、A3 1.003×。
四机制问题：protocol_mature≠就绪但被当就绪；ticket 收益一刀切
排除；exploit 不计入显现；handoff 清零私有信用。

**v4.2 修正**（三项语义 + 一护栏，不动 K/max-2/阈值）：
1. evidence_fes = 实际运行 FE（funded，全授予类型）——收据新增
   cumulative_runtime_fes，evidence_revealed 读它；
2. 信用折扣承载：adoption 时 private_credit ×0.5 而非清零（基线
   平移的证据打折保留，S5 的 CTP 2.21 不再归零）；
3. 信用续展道（credit continuation）：已显现但私有信用 > material
   阈值的臂以 challenger 类窗口继续开发（语境 horizon 平移的
   补偿，S5 型长程动作的专属通道），排在 pending 梯子之后；
4. 尾部护栏：challenger/escalation 窗口钳制保留一个续展片
   （min(w1, remaining/2)）——R2 的"最后一段发现赢家"不再发生。

CTP handoff 重锚问题机制澄清：重锚只发生在段边界，polish 单段
2.4M 内优化器状态从不重锚——S5 伤害来自信用清零+预算不足。
HCC 24 函数参照提取入 references/hcc_aob24_reference.json
（gate case 快照：R2 HCC 3.72e5 vs 我 AOR 2.21e5 优；R6 8.15e5
vs 3.17e5 优；S5 9.23e3 vs CTP 1.32e4 近；A3 7.86e4 vs 7.82e4 近）。

测试：调度器 22/22；全量 563 过零失败（除 4 个预存隔离类）。
51a v4.2 复筛启动。

## Gate 51b（v4.2）判定失败 + v4.3 综合（2026-08-18 凌晨）

**v4.2 结果**（12 cell 全落，A3 1.007✓；R2 1.79✗ / S5 43.2✗ /
R6 1.35✗ 且 ON>OFF 1.32✗；无严格胜）。四个全预算 cell 仍产出分岔
值 → **分岔与分块无关，是依赖树本身**：optimizers.py 的 sigma 上限
安全修复晚于 Gate 50（50c 夜间），溢出 cell 的 Gate 50 值是安全
修复前产物；现值全部更优（R2/gcb 295,663 vs 345,712 等）。四个
cell 的参照锚定需按"当前树全预算值"修订（待晨间决策）。

**收据归因（S5/on + R6/on 逐段）**：
1. gss 的 20k 探针速率（5e-5/FE）长期加冕——rate 窗口含探针，
   小分母大速率永不老化；
2. 信用续展道从未开火——它藏在 P3，而 P3 只在 leader 被 max-2
   卡住时触发：**救援通道被健康 leader 扼流**；
3. max-2 交错破坏 v3 的获胜动力学（R6 smp material 长跑 + 停滞
   换道），v4.0→v4.2 每修一 case 破另一 case 的根因。

**v4.3 综合**（恢复 v3 核心 + 保护通道保留节奏）：
- leader = exploit 样本速率排名（**探针/票据速率永不加冕**）；
  引导期（无 exploit 历史）按私有信用选首段（S5: ctp 2.21）；
- 撤销 max-2 与全局 streak——material leader 连续跑（v3 语义）；
  停滞换道恢复（非 material exploit 段 → 一段回避期）；
- **保留节奏**：exploit_since_event ≥ 2×segment_fes 即强制开放
  一个开发事件（梯子/续展/轮转），与 leader 健康度解耦——计划
  的 reservation 语义落地；
- 信用续展、缺席下限、尾部护栏保留。

玩具验证：票据→信用引导 ctp exploit→节奏交错→ctp 长跑（final
9.3e-16）。22/22 调度测试过、563 全量过。51a v4.3 复筛启动。

## v4.3 首跑（误标 v4.2 目录）：S5 突破 + R2/R6 解剖 + 停滞失效 bug（2026-08-18 凌晨）

**结果**（目录已改名 v4_3_preview 留档）：S5/on **66,820**（前 572,420，
**好 8.6×**；比值 43×→**5.05×**；ON/OFF=0.043——协调本身在赢）；
R2/on 433,014（1.96×）；R6/on 433,482（1.37×）；A3/on 被旧 streak
审计误拦。exit 1。

**S5 获胜解剖**（设计首次完整兑现）：CTP 票据信用 2.21 引导首段
exploit（0.15 material）→ 1.752 → 节奏事件 → 1.154 / 0.679 →
轮转 → 收尾 0.082，CTP 累计 1.59M 运行——正是它需要的长跑。

**R2/R6 失败解剖**（同一机制）：CTP 票据信用引导后连跑 **4 个零增益
exploit**（1.59M 全废），AOR 只拿 225k。根因是自埋 bug：停滞回避
在授权内先武装又被同次尾部衰减清零——**从未生效**。已修（武装移到
衰减后）。附带确认两个残留设计缺口（晨间决策项）：
1. 采样垄断：唯一有 exploit 历史的臂零速率也无对手（其它臂从未
   获得速率样本）→ 需要"每 matured 臂至少 1 个 exploit 样本"的
   采样公平下限；
2. 节奏槽竞争：overdue 轮转下限抢占了 AOR 梯子的节奏事件——
   pending 梯子应排在 revealed 轮转之前。

**两处流程失误入档**：版本常量未随 v4.3 提升（cell 误写 v4_2 目录，
已改名留档）；streak 审计未随 v4.3 语义更新（A3/on 误拦）——均已修
（版本 v4.3、审计改 exploit_stagnation_yields）。全量 563 过。
干净 51b v4.3 全量重跑已启动。

## v4.3.1：采样公平下限 + 梯道优先（2026-08-18 凌晨，最后一轮）

干净 v4.3 判定：A3 1.007✓ / S5 **5.05×**（66,820，较 v3 好 25×）
/ R2 1.77✗ / R6 **1.15**✗（从 1.37 改善，on/off 0.77✓）。停滞
修复生效（R2 的 ctp 零增益段后正确交错），但解剖暴露最后两个
设计缺口：

1. **采样垄断**：ctp 引导后连烧 1.24M 零增益——唯一有 exploit
   历史的臂零速率也无对手（其它 matured 臂从未获得速率样本，
   leadership 无从比较）→ **采样公平下限**：leader 已有 ≥2 样本而
   某 matured 臂 0 样本时，最高信用未采样臂获得引导 exploit；
2. **节奏槽竞争**：overdue 轮转下限抢走 AOR 梯子的节奏事件（R2
   上 AOR 终生 225k < 300k 穿越线）→ **梯道优先**：pending
   revelation 梯子排到 revealed 轮转之前。

测试更新至 v4.3 契约（material 长跑合法、引导采样例外）22/22。
51a+51b v4.3.1 链式终轮启动（过夜最后一轮，无论结果如何出
晨间交接）。

## v4.3.1 终轮判定：跷跷板完全显形，停止迭代交晨间决策（2026-08-18 晨）

**终轮结果**：A3 1.005✓ / S5 **25.9×**✗（回退！v4.3 曾 5.05×）/
R2 1.94✗ / R6 1.34✗（v4.3 曾 1.15）。无严格胜。

**S5 回退解剖**：采样公平下限在 ctp 的 material 长跑中段（增益
1.75 之后）强制让位给 gcb/smp/aor 各一个 300k 样本段（全部低增益
或零增益），ctp 截断在 957k（v4.3 给了 1.59M）——**为 R2 设计的
公平破坏了 S5 的赢家保护**。

**跷跷板的精确刻画**（五个版本收据证据的总结）：
- R2 需要：打破零增益垄断（采样）+ AOR 梯道节奏槽（梯道优先）；
- S5 需要：material 领跑者不被打断（保护长跑）；
- 两者共享同一笔预算与同一批节奏槽——单条件修复必然此消彼长。

**判别条件已在收据中**：R2 的垄断段全部零增益，S5 的领跑段全部
material——**采样下限应以 leader 上一段的 material 状态为门**
（material 时禁打断、零增益时强制采样）。这是晨间第一候选修复
（三行改动），但按纪律停止夜间自主迭代，交用户裁决。

**夜间的确定成果**：
1. S5 5.05×（v4.3）——框架核心机制（信用引导→material 长跑→
   节奏保护）的完整验证，比 v3 好 25 倍，ON/OFF=0.043；
2. A3 全程非劣（所有版本）；
3. R6 最佳 1.15×（v4.3）；
4. 每个版本的全部收据级归因（本轮日志连续五节）；
5. HCC 24 函数参照就绪 + vendor 2.4× 提速 + 快迭代基础设施。

## v4.4 预注册实现：健康度条件公平 + overflow 锚定切换（2026-08-18 晨）

按用户评审裁决实现 MATERIAL_RUN/DISCOVERY_RUN 状态机（leader
上一段 material 门控采样；节奏梯档在 material 长跑中保留——v4.3
的 S5 获胜解剖证明 75k 梯档不打断兑现）；sampling_debt 状态入账。
四 overflow cell 判定锚切换至当前树全预算值（overflow_reference.
json；R2_ctp 实为差 0.08%，三好一差如实记录）。判定口径固定为
frozen-version。22/22 测试过。预注册见协议附录 3。

## Gate 51b（v4.4 冻结候选）正式判定：协议层全绿，性能层 1/4（2026-08-18 午）

**协议层首次全绿**：全部 cell 审计通过、overflow 参照锚定生效
（R2_gcb 判定锚 = 295662.808 当前树值 ✓）、16 个 standalone 齐备。
gate_passed = false（性能层）。

**Frozen-version 结果**：
- A3 1.007× ✓（历版本稳定非劣）；
- **S5 5.045×（66,820）——冻结版复现 v4.3 最佳值（逐位相同），
  ON/OFF=0.050**。跷跷板正式打破：健康度门恢复 CTP material 长跑
  且不再被采样公平打断。S5 收益从 best-of-version 诊断证据升级为
  frozen-version 稳定收益（较 v3 的 126× 好 25 倍）；
- R2 1.97×✗（ON 比 OFF 差 1.234——handoff 负贡献；AOR 梯道供血
  仍不足是主因）；
- R6 1.34×✗（v4.3 曾 1.15×，冻结版回退；ON/OFF 0.907 方向对）。

**下一步候选（按预注册纪律，需用户裁决）**：R2 = DISCOVERY 态的
梯道节奏密度（AOR 越过 300k 穿越线所需梯档数 vs cadence 间隔）；
R6 = material 长跑动力学在 v4.4 门控下的衰减路径。两者都有冻结
基线可做差分归因。

## Gate 51c 启动：v5.1 fresh-seed 确认（2026-08-18 夜，用户授权自主推进）

冒烟验证通过：A3/20260901 的 Phase-I 全新发现（67.8s，44 块/110
关系）+ 完整 OC on 臂 3M（2048s ≈ 34 分钟，audit 全绿，终值
78,590 与设计 seed 的 78,495 同量级——seed 稳健的初步信号）。

全量 72 cell（3 fresh seed × 4 case × 6 臂）8 worker 钉 P 核启动。
判定标准预注册于 docs/arac-oc-gate51c-protocol.md（中位数 ≤1.05、
最差 seed ≤1.10、≥2 case 严格胜、胜例 ON<OFF ≥2 seed）。参照 =
同 seed 四 standalone 最优（不跨 seed、无 Gate 50 逐位要求）。
预计 ~5-6 小时（Phase-I 12 × ~1min + 3M cell 60 × ~25min / 8）。

## Gate 51c 判定：未过，但结构信息极其丰富（2026-08-19 晨）

**结果**（3 fresh seed × 4 case × 6 臂，72 cell 零失败、审计全绿）：

| case | 中位数 | 最差 seed | ON<OFF seed 数 | 判定 |
|---|---|---|---|---|
| A3 | **1.003** | 1.008 | 1/3（两平） | ✓ 非劣稳健 |
| R2 | 1.271 | 1.377 | **3/3** | ✗（但见下） |
| R6 | 1.126 | 1.162 | 2/3 | ✗ 边缘 |
| S5 | 4.667 | **46.6** | 0/3 | ✗ 一个 seed 灾难 |

**R2 的隐藏头条**：seed 20260901 上 ratio = **0.934——严格胜**
（ON 297,118 < 最优 standalone 318,270），ON<OFF=0.641。同一
机制在三个 seed 上 ON<OFF 全胜（0.641/0.866/0.730）——handoff
因果在 R2 上稳健成立，是 ratio 方差大（gcb/aor 各 seed 强弱交替）。

**S5 的灾难解剖**（seed 20260901，46.6×）：CTP 票据 1.709 →
adaptive_lock 兑现 1.495 → **protected_runway 段零增益**（跨过
archive 后无新增益）→ 被 horizon 挤出 → gcb 接管（0.631）→ CTP
终身 981k（设计 seed 上是 1.87M）。CTP 的 runway 保护在"零增益
plateau"时没有释放——**plateau 释放（v5.2 预注册杠杆）正是此病
的解**：released 后 runway 不再独占，horizon/其它臂可接管剩余
预算，而 CTP 保留其已贡献的 archive。

**R6 边缘**：三 seed 1.126/1.047/1.162——中位数 1.126 未过线，
方向一致但差距略大于 A3 型噪声。horizon 接力链工作正常。

**结论**：v5.1 未通过 51c 冻结。但两个 v5.2 杠杆（plateau 释放 +
R2 梯档）现在有了明确的 fresh-seed 收据证据支撑。A3 非劣是
稳健的。下一步等用户裁决 v5.2。

## v5.2 版本提升落地：版本隔离 + 旧入口退役 + 51c v5_2 复判入口（2026-08-19 午）

按预注册纪律完成 v5.2 升版（用户已实现的杠杆 1 = w1 有界验证窗随本版本定名）：

- **常量与校验**（episodes.py）：新增 `DEFAULT_SCHEDULER_VERSION_V5_2="v5.2"` +
  policy/schema 字符串；v4 入口处校验——仅 v4.4（冻结位级路径）与 v5.2 可产出，
  手工构造的 v5.0/v5.1 标签或"v5 特性 + 非 v5.2 版本"响亮报错；policy/schema
  映射改为按版本字符串判定；
- **旧入口退役**：`run_oc_episode_schedule_v5/v5_1` 保留签名、调用即
  RuntimeError 指向 v5_2（防止新行为被贴旧标签；v5.0/v5.1 行为已不在树中，
  冻结 cell 以其记录的 manifest 保持溯源）；
- **生产入口跟进**（overlap_core）：`ARAC_OC_SCHEDULER_MODES` 增 `v5_2`，
  新增 `run_arac_oc_v5_2` 命名入口并导出包级；`v5_1` 模式保留（调用经退役
  入口自然报错）；
- **Gate 51c v5_2 复判入口**（`experiments/oc_phase_aware_gate51c_v5_2.py`）：
  OC 臂（on/off × 12）在版本隔离目录 `oc_phase_aware_gate51c_v5_2` 重跑；
  standalone 48 cell 与 Phase-I 12 checkpoint **复用冻结 v5.1 产物**——锚定 =
  v5.1 confirmation 的 implementation manifest + checkpoint_hash + 结构完整性，
  冻结 cell 早于 anytime 层，其 anytime 轨迹从自带 segments 确定性重算
  （与冻结 anytime_auc.json 12/12 逐位一致），只写入 v5_2 副本、冻结源不动；
  判定数学从 v5.1 入口原样导入，两次判定可构造性可比；
- **测试**：v5_1 契约测试迁至 `test_oc_episode_schedule_v5_2.py`（4 个机制
  测试 + 5 个新纪律测试：双退役入口报错、版本标签不可产出）；v5 HPR 机制
  测试改走 v4 入口显式 v5.2 标签；unified_loop 生产入口测试切 v5_2 模式 +
  v5_1 响亮退役断言。调度器/入口定向 44 过；全量 592 过 / 4 败（stash 对照
  验证 4 败全部预存隔离类，零新回归）；ruff 全绿；
- **版本控制**：快照提交 40d3f29（整条 ARAC-OC 工作线首次入库，13,769 文件
  含 410MB artifacts 证据链；agent 会话目录入 .gitignore）。

**待用户裁决**：杠杆 2（R2 horizon 顶档 300k→450k，诊断表已证三 seed AOR
运行时恰停在 450k 穿越线）是否并入 v5.2 再跑 51c 复判，还是 v5.2 按当前
内容（仅杠杆 1）出判。复判命令：
`.venv\Scripts\python.exe -m experiments.oc_phase_aware_gate51c_v5_2 --workers 8 --pin-p-cores`（OC 24 cell，约 2 小时）。

## v5.2 杠杆 2 落地：material horizon promotion（2026-08-19 午后）

用户裁决取代原"R2 顶档 300k→450k"表述（协议 §6 已冻结最终定义）：
诊断表显示三 seed AOR 恰停 450k 穿越线，但仅 20260902 material
（0.0511），另两 seed 零增益且后续收益在 GSS——无条件加档会挤占已证明
有效的利用预算。根因（用户定位）：`_current_leader_name` 只认
exploit_history，material horizon 属 challenger 通道，发现的价值无法
进入利用阶段。

**实现**（episodes.py，v5.2 语义、hpr 门控、无新配置旗标）：
- 触发：`_execute` 中 horizon 保留段 material 且未用过 promotion →
  `horizon_material_pending=True`（pending 标记使触发能穿越 material
  leader 的 runway 交错存活，不重蹈 v4.2"救援通道被健康 leader 扼流"）；
- 授予：`_horizon_promotion_grant` 返回 probe_order 首个 pending 且
  可执行的 episode，窗 = min(2×w1, segment, remaining)，exploit 类、
  exploitation 账本、reservation_kind=horizon_promotion、每 episode
  每 run 一次；插入两处——adaptive 块（adaptive_lock 之后、P1 之前，
  "立即"语义）与 P2 hpr 路径（runway 之后、新 horizon 保留之前）；
  material leader 的 runway 不被打断（S5 保护保持）；
- 后果链：material 验证窗写入首个 exploit 速率样本 → 按常规排名竞争
  leadership → 赢得则进入有界 runway；平坦验证窗 plateau 释放——
  损失上界 2×w1=150k；
- 审计四条：promotion 收据必须是 exploit、必须由同 episode 更早的
  material horizon 收据挣得、窗 ≤ 2×w1、每 episode 单次。

**测试**：翻转计数玩具（前 N 次评估 1.0、之后 0.5）确定性复现 R2
机制缩影——N 落在 AOR horizon 窗内（18,200）：material horizon
0.693 → 下一段立即 promotion（窗 975 ≤ 1,600）→ 平坦释放，审计绿；
N 晚一段落在 smp exploit 窗（18,800）：exploit 通道 material 不铸
promotion（负对照）。v5_2 契约测试 11 个全过；定向 47 过；ruff 绿。

51c v5_2 复判待启动（协议 §6 已定稿，manifest 就绪）。

## Gate 51c（v5.2）判定：未过；promotion 兑现、runway 粒度伪释放显形（2026-08-20 晨，夜间自动完成）

72/72 cell 零失败、审计全绿（含 promotion 四条新审计）。gate_passed=false。

| case | v5.2 中位数/最差 | v5.1 中位数/最差 | 读数 |
|---|---|---|---|
| A3 | 1.003/1.008 ✓ | 1.003/1.008 | 非劣稳健 |
| R2 | 1.537/1.575 ✗ | 1.271/1.377 | 回退；ON<OFF 3/3→1/3 |
| R6 | **1.049/1.084 ✓** | 1.126/1.162 ✗ | **双条款达标** |
| S5 | 39.65/46.64 ✗ | 4.667/46.64 | 灾难性回退 |

**杠杆 2 完全兑现**：10 次 promotion、6 material；R2/20260902 AOR 显现→promotion
0.206→连续 3×75k runway 链（显现后 exploit 达成）；R6 三 seed 全 material 是其
达标的直接机制；4 次平坦 promotion 一窗识破、损失有界。

**杠杆 1 的结构性毒药显形**：S5 三 seed 的 CTP runway 全部 = 1×75k×零增益×释放。
铁证（S5/20260902）：v5.1/v5.2 轨迹 seg 9 前逐位相同（ticket 1.90→lock 1.31），
v5.1 首 300k runway 窗增益 0.338 但到达点在 75k 之后 → v5.2 的 75k 子窗增益恰为
0 → 伪释放 → 1.22M 打磨链（0.34+1.24+0.81+0.70）全灭 → ON 449,810（OFF 9.1×）。
**CTP 型 material 增益到达尺度 > 75k，逐窗 materiality 判定必然伪释放**——释放
耐心成为新跷跷板轴（R2 要快、S5 要慢）。

R2 回退另含一笔：adaptive_lock 验证窗本身仍无界整段 300k（R2/20260901 平坦
lock 烧 300k）。候选下一步（晨报 docs/morning-report-2026-08-20.md）：runway
释放耐心 N 窗参数化（先从 v5.1 轨迹定 CTP 增益到达分布）、lock 窗 w1 化、或
接受 S5 税保 promotion 收益的正交组合。

夜间运行事故与处置：宿主环境 22:39 前静默终止首实例（9 cell 无损）→ OS 脱离式
重启续跑 → 01:06 齐全。看护自动化（30 分钟周期）已自删；事件见
artifacts/oc_phase_aware_gate51c_v5_2/.watchdog_log.md。

## v5.3 实现：几何验证阶梯 + 收养宽免（2026-08-20 午）

用户裁决采纳升级方案（docs/arac-oc-v5_3-design.md，评审修正①-④后冻结：
平坦暴露上界 375k/任职期取代错误的 525k 累计口径、伪代码补 grace 清除行、
lock 玩具场景 + 预注册决策规则、R6 风险行 + 可证伪预测）。

**实现**（episodes.py，v5.3 语义、hpr 门控、无新配置旗标）：
- 阶梯：per-episode `rung`，w(rung)=min(w1·2^rung, segment)；lock 起始
  rung 0（修 F2：R2 平坦 lock 暴露 300k→75k）、promotion 固定 rung 1
  （=v5.2 语义）、runway 用 w(rung[e])；窗口不再向 min_step 扩张，
  不可执行即让位后续车道；
- 转移：material → rung+1 封顶 + released=False + **grace 清除**；
  flat + grace 武装 → 消费宽免不释放不降档；flat 无 grace → 释放 +
  rung 归 0。收养（adopted=True）→ rung 归 0 + grace 武装（修 F1：
  重锚暖机窗不再被误杀）；
- 影子字段：verification_rung / grace_consumed / would_release_v5_2
  （精确口径：仅宽免消费或 runway 窗 > w1 的 flat 窗打标——promotion/
  lock 与 v5.2 同径释放不打标）/ gain_rate；
- 审计四条（收据表面确定性重放阶梯/宽免状态机）：rung 单调、平坦暴露
  ≤ w1+segment=375k/任职期、宽免每次收养至多一次且须武装才可消费、
  反事实双条件一致；adaptive_verification_window_bounded 更新为
  w(1)=2w1（material lock 升档后首窗）；
- 版本纪律：v5.2 入口退役（保签名报错）、v5_3 常量/校验/policy/
  run_oc_episode_schedule_v5_3；overlap_core 增 v5_3 模式 +
  run_arac_oc_v5_3 并导出。

**测试**：v5_2 契约文件迁移为 test_oc_episode_schedule_v5_3.py（9 测）：
improving 玩具验证 lock@800→material→runway@1500 升档链；翻转玩具
（30k 预算）确定性复现宽免（smp/ctp flat 窗消费宽免不释放 + cf52 打标）、
平坦 lock 底档速释、promotion 有界；**lock@rung-0 脆弱性判定：三种翻转
位置（16600/17000/17300）脉冲值均被票据通道捕获、终值全 0.5、审计绿
——按预注册决策规则 lock 维持 rung 0**。unified_loop 切 v5_3 模式 +
v5_2 退役断言；17/17。定向测试全绿；全量回归待出。

**Gate 51c v5_3 入口**（experiments/oc_phase_aware_gate51c_v5_3.py）：
standalone/Phase-I 复用锚不变（v5.1 confirmation，48/48 冒烟验证）；
DO 入档发现结构性事实——soft-RDDSM 划分 disjoint、HCC 式共享变量 DO
恒 0，故补 relation_coupling（相关块覆盖变量比/关系数/强度）作为
v5.4 实际设计输入（A3 0.728/110、R2 0.676/56、R6 0.454/208、
S5 0.657/200 @20260901）。预注册预测见设计文档 §4-5。

## Recovery action lifecycle diagnostic（2026-08-23）

`artifacts/recovery_action_lifecycle_diagnostic_v1/summary.json` 已完成 55 个 paired
case/seed、共 110 arms（24 workers，0 failures）。55/55 pair 复用相同 checkpoint，
terminal FE 与 receipt/action-result hash contract 全通过；该诊断不使用历史阈值做运行时决策。

- SMP E2-E6：历史兼容 schedule 在 24/25 pair 胜出，case geometric ratio 为 1.789、
  4.559、8.687、11.756、10.663。当前 SMP 将 2.82M FE 几乎全部放入 stateful visits，
  尾部 no-op；历史兼容版本保留 rescue 和约 1.41M MMES global polish，生命周期差异已隔离。
- GCB R1-R6：结果混合（历史兼容 16/30 pair 胜出）。当前 schedule 在 R1/R3 更好，
  R2/R4/R6 略差，R5 近似持平，不能据此统一回退 GCB。
- `historical_compatible` 是冻结 source 的 schedule 对照，不是 bitwise historical replay；
  不授权历史 superiority、B1-Final、soft routing 或生产 patch 实验。

下一步：先恢复 SMP rescue/global-polish budget ownership，做 E2-E6 五 seed paired smoke，
再决定是否重跑 24x5 screen；GCB 继续冻结并单独做 schedule attribution。

## SMP lifecycle repair and 24x5 recovery screen（2026-08-23）

按上述诊断执行最小修复：保留 `RecoveredSmpExecutor` 的可恢复 v2 状态机，
将 `RecoveredActionRegistry` 的 one-shot SMP dispatch 改为历史兼容生命周期 wrapper。
wrapper 复用冻结 `SmpExecutor`，恢复 stateful visits、rescue、MMES global polish 和
terminal Sep-CMAES；route 显式标注 `historical_compatible_smp_v1_clip_offspring_true`。

paired smoke：`artifacts/recovery_smp_lifecycle_smoke_v1/summary.json`。
`50/50 arms`、`0 failures`、`25/25 checkpoint pairs`、terminal FE/hash 全通过；
E2-E6 的 25 对 final error 逐位一致，所有 current route 均有 rescue/global-polish，
无 noop 尾部。该结果证明修复已挂载且生命周期可归因，不作 superiority 声明。

随后用新 output root 重跑 `24 × 5` mapped-action screen：
`artifacts/recovery_first_screen_smp_lifecycle_v2/summary.json`。
结果为 `120/120`、`0 failures`、所有 contract 通过，但 overall screen gate 仍为 false：

- SMP：E2-E6 通过，E1 失败；
- AOR：A4、A6 失败；
- GCB：R1-R6 全部失败；
- CTP：S6 失败，其余 S1-S5 通过。

因此 SMP 生命周期修复可以保留为恢复基线的一部分，但不能把未恢复动作的回归
归因到 SMP 或 shared-patch。B1-Final、soft routing 和 production patch 继续关闭；
下一步按失败动作族做最小 fixed-action attribution，优先 E1 与 GCB schedule，
不重新启动旧 P2/P4。

## E1 zero-relation preservation（2026-08-23）

24x5 screen 后发现 SMP historical-compatible 生命周期不应覆盖零关系 E1：
旧 recovered zero-relation hybrid route 的 E1 均值约为 near-zero，而统一历史
生命周期引入 clipping/rescue 分配后明显回归。已将 recovered SMP one-shot registry
改为 topology-conditioned dispatch：正关系沿用 historical-compatible lifecycle，
零关系沿用原 recovered hybrid lifecycle。

独立 gate：`artifacts/recovery_smp_zero_relation_preservation_v1/summary.json`。
5/5 seed、0 failures、FE/hash/checkpoint 全通过，final error 与旧 E1 screen receipts
逐位一致。此前 `recovery_first_screen_smp_lifecycle_v2` 的整体 Gate 仍只代表条件
路由修复前版本；如需新的 B1 汇总，必须重新跑完整 24x5。

## Topology-conditioned 24x5 recovery screen（2026-08-23）

条件路由修复后重新运行完整 `24 × 5` screen，产物为
`artifacts/recovery_first_screen_smp_topology_v3/summary.json`：
`120/120 arms`、`0 failures`，checkpoint、terminal FE 与 receipt hash 全通过。

- SMP E1-E6 全部通过；E1 均值恢复为 `7.75e-06`；
- AOR A4/A6、GCB R1-R6、CTP S6 仍失败；其余 case 通过；
- overall screen gate 仍为 false，不能宣布 B1-Final 或历史四动作完全恢复。

SMP 条件路由冻结为当前 recovered baseline；后续只做 AOR/GCB/CTP 各自的
fixed-action attribution，不把残余回归归因给 shared-patch。

## GCB schedule attribution（2026-08-23）

基于已完成的 `recovery_action_lifecycle_diagnostic_v1` receipts，新增无重算
归因报告 `artifacts/gcb_recovery_attribution_v1/report.json`：60 arms、30 pairs，
checkpoint/terminal FE/route 解析全通过。

current 相对 historical-compatible 平均少约 235k warmup/source FE、少约 74k
coordination FE，多约 309k continuation/native FE；但性能方向混合（R1/R3 current
更好，R2/R4/R6 historical-compatible 更好，R5 近似持平）。因此不做统一 GCB 回退，
生产 GCB 保持冻结；后续若继续，只运行预注册 fresh-seed schedule ablation。

## AOR A4/A6 + CTP S6 fixed-action attribution（2026-08-23）

归因协议与报告：
`experiments/historical_recovery/aor_ctp_recovery_attribution_protocol_v1.json`、
`artifacts/aor_ctp_recovery_attribution_v1/report.json`。验证命令：
`$env:PYTHONPATH='.;src'; python -m experiments.historical_recovery.aor_ctp_recovery_attribution verify`，
通过；定向测试 `tests/test_aor_ctp_recovery_attribution.py` 为 2/2。

- AOR 当前/冻结历史 source SHA-256 完全一致
  （`15f7d567351ce660658079b70f3cc00d9feceeab7c962a4779246e1f442e2805`）。
  A4=`78,257.56` vs `78,200`（ratio `1.000736`），A6=`78,123.18` vs `78,000`
  （ratio `1.001579`）；结论为 source/lifecycle 非归因，不修改 AOR。
- CTP S6 screen mean=`5,283.95` vs `4,180`（ratio `1.264102`），虽已出现
  reserved tail，仍未恢复。matched checkpoint tail ablation（31001-31003）
  为 3/3 candidate wins，几何 ratio=`0.289733`；但与 screen seeds 不配对，
  仅作机制证据，不作 screen recovery 或生产替换声明。
- `aor_code_change_authorized=false`、`ctp_production_change_authorized=false`；
  B1-Final、patch、soft routing、selector 继续关闭。若继续 CTP，下一 gate 是
  fresh matched S6 screen-seed tail attribution。

## Recovered baseline freeze（2026-08-23）

当前修复后的四动作执行锚点已冻结为
`arac-recovered-baseline-20260823-v1`。冻结协议：
`experiments/historical_recovery/recovered_baseline_freeze_protocol_v1.json`；
校验入口：
`experiments/historical_recovery/verify_recovered_baseline_freeze.py`。

- 21 个 source/protocol/evidence 文件的 SHA-256 全部锁定；
- recovery screen 的 120/120 arms FE、checkpoint、receipt contract 全通过；
- SMP lifecycle smoke 与 E1 preservation 均通过；
- patch、soft routing、selector 仍为关闭，冻结 baseline 是默认生产回滚目标；
- 下一步升级只允许进入 `experiments/upgrade/`，按 U0-U4 独立验证，不覆盖冻结源。

## Stepwise upgrade plan v2 + literature review（2026-08-23）

完成 v1 阶梯方案的契约审阅与跨来源文献核验。正式方案：
`docs/arac-oc-stepwise-upgrade-plan-v2.md`；检索记录：
`references/lit-review/arac_oc_stepwise_upgrade_literature_2026-08-23.md`。

- S1/S2 保留为零新增 FE 的重排机制；confidence 在当前 checkpoint 未证明可用，
  首版只用 shared-variable leverage count；
- S3 明确为每 scope 最多四个 shared coordinates、每个坐标一对 ± 候选、
  固定 8-FE reservation cap；
- S4 只引入 `z,r` 和局部 context hash，写集排除 self-accept reset；
- S5 拆为 S5a disagreement candidate 与 S5b soft owner weighting；
- DCCC/CCFR3/SACC/AOS 资源调度留到 S6，不与首版 kernel 混合。

下一步从 S0 freeze verifier 与 S1 screen 开始，不直接接入 S5 或学习型调度。

## Stepwise upgrade execution: U0 passed, U1 launched（2026-08-23）

按 `docs/arac-oc-stepwise-upgrade-plan-v2.1.md` 启动 `shared_patch_v1` 升级候选，
代码全部位于 `experiments/upgrade/shared_patch_v1/`，不触碰冻结源。

- **U0 baseline guard**：`artifacts/upgrade_u0_baseline_guard_v1`，verifier
  8/8 通过（status frozen、三项 green、patch/soft routing/selector 全 off）。
  运行入口：
  `PYTHONPATH='.;src' ./.venv/Scripts/python -m experiments.upgrade.shared_patch_v1.u0_baseline_guard`。
- **conflicting overlap generator v1**：3 拓扑（chain/hub/pairs，各 10 块、
  每链 8 个共享变量）× 2 冲突级（mild δ=0.10、strong δ=0.25）= 6 个预注册
  cell。冻结 Phase-I（soft-RDDSM v9）在全部 6 cell 上发现正关系
  （28 条/cell，最大强度 0.039-0.169，strong 一致高于 mild）。
- **S1 leverage-priority sweep 实现**：executor-lane 的 leverage 预注册为
  "checkpoint 块的关系关联数"（恢复 lane 的 checkpoint 只有块级
  RelationEvidence，无逐变量 shared 标记）；首个 block sweep 的前
  `ceil(20%·n)` 槽位按 `(-leverage, original_rank)` 重排，其余槽位与后续
  sweep 保持冻结顺序。零关系时顺序恒等（内置 ov0 no-tax 控制）。
  机制通过包裹冻结扫掠函数的 `block_order` 参数实现，零新增 FE。
- **S2 propagation handoff 实现**：`HandoffState` 单 transient 字段
  `last_improved`；gcb 逐冷访问槽位、ctp 以 polish 段为槽位（coverage
  交错段无槽位语义，保持 S1 静态序，已预注册为粒度解释）；邻居分别定义为
  关系边与 polish 覆盖的变量集交集。`no_acceptance_event` 与
  `host_unreachable` 分开计数。零关系时逐位回退（冒烟验证通过）。
- **解释器事故与处置**：U1 首次执行意外跑在 PATH 上的 hermes-agent venv
  （numpy 1.26.4 + libopenblas 0.3.23.dev）下，AOB 仪器化重跑与冻结 screen
  收据的逐位一致性全部失败（checkpoint hash 一致、final_error/route 漂移）。
  根因是 BLAS 数值栈不同导致轨迹发散，不是代码或仪器化问题。已停止战役、
  删除该批产物（未做任何 gate 判定），并在 U1 协议中预注册运行环境要求
  （`E:/ARAC/.venv` + libscipy_openblas，收据记录 python_executable 与
  threadpools，identity gate fail-closed）。升级 lane 的所有命令改用
  `./.venv/Scripts/python`。这次事故同时确认：冻结 screen 的逐位复现
  依赖解释器/BLAS 栈，跨环境不可假设。
- **U1 重跑**：24 臂（AOB 12 mapped-host 仪器化重跑 + 逐位一致性校验、
  generator 6 cell × 2 强制 host）在正确解释器下进行中，产物将写入
  `artifacts/upgrade_u1_host_reachability_v1`。S1 screen（240 臂，A0/A1
  配对，里程碑 anytime 采样）在 U1 gate 通过后启动。

## U1 host reachability passed（2026-08-23）

`artifacts/upgrade_u1_host_reachability_v1/summary.json`：8/8 checks 通过，
`s1_screen_authorized=true`，`performance_comparison_authorized=false`。

- **AOB 逐位一致性**：12/12 mapped-host 仪器化重跑（S1-S6 ctp、R1-R6 gcb，
  seed 117，3M FE）的 final_error/route/action_result_hash 与冻结 screen
  收据逐位一致——同时证明了仪器化本身零税、冻结 anchor 在正确解释器栈
  （`E:/ARAC .venv` + libscipy_openblas）下可精确复现；
- **reachability 表**：S2-S6（ctp）与 R2-R6（gcb）host 可达且存在
  leverage>0 的 scope 访问；S1/R1 为零关系（mechanism_silent，S 阶梯按
  构造 no-op）；A/E 12 case 按契约 mount_absent；
- **generator lane**：6/6 cell Phase-I 发现正关系，12/12 强制 host 运行
  完成终值契约且全部访问到 leverage>0 scope；
- **收据哈希口径修复**：首执行把 `segments[*].per_block_visits` 写成 int 键，
  `sort_keys` 对 int 键按数值序、JSON 回读按字典序，导致重算哈希与声称
  哈希不一致；校验器改为按写入时规范形（int 键）重算并与声称哈希精确
  对账，后续 stage 的仪器化一律 str 键。产物本身未改动。

S1 screen 已按预注册协议启动
（`experiments/upgrade/shared_patch_v1/s1_leverage_sweep_protocol_v1.json`：
240 臂 = AOB 12 mapped hosts × 5 seeds × {a0,a1} + generator 6 cells × 5
seeds × {ctp,gcb} × {a0,a1}；eps_ref 表在首臂运行前冻结；anytime 里程碑
600k/1M/2M/3M；判据为 ov0 逐位 no-tax + 每宿主 lane 的 final/anytime
非劣 paired bootstrap）。S2 协议已同步冻结，加载器强制 S1 gate 先通过。

## S1 leverage-priority sweep screen: gate FAILED, reorder closed（2026-08-23）

`artifacts/upgrade_s1_leverage_sweep_v1/summary.json`：240/240 arms 完成、
0 失败，`gate_passed=false`、`s2_screen_authorized=false`。按 v2.1 止损线
（S1 失败：关闭重排，不接入 patch），S1 关闭，S2 保持 blocked，生产默认
不变（重排从未接入生产，冻结 baseline 无需回滚动作）。

实质结果：

- **ctp 宿主结构惰性**：S2-S6 全部 30 对 ratio 精确 1.0000。根因是冻结
  coverage 交错段的所有块会话一次性创建且从不重锚（`begin_visit` 只在
  冷访问路径调用），交错顺序对数值轨迹不敏感——"纯重排"机制在该宿主
  无法表达，除非 fork coverage 语义（超出 S1 的零 FE 重排边界）。
- **generator lane 全部 1.0000**：6 个 cell 的 Phase-I 发现为 8 块完全图
  （28 关系、leverage 全等），按平局规则 S1 序 = baseline 序，机制静默。
  v2 generator 需要更稀疏的拓扑才能真正检验 leverage 排序。
- **gcb 宿主为唯一活跃 lane 且 final 非劣失败**：R2-R6 每对 route 全部
  改变（早停时点随访问顺序移动，FE 分布漂移属预期），pooled geo≈0.998
  但 per-case CI 上界至 1.26（R3 geoR=1.056、R5=1.054），`gcb_final_nontax
  =false`；anytime 非劣通过（CI 下界 > -0.05）。机制在 gcb 上可激活、
  可观测，但没有稳定收益方向。
- **ov0 控制**：R1/S1 全部 5 seed 逐位一致（identity 通过），`ov0_exact
  _no_tax=true`；全部 a0 AOB 臂与冻结 screen 收据逐位一致。

检查规则缺陷（记录在案，不做事后放宽）：`routes_unchanged_all_pairs`
把顺序改变宿主的 route 漂移当失败；`a0_orders_never_changed` 的扫描误含
a1 收据；`a1_order_changed_where_leverage_exists` 未允许 leverage 平局
导致的合法零变化。即使全部修正这三项，`gcb_final_nontax=false` 仍单独
构成实质失败。

结论与去向：S1 关闭；重排不进入任何候选；S2 协议与加载器保持 blocked
留档。若重启升级，需要新的候选版本目录（例如仅在 gcb 宿主挂载的
leverage 排序 v2 + 稀疏拓扑 generator v2），不得复用本轮结论改判。

## v5.0 hyperedge_ctp_v1：H0 认证失败，线 B 按止损线终止（2026-08-23）

按 v5.0 计划（CTP 尾部因果复验 → 超边认证共享变量内核）执行。代码在
`experiments/upgrade/hyperedge_ctp_v1/`，冻结源/Phase-I v9/selector/
AOR/SMP/GCB 全程未改；每阶段前置冻结校验器。

- **generator v3 冻结**：`(linked-elliptic, shared_width=8, λ=2.0)`
  （预注册网格最优 26/30；强制零最优值约定——Phase-I 结构通道按
  `best_error + optimum` 重建基值，非零最优值会把最小值注入每个交互
  差分并把发现饱和成完全图，这是 v2/v3 初版退化的根因）。真值增加
  显式 `shared_owner_pairs`。
- **T0（线 A）**：S3/S6 × 20270111-15 × {tail_20pct, no_reserved_tail}
  运行中；变体执行器与冻结版在零关系 checkpoint 上逐位一致、正关系时
  仅移除 20% 尾部预留（冒烟验证 FE 重分布精确）。
- **H0（线 B）GATE FAILED**：`artifacts/upgrade_hyperedge_ctp_v1_h0_v1`。
  soft-RDDSM（计划冻结配置 + 180k 精确补足）在 6 cell × 3 discovery
  seed 上的认证结果：
  - precision 几乎处处 1.0（唯一例外 chain4-strong/20270103 = 0.0：
    该 seed 的发现把 planted 0∪1 合并成一个 200 变量区域——链式传递
    失败模式，认证准则正确拒绝）；
  - **chain4（链拓扑）结构性失败**：每个 seed 都有 8 个 2-region 超边
    + 16 个 3-region 超边（链变量的证据跨三个区域），违反"每个超边
    恰有两个 region"；
  - **pairs3-strong 差一个 seed**：precision 1.0、无 3-region、
    forest/度数全过，但 20270101 的 recall=16/24=0.667<0.75（第三条
    link 的确认在预算边界未完成）；
  - hub3 度数 3，按准则正确不激活。
- **处置**：按预注册止损线（H0 失败 → 停止线 B），H1/H2/AOB
  preservation 均未执行，未做任何性能比较。认证机器本身按设计工作：
  它拒绝了链式传递与 hub 结构。pairs3 拓扑在 2/3 seed 上干净认证
  （precision 恒 1.0），如实记录为仪器能力边界；不做任何 seed 更换
  或阈值放宽（属事后调参）。若重启线 B，需要新候选版本（例如更高
  dsm_budget 的仪器修复 + 只预注册 pairs 族），不得复用本轮结论改判。
- **seed 登记冲突**：20270101-25 段与 `current_selector_fresh_e2e`
  协议（从未执行、无产物）重叠；已在根协议记录，未静默忽略。

## v5.0 T0 因果复验通过（方向修正后判定）（2026-08-23）

`artifacts/upgrade_hyperedge_ctp_v1_t0_v1/summary.json`：`gate_passed=true`。

- S3：R(tail_20pct/no_reserved_tail)=0.178，CI=[0.058, 0.585]；
- S6：R=0.419，CI=[0.358, 0.518]；
- 两个 case 均满足 R<1 且 CI 上界<1：冻结 CTP 的正关系 20% MMES 尾
  在全新 seed（20270111-15）上有强因果收益（尾部臂误差低 2.4-5.6 倍），
  checkpoint 共享/正关系/终值 FE 契约全部合规；
- 记录一次判定修复：汇总初版把配对方向写反（算成 no_tail/tail），
  抽查原始收据发现方向与判定矛盾后修正为预注册口径
  R=tail/no_tail 并重算；臂级收据从未有误，全部 20 臂 0 失败；
- 按计划：冻结 20% 尾保持不变（其因果有效性现在有 fresh-seed 证据），
  "再调 tail 比例 / 按 Phase-I 特征调 tail"分支永久关闭。

**v5.0 总结**：线 A（T0）通过并确认尾部因果；线 B（H0 认证）失败并按
止损线终止，未进行任何性能比较。冻结基线全程未动，校验器始终绿。

## v5.1 pairwise_edge_ctp_v1：P0 证书门失败（区域合并伪影），链式可分离性诊断通过（2026-08-24）

按 v5.1 协议执行：不放宽任何阈值、不改冻结源；二元共享边证书
`PairwiseSharedEdge(j, region_a, region_b)`（恰两区域 + 双侧证据 +
无第三区域未解释证据；三区域输出整体拒绝、禁止事后拆分）。

- **P0（pairs3-strong × 20270401-05，DSM/RDG 预算上限 100k 预注册）**
  GATE FAILED：`artifacts/upgrade_pairwise_edge_ctp_v1_p0_v1`。
  - 3/5 seed 完美：precision=1.0、recall=24/24、零区域合并、重放哈希
    一致、180k FE 精确；
  - 2/5 seed 失败且签名一致：粗覆盖把两个 planted 块合并成一个
    ~200 变量区域 → 该对 link 无法出证书 → recall=16/24=0.667；
  - **合并机制定位**：粗 RDG 按变量索引升序取种子，当某共享变量
    最先成为其组件种子时，其交互邻域同时含两个 owner 块（共享变量
    同时进入两个块的联动和）→ 合并；概率 ~8%/对/seed，与 2/5 观测
    吻合；与预算无关（55k→100k 无改善）；
  - 全部 5 seed 已出证书 15/15 全对（precision 恒 1.0）；
  - 按预注册：patch 阶段不授权，未做任何性能比较。
- **chain_pair_isolation_diagnostic_v1（chain4-strong × 3 seed，
  每链接逐变量 residual 三检验，289 FE/seed）通过**：链的每个共享
  变量都与自己两个 owner 的残差交互、与远侧块残差不交互——
  **链式 overlap 可被 pair-specific residual 证据分离**；v5.0 的链式
  失败源于传递式 resolved-hyperedge 构造与区域合并伪影，不是证据
  模型不可行。若做 v5.2：需要按链接分别解析 + 对种子序合并伪影的
  防护（例如残差互检后拆分或重种子），仍不得事后放宽阈值。
- 冻结基线全程未动，校验器绿；生产默认（patch/soft routing/selector
  off）不变；CTP 20% 关系尾维持 T0 验证的冻结状态。

## shared_transaction_v1（SCST v3.0）：T0 结构认证 PASSED，T1 战役启动（2026-08-24）

按用户目标（不抛弃失败案例、在冻结基线上升级、真正的重叠机制创新、
底线非劣）裁决执行 SCST v3.0 主线（`experiments/upgrade/shared_transaction_v1/`），
evidence-sinking 的 E0 判决先行落地：生产契约不动（contracts.py 在冻结清单内），
逐变量共享身份走 soft-RDDSM 旁路腿，AOB 上机制结构性静默（T4 预注册预测）。

**失败案例教训的条款化**（详见候选 README 表格）：v5.1 区域合并伪影 →
端点细化拆分修复；v5.0 链式三区域拒绝 → 逐对目标验证（chain_pair_isolation
管线化）；54a 不可辨识 → 内核零分类；P1 零挂载 → T1 先证边界；S1 宿主惰性 →
只挂自然边界；v4-v5.3 跷跷板 → 无状态固定 8 FE；Gate 53 → AOB 预期静默。

**T0（结构认证）gate_passed=true**（`artifacts/upgrade_shared_transaction_v1_t0_v1`，
result_hash b84e5423…，t1_authorized=true）：
- 6 cell × 5 seed（20270501-05）全部 precision=1.0、recall=1.0、
  merged_region_count=0、rejected_hyperedge_count=0、证书图森林；
- v5.0/v5.1 的两个仪器级失败被同时修复：链式 recall 0.333→1.0
  （16 条三区域超边经逐对残差验证后全部正确出证书）；Stage-1 天然合并
  （chain4-mild×2、pairs3-strong×1、hub3-mild×2，共 5 次）全部被
  端点细化拆分修复（升序主分配约定保持，块内 owner 混杂为零）；
- checkpoint relations 从认证证书派生（strength=证书数）——修复 soft-DSM
  边池在 generator v3 上零跨块边导致 relation_count=0 的挂载缺陷
  （p0 收据佐证该缺陷先于本候选存在）；
- 精确 180k FE 边界与双跑重放哈希一致全部通过。

设计迭代记录（发生在烟测阶段，gate 判据从未回调）：第一版 DSM 强边拆分
修复不可用（kNN 候选池不采样块内对）；overlap 判据从 min_residual_size
改为互不包含（shared_width=8 > 3，真拆分曾被误拒；截断包含式假拆分仍被拒）。

**T2（内核契约）实现即绿**（9/9：接受+重复拒绝、fail-closed 静默、选择
上限/排序、边界裁剪、无状态确定性、mini 宿主两臂精确 FE、A0 静默）；
按阶梯顺序在 T1 判定后重跑留档。T3/T4 模块与协议已就绪：
T3 eps_ref 将从 T1 收据冻结（min positive patch-off final / 10），
confirmation seeds 20270511-15 预注册；T4 预注册预测 = AOB 全矩阵逐位恒等 +
零 lane 消耗。生产默认（patch/soft-routing/selector off）全程未动。

## shared_transaction_v1 T1 判定：CTP lane 判负（机制发现），SMP 全合格；T3 收窄 SMP-only（2026-08-24）

`artifacts/upgrade_shared_transaction_v1_t1_v1`（32 臂零失败）：

- **identity_arms_bit_identical=true**：S3/ctp、E3/smp 在仪器安装下与冻结
  recovery-screen 收据四字段逐位一致——T1 仪器在生产路径零税获证。
- **SMP 宿主 10/10 臂合格**：每臂恰 1 个合格边界（stateful→rescue），
  reanchor 全证，24/24（或 16/24）链接双 owner proposal 新鲜；
  proposal 归因横跨全部 10 块（每臂 ~19k 次 strict-best 改进）。
- **CTP lane 失败（gate_passed=false 的唯一来源）**：coverage 阶段
  strict-best 写回集中于每 run 1-4 个块（共 ~850 次改进）——交错持久
  会话一旦某块领跑，移动的全局 best 门槛使旧锚块极少再写回；结构（T0）
  与传播（reanchor=true）均在，唯 §3.4 双 owner 条件永不能满足。
  这是"共享写回经济集中"的收据级机制发现，按预注册判据如实判负。
- 判据修正记录（首轮战役即停即改，产物删除重跑）：reanchor 第一版
  要求锚==边界入口哈希，过严（rescue 斜率探针自身可改进 incumbent）；
  修正为活锚定判据（锚==出生时刻 timeline incumbent，install 播种基行）。

**T3 host 范围预注册（协议在首臂运行前冻结）**：依据 T1 普查收窄为
SMP-only；CTP lane 预算冻结为 0。eps_ref 从 T1 SMP 收据冻结
（chain4=2.4533、pairs3=2.3103）；confirmation seeds 20270511-15 预注册。
T3 screen（20 臂）已启动。

## shared_transaction_v1 T3 screen：判负，根因=互斥划分下 proposal 恒退化（2026-08-24）

`artifacts/upgrade_shared_transaction_v1_t3_screen_v1`（20 臂零失败）：

- contract/reachability/非劣全绿；**at_least_one_strict_win=false**。
- 全部 10 对 A0/A1 终值逐位相同；每 run 内核消耗 8 FE、接受 0 次。
- **收据级根因**：每候选评估误差恒等于边界前误差 → 候选==incumbent。
  互斥划分下共享坐标 j 只被主块 A 移动；B 的"proposal 快照"在 j 上只是
  A 最后设定值的拷贝 → P_j=[v,v] 恒退化，median/mean 无事可仲裁。
  **结构性结论：冻结四动作的互斥划分架构从不表达共享变量的值冲突；
  仲裁型事务（FEA 语义）在该架构上原理性空转。**
- 按预注册止损线：v1 候选关闭于 T3；保留 A0；T5/confirmation 不启动；
  生产默认未动（floor 天然保持）。

**v2 方向（由根因直接导出，新候选目录+新协议）**：结构认证的共享坐标
重整内核——owner 非主块移动后，j 的全目标切片最优点偏移；以 3 点二次
拟合（每坐标 3 FE，仅信完整目标反馈，无 owner 权威/分类器/状态）把 j
重新对中。复用 v1 的 T0 认证资产与 T1 SMP 边界资格。

## shared_transaction_v2/v3/v4 三连 screen：全部判负，边界干预空间以递进收据穷尽（2026-08-24）

- **v2 重整内核**（3 点二次拟合，12 FE）：机械上完全工作（每臂 4/4 坐标
  strict-best 接受，边界即时改进 4541→4537）；pairs3 geoR 0.9685、
  CI [0.9487, 0.9902]（CI 上界<1.0，统计显著改善）但未达 0.95 严格线；
  chain4 混合（geoR 1.046，一个 1.29 败例）。**撞尺度公理**：12 FE 微窗口
  对 2.82M 轨迹，终值被锚点扰动方差支配（F2/Gate47b 家族复现）。
- **v3 联合救援**（认证链接合并 scope，从 rescue 窗口划 20k FE/链接）：
  accepted_links=0 全臂，默认 σ warm-up 在 1e6 条件数 192 维 scope 上
  起不来；预算被偷 → chain4 一致回归（geoR 1.0665，CI 全>1.0）。
- **v4 联合救援+rescue 原生 warm-up**（STATE_RESCUE 配方逐字照抄）：
  仍 accepted_links=0，终值与 v3 逐位相同（拒绝评估不改变下游状态）。
  **闭环诊断：边界处 incumbent 已被 1.2M FE stateful 打磨到全新会话无
  收益空间；SMP stateful→rescue 边界的干预空间穷尽。**
- 全部判定按各自冻结协议执行；A0/生产默认未动（floor 全程保持）；
  四个候选目录与收据完整留档（upgrade_shared_transaction_v1..v4）。

**证据指向的下一候选**（未启动）：CTP 侧 coverage 阶段的认证 scope
干预——T1 已证 CTP 写回集中在领跑块（1-4/10），集中锁定发生在 coverage
内部；早期（而非边界）对认证链接 scope 的联合访问是唯一未探索的
预算尺度窗口，与 G5（拓扑条件化生命周期）同族。

## shared_transaction_v5 screen：判负 + 实现缺陷定位（证书级 vs 块对级选择）（2026-08-24）

`artifacts/upgrade_shared_transaction_v5_screen_v1`（20 ctp 臂零失败）：

- chain4 geoR 0.9984 CI [0.9882, 1.0086]（持平）；pairs3 geoR 0.9662
  CI [0.8715, 1.039]（3W/2L，含 0.795 大胜与 1.080 大败，高方差）。
- **accepted_links=0 全臂**：执行缺陷定位——`_links` 取了 24 张证书
  （每块对 8 个共享变量 → 8 张重复证书）而非 3 个唯一块对，25% 上限
  被摊成 24×4.7k FE 的重复 scope 会话，warm-up 不足全部零写回。
  冻结协议字面（"all T0-certified links"）被执行成了证书级；README
  本意（每链接 20k FE）是块对级。判定按冻结协议如实记录 false。
- v6 = 去重修正（唯一块对 × 完整 20k FE/对），已启动。

## shared_transaction_v6 screen：判负，第三个机制事实锁定（2026-08-24）

`artifacts/upgrade_shared_transaction_v6_screen_v1`（20 ctp 臂零失败，
scope 正确：3 唯一块对 × 200 维 × 20k FE）：chain4 geoR 0.9878
CI [0.957, 1.0165]；pairs3 geoR 0.999 CI [0.9617, 1.0659]；
**accepted_links 仍为 0（全部 30 个联合会话）**——全新 coverage 配方
会话在 20k FE 内无法赢得一次 strict-best 写回。

**机制事实三联**（v1-v6 收据收敛）：(1) 互斥划分不表达共享变量值冲突
（仲裁无输入）；(2) 一次性边界干预（微窗口或重启会话）无法对抗
2.82M FE 轨迹的扰动方差；(3) 全新重启会话竞争不过宿主的持久机制
（coverage 的 853 次写回来自交错持久状态）。唯一稳定正信号：v2 的
重对中修复本身总是被接受（4/4×全臂），失败在时机与频率。

**v7（证据链终点设计）**：逐访问连续协调——owner 块每次 stateful
visit 结束后，对其认证链接的共享坐标各做一次 3-FE 二次拟合重对中
（每次 visit ≤24 FE，全程 ≤1% Phase-II 预算硬上限，超限即停）。
确定性时刻表、零在线选择信号、strict-best-only；对应 FEA/OCC 文献的
周期性共享变量同步，把 v2 的有效局部修复聚合到访问尺度。

## v7 逐访问重对中 screen：最强正信号；统计口径 bug 修正；v8 fresh-seed 确认启动（2026-08-24）

`artifacts/upgrade_shared_transaction_v7_screen_v1`（20 smp 臂零失败）：

- **pairs3-strong geoR 0.8801**（4W/1L，胜例 0.79–0.85×，即 12–21% 终值改善），
  412 次重对中被 strict-best 接受、内核仅耗 2.2k FE（≤1% 上限内）；
  chain4 geoR 0.9861（2W/3L，79 次接受，48 FE/run）。
- **统计口径修正**：`_paired_bootstrap` 返回 log 尺度，而 strict_win 判据
  直接与其比较线性阈值 1.0（bug 继承自 v1 T3 模块）。修正为 exp() 后比较，
  阈值未动；五个受影响模块全部修正并重算——v1–v6 判定不变（皆大边距失败），
  **v7 按正确口径 CI 上界 1.0041 > 1.0，差 0.0041 未过**（初版误报 pass 已撤回）。
- v8 = v7 机制原样（内核别名导入）、修正统计、预注册 fresh seeds
  20270511-15（从未运行过、on-the-fly T0 认证）的确认战役；判定不重用
  v7 的 5 个 screen seeds。confirmation 战役运行中。

## v8 fresh-seed 确认：判负，v7 信号未复现——八候选证据链闭环（2026-08-24）

`artifacts/upgrade_shared_transaction_v8_confirm_v1`（20 臂零失败，
seeds 20270511-15，on-the-fly T0 认证全过）：

- chain4 geoR 1.0983 CI [0.9442, 1.325]；pairs3 geoR 1.0368
  CI [0.9231, 1.1571]；非劣与 reachability 绿；strict_win=false。
- v7 screen 的 pairs3 强信号（0.8801）在从未运行过的 seeds 上未复现；
  合并 n=10：pairs3 geoR 0.9553 CI [0.8692, 1.0536]——微弱正漂移，
  远低于预注册门槛。
- 机制接受修复 894 次（pairs3）/72 次（chain4），内核 FE ≤1% 上限内——
  **局部修复总是被接受，但 ≤1% 预算的连续微协调无法主导宿主轨迹方差。
  尺度公理的最终确认。**

**八候选总结**（v1 仲裁退化 / v2 微窗口方差 / v3v4 晚期无空间 /
v5v6 重启零写回 / v7v8 逐访问不复现）：在冻结四动作架构内、generator
cells 上、按预注册严格判据，任何边界/访问尺度的共享变量协调机制
都无法清除 0.95&CI 门槛——宿主预算结构对这些 cell 已近最优。
正资产：T0 认证仪器（PASSED）、T1 零税仪器与写回集中发现、
互斥划分不表达值冲突的结构性发现、以及八份完整判定书。
下一步候选（未启动）：evidence-sinking E1（CTP tail 的证据条件化，
G3 机制证据 geo 0.2897，唯一未消耗的历史正收益点）。

## evidence_sinking_e1_v1：判负，G3 未配对证据不可迁移的边界预测兑现（2026-08-24）

`artifacts/upgrade_evidence_sinking_e1_v1`（30 Phase-I + 60 臂零失败，
seeds 20270601-05 全新，机制 = tail 份额 0.20×(1+min(1, mean_strength/0.08))
运行前常量替换，S1 恒等按构造成立）：

- **S1 逐位恒等 ✓（5/5 seeds，final/route/hash 全同）**——保守默认公理
  再次完美保持。
- **S2-S5 非劣 FAIL**：S3 geoR 1.7276 CI [1.376, 2.305]、S4 1.3912
  CI [1.135, 1.700]（均显著回归）；S5 1.108；S2 1.049（方差极大
  0.553–2.065）。
- S6 geoR 0.9965 CI [0.772, 1.228]（持平，未达 0.98）。
- **机制解读**：加大 MMES 尾 = 偷走 relation-cover polish，而 polish 的
  边际价值在 S3/S4 高于尾部。G3 的 0.2897 是"尾 vs 无尾"的存在性证据，
  不是">20%"的剂量证据；v4.0 计划预注册的边界预测（"若配对后不兑现，
  如实记录为未配对机制证据不可迁移"）兑现。按止损线：E1 关闭，
  CTP tail 保持冻结 20%。
- 运行注记：战役曾被宿主环境中断一次（40/60），--resume 续跑完成，
  零失败、零重复臂。

**九候选全景（v1-v8 + E1）**：全部按冻结协议判定，无一通过；每份
失败都有收据级根因；生产默认全程 frozen（非劣底线严格保持）。
剩余未消耗的候选方向：E2（SMP 预算连续化）/E3（GCB source window
加权）——同杠杆类但先验弱于 E1（E1 是唯一有历史正收益点支撑的）；
或转入论文线（C1-C6 + 九候选边界地图 + T0 认证仪器）。

## 外部代码审阅（51 文件快照）逐条裁定（2026-08-24）

来源：用户提交的独立审阅（基于不完整上传快照，无运行环境）。逐条对照
真实仓库核实：

| # | 审阅主张 | 裁定 | 证据 |
|---|---|---|---|
| P0-1 | coordination/state.py 缺失，run_arac_oc 无法导入 | **证伪（快照伪影）** | 文件存在于 src/arac/coordination/state.py；T1/E1 战役刚跑通大量生产导入 |
| P0-2 | Phase-I 混合差分步长不一致，可分函数产生非零残差 | **证实（数学边界，非实效 bug）** | phase1.py:135/146：单块步长 c·s/√n vs 联合 c·s/√(n_l+n_r)；均匀 span 下残差为常数被 median+3MAD 稳健阈值裁掉（零关系 case 实测 0 关系佐证）；非均匀 span/elliptic 条件下伪关系原理上可越阈——Phase-I 的可分性保证依赖稳健阈值而非精确零残差 |
| P0-3 | 非零 optimum 产生伪交互 | **证实（已知已文档化）** | generator v3 强零最优值约定（2026-08-23 进度日志）；本审阅独立再发现 |
| P0-4 | sparse pilot 默认 315k FE > 180k 上限 | 属实但作用于 **OC 线（已被恢复基线取代），非冻结四动作链** | pilot 只被 overlap_core/phase1_overlap_pilot 引用 |
| P0-5 | OC loop value gate 回滚丢弃已观察最优点 | **证实（OC 线语义）** | loop.py:519-527：operator_value_ratio 未达即 restore_archive——违反 best-observed 不变量；恢复基线链（strict-best 永不回退）无此问题 |
| P0-6 | SMP hybrid rescue/GCB 异质 population 崩溃路径 | **未验证（需定向复现）** | 边缘拓扑下的可达性主张，尚未复现 |
| P0-7 | Shared Patch 入口 NameError 等 | **证实为潜伏死代码** | loop.py:24 只 import dataclass 非 dataclasses 模块，:494 调用 dataclasses.replace 必 NameError——但 patch 默认关且 P1-P4 已按协议退役，分支不可达 |
| P0-8 | v2 snapshot 序列化未初始化 scratch | **未证实** | optimizers.py:425 y=np.full 初始化（非垃圾内存）；hash 不稳定主张需复现 |
| 边界 | additive conflicting ownership 聚合目标不可辨识 | **证实并与 Gate 54a 独立收敛** | 审阅者的 (x−a)²+(x−b)² 等价性与 54a 的 Σw(v−o)²=(Σw)(v−m)²+const 同构；其"oracle 结构下 conflict_score=0"推论与 v1-T3 的 proposal 恒退化收据完全吻合——两个独立分析强化同一结论 |

**处置**（映射治理结构）：冻结锚不动（经验行为已被逐位复现锚定）；
P0-5/P0-7 在 OC 线——标注 retired 或修复进 upgrade 命名空间；
P0-2/P0-3 写入论文诚实边界节（Phase-I 适用边界：近零 optimum +
稳健阈值脆弱性）；P0-6/P0-8 立定向复现任务后再定。

## 论文入口复活：P0-4 实证 + blessed 配置确认（2026-08-24）

- **P0-4 逐位证实并在当前 HEAD 实证复现**：sparse screening 默认参数
  （bucket_size=8、rounds=8、5 anchors）在 1000 维需 315,040 FE >
  Phase-I 180,000 上限；`run_arac_oc` 默认参数在 S3 上以
  "overlap discovery did not populate the Phase-I archive" 崩溃
  （discovery 预算不足返回零消费 incomplete → 空 ledger 异常）。
  OC 论文入口默认不可用，与外审 P0-4 一致。
- **blessed 配置定位**：`aob24_overlap_applicability_audit.py` 的
  PHASE1_KWARGS（anchor_count=5、step=0.25、rounds=12、bucket_size=16、
  max_candidate_pairs=128）是历史上跑通 1000 维 AOB-24 的调用形态，
  screening 预算 121,020 ≤ 180k。S3 冒烟（当前 HEAD、完整 3M OC 管线、
  seed 20260824）已启动。
- 处置方向：论文入口文档必须显式携带 phase1_kwargs（默认参数是
  小维度设计遗留）；是否把默认值修为 blessed 配置待裁决
  （会改变依赖默认值的历史脚本行为）。

## 论文入口复现地图：C1/C3 在当前 HEAD 活体复现（2026-08-24）

用户裁决：AOB 之外（校准 cell）的入口冒烟无意义——据此停止并改走
AOB 主张的活体复验。结果（详见 docs/arac-oc-paper-reproduction-map.md）：

- **C1/C2 ✅ 活体复现**：soft-RDDSM 发现 on AOB 六 case（当前 HEAD、
  `artifacts/upgrade_paper_entry_c1_headrev_v1`）——召回 0.789(R6)–
  1.000(R2) 与冻结区间逐端点吻合、精度 1.0 全程、E1/R1 零误报、
  预算逐笔对账。
- **C3 ✅ 活体复现**：Gate 41a 离线评估（`upgrade_paper_entry_c3_headrev_v1`）
  四检查全绿（21/3 vs HCC、geo 0.0645、oracle 命中 ≥0.95、全 case
  ≤1.10× 历史列）；其活体基质（四臂 AOB 执行）由 T1 identity 逐位
  一致早已证实。
- **入口地图澄清**：C4=24 维受控生成器主张（边界 #5）；统一 OC 循环
  非 AOB 入口（边界 #2）且默认 pilot 参数在 1000 维崩溃（P0-4 实证，
  blessed kwargs 必须显式传递）；C5 只能从 tag v5.3-prealation 复现
  （episodes 已删，runner 断裂）；P0-5/P0-7 默认配置均不可达。
