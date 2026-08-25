# LSGO 重叠问题领域背景与文献地图（长期记忆版）

调研日期：2026-08-24。
检索平台：OpenAlex / Crossref / arXiv REST API（引用数为当日快照，OpenAlex/Crossref 双口径）。
定义来源：本地 PDF 正文（Two-Phase CC.pdf 等）+ API 元数据。
用途：领域背景定义、方法谱系、基准沿革的权威参考；后续会话讨论 ARAC 定位时以此为底图。
与 `docs/arac-oc-related-work-gaps.md`（定位叙事）和 `references/overlap_AOB_literature_survey_2026-07.md`（近作详解）互补，不重复其内容。

## 一、领域背景定义（以 HCC 原文表述为准，arXiv:2503.21797）

- **LSGO**：大规模全局优化，决策维度 d≥1000 的连续黑盒最小化。变量交互结构决定可分性：
  完全可分 / 部分可分（多个不可分子分量）/ 完全不可分。
- **CC（协同进化）**：Potter & De Jong 1994 开创的分而治三步循环——分解（disjoint 子分量）
  → 各子分量独立优化（round-robin 或调度）→ 组合成完整解。效率高度依赖分解质量。
- **重叠问题（overlapping problem）**：变量属于两个及以上子空间（overlapping variables），
  使"子空间两两不交互"的理想分解条件被结构性打破。HCC 原文定义理想分解为：
  ①任意两子空间互不交互；②子空间内变量两两交互。重叠变量的存在使 ① 不可能同时满足，
  只能"排除重叠变量引起的交互后近似满足"。
- **重叠给 CC 的三大难点**（HCC 原文归纳）：(1) 理想分解难以可靠达成；
  (2) **耦合问题**——在一个子空间直接改重叠变量值会显著影响其他子空间，使其已完成的
  优化失效（这是所有"共享变量值协调"机制要解决的核心问题）；(3) NDA（非分解算法）
  前期下降快 vs CC 后期精修强的两阶段互补性。
- **conforming / conflicting 重叠**：共享变量在所涉各子函数中最优值一致（conforming，
  协调容易、改善可传播）vs 不一致（conflicting，存在真值冲突，必须仲裁）。术语出自
  OCC/FEA 一线（本地 PDF 有 OCC 全文可核）。
- **重要事实**：CEC2013LSGO 15 个函数中仅 2 个含重叠（HCC 原文表述；F11 无重叠 vs
  F13 有重叠是标准对照对）。重叠专用基准长期稀缺，是 AOB/RB 系列出现的动机。

## 二、方法谱系（四条线）

### 线 1：变量交互识别 / 分解（把分组做对）

| 方法 | 出处（已核验） | 要点 |
|---|---|---|
| DG | Chen, Weise, Yang, Tang, PPSN 2010, doi:10.1007/978-3-642-15871-1_31；期刊版 Omidvar et al., TEVC 18(3) 2014, doi:10.1109/TEVC.2013.2281544（744 引） | 差分分组检测两两交互的开山 |
| DG2 | Omidvar et al., TEVC 21(6):929-942, 2017, doi:10.1109/TEVC.2017.2694221（353 引） | 无参阈值 + 非平衡子分量（MiVD/MaVD 为其正文策略）；当前事实基线。**只有 TEVC 2017 一个版本** |
| RDG | Sun, Kirley, Halgamuge, TEVC 22(5):647-661, 2018, doi:10.1109/TEVC.2017.2778089（211 引） | 递归分组，把两两检测降为递归层级。**注意：非 Qiao/TCYB（旧笔记误记）** |
| RDG2 | Sun, Li, Ernst, Omidvar, CEC 2019, doi:10.1109/CEC.2019.8790204（82 引） | RDG 的重叠版本：在共享变量处断开链接以降维。重叠分解方向的奠基文 |
| RDG3 | **无正式提出论文**（仅见于 CBCC-RDG3 等应用文献，HCC 原文将其与 DOV 并列为"面向特定问题"的重叠分解） | 引用时需查原始出处，不可笼统引用 |
| XDG | Sun et al., GECCO 2015, doi:10.1145/2739480.2754666（140 引） | 直接/间接交互 |
| ERDG | Yang et al., TEVC 2020, doi:10.1109/TEVC.2020.3009390（95 引） | 高效递归 DG |
| EDG | Kumar, Das, Mallipeddi, TEVC 2022, doi:10.1109/TEVC.2022.3230070（44 引） | **注意：非 Qiao/eigenvector（旧笔记误记）** |
| MDG / DDG / IRRG | TEVC 2022 doi:10.1109/TEVC.2022.3144684；TCYB 2022 doi:10.1109/TCYB.2022.3158391；TEVC 2022 doi:10.1109/TEVC.2022.3216968 | 合并式/乘法可分/增量递归 |
| GDD | Zhang et al., IEEE TSMC:Systems 2022, doi:10.1109/TSMC.2022.3212045（36 引） | 图论最小顶点分隔符分解 + **自带重叠函数生成器**（第三条重叠基准来源） |
| OEDG | Tian, Chen, Du, Tang, Jin, TEVC 29(6):2272-2286, 2024/2025, doi:10.1109/TEVC.2024.3390719, arXiv:2404.10515 | 两阶段：有限差分识别子分量与共享变量 → SUD/SD 精化；提出考虑拓扑/重叠度/可分性的 RB 系列基准（RB-ILD 为成员，名称出自正文） |
| RLDO | Tian et al., TEVC 2025, doi:10.1109/TEVC.2025.3622888 | RL 学分解知识（OEDG 同组后续） |
| CDM | Tian et al., IEEE TAI 2024, doi:10.1109/TAI.2024.3373391 | 加性/非加性复合结构分解 |

### 线 2：重叠共享变量的处理策略（本领域的核心分歧点）

| 策略 | 代表工作 | 机制 | 代价 |
|---|---|---|---|
| 断链降维 | RDG2 (CEC 2019)、Blanchard et al. (PPSN 2021, doi:10.1007/978-3-030-85672-4_19) | 共享变量处断开，各分量独立 | 丢失跨分量信息 |
| 单 owner 贡献分配 | Jia, Mei, Zhang, CBCC-overlapping, TCYB 52(6):4246-4259, 2022, doi:10.1109/TCYB.2020.3025577（47 引）；前作 Jia, IEEE Access 2019 doi:10.1109/ACCESS.2019.2897282 | 共享变量归贡献最大的子分量，贡献奖励 + round-robin 骨架 | owner 误判即丢信息 |
| 多 owner + 完整目标仲裁 | FEA: Pryor, Peerlinck, Sheppard, SSCI 2021, doi:10.1109/SSCI50451.2021.9659875 | 各 factor 注入自己的值，谁全局适应度好谁写回 | 需要信息性候选；FE 开销 |
| 多 owner + 资源耦合 | OCC: Komarnicki, Przewozniczek, Tinós, Li, GECCO 2024, doi:10.1145/3638529.3654171；前身 Song et al., SMC 2017, doi:10.1109/SMC.2017.8123206 | 共享变量多重归属；组件资源受共享伙伴影响 | 不解决共享变量取值协调本身 |
| 贡献合并 + 资源平衡 | CCMRO: Chen, Du et al., CEC 2023, doi:10.1109/CEC53210.2023.10254019 | 基于贡献的子分量合并 + CCRO 资源分配 | 引用尚少（0 引） |
| 两阶段 NDA+CC + 数学性质分解 | HCC: Qiu, Guo, Ma, Gong, GECCO 2025 Companion, doi:10.1145/3712255.3726560, arXiv:2503.21797 | Phase1 NDA 全局探索 + Phase2 RDDSM 递归 DSM 分解 + CMA-ES | RDDSM 分解 oracle-fed（ARAC 源码审计结论，见 docs/sota-oracle-confirmation.md） |

### 线 3：资源分配 / 调度（CC 通用，重叠无关但常被组合）

| 方法 | 出处（已核验） | 要点 |
|---|---|---|
| CBCC | Omidvar et al., GECCO 2011, doi:10.1145/2001576.2001727（125 引） | 贡献度取代 round-robin 的开山 |
| CCFR | **Ming Yang et al., TEVC 2017, doi:10.1109/TEVC.2016.2627581（126 引）**——即 "Efficient Resource Allocation in CC for LSGO"，旧笔记把 CCFR 与此文列为两篇是错的 | 贡献衰减中断 |
| SACC | Mahdavi et al., Applied Intelligence 2017, doi:10.1007/s10489-017-0926-z | 敏感性分析定预算 |
| CCFR3 | Yang et al., ESWA 2022, doi:10.1016/j.eswa.2022.117397 | CCFR 改进 |
| DCCC | Xu, Luo, Lin, Chang, TEVC 2023, doi:10.1109/TEVC.2022.3201691（31 引） | 难度×贡献双轴（"困难子问题投入难转化"与 ARAC 跷跷板直接对应） |
| AOS survey | Pei et al., IEEE TAI 2025, doi:10.1109/TAI.2025.3545792 | 自适应算子选择综述；extreme credit / 秩归一 |
| 新作 | Liu et al., TEVC 2025, doi:10.1109/TEVC.2025.3629151（贡献+启发式集成）；Chen et al., TEVC 2025, doi:10.1109/TEVC.2025.3564335（多目标自感知分组） | 2025 资源分配线仍活跃 |

### 线 4：基准测试集沿革

| 基准 | 出处 | 重叠覆盖 |
|---|---|---|
| CEC2008/2010 LSGO | 技术报告（不入 Crossref/OpenAlex，需按技术报告格式引用） | CEC2010 F1-F20 为 DG2 等标准套件 |
| CEC2013 LSGO | Li, Tang, Omidvar, Yang, Qin, 2013 技术报告（372 引，无 DOI） | 仅 2 个重叠函数（F13 典型） |
| 基准设计方法学 | Omidvar et al., Information Sciences 2015, doi:10.1016/j.ins.2014.12.062 | 部分可分结构构造 |
| GDD 生成器 | TSMC 2022（同上） | random + complicate overlap 类型 |
| RB 系列（含 RB-ILD） | OEDG, TEVC 2024 | 拓扑/重叠度/可分性可控 |
| AOB | HCC, GECCO 2025 Companion | 7 组件解耦：基函数、子空间大小表、维度置换、shift、权重、旋转、各子空间重叠数 Γ；1000 维、重叠度 0/1/3/5/7/10、24 问题 |
| hop-like 结构 | Przewozniczek, Frej, Komarnicki, GECCO 2026, doi:10.1145/3795095.3805173 | 真实 NP 难问题的新结构特征建模 |

综述锚点：Omidvar, Li, Yao, "Population-Based Metaheuristics for Large-Scale Black-Box
Global Optimization" Part I (TEVC 26(5):802-822, doi:10.1109/TEVC.2021.3130838) /
Part II (823-843, doi:10.1109/TEVC.2021.3130835)。2022 后无新的 LSGO 专属大综述。

## 三、2025-2026 前沿动态（新增于本调研）

1. **Qiu/Gong 组（SCUT）三连**：HCC（GECCO 2025 Companion）→ LCC（GECCO 2026,
   doi:10.1145/3795095.3805053, arXiv:2504.17578，NN selector 调度分解策略）→
   LH-CC（GECCO 2026, doi:10.1145/3795095.3805054, arXiv:2604.01241，异构 LSGO 学
   优化器配置）。该组正快速占领"重叠 + 学习式 CC"交叉位。
2. **ECNU Tian/Tang 组**：OEDG（TEVC 2024）→ CDM（TAI 2024）→ RLDO（TEVC 2025）→
   CC 多任务框架（TEVC 2026, doi:10.1109/TEVC.2026.3699585）。分解知识学习线。
3. **Komarnicki/Przewozniczek 组**转向链接发现与问题结构分析（FOGA 2025 两篇、
   GECCO 2025 非对称依赖、GECCO 2026 hop-like），OCC 本体暂无直接续篇。
4. **AOB 仍无第三方使用者**（2026-07 调研结论仍成立）；RB 系列使用者也少——重叠
   benchmark 生态整体远未标准化，第三方使用本身即有引用价值。
5. 资源分配线 2025-2026 仍持续产出（见线 3 末两行），但全部在 disjoint 分解语境。

## 四、共享变量处理策略与 ARAC 的映射（简表，详见 related-work-gaps.md）

- ARAC 的完整候选仲裁 + strict-best 写回 = FEA 原则 + counted probe 证据 + fail-closed
  预算契约（贡献在"预算受限下何时仲裁值得"，Gate 29/30 语境证据 36/36）。
- ARAC 的 budget-fed 发现 vs HCC 的 oracle-fed RDDSM = 信息条件边界的差异化主张。
- ARAC 已证伪/边界：运行时 conforming/conflicting 分类不可辨识（Gate 54a 解析证明）；
  仲裁移植到派发路径零接受（Gate 53）；调度天花板 ~1.6%（G3a）——这三条分别约束
  线 2 策略表的上三行，是文献中未被系统研究过的阴性边界。

## 五、本调研完成的出处纠错（旧笔记需更新）

1. RDG = Sun, Kirley, Halgamuge, TEVC 2017/2018（非 Qiao/TCYB）。
2. EDG = Kumar, Das, Mallipeddi, TEVC 2022（非 Qiao/eigenvector）。
3. DG2 只有 TEVC 2017 版本；"GECCO 2014 会议版"不存在。
4. CCFR 与 "Efficient Resource Allocation..."（TEVC 2017, Ming Yang）是同一篇。
5. CBCC 原始出处 = Omidvar et al., GECCO 2011。
6. CEC2013 报告作者 = Li, Tang, Omidvar, Yang, Qin（无 Xin）。
7. LIM（LSGO 变量交互方向）与 DGFD 未检索到正式论文，疑为记忆混淆；MiVD/MaVD 为
   DG2 正文级策略名，引用需翻原文。
