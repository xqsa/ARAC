# 大规模重叠（Overlapping）LSGO 近两年文献调研 与 AOB 使用情况核查

调研日期：2026-07-21
附件论文：Wenjie Qiu, Hongshu Guo, Zeyuan Ma, Yue-Jiao Gong. *A Novel Two-Phase Cooperative
Co-evolution Framework for Large-Scale Global Optimization with Complex Overlapping*
（HCC + RDDSM + AOB；arXiv:2503.21797，2025-03；GECCO 2025 Companion，DOI: 10.1145/3712255.3726560）。

## 一、结论速览

1. 近两年（2024–2026）专门针对"大规模重叠问题"的代表工作共 5 篇，见下表；其中分解类 2 篇
   （OCC、OEDG）、框架/学习类 3 篇（HCC、LCC-CMAES、LH-CC）。
2. **截至目前，没有发现任何其他论文在 AOB 测试集上做实验。** Semantic Scholar 收录的 4 篇
   引用文献已全部逐篇核查全文：两篇同组 GECCO 论文只在参考文献中引用 HCC，实验分别用
   CEC2013LSGO（F12–F14 重叠函数）和自建 Auto-H-LSGO；另外两篇是无关领域的顺带引用。
   Google Scholar 显示被引约 5 次，量级与 S2 一致，不存在遗漏的大批使用者。
3. AOB 目前仍是"只有提出者自己用过"的 benchmark——这恰好说明 ARAC 在 AOB 上做
   action-ceiling 验证属于首批第三方使用，没有现成的外部对比数据可依赖。

## 二、近两年代表工作详解

### 1. OCC — Overlapping Cooperative Co-Evolution（GECCO 2024）
Marcin M. Komarnicki, Michal W. Przewozniczek, Renato Tinós, Xiaodong Li.
DOI: 10.1145/3638529.3654171

- 干了什么：传统 CC 把每个变量分进唯一子组件（disjoint），对含共享变量的重叠问题不存在
  完美分解。OCC 反其道而行——**故意构造重叠的子组件**，允许共享变量被多重分配，并设计了
  组件间相互影响的计算资源分配机制（一个组件分到的资源受与其共享变量的其他组件影响）。
- 实验：CEC'2013 LSGO 重叠函数 + conforming/conflicting 子函数设置；结果在 conforming
  重叠函数上优于 CBCC、CC-RDG3、SHADE-ILS 等。
- 与 HCC 的关系：HCC 论文将其作为主要对比/引用对象；OCC 发表于 HCC 之前，未使用 AOB。

### 2. OEDG — An Enhanced Differential Grouping Method for Large-Scale Overlapping
Problems（IEEE TEVC 2024）
Maojiang Tian, Mingke Chen, Wei Du, Yang Tang, Yaochu Jin. arXiv:2404.10515

- 干了什么：针对重叠问题的**分解精度**改进。DG 类方法在重叠变量存在时交互判断失效，
  OEDG 增强差分分组以正确识别重叠变量归属，属于"把分组做对"的路线。
- 同组延伸：Tian 等 2024 在 IEEE TAI 另有 Composite Decomposition Method（CDM），
  处理加性/非加性混合可分的复合结构。

### 3. HCC — 附件论文本身（GECCO 2025 Companion）
- 两阶段框架：Phase1 由 NDA（MMES）做全局探索，Phase2 切换为 CC（RDDSM 分解 + CMA-ES）
  分而治之；RDDSM 基于设计结构矩阵的数学性质递归识别重叠变量，达到理想分解。
- 同时提出 AOB（Auto Overlapping Benchmark）：把基函数池（Schwefel/Elliptic/Rastrigin/
  Ackley）与结构组件（子空间大小、排列、shift、权重、旋转、重叠数 Γ）解耦，自动生成
  1000 维、重叠度 0/1/3/5/7/10 的 24 个问题。

### 4. LCC-CMAES — Advancing CMA-ES with Learning-Based Cooperative Coevolution
（GECCO 2025，同组）
Wenjie Qiu, Hao Guo, Zeyuan Ma, Yue-Jiao Gong. arXiv:2504.17578

- 干了什么：把 CC 中"选哪种分解策略"建模为 MDP，用 PPO 训练一个神经网络 selector，
  在 RD / MiVD / MaVD 策略池中按优化状态动态选择，替代人工设计的策略选择规则。
- 与 AOB 的关系：**引用了 HCC 论文（参考文献 [54]），但实验只用 CEC2013LSGO**（含
  F12–F14 Overlapping Functions），未使用 AOB。已全文检索确认无 "AOB" 字样。

### 5. LH-CC + Auto-H-LSGO — A Learning-Based Cooperative Coevolution Framework for
Heterogeneous LSGO（GECCO 2026，同组）
Wenjie Qiu, Zixin Wang, Hongyu Fang, Zeyuan Ma, Yue-Jiao Gong. arXiv:2604.01241

- 干了什么：面向"异构 LSGO"（各子问题维度、地貌不同），用 meta-agent 为每个子问题动态
  选择最合适的 optimizer（而非固定低维优化器）；提出 Auto-H-LSGO benchmark，可配置
  重叠结构与子问题类型自动生成 3000 维实例。
- 与 AOB 的关系：**引用了 HCC 论文（参考文献 [36]），但使用自建 Auto-H-LSGO**，
  未使用 AOB（已全文检索确认）。可视为 AOB 思路在"异构"方向上的演进。

## 三、引用 HCC 论文的文献核查明细（Semantic Scholar，2026-07-21 抓取）

| 引用文献 | 年份/出处 | 是否在 AOB 上实验 | 说明 |
|---|---|---|---|
| LCC-CMAES | GECCO 2025 | 否 | 实验用 CEC2013LSGO F12–F14 |
| LH-CC | GECCO 2026 | 否 | 用自建 Auto-H-LSGO benchmark |
| Systems optimization for enhancing community resilience (review) | 2026, Environment Systems and Decisions | 否 | 综述类顺带引用，领域无关 |
| Competitive Parallel Animated Oat Optimization | 2026, MiTA | 否 | 元启发式算法，related work 引用 |

数据源：Semantic Scholar Graph API（arXiv:2503.21797 的 citations 端点，共 4 条）；
Google Scholar 页面显示被引约 5 次（arXiv 页面 "Cited by 5"），与 S2 量级一致。

## 四、对 ARAC 项目的含义

- AOB 尚无第三方实验数据，ARAC 的 action-ceiling 结果无法与外部工作直接对比，
  native HCC baseline 必须由本项目同 runner/同 checkpoint 自产（与 AGENTS.md §4 一致）。
- 文献定位上，ARAC 的"Phase1 证据 -> 显式动作 -> Phase2 执行"与 LCC/LH-CC 的
  "学习式 selector"路线不同：他们学的是分解策略/优化器选择，ARAC 验证的是显式动作
  在同 FE 预算下的因果收益，可在 related work 中以此区分。
- 若后续需要外部有效性论据，OCC（GECCO 2024）与 OEDG（TEVC 2024）是 AOB 之外
  仅有的两条独立技术路线，适合作为重叠问题处理方式的对比背景。

## 附：原始抓取文件

- `references/raw/s2_citations.json` — Semantic Scholar 引用列表原始返回
- `references/raw/lcc_cmaes.pdf/.txt`、`references/raw/lhcc.pdf/.txt` — 两篇同组论文全文及检索用文本
- `references/raw/scholar_overlap*.csv` — scholar 插件检索结果
