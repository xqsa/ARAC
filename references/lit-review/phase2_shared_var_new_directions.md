# ARAC-OC 阶段二共享变量机制：文献增量调研笔记

日期：2026-08-22
目的：为"阶段二对共享变量的处理"寻找已知文献（CBCCO/OCC/HCC/RDG3/ORDG/FEA/CCFR/DCC/AOB）
之外的创新机制。来源：Google Scholar 检索（scholar 插件当时不可用，改用网络学术检索），
所有条目均来自本轮实际检索结果。

## 0. 对既有升级失败模式的诊断（用于定位新机制）

已尝试的升级全部是"硬裁决"族：
- 完整目标仲裁（tournament 选一值）→ Gate 53 在 conforming 实例上零接受；
- CBCCO 式归属（选一 owner）→ 尚未证明超越消融臂；
- G_coupled / G_int / lagged EMA（选一动作/优先级）→ 预测 Spearman 均不过阈。

共同特征：**离散、赢者通吃、一次性**。文献检索提示了三个连续的、非赢者通吃的机制族。

## 1. 软共识：变量复制 + 增广拉格朗日（最强的差异化方向）

- Distributed Derivative-Free Optimization Using Inexact ADMM and Trust-Region Methods
  (arXiv:2510.27396, 2025)。黑盒 DFO 场景下把链式耦合问题分解为**重叠块**，
  共享变量复制副本 + 一致性约束 + 二次罚，信任域无导数求解子问题。
  这是"重叠分解 + 共识约束"在黑盒优化中的直接先例，但**不在进化 CC 文献内**。
- Boyd et al. consensus ADMM：u 变量是"不一致性的累加和"（原文称 PI control），
  协调通过对局部副本求平均完成，不要求凸性假设即可实践使用。
- Cooperative Learning (Ding & Tibshirani, arXiv:2112.12337)：多视图学习的
  "agreement penalty"——通过调节一致性罚权重在连续谱上插值 early/late fusion。
- 多能源系统文献：Nash 谈判 + ADMM 求解交互变量已成标准做法
  （Processes 2025, 13(7):2022；Energies 2025, 18(21):5729）。

对 ARAC-OC 的映射：
- 每个 owner 保持共享变量副本 x_j^(g)，仲裁不再是 tournament 而是共识值 z_j 的
  带对偶偏移的更新；对偶量 u_j 是**免费、有符号、可积分的冲突压力信号**——
  直接回应"C_j 无稳定阈值间隙"（Gate 47）：不找静态阈值，改用动态积分器。
- conforming 时各副本自然收敛到同一值，罚项自然趋零 → **结构性满足零税条款**，
  不需要额外的静默开关；conflicting 时不一致持续，u_j 持续增长，信号从动力学中涌现。

## 2. 双层/主从视角：共享变量 = 上层复杂变量

- A Review of Bilevel Optimization (arXiv:2511.03448, 2025) §6.1 BOBD：
  bilevel optimization-based decomposition——把"造成复杂的变量"归入上层，
  其余变量下层条件求解；Ψ-mapping/φ-mapping（反应集映射/最优值映射）
  用元模型近似下层响应，避免每个上层候选都真跑下层。
- BLMOCC / CODBA 等进化双层工作：下层响应可用代理模型或继承加速
  （CCBMO, jsegc.2023.18：父子个体下层解继承）。

对 ARAC-OC 的映射：
- 新增一类 episode："shared-level 搜索"——只在共享变量子空间扰动，
  用**截断的、有界 FE 的** owner 私有变量条件重优化（或代理响应面）评估候选。
  这不同于 AOR（全局校正所有变量），也不同于仲裁（在已有 proposal 中选）：
  它**直接搜索共享变量空间且带响应感知**。
- Ψ-mapping 代理回应"即时收益 ≠ 未来收益"（Spearman -0.85 的失败）：
  不再用瞬时耦合信号预测，而用下层响应模型做短期 rollout 预测 continuation gain。

## 3. 博弈论：Nash 谈判解替代归属

- A distributed optimization algorithm for Nash bargaining in multi-agent systems
  (IFAC-PapersOnLine, 2020)：多智能体网络上分布式计算 Nash 谈判解，
  加权和方法 + 在线更新谈判权重。
- 不对称 Nash 谈判（bargaining power ≠ 均等）在能源共享中广泛使用。

对 ARAC-OC 的映射：
- conflicting 共享变量的取值 = 谈判解：max Π_g (f_g(d_g) − f_g(x_j))，
  d_g 为 disagreement point（不合作时各 owner 的值）；
- 谈判权重 = 贡献 → 与 CBCCO 的贡献口径衔接，但输出的是**值**而不是**owner**，
  且 Nash 谈判解公理化保证 Pareto 最优——比"归贡献最大者"多一层原则性；
- 谈判解仍过 strict-best archive，写回契约不变。

## 4. 代理模型辅助的子问题响应（FE 预算友好）

- ASMCC (arXiv:1803.00906)：CC 中按子问题特性自适应建代理（PR/RBF）。
- SA-LSEO-LE (d-nb.info/1387101846/34, 2025)：每个子问题训 RBF，
  用 K 个子模型均值做 infill 准则。
- BSCo-GPLM (Swarm Evol Comput 2024)：GP 驱动的子问题线性模型协作训练。
- Efficient Large-Scale Expensive Optimization via Surrogate (Hayat et al. 2024)：
  SACC 谱系，子问题分解降低代理维度需求。

对 ARAC-OC 的映射：per-owner 响应面 f̂_g(x_shared) 让仲裁/复核/谈判
在代理空间进行，真实 FE 只花在最终写回验证——回应"sense 吃掉 93.7% 预算"
（Gate 48a）的结构性解法。

## 5. Phase-I / 基准侧的新资产（非阶段二，但相关）

- OEDG: An Enhanced Differential Grouping Method for Large-Scale Overlapping Problems
  (Tian et al., IEEE TEVC 2024)：两阶段分组 + SUD/SD 精化；
  **自带控制拓扑/重叠度/可分性的新基准生成器**——与自有函数发生器互补/对照。
- Xu et al. 2023 (Swarm Evol Comput 78:101280)：可定制耦合异构模块基准。
- BICCA (Ge et al.)：双空间交互 CC——pattern space 中分组结构随优化共同进化，
  分解不花额外 FE。与"归属/结构可迁移"的思路同族。
- A Learning-Based Cooperative Coevolution Framework for Heterogeneous LSGO
  (arXiv:2604.01241, 2026)：学习型 CC 最新工作，含 HCC/OEDG 引证网络。
- DCCVI (Chen et al., Appl Soft Comput 2025)：按变量重要性动态重组 +
  收敛变量重置——动态归属思想的近邻。

## 6. 新颖性核查记录

检索"variable duplication consensus penalty cooperative coevolution"未命中任何
进化 CC 文献把增广拉格朗日/共识约束用于共享变量——命中的是统计学习
（cooperative learning）与多智能体能源系统。初步判断：§1 方向在 EC 文献内
尚未被占据，但投稿前需再做一轮正式查新（Web of Science / Scopus）。

## 7. 优先级建议（详见对话回复）

1. 软共识 + 对偶积分信号（§1）——与现有证据链咬合最紧；
2. shared-level 双层搜索 + Ψ-mapping（§2）——回应预测失败的根因；
3. Nash 谈判取值（§3）——conflicting 支路的候选升级；
4. 代理响应面（§4）——作为以上三者的 FE 放大器而非独立机制。
