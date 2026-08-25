# 软路由（连续活跃度缩放）机制：文献调研与升级映射

日期：2026-08-22
问题：如何在不分类（54a 已证不可辨识）的前提下，让机制在 conforming 时自动退化为
v1 四动作行为、在 conflicting 时自动激活——即"连续软路由"。
来源：本轮网络学术检索实际返回结果。

## 1. 自适应罚参数 / 残差平衡（强度缩放的最成熟先例）

- He et al. 2000（RB 原始）；Wohlberg, "ADMM Penalty Parameter Selection by
  Residual Balancing"（arXiv:1704.06209）：罚参数 τ 在线调节，使原始残差
  （不一致性）与对偶残差（进展速度）保持同量级；τ 有上界 τ_max，
  **有界自适应保持收敛性**。
- Boyd et al. 共识 ADMM 讲义：自适应罚在有限次迭代后冻结以保证收敛。
- Xu et al., "Adaptive Consensus ADMM"（ICML 2017, arXiv:1706.02869）：
  **逐节点（per-node）自适应罚**，给出 O(1/k) 收敛率；有界自适应假设
  Ση²<∞ —— 即自适应总量必须可求和/有界。
- Spectral penalty（arXiv:1605.07246）：BB 步长式罚更新；**明确指出 RB 的
  缺陷是尺度依赖**——残差比随问题缩放而变，必须先归一化。

对 ARAC-OC 的映射：
- 机制强度（patch 半径/激活度）按"分歧残差 vs 进展残差"的比值连续调节——
  不需要分类器，强度与问题性质自动共变；
- **尺度归一化是硬前提**（RB 的已知失败模式）→ 用项目已有的秩归一化口径；
- **有界自适应条款**：强度上调次数/总量有界，或末段预算冻结强度——
  与 fail-closed/审计文化兼容，且对应 ACADMM 的收敛假设形式。

## 2. 软分组 / 模糊隶属（EC 侧的软路由先例）

- Liu, Zhou, Li et al., "Cooperative Co-evolution with Soft Grouping for
  Large Scale Global Optimization", IEEE CEC 2019（被引 30）：
  变量对分组的隶属是**概率/连续的**，不做硬分配——"软路由"在分解层的
  直接先例。
- Fan, Wang, Han, "Cooperative Coevolution Based on Kernel Fuzzy Clustering
  and Variable Trust Region Methods", IEEE TFS 22(4), 2014（FT-DNPSO）：
  **核模糊 C 均值隶属度**做分组 + **逐变量信任域**自适应调节搜索范围。
  这是"软隶属 + 逐变量 trust-region"两要素在 EC 中的最近先例，必须引用。

差异化定位（诚实边界）：
- Soft grouping 的隶属用于**分组本身**（全重叠概率隶属），不针对重叠问题
  共享变量的 owner 权重；
- FT-DNPSO 的信任域调节的是 PSO 粒子的**搜索范围**，不是跨 episode 持久的
  共享变量仲裁状态；
- 因此"软 owner 权重 + 残差平衡驱动强度 + 持久 (z,u,r) 状态"的组合在
  重叠 CC 文献内仍未被占据，但上述两篇是 related work 的必引项。

## 3. 其他相关

- DECC-CLV（王玉峰等，武汉大学）：变量相关学习 + EM/PPCA 分组，
  针对 CEC2013 重叠基准——分解层，中文期刊。
- DCCVI（Chen et al. 2025）：变量重要性动态分组 + 收敛变量重置——
  "重要性连续量驱动分组调整"的同族思想。
- iCC（Vakhnin & Sopov 2021）：子组件变量数自适应增减。
- 重叠 CC 最新：OCC（GECCO'24）与 CBCCO 仍是最新主线，2025–2026 未见
  新的共享变量处理机制（本轮检索确认）。

## 4. 升级映射结论（详见对话回复）

软路由 = 三个连续量替代三个离散决策：

| 原离散决策 | 连续替代 | 文献先例 |
|---|---|---|
| conforming/conflicting 分类 | 强度标量（分歧残差/进展残差比值驱动） | RB/ACADMM |
| 归属唯一 owner | 软 owner 权重 w_jg（单纯形上连续更新） | Soft grouping / 模糊隶属 |
| 固定扫描范围 | 逐变量信任域半径 r_j（持久状态） | FT-DNPSO / DFO trust-region |

三条公共纪律：全部量秩归一化（防 RB 尺度陷阱）；自适应总量有界
（防抖动、保审计）；一切写回过 strict-best（失败语义内生）。
