# 重叠变量协调文献与 ARAC 下一步建议

调研日期：2026-08-14  
执行者：Codex  
检索流程：`nature-academic-search` multi-source-search；Crossref/arXiv 元数据检索，并与项目已有 `references/overlap_AOB_literature_survey_2026-07.md` 交叉核对。

## 结论先行

ARAC 当前 Gate 16/17 暴露的问题，在文献中并不是靠“把多个 proposal 做一次平均”解决的。更接近的成熟路线有三点：

1. 保留重叠变量的多重归属，但对共享变量做显式的重叠 patch / context-vector 协调，而不是把分歧降级成 sweep 顺序。
2. 用贡献、交互度或近期改进动态分配子问题预算，而不是给所有子组件固定同样的 repair 预算。
3. 采用有反馈的顺序式局部更新（trust-region / sufficient-decrease / accept-reject），每次真实评价完整解，再决定继续、扩大或缩小 patch；不是一次性批量随机采样。

这与当前证据一致：Gate 16 中 post-arbitration CTP 与 current CTP 的 win-or-tie 为 1.0、median gain 为 0，说明 search-base 时序不是主因；Gate 17 的四中心一次性 mixture 相对 shared-core owner 只有 0.20 win-or-tie，说明“多中心批量采样”不是有效修复。

## 最相关文献

| 文献 | 直接贡献 | 对 ARAC 的可借鉴点 | 不应直接宣称 |
|---|---|---|---|
| Sun, Li, Ernst, Omidvar, 2019, *Decomposition for Large-scale Optimization Problems with Overlapping Components*, CEC. [DOI](https://doi.org/10.1109/cec.2019.8790204) | 直接研究 overlapping components 的大规模分解 | 让 overlap 成为正式子问题结构；组件间资源分配应考虑共享变量和相互影响 | 该文不是 ARAC 的 Phase-I evidence-to-action 框架，也不能替代黑盒因果验证 |
| Komarnicki, Przewozniczek, Tinós, Li, 2024, *Overlapping Cooperative Co-Evolution for Overlapping Large-Scale Global Optimization Problems*, GECCO. [DOI](https://doi.org/10.1145/3638529.3654171) | 有意构造重叠子组件，并按共享变量影响分配计算资源 | 将共享变量影响用于预算和组件协作，而不只是排序；可作为 ARAC 的最直接外部基线 | 不能把其 overlap CC 直接等同于 ARAC 的共享变量协调器 |
| Michelena, Papalambros, Park, Kulkarni, 1999, *Hierarchical Overlapping Coordination for Large-Scale Optimization by Decomposition*, AIAA Journal. [DOI](https://doi.org/10.2514/2.7538) | 层次化 overlapping coordination | 采用“局部子问题 -> overlap coordination -> 更高层修正”的层次结构；支持把共享 patch 作为独立层 | 该工作主要是经典分解协调，不是当前黑盒 LSGO benchmark 的直接性能证据 |
| Baraldi, Manns, 2024, *Domain decomposition for integer optimal control with total variation regularization*, arXiv:2410.15672. [arXiv](https://arxiv.org/abs/2410.15672) | 重叠 subdomain 上求 trust-region patch，并用 greedy patch selection 和 sufficient decrease 控制更新 | 将 CTP 改为 sequential trust-region patch：真实完整目标评价后 accept/reject，并按收益调半径 | 这是整数最优控制领域；只能借鉴机制，不能借用其收敛定理到我们的非凸黑盒问题 |
| Omidvar, Li, Yao, 2010, *Cooperative Co-evolution with delta grouping for large scale non-separable function optimization*, CEC. [DOI](https://doi.org/10.1109/cec.2010.5585979) | 面向非可分函数的变量交互/分组 | Phase-I 的交互证据应最终影响重叠 patch 和联合搜索范围，而不是只影响调度 | 这是分组/交互识别路线，不是共享变量冲突修复算法 |
| Chen, Weise, Yang, Tang, 2010, *Large-Scale Global Optimization Using Cooperative Coevolution with Variable Interaction Learning*, PPSN. [DOI](https://doi.org/10.1007/978-3-642-15871-1_31) | 用 variable interaction learning 改善 CC 的非可分处理 | 记录变量/组件的在线贡献和交互，把它用于后续搜索策略与预算 | 不能把 interaction learning 自动视为 proposal conflict resolver |
| Duan, Shao, Zhou, Yang, Zhao, Shi, 2024, *Cooperative coevolution for non-separable large-scale black-box optimization: Convergence analyses and distributed accelerations*, Applied Soft Computing. [DOI](https://doi.org/10.1016/j.asoc.2024.112232) | 非可分 LSGO 的 CC 收敛与分布式加速 | 强调 CC 需要一致的全局 context 和可分析的子问题更新过程；支持做 stateful sequential update | 该文不提供我们所需的 overlap-specific conflict coordinator |
| Omidvar, Li, Yao, 2010, *Cooperative Co-evolution for large scale optimization through more frequent random grouping*, CEC. [DOI](https://doi.org/10.1109/cec.2010.5586127) | 通过更频繁重组降低固定分组偏差 | 可作为未来“重叠 patch 周期性重组”的备选对照 | 不是共享变量副本/共识机制 |
| Chen, Ouyang, Liu, Zhang, Gan, 2025, *Dynamic cooperative coevolution based on variable importance for non-separable large-scale global optimization*, Applied Soft Computing. [DOI](https://doi.org/10.1016/j.asoc.2025.113363) | 根据变量重要性动态调整 CC | 支持将 Phase-I evidence 与在线 contribution 结合，做 patch/预算动态化 | 仍主要是调度和重要性，不等于重叠变量联合修复 |
| Qiu et al., 2025, *Advancing CMA-ES with Learning-Based Cooperative Coevolution for Scalable Optimization*, arXiv:2504.17578. [arXiv](https://arxiv.org/abs/2504.17578) | 用 PPO 学习分解策略选择 | 说明 selector 可以作为后续工作，但前提是底层动作本身先证明有效 | 它选择分解策略，不解决当前 shared-core repair kernel 的失败 |

## 与 ARAC 当前失败的精确对应

### 当前实现缺少的机制

- `OverlapCoordinator._repair_shared_core` 一次性生成 32 个围绕 weighted mean 的 batch；没有逐步使用真实评价反馈更新中心。
- 多个 proposal 的分歧被压缩成一个 `weighted_mean` 和一个 spread；没有保留“哪个 owner 在当前 incumbent 上实际有效”的方向信息。
- CTP 没有 trust-region 半径、sufficient-decrease 判据、失败后的缩小或成功后的扩张。
- GCB 目前主要按 residual/topology/贡献排序；OCC 类工作提示共享变量应直接影响预算和协作，而不是只影响排序。

### Gate 结果说明了什么

- Gate 16：改变 search-base 时序没有实质收益，因此先修时序不是最高价值动作。
- Gate 17：增加 incumbent/mean/median/owner 四个中心的一次性随机样本反而更差，因此不能继续沿“多采样中心”盲目扩展。
- 下一步应测试“有反馈的共享 patch 搜索”，而不是再测试新的静态候选混合。

## 建议的下一道最小验证门

暂定名称：**Gate 18 — Sequential Shared-Patch Trust-Region Diagnostic**。

固定组件选择和 Gate 15/16 的 paired protocol，只替换 32-FE repair kernel：

1. 从候选仲裁后的 incumbent 开始，取 shared variables 为 patch；weighted mean 只作为初始点，不作为唯一搜索中心。
2. 维护 patch center `y_k` 和半径 `r_k`。每轮生成少量完整候选，只改变 patch 坐标，真实评价完整解。
3. 若 `strict-best` 且达到预设 sufficient-decrease，则接受候选、更新 `y_k`，并有限扩大 `r_k`；否则拒绝并缩小 `r_k`。
4. 共享变量 proposal 方向按 owner disagreement 和近期真实改进加权；不把所有 owner 简单平均。
5. 用同一 32 FE 与两个控制比较：
   - shared-core owner continuation；
   - current one-shot CTP。
6. 预注册输出：win-or-tie、median gain、成功/失败半径轨迹、每个 patch 的有效 FE，以及是否真正改善共享变量而非只改善独立变量。

该门的科学问题是：

> 在相同共享变量 patch 和相同 FE 下，顺序式、反馈驱动的协调是否比一次性随机 repair 更有效？

只有当它先通过 shared-core owner control 的配对门，才考虑把该机制接入 production coordinator；selector 和 AOB-24 继续冻结。

## 来源与限制

- Crossref/arXiv 返回的部分会议论文只有书目信息，没有可用摘要；方法细节只采用项目已有文献核查文件或论文公开摘要中明确的内容。
- 本备忘录不把经典分解、trust-region、CC 调度和 overlap CC 混为同一算法；每篇文献的可借鉴范围都单独标注。
- 文献启发不等于 ARAC 已验证。所有“能否改善”仍需在当前 Gate 12 interaction benchmark 和等 FE paired protocol 上单独验证。

## 本轮复核：论文到底如何处理共享变量

本轮通过 Crossref、Semantic Scholar、arXiv 和 OpenAlex 复核了摘要与公开元数据。关键事实如下。

### OCC 不是简单的重叠分组

Komarnicki et al. (2024) 的公开摘要明确指出：传统 CC 中共享变量通常仍被固定分配给单一组件；OCC 允许共享变量有多个 assignment，并让一个组件的计算资源受与其共享变量的其他组件影响。也就是说，OCC 的主要改变是“多重归属 + 组件间资源耦合”，而不是把 proposal 做一次平均。它仍主要解决组件如何共同获得搜索机会，不等于 ARAC 的在线共享变量值协调。

### Sun et al. 的路线是“在共享变量处断开链接”

Sun et al. (2019) 的公开摘要明确说明：把所有 linked variables 放入同一组会使子问题仍然过大；他们修改 RDG，在多个组件共享的变量处打断 linkage，从而得到更小的分解组件。这个路线解决的是“如何保留可分解规模”，不是运行时副本一致性；但它给 ARAC 一个重要边界：共享变量不能只作为图上的标签，必须进入子问题定义和后续更新规则。

### Baraldi & Manns 给出了最接近当前修复问题的更新节奏

Baraldi & Manns (2024, arXiv:2410.15672) 在整数最优控制中使用有重叠的 subdomain patch：每次只修改一个局部 patch，patch 内部求 trust-region 子问题，用 sufficient-decrease 判断是否接受，并用 greedy patch selection 选择下一块。这个方法不是黑盒 LSGO，但“重叠 patch + 局部半径 + 真实反馈 + accept/reject + 按收益继续或换 patch”的控制逻辑与 ARAC 的 repair kernel 问题高度同构。

### Schwarz/partition-of-unity 的邻域启发是“重叠区要有显式权重”

Schwarz/ORAS 文献（例如 May et al., 2019, arXiv:1909.08734；Bonazzoli et al., 2022, arXiv:2212.03132）把重叠区域作为接口，并用 restricted update、transmission condition 或 partition of unity 防止多个局部解无约束地叠加。它们求解的是 PDE 离散系统，不是黑盒目标；因此不能移植其收敛结论，但可以借鉴“共享坐标有显式写回规则，而不是隐式平均”的接口思想。

### LCC/LH-CC 解决的是策略选择，不是共享变量修复

Guo et al. (2025, arXiv:2504.17578) 用 PPO 根据状态选择 RD/MiVD/MaVD 分解策略；这与 ARAC 的 selector 方向相关，但它选择的是 decomposition strategy，底层 CC 仍没有提供 shared-variable conflict repair。因而 LCC 不能作为 Gate 17 失败的直接修复方案，也不应在 repair kernel 尚未成立前引入 RF/RL selector。

## 由文献收敛出的 ARAC 机制候选

文献共同支持的最小候选不是“四中心混合”，而是以下闭环：

1. Phase-I 输出带置信度的 overlap patch 和 owner 集合；
2. 在同一 incumbent 上，让一个共享 patch 产生少量局部候选；
3. 候选只改 patch 坐标，并对完整黑盒目标进行真实评价；
4. 以 strict-best / sufficient-decrease 作为写回条件；
5. 成功后沿当前 patch 方向继续或有限扩大半径，失败后缩小半径或换下一个 patch；
6. 用实际改进/失败次数更新 patch 优先级和预算，而不是只按静态 residual 排序。

这可以称为 **evidence-guided sequential shared-patch search**。它借鉴 OCC 的多重归属、Sun 的共享链接处理、Baraldi 的 trust-region patch 反馈和 Schwarz 的显式重叠接口，但组合成黑盒 LSGO 版本；不能宣称继承任何凸性、PDE 或一阶收敛定理。

## 本轮新增核对来源

- Komarnicki, Przewozniczek, Tinós, Li (2024), *Overlapping Cooperative Co-Evolution for Overlapping Large-Scale Global Optimization Problems*, GECCO, DOI: [10.1145/3638529.3654171](https://doi.org/10.1145/3638529.3654171). Semantic Scholar abstract confirms multiple assignments and resource coupling.
- Sun, Li, Ernst, Omidvar (2019), *Decomposition for Large-scale Optimization Problems with Overlapping Components*, CEC, DOI: [10.1109/cec.2019.8790204](https://doi.org/10.1109/cec.2019.8790204). Semantic Scholar abstract confirms breaking linkage at shared variables.
- Baraldi & Manns (2024), *Domain decomposition for integer optimal control with total variation regularization*, arXiv: [2410.15672](https://arxiv.org/abs/2410.15672). Abstract confirms overlapping subdomains, trust-region local patches, sufficient decrease and greedy patch selection.
- May, Haynes & Ruuth (2019), *Schwarz solvers and preconditioners for the closest point method*, arXiv: [1909.08734](https://arxiv.org/abs/1909.08734).
- Bonazzoli, Claeys, Nataf & Tournier (2022), *How does the partition of unity influence SORAS preconditioner?*, arXiv: [2212.03132](https://arxiv.org/abs/2212.03132).
- Guo et al. (2025), *Advancing CMA-ES with Learning-Based Cooperative Coevolution for Scalable Optimization*, arXiv: [2504.17578](https://arxiv.org/abs/2504.17578).

## Gate 23 后的分解边界复核

Gate 23 在固定 24-D interaction context 上得到一个关键的可辨识性分离：pairwise
mixed-difference graph 对共享变量的 precision/recall 都为 `1.0`，但 maximal cliques
没有复原 benchmark 的六个隐藏生成器 group，而是形成 18 个较小 evidence cliques。
这与以下分解文献的边界一致：

- Chen, Weise, Yang & Tang (2010), *Large-Scale Global Optimization Using Cooperative
  Coevolution with Variable Interaction Learning*, DOI
  [10.1007/978-3-642-15871-1_31](https://doi.org/10.1007/978-3-642-15871-1_31)，
  以变量交互学习支持 CC 分组；其目标是得到可优化的 linkage，而不是识别 benchmark
  生成器身份。
- Chen & Tang (2013), *Impact of problem decomposition on Cooperative Coevolution*, DOI
  [10.1109/cec.2013.6557641](https://doi.org/10.1109/cec.2013.6557641)，强调分解质量会
  直接影响 CC，但不要求分解与某个隐藏真值标签逐项相同。
- Duan et al. (2019), *Hierarchical Decomposition based Cooperative Coevolution for
  Large-Scale Black-Box Optimization*, DOI
  [10.1109/ssci44817.2019.9003169](https://doi.org/10.1109/ssci44817.2019.9003169)，
  支持在 pairwise/局部证据不够稳定时使用层次化子问题，而不是一次性声称精确超图恢复。

因此 ARAC 的 Phase-I 输出应正式定义为 **evidence interaction hypergraph/clique cover**：
它需要覆盖可复现的变量交互与共享变量多重归属，但不宣称唯一恢复 objective 生成时的
隐藏 group。Phase-II 必须直接在这个证据 cover 上验证 proposal 和 context write-back；
benchmark truth groups 只用于离线审计，不能注入算法。

## 本轮：soft-RDDSM 四个结构性失败的文献对照

本轮检索日期：2026-08-16；执行者：Codex。检索源：Crossref、Semantic Scholar
公开摘要、OpenAlex、arXiv。检索对象不是泛化的“重叠优化”，而是：多重归属、共享变量
断链、关系边聚类、黑盒差分分组和重叠副本协调。

### 1. 连通分量吞掉重叠：不要再对变量节点做普通 component closure

Sun et al. 的 RDG3 明确指出：普通 RDG 会把直接或间接相连的变量全部放入同一组，
导致重叠问题仍然退化成一个大组件。RDG3 的做法是在共享变量处**主动打断 linkage**，
用大小阈值 `epsilon_n` 在递归扩张尚未吞掉整个连通块时提交组件。它解决的是可优化
分解，不是共享变量身份恢复；RDG3 会把共享变量放入某一侧，因此不能直接作为 ARAC
的最终 evidence 输出，但可作为“断链式递归探测”的机制来源。

- Sun, Li, Ernst & Omidvar (2019), *Decomposition for Large-scale Optimization Problems
  with Overlapping Components*, CEC, DOI: [10.1109/cec.2019.8790204](https://doi.org/10.1109/cec.2019.8790204)。
- 公开版本：<http://eprints.whiterose.ac.uk/156232/1/RDG3_v3.pdf>。

更直接的图表示借鉴来自 Ahn et al. 的 link communities：他们不聚类节点，而是聚类
关系边；节点属于哪些社区由其 incident edges 的社区标签导出。这样即使一个节点同时
连接两个密集团，也不会因为节点连通性把两个社区强制合并。这个思想正好对应 ARAC：
应先形成 evidence-edge/clique cover，再从边簇推导变量的多重 membership，而不是从
变量连通分量反推重叠。

- Ahn, Bagrow & Lehmann (2010), *Link communities reveal multiscale complexity in networks*,
  Nature 466, 761–764, DOI: [10.1038/nature09182](https://doi.org/10.1038/nature09182)。

### 2. Tarjan 单点割点失效：改成边簇/最小分离集，不再寻找单个 articulation vertex

R6 每条边界有 10 个共享变量时，删除一个变量不会断开图，这是算法假设与数据结构
不匹配，不是阈值问题。link-community 的边聚类绕开了“删掉一个点是否断开”的假设。
在优化侧，RDG3 也不是找割点，而是对集合与集合做递归 `INTERACT` 测试，在达到
规模阈值时切断 linkage。对 ARAC，优先级应是：

```text
region pair interaction
  -> recursive set-vs-set split
  -> edge/cut evidence
  -> edge cluster membership
  -> shared variable = variable incident to >=2 edge clusters
```

### 3. 签名双簇性不成立：不要把 cosine/kNN 当作共享变量判别器

Li et al. 的 DGSC 确实把差分得到的设计结构矩阵作为相似矩阵再做谱聚类，但其输入
已经是差分交互矩阵；它不能证明 noisy signature cosine 在 ARAC 的 12 个随机 batch 上
能够区分“同组”和“跨组共享”。因此 DGSC 可以借鉴“用交互矩阵而不是原始索引聚类”，
不能继续作为当前 kNN 候选召回的理论依据。

- Li, Fang, Wang & Sun (2019), *Differential Grouping with Spectral Clustering for Large
  Scale Global Optimization*, CEC, DOI: [10.1109/cec.2019.8790056](https://doi.org/10.1109/cec.2019.8790056)。

Liu et al. 的 Soft Grouping 更贴近当前现象：不把变量硬分到一个组，而是控制变量对
多个组的 membership degree/probability。它的启发不是“把 cosine 相似度再调一个阈值”，
而是把 membership 当作连续证据，允许一个变量同时保留多个候选归属，再由后续真实
目标评价决定哪些归属具有优化价值。

- Liu, Zhou, Li & Tang (2019), *Cooperative Co-evolution with Soft Grouping for Large
  Scale Global Optimization*, CEC, DOI: [10.1109/cec.2019.8790053](https://doi.org/10.1109/cec.2019.8790053)。

### 4. 候选图没有跨组边：从“近邻检索”改为“集合交互测试 + refinement”

你现在已经实验证明条件探针有 5 个数量级的判别力，但 kNN 图在探针前就把跨组边
删掉了。文献中的成熟做法不是继续增大 k，而是先对集合做差分交互测试，再递归细化：

- RDG/RDG3 用 `INTERACT(X1, X2)` 递归二分，不要求目标变量先出现在 kNN 中；
- OEDG（Tian et al., 2024）采用“两阶段增强分组”：第一阶段用有限差分识别子组件和
  共享变量，第二阶段用 Subcomponent Union Detection (SUD) 与 Subcomponent Detection
  (SD) 做 union/split 修正；
- 这与当前需求完全对应：条件探针应是 refinement oracle，不能只作用于已经被 cosine
  kNN 选中的边。

- Tian, Chen, Du, Tang & Jin (2025), *An Enhanced Differential Grouping Method for Large-Scale
  Overlapping Problems*, IEEE Transactions on Evolutionary Computation 29, 2272–2286,
  DOI: [10.1109/TEVC.2024.3390719](https://doi.org/10.1109/TEVC.2024.3390719)。

### 5. 多重归属之后如何优化：借鉴副本协调，但不要把资源贡献当作充分条件

Jia et al. 的 DCCMAES 明确允许共享变量被分到多个子组件，并用 random orthogonal
experiment 生成全局解，同时提供三种 cooperation schemes 来协调多个优化器。
Komarnicki et al. 的 OCC 也允许 shared variables multiple assignments，并让组件资源
分配受到共享组件影响。Chen & Du 的 CCMRO 则用 contribution-based subgroup merging
降低子组交互，并配合资源分配。

- Jia et al. (2019), *A Distributed Cooperative Co-evolutionary CMA Evolution Strategy for
  Global Optimization of Large-Scale Overlapping Problems*, IEEE Access 7, 19821–19834,
  DOI: [10.1109/ACCESS.2019.2897282](https://doi.org/10.1109/ACCESS.2019.2897282)。
- Komarnicki, Przewozniczek, Tinós & Li (2024), *Overlapping Cooperative Co-Evolution for
  Overlapping Large-Scale Global Optimization Problems*, GECCO, 665–673,
  DOI: [10.1145/3638529.3654171](https://doi.org/10.1145/3638529.3654171)。
- Chen & Du (2023), *A Novel Cooperative Co-Evolutionary Framework for Large-Scale Overlapping
  Problems*, CEC, DOI: [10.1109/CEC53210.2023.10254019](https://doi.org/10.1109/CEC53210.2023.10254019)。

但 Sun et al. 在 RDG3 的实验中发现，单纯按 component contribution 自适应分配资源在
overlapping、尤其 conflicting 问题上反而可能变差，因为一个组件的贡献依赖其他组件
的当前状态。因此 ARAC 不能把 `Delta_g` 单独当作 GCB 预算依据；它最多参与 proposal
权重，最终仍需冲突残差、完整目标评价和 strict-best。

## 文献收敛出的 v11 方向

soft-RDDSM 不应再修补“连通分量 + kNN + 割点”三件套。建议的最小结构是：

```text
签名：只做 seed/region 排序，不做共享判别
  -> region-pair set-vs-set interaction tests
  -> recursive refinement（RDG3/OEDG 思路）
  -> edge-first / link-community evidence cover
  -> soft membership（一个变量可保留多个 group 候选）
  -> 多重归属副本 + ARAC-OC 冲突协调
```

其中最重要的工程变化是：**删掉 Tarjan 割点和 kNN 作为必要入口**。若 FE 不允许全量
逐变量探针，就用 region-pair 的分层 group test 做候选生成，再在阳性 cut 上逐变量
refinement；共享变量通过“同时属于两个 edge cluster”产生，而不是通过“删除一个点后
图是否断开”产生。

这条路线保留你已经验证的条件探针判别力，并直接修复四个根因；它也比继续增加签名
probe 数更符合文献中的处理方式。
