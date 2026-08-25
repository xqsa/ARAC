# ARAC-OC 阶梯升级文献检索记录

日期：2026-08-23  
执行者：Codex  
工作流：`multi-source-search` + `citation-verification`  
来源优先级：CrossRef / arXiv（T1），Scopus 作为补充尝试

## 检索范围

检索词覆盖：

- `overlapping cooperative co-evolution shared variables`
- `contribution based cooperative coevolution overlapping subcomponents`
- `cooperative coevolution variable interaction grouping large scale optimization`
- `derivative-free trust-region black-box optimization`
- `soft grouping overlapping cooperative coevolution`
- `adaptive operator selection metaheuristics`
- `efficient resource allocation cooperative co-evolution`

Scopus 查询因本机缺少 `pybliometrics` 配置而未返回结果；不影响 CrossRef/arXiv
主检索。宽泛 CrossRef 查询包含大量无关记录，最终只保留能通过 DOI/arXiv ID
回查的条目。

## 已核验条目与可借鉴点

| 条目 | 标识 | 核验结果 | 对 ARAC-OC 的合法借鉴 |
|---|---|---|---|
| Jia, Mei, Zhang, *Contribution-Based Cooperative Co-Evolution for Nonseparable Large-Scale Problems With Overlapping Subcomponents* | DOI `10.1109/TCYB.2020.3025577` | CrossRef metadata verified；IEEE TCYB 52(6), 4246-4259 | 贡献归属可作为候选/资源分配信号；不能直接证明 ARAC 的 shared-patch 增益 |
| Komarnicki et al., *Overlapping Cooperative Co-Evolution for Overlapping Large-Scale Global Optimization Problems* | DOI `10.1145/3638529.3654171` | CrossRef metadata verified；GECCO 2024, 665-673 | 多重 shared-variable assignment 与共享关系调制资源；支持 S1/S5a 动机 |
| Sun et al., *Decomposition for Large-scale Optimization Problems with Overlapping Components* | DOI `10.1109/CEC.2019.8790204` | CrossRef metadata verified；CEC 2019, 326-333 | 重叠分解与结构拓扑的先例；不支持在线 classifier 的因果结论 |
| Blanchard, Beauthier, Carletti, *Investigating Overlapped Strategies...* | DOI `10.1007/978-3-030-85672-4_19` | CrossRef metadata verified；CCIS 1443, 254-266 | 支持把 conforming 与 conflicting 的预期收益分开；不把阴性结果改写为终值 superiority |
| Pryor, Peerlinck, Sheppard, *A Study in Overlapping Factor Decomposition for Cooperative Co-Evolution* | DOI `10.1109/ssci50451.2021.9659875` | CrossRef metadata verified；SSCI 2021 | 完整目标评价决定 shared write-back 的直接先例；支持 strict-best gate |
| Fan, Wang, Han, *Cooperative Coevolution for Large-Scale Optimization Based on Kernel Fuzzy Clustering and Variable Trust Region Methods* | DOI `10.1109/tfuzz.2013.2276863` | CrossRef metadata verified；IEEE TFS | 支持“软隶属 + variable trust-region”作为 related-work 先例；不是跨 episode shared state 的证明 |
| Liu et al., *Cooperative Co-evolution with Soft Grouping for Large Scale Global Optimization* | DOI `10.1109/cec.2019.8790053` | CrossRef metadata verified；CEC 2019, 318-325 | 支持连续/软分组；不等同于 owner weight steering |
| Jia et al., *A Distributed Cooperative Co-evolutionary CMA Evolution Strategy...* | DOI `10.1109/access.2019.2897282` | CrossRef metadata verified；IEEE Access 7, 19821-19834 | 支持 overlapping CC 的分布式资源与 CMA-style host 讨论 |
| Chen, Du, *A Novel Cooperative Co-Evolutionary Framework for Large-Scale Overlapping Problems* | DOI `10.1109/cec53210.2023.10254019` | CrossRef metadata verified；CEC 2023 | 子组 merging 与 resource allocation 的相关先例；不直接支持 ARAC patch |
| Tian et al., *An Enhanced Differential Grouping Method for Large-Scale Overlapping Problems* | arXiv `2404.10515` | arXiv metadata/abstract verified | 支持 topology/overlap/separability 分层 generator 与结构证据成本审计 |
| Yang et al., *Efficient Resource Allocation in Cooperative Co-Evolution...* | DOI `10.1109/tevc.2016.2627581` | CrossRef metadata verified；IEEE TEVC 21(4), 493-507 | 支持按动态 contribution 分配资源；S6 只作为后续候选 |
| Yang et al., *CCFR3...* | DOI `10.1016/j.eswa.2022.117397` | CrossRef metadata verified；ESWA 203, 117397 | 支持更频繁的 contribution evaluation；不能直接替代当前 CTP lifecycle |
| Xu et al., *Difficulty and Contribution-Based Cooperative Coevolution...* | DOI `10.1109/tevc.2022.3201691` | CrossRef metadata verified；IEEE TEVC 27(5), 1355-1369 | 支持 difficulty × contribution 双轴；列为 S6，不塞进 S1-S5 |
| Mahdavi et al., *Cooperative co-evolution with sensitivity analysis-based budget assignment...* | DOI `10.1007/s10489-017-0926-z` | CrossRef metadata verified；Applied Intelligence 47 | 支持按实测 sensitivity 分配 probe 预算；不得在冻结 baseline 上直接改 sense 税 |
| Liuzzi et al., *Trust-Region Methods for the Derivative-Free Optimization of Nonsmooth Black-Box Functions* | DOI `10.1137/19m125772x` | CrossRef metadata + abstract verified；SIAM J. Optim. 29(4), 3012-3035 | 支持 bounded trust-region 的黑盒安全性动机；ARAC 不声称继承其收敛定理 |
| De Falco et al., *Investigating surrogate-assisted cooperative coevolution...* | DOI `10.1016/j.ins.2019.01.009` | CrossRef metadata verified；Information Sciences 482, 1-26 | 后续 surrogate candidate lane；不进入首个阶梯 |
| Pei et al., *Adaptive Operator Selection for Meta-Heuristics: A Survey* | DOI `10.1109/tai.2025.3545792` | CrossRef metadata verified；IEEE TAI, 2025 | 支持把 credit、state、exploration 分开；不把 AOS 综述当成 shared-variable 证据 |
| De Rainville et al., *Sustainable Cooperative Coevolution with a Multi-Armed Bandit* | arXiv `1304.3138` | arXiv metadata/abstract verified | 资源分配的 bandit 先例；列为长期路线，不进入首版 |
| Xu et al., *Adaptive Consensus ADMM for Distributed Optimization* | arXiv `1706.02869` | arXiv metadata/abstract verified | 仅支持“有界自适应参数 + residual normalization”的设计警示；不宣称 ARAC 有 ADMM 收敛 |
| Wohlberg, *ADMM Penalty Parameter Selection by Residual Balancing* | arXiv `1704.06209` | arXiv metadata/abstract verified | 明确提醒 residual balancing 的尺度陷阱；支持 rank/scale normalization 的硬约束 |

## 重要排除

- `10.5220/0006903102610278` 的 CrossRef DOI 回查返回 404，不能作为已核验
  引用；如需使用该会议条目，应以本地 PDF 的正式出版信息人工复核。
- 现有方案中把 OCC 直接表述为“相邻组件交替产生新最优”的说法过强；本次只
  采用其“shared-variable 多重归属与资源相互影响”的可核验表述。
- ADMM 文献不是 ARAC 的收敛证明。ARAC 的 patch 仍是固定 FE、strict-best、
  fail-closed 的经验机制。
- LCC/LH-CC/RLDO 可作为未来学习路线的协议参考，但不进入首个升级阶梯，避免
  学习器、动作选择和 shared-patch 同时变化。

## 检索结论

文献支持“分解/归属/资源分配/信任域”分别有成熟先例，但没有一篇被核验的工作
同时证明：在冻结 Phase-I 与外层动作的情况下，使用固定 FE 的 shared-variable
patch、跨 episode 局部状态和严格嵌套消融能稳定改善 ARAC 的 conflicting host。
因此 ARAC-OC 的可申报创新应写成“受文献启发的工程组合 + 独立因果验证”，而不是
声称某篇文献已经提出该完整机制。
