# ARAC-OC Related Work 引用缺口与定位备忘

日期：2026-08-21
来源：`docs/lit-review-2026-08-21.md` §二-D/§二-G
用途：论文 related work / positioning 写作时的必引清单与叙事锚点。

## 1. 直接先例（必须引用，当前文档未见引用）

### FEA —— 完整目标函数仲裁的先例

Pryor, Peerlinck, Sheppard, "A Study in Overlapping Factor Decomposition for
Cooperative Co-Evolution"（本地 `references/lit-review/`）。

FEA 对共享变量的处理：各重叠 factor 把自己的值注入全局解，**谁的全局适应度
好谁写回**。这正是 ARAC-OC 的完整候选仲裁 + strict-best 写回机制。写作时
应表述为：ARAC-OC 的仲裁机制继承了 FEA 的"共享变量由完整目标裁决"原则，
并在此基础上加入了 owner proposal 语义、counted probe 证据与 fail-closed
预算契约。

### CBCC-overlapping —— 贡献归属分解的先例

Jia, Mei, Zhang, "Contribution-Based Cooperative Co-Evolution for
Nonseparable Large-Scale Problems With Overlapping Subcomponents",
IEEE TCYB 52(6), 2022（本地有）。

共享变量分配给贡献最大的子分量 + round-robin 骨架 + 贡献奖励。与 ARAC-OC
的差异：CBCC 在分解层做归属，ARAC-OC 在候选层做仲裁。kernel v4 的
"贡献归属 candidate"（把 CBCC 式归属作为一个候选进入 tournament）可直接
引用此先例。

### Overlapping CC (GECCO 2024)

Komarnicki, Przewozniczek, Tinós, Li, "Overlapping Cooperative
Co-Evolution"（本地有）。共享变量多重归属 + 组件间共享关系调制资源分配
——为"owner 冲突分析调制预算"提供另一种形式化的引用支点。

## 2. 定位叙事（related work 结尾段用）

### HCC 自己留的口子

Two-Phase CC（HCC，arXiv 2503.21797）结论原文：

> "Exploring more intelligent and diverse methods for balancing
> exploration and exploitation within HCC."

引用并接续：ARAC-OC 的 GCB（oracle-free 结构发现之上的证据驱动在线调度）
是对该开放问题的直接回答。配合既有发现（HCC-ES 读取 oracle 构造文件、
零黑盒检测），定位表述为：**"HCC 测的是 oracle 上界，ARAC-OC 测的是黑盒
可达能力 + 在线调度"**。

### AOS lineage（为 strict-best archive 提供理论支撑）

Pei et al., AOS survey（本地有）：稀疏大改进场景的 **extreme credit** 是
credit assignment 的公认正确形式之一——ARAC-OC 的 strict-best archive 即
extreme credit 的实例，写作时引用该 lineage；rank/相对中位数归一是
lagged credit 的标准解法（对应 Gate：尺度归一 lagged EMA 回放）。

## 3. 机制借鉴（正文方法节引用）

| 机制 | 文献 | 嫁接点 |
|---|---|---|
| 难度×贡献双轴资源分配 | DCCC (TEVC 2023) | episode 状态从 material 单轴 → (recent_contribution, difficulty) 双轴 |
| 贡献衰减中断 | CCFR (TEVC 2017) / CCFR3 (ESWA 2022) | 替换 w1 固定窗 ladder：释放判据 = 贡献衰减拐点 |
| 按需 sense 预算 | SACC (Appl Intell 2017) | probe 份额 ∝ 实测灵敏度，残差触发 |
| 两级 surrogate 候选生成 | Zhao et al. (TEVC 2025) | surrogate 只生成候选、完整目标决定接受 |
| 学习路线协议 | LCC / LH-CC / RLDO | 状态特征表 / MDP 形式化 / 泛化划分 |
