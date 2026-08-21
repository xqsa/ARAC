# ARAC-OC 问题诊断与文献机制对照报告

日期：2026-08-21
输入：`docs/arac-oc-progress-log.md`（全量时间线）、`docs/handoff-2026-08-20.md`、
`一句话概括.txt`、`评审意见.txt`、本地文献库（`references/lit-review/` 16 篇精选已拷入）、
外部检索（MetaBBO / AOS / CC 资源分配近年进展）。

---

## 一、当前问题的重新诊断（按根因分层）

通读全部 gate 时间线后，我认为 ARAC-OC 当前不是六个孤立问题，而是
**一个根因 + 四个下游症状**。

### 根因：调度器是"无状态规则链"，而问题本质是"双轴状态决策"

v4.0 → v5.3 的全部收据证据显示同一个反复模式——**跷跷板**：

- R2 要：打破零增益垄断（采样公平）+ AOR 梯道供血（显现慢）；
- S5 要：material 领跑者不被打断（CTP 增益到达尺度 > 75k）；
- 每个版本修一边、破另一边（v4.3 S5 5.05× 但 R6 1.37；v4.3.1 S5 25.9×；
  v5.2 R6 达标但 S5 39.65×；v5.3 为 S5 再加机制）。

跷跷板不是"参数没调好"，而是**决策状态空间少了一个轴**。现在的门控本质上是
单轴的（material / 非material），但真实的状态空间至少是两轴的：

```text
                当前还有贡献
              ┌───────┬───────┐
   难度低     │ 正常exploit│ 快释放  │   ← R2 的平坦臂在这
（已开发的）  ├───────┼───────┤
   难度高     │ 保护长跑 │ 供血等待 │   ← S5 的 CTP 在左下，
（未开发的）  │ (S5-CTP)│(R2-AOR)│      R2 的 AOR 在右下
              └───────┴───────┘
```

单轴规则永远只能区分左右列，区分不了上下行——所以"保护 S5"的规则必然
误伤 R2 的平坦臂，"释放 R2"的规则必然误杀 S5 的 CTP。**v5.3 的 grace/ladder
是在单轴框架里打补丁模拟第二轴，这是机制堆叠的直接来源。**

### 症状 1：窗口粒度与增益到达尺度的错配

75k 固定窗的逐窗 materiality 判定，对增益到达尺度 >75k 的 episode 必然伪释放
（v5.2 S5 铁证：seg 9 前逐位相同，75k 子窗增益恰为 0 → 1.22M 打磨链全灭）。
**这是"固定窗口贡献评估"的已知缺陷，文献中已被 CCFR3 正面解决**（见 §二-B）。

### 症状 2：无事前可调度信号

C_j（Gate 47 无间隙）、G_coupled（fresh 0.0457）、G_int（-0.1109）、
lagged EMA（0.1163，且被跨 context 尺度混淆）全部未过 authority。
**这是 AOS（Adaptive Operator Selection）文献里的经典 credit assignment
问题**，有成熟 taxonomy 可对照（见 §二-C）。

### 症状 3：协调开销与候选质量

sense 税 45%；算子脉冲 8 FE 象征性规模 vs 历史车道千级 FE；
shared kernel 候选 0/60 → 17/60。**候选生成是独立于调度器的瓶颈**，
文献有 surrogate 路线（见 §二-D）。

### 症状 4：归属缺失

机制堆叠速度超过证据增长（v5.3 的 A3/R6 审计失败 + S5 15-20× ON 劣于 OFF）。
全局消融表缺席——这已是共识，不展开。

---

## 二、文献中可直接借鉴的机制（按问题映射）

### A. 治根因：双轴状态 —— DCCC（难度 + 贡献双信号）

**文献**：Xu, Luo, Lin, Chang, Tang, "Difficulty and Contribution-Based
Cooperative Coevolution for Large-Scale Optimization", IEEE TEVC 27(5), 2023。
（本地：`Difficulty_and_Contribution-Based_Cooperative_Coevolution_for_Large-Scale_Optimization.pdf`）

核心观察与 ARAC-OC 的跷跷板**逐字对应**：

> "difficult subproblems 的资源投入不能快速改善 fitness，导致早期贡献小、
> 被现有贡献分配方案忽略。"

DCCC 的做法：用 fitness-distance correlation（FDC）在进化过程中**在线量化
每个子问题的难度**，资源分配 = f(贡献, 难度) 双输入，而不是贡献单输入。

**借鉴形式**：把 GCB 的 MATERIAL_RUN / DISCOVERY_RUN 二值状态机推广为
连续双轴状态：每 episode 维护 (recent_contribution, difficulty_estimate)。
difficulty 可以用 episode 自身轨迹的廉价代理（增益到达间隔的历史分布、
rate 衰减曲线形状），不需要真 FDC。S5-CTP = 高难度高潜力 → 保护；
R2 平坦臂 = 低难度零贡献 → 快释放；R2-AOR = 高难度未显现 → 梯道供血。
**三个 case 的规则第一次可以用同一个两输入函数表达，而不是三个 if。**

### B. 治症状 1：自适应贡献评估替代固定窗口 —— CCFR / CCFR3

**文献**：Yang et al., "Efficient Resource Allocation in Cooperative
Co-Evolution", IEEE TEVC 21(4), 2017（CCFR，本地有）；
Yang et al., "CCFR3", Expert Systems with Applications 203, 2022（前沿/CCFR3.pdf）。

CCFR 三个可直接搬的机制：

1. **贡献用最近周期平滑**（ΔF_i = 上次与本次平均），不用全程累计——
   对应 ARAC-OC 的 material 判定应读 recent rate 而非历史票据信用；
2. **停滞清零**：子种群停滞时贡献置零、自动退出后续周期——对应
   plateau release，但判定依据是**贡献衰减**而非固定窗口；
3. **全平等时重启**：所有贡献相等时重置进程，误判停滞的臂可恢复——
   对应 released 臂的可逆性（现在 v5.x 的 release 近乎不可逆）。

CCFR3 更进一步：**去掉固定代数参数 GEs，按每周期贡献自适应中断**——
这正中"75k 窗口粒度毒药"：释放判据应该是"贡献衰减曲线拐点"，
不是"w1 窗口内为零"。CTP 在 S5 的增益到达尺度 >75k 时，
衰减检测天然等待，不需要 grace 补丁。

### C. 治症状 2：credit assignment 的正确形式 —— AOS 文献

**文献**：Pei, Mei, Liu, Zhang, Yao, "Adaptive Operator Selection for
Meta-Heuristics: A Survey"（本地有）；Pei et al., "Learning from Offline
and Online Experiences: A Hybrid Adaptive Operator Selection Framework",
GECCO 2024（HF，外部检索到）。

AOS 五十年文献对"用收益选算子"踩过的坑有完整记录，三条直接可用：

1. **credit 必须归一化/秩化**：raw gain 跨 context 不可比是已知问题，
   主流做法是 rank-based 或相对中位数归一——这正是评审第 1 条指出的
   lagged EMA 尺度混淆的文献级解法；
2. **extreme credit vs average credit**：稀疏大改进场景下用极值信用
   （ARAC-OC 的 strict-best archive 本质上已是 extreme credit——
   可以在论文里引用这条 lineage 作为理论支撑）；
3. **exploration 项显式化**：Dynamic Multi-Armed Bandit / Adaptive Pursuit
   把"选估计最优"和"采样不确定臂"写进同一个公式（UCB 式），
   比手写的 ticket/challenger/escalation 三车道更紧致、更少交互 bug。

HF（GECCO 2024）= 离线经验 + 在线学习的混合 AOS——这正是评审建议的
"offline contextual value model 作 shadow challenger"的文献先例与协议模板。

### D. 治症状 3：共享变量与候选生成

**文献 1**：Jia, Mei, Zhang, "Contribution-Based Cooperative Co-Evolution
for Nonseparable Large-Scale Problems With Overlapping Subcomponents",
IEEE TCYB 52(6), 2022（本地有）。两个机制：

- 共享变量分配给**贡献最大的**子分量（contribution-based decomposition）——
  可作为一个**廉价 candidate** 进入现有完整目标仲裁（不替代 tournament，
  只增加候选多样性）；
- round-robin 骨架 + 贡献奖励的优化框架，同时保合作频率与资源分配。

**文献 2**：Pryor, Peerlinck, Sheppard, "A Study in Overlapping Factor
Decomposition for Cooperative Co-Evolution"（本地有，FEA）。
FEA 对共享变量的处理 = 各重叠 factor 把自己的值注入全局解、
**谁的全局适应度好谁写回**——这就是 ARAC-OC 的完整目标函数仲裁，
说明机制选择正确，且 FEA 是**必须引用的直接先例**（目前文档未见引用）。

**文献 3**：Komarnicki, Przewozniczek, Tinós, Li, "Overlapping Cooperative
Co-Evolution", GECCO 2024（前沿有）。共享变量多重归属 + 
**组件间共享关系影响资源分配**——给"owner 冲突分析"提供了另一种
形式化（不仲裁值，而是让共享结构直接调制预算）。

**文献 4**（kernel 候选生成）：Zhao, Wang, Sun, Jin, Hayat, "Efficient
Large-Scale Expensive Optimization via Surrogate-Assisted Subproblem
Selection", IEEE TEVC 29(5), 2025（前沿有）。surrogate 在随机维上建模、
扰动预测识别最重要变量 → 构造 active subproblem → 二级 surrogate 指导
子代生成。**直接对应评审建议的"小 surrogate 只生成候选、不决定接受"**，
且给出了两级 surrogate 的具体结构。

### E. 治 sense 税：灵敏度自适应预算 —— SACC

**文献**：Mahdavi, Rahnamayan, Shiri, "Cooperative co-evolution with
sensitivity analysis-based budget assignment", Appl Intell 47, 2017（本地有）。
按子分量对目标函数的影响分配计算时间。**借鉴形式**：sense/probe 预算
份额 ∝ 该 component 最近实测灵敏度；灵敏度残差无变化的 component 
跳过 probe（残差触发式探测）——直接把 45% 的固定税变成按需税。

### F. 学习路线的协议模板 —— LCC / LH-CC / RLDO

**文献**（三篇均本地/前沿）：

- LCC（Guo et al., arXiv 2504.17578, 2025）：NN 在优化过程中动态调度
  **分解策略**，PPO 训练，精心设计的状态特征集；
- LH-CC（Qiu et al., 2026）：把"为每个子问题选优化器"建成 MDP，
  meta-agent 选择——**这就是 GCB 选 episode 的 RL 版**；
- RLDO（Tian et al.）：PPO 学分解（交互概率矩阵 → 采样分解 →
  适应度归一化作 reward）。

三篇共同提供的不是"用 RL"这个口号，而是**可抄的工程协议**：
状态特征设计（optimization status features）、reward 归一化方式、
跨问题训练/泛化评估划分。**如果走学习路线，特征集直接参考 LCC 的
状态特征表，训练/测试划分参考 LH-CC 的 generalization 协议。**

### G. 定位强化：HCC 自己留了口子

Two-Phase CC（HCC，arXiv 2503.21797）结论原文：

> "Exploring more intelligent and diverse methods for balancing
> exploration and exploitation within HCC."

HCC 作者自己承认 exploration/exploitation 平衡是开放问题。
ARAC-OC 的 GCB（在 oracle-free 结构发现之上做证据驱动调度）
正是这句话的直接回答——**论文 related work / positioning 里应引用这句**，
把竞争叙事变成"接续 HCC 声明的开放问题"。配合已有发现（HCC-ES 读
oracle 构造文件、零黑盒检测），定位 = "HCC 测的是 oracle 上界，
我们测的是黑盒可达能力 + 在线调度"。

---

## 三、建议的落地顺序（与既有共识合并）

```text
0. 机制冻结（共识）
1. 纯回放三件套（共识：lagged EMA 归一重跑 / production 基线 / oracle-gap 表）
   + 新增：FEA 与 CBCC-overlap 的引用缺口补进 related work
2. 全局消融表（共识）——但消融轴按 §一 的"双轴状态"重新组织：
   不是 7 个机制各自开关，而是先验证"双轴表示"能否用更少机制
   覆盖现有行为（DCCC 式贡献+难度 → 替代 material 单轴 + grace + ladder）
3. CCFR3 式自适应贡献中断 替换 w1 固定窗 ladder（v6.0 候选，预注册）
4. kernel：CBCC 贡献归属 candidate + 两级 surrogate 候选生成（共识的 kernel v3）
5. SACC 式按需 sense 预算（治 45% 税）
6. 学习路线：先 AOS 式秩化 credit + UCB 选择（廉价、可审计），
   再视 oracle-gap 表决定是否上 LCC 式 PPO meta-controller
```

关键纪律：第 2/3 步是**减法**（用文献机制替换手写机制，净减代码），
第 4/5 步是**加法**（新候选、新预算规则）。先做减法再考虑加法。

---

## 四、风险与诚实边界

1. DCCC 的 FDC 难度估计在 1000-D AOB 上的计算成本未验证——先用
   episode 内部轨迹的廉价代理，不直接搬 FDC；
2. CCFR/CCFR3 的全部实验在 disjoint 分解上，共享变量存在时贡献定义
   需要 ARAC-OC 的 global materiality 口径（已有）——嫁接点在
   "贡献衰减检测"，不在其分解层；
3. RL 路线（LCC/LH-CC）的训练成本与 ARAC-OC 的逐 FE 审计文化冲突——
   只适合作为 shadow challenger，不适合进 production 主链；
4. 以上文献映射是**机制级**的，不代表这些论文报告的数字可横向比较
   （benchmark 不同：CEC'2010/'2013 vs AOB-24）。
