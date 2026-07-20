# 动作库设计的文献与证据综述（Action Validation 阶段）

日期：2026-07-20
执行者：Codex
范围：回答"动作库应包含哪些动作，才能在真实 AOB 上对 native HCC 产生可重复正收益"
来源：11 路并行调研（HCC 原文精读、AOB 基准构造、预算/贡献调度文献、overlap 写回文献、
优化器选择文献、分组/LSGO 优化器文献、2023–2026 网络检索、方法论检索、HCC 源码可干预点、
ARAC 现有实现与实验证据、meta-BBO 前沿相关性判断）
状态：设计建议文档；不改变阶段锁，不授权 selector 工作

---

## 1. 关键结构事实（设计动作的约束条件）

1. **AOB 全部 case 是 conforming 构造**。vendor 与真源的 4 个 base 函数（elliptic/rastrigin/
   ackley/schwefel）全部只调用 `rotateVectorConform`，所有子空间共享同一全局 Ovector；
   `rotateVectorConflict` 是死代码。共享变量在 x_opt 处两 owner 取值一致，不存在结构性
   冲突。coupling issue 是**优化中期的瞬态轨迹冲突**（同一共享变量经不同旋转矩阵 R_i
   进入两个子空间的二次型），随收敛自然消失。
   含义：写回类动作的 headroom 来自修复瞬态耦合，**收益窗口集中在早中期**，后期趋近于
   `true_no_writeback` 的零收益；material-positive context 天然稀疏（对应决策门第 3 条，
   必须按 FE 进程分层统计 prevalence）。
2. **子空间权重 w 跨度约 3×10^10**（w_i = 10^{3·N(0,1)}，F4-w.txt 实测最小 1.1e-7、
   最大 3385）。native eq.8 的 Δ 加权和等额预算分配都对 w 盲视——这是写回类与预算类
   动作最硬的结构性 headroom 来源。
3. **链式两两重叠**：AOB 仅相邻组重叠，每对相邻组恰好共享 Γ 个变量，每变量恰好 2 个
   owner，共 19 个 relation/sweep。写回动作作用域严格局部、参数空间小。case 命名 =
   base 首字母 + Γ 下标：E1(Γ=0 控制)、E3(Γ=3)、A4/R4(Γ=5)、S5(Γ=7)。
4. **Γ 提供剂量-反应检验**：机制正确的写回/预算动作收益应随 Γ 单调增长
   （E3 → A4/R4 → S5）。收益不随 Γ 增长的动作其机制叙事可疑。
5. **native 循环的唯一跨 sweep 状态载体是 incumbent**：每次组 dispatch 新建 CMAES
   （mean=gbest[dims], σ=0.5, is_restart=True），协方差/进化路径/早停计数全部重置。

## 2. native HCC 的失效点清单（动作切入点，含代码级证据）

| # | 失效点 | 证据 | 对应动作面 |
|---|--------|------|-----------|
| W1 | eq.8 混合写回**从不被重新评价**（盲写），是整个循环中唯一不受改进守卫的状态变更 | HCC-ES.py L261–268；runner L4387 | 评价守护写回 |
| W2 | 双停滞（Δ_prev+Δ_curr=0）时退化为**算术平均**覆盖，把从未评估的中点写进 incumbent | shared_variable_blend.py L27–28 | 停滞守卫 |
| W3 | Δ 归因错误：整组全空间 Δ（含 unique 贡献）被用来给 overlap 取值加权；且对 w 盲视 | runner L3460/L3546；Benchmarks.py L203–222 | winner-take-all / w 感知写回 |
| W4 | 等额预算：sub_FEs 均分 20 组，不区分 25 维组与 103 维组、不区分 w=1e-7 与 w=3385 | HCC-ES.py L226 | 预算再分配 |
| W5 | 固定拓扑 sweep 序，高 Δ 潜力组与停滞组同等优先级；早期写回误差沿链传播 | HCC-ES.py L228 | sweep 顺序 |
| W6 | 组优化器每 sweep 冷启动（σ=0.5 固定），跨 sweep 无状态 | HCC-ES.py L231–250 | warm start / σ 状态携带 |
| W7 | restart 时 mean 均匀随机重采样 + λ 翻倍，单峰已收敛 case 上纯烧 FE；当前动作库无任何 restart 策略动作 | vendor es.py L213–266 | restart 策略动作 |
| W8 | GloFEs 由 eq.7 按 DO 静态定死（S5 约 30.6% 预算先给 MMES），无法按实际进展调整；HCC Table 2 显示 E4/E6 上全程 MM-ES 反而显著胜 HCC-ES | Two-Phase-CC p.5/p.7 | 全空间 NDA 续跑 |

注意：GitHub 公布代码的 eq.8 权重方向与论文相反，真源（HCC-main）与 ARAC 实现的是论文
方向；任何"复现官方仓库"的比较都会得到不同 native。vendor 的 F*-design.txt 磁盘截断，
目前靠 metadata 重建（已数值验证正确但脆弱）。ARAC 对 E1 的 GloFEs 保底 20% 是有意协议
偏离，对比论文值时必须声明。

## 3. 文献证据要点（按动作族）

### 3.1 共享变量写回

- 写回策略收益**集中在 conforming 关系**；conflicting 上没有任何策略稳定胜出
  （Blanchard 2021；OCC GECCO'24；CCMRO 2023；Sun 2019）。AOB 全局 conforming →
  写回动作有真实但不丰厚的 headroom。
- **贡献定 owner（CBCCO 的 CBD）**：共享变量分给贡献更大的 owner，conforming 上大幅最优，
  conflicting 上持平；反向分配（CBD-R）全面最差——**贡献方向是真实信号，且方向选错的
  代价真实存在**（Jia/Mei/Zhang, IEEE TCYB 52(6):4246–4259, Table VI/Fig.8；
  贡献比 we≥2 起显著，we≥3 稳定）。
- **评价竞争写回（FEA 式 compete / 贪婪接受）**：候选值代入全局评估取 argmin，
  按定义不劣于任选一值；OCC GECCO'24 的消融显示贪婪写回+邻接预算在 conforming 几乎全胜
  （Komarnicki et al., 10.1145/3638529.3654171）。
- 双 owner 均值的失效：非线性耦合下中点可能两不靠，Rastrigin/Ackley 多峰 case 最危险。
- OCC GECCO'24 明确警告**不要向 owner optimizer 内部注入共享变量最优值**（会打断
  优化器自适应）——写回 incumbent 可以，篡改 optimizer 状态须以显式动作形式另行声明。

### 3.2 预算再分配

- **overlap 上唯一被验证安全的倾斜方式：保留 round-robin 主干 + 第一梯队奖励**
  （CBCCO 的 CBO，Algorithm 3）。放弃 RR 的 winner-take-all（CCFR/CCFR3）在 overlap 上
  三篇文献一致报告差于或平于 RR（CCFR3 f13 显著更差；Sun 2019 的 CBCC-RDG3 在 20 个
  overlap 问题一致差于朴素 RR）。
- 贡献度量：近期贡献 EMA（半衰衰减）证据强度最高（CCFR/CBO/CBCCO 三家独立使用）；
  数量级贡献 δ=⌊log10 Δf⌋（DCCC）对尺度差异鲁棒；Morris 敏感度需 2 万 FE 探测且在
  部分函数自报失效，**不可用于 ARAC**（除非能从轨迹免费估计）。
- 必须带防锁死：预算地板（P_min 原则）+ 贡献时间衰减；CBCC 的 rich-get-richer 是
  catastrophic 的文献模板。LSGO review 明确指出 contribution-aware 调度会过度开发
  单一 component。
- 迁移性警告：文献收益几乎全是 3e6 FE 全程终点比较；ARAC 的 Δ 度量 t+h best-so-far，
  短 horizon 内重分配类动作可能尚未摊薄间接成本——t+3h 必须保留。

### 3.3 优化器状态与全空间动作

- overlap 问题偏好**更大的有效搜索单元**：CC-SHADE-ML 静态最优组数 f13→5、f14→2；
  RAG m=8 最优；SLR-ES（稀疏+低秩 ES，不分解）在 CEC'13 overlap/不可分类 F12–F15
  中位数全部第 1（SWEVO 2025）。SGCC-local（软压缩 w=0.3 向 incumbent 收缩）在 C3 类
  显著优于显式分组 CC。三者共同指向：对 overlap case，"扩大有效搜索单元 + 向
  incumbent 收缩 + 全空间方向性搜索"比"更细的正确分组"更有收益。
- HCC Table 2：E4/E6（高重叠 Elliptic）上全程 MM-ES 显著胜 HCC-ES——全空间 NDA 续跑
  在该类 case 有真实 headroom；**全空间动作首选 MM-ES/SLR-ES 类，sep-CMA-ES 在所有
  AOB case 全程大幅落后（E1 差 ~2.3 个数量级）**。
- 组级 optimizer 切换（LH-CC, GECCO 2026）：学习型选择 0.967 > 最优固定 0.864 >
  random 0.756，但收益前提是**异构子问题**；AOB 同构，收益必须实测，且切换必须携带
  per-(group, optimizer) context memory（LH-CC 的 Γ 方案是文献中唯一工程级答案）。
- restart/warm start 的生效条件（Hansen & Kern；Nomura AAAI 2021）：弱全局结构 →
  小步长重启有效；强多模态 → 大种群；错误方向的 warm start 显著有害；无条件热启动
  是文献负面案例。IGKT 式整体迁移在可分结构 catastrophic（f1 差 5.7 个数量级）。
- exp_026 已证伪：R4 上 diagonal_covariance 固定切换 catastrophic rate 0.4，否决。

### 3.4 动作集设计方法论

- arm 总数控制在个位数~12 个：Balcan et al. (AAAI 2021) 组合规模-过拟合界；arm 数
  增加时选择误差线性增长而 credit 噪声不变。参数档位小而离散（每动作 1–3 个参数、
  每参数 2–4 档），**离线按文献先验一次性固定，AOB 上只做执行/不执行的 0-1 检验**。
- abstain 成本为 0（回 native），宁多拒（selective prediction 原则）。
- 信用汇总取窗口内极值/分位数而非均值（Fialho：极端值 credit assignment），否则
  罕见大收益动作被均值饿死——与 ARAC material-positive 阈值设计一致。
- 方向性动作必须配反向对照臂（仿 CBD-R），即使动作整体无收益也能标定信号方向。
- 特征若消耗 FE 或触及未来信息即 leakage（SATzilla 传统）；汇报固定 SBS/VBS/native
  三口径，VBS-SBS gap≈0 时直接判定不需要 selector。

### 3.5 创新空间确认

HCC 目前仅被同团队（SCUT）的 learning-based 调度工作引用（LCC GECCO'25 换分解、
LH-CC GECCO'26 换 optimizer）。"冻结 checkpoint + 显式可审计动作实例 + 同 FE 配对
评估"的协议空间无人占据；显式 sweep 顺序单点动作在文献中基本空白。ARAC 的差异化
应坚持"显式动作 + action ceiling + 证据可分性"，避免走向神经网络动态调度。

## 4. 推荐动作库（进入 action-ceiling 的候选，按优先级分层）

### Tier 1：立即进入 E3/S5 diagnostic（直接攻击 W1–W5，风险可控）

| 动作 | 机制（规则级） | 生效条件 | FE | catastrophic 风险 | 依据 |
|------|---------------|---------|----|--------------------|------|
| **A1 guarded_eq8_writeback**（新增，最优先） | 按 eq.8 算混合点 b；花 1–3 FE 评价候选集 {b, v_prev, v_cur}，写 argmin（平局保 current）；合规版 reprobe_then_exact：候选与顺序冻结进动作 hash，探测 FE 从同 horizon 扣 | 全部 overlap case；Δ 噪声化区域（A4 近收敛、E3 后期） | 1–3 FE/次 | 低：严格接受不劣于 native 盲写，唯一风险是探测 FE 机会成本 | W1/W2；v2 SBS reprobe_then_exact +0.00346（material+ 33%, catastrophic 0/40）；FEA compete |
| **A2 true_no_writeback**（已在 v5） | 目标 dispatch 冻结当前值 | 全部 overlap case | 0 | 低 | v2 mean +0.00345、VBS 赢家 11/40：eq.8 在 300k 阶段平均略微有害 |
| **A3 contribution_owner_writeback**（新增） | shared[γ] ← argmax_j(Δ_j) owner 的值；w 感知变体 argmax_j(w_j·Δ_j)；双停滞时 abstain（保前值） | Δ 差异大的 relation；单峰 case（E3/S5）方向性最明确 | 0 | 中：归因错误反向时覆盖真正更好的值；单选不落"两不靠"中点，风险小于均值 | W3；CBCCO CBD + CBD-R 反向对照证据 |
| **A4 stagnation_guard_writeback**（新增，最小消融臂） | Δ_prev+Δ_curr=0 时**不执行算术平均**，保持前值；其余按 native eq.8 | 停滞 relation；多峰 case（R4/A4）中点劣化风险最高 | 0 | 极低（严格比 native 少一次改动）；收益也可能小 | W2；源码 L172–173 退化分支 |
| **A5 efficiency_budget_reallocation**（已在 v5，按 CBCCO 对齐） | 保留 RR 主干：每组保底 floor=population size，效率 EWMA（Δ/FE）第一梯队组奖励加成，cap 3×uniform，sweep 总量不变 | 贡献不均衡 case（w 跨度保证 AOB 天然不均衡）；S5 优先 | 0 | 低-中：有地板+cap 封顶；饿死低 w 组近乎无害 | W4；CBCCO CBO 是 overlap 上唯一安全的倾斜方式 |
| **A6 delta_priority_scan**（已在 v5） | sweep 边界按上 sweep Δ 降序重排组序 | 组间 Δ 差异大的 case | 0 | 中：改变 eq.8 触发顺序，早期混合误差链式传播路径改变 | W5；v4 smoke E3 +0.039（仅单轨迹，非推断证据） |

Tier 1 的对照设计要求：
- A3 必须同时挂**反向臂**（取 argmin Δ owner 的值）作方向性对照（仿 CBD-R）。
- A1 的 FE 记账在协议中显式化；与 true_no_writeback、native_eq8 同 context 配对。
- 全部写回臂在 E1 上必须无效/abstain（协议 sanity check）。
- 按 FE 进程分层统计 material-positive prevalence（早/中/晚），验证"收益窗口在早中期"的推断。

### Tier 2：Tier 1 gate 通过后进入（机制合理但风险或不确定性更高）

| 动作 | 机制 | 生效条件 | FE | 风险 | 依据 |
|------|------|---------|----|------|------|
| **B1 full_space_nda_continuation_mm_es**（新增；与现有 full_space_sep_cma 对比） | checkpoint 从 incumbent 初始化全空间 MM-ES（或 SLR-ES），horizon 内消耗声明预算，strict improvement 接受 | 高重叠 Elliptic（E3；E4/E6 全程反超证据）；CC 段停滞时发行 | 声明预算（horizon 内扣） | 中-高，强 case 依赖；sep-CMA 版先验最差 | W8；HCC Table 2；SLR-ES F12–F15 全第 1 |
| **B2 restart_policy_action**（新增） | 组 dispatch 构造时冻结 is_restart=False（suppress），或将 restart mean 改为 incumbent[dims]（incumbent_mean_restart） | E3/S5 单峰已收敛段（suppress）；R4/A4 多模态（对照方向） | 0 | 中：多峰上抑制跳出机制 | W7；Hansen & Kern 小步长重启条件 |
| **B3 group_warm_start 族**（已有 stagnation_cross_group_warm_start；扩展 σ 状态携带/软压缩） | (a) 跨 sweep 携带 mean/σ 而非冷启动；(b) SGCC 重构：mean ← x_best + w·(mean − x_best)，w=0.3，σ 同步缩减；仅在停滞证据强时签发 | S5（背景漂移快）；(b) overlap/不可分 case 停滞后 | 0 | 中-高：陈旧 σ 探索不足烧 FE；方向错误的 warm start 文献级负面 | W6；SGCC C3 全胜；Nomura 条件性 |
| **B4 exploration_injection_ats**（新增） | 组停滞 T_i≥τ 时按剂量 prob_r=0.1+0.09·min(T_i,10) 注入指引个体，strict improvement 接受 | 多模态 case（R4） | 1–数 FE | 低-中：严格接受保底，过度注入则 Δ<0 | MSORL ATS 逃逸证据 |

### 明确排除（文献或本项目已证伪/不可行）

- winner-take-all 预算（CCFR/CCFR3 式）：overlap 上三篇文献一致负面。
- Morris 敏感度静态预算（SACC3 原版）：2 万 FE 探测 + 自报失效案例。
- 动态 regrouping / 组粒度切换 / DCC DyG：违反 frozen topology 契约；如需测试只能以
  "显式 topology_deviation 臂 + horizon 末恢复"单独立项，不与冻结拓扑臂混合汇报。
- DG/DG2/HDG 分解学习：单次 ≥3e5 FE，超过 pilot 全部预算。
- 空间压缩（DCBA 式 bounds 收缩）：catastrophic 尾巴最厚（错半区永久排除全局最优），
  仅在 Tier 1/2 全部明确后配保守变体单测。
- 固定全量 bridge 仲裁（exp_019 oracle_no_go）、clipped consensus blend（exp_020 无正收益）、
  probe-utility 触发的 winner-take-all repair（exp_023 负 credit，utility 方向性存疑）、
  cohen_d 门控（exp_020 不可分）、R4 固定 diagonal_covariance（exp_026 catastrophic 0.4）。
- surrogate 内部组件：300k FE 非昂贵场景无必要，且引入不可审计隐藏状态。
- 篡改 owner optimizer 内部状态的隐式注入（OCC GECCO'24 警告）；须以显式动作声明。

## 5. 当前最大的缺口是执行而非设计

1. **v5 arm 集合（9 arms）在诊断规模完全无数据**：唯一推断级证据（v2 oracle300k，
   VBS +0.0107、LCB +0.0069）属于已被替换的 v2 arm 集合，且 v2 实现是复合策略，
   数字只能作 ceiling 线索，不能拼接进 v5 汇总。
2. 推进顺序建议：
   (a) 先按 Tier 1 补齐 A1/A3/A4 三个新 arm（实现量小，均为 0–3 FE）；
   (b) 跑 v5+A1/A3/A4 的 E3/S5 × ≥5 seeds × 300k diagnostic（每 case ≥10 有效 context）；
   (c) 按决策门判定：VBS point estimate 与 95% LCB、material-positive prevalence
       （按 FE 进程分层）、catastrophic 频率、Γ 剂量-反应一致性；
   (d) exp_024/025 的 15 跑已完成但无 gate 评估，尽快补分析（判断 repair 类动作残值）。
3. v2 的 immediate horizon VBS=0：收益完全经 continuation 体现，任何"写回即赚"的
   动作形态不存在——新动作设计必须声明目标 horizon，评价看 t+1/t+h/t+3h 三点。

## 6. 主要来源

- HCC 原文：Two-Phase-CC.pdf（arXiv:2503.21797，GECCO 2025）；真源 E:\HCC-main
- CBCCO：Jia/Mei/Zhang, IEEE TCYB 52(6):4246–4259（贡献定 owner + CBO 预算）
- OCC：Komarnicki et al., GECCO 2024, 10.1145/3638529.3654171（贪婪写回+邻接预算）
- Sun et al. 2019（RDG3；CBCC 在 overlap 差于 RR 的反面证据）
- CCFR/CCFR3：Yang et al., IEEE TEVC 21(4) 2017 / ESWA 203:117397 2022
- DCCC：Xu et al., IEEE TEVC 27(5):1355–1369, 2023
- SGCC：Cooperative Co-evolution with Soft Grouping, IEEE CEC 2019
- SLR-ES：Exploring high-dimensional optimization by sparse and low-rank ES, SWEVO 2025
- LH-CC：arXiv:2604.01241（GECCO 2026，优化器选择+context memory）
- TSCC-CMAES：GECCO 2020（按可分性混合求解器；资源分配 7 个数量级效应）
- MSORL：SWEVO 2024（ATS 停滞剂量注入）；IGKT：CEC 2025（迁移 catastrophic 反例）
- Nomura et al., AAAI 2021（warm start 条件）；Balcan et al., AAAI 2021（组合-过拟合界）
- Fialho thesis（extreme-value credit assignment）
- ARAC 自有证据：results/exp019_g1_oracle_vbs_e3_s5_300k_5seed/oracle300k/（v2 ceiling）、
  exp_020/023/026 证伪记录、src/arac/actions/ 现有实现
