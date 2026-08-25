# ARAC-OC 阶梯式升级方案 v2.1

日期：2026-08-23  
基线：`arac-recovered-baseline-20260823-v1`  
状态：文献复核后的可执行方案，逐级预注册；尚未接入生产

文献检索记录：`references/lit-review/arac_oc_stepwise_upgrade_literature_2026-08-23.md`

## v2 → v2.1 修订记录

1. **S3/S4 增加 patch lane 总量上界与逐坐标有界退避**：修复"半径下界
   base_radius 导致慢性拒绝坐标每访问必烧 8-FE lane"的常驻税泄漏
   （修订见 §4.3、§5-S3、§5-S4）。
2. **统计口径增加零误差守卫**：ratio 类指标在 baseline error 接近数值零时
   的 NaN/爆炸风险，改为预注册 eps floor + log10 误差差值 paired CI
   （见 §4.2）。
3. **S5a 分歧方向重定义**：`top − second` 两 owner 规则在 |M(j)|≥3 时不稳定，
   改为最大权重 owner 对其余加权均值方向，平局按 coordinate_id 字典序
   （见 §5-S5a）。
4. **U1 per-case reachability 表升级为 S1 协议的强制附件**；S2 的
   `host_unreachable` 拆分为 `no_acceptance_event` 与 `host_unreachable`
   两个语义（见 §5-S2、§7）。
5. v2 其余内容（冻结边界、判据、阶梯定义、消融矩阵、止损线、参考文献）
   未改动。

## 1. 总评与修改结论

原 v1 的方向是对的，尤其是"恢复基线不可侵犯、每级只加一个机制、把
conforming 与 conflicting 分开、机制必须真实触发"四条纪律。这比直接把
state、radius、soft routing 一次性并入生产更容易获得可归因结果。

但 v1 不能原样执行，存在六个契约级缺口：

1. `confidence` 没有证明在当前 checkpoint 中可用；现有历史记录中该量可能
   恒定为 1.0，直接使用会产生伪创新。
2. S3 的"8 FE 试共享坐标"没有规定 scope 内有多个共享变量时怎么分配，无法
   精确复现，也无法保证预算不借用。
3. S4 的 `base_radius`、写集、上下文 reset 与跨 episode 持久化缺少完整状态
   契约。
4. S5 同时改变 disagreement 候选和 soft owner weight，违反"一级一个机制"，
   归因会再次混淆。
5. "触发次数 > 0"不能作为所有列的统一判据：AOB/conforming 的正确行为可能
   是静默；必须区分目标 host 的 reachability 与 conforming 的 no-tax。
6. `ratio <= 1.05` 只能作为预注册的非劣 margin，不能单独支撑 superiority；
   必须同时报告 paired CI、anytime AUC 和 exact contract。

因此 v2 保留 S1-S5 的故事，但把 S5 拆成 S5a/S5b，并把 S3/S4 写成可直接
实现的固定契约。DCCC、CCFR3、SACC 和学习型调度保留为 S6 以后路线，不进入
首个升级候选。

## 2. 研究主张边界

本方案只主张验证以下组合：

> 在相同 Phase-I checkpoint、相同外层 action、相同总 FE 和相同 strict-best
> ledger 下，CTP/GSS 内部的结构化扫掠、共享坐标 patch、局部持久 trust-region
> 是否能在 conflicting overlap host 上产生可归因增益，同时在 ov0/AOB
> conforming host 上保持非劣。

不主张：

- AOB conforming 上必然有终值改善；
- ADMM 收敛、全局收敛或任何数学等价性；
- shared-patch 改变 Phase-I 或 selector 的最优动作；
- 一篇已有文献已经提出 ARAC 的完整组合；
- patch 在当前 S6 screen 上已经恢复历史均值。

## 3. 冻结边界与共用输入

### 3.1 不可修改

- freeze manifest、恢复 source、历史 checkpoint/vendor tree；
- Phase-I 180,000 FE 协议和输出结构；
- 外层 selector 输入、输出和动作类型；
- AOR/SMP/GCB 的默认实现；
- patch-off baseline 的 receipt、FE、state/hash contract。

### 3.2 允许挂载

首个升级只允许挂载在**真实可达的 CTP/GSS block-sweep host**。SMP/AOR/GCB
不挂 patch。若某个生产 loop 在 preflight 中没有 CTP/GSS scope 访问，标记
`host_unreachable`，不通过"强行触发"解决。

### 3.3 三类数据源

| 数据源 | 用途 | 结论口径 |
|---|---|---|
| ov0 generator | 证明机制静默与零税 | exact route/FE/hash，允许无触发 |
| AOB 24 cases | conforming preservation | paired anytime AUC 非劣，终值非劣；不要求 patch 触发 |
| conflicting overlap generator | 机制增量 | matched host reachability、嵌套归因、fresh seed 相对收益 |

AOB 的历史均值只作为背景，不再作为升级候选的唯一 oracle。候选首先与冻结
recovered baseline 做 paired comparison。

## 4. 通用 Gate 与统计口径

所有 level 共用以下 contract：

1. checkpoint hash、Phase-I FE、terminal FE、action seed、boundary profile 完全一致；
2. exact FE、strict-best 单调、receipt/action/state hash 可重放；
3. patch lane 只能从 CTP/GSS 自身 reservation 划出，不得从 sense/probe/tail
   或其他 action 借预算；
4. 发生器的构造标签、owner proposals、共享变量真值和 host route 写入 receipt；
5. seed registry 冲突时 preflight 失败，不静默替换。

### 4.1 快速筛查与确认

- **screen**：AOB 24 cases × 5 paired seeds（117、123、129、135、141）；
  conflicting generator 6 个预注册 cell × 5 seeds；用于暴露 reachability 和
  明显回归，不作最终 superiority。
- **confirmation**：新 seed registry，至少 10 paired seeds；只在 screen
  通过后运行。建议 fresh seeds `[151, 157, 163, 169, 175]` 作为第一批，运行前
  必须做冲突预检并登记。

### 4.2 判据

**零误差守卫（v2.1 新增，先于一切 ratio 计算）**：所有 ratio/几何均值指标
在计算前对误差施加预注册 floor：

```text
err_safe = max(error, eps_ref)
eps_ref  = 逐 case 预注册，取该 case 冻结 baseline 全 seed 的
           final-error 噪声地板（定义：全 seed final error 的
           min-positive 值的 1/10），运行前写入协议，不得事后调整
```

主统计同时报告 `log10(err_safe_candidate) - log10(err_safe_baseline)` 的
paired CI（与几何均值比等价但零安全）。任何 arm 出现 raw error == 0 时，
该 cell 只使用 log 差值口径，并在 receipt 标记 `zero_error_floor_used`。

设 `R = candidate_err_safe / baseline_err_safe`，越小越好。

- **ov0 no-tax**：route、FE 分类、terminal FE、receipt hash 和 final error
  必须逐位一致；机制触发为 0 是通过条件。
- **AOB/conforming preservation**：paired geometric mean `R <= 1.05`，且
  95% paired bootstrap CI 上界 `<= 1.05`；anytime AUC 的 paired CI 不得低于
  `-0.05` 的相对 margin。只作非劣，不作 superiority。
- **conflicting efficacy**：至少一个预注册 conflicting cell 满足 geometric
  mean `R < 0.95` 且 95% paired bootstrap CI 上界 `< 1.0`；同时不得有任何
  cell 的 CI 下界超过 `1.05`。这只是 level promotion gate，不是最终论文的
  全部 superiority 证据。
- **reachability**：目标 conflicting host 必须有非零 patch receipt、候选
  trace 非空、消耗 FE 大于 0；AOB/ov0 允许为 0，但 receipt 必须明确记录
  `silent_conforming`。

### 4.3 patch lane 总量上界（v2.1 新增）

S4 的半径下界（`r_j >= base_radius_j`）意味着长期被拒绝的坐标会在每次
scope 访问时照常消耗 lane。为防止常驻税泄漏，预注册两条有界规则：

1. **总量上界**：单一 run 内 patch lane 累计消耗不得超过该动作自身
   operator reservation 的 `5%`；达到上界后该 run 内 lane 自动停用，
   receipt 记录 `lane_budget_exhausted`。上界值运行前冻结，不得按结果调整。
2. **逐坐标有界退避**：每个 active coordinate 维护连续拒绝计数
   `rej_j`（strict-best 拒绝即 +1，接受即清零）：

   ```text
   rej_j >= 4  ->  该坐标 lane 触发间隔翻倍（stride *= 2）
   stride 上界 = 16 次 scope 访问；rej_j 清零时 stride 复位为 1
   ```

   退避只调节单一坐标自身的触发频率，不改变动作选择、scope 顺序或
   预算分配；`rej_j`、`stride_j`、跳过次数全部入 receipt。

## 5. 阶梯定义

### S0：Baseline guard（无新机制）

执行 freeze verifier，并把 patch-off recovered baseline 的所有输入/输出 hash
写入本 level manifest。S0 不跑性能创新，不产生新的 production 行为。

通过后才允许 S1。S0 失败意味着环境或基线漂移，停止全部升级。

### S1：Leverage-priority sweep（纯重排，零新增 FE）

#### 机制

对 CTP/GSS 的 scope 建立静态优先级：

```text
leverage(scope) = number of Phase-I shared variables in scope
priority(scope) = (-leverage(scope), original_scope_rank)
```

首个 block sweep 的前 `20%` 槽位使用该顺序，之后恢复冻结 baseline 顺序。
ov0 时 `leverage=0`，构造保证顺序和 route 不变。

#### confidence 处理

v1 中的 Phase-I member confidence 只有在 checkpoint 中存在、非恒定、且有 schema
和 hash 证明时才可作为乘数。当前版本默认**不使用 confidence**，只使用共享变量
计数；否则容易把恒定 `q=1.0` 当作新信号。

#### Gate S1

- ov0 exact no-tax；
- AOB CTP/GSS host 的 anytime AUC 非劣；
- scope order trace 非空且与 baseline 有可审计差异；
- 不改变 action route、总 FE 和 selector；
- **强制附件（v2.1）**：U1 的 per-case reachability 表必须作为本 gate 协议
  的附件预先归档，S1 的适用范围以该表为准。

### S2：Propagation handoff（纯重排，零新增 FE）

#### 机制

在一个 action episode 内维护唯一 transient 字段 `last_improved_scope`。若 scope
`s` 产生 strict-best 接受，则下一个可执行槽位优先选择与 `s` 共享变量的相邻
scope；没有邻居或没有接受时回到 S1 顺序。该状态不跨 checkpoint 持久化，避免
上下文漂移伪记忆。

#### 必须补的 receipt

每次选择写入：`scope_rank_before`、`selected_scope`、`handoff_source_scope`、
`shared_neighbor_count`、`handoff_reason`。

#### 静默语义（v2.1 修订）

没有真实 scope acceptance event 时**不再标记** `host_unreachable`，而标记
`no_acceptance_event`（host 可达、只是本期无接受——正常事件）；
`host_unreachable` 仅用于 preflight 证实 CTP/GSS scope 访问路径不存在
（结构问题）。两者在汇总中分开计数，不得混用。

#### Gate S2

S2 与 S1 成对比较；必须证明 handoff trace 实际发生，并且 AOB/AOB-conforming
列不劣。S2 失败时保留 S1，不进入 S3。

### S3：Bounded shared-coordinate micro-patch（首个新增 FE 机制）

#### 机制与候选契约

每次 CTP/GSS scope 访问预留一个固定 **8-FE lane cap**，但不强制浪费 FE：

1. 从 scope 内共享变量按 `(-owner_count, Phase-I rank, coordinate_id)` 选最多
   4 个 active coordinates；
2. 每个 active coordinate 生成一对完整目标候选：

   ```text
   x_plus[j]  = clip(x[j] + rho * range[j])
   x_minus[j] = clip(x[j] - rho * range[j])
   rho = 1/16
   ```

3. 每对候选各消耗 1 FE，因此 `consumed_fes = 2 * min(4, active_count) <= 8`；
   未用的 reservation 明确返回 CTP/GSS 自身 continuation，receipt 记录
   `reserved_fes=8`、`consumed_fes`、`returned_fes`；
4. `best_error_before` 和 `incumbent_before` 从 ledger 读取，禁止重复评价；
5. 所有候选过 bounds、finite-value 和 strict-best 检查；不使用 owner proposal、
   classifier、`u` 或新 probe FE。

这一定义修复了 v1 的歧义：一个 scope 有多个 shared coordinates 时，最多四个
坐标、每个坐标一对候选，FE 账永远可对账。

#### 预算纪律（v2.1 新增）

S3 起，patch lane 受 §4.3 两条规则约束：run 级总量上界（operator reservation
的 5%）与逐坐标有界退避（rej ≥ 4 触发 stride 翻倍、上界 16）。lane 停用或
跳过必须显式入 receipt，不允许静默。

#### Gate S3

- 对照为 S2，A0 是 patch-off baseline；
- conflicting host 必须产生非零 patch receipt 和 candidate trace；
- acceptance rate、consumed/returned FE、lane 总量与退避计数、strict-best
  全可审计；
- AOB 只要求非劣和不改变外层 route；
- 失败不进入 S4。

### S4：Persistent trust-region state（`z_j,r_j`，不加 `u_j`）

#### 状态

每个 shared coordinate 维护：

```text
z_j              patch center
r_j              current radius
base_radius_j    rho * range_j
context_hash_j   local context hash
last_scope       last access
reset_count      reset counter
rej_j            consecutive rejection counter（v2.1，见 §4.3）
stride_j         lane trigger stride（v2.1，见 §4.3）
```

初始化 `z_j=incumbent[j]`、`r_j=base_radius_j`。成功接受时：

```text
z_j <- accepted_value
r_j <- min(4 * base_radius_j, 1.25 * r_j)
```

失败或无收益时：

```text
r_j <- max(base_radius_j, 0.5 * r_j)
```

半径无法缩至 base 以下，因此慢性拒绝坐标的热量由 §4.3 的退避与总量上界
兜底，而不是由半径收缩兜底——这是有意设计：半径语义保持单纯（搜索尺度），
频率控制全部交给可审计的 stride。

#### Local context hash

`context_hash_j` 只覆盖 checkpoint hash、selector hashes、component id 和
component 相关但不属于本 patch 写集的 incumbent 坐标。被 patch 写入的坐标必须
排除；否则 self-accept 会抹掉半径记忆。发生外部上下文变化时：

```text
z_j <- incumbent[j]
r_j <- base_radius_j
reset_count <- reset_count + 1
context_reset_reason = external_context_change
```

`state_hash`、`radius_trace`、`context_hash_before/after` 和 reset reason 必须
进入 receipt。S4 不引入 `u_j`，隔离状态本身的增量。

#### Gate S4

对照为 S3；必须在 toy、matched CTP/GSS 和 snapshot/restore 上证明：成功扩径、
失败缩径、self-accept 不 reset、外部变化 reset、退避/总量上界触发可见、
v1/v2 state fail-closed。

### S5a：Disagreement-direction candidate（conflicting only）

#### 输入边界

owner proposals 必须来自已有 checkpoint/sense receipt 或 matched-host fixture；
没有 proposal 就 fail-closed，不新增隐式 probe。

#### 方向定义（v2.1 修订，替代 v2 的 top−second 规则）

对 shared coordinate `j`，owner 集合 M(j) 可能含三个以上 owner，
`top − second` 对平局与 proposal 噪声不稳定。固定为：

```text
g*           = argmax_g w_g（平局按 coordinate_id 字典序，确定性）
rest_mean_j  = Σ_{g≠g*} w_g * proposal_g[j] / Σ_{g≠g*} w_g
d_j          = proposal_{g*}[j] - rest_mean_j
consensus_j  = Σ_g w_g * proposal_g[j]
```

其中 `w_g` 在 S5a 阶段为均匀权重 `1/|M(j)|`（S5b 才引入更新）；候选只增加
`consensus_j`、`consensus_j + sign(d_j)*r_j` 和 `consensus_j - sign(d_j)*r_j`
三类；候选仍由完整目标 + strict-best 决定。`u_j` 不存在，方向不得由历史
积分器生成。

#### Gate S5a

只在 conflicting generator 上运行；AOB 只作 silent/conforming preservation。
候选 trace 必须证明 disagreement candidate 与 S4 候选集不同，且相同 FE、相同
checkpoint、相同 host。

### S5b：Soft owner weighting（单独升级）

S5b 不再同时改变候选族。它只把 owner proposal 权重做有界连续更新：

```text
w_g <- (1-alpha) * w_g + alpha * rank_normalized(recent_owner_progress_g)
w_g <- project_to_simplex(w, w_g >= w_min)
alpha <= 0.2
```

所有 progress 先做 context-local rank normalization；不得把 raw error 跨 case
直接比较。权重只能影响 consensus candidate 的中心与 S5a 的 `g*`/`rest_mean`
计算，不得改变 action choice、scope order、sense/probe/tail budget 或
component 停用。

S5b 对照为 S5a。若 S5a 通过而 S5b 失败，保留 S5a；不允许把两个机制合并后
宣称"soft routing"有效。

### S6（后续，不进入首个升级）

S6 才考虑 DCCC 式 difficulty × contribution、CCFR3 式贡献衰减中断、SACC
式 sensitivity budget 和 AOS/UCB credit。它们改变的是资源调度，不是 shared
candidate kernel，必须另开协议和消融，不能与 S5b 同批加入。

## 6. 嵌套消融矩阵

| 臂 | 扫掠 | patch | 状态 | disagreement | soft weight |
|---|---|---|---|---|---|
| A0 | baseline | off | none | off | off |
| A1 | S1 | off | none | off | off |
| A2 | S2 | off | none | off | off |
| A3 | S2 | S3 | none | off | off |
| A4 | S2 | S3 | S4 `z,r` | off | off |
| A5 | S2 | S3 | S4 `z,r` | S5a | off |
| A6 | S2 | S3 | S4 `z,r` | S5a | S5b |

归因链：

```text
A1-A0：静态 leverage 排序
A2-A1：传播接续
A3-A2：8-FE shared-coordinate patch
A4-A3：持久 trust-region state
A5-A4：disagreement candidate
A6-A5：soft owner weighting
```

每次只改变一个列；A0/A1/A2 的"无新增 FE"不能和 S3 之后的 FE 机制混合解释。
S3 及以后的臂全部受 §4.3 lane 预算纪律约束（统一规则，不作为归因变量）。

## 7. 实验执行顺序

```text
U0 freeze verifier
  -> U1 host reachability
  -> S1 screen
  -> S2 screen
  -> S3 contract + matched host
  -> S4 state contract + matched host
  -> S5a conflicting screen
  -> S5b conflicting screen
  -> AOB preservation/no-tax
  -> fresh-seed confirmation
  -> production E2E（仅在全部前置 gate 通过后）
```

**U1 输出规约（v2.1 新增）**：U1 必须产出 per-case reachability 表
（case × action × scope-visit 计数 × 是否可达），作为 S1 gate 协议的强制
附件归档；S1 的适用范围声明不得超出该表覆盖的 case 集合。

任一级失败立即关闭该级并回退到上一级；不在同一级内部调阈值。每一级都生成
独立 manifest、raw receipts、summary、protocol hash 和 result hash。

## 8. 论文与创新表述

可以声称：

1. 在冻结 Phase-I/outer action 下，提出一个固定 FE、strict-best、可恢复的
   shared-variable upgrade ladder；
2. 通过 nested attribution 分开 leverage order、propagation、coordinate patch、
   persistent radius、disagreement candidate 和 soft owner weight；
3. conforming 与 conflicting 使用不同的预期收益与数据源，避免把 AOB 静默误判
   为机制失败。

不能声称：

- 文献已经证明该组合优于所有 overlapping CC；
- AOB 上 patch 必须带来终值收益；
- 当前 CTP S6 screen 已被 tail ablation 恢复；
- soft owner weighting 等价于 consensus ADMM；
- 任一 level 通过就代表生产 selector 已升级。

## 9. 失败与止损线

- U0 失败：冻结 baseline 漂移，停止升级；
- U1 失败：挂载点不可达，停止性能比较，修 host contract；
- S1/S2 失败：关闭重排，不接入 patch；
- S3 失败：保留 S2，放弃当前 patch candidate；
- S4 失败：回退无状态 S3，不保留半径记忆；
- S5a 失败：保留 S4，不加入 disagreement；
- S5b 失败：保留 S5a，不加入 soft weight；
- lane 总量上界触发：属正常预算事件，记录 `lane_budget_exhausted`，
  不构成该级失败；但 AOB preservation 因 lane 开销失败时，patch 默认关闭；
- AOB preservation 失败：patch 默认关闭；
- production E2E 失败：只保留 matched-host 机制结果，不宣称 production
  superiority。

## 10. 核心参考文献

完整检索记录与核验状态见
`references/lit-review/arac_oc_stepwise_upgrade_literature_2026-08-23.md`。

1. Jia, Mei, Zhang. Contribution-Based Cooperative Co-Evolution for Nonseparable Large-Scale Problems With Overlapping Subcomponents. IEEE TCYB, 2022. DOI: `10.1109/TCYB.2020.3025577`.
2. Komarnicki et al. Overlapping Cooperative Co-Evolution for Overlapping Large-Scale Global Optimization Problems. GECCO, 2024. DOI: `10.1145/3638529.3654171`.
3. Sun et al. Decomposition for Large-scale Optimization Problems with Overlapping Components. CEC, 2019. DOI: `10.1109/CEC.2019.8790204`.
4. Pryor, Peerlinck, Sheppard. A Study in Overlapping Factor Decomposition for Cooperative Co-Evolution. SSCI, 2021. DOI: `10.1109/ssci50451.2021.9659875`.
5. Fan, Wang, Han. Cooperative Coevolution for Large-Scale Optimization Based on Kernel Fuzzy Clustering and Variable Trust Region Methods. IEEE TFS, 2014. DOI: `10.1109/tfuzz.2013.2276863`.
6. Liuzzi et al. Trust-Region Methods for the Derivative-Free Optimization of Nonsmooth Black-Box Functions. SIAM J. Optim., 2019. DOI: `10.1137/19m125772x`.
7. Yang et al. Efficient Resource Allocation in Cooperative Co-Evolution for Large-Scale Global Optimization. IEEE TEVC, 2017. DOI: `10.1109/tevc.2016.2627581`.
8. Yang et al. CCFR3: A cooperative co-evolution with efficient resource allocation for large-scale global optimization. ESWA, 2022. DOI: `10.1016/j.eswa.2022.117397`.
9. Xu et al. Difficulty and Contribution-Based Cooperative Coevolution for Large-Scale Optimization. IEEE TEVC, 2023. DOI: `10.1109/tevc.2022.3201691`.
10. Liu et al. Cooperative Co-evolution with Soft Grouping for Large Scale Global Optimization. CEC, 2019. DOI: `10.1109/cec.2019.8790053`.
11. Mahdavi et al. Cooperative co-evolution with sensitivity analysis-based budget assignment strategy for large-scale global optimization. Applied Intelligence, 2017. DOI: `10.1007/s10489-017-0926-z`.
12. Pei et al. Adaptive Operator Selection for Meta-Heuristics: A Survey. IEEE TAI, 2025. DOI: `10.1109/tai.2025.3545792`.
13. Tian et al. An Enhanced Differential Grouping Method for Large-Scale Overlapping Problems. arXiv: `2404.10515`.
14. De Falco et al. Investigating surrogate-assisted cooperative coevolution for large-scale global optimization. Information Sciences, 2019. DOI: `10.1016/j.ins.2019.01.009`.
