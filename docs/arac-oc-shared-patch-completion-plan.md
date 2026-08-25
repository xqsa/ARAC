# ARAC-OC Shared-Patch 完成方案

日期：2026-08-23
状态：恢复基线已冻结；shared-patch 升级候选待实现与预注册实验
上游：`arac-oc-completion-plan.md`、`arac-oc-dual-overlap-upgrade.md`、
`arac-oc-gate54a-judgment.md`、`references/lit-review/phase2_shared_var_new_directions.md`

本文档替代旧的“一致性分类 + 双支路”升级路线，改以 CTP/GSS 内部的
stateful shared-patch kernel 为唯一新增机制主线。

## 1. 最终裁决

### 1.0.3 Recovered baseline freeze

恢复后的四动作执行锚点已冻结为
`arac-recovered-baseline-20260823-v1`。冻结协议、21 个文件的 SHA-256
和 verifier 见：

- `experiments/historical_recovery/recovered_baseline_freeze_protocol_v1.json`；
- `experiments/historical_recovery/verify_recovered_baseline_freeze.py`；
- `docs/arac-oc-recovered-baseline-freeze.md`。

冻结点的生产默认值为 `patch=false`、`soft_routing=false`、`selector=false`。
下一步升级只允许进入 `experiments/upgrade/`，并按 U0-U4 通过后才可申请
promotion；不得直接覆盖冻结 source、protocol 或 evidence artifacts。升级准备
说明见 `docs/arac-oc-next-upgrade-preparation.md`。

### 1.0 P1 实质判定与恢复优先路线

旧 P1 的形式检查虽然通过，但 12/12 cell 没有 patch receipt，五臂轨迹
逐位相同，归因比均为 1.0000。因此该 P1 只能标记为“机制不可评估”，
不能作为 patch 失败或性能证据。根因是 AOB 的 conforming owner proposal
使冲突级保持 low，planner 选择 `arbitration_only`，CTP/GSS 没有生产挂载点。

恢复路线固定为三条互不混淆的证据链：

```text
AOB：历史性能恢复 + selector 非劣 + patch 零税
conflicting overlap generator：shared-patch 机制增量
生产统一 loop：恢复基线后的端到端组合验证
```

执行入口：

- `experiments/historical_recovery/recovery_first_protocol_v1.json`
- `experiments/historical_recovery/recovery_first_campaign.py`
- `experiments/overlap_shared_patch_matched_host_gate.py`

旧 P2/P4 在 B1/M0 之前不启动。恢复实验关闭 patch、soft routing 和新
selector；matched-host 只在自有 conflicting generator 中强制 CTP/GSS，
不修改生产 planner。

### 1.0.1 B0-B3 恢复 Gate

- **B0 Provenance**：逐 case/seed 校验 checkpoint hash、Phase-I/terminal FE、
  action seed、boundary profile 和 vendor tree hash。
- **B1 Fixed-Action**：完整 24×25×4 矩阵，四动作族都必须覆盖；按历史映射
  `A→AOR, E→SMP, S→CTP, R→GCB` 检查 mapped-action coverage。历史表只保留
  ARAC aggregate 时，结果明确标为 `inferred_protocol_not_bitwise_recovered`，
  不把代表性 lane 当作历史逐位恢复。
- **B2 Selector Parity**：复用 checkpoint 的 Phase-I 特征输入，验证 input
  hash→output hash→selected action；不重新评价动作。
- **B3 End-to-End**：验证 Phase-I→selector→selected action 的 terminal contract，
  每个未恢复 case 单独列出，不用统一平均掩盖失败。

### 1.0.2 当前 Gate 顺序（取代旧 P1-P4）

旧的 AOB P1/P2/P3/P4 链不再作为当前执行入口。当前顺序是：

```text
B0 -> B1 -> B2 -> B3 -> M0 -> M1 -> M2 -> AOB preservation -> production E2E
```

其中 B1 未通过时停止创新；M0 未通过时禁止性能比较；M1 未通过时保留
历史基线并放弃当前 patch 版本；软路由只能在 M1 后作为 CTP/GSS 内部
utility 进入单独 Gate。旧 P1 的形式检查产物保留为历史诊断，不得解释为
机制接入或 superiority 证据。

最终方法固定为：

```text
Phase-I overlap evidence
    -> 原有 selector
    -> 原有阶段二动作
    -> CTP/GSS 内部 Stateful Shared-Patch Kernel
```

冻结边界：

- Phase-I 重叠发现协议不变；
- selector 输入、输出和动作类型不变；
- SMP/AOR 不挂载 patch；
- patch 不能改变外层动作选择；
- 不引入在线 consistency classifier；
- 不引入 RL、在线动作选择器或 `u_j` 驱动的方向 steering；
- Gate 54a/54b 的一致性分类路线保留为“聚合黑盒目标上的不可辨识性”
  失败边界，不再作为主线。

本方案的目标不是重新设计阶段一或调度器，而是证明：在相同的 Phase-I
证据和相同的 CTP/GSS 动作下，持久共享变量状态、共识/分歧候选和局部
trust-region 半径能否相对 local competition v2 产生可归因增量。

## 2. 文献依据与创新定位

机制依据：

- Jia et al., CBCCO，DOI `10.1109/TCYB.2020.3025577`：贡献驱动的
  shared-variable 处理；
- Komarnicki et al., OCC，DOI `10.1145/3638529.3654171`：多 owner
  共享资源协调；
- Sun et al.，overlapping decomposition，DOI `10.1109/CEC.2019.8790204`；
- Blanchard et al.，overlapping CC，DOI `10.1007/978-3-030-85672-4_19`；
- Xu et al., DCCC，DOI `10.1109/TEVC.2022.3201691`：difficulty/contribution
  双轴；
- Yang et al., CCFR/CCFR3：贡献反馈和有界衰减；
- Fasiku & Tang，arXiv `2510.27396`：无导数 trust-region 与
  consensus/ADMM；
- De Falco et al.，DOI `10.1016/j.ins.2019.01.009`：surrogate-assisted CC，
  作为后续 FE 优化方向，不进入第一版实现。

本文不宣称非凸黑盒场景下的 ADMM 收敛。创新点是将共识、局部半径和
持久状态压缩成固定 FE、可审计、可恢复的 CTP/GSS shared-patch 协议。

## 3. 总体架构

```text
Phase-I（冻结）
    soft-RDDSM v3 -> shared variables + owner groups + checkpoint
                         |
                         v
原 selector（冻结） -> 选择 CTP/GSS/SMP/AOR
                         |
                         v
                 CTP/GSS patch kernel
                   - v2 owner candidate
                   - consensus candidate
                   - disagreement candidates
                   - z/u/r 持久状态
                   - 局部 context reset
                         |
                         v
                 strict-best ledger 写回
```

CTP/GSS 的 full-space continuation 不调用 patch；SMP/AOR 完全保持原实现。

## 4. Patch Kernel 设计

### 4.1 Proposal 输入

所有统计量来自已有 SMP sense 的 `LocalProposal`，不新增探测 FE：

```text
p_gj       owner g 对共享变量 j 的 proposal value
sigma_gj   owner g 对 j 的 uncertainty
I_g        owner g 的 proposal improvement
```

owner 权重为：

```text
w_g = max(I_g, 0)
```

当所有 `w_g=0` 时，使用均匀 owner 权重。

共识值：

```text
consensus_j = Σ_g w_g p_gj / Σ_g w_g
```

owner 分歧：

```text
disagreement_j = max_g |p_gj - consensus_j|
```

归一化分歧：

```text
normalized_disagreement_j =
    disagreement_j /
    max(variable_range_j, max_g sigma_gj, eps)
```

### 4.2 Candidate families

每个 patch lane 使用以下候选族：

1. v2 owner-conditioned candidate；
2. consensus candidate；
3. disagreement-positive candidate；
4. disagreement-negative candidate。

disagreement 方向只来自当前 proposal 的 owner 差异：

```text
d_j = p_top_owner,j - p_second_owner,j
```

禁止从 `u_j` 生成或修正候选方向。

### 4.3 持久状态

每个共享变量维护：

```text
z_j              patch 中心
u_j              有界不一致积分器
r_j              当前 patch 半径
base_radius_j    初始半径
context_hash_j   局部上下文哈希
last_update      最近 scope 访问周期
reset_count      context reset 次数
```

初始中心：

```text
z_j = 当前 strict-best incumbent[j]
```

### 4.4 Base radius 与 conforming 静默

为避免 conforming 变量因 proposal uncertainty 非零而继续被 patch，采用：

```text
if disagreement_j <= 1e-12 * variable_range_j:
    base_radius_j = eps
else:
    base_radius_j = clip(
        max(disagreement_j, max_g sigma_gj),
        eps,
        variable_range_j
    )
```

因此：

- conforming 或近似 conforming 变量的 patch 自动静默；
- conflicting 变量的半径由 proposal 分歧和 uncertainty 决定；
- 不需要运行时一致性分类器；
- 活跃 patch 域由 owner 分歧自动划定。

这意味着 conforming case 只要求非劣，不要求额外终值改善；主要性能
收益预期放在高 disagreement 的 conflicting case。

### 4.5 Radius update

成功条件：

```text
gain >= max(1e-12, 1e-6 * max(abs(error_before), 1))
```

成功时：

```text
z_j <- accepted_candidate[j]
r_j <- min(4 * base_radius_j, 1.25 * r_j)
```

失败或无收益时：

```text
z_j 保持当前 incumbent
r_j <- max(eps, 0.5 * r_j)
```

## 5. 局部 Context Hash 与 Reset

### 5.1 不使用 full incumbent

不能把完整 incumbent 放进 `context_hash`。否则 patch 自己被接受会改变
incumbent，并在下一次访问时触发 reset，使半径无法累积。

定义当前 patch 的写集：

```text
write_set =
    当前 shared scope
    + CTP/GSS 本轮会修改的 boundary/private coordinates
```

对变量 `j` 定义：

```text
context_coordinates_j = component 相关坐标 - write_set
```

局部上下文哈希为：

```text
context_hash_j = SHA256(
    checkpoint_hash,
    selector_input_hash,
    selector_output_hash,
    component_id,
    incumbent[context_coordinates_j]
)
```

因此：

- 本 patch 修改的 scope 不参与 hash；
- 本 patch 被接受不会触发 reset；
- 外部 component 修改了本 component 的上下文时触发 reset；
- 同 component 中未被本轮写入的上下文变化时触发 reset；
- selector 输入或输出变化时触发 reset。

### 5.2 Reset 规则

发生局部 hash 变化时：

```text
z_j <- 当前 incumbent[j]
r_j <- base_radius_j
u_j <- 0.25 * u_j
reset_count <- reset_count + 1
```

receipt 记录：

```text
context_reset_reason ∈ {
    none,
    external_context_change,
    scope_change,
    checkpoint_change,
    restore
}
```

不采用“自致豁免”作为主规则，因为局部 context hash 已从定义上排除
本 patch 写集，审计更直接。

## 6. `u_j` 的更新与作用范围

### 6.1 更新时机

每次 scope 被访问时，每个 scope 变量只更新一次：

1. 读取当前 proposal；
2. 检查局部 context hash；
3. 如有变化，先执行 reset；
4. 更新 `u_j`；
5. 生成候选；
6. 执行 patch lane；
7. 根据 strict-best 更新 `z_j` 和 `r_j`。

更新公式：

```text
u_j <- min(4.0, 0.80 * u_j + normalized_disagreement_j)
```

即使 patch 预算不足，`u_j` 仍然更新；但此时不更新 `z_j` 或 `r_j`。

### 6.2 唯一因果通道

`u_j` 唯一可以影响下一次 scope 排序：

```text
(-u_j, -proposal_priority_j, j)
```

明确禁止 `u_j`：

- 生成候选方向；
- 修改 patch 半径；
- 改变 CTP/GSS/SMP/AOR 动作选择；
- 改变 Phase-I selector；
- 关闭 component；
- 从 sense lane 借预算。

## 7. 固定 FE 契约

每次 CTP/GSS patch scope 访问预留：

```text
k_patch = 8 FE
```

固定为四轮双候选：

```text
8 FE = 4 rounds × 2 candidates
```

`best_error_before` 和 `incumbent_before` 直接从 `EvaluationLedger` 读取：

```text
best_error_before = ledger.best_error
incumbent_before = ledger.best_x
```

不得重新评价 incumbent。

预算规则：

- patch lane 从 CTP/GSS operator reservation 中划出；
- 不从 sense、probe、tail 或其他 component 借预算；
- 预留不足 8 FE 时产生显式 `patch_budget_unavailable`；
- 可以执行原 v2 CTP/GSS 行为，但 receipt 必须说明 patch 未执行；
- 不允许静默减少候选或静默 fallback。

## 8. 代码接口与状态契约

### 8.1 代码挂载点

主要修改位置：

- `src/arac/coordination/overlap.py`：实现 `SharedPatchKernel`、候选生成、
  radius update 和局部 context hash；
- `src/arac/coordination/state.py`：增加 shared-patch 状态并升级 schema；
- `src/arac/coordination/contract.py`：增加 patch receipt 字段；
- `src/arac/coordination/loop.py`：在已选择的 CTP/GSS operator 内挂载 patch；
- `src/arac/actions/phase2_v2.py`：CTP/GSS block-sweep 阶段接入持久状态。

full-space continuation 不调用 patch；SMP/AOR 不修改。

### 8.2 Kernel 接口

```text
apply(
    component,
    proposals,
    scope,
    context_hash,
    budget_fes=8,
    seed,
    mode
) -> SharedPatchResult
```

`SharedPatchResult` 至少包含：

```text
component
scope
candidate_trace
consumed_fes
best_error_before
best_error_after
accepted_candidate
context_reset
context_hash_before
context_hash_after
state_hash
radius_trace
u_trace
budget_status
```

### 8.3 State schema

状态 schema 升级为：

```text
arac-oc-coordinator-state-v2
```

`CoordinatorState.payload()` 增加排序稳定的 `shared_patch` 字段。

v1 snapshot 恢复到 v2 时：

- 缺少 patch 状态则初始化为空；
- qhat、EMA、stall、cooldown 等旧字段保持；
- checkpoint hash、structure hash、config hash 不匹配时 fail-closed。

### 8.4 Receipt schema

operator receipt 升级为 v2，增加：

```text
patch_enabled
patch_lane_fes
patch_budget_status
patch_candidate_names
patch_accepted_candidate
patch_context_hash_before
patch_context_hash_after
patch_context_reset
context_reset_reason
patch_state_hash
patch_reset_count
patch_radius_min/max
patch_u_min/max
```

原有 `plan_hash`、`state_hash`、`reserved_fes`、`actual_fes`、
`best_error_before/after`、strict-best、exact-FE 和 fail-closed 语义不变。

## 9. 消融矩阵

所有消融共享：

- 相同 Phase-I checkpoint；
- 相同 proposal；
- 相同 selector 输入输出；
- 相同 seed；
- 相同总预算；
- 相同 CTP/GSS operator reservation；
- 相同 strict-best ledger。

| 臂 | 候选 | 状态 | 半径 | 归因 |
|---|---|---|---|---|
| A0 | 原始无 patch | 无 | 不适用 | 原始方法基线 |
| A1 | v2 owner-conditioned | 无 | v2 固定 | v2 基线 |
| A2 | v2 + consensus/disagreement | 无 | 固定 | 新候选增量 |
| A3 | v2 + consensus/disagreement | `z,u,r` | 固定 | 持久状态增量 |
| A4 | v2 + consensus/disagreement | `z,u,r` | 自适应 | 自适应半径增量 |

归因关系：

```text
A1 - A0：v2 候选增量
A2 - A1：consensus/disagreement 候选增量
A3 - A2：持久状态增量
A4 - A3：自适应半径增量
A4 - A1：相对当前生产 v2 的总增量
```

A2 是强制消融，不能用“有 patch/无 patch”替代。

## 10. Gate 验收协议

### Gate P0：Kernel Contract

范围：toy overlap、CTP/GSS 单动作、snapshot/restore、context reset、
fixed FE lane。

必须通过：

- 候选始终在变量边界内；
- exact FE；
- strict-best 单调；
- incumbent 不重复计费；
- `u_j` 不进入候选方向；
- 局部 context hash 只检测局部上下文；
- patch 自接受不触发 reset；
- 外部上下文变化触发 reset；
- 成功半径扩大、失败半径缩小；
- v1 snapshot 可恢复为 v2；
- 预算不足状态显式；
- operator 异常 fail-closed；
- receipt/state hash 可重复。

P0 失败则停止后续实验。

### Legacy Gate P1：小型归因（已退休，仅作历史记录）

使用四个代表性 case：

```text
R2 / A3 / S5 / R6
```

使用三枚 replay seed：

```text
20260880 / 20260881 / 20260882
```

运行 A0-A4 全部消融。

通过条件：

- Phase-I checkpoint hash 完全一致；
- proposal hash 完全一致；
- selector input/output hash 完全一致；
- 五臂总 FE 和 FE 分类可审计；
- A2 与 A1 的候选 trace 有明确差异；
- A3 与 A2 的 state trace 有明确差异；
- A4 与 A3 的 radius trace 有明确差异；
- A4 相比 A2 的 median anytime AUC 不得劣化超过 5%；
- 所有 receipt/state hash 审计通过。

该旧 P1 已被 2026-08-22 的实质判定覆盖：其形式 checks 通过但实际无 patch
挂载点，因此不能证明机制接入。当前等价的可达性与归因由 matched-host
M0/M1 承担。

### Legacy Gate P2：四 case fresh-seed 筛查（已退休）

使用四枚全新 seed：

```text
20260890 / 20260891 / 20260892 / 20260893
```

运行：

```text
R2 / A3 / S5 / R6 × A0-A4
```

P2 是筛查门，不作最终性能优越性判断。

必须通过：

- exact-FE、strict-best、receipt、state-hash chain；
- A4 相对 A1/A2 无预注册安全性回归；
- conforming/低 disagreement case 中 A4 相对 A1 非劣；
- conflicting/高 disagreement case 记录方向性结果；
- 能观察到真实 state、radius、context reset 事件；
- 不出现 `u_j` 越界或 patch FE 泄漏。

不再使用“至少两个 case 严格改善”作为 P2 条件。所有改善结果标记为
exploratory，正式主张留给 P4。

### Legacy Gate P3：Selector Boundary Parity（已退休）

P3 只验证相同 selector 输入是否产生相同 selector 输出。

若：

```text
selector_input_hash 相同
```

则必须满足：

```text
selector_output_hash 相同
selected_action 相同
```

如果 patch 改变了后续动态 selector 的合法输入，则允许输出不同，但必须记录：

```text
selector_input_changed = true
selector_input_hash_before
selector_input_hash_after
```

如果当前 selector 只有一次基于 Phase-I evidence 的选择，则只比较这一处。

P3 失败仅指“相同 selector 输入产生不同 selector 输出”，不能因为 patch
合法改变运行状态，就强制后续轨迹逐位一致。

### Legacy Gate P4：24 × 25 确认实验（已退休）

复用已有 24-case 矩阵，使用 25 枚新 seed：

```text
20261001 ... 20261025
```

主比较：

```text
A4 vs A1
A4 vs A2
```

按 benchmark 构造标签分层：

- conforming：只要求非劣；
- conflicting：要求体现 patch 增量。

不使用运行时 consistency classifier 生成标签。

建议确认标准：

总体：

```text
A4/A1 geometric mean final-error ratio <= 0.98
A4/A1 win-or-tie >= 0.60
median anytime AUC 改善 >= 2%
```

conforming strata：

```text
A4/A1 ratio <= 1.05
```

conflicting strata：

```text
A4/A1 ratio <= 0.98
win-or-tie >= 0.60
```

所有主结论使用 paired bootstrap 95% CI 或预注册 permutation test，并进行
多重比较校正。

## 11. 统计指标与实验记录

每个 cell 记录：

- final error；
- `log10(error)` anytime AUC；
- 600k/1M/2M/3M checkpoints；
- win/tie/loss；
- immediate gain；
- handoff gain；
- patch acceptance rate；
- context reset count；
- radius expansion/shrink count；
- `u_j` 最大值和均值；
- sense/operator/probe/tail FE；
- patch budget unavailable 次数；
- strict-best violation；
- receipt/state hash violation。

统计纪律：

- common-random-number 配对；
- paired bootstrap 10,000 次；
- paired permutation 10,000 次；
- seed、case、参数和判据运行前冻结；
- 禁止根据结果调参；
- seed registry 冲突时预检失败，不能静默替换。

## 12. 失败语义与回退

### 12.0 当前恢复优先止损线

- **B1 不通过**：停止创新实验，继续隔离未恢复的 AOR/SMP/CTP/GCB 动作。
- **B2 不通过**：冻结 selector，不接入任何新调度逻辑。
- **B3 不通过**：保留 fixed-action 结果，逐 case 修复 handoff，不启动 M0。
- **M0 不通过**：实验设计不可评估，禁止性能比较。
- **M1 不通过**：保留历史基线，放弃当前 patch 版本。
- **M2/软路由不通过**：回退到无软路由的 matched-host kernel。
- **AOB preservation 不通过**：patch 默认关闭。
- **生产 E2E 不通过**：只保留 matched-host 机制结论，不宣称生产 superiority。

### P0 失败

停止所有后续实验，优先修复 contract、state 或 FE 账本。

### P1 失败

检查候选、状态、context hash 和 FE 归因；不得直接进入 P2。

### P2 失败

关闭 patch，生产回退 A1/v2；保留失败 artifacts。

### P3 失败

若相同 selector 输入产生不同 selector 输出，禁止生产接入。

### P4 失败

保留 A1/v2 作为生产方法；论文如实报告 patch 未证明 superiority。

任何阶段都禁止：

- 从 sense lane 借 FE；
- 静默减少候选；
- 用 `u_j` 直接 steering；
- 改动 Phase-I；
- 改动外层 selector；
- 改动 SMP/AOR；
- 通过 benchmark 特判修复结果。

## 13. 文档与论文收尾

需要更新：

- `docs/arac-oc-completion-plan.md`：标记旧的一致性分类方案 superseded；
- `docs/arac-oc-dual-overlap-upgrade.md`：改写为 shared-patch 主线；
- `docs/arac-oc-operator-contract.md`：增加 patch lane、state v2、receipt v2；
- 新增 P0-P4 gate protocol 和实验配置；
- 更新 `docs/arac-oc-paper-materials.md` 与论文草稿。

论文主张顺序：

1. Phase-I 黑盒重叠发现；
2. 证据驱动 selector；
3. CTP/GSS 内部 shared-patch kernel；
4. v2、候选、状态、半径的嵌套消融；
5. fresh-seed 和 24×25 结果；
6. conforming 自动静默；
7. conflicting 场景中的局部 trust-region 增量；
8. consistency classifier 不可辨识、SMP/AOR 未改、selector 未改等边界。

必须保留：

```text
references/lit-review/phase2_shared_var_new_directions.md
```

## 14. 默认参数冻结

```text
k_patch                  = 8 FE
radius_success_scale     = 1.25
radius_failure_scale     = 0.50
radius_upper_bound       = 4 * base_radius
u_decay                  = 0.80
u_context_decay          = 0.25
u_upper_bound             = 4.0
min_relative_gain        = 1e-6
min_absolute_gain        = 1e-12
conforming_threshold      = 1e-12 * variable_range
patch_mount_points        = CTP/GSS only
outer_selector            = unchanged
Phase-I                   = unchanged
SMP/AOR                   = unchanged
```

最终生产决策：

```text
B0-B3、M0-M1、AOB preservation 全部通过 -> 才可申请 production E2E
M2/soft-routing 通过 -> 才可分别开启对应机制
任一恢复、可达性、归因或 parity gate 未通过 -> 保留历史基线，patch 默认关闭
```
