# Gate 37 协议：最小 GCB 派发协调器对比固定 CTP

日期：2026-08-14
状态：已执行完毕。判定：gate_passed = false（3 项筛查过 1 项）；8 项协议
检查全部通过。归因见 §7。

## 1. 研究问题

Gate 36 证明了固定调用 coordinate CTP 会在 `conflicting/star/overlap=6` 上产生
实质性回归（gain = −17.8712），且回归通道是隐式预算挤占（CTP 消耗直接从
proposal-conditioned neighborhood 预算中扣减）。本 gate 检验：

> 预注册的 streak+拓扑派发协调器，能否在不损失 `conflicting/chain/overlap=3`
> 收益（+3.5727）的前提下，消除 star/overlap=6 的实质性回归，并在所有 cell
> 上不劣于固定 CTP？

## 2. 派发规则（预注册，v1 固定阈值）

每个 refresh cycle、每个 overlap component 依次执行：

1. 候选仲裁（incumbent/owner/weighted_mean/weighted_median，strict-best 接受）；
2. `GcbDispatchPlanner.plan()` 依据持久冲突 streak 与组件 hub degree 产生
   显式预留预算的派发计划；
3. 计划为 neighborhood 时，32 FE 信封全额留给 proposal-conditioned
   neighborhood 搜索；
4. 计划为算子时，算子只消费信封内显式预留的 FE，neighborhood 获得信封
   余额——不存在计划外的隐式扣减。

派发决策表（`GcbDispatchConfig` v1 默认值）：

| 条件 | 动作 |
|---|---|
| streak < 2 | neighborhood（不派发） |
| streak ≥ 2 且 hub degree ≤ 2 | `sequential_coordinate_patch`（受限坐标修复） |
| streak ≥ 2 且 hub degree ≥ 3 | `sequential_joint_patch`（共享核心+边界联合修复） |
| streak ≥ 6 且本组件未升级过 | `joint_cmaes`（组件级联合优化，每组件每运行至多一次） |
| cooldown 期内 | neighborhood（冷却） |
| 连续 2 次派发无收益 | 该组件停止派发（stall cap） |

hub degree 定义：组件内某组通过共享变量连接的不同伙伴组数量的最大值。
star 拓扑 hub degree = 3，chain/random = 2（12-cell 真实结构已验证分离）。

## 3. 与 ARAC-OC 设计契约的诚实差距

本 gate 是最小协调器，不声称实现了 `docs/arac-oc-design.md` 的完整契约：

- 冲突等级沿用 SMP proposal 残差 streak，不是独立 counted two-sided probe；
- 无 EMA、滞回、budget pulse、`qhat` 运行时信任值与 `CoordinatorState`；
- 组件处理顺序仍为固定顺序，无优先队列；
- AOR 升级以组件级 `joint_cmaes` 代替全局 AOR 动作。

## 4. 协议检查（全部通过才计入筛查）

- `cell_count_12`：12 个 cell 全部完成；
- `phase1_exact` / `checkpoint_parity` / `proposal_budget_parity` /
  `terminal_exact` / `strict_best`：与 Gate 35/36 同义；
- `dispatch_consumption_parity`：每张收据 consumed_fes == reserved_fes；
- `envelope_no_encroachment`：每个 cycle 满足
  `ctp_fes + neighborhood_fes == 32 × 组件数`（预算挤占结构性不可达的回归
  测试，对应设计契约 §7.1）。

## 5. 筛查判定（预注册）

相对 proposal_neighborhood 基线（Gate 29 冻结值）的 gain，容差 1e-9：

1. `star_ov6_no_material_regression`：`conflicting/star/overlap=6` gain
   ≥ −1e-9；
2. `chain_ov3_positive_gain`：`conflicting/chain/overlap=3` gain > 1e-9；
3. `not_worse_than_fixed_ctp_all_cells`：全部 12 个 cell 的 gain ≥
   Gate 36 固定 CTP 同 cell gain − 1e-9。

三项全部通过 = "自适应协调优于固定动作"的第一个配对证据点。任何一项失败，
按失败 cell 的收据归因（派发过少、算子选错还是升级路径未触发），不回调
阈值重跑；阈值修改只允许进入新的预注册 gate。

## 6. 运行方式

```powershell
.venv\Scripts\python.exe experiments\overlap_gcb_coordinator_gate37_screening.py --workers 4
```

产出：`artifacts/overlap_gcb_coordinator_gate37/confirmation_fresh.json`
（含每 cell 派发收据摘要、按动作分类的 FE 与收益统计）。

对照来源（只读，不重跑）：

- Gate 29 `confirmation_fresh.json`：proposal_neighborhood / full_context 基线；
- Gate 36 `confirmation_fresh.json`：固定 coordinate CTP 逐 cell 终值。

## 7. 结果与归因（2026-08-14 执行）

协议检查 8/8 通过（含 `dispatch_consumption_parity` 与
`envelope_no_encroachment`）。筛查判定：

- `star_ov6_no_material_regression`：通过（gain = −1.1e-13，数值平局；
  相对固定 CTP 改善 +17.87）；
- `chain_ov3_positive_gain`：失败（gain = +1.7e-13，平局而非正收益）；
- `not_worse_than_fixed_ctp_all_cells`：失败（最小 −3.5727，恰为
  chain/overlap=3）。

全局：vs proposal 基线 12/12 win-or-tie（最小 gain −1.3e-13），协调器在
任何 cell 都不再回归。

归因（收据证据）：全部 25 次派发均为 joint 动作（22 次 `joint_ctp`、
3 次升级），`coordinate_ctp` 分支从未触发——因为推断结构的 hub degree
为 10–20（真实结构为 2–3）。Phase-I 推断的重叠图远比真实拓扑稠密，
绝对阈值 `hub_degree >= 3` 在推断证据上恒真，预注册规则退化为
"永远派发 joint"。chain/ov=3 的 +3.57 存在于从未被执行的坐标修复
分支中。

按 §5 的失败条款，不回调本 gate 阈值重跑。修正进入下一个预注册 gate
（Gate 38）：拓扑信号必须在推断结构统计上归一化——相对 hub degree
（hub / (组件组数 − 1)）、或按 `q_jg` 成员置信度过滤/加权后再计算，
推断结构已携带逐成员置信度（`overlap_adapter.membership_confidences`）。

附带发现：即使算子选择在全部 cell 上都是"错误"的（joint 而非坐标），
stall cap + cooldown + 信封纪律仍把最坏结果限制为平局——错误派发在
该架构下的代价上界是零收益，不再是 Gate 36 式的 −17.87。
