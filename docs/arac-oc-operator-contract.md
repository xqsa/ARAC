# ARAC-OC 统一 Operator Contract（冻结 v1）

日期：2026-08-15
状态：**冻结**。字段级规范，实现为 `src/arac/coordination/contract.py`
（测试 `tests/test_oc_operator_contract.py`）。本契约是
`docs/arac-oc-design.md` §2/§3/§5/§6/§7/§8 的实现级冻结，不引入新语义。
修改本契约需要新版本号（v2）并重新过契约测试。

---

## 1. OperatorPlan：GCB 一次派发决策的完整载体

`GCB.make_plan()` 的唯一输出，回答协调器四问（处理哪里 / 多大范围 /
多少预算 / 哪个动作）。

| 字段 | 类型 | 语义 |
|---|---|---|
| `cycle_index` | int ≥ 0 | 协调周期序号 |
| `component` | tuple[int, ...] | 重叠组件（组索引，非空） |
| `scope` | tuple[int, ...] | 变量级作用范围，有序去重；可负担性收缩后的产物 |
| `conflict_level` | str ∈ {low, medium, high, complex} | counted B/W/C + 滞回后的冲突等级 |
| `action` | str ∈ 见 §3 | 派发动作，必须等于等级映射表的值（固定设计） |
| `reserved_fes` | int | 算子预留 FE；仅 low/仲裁级为 0 |
| `predicted_gain` | float ≥ 0 | 派发前预测收益（qhat credit 的分母输入） |
| `seed` | int ≥ 0 | 本次派发的确定性种子 |
| `reason` | str | 决策理由（如 `ema_enter_high`、`persistent_escalation`） |
| `hub_degree` | int | 绝对 hub 度（收据强制携带） |
| `relative_hub` | float ∈ [0,1] | 相对 hub 度（收据强制携带，Gate 38 §5） |

`plan_hash = canonical_sha256(payload)`，payload 注入
`schema_version: arac-oc-operator-plan-v1`。

## 2. OperatorReceipt：一次执行的审计结果

| 字段 | 类型 | 语义 |
|---|---|---|
| `plan_hash` | 64 hex | 关联计划 |
| `cycle_index / component / action / conflict_level / reason / hub_degree / relative_hub` | 同计划 | 审计冗余携带 |
| `reserved_fes / actual_fes` | int | 预留 vs 实际消耗 |
| `status` | str | `completed` / `no_gain` / `operator_failed` |
| `realized_gain` | float ≥ 0 | strict-best 前后差（账本单调，恒非负） |
| `best_error_before / after` | float | 严格最优前后 |
| `candidates` | tuple | 算子产出的候选解（审计用） |
| `state_hash` | 64 hex | 状态更新后的 CoordinatorState 哈希（每次更新必入收据，设计 §3） |
| `remaining_fes` | int | 仅 operator_failed 时有意义（设计 §2.2） |
| `exception_name` | str | 仅 operator_failed 时非空 |

校验不变量（实现于 `__post_init__`，违反即构造失败）：

- **消耗平价**：正常完成要求 `actual_fes == reserved_fes`（Gate 37 §4 纪律）。
- **状态一致**：`completed ⇔ realized_gain > 0`；零收益完成是 `no_gain`
  （正常路径，进 stall/cooldown，设计 §2.3）。
- **fail-closed**：`operator_failed` 必须携带 `exception_name`，允许
  `actual_fes < reserved_fes`（已花费 FE 留账），且必须记录 `remaining_fes`。
  调用方不得重试该动作、不得改派其他动作、不得把剩余 FE 静默交给 AOR
  （设计 §2.2，由统一循环保证）。

## 3. 动作与等级映射（固定设计，§6）

```
OC_ACTION_ARBITRATION    arbitration_only   low     仅候选仲裁（incumbent+owner+consensus+median，各 1 FE）
OC_ACTION_CTP_RESTRICTED ctp_restricted     medium  受限 CTP 局部联合修复（coordinate patch）
OC_ACTION_CTP_SHARED_CORE ctp_shared_core   high    shared-core CTP 联合优化（joint patch）
OC_ACTION_SMP            smp                high*    状态记忆重建（信任地板击穿时的替代路径）
OC_ACTION_AOR            aor                complex 预留预算内的一次 AOR 全局校正
```

映射冻结为 `OC_LEVEL_ALLOWED_ACTIONS`（每级允许集）与
`OC_LEVEL_ACTION_MAP`（每级默认动作）。`high` 级的 SMP 替代路径：
当 scope 所属 owner 的运行时信任 `qhat` 均值低于 `smp_trust_floor`
（未校准占位，默认 0.5）时派发 SMP 重建状态记忆（设计 §4 的
sense/execute 二分）。运行时按基准身份或最终结果调参是禁止的。

计数探针成本：`OC_PROBE_FES_PER_VARIABLE = 2`（f(x0) 复用 incumbent，
不重复计费，设计 §5）。

## 4. OcCoordinatorConfig：版本化常数

| 块 | 字段 | 状态 |
|---|---|---|
| 已校准（Gate 38 v2 冻结） | `persistent_streak=2, escalation_streak=6, hub_mode="relative", complex_hub_degree=3, complex_hub_ratio=0.9, stall_cap=2, cooldown_cycles=1` | 36 个新 seed 结构上离线校准 |
| **未校准占位**（`UNCALIBRATED_FIELDS`） | `ema_alpha, tau_enter/tau_exit, k_enter/k_exit, gamma_up/gamma_down, pulse_min_fes/pulse_max_fes, operator_episode_min_fes, gain_floor, k_window, probe_budget_share` | 中性默认值，仅保证可运行；任何比较性结论之前必须过预注册校准门 |

不变量：`tau_exit < tau_enter`（滞回）、`gamma_up > 1 > gamma_down > 0`、
`pulse_min ≤ pulse_max`、`escalation_streak ≥ persistent_streak`。

`operator_episode_min_fes` 是 CTP restricted 的最小有效 episode 预留。若可用
operator pool 小于该值，GCB 必须输出 arbitration-only/预算不可行收据，不能
把较小的 FE 脉冲包装成完整动作。Gate49 起比较性实验必须显式登记该值。
`config_hash = canonical_sha256(asdict)`，进入重放四元组
`(checkpoint_hash, state_hash, seed, config_hash)`（设计 §3）。

## 5. 重放与审计

- 同一 `(checkpoint_hash, state_hash, seed, config_hash)` 必须产生相同
  `plan_hash` 链与 `receipt_hash` 链。
- 每周期收据必须同时携带绝对与相对 hub 信号（Gate 38 §5）。
- 预算类别互不侵占（sense/probe/算子/邻域各自预留，设计 §7.1）；
  `probe_budget_unavailable` 在 scope 收缩为空时显式开票。
