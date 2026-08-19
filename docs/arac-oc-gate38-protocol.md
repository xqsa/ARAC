# Gate 38 协议：归一化拓扑信号下的 GCB 派发协调器

日期：2026-08-15
状态：已执行完毕。判定：**gate_passed = true——8 项协议检查与 3 项筛查
判据全部通过**。这是"自适应协调不劣于固定动作"的第一个完整闭环证据。
结果见 §6。

## 1. 研究问题

Gate 37 的归因证明：绝对 hub degree 在 Phase-I 推断结构上恒为 10–20，
坐标修复分支从未触发，chain/overlap=3 的 +3.5727 收益被还给基线。本 gate
检验：

> 将拓扑复杂度信号换成推断结构统计上校准的相对 hub degree 后，预注册
> 派发能否同时满足 Gate 37 失败的两条判据（chain/ov=3 保正收益、全 cell
> 不劣于固定 CTP），并保持已通过的全部协议检查？

## 2. 离线校准（已完成，`artifacts/overlap_topology_calibration_gate38/`）

3 个新 seed（20260815/16/17，排除 gate seed 20260829）× 12 个拓扑组合
= 36 个 Phase-I 推断结构，只跑 Phase-I（180k FE/cell），不跑优化。

相对 hub degree = hub / (组件组数 − 1)：

| 真实拓扑 | rel_hub 范围（12 样本） |
|---|---|
| star | 1.000（12/12 精确饱和） |
| chain | 0.706–0.824（12/12 < 0.9） |
| random | 0.765–1.000（8 个饱和、4 个 < 0.9） |

附加发现（写入归因记录）：

- 推断结构共享变量 owner 数 6–11（真实为 2），重叠对数 59–126；
- Phase-I 不输出真实成员置信度（适配器全部默认 1.0），置信度过滤方案
  在当前证据下不可用；
- gate seed 上两个决定性 cell 的诊断值与校准分布一致（chain/ov=3：
  0.700；star/ov=6：1.000）。

冻结决定：`hub_mode="relative"`，`complex_hub_ratio=0.9`（阈值位于
chain 上确界 0.824 与 star 下确界 1.000 的间隙内）。random 的混合分类
被接受为已声明边界：Gate 36/37 中 random cell 在两个算子下收益均为 0，
分类对判据中性。

## 3. 派发规则（v2，其余同 Gate 37）

| 条件 | 动作 |
|---|---|
| streak < 2 | neighborhood |
| streak ≥ 2 且 rel_hub < 0.9 | `sequential_coordinate_patch` |
| streak ≥ 2 且 rel_hub ≥ 0.9 | `sequential_joint_patch` |
| streak ≥ 6 且未升级过 | `joint_cmaes`（每组件每运行一次） |
| cooldown / stall cap | 同 Gate 37 |

预算语义、收据、信封纪律与回归测试不变（`tests/test_gcb_coordinated_mode.py`）。

## 4. 判定（与 Gate 37 完全相同）

协议检查 8 项 + 筛查 3 项（star/ov=6 无实质回归、chain/ov=3 保正收益、
全 cell 不劣于 Gate 36 固定 CTP，容差 1e-9）。对照来源同 Gate 37
（Gate 29 基线、Gate 36 固定 CTP，均只读）。

失败处置同前：按收据归因，不回调阈值重跑。

## 5. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.overlap_gcb_coordinator_gate38_screening --workers 4
```

产出：`artifacts/overlap_gcb_coordinator_gate38/confirmation_fresh.json`，
收据含每次派发的 absolute 与 relative hub 两个信号值。

## 6. 结果（2026-08-15 执行）

gate_passed = true。协议检查 8/8、筛查判据 3/3 全部通过：

- `chain_ov3_positive_gain`：+3.5727——坐标修复分支首次触发
  （3 次派发、3 次产生收益），数值上精确复现 Gate 36 固定 CTP 在该
  cell 的收益；
- `star_ov6_no_material_regression`：−1.1e-13（数值平局），相对固定
  CTP 改善 +17.8712；
- `not_worse_than_fixed_ctp_all_cells`：最小 −1.1e-13（平局）；
- vs proposal 基线 12/12 win-or-tie，中位数 gain 0.0。

派发分类与校准预测一致：全部 chain cell（rel_hub 0.70–0.71）走
`coordinate_ctp`；star 与 random 饱和 cell（rel_hub 1.00）走
`joint_ctp`/升级；conforming cell 派发极少且无收益无损害。总派发
25 次 / 800 FE / 3 次升级，与 Gate 37 相同的派发次数下把收益从 0
恢复到 +3.5727——差异只来自信号归一化后的算子选择。

科学结论：**"协调器按证据选择算子"首次同时做到固定动作的优点
（chain 收益）而不承担其缺点（star 回归）**。此结论限于 seed
20260829 单批；多 seed 配对确认见 Gate 39
（`docs/arac-oc-gate39-protocol.md`）。
