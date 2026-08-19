# Gate 48b：同 checkpoint 配对 pilot

日期：2026-08-16

## 目的

Gate48a 发现 AOB 上 sense 占约 93.7% 的 Phase-II 预算、operator FE 为零，
且与 Gate41b 不是同 checkpoint 对照。Gate48b 只回答预算修复后统一循环是否
真正获得 operator 执行机会，以及在同一 Phase-I 边界下相对终端 MMES 对照的
变化。

## 固定协议

- cases：`R2/A3/S5/R6`；seed：`20260845`；Phase-I：精确 `180,000 FE`；
  terminal：精确 `3,000,000 FE`；
- Phase-I：Gate48a 同一 v3 soft-RDDSM 配置；每个 case 只生成一次 checkpoint；
- sense 总预算上限：Phase-II 的 `45%`，通过通用
  `capped_proposal_budget` 转成每组每周期预算；
- unified 臂：当前 ARAC-OC config v3，16 周期，strict-best；
- control 臂：从同一 checkpoint 使用完整剩余 `2,820,000 FE` 的 MMES；
- Gate41b 和历史均值只做报告，不参与本 pilot 判定。

## 判定

必须检查 Phase-I/terminal FE、同 checkpoint hash、strict-best、receipt FE
平价和 state hash。`operator_fired_all` 是机制检查：若失败，pilot 失败且
不得宣称循环收益；若通过，报告每个 case 的 operator FE、预算流向和
`control_final - unified_final`。

运行：

```powershell
.venv\Scripts\python.exe -m experiments.oc_aob_paired_gate48b --workers 4
```

产出：`artifacts/oc_aob_paired_gate48b/`。

## 结果（2026-08-16）

六项检查全部通过，且四个 case 都有真实 operator 执行。统一循环相对同
checkpoint、完整 Phase-II MMES 对照：R2/R6 胜，A3/S5 负；四个 case 的
operator 均为 `8 FE`，表明预算上限解决了“operator=0”，但当前残差/停滞
反馈只允许一次最小 pulse。该 pilot 不能替代 25-seed 正式性能门。
