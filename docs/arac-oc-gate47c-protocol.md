# Gate 47c 协议：预算生命周期修复后的 fresh-seed 对照

日期：2026-08-16  
执行者：Codex  
状态：执行中

## 目的

验证 Gate47b 暴露的 dispatch 生命周期错误已经修复，而不是只在
`rastrigin/star/ov3/seed=20260835` 单元上恢复结果。

Gate47c 完全继承 Gate47b 的 12 个 gate40 cell 和 6 个 rastrigin cell，唯一变化是：

- 使用修复后的 `arac-oc-unified` 循环；
- 层 2 使用 fresh seed `20260837`；
- 输出写入 `artifacts/oc_streak_confirmation_gate47c/`；
- Gate47b 的失败 artifacts 不被覆盖。

## 修复假设

1. `arbitration_only` 不产生 operator feedback，不增加 stall/cooldown，不更新 qhat/pulse。
2. `stall_cap` 和 cooldown 只禁止新的 operator plan；所有 component 的 owner-local
   `SMP.sense` 继续消耗预留预算。
3. 每个可负担 cycle 保留 32-FE/component 的 proposal-neighborhood writeback 基线
   lane；operator pool 在此预算之后计算。
4. 其余 residual、relative hub、operator contract、strict-best 和 terminal policy 不变。

## 判定

沿用 Gate47b 的 6 项协议检查、`path_fires_on_chain`、
`not_worse_than_kernel_all` 和 `star_no_regression_vs_proposal`。

额外报告每个 cell 的 `sense_fes`、`operator_fes`、`tail_fes`，确认预算没有再次
从 owner-local proposal/writeback lane 大规模转入 MMES tail。

```powershell
.venv\Scripts\python.exe -m experiments.oc_streak_confirmation_gate47c --workers 16
```
