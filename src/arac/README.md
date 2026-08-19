# ARAC 源码边界

当前生产方法是 `ARAC-Core`，源码职责如下：

- `core.py`：根据 Phase-I checkpoint 选择一个动作，并提供完整 `run_arac` 入口。
- `benchmarks/`：把测试函数转换为统一数值问题；benchmark 身份不进入 checkpoint。
- `evidence/`：运行身份盲 Phase-I 协议并生成不可变 checkpoint。
- `actions/`：实现 `CTP`、`SMP`、`GCB`、`AOR` 四个同接口动作。
- `runtime/`：统一 FE 账本、checkpoint/action 契约和优化器端口。
- `analysis/`：保留历史 RF、兼容 baseline 和未来工作的轨迹分析工具。

正式执行顺序为：

```text
OptimizationProblem
  -> EvaluationLedger
  -> run_phase1(...)
  -> select_core_action(checkpoint)
  -> ActionRegistry.execute(selected_context)
  -> exact terminal FE
```

`PhaseCheckpoint` 不包含函数 ID、family 或 benchmark 名称。`run_arac` 只调用一次
`ActionRegistry.execute`；Phase-II v2 的分支 probe 和 delayed commit 不在当前生产链路中。

