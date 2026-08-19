# Independent action semantic parity pilot

状态：冻结设计，尚未运行。

这个 pilot 只回答一个问题：当前独立 ARAC 动作的生命周期是否仍然实现了
历史动作 JSON 中声明的机制。冻结 v3 源码只作为迁移差异参照。pilot 不比较四个
动作谁最终更好，也不训练或调用 selector。

## 公平边界

- 四条 lane 都从已有 Phase-I checkpoint 开始；checkpoint 只作为审计元数据定位，
  不传入 runtime 的函数身份或 family 标签。
- 每条 lane 的两臂使用相同 checkpoint、相同 action seed、相同 `120000 FE` 屏幕预算、
  `native_threads=1`。
- 允许四个 lane 并行，但每个 lane 内部串行，所有 FE 必须计入 ledger。
- 输出只能写入新的 `artifacts/independent_action_semantic_parity_pilot_v1`，不得覆盖
  历史结果或 frozen source。

## 两个 arm

1. `current_production`：当前 `src/arac/actions` 的独立实现。
2. `historical_semantic_port`：只把历史 action contract 的调度语义移植到 ARAC
   独立 `ActionContext`/`EvaluationLedger`/pypop7 端口中；不得复制 HCC 源码。

## 通过条件

- 两臂 checkpoint hash 完全一致；
- 每条 arm 精确消耗 `120000 FE`，没有隐藏探测或 fallback；
- 收据记录动作事件序列、块会话数量、覆盖/协调/精修预算和 optimizer package 版本；
- production source 扫描没有 HCC runtime import；
- 不产生终值优劣或 selector 正确率结论。

四条 lane 的机制屏幕全部通过后，才可以另行冻结 terminal parity 实验。即便
terminal parity 通过，也不能自动授权 selector；selector 仍需单独的共同 checkpoint
四动作对照和泛化验证。
