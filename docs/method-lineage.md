# ARAC 方法边界与创新归属

日期：2026-08-09  
执行者：Codex

## 当前主线

ARAC-Core 的核心是“Phase-I 身份盲证据 -> 一次结构动作选择 -> 一个 Phase-II 动作”。
当前主方法不训练 RF，不在运行时试跑四个动作，也不使用 benchmark 身份。

方法贡献边界：

1. Phase-I 通过计数黑盒评价构造 incumbent、变量块、关系图和进度证据。
2. 结构完成度与关系连通性确定 `AOR`、`SMP`、`CTP`、`GCB` 中的一个动作。
3. 被选动作从同一 checkpoint 和 FE 边界运行到终点；未选动作不执行。
4. checkpoint、选择原因、动作结果和 FE 由公共契约绑定，可逐条审计。

## 不宣称原创的部分

- AOB、IOH/BBOB 只是 benchmark。
- CMA-ES、MMES、Sep-CMA-ES 和 PyPop7 不是 ARAC 原创。
- FE 账本、哈希收据和并行调度属于可复现工程，不是算法创新。
- 在线 algorithm portfolio、warm-start switching 和 racing 已有先行工作。

## 历史路线

- RF v3：在同一 AOB 函数集合的新 seed 上完成 600-run proof-of-concept。
- RF v4/v5/v6：严格 holdout、tail-risk 或跨 family 验证失败，未形成可发布选择器。
- Phase-II v2：common-anchor、resume 和 FE accounting 协议可执行，但 probe policy 未通过
  晋级门，并出现数值稳定性失败。

这些实现和冻结结果保留用于审计，不作为当前方法的正面证据。

## 未来工作

当 ARAC-Core 的固定动作对照和跨 suite 验证完成后，才重新研究 trajectory forecasting、
common-anchor probe、delayed commitment、survivor racing 和 risk-aware online allocation。

当前方法规范见 `docs/arac-core-method.md`。

