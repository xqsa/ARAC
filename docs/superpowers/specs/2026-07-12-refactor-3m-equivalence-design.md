# ARAC 重构前后 3M-FE 等价性验证设计

## 目标

验证 canonical v3.2 `b88a4d9` 与重构后 `cd69d90` 在完整 24 个 AOB case、
同一 `seed=1`、同一 `3,000,000` FE 下是否产生一致的数值结果和 FE 账本。

## 固定协议

- cases：`E1-E6 S1-S6 R1-R6 A1-A6`
- seed：`1`
- lane profile：`canonical_evidence_controller_v1`
- runtime action：`arac_evidence_action_controller_v32`
- budget accounting：`strict`
- CMA-ES/MMES restart：保持入口默认开启
- Python：`E:\ARAC\.venv\Scripts\python.exe`
- 环境线程：`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、
  `MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`
- 并发：before/after 各 `jobs=12`，同时运行

两个被测版本分别位于 exact-commit detached worktree。AOB 输入文件在运行前按相对
路径和 SHA256 对齐；输出写入 `E:\ARAC\results\refactor_3m_equivalence_20260712`
下独立的 `before_b88a4d9` 与 `after_cd69d90` 目录。

## 对照字段

每个 case 输出：before/after final error、绝对差、相对差、FE used、optimizer final
FE、状态、same-budget violation、action trace 行数和 trace SHA256。最终分为：

1. `exact_equal`：记录值、FE、状态和 trace hash 全部相同；
2. `numeric_equal`：final error 在 `abs_tol=1e-12, rel_tol=1e-12` 内，且 FE/状态一致；
3. `different`：超出容差，或 FE/状态不一致；
4. `incomplete`：任一侧缺失、失败或没有完整 3M-FE 账本。

运行时间不参与一致性判定。该实验只验证重构等价性，不与论文值或历史最优比较，
也不构成性能声明。

## 成功标准

- 48 条真实 optimizer trajectory 全部完成；
- 两边 AOB 输入 manifest 完全一致；
- 24/24 case 至少达到 `numeric_equal`；
- 两边均无 same-budget violation；
- 若不一致，保留原始 artifacts 并按 case 报告，不用容差掩盖 FE 或动作轨迹差异。
