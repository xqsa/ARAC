# ARAC 重构前后 3M-FE 等价性验证

- 日期：2026-07-12
- 执行者：Codex
- Before：`b88a4d9`，detached worktree `refactor-equivalence-before`
- After：`cd69d90`，detached worktree `refactor-equivalence-after`
- 协议：24 个 AOB case、`seed=1`、每 case `3,000,000 FE`、strict accounting
- lane：`canonical_evidence_controller_v1`
- 并发：两侧同时运行，各 `jobs=12`

## Preflight

- before 聚焦 HCC 测试：`43 passed`
- after 聚焦 HCC 测试：`49 passed`
- AOB datafile：两侧各 60 个文件，SHA256 全部一致
- 环境：Python 3.12.13、NumPy 2.3.5、SciPy 1.18.0、Torch 2.12.1+cpu、
  matplotlib 3.11.0、PyYAML 6.0.3
- 线程控制：`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、
  `MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`

## 结果

原始结果目录：
`E:\ARAC\results\refactor_3m_equivalence_20260712`

逐 case 对照：[paired_case_comparison.csv](E:/ARAC/results/refactor_3m_equivalence_20260712/paired_case_comparison.csv)

汇总：[equivalence_summary.md](E:/ARAC/results/refactor_3m_equivalence_20260712/equivalence_summary.md)

| 检查项 | 结果 |
|---|---:|
| before completed rows | 24/24 |
| after completed rows | 24/24 |
| evaluation records | 48 |
| budget summaries | 48 |
| same-budget violations | 0/48 |
| exact_equal | 24/24 |
| numeric difference | 0/24 |
| action trace hash mismatch | 0/24 |
| budget field mismatch | 0/24 |

24 个 case 的 final error、FE、状态、动作选择、action trace hash 和预算字段均逐项一致。

## 退出码说明

两侧 optimizer 轨迹和 FE 账本均已完整写出，但两个 `exp_005` wrapper 最后都返回
退出码 `1`：离线 `paper_best` 收尾矩阵只有原来的 13 个 case，处理到 `A1` 时找不到
阈值并抛出 `KeyError`。该错误发生在 24 条 optimizer trajectory 完成之后，不影响本次
reference-blind 等价性对照，但说明当前 final wrapper 还不能直接作为 24-case 成功入口。

本次比较没有读取论文值、历史最优值或 prior outcome；也没有覆盖原有 results。该结果证明
的是重构前后行为等价，不是性能提升或论文胜场声明。
