# ARAC Refactor 3M-FE Equivalence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 24 个 AOB case 上完成 `b88a4d9` 与 `cd69d90` 的同 seed 3M-FE 配对验证并输出可审计报告。

**Architecture:** 两个 exact-commit detached worktree 只负责运行 optimizer；当前重构分支负责协议、离线对照和报告。运行时不读取论文或历史结果。

**Tech Stack:** Python 3.12、pytest、HCC/AOB、PowerShell、Git worktree、CSV。

---

## Task 1: 建立不可变被测环境

- [x] 创建 `b88a4d9` 与 `cd69d90` detached worktree。
- [x] 确认两个 worktree clean，并运行各自聚焦测试。
- [x] 比较 AOB datafile 的相对路径、文件数和 SHA256。
- [x] 记录 Python/NumPy/SciPy/Torch/OpenBLAS 版本。

## Task 2: 执行完整配对实验

- [x] 删除目标不存在的前提下创建全新的 before/after 输出目录。
- [x] 固定 seed、3M FE、strict accounting、restart 和线程环境。
- [x] 同时启动 before/after runner，各 `jobs=12`。
- [x] 持续记录进度，直到两个 runner 都退出。
- [x] 检查各 24 条 completed 状态和 same-budget ledger。

## Task 3: 逐项对照与报告

- [x] 按 `problem_id,seed` 对齐两边结果。
- [x] 比较 final error、FE、状态、trace 行数与 SHA256。
- [x] 输出 `paired_case_comparison.csv` 和 `equivalence_summary.md`。
- [x] 将命令、提交、输入哈希、环境和限制写入 `run_manifest.md`。
- [x] 运行结果审计并提交正式报告；不提交 generated results payload。
