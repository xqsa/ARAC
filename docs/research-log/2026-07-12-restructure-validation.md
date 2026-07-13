# ARAC 重构验证记录

- 日期：2026-07-12
- 执行者：Codex
- 分支：`codex/research-project-structure`
- 验证基线提交：`77915c2`（HCC runtime 职责拆分）
- 目标：完成 Task 9 的可复现性、runtime 边界和 Git 清洁门

## 验证命令

以下命令均在
`C:\Users\83718\.config\superpowers\worktrees\ARAC\codex-research-project-structure`
执行：

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest -q
E:\ARAC\.venv\Scripts\python.exe scripts/audit_project_structure.py --root .
E:\ARAC\.venv\Scripts\python.exe scripts/build_results_manifest.py --root . --results E:\ARAC\results --output .codex/tmp/results-manifest.csv
git diff --check
git status --short --branch
```

## 结果

- pytest：`437 passed, 1 skipped`
- structure audit：通过；`results/` 被识别为 generated，payload scan 按规则跳过
- HCC 模块编译：通过
- HCC 相关聚焦回归：`64 passed`
- results manifest：`3152` 条记录，`status=partial` 为 `3152` 条
- manifest SHA256：`7FAD974CDD9AAEA4EE0F8D3AA0BFA81E48C7811460C42CFA16E74E1E3BEB2ADE`
- `git diff --check`：通过

## 边界确认

- `hcc.py` 继续拥有 subprocess 执行编排，因此现有对模块级 parser/trace 名字的
  monkeypatch 语义保持不变。
- `hcc_budget.py` 和 `hcc_trace.py` 不导入 subprocess；共享写回只接收显式
  `ActionDecision` 和 `optimizer_consumed`，不读取 paper、历史结果或 case label。
- 论文 reported values、历史 final/pilot outcome、relative gain 和 problem-family
  信息仍属于 offline evaluation，不进入 runtime dispatch。
- 当前验证没有重跑 25-run AOB final protocol，也没有产生 final performance claim。
  旧 results 因 metadata 不完整全部保持 `partial`，不能被解释为 complete。

## Git 交接

本验证记录随后续文档提交进入 Git；未暂存 `results/`、`.venv/`、缓存、日志或
`.codex/tmp/` manifest。远端 push 不在本次操作范围内。
