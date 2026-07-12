# ARAC 重构基线盘点

- 日期：2026-07-12
- 执行者：Codex
- 任务：Task 1，冻结 v3.2、盘点当前 workspace，并建立干净重构 worktree

## Canonical 基线

- 当前 root：`E:\ARAC`
- 当前 branch：`main`
- canonical v3.2：`b88a4d9a68edd823b217ffa70ee06e7c10d00c73`（对象类型为 `commit`）
- 稳定 runtime 规则：**只允许 v3.2（`b88a4d9`）进入 stable runtime。** 当前 root 的未提交 v3.3 材料、现有 v3.3 worktree 改动和历史结果均不改变该规则。

`b88a4d9` 是本次重构唯一稳定算法基线。论文 reported values、历史结果、relative gain、problem family、final outcome 和 v3.3 实现不得作为 runtime dispatch 输入。

### 执行前快照

- 采集时间：`2026-07-12T15:34:34.2275330+08:00`
- root HEAD：`30d2517f062165f379be0de3361698eaf796920b`（`docs: plan research project restructuring`）
- root 相对 `origin/main`：ahead 2、behind 3
- 时间依据：第一份 workspace 清单 `.codex/tmp/tracked-files.txt` 的创建时间；此时尚未写入或提交 Task 1 盘点文件。

### Task 1 提交后快照

- 采集时间：`2026-07-12T15:38:53+08:00`
- root HEAD：`7fdb3d8bf5537cc8765b4d604d8a03bec52b2105`（`docs: record restructuring baseline and path inventory`）
- root 相对 `origin/main`：ahead 3、behind 3
- 时间依据：`git reflog --date=iso-strict` 中该提交最终 amend 的时间。

上述两个状态是不同时间点的事实，不能把执行前的 `30d2517`/ahead 2 与提交后的 `7fdb3d8`/ahead 3 混写为同一“当前状态”。

## Git 与 worktree 事实

执行前采集命令：

```text
git status --short --branch
git log -5 --oneline --decorate
git worktree list --porcelain
git branch -vv --all
git remote -v
git cat-file -t b88a4d9
```

最近提交：

```text
30d2517 docs: plan research project restructuring
16bda5c docs: define research project restructuring design
0815163 Implement auditable canonical HCC controller
795dcc2 Add targeted AOB historical diagnostics
80cece9 Design canonical runtime action controller
```

执行前已存在的 worktree：

```text
E:/ARAC                                                     30d2517  main
C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-nondense-v24
                                                               b88a4d9  codex/nondense-runtime-lock
C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-s1-backend-repro
                                                               f1e9c74  codex/s1-backend-repro
```

现有 v3.3 worktree `codex/nondense-runtime-lock` 保持原样；其状态为 `ahead 9`，有 9 个已修改 tracked 文件和 2 个未跟踪设计/计划文件。该 worktree 未被本任务修改。

Remote：

```text
origin  https://github.com/xqsa/ARAC.git (fetch)
origin  https://github.com/xqsa/ARAC.git (push)
```

## Workspace 盘点

盘点基于执行前快照 `30d2517` 和当时的未跟踪 workspace。临时明细位于 `.codex/tmp/`，不提交：

| 临时文件 | 生成时间 | SHA256 | 行数 | 记录数 |
| --- | --- | --- | ---: | ---: |
| `.codex/tmp/tracked-files.txt` | `2026-07-12T15:34:34.2311043+08:00` | `61AE31F1E812A78230CF91C1DD810323DA8A886AEE00194CBF8F6481D86791F6` | 159 | 159 |
| `.codex/tmp/untracked-files.txt` | `2026-07-12T15:34:34.3233959+08:00` | `28374EE476E5BCA8D11FFA5AEED0E51B4CC6D91B2BCD226EFF52599E2258520E` | 298 | 298 |
| `.codex/tmp/results-inventory.csv` | `2026-07-12T15:34:39.5087700+08:00` | `7A298E548988628F253348CF719F8C4E39168DCFA078D5896E8FB6AAB7CE6C21` | 170 | 169（不含 header） |

生成命令如下；复核 tracked 清单时必须使用执行前提交 `30d2517`，复核 untracked/results 时还必须保持当时的本地材料和 results payload：

```powershell
git ls-files | Set-Content -Encoding utf8 .codex/tmp/tracked-files.txt
git ls-files --others --exclude-standard | Sort-Object | Set-Content -Encoding utf8 .codex/tmp/untracked-files.txt

$rows = @(Get-ChildItem results -Recurse -File -ErrorAction SilentlyContinue)
$summary = [PSCustomObject]@{
    path = 'results'
    file_count = $rows.Count
    bytes = (($rows | Measure-Object -Property Length -Sum).Sum)
    gib = [math]::Round((($rows | Measure-Object -Property Length -Sum).Sum) / 1GB, 3)
}
$children = @(Get-ChildItem results -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
    $files = @(Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue)
    [PSCustomObject]@{
        path = $_.FullName.Replace((Get-Location).Path + '\', '')
        file_count = $files.Count
        bytes = (($files | Measure-Object -Property Length -Sum).Sum)
    }
})
@($summary) + $children | Export-Csv -NoTypeInformation -Encoding utf8 .codex/tmp/results-inventory.csv
```

可复核统计：tracked 159 个；未忽略 untracked 状态项 30 个、展开文件 298 个；`results/` 有 22,265 个文件、168 个直接子目录、1,182,392,766 bytes，约 1.101 GiB。

- `results/` 已由 `.gitignore` 忽略；本任务不移动、不删除、不加入 Git

结果目录的最大单文件是 `results/exp_005_hcc_ackley_landscape_escape/action_trace.csv`，18,903,229 bytes。结果规模说明本阶段只建立索引，不做批量重排。

## Canonical tracked 内容边界

`git ls-tree -r --name-only b88a4d9 -- HCC_SRC results/.gitkeep` 证明：

- `HCC_SRC/` 是 `b88a4d9` 已跟踪的 canonical 内容，必须保留到 Task 3 再迁移到 `vendor/hcc/`。
- `results/.gitkeep` 是 `b88a4d9` 已跟踪的目录占位文件，必须保留。
- 二者都不是 v3.3 泄漏。只有 `results/.gitkeep` 之外的本地 results payload 属于 ignored/generated 结果，并继续留在 `E:\ARAC\results`。

迁移 CSV 的 `source_root` 取值含义固定为：

- `canonical_worktree`：`C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure`，即从 `b88a4d9` 建立的重构分支。
- `legacy_root(E:/ARAC)`：当前 root 中尚未审阅或迁移的用户材料。
- `results_root(E:/ARAC/results)`：只读索引的 ignored results payload；不包括已跟踪的 `results/.gitkeep`。

## Migration preflight

迁移表中的 `verification` 是 preflight，不执行移动：不运行 `Move-Item`、`git mv`，不写入目标目录，不修改 results payload。每条命令同时检查 source_root 下源路径存在、`source_state` 的精确 Git 状态，以及目标路径当前不存在或明确允许同一绝对路径。`review-only` 和 `defer-to-task-*` 行同样执行目标冲突检查。

实际执行命令：

```powershell
$rows = Import-Csv docs/migrations/2026-07-12-path-migration.csv
foreach ($row in $rows) {
    & pwsh -NoProfile -Command $row.verification
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
```

本轮四批次复跑记录：

| 批次 | 开始 | 结束 | 通过 |
| --- | --- | --- | ---: |
| 1-10 | `2026-07-12T16:19:42.3441980+08:00` | `2026-07-12T16:19:54.9971725+08:00` | 10/10 |
| 11-20 | `2026-07-12T16:19:55.8369216+08:00` | `2026-07-12T16:20:08.5006816+08:00` | 10/10 |
| 21-30 | `2026-07-12T16:20:09.3538109+08:00` | `2026-07-12T16:20:22.0166890+08:00` | 10/10 |
| 31-40 | `2026-07-12T16:20:22.8598779+08:00` | `2026-07-12T16:20:35.2287443+08:00` | 10/10 |

总计：**40/40 verification 通过，失败 0**。

## 当前 root 的未提交 v3.3/待审材料

以下材料在本次盘点时存在于 `E:\ARAC`，全部保持原路径，不因本任务自动加入 Git：

- 论文、历史证据和分析：`docs/aob_13_historical_win_mechanism_gap_audit.md`、`docs/aob_24_historical_best_results.md`、`docs/aob_paper_best_win_replay_matrix.md`、`docs/aob_table2_perf_with_arac.docx`、`docs/arac_action_guided_cc_manuscript_draft.md`、`docs/arac_action_guided_cc_manuscript_draft_zh.md`、`docs/flyki_12_historical_best_results.md`、`docs/hcc-focused-ablation-5000fe-summary.md`、`docs/hcc-focused-core-profile-summary.md`、`docs/hcc-targeted-ablation-pilot-analysis.md`
- 未提交设计/计划：`docs/superpowers/plans/2026-07-06-flyki-smoke-adapter.md`、`docs/superpowers/plans/2026-07-07-arac-guarded-final-protocol.md`、`docs/superpowers/plans/2026-07-10-canonical-runtime-action-controller.md`
- 未提交实验：`experiments/exp_006_flyki_adapter_smoke/`、`experiments/exp_007_flyki_cbocco_runner/`、`experiments/exp_008_arac_guarded_final_protocol/`
- 未提交外部/基准源码与数据：`Large-Scale-Overlapping-Optimization-master/`
- 未提交历史表格：`references/aob_13_historical_win_mechanism_gap_audit.csv`、`references/aob_paper_best_win_replay_matrix.csv`、`references/aob_table2_style_corrected_results.csv`
- 未提交 FlyKi 构建与 runner：`scripts/build_flyki_cbocco.ps1`、`scripts/build_flyki_objective_runner.ps1`、`scripts/flyki_gcc_compat.h`、`scripts/flyki_objective_runner.cpp`
- 未提交 benchmark 适配：`src/arac/benchmarks/`
- 未提交测试：`tests/test_exp_006_flyki_adapter_smoke.py`、`tests/test_exp_007_flyki_cbocco_runner.py`、`tests/test_exp_008_arac_guarded_final_protocol.py`、`tests/test_exp_008_arac_guarded_final_protocol_cli.py`、`tests/test_flyki_benchmark_adapter.py`

现有 v3.3 worktree 的 dirty 文件另由上述 worktree 状态命令保留为证据；本任务不复制其未提交实现。

## Worktree 交接

本 Task 1 完成后创建：

```text
path:   C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure
branch: codex/research-project-structure
start:  b88a4d9
```

该 worktree 从当前 `main` 恢复并提交以下四份交接文档：

- `docs/superpowers/specs/2026-07-12-research-project-restructure-design.md`
- `docs/superpowers/plans/2026-07-12-research-project-restructure.md`
- `docs/research-log/2026-07-12-restructure-baseline.md`
- `docs/migrations/2026-07-12-path-migration.csv`

分支以 `b88a4d9` 为祖先基线；四文档提交后 HEAD 是 `b88a4d9` 的后代，不要求也不应继续等于 `b88a4d9`。

Task 2 及以后工作不得在本 Task 1 中执行。

## 盘点结论

1. `b88a4d9` 冻结为 v3.2 canonical baseline。
2. 当前 root 的 30 个未忽略 untracked 状态项（展开为 298 个文件）和现有 v3.3 worktree 的未提交改动均保持原状。
3. 结果目录是本地生成产物，不进入 Git；本阶段只记录规模和路径索引。
4. 迁移路径、归档路径和 stable runtime 入口必须经过后续任务逐项审阅；本文件和迁移 CSV 不执行材料移动。

## Task 8/9 checkpoint

HCC runtime 拆分已在 `codex/research-project-structure` 的 `77915c2` 完成：
`hcc.py` 保留执行编排和可 monkeypatch 的兼容名字，纯动作计划、预算解析、trace
读取和共享写回分别位于 `src/arac/backends/hcc_plan.py`、`hcc_budget.py`、
`hcc_trace.py` 和 `hcc_shared_writeback.py`。完整验证记录见
`docs/research-log/2026-07-12-restructure-validation.md`。
