# ARAC Research Project Restructuring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 将 ARAC 收敛为以 v3.2 为唯一稳定算法基线、具有清晰代码/实验/证据/论文边界且可复现可审计的科研项目。

**Architecture:** 先隔离并冻结 v3.2，再把现有源码分为稳定 `src/arac`、只读 `vendor/hcc`、按阶段组织的 `experiments`、离线 `references` 和论文材料。第一轮只做路径、入口和配置边界迁移；第二轮拆分两个过大的运行时模块，所有阶段都用现有测试、结构审计、FE 账本和 anti-leakage 检查验证。

**Tech Stack:** Python 3.11+, pytest, PyYAML, NumPy/SciPy/Torch HCC backend, PowerShell, Git。

---

## 约束与执行顺序

- 稳定基线固定为提交 `b88a4d9`，不从 v3.3 未提交 diff 复制实现。
- `E:\HCC-main` 只读；ARAC 内的 HCC 源码快照只作为 vendor/backend 依赖。
- 不删除结果目录；1.1 GB 结果先建立索引，不做批量移动。
- 不把当前未跟踪材料直接批量加入 Git；每一类材料先归类、检查、再单独提交。
- 每个任务完成后运行任务级验证，并只提交本任务文件。
- 第一阶段不改变 runtime 策略、随机数流程、FE 预算或实验结果语义。

## 文件变更总览

### 创建

- `docs/research-log/2026-07-12-restructure-baseline.md`：基线和 worktree 事实记录。
- `docs/migrations/2026-07-12-path-migration.csv`：旧路径到新路径的单一迁移表。
- `docs/project-structure.md`：最终目录和科研产物规则。
- `scripts/audit_project_structure.py`：结构、缓存和路径边界审计。
- `scripts/build_results_manifest.py`：从结果目录生成可追溯 manifest。
- `analysis/README.md`、`results/README.md`、`paper/README.md`：科研产物契约。
- `vendor/hcc/README.md`：HCC 来源、只读边界和版本说明。
- `archive/README.md`：旧实现和失败实验的归档规则。
- `archive/failed-experiments/v33-late-stagnation-nda-takeover/README.md`：v3.3 负结果、结果路径和升级门记录。

### 移动或重命名

- `HCC_SRC/` → `vendor/hcc/`。
- `experiments/exp_001_schema_smoke/` → `experiments/pilots/exp_001_schema_smoke/`。
- `experiments/exp_002_aob_1run_pilot/` → `experiments/pilots/exp_002_aob_1run_pilot/`。
- `experiments/exp_003_hcc_runtime_consumer_smoke/` → `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/`。
- `experiments/exp_004_hcc_main_historical_result_recovery/` → `experiments/recovery/exp_004_hcc_main_historical_result_recovery/`。
- `experiments/exp_005_hcc_final_protocol_pilot/` → `experiments/final/exp_005_hcc_final_protocol_pilot/`。
- `experiments/exp_006_flyki_adapter_smoke/` → `experiments/pilots/exp_006_flyki_adapter_smoke/`。
- `experiments/exp_007_flyki_cbocco_runner/` → `experiments/infrastructure/exp_007_flyki_cbocco_runner/`。
- `experiments/exp_008_arac_guarded_final_protocol/` → `experiments/final/exp_008_arac_guarded_final_protocol/`。
- 论文草稿 → `paper/drafts/`；论文表格文档 → `paper/tables/`。
- 历史审计 Markdown/CSV → `docs/audits/` 和 `references/historical/`。
- `docs/core-method.md`、`docs/boundaries.md` → `docs/design/`。
- `docs/aob-final-evaluation-protocol.md` → `docs/protocols/`。

### 第二阶段拆分

- `src/arac/policy/relation_policy.py` → `evidence_model.py`、`action_policy.py`、`policy_gates.py`。
- `src/arac/backends/hcc.py` → `hcc_plan.py`、`hcc_budget.py`、`hcc_shared_writeback.py`、`hcc_trace.py`。
- `src/arac/action_space.py` → `src/arac/actions/contracts.py`，保留短期兼容导入。
- `src/arac/evaluation.py`、`src/arac/audit.py` → 对应子包。

## Task 1: Freeze v3.2 and inventory the current workspace

**Files:**
- Create: `docs/research-log/2026-07-12-restructure-baseline.md`
- Create: `docs/migrations/2026-07-12-path-migration.csv`
- Test: Git/worktree command checks

- [ ] **Step 1: Record repository and worktree facts**

Run:

```powershell
git status --short --branch
git log -10 --oneline --decorate
git worktree list --porcelain
git branch -vv
git remote -v
```

Expected: 记录 `main`、v3.2 `b88a4d9`、v3.3 dirty worktree 和当前未跟踪材料；不修改任何文件。

- [ ] **Step 2: Inventory tracked, untracked, generated, and large files**

Run:

```powershell
git ls-files > .codex/tmp/tracked-files.txt
git ls-files --others --exclude-standard > .codex/tmp/untracked-files.txt
Get-ChildItem results -Recurse -File | Measure-Object Length -Sum
```

将每个未跟踪路径分类为 `retain`、`migrate`、`archive` 或 `review`。`results/` 继续作为本地生成产物。

- [ ] **Step 3: Verify v3.2 without changing the dirty root**

Run:

```powershell
git cat-file -t b88a4d9
git diff --stat main..codex/nondense-runtime-lock
git -C C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-nondense-v24 status --short --branch
```

Expected: 第一条输出 `commit`；v3.3 的已知未提交文件仍在原 worktree；当前根目录未跟踪文件未被覆盖。

- [ ] **Step 4: Write the baseline record and migration table**

基线记录写明执行前和 Task 1 提交后两个快照、canonical SHA、v3.3 dirty-file list、结果目录规模和“仅 v3.2 可进入稳定 runtime”。CSV 保留原七列并增加 `source_root`，固定列为：

```text
old_path,new_path,category,source_root,source_state,action,git_policy,verification
```

- [ ] **Step 5: Validate and commit the inventory artifacts**

Run:

```powershell
git diff --check -- docs/research-log/2026-07-12-restructure-baseline.md docs/migrations/2026-07-12-path-migration.csv
git add docs/research-log/2026-07-12-restructure-baseline.md docs/migrations/2026-07-12-path-migration.csv
git commit -m "docs: record restructuring baseline and path inventory"
```

Expected: diff check 无输出；提交只包含两份盘点文件。

- [ ] **Step 6: Create a clean implementation worktree from v3.2**

Run from `E:\ARAC` after the plan commit exists on `main`:

```powershell
git worktree add C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure -b codex/research-project-structure b88a4d9
git -C C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure restore --source 95bbbaa -- docs/superpowers/specs/2026-07-12-research-project-restructure-design.md docs/superpowers/plans/2026-07-12-research-project-restructure.md docs/research-log/2026-07-12-restructure-baseline.md docs/migrations/2026-07-12-path-migration.csv
git -C C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure add -- docs/superpowers/specs/2026-07-12-research-project-restructure-design.md docs/superpowers/plans/2026-07-12-research-project-restructure.md docs/research-log/2026-07-12-restructure-baseline.md docs/migrations/2026-07-12-path-migration.csv
git -C C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure commit -m "docs: seed restructuring worktree"
git -C C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure merge-base --is-ancestor b88a4d9 HEAD
git -C C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure ls-files --error-unmatch -- docs/superpowers/specs/2026-07-12-research-project-restructure-design.md docs/superpowers/plans/2026-07-12-research-project-restructure.md docs/research-log/2026-07-12-restructure-baseline.md docs/migrations/2026-07-12-path-migration.csv
git -C C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure status --short
```

Expected: 新 worktree 分支以 `b88a4d9` 为祖先基线；固定提交 `95bbbaa` 是四份交接文档的快照来源，不是新的算法基线。四份文档已 tracked 并提交，`status --short` 无输出。提交后的 HEAD 是 `b88a4d9` 的后代，不要求 HEAD 永远等于 `b88a4d9`。后续文档修订直接在 `codex/research-project-structure` 继续提交，不依赖重新读取可变的 `main`。当前 `E:\ARAC` 未跟踪材料和原 v3.3 dirty worktree 均保持原状。后续 Tasks 2-9 在该干净 worktree 执行。

后续命令统一使用：

```powershell
$RepoRoot = 'C:/Users/83718/.config/superpowers/worktrees/ARAC/codex-research-project-structure'
$ResultsRoot = 'E:/ARAC/results'
Set-Location $RepoRoot
```

`$RepoRoot` 是被重构和提交的代码树；`$ResultsRoot` 是只读索引的现有大结果目录。

## Task 2: Create the scientific project skeleton and structure audit

**Files:**
- Create: `docs/project-structure.md`
- Create: `analysis/README.md`, `results/README.md`, `paper/README.md`
- Create: `vendor/hcc/README.md`, `archive/README.md`
- Create: `scripts/audit_project_structure.py`
- Modify: `.gitignore`
- Test: `tests/test_project_structure_audit.py`

- [ ] **Step 1: Write failing structure-audit tests**

测试必须验证：目标顶层目录合法；被跟踪的 `__pycache__`/`.pyc` 会失败；`src/arac` 引用 paper/historical 路径会失败；`results/` 被识别为 generated。测试使用临时目录，不扫描 1.1 GB 结果。

- [ ] **Step 2: Confirm the test fails before implementation**

Run: `pytest tests/test_project_structure_audit.py -q`

Expected: FAIL，原因是 `scripts.audit_project_structure` 尚不存在。

- [ ] **Step 3: Create directory contracts**

`docs/project-structure.md` 写明目标树、命名、raw/generated 区分、claim level 和 reference-blind 规则。每个 README 写明输入、输出和 Git policy，不能是空目录占位。

- [ ] **Step 4: Implement the minimal audit CLI**

CLI:

```powershell
python scripts/audit_project_structure.py --root $RepoRoot
```

只使用标准库；违规输出格式为 `path: rule`；忽略 `.git`、`.venv`、`.pytest_cache` 和 `results` 内容；验证 `results/` 被 Git 忽略；有违规返回非零。

- [ ] **Step 5: Update ignore rules**

保留 `results/*`，增加 `analysis/generated/`、`logs/`、vendor 缓存和临时 manifest；README/schema 仍可跟踪。

- [ ] **Step 6: Verify and commit**

Run:

```powershell
pytest tests/test_project_structure_audit.py -q
python scripts/audit_project_structure.py --root $RepoRoot
git diff --check
```

Expected: 测试通过；audit 仅报告尚未迁移的已知路径；无 result 文件被暂存。提交信息：`feat: add scientific project structure audit`。

## Task 3: Establish the v3.2 canonical source and HCC vendor boundary

**Files:**
- Move: `HCC_SRC/` → `vendor/hcc/`
- Modify: `vendor/hcc/README.md`, `pyproject.toml`, `README.md`
- Modify: all path consumers found by `rg`
- Test: HCC adapter and smoke tests

- [ ] **Step 1: Find every hard-coded HCC path**

Run:

```powershell
rg -n "HCC_SRC|arac_hcc_smoke_runner|HCC-main|HCC\\|AOB\\" src tests scripts experiments configs README.md docs
```

把结果写入迁移记录，分类为 runtime、documentation、test fixture 或 historical evidence。

- [ ] **Step 2: Add a failing backend-root contract test**

测试使用临时 backend root，验证 adapter 从显式 root 解析 `AOB`、`HCC` 和 runner；并验证解析不依赖当前工作目录。

- [ ] **Step 3: Move the vendor snapshot and update paths**

整体移动到 `vendor/hcc/`。只改路径和 import，不改 vendor 优化逻辑。runner 接收显式 vendor root 或从 repository root 推导。

- [ ] **Step 4: Record vendor provenance**

README 写明来源 `E:\HCC-main`、快照用途、只读规则、v3.2 兼容性和 smoke 命令。

- [ ] **Step 5: Run HCC regression and cwd-equivalence checks**

Run:

```powershell
pytest tests/test_hcc_backbone_adapter.py tests/test_hcc_execution_adapter.py tests/test_hcc_smoke_runner_cli.py -q
```

Expected: 全部通过；从仓库根和无关 cwd 运行 smoke 都解析到同一 vendor root，并保持 FE accounting。

- [ ] **Step 6: Commit the vendor migration**

检查 staged rename 和 `git diff --cached --check`，提交信息：`refactor: isolate HCC vendor source boundary`。

## Task 4: Normalize stable package boundaries without behavior changes

**Files:**
- Create: `src/arac/actions/`, `execution/`, `evaluation/`, `audits/`
- Move/compatibility: `action_space.py`, `backend_adapter.py`, `evaluation.py`, `audit.py`
- Modify: package imports and tests
- Test: `tests/test_package_boundaries.py` plus full suite

- [ ] **Step 1: Write failing package-boundary tests**

验证新子包可导入且不会导入 `vendor.hcc`；新旧公开 action/evaluation 类型行为一致。

- [ ] **Step 2: Confirm boundary tests fail**

Run: `pytest tests/test_package_boundaries.py -q`

Expected: 新包路径不存在导致 FAIL。

- [ ] **Step 3: Move modules by responsibility**

action contracts 进入 `actions/`；backend interfaces 进入 `execution/`；FE helpers 进入 `evaluation/`；claim/leakage gates 进入 `audits/`。旧路径仅保留无重复逻辑的 re-export shim。

- [ ] **Step 4: Update imports and exports**

更新 `src/arac/__init__.py`、实验和测试；任何 package 模块都不能读取 paper/reference CSV。

- [ ] **Step 5: Run package regression**

Run:

```powershell
pytest -q
python -c "import arac; import arac.actions; import arac.execution; import arac.evaluation; import arac.audits"
```

Expected: 全部通过，导入过程不访问 paper/historical files。

- [ ] **Step 6: Commit**

提交信息：`refactor: separate ARAC package responsibilities`。

## Task 5: Organize experiments by research stage

**Files:**
- Move: `experiments/exp_001` through `exp_008` into approved stage folders; `exp_006` is a pilot smoke and `exp_007` is infrastructure, not an ablation
- Modify: `experiments/README.md`, moved runners and tests
- Test: `tests/test_experiment_layout.py` and all `test_exp_*` files

- [ ] **Step 1: Write the experiment-layout test**

每个非 archive 实验必须有 README、入口和明确阶段；阶段包括 `pilots`、`infrastructure`、`recovery`、`ablations` 和 `final`；archive 不被默认 runner 导入；输出仍位于 `results/`。

- [ ] **Step 2: Add explicit repository-root resolution**

在唯一 helper 中用 `Path(__file__)` 解析 root；runner 用它定位 config、vendor 和 results，不使用 cwd。

- [ ] **Step 3: Move all experiment packages**

保持 `exp_00N_*` ID、输出 schema、claim level 和 runtime 参数不变；`exp_006_flyki_adapter_smoke` 归入 `pilots`，`exp_007_flyki_cbocco_runner` 归入 `infrastructure`（benchmark/build/外部 runner 基础设施），二者不归入 `ablations`；本步不重命名现有结果目录。

- [ ] **Step 4: Update imports and commands**

更新 `test_exp_*`、README 和模块调用命令；每个 README 标明 pilot/infrastructure/recovery/ablation/final。

- [ ] **Step 5: Run experiment tests in three batches**

Run:

```powershell
pytest tests/test_experiment_layout.py tests/test_exp_001_schema_smoke.py tests/test_exp_002_aob_1run_pilot.py -q
pytest tests/test_exp_003_hcc_runtime_consumer_smoke.py tests/test_exp_004_hcc_main_historical_result_recovery.py -q
pytest tests/test_exp_005_hcc_final_protocol_pilot.py tests/test_exp_006_flyki_adapter_smoke.py tests/test_exp_007_flyki_cbocco_runner.py tests/test_exp_008_arac_guarded_final_protocol.py -q
```

Expected: 全部通过，且不导入失败 v3.3 worktree。

- [ ] **Step 6: Commit**

提交信息：`refactor: organize experiments by research stage`。

## Task 6: Separate documents, paper material, and offline evidence

**Files:**
- Move: method/protocol/audit documents according to migration CSV
- Move: manuscripts and table documents into `paper/`
- Modify: root and documentation indexes
- Test: `tests/test_offline_boundary.py`

- [ ] **Step 1: Write the offline-boundary test**

扫描 `src/arac` Python 文件；引用 `paper/`、`references/paper/`、`references/historical/` 或 result output path 时失败；offline analysis 仍可读取这些文件。

- [ ] **Step 2: Move only paths approved in the migration CSV**

优先使用 `git mv` 保留历史；`docs/superpowers/` 保持原位；未归类的用户材料不移动。

- [ ] **Step 3: Archive the failed v3.3 experiment without importing its runtime code**

创建 `archive/failed-experiments/v33-late-stagnation-nda-takeover/README.md`，记录设计/计划原路径、结果目录 `E:\ARAC\results\exp_017_controller_v33_r3_seed123_3m_jobs16`、三 seed 结果、未通过 `3.28e5` 升级门的结论，以及“641 行未提交实现不得进入 stable runtime”。不复制 v3.3 Python 实现。

- [ ] **Step 4: Update indexes and root README**

README 写明 v3.2 canonical baseline、研究 pipeline、claim ladder、25-run 未完成限制，以及 paper/offline evidence/results 的位置。

- [ ] **Step 5: Check links and offline boundary**

Run:

```powershell
pytest tests/test_offline_boundary.py -q
rg -n "docs/(core-method|boundaries|aob-final-evaluation-protocol)" README.md docs experiments tests scripts
```

Expected: 测试通过，搜索无需要修复的旧路径。

- [ ] **Step 6: Commit**

提交信息：`docs: separate research records from runtime code`。

## Task 7: Add deterministic result manifest generation

**Files:**
- Create: `scripts/build_results_manifest.py`
- Modify: `results/README.md`, `.gitignore`, experiment docs
- Test: `tests/test_results_manifest.py`

- [ ] **Step 1: Write manifest schema tests**

用临时 fixture 验证列 `experiment_id,protocol,git_commit,config_path,seed,case_id,total_fe,status,claim_level,output_path`；不完整 metadata 标记为 `partial`；不能把 paper values 写入 runtime artifacts。

- [ ] **Step 2: Confirm the test fails**

Run: `pytest tests/test_results_manifest.py -q`

Expected: generator 不存在导致 FAIL。

- [ ] **Step 3: Implement a standard-library CSV generator**

CLI:

```powershell
python scripts/build_results_manifest.py --root $RepoRoot --results $ResultsRoot --output "$RepoRoot/.codex/tmp/results-manifest.csv"
```

按路径稳定排序，只读 run metadata；缺失字段标记 `partial`；不改源码、配置和结果 payload。

- [ ] **Step 4: Verify deterministic output**

连续运行两次并比较 SHA256；Expected: 两份 CSV hash 相同，`results/` 无修改。

- [ ] **Step 5: Commit**

提交信息：`feat: add reproducible result manifest protocol`。

## Task 8: Split large runtime modules behind regression gates

**Files:**
- Split: `src/arac/policy/relation_policy.py`
- Split: `src/arac/backends/hcc.py`
- Create: focused policy and HCC execution modules from the approved design
- Test: policy/HCC suites and `tests/test_runtime_module_boundaries.py`

- [ ] **Step 1: Capture v3.2 behavior**

运行现有 policy/HCC tests，在 research log 记录命令、commit 和测试摘要，不提交 generated results。

- [ ] **Step 2: Write responsibility-boundary tests**

验证 evidence model 不执行 optimizer；policy 不读 final/paper；budget 只负责 FE；shared writeback 接收显式 decision 而非 case label。

- [ ] **Step 3: Extract one responsibility at a time**

只移动定义，不改语义；每次提取后运行对应窄测试；旧模块先做 re-export。

- [ ] **Step 4: Remove duplicate implementations**

调用方迁移后，兼容模块只保留 re-export；测试验证旧新 public contract object identity。

- [ ] **Step 5: Run full regression and audit**

Run:

```powershell
pytest -q
python scripts/audit_project_structure.py --root $RepoRoot
```

Expected: 全部通过，无重复 runtime 实现，无 paper/historical runtime 引用。

- [ ] **Step 6: Commit**

提交信息：`refactor: split policy and HCC execution responsibilities`。

## Task 9: Final reproducibility gate and handoff

**Files:**
- Modify: `README.md`, baseline research log
- Create: `docs/research-log/2026-07-12-restructure-validation.md`
- Test: full repository verification

- [ ] **Step 1: Run the complete verification matrix**

```powershell
pytest -q
python scripts/audit_project_structure.py --root $RepoRoot
python scripts/build_results_manifest.py --root $RepoRoot --results $ResultsRoot --output "$RepoRoot/.codex/tmp/results-manifest.csv"
git diff --check
git status --short
```

- [ ] **Step 2: Verify canonical and anti-leakage boundaries**

确认测试代码来自 v3.2；v3.3 失败实现未被导入；runtime 不含 paper/historical/final-only 字段；HCC smoke 保持 same-budget FE。

- [ ] **Step 3: Write the validation record**

记录命令、日期、commit、测试结果、structure audit、manifest、已知限制和“未重跑 25-run final protocol”。

- [ ] **Step 4: Review Git cleanliness**

不得暂存 `results/`、`.venv/`、缓存、`.codex/tmp/`、日志或未审阅用户材料。运行 `git diff --cached --check` 和 `git status --short --branch`。

- [ ] **Step 5: Commit the handoff**

提交信息：`docs: record research project restructuring validation`。推送必须另行获得用户确认。

## Self-review against the design spec

- Canonical v3.2 和 v3.3 archive 边界：Tasks 1、3、8、9。
- 科研目录结构：Tasks 2、3、5、6。
- runtime package 边界：Tasks 3、4、8。
- 实验和 manifest 协议：Tasks 5、7。
- offline evidence 和 paper 隔离：Task 6 及 Task 9。
- 不批量移动结果：Tasks 1、7。
- 验证与 Git 清洁：Tasks 2 至 9。

计划不包含未决占位符；所有产出任务都给出了准确路径、验证命令和预期行为。
