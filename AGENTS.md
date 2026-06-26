# AGENTS.md - ARAC 项目规则

本项目是从 `E:\HCC-main` 移植出来的 clean ARAC 包。`E:\HCC-main`
是源工程和证据库，但结构很乱；`E:\ARAC` 的目标是逐步收敛成可维护、
可验证、可复现的实现，而不是复制旧项目的文件膨胀。

## 1. 语言与来源

- 默认使用简体中文沟通。
- 新增或修改的代码、CSV、Markdown、配置文件统一使用 UTF-8。
- 做 HCC/AOB/MI-ARAC 相关任务前，必须先检查 `E:\HCC-main` 的相关源文件、
  artifacts 或测试，避免凭空重写旧逻辑。
- `E:\HCC-main` 默认只读。除非用户明确要求修源工程，否则不要修改它。

## 2. 移植原则

- ARAC 不是从 0 开始；它是从 `E:\HCC-main` 抽出的核心方法。
- 优先移植“稳定接口、证据抽取、runtime 边界、backend binding、gate/audit”
  这些可复用部分，不搬运历史 milestone 垃圾、临时 runner、大量旧 artifacts。
- 不能把 paper reported baseline、oracle、final error、relative gain、problem
  family label、prior outcome 作为 runtime dispatch 输入。
- 论文表格和历史 final/pilot 结果只允许用于 offline evaluation、blocker
  classification 和报告对比。

## 3. 版本收敛规则

- 每完成一个可验证版本，必须本地验证、提交到 Git，并按用户要求推送到
  `origin/main` (`https://github.com/xqsa/ARAC.git`)。
- 有更新时优先“替换/升级已有入口和文档”，不要不断新增平行文件、重复 runner、
  重复协议文档或第二事实源。
- 新增文件必须回答三个问题：
  1. 它是否是新的稳定接口、测试、配置、实验入口或必要文档？
  2. 是否已有文件可以被更新替代？
  3. 如果它取代旧文件，旧文件是否已删除、归档或在文档中标明废弃？
- 不允许因为怕覆盖旧版本而无限增加 `v2/v3/final_new/latest_fixed` 文件。
  版本历史交给 Git 保存。
- 废弃实现要删除；确需保留的历史证据放在明确的 `archive` 或 `references`
  位置，并说明为什么不能删除。

## 4. 文件与目录边界

- `src/arac/`：稳定库代码和 backend adapter。
- `experiments/exp_*`：可运行实验入口；一个实验一个目录。
- `configs/`：当前有效配置。过期配置要删除或明确归档。
- `docs/`：协议、设计、验证说明。避免同一协议散落多份。
- `references/`：论文表格、源工程索引、外部只读证据。
- `results/`：可重生成实验输出，默认不入 Git。
- `.codex/tmp/`：临时抽取/中间材料，不提交，除非用户明确要求沉淀为正式文档。

## 5. 实验推进规则

- 先做小的 1-run pilot，再考虑 25-run final protocol。
- 不重跑论文 baseline；只跑我们自己的方法，再和论文 reported values 做
  offline comparison。
- scaffold/proxy 必须显式标注，例如 `scaffold_synthetic_proxy`，不能包装成真实
  full optimizer performance。
- 从 scaffold 升级到真实实验时，必须接 HCC 的真实 AOB metadata、RDDSM grouping、
  HCC optimizer execution、trace evidence 和 same-budget FE 账本。
- catastrophic loss 是硬门。不能只用 win count 或 mean gain 宣称成功。

## 6. 验证与提交

- 修改代码后优先运行聚焦测试；能自动验证就不要只靠人工判断。
- 提交前至少检查：
  - `git status --short`
  - 相关 pytest
  - `git diff --check` 或 `git diff --cached --check`
  - 是否误提交 `results/`、缓存、临时文件或大日志
- 每次提交信息要描述实际版本目标，不写空泛的 “update”。
- 提交后必须推送到 GitHub remote。若推送失败，要报告失败原因和当前 commit。

## 7. 清洁底线

- 不硬编码密钥、令牌、私有凭据。
- 不提交 `__pycache__`、`.pytest_cache`、临时日志、实验大输出。
- 不保留死代码、重复逻辑和第二事实源。
- 不为一次性探索新增长期维护依赖。
- 不回滚用户已有改动，除非用户明确要求。
