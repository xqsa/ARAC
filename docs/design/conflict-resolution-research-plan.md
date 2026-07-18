# exp_019 冲突取值消解研究计划

Date: 2026-07-18
Status: 第 1 步已运行并得到 `oracle_no_go`；STOP，不进入第 2 步
Executor: Codex（实现），用户与 Claude（阶段审查）

## 1. 研究结论与当前起点

exp_018 已真实运行 mechanism gate，结果为 `pilot_no_go`。AOB E3/A4/S5
使用 `rotateVectorConform`，共享变量在相邻子组内具有同一局部最优值；因此，
exp_018 的 reliability-weighted bridge 在这些 conform 案例上缺少可辨识的收益
天花板。

exp_019 先建立隔离的 synthetic conflict 孪生基准，再判断同 checkpoint 的取值
仲裁是否具有可观测收益。全程 observer-only，不修改 `vendor/hcc`、exp_018、
`scripts/hcc_smoke_runner.py` 或 runtime state。

原计划所写的“纯离线重放”不可实现：exp_018 没有保存候选向量，无法在不伪造
输入的前提下重算 conflict-side objective。第 1 步因此改为 fresh conflict-side
HCC trajectory 和四点探测；conform 阴性侧只读取并哈希绑定 exp_018 已有的 15 条
paired-owner trajectories。

第 0.5 步经用户批准后，第 1 步已按冻结矩阵真实运行。结果为
`oracle_no_go`：conflict material win 0/15、paired-win LCB 0、median delta
`-7.307711256377876e-04`、trajectory-level large loss 0。conform 阴性侧通过其
冻结条件。详细证据见 `docs/design/conflict-resolution-diagnostic.md`。按本计划
第 4.2 节，当前必须 STOP，第 2/3 步保持未授权。

## 2. 不可逾越的边界

- 新工作只落在 `experiments/pilots/exp_019_conflict_resolution_pilot/` 和对应设计
  文档、测试；`E:\HCC-main` 与 `vendor/hcc` 只读。
- synthetic 案例必须使用 `conflict_variant_synthetic` 标识，不得冒充官方 AOB。
- paper reported baseline、oracle、final error、relative gain、problem family label
  和 prior outcome 不得成为 runtime dispatch 输入。
- conflict score、BA、agreement、VOI 只允许决定是否产生影子候选，不得进入
  outcome 检验统计量。
- 禁止 runtime writeback、动态 regrouping、结果后调阈值和新增平行版本入口。
- 每个阶段单独验证、提交、推送并等待用户 go/no-go；未获准时不得跨阶段实施。

## 3. 第 0.5 步：受控冲突基准

本阶段建立同一个 exp_019 目录，但不创建正式在线 pilot 的 `protocol.py`、
`run.py`、`README.md` 或 `expected_outputs.md`。

### 3.1 构造

- experiment-local adapter 子类化 vendor 的 elliptic/ackley/schwefel；只加载本
  实验的 `OvectorVec`，并在本地覆盖 batch-safe `rotateVectorConflict` 计算。
- topology、rotation、design、weights、Pvector 和基准 xopt 继续从只读 vendor
  数据目录加载，不复制约 60 MB 的基础数据。
- CSV schema 固定为：
  `variant_id,group_index,local_index,global_variable_index,base_optimum,conflict_optimum,is_shared`。
- 冲突强度冻结为 `rho=0.10`。对共享全局变量的左右 owner，分别使用
  `v + rho * (lower - v)` 和 `v + rho * (upper - v)`；非共享变量保持 `v`。
- 三个 ID 固定为 `E3_conflict_variant_synthetic`、
  `A4_conflict_variant_synthetic`、`S5_conflict_variant_synthetic`。
- manifest 绑定 generator version、rho、全部 F3/F4/F5 vendor 基础文件 SHA256
  和 synthetic CSV SHA256。任一缺失、schema 不符或 hash 不符均 fail-closed。
- CEC 2013 f13/f14 本轮暂缓，待 AOB conflict 阳性 gate 通过后再独立处理许可和
  适配。

### 3.2 验证与暂停点

单测覆盖 rho=0 与 conform 数值等价、共享变量是唯一变化、边界内取值、1D/batch
objective、`fitness_record`、确定性 hash，以及缺失、重复或篡改数据 fail-closed。
完成聚焦测试、全量回归、受保护路径 diff 审计、提交和推送后暂停，等待第 0.5 步
go/no-go。

构造与声明边界见 `docs/design/conflict-benchmark-construction.md`。

## 4. 条件性第 1 步：oracle-ceiling 诊断

前提：用户明确批准第 0.5 步。

### 4.1 执行边界

- 新增 experiment-local `diagnostic --stage smoke|oracle`。runner 只在子进程内
  替换 `scripts.hcc_smoke_runner` 的 benchmark factory，复用 HCC、四点 probe、
  FE ledger 和 artifact writer，不修改共享 runner。
- smoke：A4 conflict、seed 1、100k FE。
- oracle：E3/A4/S5 conflict、seeds 117-121、3M FE、`jobs=12`，共 15 条 fresh
  conflict trajectories。
- conform 阴性侧只读取 exp_018 mechanism 的 15 条 paired-owner trajectories，
  并绑定源 artifact hash；缺失、schema 错误或完整性 gate 失败时直接中止。
- 主 baseline 为可靠度较高的 owner，可靠度相同选 left。bridge 使用现有
  `1+reliability` 权重和 0.65 cap。最佳 owner 只作离线敏感性分析，不进入主 gate。

### 4.2 独立单位与统计门

每条 trajectory 对四个冻结 relations 计算：

```text
delta = log((f_baseline + eps) / (f_bridge + eps))
```

以四个 relation delta 的中位数作为一个独立 case-seed 配对单位。material win
定义为 `delta > log(1.01)`；large loss 定义为 `delta <= -log(1.20)`。对 15 个
case-seed 单位计算 one-sided 95% Wilson `paired_win_lcb` 和 `large_loss_ucb`。

Conflict 侧通过条件全部固定为：

- `paired_win_lcb > 0.5`；
- 总体 median delta `> log(1.01)`；
- 实际 large-loss count 为 0。

Conform 侧通过条件全部固定为：

- `paired_win_lcb <= 0.5`；
- 总体 `abs(median delta) <= log(1.01)`；
- 实际 large-loss count 为 0。

Conform 结论只能写“与无材料收益相容”，不能写成统计等价性证明。报告必须给出
逐 case/seed 原始结果、两侧 LCB/UCB、median 和失败原因，并明确验证对象是同
checkpoint 即时 objective value，不是 writeback 后的终局收益。

任一条件失败即 STOP；全部通过后单独提交、推送并等待用户批准第 2 步。

## 5. 条件性第 2 步：只写机制规格

前提：第 1 步通过且用户批准。

- conflict score 固定为 `owner proposal normalized gap` 与“两个 proposal 位于
  anchor 两侧”的变量比例乘积，阈值固定 `0.05`。
- 输入缺失、非有限、越界或 score 低于阈值时必须 fallback。
- 仲裁值继续使用 capped reliability bridge。
- PACE/LH-CC 与三条“不能声称”必须写入规格；e-process 不进入 exp_019 v1，
  待基础 conflict gate 通过后另行设计。
- 本阶段只产出 `docs/design/conflict-resolution-spec.md`，不写 pilot 代码。

三条不能声称固定为：

1. 该方法不是新的分解算法，而是既有候选池上的取值决策层。
2. 该方法不创造收益，只尝试回收真实冲突型重叠中被现有取值策略损失的价值。
3. 不得声称对 anytime-valid runtime gate 范式的优先权。

## 6. 条件性第 3 步：正式 observer-only pilot

前提：第 2 步规格经用户与 Claude 批准。

第 3 步只向第 0.5 步已经建立的同一 exp_019 目录增加 `protocol.py`、`run.py`、
`README.md` 和 `expected_outputs.md`，不得另建平行目录。

- fresh seeds 217-221；E3/A4/S5 conform/conflict；3M FE；`jobs=12`；共 30 条
  单 observer-lane trajectories。
- 必须 fresh/完整、精确 FE，且通过 hash、anti-leakage 和 runtime forbidden
  fields 审计。
- promotion gate：conflict material-win LCB `>0.5`，material win 覆盖三个
  cases 和至少四个 seeds；conform non-fallback count 为 0；large-loss count 为 0。
- 通过只授权设计独立 writeback 实验，不授权 runtime action 或性能声明。

## 7. 接口、文献和提交纪律

新增公开面仅限 experiment-local `ConflictBenchmarkFactory`、synthetic manifest，
以及获准第 1 步后才增加的 `diagnostic --stage smoke|oracle` CLI。`src/arac`、vendor、
exp_018 接口保持不变。

文献定位必须引用 OEDG (`10.1109/TEVC.2024.3390719`)、OCC
(`10.1145/3638529.3654171`)、HCC (`10.1145/3712255.3726560`)、重叠策略研究
(`10.1007/978-3-030-85672-4_19`)、PACE (arXiv:2606.08106) 和 LH-CC
(arXiv:2604.01241)。DOV 元数据核实前不引用。novelty 只写成有限检索下的窄定位，
不得使用“没有人做过”或全球首次声明。

每个获准阶段均运行相关测试及全量回归，并检查：

- `git diff -- vendor/hcc` 为空；
- `git diff -- experiments/pilots/exp_018_rddsm_evidence_overlay_pilot` 为空；
- 无 `results/`、缓存或大日志进入提交；
- e-process 计划不与 exp_019 阶段提交捆绑。

## 8. Novelty 先例修正（2026-07-18 晚，基于用户本地 17 篇重叠文献排查 + 亲验 HCC 原文）

**这是本计划写就之后才查实的关键情报，第 5/7 节的 novelty 措辞必须据此收窄。
此前"取值消解无先例"的隐含前提是错的。**

value-side（取值消解）已有两个先例，必须在 spec 与 related work 显式处理：

1. **HCC / Two-Phase CC（arXiv 2503.21797，本项目骨干）Eq.8 —— 最强威胁。**
   重叠变量最终取值 = 各子空间优化值的**贡献加权和**，权重
   γ = Δ_i/(Δ_i + Δ_j)（Δ = 各子空间 fitness 改善量），并把"重叠变量取值
   耦合"列为其三大待解问题之一。**这与本计划第 5 节"capped reliability
   bridge 仲裁"是同一类动作（两侧提议的可靠性/贡献加权折中）。** 若不区分，
   本方法会与骨干论文正面重叠。HCC 的关键缺口：**对所有重叠变量无差别套用
   Eq.8，从不区分 conforming/conflicting（全文无此二词）。**

2. **DCCMAES（IEEE Access 2019）。** 专打冲突型 f14，操纵共享变量分布均值
   （WC/GS/EO + ROE 正交竞争）。有冲突针对性，但取值为 consensus-forcing
   或朴素中点 (a+b)/2，且假设"最终必须收敛到单一值"。

3. DOV（Meselhi 2022，未获原文）：HCC/OEDG 均引用其"跨子空间取均值"。
   如第 7 节所述，元数据核实前不引用。

**据此强制约束（覆盖第 5、7 节相关措辞）：**

- **严禁**声称"首次证据驱动消解共享变量取值"——会被骨干 HCC Eq.8 当场反驳。
- 本工作可诚实主张的收窄增量 = **冲突条件化（conflict-conditioning）**：
  先用四点探测（xL vs xR 分歧）廉价判定哪些共享变量真冲突，**只对冲突变量**
  施加仲裁，一致型变量省略/简化。相对 HCC 的差异是"选择性施加"而非"全量施加"。
- **spec 与 pilot 必须新增一条强制消融基线：HCC Eq.8 全量贡献加权 vs 本工作
  的冲突条件化仲裁。** 缺此消融，收益无法归因于"冲突条件化"这一增量，
  novelty 不成立。这条与第 4.1 节的 owner-baseline 对照并列，不可省略。
- 第 5 节三条"不能声称"新增第 4 条：不得声称取值加权本身是新的——证据加权
  取值组合已由 HCC Eq.8 确立；本工作的增量仅在"冲突条件化 + 非朴素仲裁"。
