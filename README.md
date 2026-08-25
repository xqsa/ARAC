# ARAC-OC：面向变量重叠的大规模全局优化

当前框架是**soft-RDDSM 发现 + 变量级重叠证据 sidecar + 非学习结构路由 +
四动作 Phase-II**：Phase-I 用 soft-RDDSM 从计数黑盒评价中发现重叠结构，
经确认的 hyperedge 构造变量级 sidecar，由确定性结构规则（不训练、不读
benchmark 身份）路由到恰好一个 Phase-II 动作，在精确 3,000,000 FE 终止。

```text
soft-RDDSM Phase-I（身份盲，180,000 FE）
  -> confirmed hyperedge -> 变量级重叠证据 sidecar
  -> 非学习结构路由（AOR / SMP / CTP / GCB）
  -> action-view checkpoint -> 恰好一个 Phase-II 动作
  -> EvaluationLedger：exact-FE 计费 + strict-best 仲裁 + receipt/state hash
```

## 框架分述

### 1. soft-RDDSM 发现（Phase-I v10）

协议 `arac-soft-rddsm-mainline-v10-candidate-1`
（`experiments/upgrade/soft_rddsm_mainline_v10/phase1_v10.py`，
机制在 `src/arac/evidence/soft_rddsm.py`）：

```text
240-FE v9 景观探针（与 v9 同 seed 逐位相同）
  -> 变量签名（Gate 43 共享测量基：P=12 固定 batch 池）
  -> mutual-kNN 候选边（无 d^2 矩阵）
  -> 两级计费条件对探针（screen -> confirm，逐边 support 而非单阈值判定）
  -> 软设计结构矩阵（加权 + support）
  -> RDG 粗分组 + 小区域递归细化
  -> 完全交叠可分性 + 双侧确认 -> ResolvedOverlapHyperedge（Gate 42 语义）
  -> checkpoint blocks/relations（T0 映射）+ MMES 补齐至精确 180,000 FE
```

40 个特征沿用 v9 命名（landscape 逐位 v9、structural 由发现证据馈入、
progress 分段映射）。前置 Gate 42（证据语义）、43（变量签名）、44
（区域接口）已全部通过；C1 活体复现：AOB 六 case 召回 0.789（R6）至
1.000（R2），精度全程 1.0，E1/R1 零误报。

### 2. 变量级重叠证据 sidecar

`src/arac/evidence/soft_rddsm_adapter.py` 是分层证据到 Phase-II 之间的
窄桥：叶子仍是互斥分区，只有 `ResolvedOverlapHyperedge` 能把变量加入
额外的 owner group，产出 `Phase1OverlapEvidence`（groups / memberships /
逐对置信度 / completeness），schema 为
`arac-soft-rddsm-overlap-evidence-v1` 并带 hash。区域级交互没有确认
hyperedge 时，证书标记为不完整，而不是静默变成重叠隶属。

### 3. 非学习结构路由

`src/arac/analysis/structural_router.py` 的 `route_from_overlap_evidence`
只依据证据完整性、共享变量存在性和 owner 图连通性；目标尺度、benchmark
名称、拟合 selector、在线增益信号都不进入决策：

```text
证据不完整                    -> AOR   （overlap_evidence_incomplete）
完整且无共享变量               -> SMP   （complete_disjoint_structure）
完整且多个重叠连通分量         -> CTP   （complete_disconnected_overlap_components）
完整且连通的重叠图             -> GCB   （complete_connected_overlap_graph）
```

主路由不是动作排他标签：只要变量级证据完整，SMP 保持结构兼容
（`zero_relation` / `overlap_aware` 两种 lifecycle 模式）。升级 lane 的
组合入口 `experiments/upgrade/soft_rddsm_structural_router_v1` 用
`action_view_checkpoint()` 把 sidecar 的 owner 图投影为
`PhaseCheckpoint.relations`（incumbent、Phase-I/terminal FE 边界不变，
原始 checkpoint hash 与 action view hash 分开记录）。

`src/arac/dispatch_policy.py` 保留另一条 Gate 41a 校准的两特征规则
（`tail_log10_gain` + `structural_relation_density`，阈值落在族间空隙），
供 C3 口径的历史对照；两条路由都是确定性规则，无训练组件。

### 4. 四动作 Phase-II

- **CTP**（Coverage-to-Polish）、**SMP**（State-Memory Persistence）、
  **GCB**（Graph-Conditioned Balancing）、**AOR**（Adaptive Optimizer
  Routing）实现同一 `ActionContext -> ActionResult` 契约；
- 恢复基线 `arac-recovered-baseline-20260823-v1` 以
  `RecoveredActionRegistry`（`src/arac/actions/recovered_registry.py`）
  为冻结锚点，生产默认 `patch=false`、`soft_routing=false`、
  `selector=false`（`docs/arac-oc-recovered-baseline-freeze.md`）；
- 只执行被路由选中的一个动作，不试跑其余动作；所有候选由真实目标评价，
  strict-best archive 永不回退；算子异常 fail-closed，不静默改派。

## 使用方式

升级 lane 的候选入口（Phase-I v10 + 结构路由 + 单动作执行）：

```python
from experiments.upgrade.soft_rddsm_structural_router_v1 import (
    run_and_execute_soft_rddsm_structural_route,
)

result = run_and_execute_soft_rddsm_structural_route(
    problem,
    run_seed=117,
    action_seed=9117,
)
print(result.action_result.action_name, result.action_result.route)
```

命令行冒烟：

```powershell
$env:PYTHONPATH = '.;src'
.venv\Scripts\python.exe -c "from arac.benchmarks.aob import AobBenchmark; from experiments.upgrade.soft_rddsm_structural_router_v1 import run_soft_rddsm_structural_router; print(run_soft_rddsm_structural_router(AobBenchmark().load('S3'), run_seed=117).decision)"
```

该入口属于 upgrade lane，不改变生产 selector、生产 action registry 或
默认入口；单次冒烟结果不能当作 gate。

OC 统一循环入口 `arac.run_arac_oc`（Phase-I soft-RDDSM + 两特征派发 +
仲裁接线，`docs/arac-oc-design.md`）仍在维护。注意必须显式传
`phase1_kwargs`：默认 pilot 参数是小维度设计遗留，在 1000 维 AOB 上
screening 预算（315,040 FE）超过 Phase-I 180,000 FE 上限并直接崩溃；
blessed 配置（screening 121,020 FE）见
`experiments/aob24_overlap_applicability_audit.py`：

```python
from arac import run_arac_oc

result = run_arac_oc(
    problem,
    total_budget_fes=3_000_000,
    run_seed=20260815,
    phase1_kwargs={
        "anchor_count": 5,
        "step": 0.25,
        "rounds": 12,
        "bucket_size": 16,
        "max_candidate_pairs": 128,
    },
)
```

`run_overlap_arac` / `run_overlap_from_pilot` 只保留为历史/对照臂；
`run_arac_oc_v5_1/v5_2/v5_3` 具名入口与 `scheduler_mode` 参数已随 cut-2
删除，v5.x 调度行为只能从 tag `v5.3-prealation` 复现。

## 升级与研究状态

- **升级总纲**：`docs/arac-oc-shared-patch-completion-plan.md`——在
  Phase-I、路由和外层动作全部不变的前提下，只在 CTP/GSS 内部挂载
  stateful shared-patch kernel（候选族 / `z,u,r` 持久状态 / 局部
  context hash / 固定 8 FE lane）。gate 顺序
  `B0 -> B1 -> B2 -> B3 -> M0 -> M1 -> M2 -> AOB preservation -> production E2E`。
- **恢复优先协议**：`docs/arac-oc-recovery-first-protocol.md`。当前
  B1-Screen 120 臂零契约失败，但 displayed-mean screen 未通过（AOR 4/6、
  SMP 1/6、GCB 0/6、CTP 5/6），属诊断性恢复失败；B1-Final 与创新实验
  在隔离该动作级回归前保持阻塞。
- **已关闭的路线**：一致性分类（Gate 54a 不可辨识性裁决）、阶梯式
  S1 杠杆扫掠序 / S2 传播接续（三重证据证伪）、九个 shared-transaction
  升级候选（v1-v8 + E1）全部按冻结协议判定未通过。
- **v6.0 双轴状态设计**：预注册未实现（`docs/arac-oc-v6_0-design.md`）。
- `UNCALIBRATED_FIELDS` 中的阈值尚未通过独立校准门；当前代码验证的是
  契约、FE 对账、严格接受和确定性，不构成跨 benchmark 性能结论。

## 代码结构

- `src/arac/evidence/`：Phase-I 证据（`soft_rddsm.py` 发现、
  `soft_rddsm_adapter.py` sidecar、`hierarchical.py` 三层证据 schema、
  `variable_signature.py` 签名）。
- `src/arac/analysis/structural_router.py`：非学习结构路由。
- `src/arac/actions/`：四个 Phase-II 动作与 `recovered_registry.py`
  （恢复基线锚点）。
- `src/arac/coordination/`：OC 统一循环、planner/operators、
  `shared_patch.py`（kernel）、state v2 与 receipt v2。
- `src/arac/dispatch_policy.py`：Gate 41a 两特征派发规则（C3 口径）。
- `src/arac/overlap_core.py`：`run_arac_oc` OC 入口及历史对照臂。
- `src/arac/runtime/`：严格 FE 账本、checkpoint 和优化器端口。
- `experiments/upgrade/soft_rddsm_mainline_v10/`：Phase-I v10 组合管线。
- `experiments/upgrade/soft_rddsm_structural_router_v1/`：结构路由候选入口。
- `experiments/historical_recovery/`：recovery-first 战役（B0-B3）与
  恢复基线 freeze verifier。
- `docs/arac-oc-shared-patch-completion-plan.md`：升级总纲；
  `docs/arac-oc-recovery-first-protocol.md`：恢复 gate 定义；
  `docs/arac-phase1-v10-design.md`：Phase-I v10 设计与前置 Gate。

## 安装与验证

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests experiments
.venv\Scripts\python.exe -m compileall -q src experiments
.venv\Scripts\python.exe -m pip check
```
