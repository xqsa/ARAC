# ARAC-OC：面向变量重叠的大规模全局优化

ARAC-OC 的当前框架是 **soft-RDDSM 结构发现 + 变量级重叠证据 sidecar +
非学习结构路由 + 四动作 Phase-II**：重叠不再只是测试函数的标签，而是被
显式编码进变量证据和图结构中，由确定性路由规则映射到恰好一个 Phase-II
动作，在总预算下输出可审计的终值与 receipt。

```text
目标函数、变量边界、随机种子、总 FE 预算
        ↓
Phase-I 预算预留
        ↓
soft-RDDSM 结构发现器
        ↓
确认的 hyperedge
+ 变量级 overlap evidence sidecar
+ overlap graph
+ evidence completeness
        ↓
非学习、确定性的结构路由器
        ↓
AOR / SMP / CTP / GCB
        ↓
从 Phase-I checkpoint 继续执行 Phase-II
        ↓
3,000,000 FE 下的最终最优解和完整 receipt
```

## 1. Phase-I：soft-RDDSM 发现结构

当前 Phase-I 使用 soft-RDDSM 做递归结构发现，经过 signature、warmup、
DSM/RD 类探测后，输出三类信息：

- `confirmed hyperedges`：确认存在内部交互的变量集合；
- 变量级状态：例如 `member_candidate`、`observed_separable`、
  `not_yet_resolved`；
- 组件 overlap graph：节点是组件，若两个组件共享变量则连边。

只有确认的 hyperedge 才能把变量加入额外的 owner group；区域级交互
没有确认 hyperedge 时，证书标记为不完整，而不是静默变成重叠隶属。
sidecar schema 为 `arac-soft-rddsm-overlap-evidence-v1` 并携带 hash。

## 2. Evidence contract：先判断证据是否完整

只有当所有组件和变量都已经被解析，并且不存在 `not_yet_resolved` 状态时，
才能设置：

```text
evidence.complete = True
```

`resolved_hyperedges` 是否为非空 tuple 不能代表证据完整。

当前实验中 Phase-I 通常限制为 `180,000 FE`，soft-RDDSM 的 discovery FE
必须属于这个上限，而不是在 discovery 完成以后再补检查。

## 3. 非学习结构路由

| 证据状态 | 动作 | 作用 |
|---|---|---|
| 证据不完整 | AOR | 保守探索，避免基于错误结构做强路由 |
| 证据完整且无共享变量 | SMP | 独立处理各可分离组件 |
| 证据完整且 overlap graph 不连通 | CTP | 分别处理多个独立 overlap cluster |
| 证据完整且 overlap graph 连通 | GCB | 对全局耦合结构进行联合优化 |

这里要特别区分：

SMP 的路由条件是"没有跨组件共享变量"，不等于"函数内部完全没有交互"。
一个组件内部可以有 hyperedge，仍然可以使用 SMP。历史记录中 SMP 也可能
作为通用动作或 fallback 作用于部分重叠问题，所以不能仅凭动作名称推断
该函数没有 overlap。

路由只依据证据完整性、共享变量存在性和 overlap graph 连通性；目标
尺度、benchmark 名称、拟合 selector、在线增益信号都不进入决策。
实现见 `src/arac/analysis/structural_router.py`；`src/arac/dispatch_policy.py`
另保留一条 Gate 41a 校准的两特征规则（`tail_log10_gain` +
`structural_relation_density`）作 C3 口径对照，同样是确定性规则。

## 4. Phase-II：动作执行

路由器把结构证据和 overlap graph 传给 `ActionRegistry`，然后从同一个
Phase-I checkpoint 继续执行对应动作。Phase-II 固定：

- action registry；
- boundary profile；
- warmup/continuation schedule；
- 剩余 FE 预算。

最终在总预算下输出最优解、动作轨迹、路由原因、结构证据和 FE receipt。
只执行被路由选中的一个动作，不试跑其余动作；所有候选由真实目标函数
评价，strict-best archive 永不回退；算子异常 fail-closed，不静默改派。
恢复基线 `arac-recovered-baseline-20260823-v1` 以
`RecoveredActionRegistry`（`src/arac/actions/recovered_registry.py`）为
冻结锚点，生产默认 `patch=false`、`soft_routing=false`、`selector=false`。

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
  stateful shared-patch kernel。gate 顺序
  `B0 -> B1 -> B2 -> B3 -> M0 -> M1 -> M2 -> AOB preservation -> production E2E`。
- **恢复优先协议**：`docs/arac-oc-recovery-first-protocol.md`。当前
  B1-Screen 120 臂零契约失败，但 displayed-mean screen 未通过（AOR 4/6、
  SMP 1/6、GCB 0/6、CTP 5/6）；B1-Final 与创新实验在隔离该动作级回归前
  保持阻塞。
- **已关闭的路线**：一致性分类（Gate 54a 不可辨识性裁决）、阶梯式
  S1 杠杆扫掠序 / S2 传播接续、九个 shared-transaction 升级候选
  （v1-v8 + E1）。
- **v6.0 双轴状态设计**：预注册未实现（`docs/arac-oc-v6_0-design.md`）。
- `UNCALIBRATED_FIELDS` 中的阈值尚未通过独立校准门；当前代码验证的是
  契约、FE 对账、严格接受和确定性，不构成跨 benchmark 性能结论。

## 代码结构

- `src/arac/evidence/`：Phase-I 证据（`soft_rddsm.py` 发现、
  `soft_rddsm_adapter.py` sidecar、`hierarchical.py` 证据 schema、
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
