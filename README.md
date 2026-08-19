# ARAC-OC：面向变量重叠的大规模全局优化

ARAC-OC 是本仓库当前唯一的生产方法。它用 Phase-I 的身份盲证据发现重叠结构，
再由 GCB coordinator 处理共享变量冲突，而不是把证据降级为普通 sweep 顺序。

```text
Phase-I overlap evidence
  -> GCB coordinator: component / scope / budget / episode plan
  -> scoped CTP | GSS | SMP | AOR episodes
  -> real-objective strict-best arbitration
  -> CoordinatorState feedback
```

GCB 是协调器，不是第五个搜索动作；CTP、GSS、SMP、AOR 是四个被调度的搜索 episode。
历史 receipt 中 GSS 仍使用 `episode="gcb"`，并由 `episode_kind="gss"` 和
`episode_names` 映射保持兼容。四个 episode 不会被
独立 selector 或 RF 模型一次性选定。每个 OperatorPlan 都携带明确 scope、预算、
动作类型和状态收据，所有评价由同一个 `EvaluationLedger` 计费。

## 使用方式

完整的 ARAC-OC 入口是 `arac.run_arac_oc`：

```python
from arac import run_arac_oc

result = run_arac_oc(
    problem,
    total_budget_fes=3_000_000,
    run_seed=20260815,
)
```

v5.1 四-episode 调度器通过显式模式接入，避免把旧 `oc_unified` 收据误读为
phase-aware 结果：

```python
from arac import run_arac_oc_v5_1
from arac.coordination.episodes import PhaseAwareSchedulerConfig

result = run_arac_oc_v5_1(
    problem,
    total_budget_fes=3_000_000,
    run_seed=20260901,
    scheduler_config=frozen_gate51_config,
)
```

`run_arac_oc(..., scheduler_mode="legacy_unified")` 保留旧循环的可复现兼容路径；
`scheduler_mode="v5_1"` 返回带 `scheduler_version="v5.1"`、`coordinator_name`
和 `episode_names` 的阶段感知 schedule receipt。

`run_overlap_arac` 只保留为兼容包装；`run_overlap_from_pilot` 与
`gcb_coordinated` 只用于历史/控制实验，不能作为当前方法定义。

## 当前边界

- Phase-I 当前协议为 `arac-identity-blind-evidence-v9`，正式 3M FE 配置下使用
  `180000 FE` 并产生 40 个身份盲特征。
- Phase-I 必须输出变量级 groups、owner membership 和成员置信度；只有适配器 ready
  才能进入协调器，结构证据不完整时 fail-closed。
- AOR 是 GCB 预注册的正常升级动作，不是算子异常后的隐式 fallback。
- 算子异常产生 `operator_failed` 收据并显式终止；不会静默改派另一个动作。
- `UNCALIBRATED_FIELDS` 中的阈值和 pulse 参数尚未通过独立校准门，当前代码验证的是
  契约、FE 对账、严格接受和确定性，不构成跨 benchmark 性能结论。

## 代码结构

- `src/arac/evidence/`：Phase-I 重叠发现、checkpoint 和适配门。
- `src/arac/coordination/`：协调器、GCB planner、四个 operator、探针和状态反馈。
- `src/arac/overlap_core.py`：`run_arac_oc` 唯一生产入口及兼容包装。
- `src/arac/runtime/`：严格 FE 账本、checkpoint 和优化器端口。
- `docs/arac-oc-design.md`：当前方法契约、预算和失败语义。
- `docs/arac-core-method.md`：历史 ARAC-Core 记录，不是当前生产规范。

## 安装与验证

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests experiments
.venv\Scripts\python.exe -m compileall -q src experiments
.venv\Scripts\python.exe -m pip check
```
