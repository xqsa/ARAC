# soft-RDDSM structural router v1

这是一个候选升级入口，不改变生产默认链路。

```text
soft-RDDSM Phase-I
  -> confirmed hyperedge -> variable-level overlap sidecar
  -> primary structural route (AOR / SMP / CTP / GCB)
  -> action-view checkpoint -> exactly one Phase-II action
```

主路由依据只有证据完整性、共享变量存在性和 owner graph 连通性；不使用 AOB
函数编号、最终误差标签或训练 selector。这里的主路由不是动作排他标签：只要变量级
证据完整，SMP 都保持结构兼容；无共享变量时记录 `zero_relation` 模式，有共享变量
时记录 `overlap_aware` 模式，即使主路由是 CTP 或 GCB。`Phase1V10Result.overlap_evidence`
和 `overlap_evidence_hash` 可用于后续 Phase-II matched-host 实验。

候选 dispatcher 使用 `action_view_checkpoint()` 将确认的 owner graph 投影为
`PhaseCheckpoint.relations`：AOR 保留不完整证据，SMP 清除 incidental relation，
CTP/GCB 使用 sidecar 诱导的 disconnected/connected relation graph。原始 Phase-I
checkpoint hash 与动作 view hash 分开记录，incumbent、Phase-I FE 和 terminal FE
边界保持不变。

执行一个候选动作：

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

该入口只属于 upgrade lane，不改变生产 selector、生产 action registry 或默认入口。

运行入口：

```powershell
$env:PYTHONPATH = '.;src'
.venv\Scripts\python.exe -c "from arac.benchmarks.aob import AobBenchmark; from experiments.upgrade.soft_rddsm_structural_router_v1 import run_soft_rddsm_structural_router; print(run_soft_rddsm_structural_router(AobBenchmark().load('S3'), run_seed=117).decision)"
```

大规模 AOB 冒烟可能需要较长时间；正式性能结论必须使用独立协议和 fresh seeds，
不能把本候选入口的单次结果当作 gate。
