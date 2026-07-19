# exp_019 受控冲突基准构造说明

Date: 2026-07-18
Executor: Codex
Status: synthetic diagnostic benchmark；非官方 AOB 案例

## 1. 用途与声明边界

本基准为 AOB E3/A4/S5 的受控 synthetic conflict 孪生体，只用于检验共享变量
取值冲突是否为 reliability-weighted bridge 提供可观测的即时 objective 收益。
它不是官方 AOB benchmark，不代表完整 optimizer 性能，也不授权 runtime
writeback、动态 regrouping 或 HCC 性能声明。

第 0.5 步只构造并验证基准，不运行任何新的 optimizer FE。后续 oracle 诊断和
正式 pilot 必须分别通过用户 go/no-go。

## 2. 只读来源与隔离方式

实现位于：

- `experiments/pilots/exp_019_conflict_resolution_pilot/benchmark.py`
- `experiments/pilots/exp_019_conflict_resolution_pilot/data/`

`ConflictBenchmarkFactory` 产生的三个类分别子类化 vendor elliptic、ackley 和
schwefel。adapter 只替换两件事：

1. 从 exp_019 CSV 加载按子组展开的 `OvectorVec`；
2. 在 exp_019 内执行 batch-safe `rotateVectorConflict`，即使用
   `candidate[:, indices] - OvectorVec[group]`。

其余 topology、Pvector、rotation、design、weights、基础 xopt 和 objective
transformation 都从 `vendor/hcc/AOB/AOBG/datafile` 原位读取。没有复制约 60 MB
的 design 文件，也没有修改 `vendor/hcc` 或 `E:\HCC-main`。

## 3. 确定性构造

CSV 字段顺序固定为：

```text
variant_id,group_index,local_index,global_variable_index,base_optimum,conflict_optimum,is_shared
```

索引均为零基。先根据 vendor `Pvector` 和每组大小，将 1000 维官方 xopt 投影到
每个局部子组。对全局变量 `j` 的基础最优值 `v_j`：

```text
non-shared:  v_conflict = v_j
left owner:  v_conflict = v_j + rho * (lower - v_j)
right owner: v_conflict = v_j + rho * (upper - v_j)
rho = 0.10
```

三个案例的边界均为 `[-100, 100]`，因此相邻 owner 的两个局部最优值相距 20，
即完整搜索区间的 10%。每个值是 `v_j` 与一个边界的凸组合，所以不会越界。
非共享变量完全不变。

| Variant | 全局变量 | 局部 CSV 行 | 共享全局变量 | 共享局部副本 |
| --- | ---: | ---: | ---: | ---: |
| `E3_conflict_variant_synthetic` | 1000 | 1057 | 57 | 114 |
| `A4_conflict_variant_synthetic` | 1000 | 1095 | 95 | 190 |
| `S5_conflict_variant_synthetic` | 1000 | 1133 | 133 | 266 |

## 4. 零偏移等价性

当 `rho=0` 时，每组 `OvectorVec[i]` 就是全局 xopt 在该组 `Pvector` 索引上的投影。
因此对任一候选 `x` 和子组 `i`：

```text
conform:  (x - Ovector)[P_i]
conflict: x[P_i] - OvectorVec[i]
```

两式逐元素相同，之后复用的 rotation、transform 和 objective 也相同。聚焦测试
对 E3/A4/S5 的 1D 和 batch 输入执行了数值比较，三者均通过。这一性质把
conform/conflict 的受控差异锁定为共享变量局部最优值，而不是拓扑或函数实现。

## 5. 完整性与哈希清单

唯一的机器可读事实源是：
`experiments/pilots/exp_019_conflict_resolution_pilot/data/conflict_variants_manifest.json`。
manifest 固定 generator version 和 `rho`，并逐文件绑定 AOBG 数据包 ID
3/4/5 的 info、xopt、Pvector、subgroup sizes、weights、rotation 和 design
文件（相应磁盘文件仍使用 `F3/F4/F5-*` 前缀）。加载任一 variant 前会
验证整个 bundle；缺失、重复、schema 错误或任一 hash 不一致均直接失败。

| Synthetic CSV | SHA256 |
| --- | --- |
| `E3_conflict_variant_synthetic.csv` | `a82f89e2c9c229ce498b8dde8b65f448eca635f08b9be01bb7bbec904b54e6c2` |
| `A4_conflict_variant_synthetic.csv` | `a327c3323213507b1f1e8852556a5c5b615d8850984ef312c775804c02ec4235` |
| `S5_conflict_variant_synthetic.csv` | `50f3f5740f6e5b9ddfed25c47ce4c594012e0145caecf28d9772e187d7e963ec` |

不在本文重复 30 个 vendor 文件 hash，以避免产生第二事实源；完整清单及相对路径
均在上述 manifest 中。生成器输出不含时间戳，同一 vendor 输入上逐字节确定。

## 6. 验证入口

```powershell
.\.venv\Scripts\python.exe -m experiments.pilots.exp_019_conflict_resolution_pilot.benchmark --check
.\.venv\Scripts\python.exe -m pytest -q tests/test_exp_019_conflict_benchmark.py
```

当前聚焦结果为 `12 passed`。测试覆盖零偏移等价、共享变量唯一变化、边界、
1D/batch 与 `fitness_record`、确定性生成，以及缺失/重复/篡改输入的 fail-closed
行为。CEC 2013 f13/f14 未纳入本阶段。
