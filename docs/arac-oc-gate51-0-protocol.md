# Gate 51-0 协议：显现 horizon 测量（先测后冻结）

日期：2026-08-17
状态：预注册（v4 升级计划 §2）
前置：v4 计划已定稿；Gate 50/50c 收据在库。

## 1. 研究问题

v4 的窗口参数（maturity 窗基数 w1、递增级数 K、开发上限、
exploitation 保底）不允许拍脑袋。两个必须实测的事实：

> (a) R2/aor：282k 探针窗 global/local 双零——它的价值显现需要
>     多长的累计窗口？
> (b) S5/ctp：全局边际 0.02、私有轨迹 1.30——它的私有轨迹何时
>     越过其它臂的探针期水平？

## 2. 设计

- 两次 3M 分段 instrumented standalone 重跑（v2 episode 机制，
  不依赖进度契约，不修改调度器）：
  - aor @ R2 checkpoint（seed 20260845，Gate 50 缓存 checkpoint）；
  - ctp @ S5 checkpoint，同规格。
- 分段：150k FE × ~19 段；每段落盘 requested/consumed FE、
  cumulative FE、error_before/after、state_hash。
- checkpoint 哈希与 Gate 50 standalone cell 逐位核验（漂移即失败）。
- 脚本与产出目录带 implementation_manifest_hash（覆盖调度器、
  state、runtime phase2、本协议与本脚本）。

## 3. 对账判据（全部成立才有效）

1. `final_error` 与 Gate 50 standalone 收据**逐位相等**（分段 ≡
   单发已由 recovered 等价性测试与 Gate 50a 证明；不等 = gate
   fail，如实报告，不得继续定标）；
2. `terminal_fes` 精确 = 3,000,000；
3. 轨迹单调：best_error 逐段非增；
4. 分段 FE 之和 = Phase-II 预算（3M − 180k）。

## 4. 分析产出（定标表，写入本文件附录后冻结）

- 每 run 的 err(cumulative FE) 轨迹表；
- 显现 horizon `h*(E)` = 轨迹首次严格低于参考水平 E 的累计 FE；
  E 取：(i) 同 case 其它臂在 50c ON 探针期末的 global_error_after；
  (ii) 该 case OC 探针期 archive 水平；
- 定标规则（累计覆盖）：冻结 w1、r=2、K，使
  `w1·(2^K − 1) ≥ h*_cal × 安全系数`（h*_cal 跨 run 取最大；
  全局常数，禁止按 case 配置）；
- `exploration_and_development_cap` 与 exploitation 保底比例由
  定标表一并给出；
- 已知局限声明：AOR standalone horizon ≠ handoff 后 contextual
  horizon（archive 注入改变私有基线）；定标采用 standalone
  曲线，偏差方向与幅度记录在案；
- 若累计覆盖在剩余预算内不可达 → 如实记录，v4 以 BIPOP 式尾部
  预留为后备杠杆（版本隔离流程），不扩冷启动税。

## 5. 约束

- 只读 Gate 50 缓存 checkpoint 与 50c 收据 + 两次新跑；
- 只作参数定标，不构成任何性能结论；
- 产出目录：`artifacts/oc_horizon_gate51_0/`（cells/ +
  confirmation.json）。

## 6. 运行方式

```powershell
.venv\Scripts\python.exe -m experiments.oc_horizon_gate51_0 --workers 2
```

## 附录：测量结果与定标表（2026-08-17 冻结）

**对账**：两 run 终值与 Gate 50 standalone 收据逐位相等（aor/R2 =
221,181.93…，ctp/S5 = 13,244.70…），terminal 精确 3M，轨迹单调，
分段 FE 对账精确。gate_passed = true。

**出处注记（诚实记录）**：两 run 执行期间 v4 调度器在同仓实现，
执行时点的 manifest 不可复原；cell 统一盖跑结束时的 manifest，
数值有效性由逐位对账锚定（强于 manifest 戳）。脚本已改为 main
单点计算 manifest 传递 worker，消除该竞态类。

**轨迹与穿越**（150k 粒度）：

| run | 150k | 300k | 450k | … | 2820k | 参照线 E | 穿越 |
|---|---|---|---|---|---|---|---|
| aor/R2 | 2.14e6 | **499,156** | 320,849 | … | 221,182 | 512,640（探针期 archive） | (150k, 300k] |
| ctp/S5 | 6.90e7 | **3.68e7** | 8.78e6 | … | 13,244.7 | 43.62e6（探针期 archive） | (150k, 300k] |

关键事实：两晚熟臂在 150k 累计时都**差于**探针期 archive
（aor 2.14e6 vs 5.13e5；ctp 6.9e7 vs 4.4e7）——任何 <300k 的
累计窗口都会把它们判为无价值，v3 的 282k 探针失败由此完全解释。

**定标推导**：

- h*_measured = 300,000 FE（两 run 一致，150k 粒度下的保守值）；
- 安全系数 1.5：覆盖测量粒度 + AOR contextual horizon 偏移
  （archive 注入后基线更低，穿越更晚；standalone 轨迹推算
  contextual 穿越 ≈ 375-450k）；
- 目标累计覆盖 = 450,000 FE；
- w1 = 75,000，r = 2，K = 3：75k+150k+300k = 525k ≥ 450k ✓；
- exploration_and_development_cap = 0.55——R2 全修复路径算术：
  gss ticket 639k + aor ladder 525k + smp ladder 375k = 1.54M
  ≤ 0.55×2.82M = 1.55M ✓；exploitation 保底 0.10；
- 已知局限：h* 来自 2 case × 2 episode 测量，A3/R6 未测（其
  晚熟臂 282k 窗口即可见，450k 上限覆盖）；fresh seed 门（51c）
  是对此定标的独立检验。

**冻结表**（`artifacts/oc_horizon_gate51_0/calibration.json`）：

| 参数 | 值 |
|---|---|
| maturity_window_fes (w1) | 75,000 |
| revelation_horizon_fes | 450,000 |
| escalation_factor | 2 |
| escalation_grants_k | 3 |
| exploration_and_development_cap | 0.55 |
| exploitation_reserve_ratio | 0.10 |
| cold_start_probe_cap | 0.25 |
| probe_min_fes | 20,000 |
| calibration_ref | gate51-0-20260817 |
