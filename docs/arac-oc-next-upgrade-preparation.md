# ARAC-OC 下一步升级准备

日期：2026-08-23  
前置锚点：`arac-recovered-baseline-20260823-v1`

## 升级目标

在不破坏已恢复四动作轨迹的前提下，继续解决大规模全局优化中的 shared-variable
overlap。第一升级候选只允许进入 CTP/GSS 内部，不改变 Phase-I、外层 selector、
SMP/AOR/GCB 的动作类型，也不把 AOB conforming case 当成 conflicting efficacy
主实验。

具体执行方案已升级为：

`docs/arac-oc-stepwise-upgrade-plan-v2.md`

文献核验记录：

`references/lit-review/arac_oc_stepwise_upgrade_literature_2026-08-23.md`

## 目录与命名

升级代码、协议和实验入口统一放在：

`experiments/upgrade/`

每个候选使用独立版本目录，例如：

```text
experiments/upgrade/shared_patch_v1/
experiments/upgrade/shared_patch_v1/protocol.json
experiments/upgrade/shared_patch_v1/README.md
```

候选不得覆盖 `src/arac/actions/` 中的冻结文件。需要修改生产挂载点时，先复制
到候选命名空间，通过 gate 后再以显式 promotion 变更冻结 manifest。

## 固定验证阶梯

### U0：Kernel contract

固定 8 FE patch lane，检查 exact FE、strict-best、候选边界、state hash、局部
context reset、snapshot/restore 和 fail-closed。失败即停止。

### U1：Matched-host reachability

使用 conflicting overlap generator，强制 CTP/GSS host；必须产生非零 patch receipt，
且不得出现隐式预算借用或全 `arbitration_only`。外层 selector 不参与。

### U2：Nested attribution

使用 A0-A4 嵌套消融，严格隔离候选增量、持久状态和自适应半径。主比较是 A4
相对 A1/A2，不接受只比较 patch on/off 的替代设计。

### U3：AOB preservation/no-tax

先运行冻结 baseline，再运行 patch-on candidate。AOB 只要求 selector/action route
不变、terminal FE contract 不变、conforming case 非劣；不要求 patch 在 AOB 上触发。

### U4：Production parity

只有 U0-U3 全通过，才允许在统一 loop 中做 patch-enabled 对照。任何失败都回退
到冻结 baseline，不修改 production 默认开关。

## 第一候选的范围

建议先做“CTP/GSS 内部 shared-patch kernel v1”：

- owner-conditioned、consensus、disagreement 三类候选；
- `(z,u,r)` 跨 episode 状态；
- 局部 context hash，排除本 patch 写集；
- conforming 变量 `base_radius=eps` 自动静默；
- `u` 只参与 scope 排序，不参与候选方向、动作选择或预算分配。

软路由暂不与 kernel 首版同时引入。先证明 kernel 的可达性和归因，再单独增加
连续活跃度，避免多个反馈回路同时变化。
