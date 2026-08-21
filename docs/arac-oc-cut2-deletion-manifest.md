# 代码第二刀：调度机制删除清单（预注册草案）

日期：2026-08-21
状态：**草案，待 Gate 53 判定后执行**（总纲 §9 依赖 #6 ← #4）。
依据：总纲 §0 裁决（调度线降级为分析资产）+ §7 第二刀纪律
（预注册清单 → 执行 → 全量回归 → 判定）。快照锚点：tag `v5.3-prealation`。

## 引用分析（2026-08-21 实测）

- 主线（Phase-I soft-RDDSM + 两特征派发 + 仲裁接线）**不引用** episodes.py
  的任何调度机制，也不引用 gcb.py。
- episodes.py 调度机制的 src 引用点：仅 `overlap_core.py`（`run_arac_oc` 的
  v5_1/v5_2/v5_3 分支与 `run_arac_oc_v5_*` 具名入口）。
- gcb.py 引用点：`overlap_core.py:444`（`run_overlap_from_pilot` 历史对照臂）、
  `coordination/__init__` 导出、31 个冻结实验脚本（不动，按收据锚定）。
- 测试覆盖：`test_oc_episode_schedule_{v4,v5,v5_3}.py`、
  `test_oc_episode_schedule_gate50b.py`（随机制删除）；
  `test_oc_unified_loop.py` 属 legacy_unified 线（第三刀范围，保留）。

## 删除清单（第二刀范围 = episodes.py 调度机制，gcb.py 留给第三刀）

1. `src/arac/coordination/episodes.py`：
   - 删除 `run_oc_episode_schedule_v4/v5/v5_1/v5_2/v5_3` 及其全部私有机制
     （P0-P5 adjudication ladder、maturity ticket、challenger rotation、
     escalation ladder、HPR、verification ladder、adoption grace、
     v4/v5 审计面与全部配套 dataclass）；
   - 目标 ≤1,000 行；保留仍被非调度代码引用的符号（执行时以 import
     报错为零容忍标准逐个裁定）。
2. `src/arac/overlap_core.py`：
   - 删除 `run_arac_oc` 的 `scheduler_mode` 参数与 v5 分支、
     `run_arac_oc_v5_1/v5_2/v5_3` 具名入口（v5_1/v5_2 分支现为退役 raise，
     一并删除）；`run_arac_oc` 只保留 legacy_unified 语义。
3. `src/arac/coordination/__init__.py`：移除随之失效的导出。
4. `tests/`：删除上列 4 个调度测试文件（tag 里有完整历史）。

## 执行协议

1. Gate 53 判定落地后执行；判定失败不改变本清单（调度降级与 Gate 53
   结果无关，仲裁去留只影响主线接线）。
2. 顺序：episodes.py → overlap_core.py → __init__ → tests，每步后
   `python -m pytest -q`（全量）+ `ruff check` 全绿才算该步完成。
3. 验收：主线 import 图不变（Phase-I/dispatch/actions/arbitration wire
   无新增 import 错误）；冻结实验脚本按 §7 纪律不要求可重跑。
4. 删除后 `git commit`（信息注明 cut-2 与清单文档哈希）。
