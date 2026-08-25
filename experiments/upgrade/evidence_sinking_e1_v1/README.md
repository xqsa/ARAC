# evidence_sinking_e1_v1 — CTP tail reserve 的证据条件化

基线：`arac-recovered-baseline-20260823-v1`（冻结，不动）。
计划出处：`docs/arac-oc-evidence-sinking-plan-v4.0.md` §E1（2026-08-23 预注册）。

## 机制（冻结于 screen_protocol_v1.json）

冻结 CTP 对正关系 run 一律给固定 20% MMES 尾。E1 把尾份额条件化在
checkpoint 静态关系证据上：

```text
mean_strength = Σ strength / 关系数
norm          = min(1, mean_strength / 0.08)   # 0.08 = S 族均值强度上端（输入校准）
tail          = 0.20 × (1 + norm)              # 单调有界 [0.20, 0.40]
```

实现 = 运行前替换 `ctp_module._POSITIVE_RELATION_MMES_TAIL_FRACTION`
后恢复；会话行为/顺序/零关系路径不动（常量只在正关系时被读 →
S1 双臂按构造逐位恒等）。A0 不动常量。

## 为什么是 E1（证据链）

- 尺度公理的正面侧：本项目全部历史真实收益都在"预算所有权 × 静态结构
  证据"层（G1/G2/G3/G4/G5/G8），shared_transaction v1–v8 在协调机制层
  的八连负则刚穷尽了反面。
- G3（未消耗的最大正收益点）：CTP S6 matched checkpoint tail ablation
  3/3 胜、geo 0.2897；v5.0 T0 用全新 seeds 复验 S6 R=0.419——尾部有强
  因果价值，而 S6 的固定 20% 可能不足（screen ratio 1.264 未恢复）。
- 判据（计划原文）：S6 全新配对 seeds geo ≤0.98；S2-S5 非劣
  CI 上界<1.05；S1 零关系逐位恒等。

## 判定口径

配对 bootstrap 10k、**exp 尺度**比较（v7 事件后全线修正的口径）；
seeds 20270601-05（从未用于任何 S-case 运行）。
