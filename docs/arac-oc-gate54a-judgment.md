# Gate 54a 判定与一致性不可辨识性发现

日期：2026-08-22 凌晨（自主夜间执行）
上游：`docs/arac-oc-completion-plan.md`（v2）§5 Gate 54a。

## 判定：Gate 54a 未通过（分析性不可辨识 + 三次仪器 pilot 证据）

按 v2 方案止损线：**双重叠升级冻结**，主线退回"Phase-I 发现 + 证据派发 +
循环语境仲裁（Gate 29/30）"，论文按既有证据 + 本发现撰写。

## 发现 F1：聚合目标上的一致性不可辨识性（可陈述为命题）

设 f(x) = Σ_g w_g·b(x_{G_g} − o_g)（可加、各组同基 b、共享变量属于多个
G_g）。对任一共享变量 v，给定其余坐标的任意上下文 c：

  f(v | c) = Σ_{g∋v} w_g·b(v − o_{g,v}) + const(v 无关项)

即 v 的条件剖面是 owner 项的**对称聚合**——不存在任何黑盒查询能区分
"owner 们 optima 相同（conforming）"与"optima 不同（conflicting）"，
因为两者在 wide 参数族上产生**相同的可观测剖面**（sphere 基下精确相同：
Σw_g(v−o_g)² = (Σw_g)(v−m)² + const，曲率同为 Σw_g，差异项不可观测）。

一致性是**分解内部参数**的属性；它的 manifestation 只存在于
分解层的 owner 条件化过程（而 owner 条件化在共享坐标上同样纠缠：
任一 owner 的坐标限制搜索优化的仍是含 co-owner 项的同一函数）。

## 证据链（三次声明的仪器 pilot，artifacts/oc_gate54a_pilot/）

| 仪器 | 机理 | 结果 |
|---|---|---|
| v1 cross-context | owner-pull 使偏置随共动上下文变化 | **方向反**（conforming 0.071 > conflicting 0.003）；可分目标下按构造为零 |
| v2 scale-instability | 双井使偏置跨尺度翻转 | 重叠不分离（0.87 vs 0.79）；被 incumbent 位置效应淹没 |
| v3 owner-calibration | 各 owner 块局部搜索读偏好值 | 分歧恒 0（分析性：两 owner 搜索同一限制函数，端点必然一致） |

三次迭代均在协议冻结前（pilot 声明角色），无判定数据泄漏。

## 对论文的影响（并入 §7 故事线）

1. "runtime 一致性分类"作为**可辨识性边界**陈述——解释了为什么文献中
   冲突处理都假设分解内部信息（CBCCO/OCC 用 DG2 已知结构），为什么
   Gate 53 的仲裁在无提案源时零接受；
2. ARAC-OC 的实际可辩护主张链不变：发现（C1/C2）+ 派发（C3）+
   循环语境仲裁（C4）+ 分析节（含本发现）；
3. v2 方案 §3 的 L2a/L2b 冻结为后续工作（需要分解内部信道的设计，
   例如把 owner 提案源引入派发路径——即已登记的 Gate 53b 方向）。

## 代码状态

- `src/arac/coordination/consistency.py`（仪器 v3，含失败机理注释）保留为
  研究工件；未接入任何生产路径；
- 第三刀前提（主线 = 派发 + 双支路，loop 线退役）被冻结失效——
  legacy_unified/loop.py 仍是循环语境仲裁（C4）的宿主，**不删**；
  gcb.py 维持现状（历史对照臂引用）。第三刀降级为文档清理。
