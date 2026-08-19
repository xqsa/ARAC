# AOB vendor 性能补丁（2026-08-17）

范围：vendor/aob/AOB/Benchmarks.py（transform_asy/elliptic/Lambda 常量
缓存、rotateVectorConform 索引缓存）+ 四个函数文件的 fitness_record
默认关闭（record_fitness 开关）。

数值等价证据：golden_reference.npy（4 case × 1/8/24 行批次 + 单行，
固定 seed 20260817）——补丁前后逐位一致；守卫测试
tests/test_vendor_perf_patch_bitwise.py 持续锁定。

提速：单进程吞吐 2,713 → 6,491 FE/s（2.4×）。
内存：fitness_record 关闭省 ~100MB/worker（3M FE）。
campaign_manifest 的 vendor_trees 修订见该文件 amendments 记录。
