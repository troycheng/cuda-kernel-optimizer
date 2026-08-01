# Task 3: Nsys collect 实现记录

## 范围与边界

- 只修改 `profile_nsys.py` 和其聚焦测试；未修改 `analyze` 输入契约、
  `workload_adapter.py`、存储层或其它 Profiler。
- `collect` 在提交 Invocation 前调用
  `resolve_profile_collection(..., capability="nsys_wrap_v1")`，并核对 Target
  固定的主机/GPU、`nsys` 绝对路径与 SHA-256。candidate 缺少 Experiment 或通过的
  correctness receipt 时直接拒绝。
- worker 获取 GPU 后再次解析全部不可变绑定；只经 adapter 物化输入、构建 driver
  request 和 argv。没有增加第二条 workload 路径、调度器或自动决策。

## 采集与证据

- 仅接受 `nsys 2026.2.x`；每次使用前重新校验固定 executable 的内容摘要。
- profile argv 固定为 `profile --trace=cuda,nvtx,osrt --sample=none
  --cpuctxsw=none --stats=false --wait=all --output <prefix> <driver argv>`。
- 只接受一个 `<prefix>.nsys-rep`，先冻结并以 `.nsys-rep` 后缀物化，再严格执行
  `nsys export --type sqlite --output <unique>.sqlite <frozen>.nsys-rep`；SQLite
  冻结后以 `.sqlite` 后缀物化，并只交给 `parse_nsys_sqlite()`。
- 结果保留绑定、命令回执、tool/parser identity、driver/raw-report/SQLite 的不可变
  对象引用、耗时和实际 cleanup 状态。任一失败均保持 observations 为空且无效。

## 冻结上限

- driver 输出：16 个文件、16 MiB、5 秒，和结构化 driver result 的小型诊断输出相匹配。
- Nsys `.nsys-rep` 与 SQLite：单文件、8 GiB、120 秒。8 GiB 与现有
  `_MAX_SQLITE_BYTES` 对齐，适用于单 case 采集，同时避免把 Nsys 报告错误地限制为
  NCU 的 16 MiB。没有改动 `artifact_store.py`；更大对象的流式处理仍留在既定存储收口阶段。

## TDD 与验证证据

1. 先新增公开成功采集与候选前置拒绝测试；实现前运行：
   `python3 -m unittest tests.test_profile_nsys.NsysAnalyzeTests.test_public_collect_wraps_the_only_driver_argv_and_freezes_raw_facts tests.test_profile_nsys.NsysAnalyzeTests.test_candidate_collect_missing_correctness_rejects_before_invocation_or_nsys`
   两项均按预期失败，原因是 CLI 尚未提供 `collect`。
2. 实现后运行同一命令：2/2 通过。
3. 聚焦回归：`python3 -m unittest tests.test_profile_nsys tests.test_workload_adapter`：17/17 通过。
4. `python3 -m py_compile skills/cuda-kernel-optimizer/scripts/profile_nsys.py`：通过。
5. `git diff --check`：通过。
