# 公开操作请求格式

本页用于构造随 skill 安装的公开工具请求。先运行目标脚本的 `--help` 确认 operation，
将一个 JSON 对象写入文件，再通过 `--request <path>` 传入。需要等待长操作完成时加 `--wait`；
否则保存返回的 `invocation_id`，之后调用同一脚本的 `status` 或 `cancel`。

所有输入都是封闭对象：不得增加未列出的字段，不适用的可选字段直接省略。`target_ref`、
`experiment_ref`、`baseline_ref`、`result_ref` 和 `selection_ref` 必须直接使用上一项操作返回的
引用，不能根据文件名猜测。Target、Experiment、Invocation result 和 Selection 引用中的
`sha256` 是对应 JSON 记录的内容摘要；`report_ref.sha256` 是冻结报告本身的内容摘要。时间戳
使用 Unix epoch 秒。

## 公共片段

```json
{
  "target_ref": {"id": "<target id>", "sha256": "<64 hex>"},
  "experiment_ref": {"id": "<experiment id>", "sha256": "<64 hex>"},
  "invocation_ref": {"invocation_id": "inv-...", "sha256": "<64 hex>"},
  "resources": {"host_id": "<frozen host id>", "gpu_uuids": ["GPU-..."]},
  "runtime_limits": {
    "operation_timeout_seconds": 600,
    "command_timeout_seconds": 300,
    "resource_wait_timeout_seconds": 120,
    "cleanup_timeout_seconds": 30
  },
  "scan_limits": {
    "max_files": 10000,
    "max_total_bytes": 1073741824,
    "max_wall_seconds": 60
  }
}
```

尖括号中的值都是必须替换的占位符。引用中的 `sha256` 是被引用 JSON 文件的内容摘要。需要 Invocation 的请求在顶层展开四项
runtime limit，并提供短期 `launch_deadline`。可选的 `absolute_deadline` 表示用户授权停止时间；
显式重试时可选 `retry_of` 指向旧 invocation id。

## readiness

优化 Target 的最小完整形状：

```json
{
  "format_version": "cuda-kernel-optimizer/readiness-input-v1",
  "operation": "check",
  "artifact_root": "/absolute/new/artifacts",
  "target_mode": "optimization",
  "claim_layer": "workload",
  "test_suite": {"path": "/absolute/tests.json", "case_ids": ["main"]},
  "correctness": {
    "reference_path": "/absolute/reference.json",
    "method": "driver",
    "acceptance": {"metric": "max_error", "operator": "less_or_equal", "value": 0.001}
  },
  "original": {"kind": "source_snapshot", "path": "/absolute/original"},
  "objective": {
    "primary_metric": {"name": "latency_ms", "unit": "ms", "direction": "lower", "aggregation": "median"},
    "minimum_effect": {"value": 0.5, "unit": "percent"},
    "constraints": []
  },
  "driver": {
    "command": ["/absolute/python", "/absolute/workload_driver.py"],
    "request_argument": "--request",
    "execution_mode": "combined",
    "protocol_version": "cuda-kernel-optimizer/driver-v1",
    "profiler_capabilities": [],
    "side_effects": [],
    "cleanup_contract": {"kind": "process_group_only", "external_tasks": false}
  },
  "environment_requirements": {"gpu_uuids": ["GPU-..."], "required_tools": ["nvidia-smi"]},
  "validity_requirements": {"minimum_pairs": 10, "confidence": 0.95, "bootstrap_samples": 10000},
  "smoke": {
    "mode": "combined",
    "case_id": "main",
    "resources": {"host_id": "<current host id>", "gpu_uuids": ["GPU-..."]},
    "runtime_limits": {
      "operation_timeout_seconds": 120,
      "command_timeout_seconds": 90,
      "resource_wait_timeout_seconds": 30,
      "cleanup_timeout_seconds": 15
    }
  },
  "scan_limits": {"max_files": 10000, "max_total_bytes": 1073741824, "max_wall_seconds": 60}
}
```

`claim_layer` 可为 `kernel`、`workload` 或 `serving`。约束项使用
`name`、`unit`、`direction`、`aggregation: "median"` 和 `max_regression_pct`。
driver 输入输出协议以 `templates/workload_driver_request.schema.json`、
`templates/workload_driver_result.schema.json` 和 `templates/workload_driver.py` 为准。
V1.4 只支持留在 Invocation 进程组内的任务，不支持远端或脱离进程组的后台任务。
optimization readiness 只支持 combined driver；最小 smoke 在 request 的开放 `sampling`
对象中发送 `{"kind":"smoke","repetitions":2}`。driver result 的 primary 和全部 constraints
必须与 request objective 的名称、unit 和集合精确一致，每组 samples 必须恰好有两个值。

只有现成报告或编译产物时使用 diagnostic Target：

```json
{
  "format_version": "cuda-kernel-optimizer/readiness-input-v1",
  "operation": "check",
  "artifact_root": "/absolute/new/artifacts",
  "target_mode": "diagnostic",
  "claim_layer": "diagnostic",
  "original": {"kind": "artifact", "path": "/absolute/source-or-binary"},
  "materials": [{
    "kind": "report",
    "path": "/absolute/report",
    "tool": "nsys",
    "tool_version": "2026.2.1",
    "dialect": "nsys-sqlite-3.25-v1"
  }],
  "environment_requirements": {"gpu_uuids": [], "required_tools": []},
  "scan_limits": {"max_files": 10000, "max_total_bytes": 1073741824, "max_wall_seconds": 60}
}
```

## workload evaluation

`baseline`：

```json
{
  "format_version": "cuda-kernel-optimizer/evaluator-input-v1",
  "operation": "baseline",
  "artifact_root": "/absolute/artifacts",
  "target_ref": {"id": "<target id>", "sha256": "<64 hex>"},
  "sampling_design": {"case_ids": ["main"], "samples_per_case": 10, "seed": 1},
  "resources": {"host_id": "<frozen host id>", "gpu_uuids": ["GPU-..."]},
  "operation_timeout_seconds": 600,
  "command_timeout_seconds": 300,
  "resource_wait_timeout_seconds": 120,
  "cleanup_timeout_seconds": 30,
  "launch_deadline": "<current epoch seconds + 60>"
}
```

新建的 combined Target 在 baseline 对每个 case 只调用 driver 一次；driver 必须在一次 result 中返回
`samples_per_case` 个 primary 样本，以及数量相同的每项 constraint 样本。样本数、名称、unit
或 constraint 集合不一致时，第一次 result 即失败，不能进入聚合。额外诊断写入已声明 artifact
或 driver 日志，不能混入 measurements.constraints。evaluator 仍可读取已冻结的旧 separate Target；
该兼容路径会对每个 case 分别调用 correctness 和 measure，共 `2 × C` 次，不代表 readiness
仍能新建 separate optimization Target。

`experiment` 是同步记录操作，不含 runtime limit：

```json
{
  "format_version": "cuda-kernel-optimizer/evaluator-input-v1",
  "operation": "experiment",
  "artifact_root": "/absolute/artifacts",
  "target_ref": {"id": "<target id>", "sha256": "<64 hex>"},
  "baseline_ref": {"invocation_id": "inv-...", "sha256": "<64 hex>"},
  "source_base": {"kind": "source_snapshot", "path": "/absolute/reference"},
  "candidate": {"kind": "source_snapshot", "path": "/absolute/candidate"},
  "hypothesis": "<one falsifiable claim>",
  "mechanism_key": "normalized.mechanism.key",
  "claim_layer": "workload",
  "cheapest_falsifier": {"kind": "none", "reason": "<why no separate command is needed>"},
  "screen_design": {"enabled": true, "kind": "conservative_bound", "reason": "<why>", "claim": "<what it can reject>"},
  "estimated_cost": {
    "screen": {"p50_seconds": 60, "p90_seconds": 120, "gpu_count": 1, "basis": "measured baseline"},
    "target": {"p50_seconds": 300, "p90_seconds": 600, "gpu_count": 1, "basis": "measured baseline"}
  },
  "minimum_effect": {"value": 0.5, "unit": "percent"},
  "reject_if": [{"kind": "correctness_failed"}],
  "promote_if": [{"kind": "formal_target_passed"}],
  "change_scope": ["src/kernel.py"],
  "max_risk": "low"
}
```

`screen`、`target` 和 `final_audit` 使用同一比较请求骨架：

```json
{
  "format_version": "cuda-kernel-optimizer/evaluator-input-v1",
  "operation": "target",
  "artifact_root": "/absolute/artifacts",
  "target_ref": {"id": "<target id>", "sha256": "<64 hex>"},
  "experiment_ref": {"id": "<experiment id>", "sha256": "<64 hex>"},
  "sampling_design": {"case_ids": ["main"], "pairs": 10, "seed": 1},
  "resources": {"host_id": "<frozen host id>", "gpu_uuids": ["GPU-..."]},
  "operation_timeout_seconds": 1200,
  "command_timeout_seconds": 600,
  "resource_wait_timeout_seconds": 120,
  "cleanup_timeout_seconds": 30,
  "launch_deadline": "<current epoch seconds + 60>"
}
```

把 `operation` 改为 `screen` 可做初筛；改为 `final_audit` 时删除 `experiment_ref`。
正式比较的 `pairs` 不得低于 Target 的 `minimum_pairs`。
combined driver 的 baseline 调用数为 `C`，screen 调用数为 `2 × P × C`；正式 target 或
final audit 还会先对两个 Variant 各做一次每-case 精度调用，因此总数为
`2 × C + 2 × P × C`。其中 `C` 是 case 数、`P` 是 pairs；单 case、3 pairs 的正式调用为
8 次。readiness smoke 另计一次。

## profiler

三种 profiler 的 `analyze` 和 `collect` 共享字段模式，但各自使用独立 format version：

| 脚本 | format version | analyze 的附加字段 |
|---|---|---|
| `profile_ncu.py` | `cuda-kernel-optimizer/ncu-input-v1` | `report_ref`、`kernel_name_hints` |
| `profile_nsys.py` | `cuda-kernel-optimizer/nsys-input-v1` | `report_ref` |
| `profile_pytorch.py` | `cuda-kernel-optimizer/pytorch-input-v1` | `report_ref` |

`analyze` 必填：`format_version`、`operation: "analyze"`、`artifact_root`、`target_ref`、
表中附加字段、空 GPU 的 `resources` 以及四项 runtime limit 和 `launch_deadline`。
`report_ref` 取自 diagnostic Target 的 `target.json` 中对应 `diagnostic_materials` 项的 `id` 和
`sha256`；readiness 响应本身只返回 `target_ref`。NCU 的 `kernel_name_hints` 是字符串数组。

`collect` 必填：`format_version`、`operation: "collect"`、`artifact_root`、`target_ref`、
`baseline_ref`、`role`、`case_id`、Target 对应的 `resources`、四项 runtime limit 和
`launch_deadline`。NCU 还需要 `kernel_name_hints`。候选采集额外提供 `experiment_ref` 和
`correctness_ref`；其中 `correctness_ref` 必须包含 `invocation_id`、`sha256` 和同一个
`case_id`。original 采集不提供这两项。

## compiler 与 SASS

`compiler_evidence.py analyze` 和 `sass_check.py analyze` 必填：各自的 `format_version`、
`operation: "analyze"`、`artifact_root`、`target_ref`、`artifact_ref`、空 GPU 的 `resources`、
四项 runtime limit 和 `launch_deadline`。format version 分别为
`cuda-kernel-optimizer/compiler-input-v1` 和 `cuda-kernel-optimizer/sass-input-v1`。

以下是 `compiler_evidence.py` 的 Target material：

```json
{"source": "target_material", "stage": "ptx", "material_ref": {"id": "<material id>", "sha256": "<64 hex>"}}
```

以下是 `compiler_evidence.py` 的 Invocation driver artifact：

```json
{"source": "invocation_driver_artifact", "stage": "binary", "invocation_ref": {"invocation_id": "inv-...", "sha256": "<64 hex>"}, "receipt_index": 0, "relative_path": "artifacts/kernel.cubin"}
```

compiler stage 可为 `source`、`ttir`、`ttgir`、`llvm_ir`、`ptx`、`sass` 或 `binary`；
`sass_check.py` 只接受 binary，并且它的 `artifact_ref` 不包含 `stage`：Target material 使用
`{"source":"target_material","material_ref":{...}}`，Invocation driver artifact 使用
`{"source":"invocation_driver_artifact","invocation_ref":{...},"receipt_index":0,"relative_path":"artifacts/kernel.cubin"}`。

## knowledge 与 Champion

离线知识的 detached 查询：

```json
{
  "format_version": "cuda-kernel-optimizer/knowledge-input-v1",
  "operation": "query",
  "identity": {"gpu_architecture": "sm_120", "cuda_version": "13.0", "frameworks": {"triton": "3.4"}, "phenomena": ["memory_bound"], "claim_layer": "kernel"},
  "filters": {"mechanism_keys": []},
  "limits": {"max_results": 8, "max_context_bytes": 32768}
}
```

基于 Target 查询时，用 `artifact_root`、`target_ref` 和 `phenomena` 替换 `identity`。

`champion.py` 使用 `format_version: "cuda-kernel-optimizer/champion-input-v1"`。
`show` 另外提供 `operation: "show"`、`artifact_root` 和 `target_ref`。
`select` 与 `restore-original` 另外提供：

```json
{
  "result_ref": {"invocation_id": "inv-...", "sha256": "<64 hex>"},
  "expected_selection_ref": null
}
```

`expected_selection_ref` 必须等于 `show` 返回的当前值；没有已选 Champion 时为 `null`。

## status 与 cancel

所有长操作都由原脚本查询或取消：

```json
{
  "format_version": "<that tool input version>",
  "operation": "status",
  "artifact_root": "/absolute/artifacts",
  "invocation_id": "inv-..."
}
```

取消时只把 `operation` 改成 `cancel`。不要绕过工具直接删除 Invocation、锁或进程。
