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
  "format_version": "cuda-kernel-optimizer/readiness-input-v2",
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
    "evidence_capabilities": ["single_variant_combined", "paired_same_process_combined"],
    "protocol_version": "cuda-kernel-optimizer/driver-v2",
    "profiler_capabilities": [],
    "side_effects": [],
    "cleanup_contract": {"kind": "process_group_only", "external_tasks": false}
  },
  "environment_requirements": {"gpu_uuids": ["GPU-..."], "required_tools": ["nvidia-smi"]},
  "validity_requirements": {"minimum_pairs": 10, "confidence": 0.95, "bootstrap_samples": 10000},
  "smoke": {
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
optimization readiness 要求 driver 至少声明 `single_variant_combined`；只有确实能在同一进程
共享已声明状态时，才声明 `paired_same_process_combined`。最小 smoke 会发送一个 original
subject 和 `{"kind":"smoke","repetitions":2}`。driver 一次返回该 subject 的正确性和性能
证据；primary 与全部 constraints 必须精确匹配 request objective，每组 samples 恰好两个值。

`same_process` 的协议保证是：两个 subject 由一次 driver 调用执行，并接收同一个
`prepare_acquisition` 返回对象。它不能自动证明 driver 内部管理的外部服务、远端进程或缓存
确实共享；若结论依赖这些状态，driver 必须把对应身份作为 artifact 或 diagnostic 保存，ChatGPT
再决定证据是否足够。仅回显 acquisition 声明不能支持更强归因。

只有现成报告或编译产物时使用 diagnostic Target：

```json
{
  "format_version": "cuda-kernel-optimizer/readiness-input-v2",
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
  "format_version": "cuda-kernel-optimizer/evaluator-input-v2",
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

baseline 对每个 case 只调用 driver 一次；driver 必须在一次 evidence bundle 中返回
`samples_per_case` 个 primary 样本，以及数量相同的每项 constraint 样本。样本数、名称、unit
或 constraint 集合不一致时，第一次 result 即失败，不能进入聚合。额外诊断写入已声明 artifact
或 driver 日志，不能混入 measurements.constraints。V2 不读取旧 Driver V1 Target。

`experiment` 是同步记录操作，不含 runtime limit：

```json
{
  "format_version": "cuda-kernel-optimizer/evaluator-input-v2",
  "operation": "experiment",
  "artifact_root": "/absolute/artifacts",
  "target_ref": {"id": "<target id>", "sha256": "<64 hex>"},
  "baseline_ref": {"invocation_id": "inv-...", "sha256": "<64 hex>"},
  "source_base": {"kind": "source_snapshot", "path": "/absolute/reference"},
  "candidate": {"kind": "source_snapshot", "path": "/absolute/candidate"},
  "hypothesis": "<one falsifiable claim bound to the production replacement boundary>",
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
  "max_risk": "low",
  "opportunity_claim": {
    "boundary": {
      "component": "qk-preprocess", "phase": "decode", "case_id": "main",
      "shape": "TP-local production shape", "lowering": "inductor",
      "graph": "full-cuda-graph", "dispatch": "production-path",
      "fallback": "none", "overlap": "exposed-critical-path"
    },
    "candidate_components": ["qk-preprocess"],
    "primary_model": "inverse_time",
    "denominator_us": 4314.0,
    "denominator_evidence": {"source": "frozen production replay", "sha256": "<64 hex>"},
    "pools": [{
      "pool_id": "qk-preprocess.decode", "component_id": "qk-preprocess", "parent_pool_id": null,
      "reference_time_us": 2.899, "candidate_time_us": 2.316,
      "occurrences": 10, "exposure_upper_bound": 1.0,
      "reference_evidence": {
        "relationship": "same_boundary",
        "execution_form": {"component": "qk-preprocess", "phase": "decode", "case_id": "main", "shape": "TP-local production shape", "lowering": "inductor", "graph": "full-cuda-graph", "dispatch": "production-path", "fallback": "none", "overlap": "exposed-critical-path"},
        "source": "production reference timing", "sha256": "<64 hex>", "reason": "exact production path"
      },
      "candidate_evidence": {
        "relationship": "same_boundary",
        "execution_form": {"component": "qk-preprocess", "phase": "decode", "case_id": "main", "shape": "TP-local production shape", "lowering": "inductor", "graph": "full-cuda-graph", "dispatch": "production-path", "fallback": "none", "overlap": "exposed-critical-path"},
        "source": "bounded prototype timing", "sha256": "<64 hex>", "reason": "same lowering and graph path"
      }
    }]
  },
  "comparison_contract": {
    "relationship": "implementation_equivalence",
    "additional_gates": [],
    "diagnostics": [],
    "acquisition": {
      "lifecycle": "isolated_process",
      "shared_state": [],
      "rebuilt_state": ["process", "weights", "allocator"],
      "rationale": "两套实现不能安全地共存于同一进程"
    }
  },
  "material_premises": []
}
```

`comparison_contract.relationship` 可为 `implementation_equivalence`、`artifact_fidelity` 或
`deployment_effect`。`additional_gates` 与 correctness acceptance 同形；`diagnostics` 只列指标名，
不能与 Target gate 或附加 gate 重叠。`same_process` 必须声明非空 `shared_state`，且 driver 已声明
`paired_same_process_combined`；否则使用 `isolated_process`。`material_premises` 只记录确实会改变
候选判断或实验设计的事实或假设，每项包含 `statement`、`component`、`version`、`status`、
`source` 和 `decision_effect`。一手资料中的命题使用 `primary_source_claim`，它仍是待 ChatGPT 结合版本和当前环境判断的来源主张，不是工具认证的事实；未核实内容使用 `unresolved_hypothesis`。

`opportunity_claim` 把当前 Candidate 的 production replacement boundary、实际 execution form、
Candidate scope、计入的时间池和完整 workload 分母冻结在 Experiment 中。`same_boundary` 的
reference 或 candidate evidence 必须逐字段匹配 boundary；eager、lowering、graph、dispatch、fallback
或 overlap 不同会在创建 Experiment 时被拒绝。`conservative_upper_bound` 只允许用于 reference，且必须
说明为什么能够约束当前 boundary。父子时间池不能同时计入，时间池的 component 必须属于
`candidate_components`。

工具根据这些声明重算可移除时间和端到端上限；低于 Target `minimum_effect` 时不创建 Experiment。
它验证显式矛盾、范围、摘要格式和算术，不声称能从任意文本或手写 JSON 证明 production 语义。
证据关系和 critical-path exposure 仍由 ChatGPT 根据原始 artifact 判断；原始 artifact 的来源与
SHA-256 必须保留。没有 production-equivalent candidate timing 时可将 `candidate_time_us` 和
`candidate_evidence` 同时设为 `null`，此时结果只是完全移除上限，不是预期收益。
`primary_model` 只允许 `direct_time`（延迟随时间同向变化）或 `inverse_time`（吞吐随时间反向变化），
并且必须与 Target direction 一致；其它 primary 不得套用这套时间贡献公式。

`screen`、`target` 和 `final_audit` 使用同一比较请求骨架：

```json
{
  "format_version": "cuda-kernel-optimizer/evaluator-input-v2",
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

把 `operation` 改为 `screen` 可做初筛；改为 `final_audit` 时删除 `experiment_ref`，并在顶层
提供同形状的 `comparison_contract`。
正式比较的 `pairs` 不得低于 Target 的 `minimum_pairs`。
baseline 调用数为 `C`。隔离进程比较为 `2 × P × C` 次；同进程成对比较为 `P × C` 次。
每次调用同时返回正确性和性能证据，不再额外重复运行每-case 精度 workload。任一 gate 失败后
不再启动用于性能结论的剩余调用；final audit 只可继续取得判断是否能恢复 original 所缺的最低
正确性证据。`C` 是 case 数、`P` 是 pairs；readiness smoke 另计一次。

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
