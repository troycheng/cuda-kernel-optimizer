# 通用知识覆盖扩展实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用
> `superpowers-zh:subagent-driven-development` 逐任务实现本计划。每项任务先写失败测试，
> 再写最小实现；共享 JSON 和 `diagnostic_knowledge.py` 串行修改，主代理统一审查和提交。

**目标：** 在不改变 V1.2 Controller、授权和候选状态机的前提下，让知识子系统能够依据
当前、可信的本地证据，为 Ampere 至 Blackwell 的 CUDA workload 提出最多三个可证伪的
通用机制候选，并证明该能力不依赖现有 RTX 5090 案例。

**架构：** 保留 `diagnostic_knowledge.py` 作为唯一运行时检索入口。Profiler 和已有
active-evidence adapter 只产生版本明确的稳定语义；方法目录负责 exact-SM 能力过滤；
机制卡负责因果路由；案例记忆只保留精确身份下的拒绝和排序。知识输出仍无执行权和
promotion 权限。

**技术栈：** Python 3 标准库、JSON、`unittest`、现有内容寻址证据链和 V1.3 知识包。
自动化测试全部离线运行，不需要 GPU、网络或外部 AI。

---

## 1. 修改范围

### 修改

- `skills/cuda-kernel-optimizer/references/method_registry.json`
  - 修正 exact-SM 能力和 `min_sm` 数据，删除不可迁移的历史收益提示。
- `skills/cuda-kernel-optimizer/references/knowledge_sources.json`
  - 补齐架构、PTX、NCCL、profiler 和框架的一手来源及适用版本。
- `skills/cuda-kernel-optimizer/references/diagnostic_cards.json`
  - 保留跨层入口路由和 5090 精确案例卡，建立 12 个通用机制族。
- `skills/cuda-kernel-optimizer/scripts/knowledge_query.py`
  - 在 legacy 方法查询中同时执行 `min_sm` 和 exact feature 过滤。
- `skills/cuda-kernel-optimizer/scripts/diagnostic_evidence.py`
  - 集中定义可进入知识检索的稳定语义和合法 producer/tool 关系。
- `skills/cuda-kernel-optimizer/scripts/profile_ncu.py`
  - 保留原始 metric，同时输出版本明确的稳定 NCU 语义。
- `skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py`
  - 复用同一语义转换，不再只输出 heuristic 主轴。
- `skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py`
  - 拒绝低质量、未知版本和身份不一致的观测；安全放行满足条件的
    `source_verified` 机制。
- `tests/test_knowledge_query.py`
- `tests/test_profile_ncu.py`
- `tests/test_analyze_ncu_rep.py`
- `tests/test_diagnostic_evidence.py`
- `tests/test_diagnostic_knowledge.py`
- `tests/test_knowledge_replay.py`
- `README.zh-CN.md`、`README.en.md`
- `docs/knowledge-and-research.md`、`docs/validation.md`

### 创建

- `tests/fixtures/knowledge_coverage/cross_layer_cases.json`
  - 12 个 case-off 路由 fixture，覆盖六个不同的公开代码路径。
- `tests/test_knowledge_coverage.py`
  - 验证机制矩阵、跨层 fixture、case-off 行为和发布声明边界。

### 冻结

- `workload_controller.py`
- `diagnostic_decision.py`
- `adaptive_investment.py`
- `budget.py`
- `knowledge_adapter.py`
- grant、ChangeSet、CandidateGate、candidate stage、promotion 和所有 JSON Schema

若实现需要修改冻结项，停止当前任务并重新审查设计，不得增加旁路或兼容补丁。

## 2. 固定的数据约定

### 2.1 机制 ID

| 编号 | `mechanism_key` |
|---|---|
| K1 | `global_memory_transactions` |
| K2 | `redundant_dram_traffic` |
| K3 | `memory_latency_hiding` |
| K4 | `register_or_shared_pressure` |
| K5 | `parallelism_or_wave_tail` |
| K6 | `compute_pipeline_or_dtype` |
| K7 | `synchronization_or_atomic_contention` |
| W1 | `framework_launch_fragmentation` |
| W2 | `host_device_transfer_serialization` |
| W3 | `cpu_or_data_pipeline_starvation` |
| W4 | `collective_wait_or_rank_skew` |
| W5 | `serving_scheduling_or_request_path` |

`diagnostic.cross-layer.triage` 继续作为入口路由，不计入机制族，也不能成为实现候选。

### 2.2 候选可用观测

一条观测只有同时满足以下条件，才可命中 positive、counter 或 invalidator：

```python
quality == "validated"
action_id 与 evidence_kind 属于现有 catalog 的固定组合
tool.name 属于该 evidence_kind 的允许集合
tool.version 与 knowledge_identity 中对应组件的 verified value 完全一致
source_digest、adapter digest 和 result digest 已通过现有封存链
```

`heuristic`、`estimated`、未知 quality、未知 tool、未知 tool version、身份不一致和
`unmodeled` 都不能形成候选。`ERR_NVGPUCTRPERM` 仍只转换为
`profile.counter_access=unavailable`。

### 2.3 稳定语义

首批只归一化能够由现有工具明确获得的语义，不补猜缺失指标：

| 机制 | 稳定语义示例 | 合法来源 |
|---|---|---|
| K1 | `kernel.global_memory_transaction_amplification` | NCU、SASS |
| K2 | `kernel.dram_throughput_pct`、`kernel.l2_hit_rate_pct` | NCU |
| K3 | `kernel.long_scoreboard_pct`、`kernel.eligible_warps_per_scheduler` | NCU |
| K4 | `kernel.occupancy_pct`、`kernel.spill_present` | NCU、SASS |
| K5 | `kernel.wave_tail_ratio`、`kernel.grid_too_small` | NCU、launch metadata |
| K6 | `kernel.tensor_pipe_pct`、`kernel.dtype_path_mismatch` | NCU、SASS |
| K7 | `kernel.barrier_stall_pct`、`kernel.atomic_contention` | NCU、SASS |
| W1 | `runtime.launch_gap_short_context`、`framework.dispatch_overhead` | Nsys、PyTorch |
| W2 | `transfer.h2d_serialized` | Nsys |
| W3 | `runtime.gpu_waiting_for_input` | Nsys、PyTorch |
| W4 | `communication.rank_arrival_skew` | Nsys |
| W5 | `serving.queue_or_request_path_dominant` | Nsys、workload KPI |

原始工具字段继续封存。稳定语义只用于路由，不代替原始 profile 和正式性能验证。

## 3. 实施任务

### 任务 1：修正 exact-SM 方法过滤

**文件：**

- 修改：`skills/cuda-kernel-optimizer/scripts/knowledge_query.py`
- 修改：`skills/cuda-kernel-optimizer/references/method_registry.json`
- 修改：`tests/test_knowledge_query.py`

- [ ] **步骤 1：写失败测试**

在 `tests/test_knowledge_query.py` 增加：

```python
def test_min_sm_rejects_method_even_when_feature_name_is_present(self):
    module = load_module()
    registry = {
        "arch_feature_map": {"sm_80": ["tensor_core", "tma"]},
        "methods": {
            "future": {
                "axis": "memory",
                "priority": 1,
                "min_sm": 90,
                "name": "future",
                "required_features": ["tma"],
            }
        },
    }
    self.assertEqual(
        module._kernel_cards(registry, "sm_80", None, None, {}),
        [],
    )

def test_registry_contains_no_transferable_speedup_claims(self):
    registry = json.loads(
        (SCRIPT.parents[1] / "references" / "method_registry.json").read_text()
    )
    self.assertFalse(
        any("typical_speedup" in method for method in registry["methods"].values())
    )
```

再增加表驱动测试，逐一查询 `sm_80`、`sm_86`、`sm_89`、`sm_90`、`sm_100`、
`sm_103`、`sm_110`、`sm_120`、`sm_121`，断言结果的 `min_sm` 不大于当前 SM，
且 `required_features` 是 exact feature map 的子集。

- [ ] **步骤 2：确认测试失败**

运行：

```bash
python3 -m unittest tests.test_knowledge_query -v
```

预期：新增的 `min_sm` 和 `typical_speedup` 测试失败；既有测试继续通过。

- [ ] **步骤 3：写最小实现**

在 `knowledge_query.py` 增加严格解析：

```python
def _sm_number(arch: str) -> int:
    match = re.fullmatch(r"sm_(\d+)", arch)
    if match is None:
        raise ValueError(f"Invalid exact architecture: {arch}")
    return int(match.group(1))
```

在 `_kernel_cards()` 中先检查：

```python
if _sm_number(arch) < int(method["min_sm"]):
    continue
```

清理 `method_registry.json` 中全部 `typical_speedup`，核对 exact feature map，不使用数值继承。

- [ ] **步骤 4：验证并提交**

```bash
python3 -m unittest tests.test_knowledge_query tests.test_compatibility -v
git diff --check
git add skills/cuda-kernel-optimizer/scripts/knowledge_query.py \
        skills/cuda-kernel-optimizer/references/method_registry.json \
        tests/test_knowledge_query.py
git commit -m "fix: enforce exact architecture method gates"
```

预期：测试通过，提交只包含上述三个文件。

### 任务 2：建立稳定 NCU 语义

**文件：**

- 修改：`skills/cuda-kernel-optimizer/scripts/diagnostic_evidence.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/profile_ncu.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py`
- 修改：`tests/test_diagnostic_evidence.py`
- 修改：`tests/test_profile_ncu.py`
- 修改：`tests/test_analyze_ncu_rep.py`

- [ ] **步骤 1：写语义转换失败测试**

测试固定输入 metric 名称、值、单位和 NCU 版本，要求：

```python
observations = module.normalize_ncu_metrics(
    {
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": (81.0, "%"),
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": (37.0, "%"),
        "unknown__metric": (9.0, "%"),
    },
    tool_version="2026.2",
)
self.assertEqual(
    [item["semantic_id"] for item in observations],
    ["kernel.dram_throughput_pct", "kernel.long_scoreboard_pct"],
)
self.assertTrue(all(item["quality"] == "validated" for item in observations))
```

另测：

- 未知 NCU major/minor 返回空语义和 `unmodeled` 原因；
- 非有限数值、缺单位和重复冲突失败关闭；
- heuristic `primary_axis` 不进入 `semantic_observations`；
- `profile_ncu.py` 与 `analyze_ncu_rep.py` 对同一 CSV 产生相同语义。

- [ ] **步骤 2：确认测试失败**

```bash
python3 -m unittest \
  tests.test_diagnostic_evidence \
  tests.test_profile_ncu \
  tests.test_analyze_ncu_rep -v
```

预期：只因 `normalize_ncu_metrics` 和 `semantic_observations` 尚不存在而失败。

- [ ] **步骤 3：实现唯一转换函数**

在 `diagnostic_evidence.py` 中维护 NCU metric 正则到稳定语义的闭合表，并实现
`normalize_ncu_metrics(metrics, tool_version)`。返回值必须且只能包含
`semantic_observations`、`unmodeled_metrics` 和
`mapping_version: "ncu-semantic-v1"`；已识别 metric 按 `semantic_id` 排序，未识别 metric
按原始名称排序。

每条语义包含 `semantic_id`、`status`、`value`、`unit`、`scope`、`aggregation`、
`tool` 和 `quality`。`diagnostic_knowledge.normalize_observations()` 继续使用封存结果的
`result_sha256` 生成 `source_digest`，两个 NCU 脚本不得自行声明证据摘要。两个脚本只调用
同一转换函数，不复制映射表。

- [ ] **步骤 4：验证两个入口一致并提交**

```bash
python3 -m unittest \
  tests.test_diagnostic_evidence \
  tests.test_profile_ncu \
  tests.test_analyze_ncu_rep -v
git diff --check
git add skills/cuda-kernel-optimizer/scripts/diagnostic_evidence.py \
        skills/cuda-kernel-optimizer/scripts/profile_ncu.py \
        skills/cuda-kernel-optimizer/scripts/analyze_ncu_rep.py \
        tests/test_diagnostic_evidence.py tests/test_profile_ncu.py \
        tests/test_analyze_ncu_rep.py
git commit -m "feat: normalize stable NCU observations"
```

### 任务 3：整理一手来源和 12 个通用机制族

**文件：**

- 修改：`skills/cuda-kernel-optimizer/references/knowledge_sources.json`
- 修改：`skills/cuda-kernel-optimizer/references/diagnostic_cards.json`
- 修改：`skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py`
- 修改：`tests/test_diagnostic_knowledge.py`

- [ ] **步骤 1：写知识包结构失败测试**

新增测试要求：

```python
expected = {
    "global_memory_transactions",
    "redundant_dram_traffic",
    "memory_latency_hiding",
    "register_or_shared_pressure",
    "parallelism_or_wave_tail",
    "compute_pipeline_or_dtype",
    "synchronization_or_atomic_contention",
    "framework_launch_fragmentation",
    "host_device_transfer_serialization",
    "cpu_or_data_pipeline_starvation",
    "collective_wait_or_rank_skew",
    "serving_scheduling_or_request_path",
}
self.assertEqual(
    {
        card["mechanism_key"]
        for card in cards["cards"]
        if card["content_status"] == "source_verified"
        and card["id"] != "diagnostic.cross-layer.triage"
    },
    expected,
)
```

同时断言：

- 12 张卡都至少有一个 positive、counter、invalidator 和只读 cheapest falsifier；
- 通用卡不含 speedup、固定 tile/warp/stage 或默认配置；
- triage 没有 runtime candidate 权限；
- 6 张 5090 精确案例卡原样保留；
- 所有 `source_ids` 指向 `status=verified` 的一手来源；
- 知识包不再硬编码“恰好 14 个来源”，但必须非空、ID 唯一并全部被校验。

- [ ] **步骤 2：确认测试失败**

```bash
python3 -m unittest tests.test_diagnostic_knowledge -v
```

- [ ] **步骤 3：更新来源**

只加入正式机制实际使用的一手来源：

- CUDA Programming Guide、Best Practices、PTX ISA；
- Ampere、Hopper、Blackwell tuning/compatibility guides；
- Nsight Compute、Nsight Systems；
- CUTLASS/CuTe、Triton、PyTorch profiler/compile；
- NCCL user guide；
- Serving 只使用 vLLM 或 TensorRT-LLM 官方文档。

每项填写具体版本或版本族、章节 locator、核对日期和摘要 SHA。论文和第三方 AI 不能作为
架构事实或 profiler metric 的正式来源。

- [ ] **步骤 4：建立机制卡**

将现有 7 张通用路由卡整理成一个 triage 和 12 张机制卡。所有机制卡：

```json
{
  "status": "routing_only",
  "content_status": "source_verified",
  "case_ids": [],
  "cheapest_falsifier": {
    "action_id": "ncu-targeted-kernel",
    "rationale": "Measure the named kernel metric that distinguishes this mechanism from its counter explanation."
  }
}
```

不得新建 action，不得修改 6 张本地案例的身份、结果或收益。

- [ ] **步骤 5：验证并提交**

```bash
python3 -m unittest tests.test_diagnostic_knowledge tests.test_knowledge_replay -v
git diff --check
git add skills/cuda-kernel-optimizer/references/knowledge_sources.json \
        skills/cuda-kernel-optimizer/references/diagnostic_cards.json \
        skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py \
        tests/test_diagnostic_knowledge.py
git commit -m "feat: add source-backed diagnostic mechanisms"
```

### 任务 4：安全放行 `source_verified` 候选

**文件：**

- 修改：`skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py`
- 修改：`tests/test_diagnostic_knowledge.py`
- 修改：`tests/test_active_diagnosis_vertical.py`

- [ ] **步骤 1：写门禁失败测试**

从同一个满足收益空间和 action availability 的 frozen input 出发，逐项变异：

```python
mutations = {
    "quality": "heuristic",
    "tool": {"name": "unknown", "version": "1"},
    "tool_version": "2025.1",
}
```

要求：

- `validated`、工具身份匹配、positive 命中时产生一个
  `confidence=inconclusive`、`promotion_authority=none` 的候选；
- low quality、未知工具、工具版本与 `knowledge_identity` 不一致时只产生 explanation；
- 没有 positive、命中 counter/invalidator、action 不可用或收益空间不足时不产生候选；
- 直接调用 `query_frozen()` 不能绕过 producer/action/tool 检查；
- 在 `tests/test_active_diagnosis_vertical.py` 单独篡改
  `adapter_implementation_sha256`，确认现有 Controller 封存链拒绝该输入；不得修改 Controller。

- [ ] **步骤 2：确认测试失败**

```bash
python3 -m unittest \
  tests.test_diagnostic_knowledge \
  tests.test_active_diagnosis_vertical -v
```

- [ ] **步骤 3：实现可信观测过滤**

在 `diagnostic_knowledge.py` 中加入纯函数和闭合映射：

```python
_ACTIVE_EVIDENCE_TOOLS = {
    "ncu_kernel": ({"ncu"}, "profiler_versions", "ncu"),
    "nsys_timeline": ({"nsys"}, "profiler_versions", "nsys"),
    "os_runtime": ({"nsys"}, "profiler_versions", "nsys"),
    "framework_trace": ({"pytorch"}, "framework_versions", "pytorch"),
    "compiler_sass": ({"nvdisasm", "cuobjdump"}, "compiler_versions", "cuda"),
}

def _active_observation_trust(
    observation: Mapping[str, object],
    *,
    action_id: str,
    evidence_kind: str,
    identity: Mapping[str, object],
) -> tuple[bool, str]:
    if _ACTIVE_ACTIONS.get(action_id) != evidence_kind:
        return False, "producer_untrusted"
    if observation["quality"] != "validated":
        return False, "quality_untrusted"
    expected = _ACTIVE_EVIDENCE_TOOLS.get(evidence_kind)
    if expected is None:
        return False, "tool_unmodeled"
    tool_names, identity_group, identity_key = expected
    if observation["tool"]["name"] not in tool_names:
        return False, "tool_unmodeled"
    fact = identity[identity_group].get(identity_key)
    if fact is None or fact["status"] != "verified":
        return False, "identity_unverified"
    if observation["tool"]["version"] != fact["value"]:
        return False, "tool_version_mismatch"
    return True, "validated"
```

固定返回原因：

```text
validated
quality_untrusted
producer_untrusted
tool_unmodeled
tool_version_mismatch
identity_unverified
```

`_rule_evidence()` 只消费通过该函数的观测。`validate_knowledge_package()` 只把具备 positive
规则、已核对来源和只读 falsifier 的 `source_verified` 卡加入 runtime eligibility。
不新增 decision、stage 或 schema。

- [ ] **步骤 4：验证并提交**

```bash
python3 -m unittest \
  tests.test_diagnostic_knowledge \
  tests.test_active_diagnosis_vertical \
  tests.test_knowledge_query -v
git diff --check
git add skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py \
        tests/test_diagnostic_knowledge.py \
        tests/test_active_diagnosis_vertical.py
git commit -m "feat: admit trusted source-backed candidates"
```

### 任务 5：证明知识不依赖 5090 案例

**文件：**

- 创建：`tests/fixtures/knowledge_coverage/cross_layer_cases.json`
- 创建：`tests/test_knowledge_coverage.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py`

- [ ] **步骤 1：创建 12 个独立路由 fixture**

fixture 根结构固定为：

```json
{
  "schema_version": "cuda-optimizer/knowledge-coverage-fixture-v1",
  "cases": [
    {
      "id": "cuda-coalescing",
      "stack_family": "cuda_kernel",
      "public_path": "NVIDIA/cuda-samples@master:cpp/6_Performance/transpose/transpose.cu",
      "mechanism_key": "global_memory_transactions",
      "semantic_observations": [],
      "counter_observations": [],
      "invalidator_observations": []
    }
  ]
}
```

12 个 fixture 一一对应 K1–K7、W1–W5；`stack_family` 合计覆盖
`cuda_kernel`、`cutlass_cute`、`triton`、`pytorch`、`serving`、`nccl`，且至少来自六个
不同的公开代码路径。fixture 只保存 adapter 输出后的语义输入和期望机制，不执行或解析公开
代码路径，也不保存或复制 5090 winner。

六个来源路径固定为：

| `stack_family` | 官方路径 |
|---|---|
| `cuda_kernel` | `NVIDIA/cuda-samples@master:cpp/6_Performance/transpose/transpose.cu` |
| `cutlass_cute` | `NVIDIA/cutlass@main:examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu` |
| `triton` | `triton-lang/triton@main:python/tutorials/03-matrix-multiplication.py` |
| `pytorch` | `pytorch/tutorials@main:recipes_source/recipes/profiler_recipe.py` |
| `serving` | `vllm-project/vllm@main:benchmarks/benchmark_serving.py` |
| `nccl` | `NVIDIA/nccl-tests@master:src/all_reduce.cu` |

实现时记录每个路径内容的 commit SHA 和摘要；若上游路径已经迁移，只允许更新到同一官方
仓库中可追溯的新路径，并在 fixture 中记录迁移来源。

- [ ] **步骤 2：写 case-off 失败测试**

测试在临时 reference 目录中：

```python
cases["cases"] = []
for card in cards["cards"]:
    card["case_ids"] = []
context = module.build_knowledge_context(frozen, limit=3)
```

逐 fixture 断言：

- 正向观测只命中预期机制；
- counter 不会被当作 positive；
- invalidator 直接拒绝；
- cheapest action 不可用时不产生候选；
- 所有候选仍是 inconclusive、无 promotion authority；
- 每个“适用”矩阵单元可追溯到卡、语义和 action；
- 未匹配知识只返回 explanation/STOP 所需输入，不报错，也不声称没有优化空间。

- [ ] **步骤 3：允许空案例记忆**

`validate_knowledge_package()` 允许 `cases: []`，但仍严格校验非空运行包中的每条案例。
通用机制不能依赖 `case_ids`；本地卡缺少精确案例时仍失败关闭。

- [ ] **步骤 4：验证并提交**

```bash
python3 -m unittest \
  tests.test_knowledge_coverage \
  tests.test_diagnostic_knowledge \
  tests.test_knowledge_replay -v
git diff --check
git add tests/fixtures/knowledge_coverage/cross_layer_cases.json \
        tests/test_knowledge_coverage.py \
        skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py
git commit -m "test: add case-independent knowledge coverage"
```

### 任务 6：整体回归和文档收口

**文件：**

- 修改：`README.zh-CN.md`
- 修改：`README.en.md`
- 修改：`docs/knowledge-and-research.md`
- 修改：`docs/validation.md`
- 修改：与精确测试总数直接相关的现有文档测试

- [ ] **步骤 1：运行知识子系统回归**

```bash
python3 -m unittest \
  tests.test_knowledge_query \
  tests.test_diagnostic_evidence \
  tests.test_profile_ncu \
  tests.test_analyze_ncu_rep \
  tests.test_diagnostic_knowledge \
  tests.test_knowledge_coverage \
  tests.test_knowledge_replay \
  tests.test_active_diagnosis_vertical -v
```

预期：全部通过；没有 GPU、网络或 Controller 代码变更。

- [ ] **步骤 2：运行完整测试和 skill 自检**

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 skills/cuda-kernel-optimizer/scripts/self_check.py
git diff --check
```

预期：完整测试和 self-check 通过。若失败，先按共同根因分类；不为单个 fixture 增加生产特例。

- [ ] **步骤 3：更新文档**

中文 README 为主，英文 README 保持等价信息。只说明：

- 覆盖 Ampere 至 Blackwell 的来源支持和路由测试；
- 12 个机制族及六个软件层；
- 精确架构能力按 exact SM 和本地身份过滤；
- 物理性能验证范围仍单独列出；
- 5090 案例是保留回归，不代表其他环境收益；
- 未命中知识卡不等于没有优化空间。

不得使用“全面支持”“普遍提速”“已在所有架构验证”等表述。

- [ ] **步骤 4：最终边界审查**

```bash
git diff d0994bd -- \
  skills/cuda-kernel-optimizer/scripts/workload_controller.py \
  skills/cuda-kernel-optimizer/scripts/diagnostic_decision.py \
  skills/cuda-kernel-optimizer/scripts/adaptive_investment.py \
  skills/cuda-kernel-optimizer/scripts/budget.py \
  skills/cuda-kernel-optimizer/scripts/knowledge_adapter.py \
  skills/cuda-kernel-optimizer/templates
```

预期：无输出。

再检查：

```bash
rg -n "typical_speedup|全面支持|普遍提速|所有架构.*验证" \
  skills/cuda-kernel-optimizer/references README*.md docs
```

预期：没有不可迁移收益或过度发布声明。

- [ ] **步骤 5：提交**

```bash
git add README.zh-CN.md README.en.md \
        docs/knowledge-and-research.md docs/validation.md \
        tests
git commit -m "docs: describe general knowledge coverage"
```

## 4. 执行组织

为减少 token 和返工：

1. 任务 1、2 分别交给中等思考的实现代理，输入只包含对应文件和验收命令；
2. 任务 3、4 由高思考代理串行处理，因为共享 `diagnostic_knowledge.py` 和知识 JSON；
3. 任务 5 交给独立代理，只读生产逻辑后编写 fixture 和覆盖测试；
4. 主代理在每个任务后只做规格 diff 审查和目标测试，不重新探索整个仓库；
5. 任务 6 由主代理执行完整回归、文档一致性检查和最终提交；
6. 任一任务出现设计级冲突时暂停该任务，回到本计划和设计文档判断，不做 case-by-case 补丁。

## 5. 完成条件

只有同时满足以下条件才算实现完成：

- exact-SM 与 `min_sm` 双重过滤生效；
- 12 个通用机制族完整且都能最低成本证伪；
- `source_verified` 只能由当前、validated、版本匹配的本地观测触发；
- case memory 关闭后 12 个 fixture 仍正确路由；
- 六个软件层和九个 exact SM 都有独立反事实验证；
- 知识候选始终不超过三个、无执行权、无 promotion 权限；
- 5090 保留回放不退化；
- V1.2 Controller、授权、状态机、预算和 schema 均未修改；
- 完整测试、self-check 和文档一致性检查全部通过。
