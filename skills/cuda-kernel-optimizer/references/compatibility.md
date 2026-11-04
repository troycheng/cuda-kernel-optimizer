# 兼容性判断

GPU 优化结论必须绑定实际硬件、驱动、CUDA、编译器、框架和 workload 身份。架构名称相近不代表代码生成、特性支持或性能行为相同。

## 精确身份

至少记录：

- GPU 型号、compute capability 和 UUID；
- 驱动版本、CUDA Runtime 与 Toolkit 来源；
- PyTorch、Triton、CUTLASS、vLLM 或 TensorRT-LLM 等实际版本；
- 容器镜像或环境摘要；
- 编译目标与关键编译参数；
- 测试集、精度校验和 workload driver 摘要。

硬件 compute capability、编译目标和具体指令要求分别核对。例如 `sm_90` 的硬件身份不等于已使用 `sm_90a` 编译；知识卡匹配硬件也不证明指令会被实际执行。只补充会影响当前判断的缺项，不为已确认的事实重复 probe。

## 能力确认

优先级从高到低：

1. 当前环境中的最小 compile probe；
2. 当前版本的官方文档和 release note；
3. `knowledge_query.py` 返回的身份匹配知识卡；
4. 相邻版本或相邻架构资料，仅用于形成待验证假设。

compile probe 只能证明某个特性在当前工具链可编译，不能证明它正确、会被实际 dispatch，或一定更快。

## Profiler 限制

NCU、Nsys 和 PyTorch Profiler 报告都按已知版本与关键字段解析。版本、schema、解释测量所需的字段或单位未知时拒绝输出语义观测，并保留原始报告与来源信息。已知格式中的非关键扩展内容只作为 `unmodeled` 保留。

`ERR_NVGPUCTRPERM` 表示宿主机不允许读取硬件 counter，不表示 NCU 缺失，也不表示 kernel 有问题。skill 不自动修改驱动权限；可以继续使用 timing、Nsys、PyTorch Profiler、编译产物或 SASS 等适合当前问题的证据。

## 迁移与复用

迁移的是机制和代价关系，不是 tile、stage、warp 数或历史收益。Ampere 的 `cp.async` 不等于 Hopper 的 TMA；SM100 与 SM120 的共享内存等资源限制也不同。按实际 shape、生成代码和资源占用重新选择参数，不能从同代产品名称推断互通。相关诊断见知识查询按需返回的 [数据供给与计算重叠说明](knowledge/playbooks/hopper-dataflow.md)。

H800/H100/H200 等型号的显存、板卡形态和互连不能相互代入。只有当容量、带宽或通信影响候选判断时，才补查实际设备配置；不把标称带宽当成 workload 可达带宽。[Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)、[Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) 和 [Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) 调优指南提供机制与资源边界，具体实现仍以部署版本为准。

旧结果只有在 Target 身份、代码对象、测试集和测量设计仍一致时才可直接比较。任何一项变化都应建立新的 Target 或 Experiment。旧结果仍可用于提出假设，但不能冒充新环境中的测量证据。
