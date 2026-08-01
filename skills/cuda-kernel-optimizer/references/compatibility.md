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

`sm_120`、`sm_100`、`sm_90` 等必须按精确目标处理。不要因为都属于某一代产品就复用能力结论，也不要把其他架构上的收益数字直接当成当前目标的预测。

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

旧结果只有在 Target 身份、代码对象、测试集和测量设计仍一致时才可直接比较。任何一项变化都应建立新的 Target 或 Experiment。旧结果仍可用于提出假设，但不能冒充新环境中的测量证据。
