# 系统层与中间表示

瓶颈不一定在 kernel。分析完整 workload 时，应先根据时间占比和可移除上限判断层级，再决定是否深入到 Triton IR、PTX 或 SASS。

## 系统层检查

- CPU launch、Python 或框架调度；
- host-to-device / device-to-host 传输；
- kernel 间空隙、同步和 stream 依赖；
- allocator、数据准备、I/O 和预处理；
- 多 GPU 通信、拓扑和负载不均；
- batch、queue、cache、路由和服务依赖；
- 共享宿主机上的频率、温度和其他进程干扰。

Nsys 适合查看时间线、launch 与系统调用；PyTorch Profiler 适合框架算子、CPU/GPU 关联和 trace；`execution_map.py` 只根据已有观测计算覆盖、重叠和可核验上限，不命名瓶颈。

## 编译与 IR

Triton 或 CUDA 候选可按需要检查：

- 高层算子和 shape-specialization；
- Triton TTIR/TTGIR/LLVM IR；
- PTX；
- cubin 与 SASS。

`compiler_evidence.py` 读取显式冻结的阶段产物；`sass_check.py` 对显式 binary 做受控反汇编。两者只提取事实，不触发 workload 构建，也不选择需要优化的 kernel。

不同层之间需要可追溯关联：源码变体、编译参数、产物摘要、kernel 名称、dispatch shape 和运行结果必须指向同一个 Experiment。只看到某条指令或某个 IR 形态，不能证明它是主要瓶颈或带来完整 workload 收益。

## 知识覆盖

`knowledge_query.py` 可按精确 GPU、CUDA、框架、claim layer 和观测查询 `references/knowledge/cards.json`。返回的机制卡用于补充检查点和反例，不是完整候选列表。空结果不限制 ChatGPT 从源码和实测证据中提出新方向。
