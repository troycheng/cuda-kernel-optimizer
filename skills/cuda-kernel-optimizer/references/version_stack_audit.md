# 版本栈审计

性能变化可能来自代码，也可能来自驱动、CUDA、编译器、框架或容器变化。只有版本栈稳定，候选比较才可解释。

## 记录范围

- GPU 型号、compute capability、UUID；
- NVIDIA driver；
- CUDA Runtime、Toolkit、NVCC 和相关库；
- PyTorch、Triton、CUTLASS、vLLM、TensorRT-LLM 等实际组件；
- NCU、Nsys、`cuobjdump` 等分析工具及来源；
- 容器镜像、Python 环境和关键编译参数。

版本字符串不足以证明工具身份。可执行文件的规范路径、摘要和来源也应绑定到调用记录。等待 GPU 锁或任务运行期间若工具身份发生变化，本次结果必须标为无效，不得静默改用新版本。

## 单变量升级

需要评估某个版本变化时：

1. 冻结相同代码、测试集、精度规则和测量设计；
2. 只改变一个明确组件；
3. 分别建立可追溯的 Target 或环境身份；
4. 先验证精度和实际 dispatch；
5. 再进行成对性能比较；
6. 报告适用版本范围，不将结果外推到未测组合。

readiness 和具体 profiler 会复用 `version_audit.py` 的只读校验能力。一次调用因版本或来源不明而失败时，保留该调用的 result 和 terminal reason；不要维护另一份全局失效名单。
