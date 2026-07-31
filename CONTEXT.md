# GPU 工作负载优化

本文统一项目中的核心术语，避免把诊断证据、候选测试和当前最佳实现混为一谈。

## 术语

**RunSpec**

一次优化任务不可变的定义，包括目标、测试集、精度校验、测量规则、环境身份、
最低有效收益和允许的修改范围。

避免使用：Workload Contract、control、manifest

**Grant**

绑定 RunSpec 的追加授权，记录用户当前允许投入的时间、GPU 资源、修改范围、
风险和验证深度。

避免使用：budget、deadline

**Variant**

可按内容识别、可被测量的实现或部署包。original、champion 和 candidate
是 Variant 在 Experiment 中承担的角色。

避免使用：branch、snapshot

**Evidence**

为回答一个明确不确定性而采集的只读观测。Evidence 可以支持或否定假设，
但不能使 Variant 晋升。

避免使用：没有候选比较时笼统称为 measurement

**Experiment**

在同一 RunSpec 下，candidate Variant 与指定 reference champion 之间的受控比较，
包含明确的正确性和性能门禁。

避免使用：iteration、round、candidate session

**Promotion**

candidate 直接战胜当前 champion 并满足 RunSpec 后，写入的显式、不可变晋升记录。

避免使用：latest PASS、automatic winner

**Champion**

没有 Promotion 时，champion 是 RunSpec 中的 original Variant；此后是最新有效
Promotion 指向的 Variant。不能根据源码继承关系或单独的 kernel 结果推断 champion。

避免使用：best branch、latest successful candidate

**Source base**

candidate 开发时基于的 Variant。它可以不同于 reference champion，但必须在
promotion 前声明。

避免使用：用 parent 表示比较基线

**Reference champion**

Experiment 中作为直接比较基线的当前 champion。

避免使用：parent、original baseline

**Claim layer**

结果能够支持的最高范围：kernel、workload 或 serving。低层结果不能推导出高层结论。

避免使用：optimization level
