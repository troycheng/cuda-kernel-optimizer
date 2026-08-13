# GPU 工作负载优化

本文统一项目中的核心术语，避免把诊断发现、候选测试和最终性能结论混为一谈。

## 术语

**优化目标（Optimization Target）**

一次任务中保持不变的结论目标和材料身份。optimization 目标包括真实测试集、精度校验、
原始版本、主指标、最低有效收益、约束和固定测量环境；diagnostic 目标明确记录暂时缺少的
测试能力，只支持已有材料分析。

避免使用：Run、Workload Contract

**结论层级（Claim Layer）**

结果能够支持的最高范围：diagnostic、kernel、workload 或 serving。低层结果可以解释
高层现象，但不能替代高层测试。

避免使用：Optimization Level

**版本（Variant）**

内容可以被准确识别和复现的源码、构建产物或部署包。original、candidate 和 champion
是版本在一次优化任务中承担的角色。

避免使用：用 branch、workspace 或 latest 表示版本身份

**原始版本（Original）**

优化开始时由用户提供、并写入优化目标的版本。原始基线和最终复测都以它作为完整收益的
起点。

**候选版本（Candidate）**

ChatGPT 为验证一个明确性能假设而创建的版本。候选必须通过精度校验，并直接与当前最佳版本
比较后，才可能被采用。

**当前最佳版本（Champion）**

当前被 ChatGPT 明确采用、并有正式目标比较支持的版本。任务开始时，当前最佳版本是原始版本。

避免使用：latest PASS、best branch、自动 winner

**最佳版本选择（Champion Selection）**

ChatGPT 明确采用或恢复一个版本后留下的不可变记录。当前最佳版本由同一优化目标下当前生效
的有效选择记录确定；没有选择记录时就是原始版本。

**比较基准（Reference）**

一次候选比较中与候选直接配对测试的当前最佳版本。开发候选时使用的源码起点可以不同于
比较基准。

避免使用：用 parent 同时表示源码起点和比较基准

**原始基线（Baseline）**

原始版本在固定测试集、精度校验和测量环境下取得的有效结果。它证明测试可以运行并建立后续
比较的起点。

**候选实验（Experiment）**

围绕一个明确假设，对候选版本执行最低成本证伪、正确性、可选短版测试和正式目标比较。
ChatGPT 决定是否创建实验以及何时继续投入；工具一次只执行一个已明确请求的操作。

避免使用：Iteration、Round、Candidate Session

**调用（Invocation）**

工具为完成一个明确请求而进行的一次独立执行。调用只记录自己的请求、进度和结果，不表示
优化任务的当前阶段，也不决定下一项调用。

避免使用：Run、Action、全局 Stage

**Profiler 事实（Profiler Facts）**

从 NCU、Nsight Systems、PyTorch Profiler、编译器或 SASS 中提取的可核验观测，包括来源、
版本、单位和内容身份。Profiler 事实用于分析，不能单独证明候选收益。

避免使用：把瓶颈判断或优化建议称为 Profiler 事实

**实质性技术前提（Material Technical Premise）**

一个外部技术事实，其真假会改变候选的可行性、安全性、收益上限或拒绝范围。它由 ChatGPT
在 Experiment 中声明并判断是否已解决；知识查询只能返回相关材料。

**技术契约（Technical Contract）**

由匹配版本的一手资料明确规定的原子语义，包括适用条件和不保证事项。它可以支持技术前提，
但不能替代当前 workload 的 profile、正确性或性能结果。

**优化经验（Optimization Heuristic）**

根据常见现象提出候选或观测方向的经验。它帮助 ChatGPT 搜索方向，不证明机制适用，也不决定
是否继续投入。

**实践案例（Practice Case）**

绑定具体环境、版本、workload 和证据的成功、失败或反例。跨环境使用时只作为参考，不能自动
升级为通用技术契约。

**正式目标比较（Target Comparison）**

候选版本与当前最佳版本在优化目标层级上的直接、有效比较。只有正式目标比较能够支持采用
候选版本。

避免使用：用 kernel benchmark 或短版 screen 代替正式目标比较

**Handoff**

ChatGPT 在暂停或结束时留下的简短任务说明，记录当前最佳版本、关键结果、已排除方向、
未解决问题、停止原因和恢复条件。Handoff 供用户和后续 ChatGPT 阅读，不参与工具准入。
