# 服务层证据

服务层结论必须来自用户真实请求路径中的成对比较。局部 kernel、离线 microbenchmark 或单次 profiler 观测只能解释机制，不能直接证明线上收益。

## 结论层级

- kernel：只说明指定 kernel、shape 和环境下的结果；
- workload：说明冻结测试集或 replay 上的完整请求结果；
- serving：说明实际服务栈、路由、并发和流量范围中的结果。

结论只能到达证据覆盖的层级。

## 测试身份

服务测试至少绑定：

- 服务镜像、代码与配置摘要；
- GPU、驱动、CUDA、框架和容器身份；
- 请求集或流量分层及其摘要；
- 路由、batch、并发、队列、缓存和依赖服务条件；
- original 与 Candidate 的部署路径证明；
- 物理 GPU/NUMA/拓扑分配，以及 variant 是否在设备集合间交叉；
- 主要指标、约束、最低有效收益和统计设计；
- 每个决策指标的实际分子、分母、数据来源和重建步骤；
- canonical request trace 或逐请求 token-id/长度摘要，以及 tokenizer、模型和采样配置身份；
- 原始样本、监控 guard、日志与清理状态。

## 执行原则

1. 先确认 original 服务能通过精度与健康检查。
2. 冻结 AB/BA 或其他成对设计，不在看到结果后修改分层和排除规则。优先让两个角色顺序使用同一物理设备集合；若角色驻留在不同 GPU 对或实例上，必须做平衡的 variant-by-device crossover，否则设备差异与代码差异混杂。
3. 两个角色使用相同入口、canonical request trace、预热和统计口径。优先 replay 同一组预 tokenized token ids；若 driver 会重新 tokenize 或从生成文本重建 token 数，必须证明逐请求身份和计数口径等价，否则相关指标只能作 diagnostic。
4. 保留每个 block 的原始顺序与环境观测；环境污染按预先规则处理。
5. 精度、错误率、超时、资源或其他约束失败时，候选不能因主要性能指标改善而被接受。
6. 正式结果有效且通过当前 Target 的 primary verdict 后，才由 ChatGPT 显式记录 Champion；工具不自动部署或扩大流量。
7. 最终交接前，用 original 与当前 Champion 完成 final audit。

正式 serving 测试前，用 original 重复样本估计各决策指标在当前环境中的测量分辨率。若候选在 primary 和预先声明的重要 secondary 上的预期效果都低于各自门槛或分辨率，不单独消耗完整 serving A/B 预算；先组合足够覆盖的机制，或只保留局部结论。

应用 minimum effect 前先做 metric semantic audit。固定 token 数的 workload 应核对“请求数 × 每请求 token 数”、driver 报告值和服务端计数；若差异来自重分词等测量实现，使用合约已知的计数重新计算，或将无法修复的比较标为 inconclusive，不能让测量伪差决定百分比级候选。

## 断连与终态

长命令由 Invocation 记录 request、事件、heartbeat、result 和清理状态。SSH 或前台断开不等于任务成功或失败，应通过 `status` 查询终态。timeout 会终止进程组；无法确认清理完成时，不应在同一资源上启动冲突任务。

服务测试结束后，报告收益区间、约束、有效范围、样本数量、污染判断、失败或跳过的证据，以及明确 terminal reason。不要只报告一个最好数字。
