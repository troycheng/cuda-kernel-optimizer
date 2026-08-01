# 一次完整优化示例

下面展示 ChatGPT 使用本 skill 优化 Triton workload 时的典型过程。用户不需要手工拼装内部 JSON，也不需要逐条运行脚本。

## 用户提供的信息

```text
请优化这个 Triton attention workload。

代码：~/work/attention/
测试集：~/work/bench/requests.jsonl
精度校验：~/work/bench/check_accuracy.py
性能入口：python3 ~/work/bench/run.py --json
主要指标：p50 latency，越低越好
最低有效收益：0.5%
允许修改：~/work/attention/ 下的代码
宿主机配置：只给建议，不直接修改
```

测试集代表真实请求分布；精度校验给出期望输出、容差或业务精度标准。缺少其中任一项时，ChatGPT 会先帮助补齐验证环境，而不会直接把 microbenchmark 当成完整 workload 结论。

## ChatGPT 的处理过程

1. 调用 `readiness.py check`，检查 driver、测试集、精度规则和环境，冻结 `target.json` 与 original Variant。
2. 调用 `workload_evaluate.py baseline` 测量原始业务基线。若 original 精度失败，先报告基础环境问题，不开始性能候选。
3. 分析源码和已有观测，形成若干竞争性解释。例如：短序列受 launch gap 主导，长序列受 KV 访问主导，或尾块正确性限制了可用 tile。
4. 选择当前最值得验证的一项，声明候选机制、预期影响、最低成本证伪、修改范围和拒绝条件，再调用 `workload_evaluate.py experiment` 冻结 Candidate 与 Experiment。
5. 创建 Experiment 前先完成最低成本证伪；已经证伪时不执行候选。随后执行 `screen`，精度失败时停止该候选。若 `conservative_bound` 已证明收益上限低于阈值，也不继续；`diagnostic_proxy` 无法定论时，由 ChatGPT 根据它实际检验的主张和下一步成本决定是否进入正式测试。
6. 只有现有证据无法区分关键假设时，才调用 Nsys、NCU、PyTorch Profiler、编译产物或 SASS 分析。profiler 返回事实，由 ChatGPT 解释。
7. 候选未被初筛拒绝，并且正式比较仍值得投入时，执行 `target`。结果有效且优于当前最佳版本时，ChatGPT 调用 `champion.py select` 显式记录选择。
8. 在输出 workload 或服务层最终结论前，执行 `final_audit`，重新比较 original 与当前 Champion。

典型 artifact 目录如下：

```text
artifacts/
├── target.json
├── objects/
├── experiments/
│   └── <experiment-id>.json
├── invocations/
│   └── <invocation-id>/
│       ├── request.json
│       ├── events.jsonl
│       └── result.json
├── champion/
│   ├── current.json
│   └── selections/
└── handoff.md
```

这些文件分别回答：优化目标是否一致、候选究竟改了什么、命令实际执行了什么、结果是否有效，以及当前最佳版本为何被选择。ChatGPT 的分析与最终交接说明应引用这些记录，而不是复制一份不可核验的流程状态。

## 可能的结束方式

- 没有候选达到最低有效收益：停止，保留已排除方向及证据。
- 候选精度失败：拒绝候选，不解释或继续消费其性能结果。
- profiler 不可用：记录限制；如果其他证据足够则继续，否则说明结论上限。
- 候选通过 target 与 final audit：报告收益区间、适用范围、环境身份、风险和可回滚的 Champion 记录。
- 下一步可能有较高收益，但预计时间、GPU 成本、风险或修改范围超出用户授权：先汇总投入建议，再请求一次授权。

任何阶段出现明确结论后都应尽早结束。单个命令的 timeout 是防卡死措施，不是必须消耗完的优化时长。
