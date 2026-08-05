# MXFP6 SM120 精确分发与 p99 TPOT

[English](case-mxfp6-sm120-tail-latency.md)

这是一个真实、简单的使用反馈示例，不是可复现的 benchmark。它记录了为什么保留一项范围明确的优化，以及优化器从这次判断中学到了什么。

## 目标与改动

任务是优化一个用于 vLLM 张量并行服务的自定义 MXFP6 CUTLASS 扩展，测试使用两张 RTX 5090 GPU。首要目标是提升端到端 token 吞吐，同时将 TPOT 和正确性作为重要的服务指标。

候选改动没有实现新 kernel，而是为 `(40, 5120, 3072)` 和 `(48, 5120, 3072)` 两个 shape 增加精确分发，复用已有 kernel。代码已通过 [Nekofish-L/mxfp6_sm120#1](https://github.com/Nekofish-L/mxfp6_sm120/pull/1) 提交给上游项目。

## 实测证据

两个目标 kernel shape 的性能变化如下：

| Shape | Kernel 提升 |
|---|---:|
| `(40, 5120, 3072)` | 17.84% |
| `(48, 5120, 3072)` | 19.34% |

随后三组成对服务测试得到以下结果：

| 指标 | 观测结果 |
|---|---|
| p99 TPOT 改善 | 2.70%、7.53%、3.05%；中位数 3.05% |
| Token 吞吐 | 中位数 +0.32%；最差一组 -0.41% |
| 吞吐保护条件 | 最差一组仍在冻结的 0.5% 非劣阈值内 |
| 正确性和请求成功率 | 无变化 |

TPOT 改善为正表示每个输出 token 的耗时下降。这些测量没有证明端到端吞吐取得了实质提升。

## 最终判断

该候选被保留为一项适用范围明确的长尾延迟优化：三组成对测试中的 p99 TPOT 均有改善，同时吞吐、正确性和请求成功率均未超过声明的退化边界。本案例不将它表述为吞吐收益。

这个案例真正有价值的地方是结果选择。完整 workload 的时间占比可以界定吞吐或平均延迟的预期收益上限，但不能单独否定受影响请求上可重复的 p95 或 p99 改善。如果一项优化改善了重要指标，并且其他关键指标整体不负向，那么在准确说明适用范围后，它仍然是有效的优化结果。

## 对项目的反馈

此前的指令可能因为目标 shape 的总时间覆盖较低、主吞吐指标变化很小而丢弃这项结果。[提交 `16d4f96`](https://github.com/troycheng/cuda-kernel-optimizer/commit/16d4f96) 已修改判断标准，要求保留稳定且整体不负向的长尾收益。[Issue #6](https://github.com/troycheng/cuda-kernel-optimizer/issues/6) 记录了这条真实使用反馈。

同一次任务还暴露了两项独立的流程成本：重复启动完整服务记录在 [Issue #7](https://github.com/troycheng/cuda-kernel-optimizer/issues/7)，小众技术栈的源码调研时机记录在 [Issue #8](https://github.com/troycheng/cuda-kernel-optimizer/issues/8)。它们目前保留在 backlog 中，不改变上述优化结论。

## 证据边界

私有多模态 workload 和原始服务 artifact 未公开。本示例中的汇总测量结果已获授权发布，只支持当前 workload 和环境下的上述判断，不预测其他场景中的收益。
