# MXFP6 SM120 MoE：从 kernel 到端到端服务优化

[English](case-mxfp6-sm120-moe-end-to-end.md)

这是一个经授权公开的真实使用反馈示例，不是可复现 benchmark。它记录了哪项结果通过了完整服务目标、哪项看起来很好的局部收益没有被包装成端到端收益，以及这次案例实际改变了哪些优化判断。

## 目标与证据契约

任务是在两张 RTX 5090、TP2 环境中优化 Qwen3.5-35B-A3B MoE 服务的 MXFP6 路径。主指标是冻结的 64 请求真实多模态 workload、并发 4 下的 output-token 吞吐；TPOT、请求完成率、服务端 token 计数、layer 正确性和参考输出保真度作为保护指标。

测试使用内部修改的 vLLM 0.25.1 环境。可执行的原始基线是开发者的 MoE feature 版本；任务开始时，公开仓库的 `main` 尚不包含 MoE 实现，无法运行同一 workload，因此没有把它伪装成原始性能基线。

## 最终保留的优化

保留的 kernel 改动针对 batch 4 的 W2 路径：对 12-byte FP6 权重片段使用对齐的 packed load，保留 scalar fallback，并保持 routed expert 与 shared expert 权重分离。公开的 CUDA Graph layer 结果通过了输出 exact 一致性，并给出以下 TP2 成对测量：

| 范围 | Reference | Candidate | 提升 |
|---|---:|---:|---:|
| 完整 batch-4 MoE layer | 26.618 us | 24.647 us | 8.00% |

完整样本和测量契约见公开的 [B4 vector-load 结果清单](https://github.com/Nekofish-L/mxfp6_sm120/blob/main/benchmarks/results/qwen35_moe_b4_vector.json)。

在冻结的服务 workload 上，最终 MXFP6 Champion 相比可执行的开发者基线取得：

| 指标 | 变化 |
|---|---:|
| Output-token 吞吐 | +4.092% |
| Mean TPOT | -3.498% |
| p99 TPOT | -3.275% |

两组按时间顺序配对的吞吐收益分别为 +4.513% 和 +3.679%。每个正式 block 均完成 64/64 请求，服务端报告的 token 数固定为 203,631 个 prompt token 和 59,193 个 completion token。

## FP8 交付对比与精度边界

同一 MXFP6 版本还与同架构的官方 FP8 checkpoint 做了对比。这个结果是部署格式对照，不是优化器相对原始实现的增量收益。在该环境中，MXFP6 的 output-token 吞吐为 +13.581%，mean TPOT 为 -10.085%，p99 TPOT 为 -7.960%。完整 block 和身份信息见公开的 [服务结果清单](https://github.com/Nekofish-L/mxfp6_sm120/blob/main/benchmarks/results/qwen35_moe_service_tp2.json)。

两种格式均完成 742/742 个参考输出案例。完整归一化字符相似度：FP8 为 35.62%，MXFP6 为 35.05%，差异 -0.57 个百分点，配对 95% 置信区间为 [-1.29, +0.15]；答案部分相似度差异为 -0.32 个百分点，95% 区间为 [-1.62, +0.95]。没有检测到统计显著的参考输出保真度下降，但这不是经过审计的业务准确率评测。

## 真正影响结果的判断

### 局部优胜方案没有在缺少服务覆盖时晋级

batch 2 扩展在独立 TP2 layer 上提升了 11.03%，但实测服务 graph 中没有 batch-2 W2 launch。因此该结果只被记录为局部收益，没有被计入服务 Champion，也没有包装成额外的端到端收益。

这体现了 kernel 候选必须继续通过完整 layer 覆盖和冻结服务目标的实际价值：单个 kernel 或 layer 的加速不能直接决定最终结果。

### 失败只关闭有证据覆盖的实现

多个 W1 load、staging、split-K、unroll 和 pipeline 实现未通过正确性或完整 layer 门槛。这些实验关闭的是已测试的实现，不是所有未来 W1 优化。通信融合与 GDN 实验也使用了同样的有限结论。

这样既能利用负向证据减少重复试错，又不会因为一个实现失败就武断宣称整个机制已经没有空间。

### 一手资料检索过晚造成了可测量的回退

在准备提交时，由于对 CUDA Programmatic Dependent Launch 的语义理解不完整，一度把有效的 PDL 链判断为不安全。移除 PDL 后，服务均值从 665.83 降到 655.70 output tokens/s，回退约 1.52%。随后核对 CUDA 和 CUTLASS 一手文档，确认 dependent grid 入口处的 wait 已提供所需的完成与内存可见性语义，因此拒绝了移除方案并恢复 PDL。

这里暴露的不是 PDL 特例，而是一个通用问题：在不熟悉的外部技术栈事实影响安全、正确性或终止判断之前，没有及时核对相关一手资料。

### 当前候选自身的收益上限计算得太晚

后续实验只把 MoE W2 从 MXFP6 替换为 MXFP4，但早期预期部分借用了 W1、W2 和 dense MXFP6 全部转换后的更大收益。限定当前窄候选所需的事实其实已经具备：W2 占实测 replay 的 7.99%；现有 MXFP6 与 MXFP4 路径的每 token 估算 global bytes 分别为 3,994,624 B 和 2,814,976 B。因此，仅由 global bytes 决定的 W2 理想加速上限约为 1.419x，对应乐观的端到端 Amdahl 上限约 +2.42%。

因为 +2.42% 仍高于 Target 的 +0.5% 最低有效收益，执行一次有界证伪是合理的，但它不足以支持更大的实现投入。TP2 layer 成对结果按实测 batch-2/batch-4 覆盖率加权并外推 40 层后，每个 replay 只节省 3.21 us：端到端约 +0.075%，比最低有效收益低约 6.7 倍。这个结果关闭的是已测试的 W2 映射，不是 MXFP4 这一模型格式方向。

### 公开方法先被还原为当前目标下的具体假设

后续研究核对了 [Cursor Warp Decode](https://cursor.com/blog/warp-decode)、[Alpha-MoE](https://github.com/Aleph-Alpha/Alpha-MoE)、[DeepGEMM MegaMoE](https://github.com/deepseek-ai/DeepGEMM) 和 CUTLASS 官方 MXFP4 支持。这些公开收益依赖不同的硬件、数据格式、expert layout、累加语义或通信重叠，不能直接作为当前 Target 的预期收益。

因此，literal output-centric Warp Decode W2 被还原成一个保持现有 route rounding 和累加顺序的 SM120/MXFP6 精确 skeleton。输出完全一致，但加权 W2 延迟从 6.696 us 回退到 23.806 us。该结果只拒绝当前目标上实测的 scalar output-centric 映射，不反驳其 B200/MXFP8 公开结果，也不关闭其他 output tiling。Alpha-MoE 和 MegaMoE 式 persistent pipeline 所依赖的跨 cluster 归约或通信重叠在当前 decode 路径中并不存在，因此仍只保留为可行性假设，而不是案例结论。

## 对项目形成的反馈

这次案例形成了五项范围明确的修改或 backlog，而不是一次宽泛的流程重写：

- [Issue #15](https://github.com/troycheng/cuda-kernel-optimizer/issues/15)：冻结派生容器的真实身份，不能只记录继承的镜像 label 或包版本。
- [Issue #16](https://github.com/troycheng/cuda-kernel-optimizer/issues/16)：当小幅 kernel 选择收益容易被启动和 cache 方差掩盖时，使用同进程成对测试。
- [Issue #17](https://github.com/troycheng/cuda-kernel-optimizer/issues/17)：区分同一实现的 parity 与不同 checkpoint 之间的 fidelity。
- [Issue #18](https://github.com/troycheng/cuda-kernel-optimizer/issues/18)：当陌生外部技术栈事实会影响安全、正确性或终止判断时，先核对相关一手资料。
- [Issue #19](https://github.com/troycheng/cuda-kernel-optimizer/issues/19)：在深入研究或实现前，先按当前候选的实际范围计算物理上限和端到端 Amdahl 上限。

优化器指令也据此收紧：高潜力候选失败时，需要提供与潜在收益相称的反证；拒绝结论只能关闭已经测试的实现和条件。

## 证据边界

私有多模态 workload、原始 trace、模型 artifact 和内部 runtime patch 未公开。本案例中的汇总测量已获授权发布，公开结果清单保留了可分享的环境与指标身份。结论只适用于记录中的 SM120、TP2、模型和 runtime 条件，不预测其他环境的收益，也不声称 MoE 的剩余优化空间已经穷尽。
