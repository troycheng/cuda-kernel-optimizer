<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/logo-wordmark-dark.svg">
    <img src="asset/logo-wordmark.svg" width="520" alt="CUDA Kernel Optimizer">
  </picture>
</p>

<p align="center"><strong>以证据驱动 ChatGPT 优化 CUDA、CUTLASS 与 Triton</strong></p>

<p align="center">
  <a href="docs/getting-started.md">快速开始</a> ·
  <a href="docs/environment-readiness.md">准备 Workload</a> ·
  <a href="docs/workflows.md">工作流</a> ·
  <a href="docs/evidence-and-safety.md">证据与安全</a> ·
  <a href="skills/cuda-kernel-optimizer/examples/walkthrough.md">示例</a> ·
  <a href="README.md">English</a>
</p>

## 项目简介

`cuda-kernel-optimizer` 是一个面向 ChatGPT 编程代理的 GPU 性能优化 skill。
它可以优化 CUDA、CUTLASS 或 Triton kernel，排查完整 workload 的瓶颈，验证修改
能否改善 serving 指标，也可以在不重新运行原程序的情况下分析已有 Nsight Compute report。

Skill 会在真实目标上做 profiling，只修改限定的项目路径，先检查正确性，再比较成对
性能数据。瓶颈不在 kernel 时，它还会检查框架调度、CPU 和数据处理、传输、通信、I/O、
allocator 与运行状态。

确定性 Controller 会在优化前冻结目标、环境、预算、测量规则和修改范围。
可恢复的主动诊断闭环会先检查必需能力，再比较能被实测推翻的竞争解释，
只执行当前决策真正需要的证据动作。签名证据与只增不改的账本防止任务中断、
噪声升高或身份漂移时悄悄改变实验，也避免等价采集重复消耗预算。

第一次全局扫描后，分析引擎会建立性能模型，算清关键路径、收益上限和仍未确认的
问题。昂贵 profiling 开始前，用户就能看到一份投入判断：目前知道什么、下一步只做
哪一项、成本属于什么量级，以及是否值得继续投入。
baseline 前的环境准入会自动执行：AI 先确认编译、GPU、profiler 和 workload smoke 等必需能力。
真实 workload 和明确授权仍由用户提供。带哈希锁定的隔离环境 pip 是唯一允许自动执行的修复，
宿主机改动只给建议。`self_check` 通过不代表 GPU 环境已经可用。

Skill 不会自动修改宿主机配置。驱动、counter 权限、频率、功耗限制、服务和系统设置
都只给建议，除非用户另行明确授权。

## 快速开始

安装由 ChatGPT 的编程代理完成，读者不需要手工运行项目脚本。在 ChatGPT 编程会话中发送：

> 从 [troycheng/cuda-kernel-optimizer](https://github.com/troycheng/cuda-kernel-optimizer) 的最新发布版本安装 `skills/cuda-kernel-optimizer`。只把这个 skill 安装到当前 skills 目录，执行 CPU/static `self_check`，并报告安装标签、commit 和目标目录。除非我明确要求，否则不要使用 `main`。

安装完成后开启新会话，让新指令生效。

正式使用 45 分钟的 `quick` 预算前，先做一次 **10 分钟适配检查**：

> 使用 cuda-kernel-optimizer 对当前项目做只读适配检查，最多用 10 分钟。不要修改源码、安装依赖或调整宿主机。确认可运行目标、正确性 reference、benchmark、目标 GPU 和 profiler 权限。报告当前能支持的结论、阻塞项、缺失证据和最低成本的下一步，不产出提速结论。

这一步只判断项目是否具备优化条件，不产出提速结论。真实 workload 必须由用户提供，
skill 不会自行下载或编造。基础条件满足后，再提供性能目标、约束和允许修改的范围，
并选择 `quick`、`balanced` 或 `thorough`。

AI 会冻结任务，运行项目原始 baseline，按照从低成本检查到昂贵测试的顺序评估候选，
并恢复被否决的修改。结束时必须报告准确的运行目录。先看 `summary.md`，需要机器可读
结论时查看 `itervN/decision.json`。只有真实 workload 目标、正确性、约束和证据完整性
全部通过，修改才适合合入。

`quick` 最长 45 分钟，`balanced` 是默认的 3 小时，`thorough` 最长 10 小时。
证据已经明确或没有值得继续的方向时，任务会提前结束。

> 使用 cuda-kernel-optimizer 优化这个 Triton workload。先确认 reference、真实输入、目标指标、允许修改的文件和目标环境。保持宿主机设置不变，只有正确性与成对性能证据都通过时才保留修改。

完整的首次运行流程见[快速开始](docs/getting-started.md)。

## 选择工作流

| 工作流 | 适用场景 | 能支持的结论 |
|---|---|---|
| **环境准备** | 缺少 workload、reference、benchmark、profiler 或目标环境 | 缺口、当前结论上限和项目内准备方案 |
| **Kernel 优化** | CUDA、CUTLASS 或 Triton 实现已有可比较 reference | 带正确性与成对测量证据的 kernel 级结论 |
| **完整 workload** | 瓶颈可能横跨 GPU、框架、CPU、传输、通信、I/O 或运行状态 | 对用户 workload 的限定范围诊断与端到端评测 |
| **Serving 验证** | 需要确认局部修改能否改善产品 KPI | 冻结 c1/c2/c4/c8/c12 分层、约束、运行身份，并分别判定性能和证据完整性 |
| **已有 NCU report** | 已有 `.ncu-rep`，不能重新运行原 workload | 只读分析；无法解析时准确记录降级原因 |

[工作流说明](docs/workflows.md)列出各路径需要的输入和结论范围。
[长任务优化](docs/long-running-optimization.md)说明 Controller、能力库、
校准、周期 audit 和恢复方式。

## 工作方式

```mermaid
flowchart LR
    goal["目标、代码和约束"] --> environment["检查测试环境"]
    environment --> baseline["冻结并校准 baseline"]
    baseline --> context["建立执行图和证据目录"]
    context --> hypothesis["提出可证伪的竞争解释"]
    hypothesis --> evidence["选择并执行最有区分力的证据动作"]
    evidence --> hypothesis
    hypothesis --> change["证据充分：创建限定范围的修改"]
    change --> evaluation["检查正确性和成对性能"]
    evaluation --> keep["证据充分：保留修改"]
    evaluation --> restore["证据不足：恢复原实现"]
```

正式计时前，Controller 会冻结目标与授权范围，估计测量噪声和最小可检测效应。
`green` 允许进入候选实验，`yellow` 暂停并改善测量或重放 baseline，`red` 停止任务。
合同还会限制两次 baseline audit 之间最多能运行多少候选。

AI 每轮最多提出 3 个竞争机制，Controller 负责计算事实，并且只返回一种决定：
`MEASURE`、`PURSUE`、`REVIEW_REQUIRED` 或 `STOP`。后续只能执行该决定指定的一项动作。
收益上限、不确定性、成本等级和下一个检查点会写入
`active_diagnosis/investment_brief.json`。只有同一动作存在身份一致的历史记录时，
才会给出具体耗时估计。

通过验证的观察只检索少量匹配的 capability card。能力卡提供方法、反例和检查办法，
不负责判定结果。每轮都从一个能被实测推翻的性能假设开始；只有重新校验通过的 V2.5
证据闭环才算真正评估过候选。环境准备在优化计时前完成；3 分钟或总预算 10% 只是检查
进展的时点，不会自动中止安装或修复。只有实际命令超时或达到 readiness 硬截止时间时，
Controller 才终止整个进程组。修工具不等于性能提升。

方向是否值得继续见[方向准入约束](skills/cuda-kernel-optimizer/references/direction_admission.md)，
候选迭代规则见[性能优先约束](skills/cuda-kernel-optimizer/references/performance_iteration.md)。

## 以证据为准，而不是选择最快样本

性能结论需要同时满足：

- 正确性和所有声明约束通过；
- 成对 A/B 样本遵守冻结的 schedule 与 aggregation 规则；
- 默认 95% 置信区间支持要求的效应，并且有效 pair 足够；
- continuous shared-host guard 完整覆盖正式计时，不缺采样、不过期、不受污染；
- 正式 serving 证据覆盖 c1/c2/c4/c8/c12，并把 measured binary 绑定到真实 execution path。

必需证据缺失、互相矛盾、受到污染、过期或身份不符时必须 fail closed。
`performance_verdict` 与 `evidence_integrity` 分开判定：更快的数字不能补救无效实验。
安装后的 `self_check` 只执行 CPU/static 检查，不验证 GPU 环境。

进一步说明见[证据与安全](docs/evidence-and-safety.md)、
[V2.5 正式证据参考](skills/cuda-kernel-optimizer/references/evidence_automation.md)和
[长任务 Controller 参考](skills/cuda-kernel-optimizer/references/long_running_control.md)。

## 验证情况

[验证情况](docs/validation.md)记录自动化检查、物理 RTX 5090 路径、工具权限，以及
使用真实 GPU pair 得到的稳定性结论。[案例](docs/case-studies.md)单独记录历史
workload 结果。两者都不承诺新项目能获得相同提速。

## 版本记录

### V1.1.0

- 新增确定性性能模型，统一计算关键路径、各层优化空间、缺失证据和同类动作耗时。
- 主动诊断最多保留 3 个竞争机制，每轮只允许一项下一步动作，并明确输出
  `MEASURE`、`PURSUE`、`REVIEW_REQUIRED` 或 `STOP`。
- 昂贵 profiling 前先生成投入判断；没有方向达到项目效果阈值时直接停止。
- 新证据可更新性能模型；仍在验证的假设保持身份不变，已有证据关闭的机制可以被
  新方向替换，但不能改名后重新进入候选池。
- 完整保留外部质证内容并记录本地证据如何处理；在 RTX 5090 上验证 4 类受控
  fixture 的 Controller 证据准入与状态转换。机制假设由 fixture 提供，不代表系统已能
  从未知 profile 自主完成诊断；另提供单独授权、用完即销毁的 NCU smoke 路径。

### V1.0.1

- 安装后的 skill 现在包含 `LICENSE` 与 `NOTICE`。
- 物理 GPU 验收路径改为参数配置，不再绑定维护者目录。
- `open-iter` 统一使用 hard deadline，并持久记录实际耗时。
- 明确区分独立项目版本和保留的 pre-V1 协议标识。

### V1.0.0

首个独立公开版本整合了环境准备、主动诊断、限定范围的代码修改、分阶段的
正确性与性能检查、证据封存和确定性长任务恢复。只有低成本检查通过后才会进入
更昂贵的阶段；只有声明的 workload 目标支持修改时才会保留候选。物理 GPU 验证说明
机制和目标机路径可用，不预测新 workload 能获得多少提速。

## 文档

- [快速开始](docs/getting-started.md)
- [准备 workload](docs/environment-readiness.md)
- [工作流选择](docs/workflows.md)
- [长任务优化](docs/long-running-optimization.md)
- [证据与安全](docs/evidence-and-safety.md)
- [兼容性](docs/compatibility.md)
- [验证情况](docs/validation.md)与[案例](docs/case-studies.md)
- [知识、搜索与独立质证](docs/knowledge-and-research.md)
- [AI 执行协议](skills/cuda-kernel-optimizer/SKILL.md)与[示例](skills/cuda-kernel-optimizer/examples/walkthrough.md)
- [性能迭代](skills/cuda-kernel-optimizer/references/performance_iteration.md)、[方向准入](skills/cuda-kernel-optimizer/references/direction_admission.md)和[长任务控制](skills/cuda-kernel-optimizer/references/long_running_control.md)
- [软件栈版本对照](skills/cuda-kernel-optimizer/references/version_stack_audit.md)
- [正式证据](skills/cuda-kernel-optimizer/references/evidence_automation.md)与[兼容性参考](skills/cuda-kernel-optimizer/references/compatibility.md)
- [RTX 5090 opt-in 测试说明](tests/gpu/sm120/README.md)
- [MIT License](LICENSE)

本项目独立于 CUDA、CUTLASS、Triton 和 Nsight Compute。相关依赖遵循各自许可证。
