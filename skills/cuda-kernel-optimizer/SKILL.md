---
name: cuda-kernel-optimizer
description: "Use when optimizing, tuning, diagnosing, or profiling CUDA, CUTLASS, Triton, PyTorch, vLLM, TensorRT-LLM, or another GPU workload; when assessing an NCU, Nsys, or PyTorch Profiler report; or when the test workload, correctness checks, measurement path, or target environment is incomplete."
---

# CUDA Kernel and Workload Optimizer

ChatGPT 负责优化判断：识别瓶颈、提出候选、评估投入产出并决定下一步。随 skill 安装的工具只执行一次明确操作，完成检查、测量、解析或事实记录；工具不选择方向，不判断 ROI，也不生成下一步计划。

优化目标是用户的完整 workload，而不是孤立的 kernel 指标。kernel 测量只能支持 kernel 层结论；要得出 workload 或服务层结论，必须先取得用户提供的真实测试集和精度校验。不得自行编造、下载或替换测试 workload。

当前 Target 的 primary 指标决定候选排序和任务是否完成。长尾或其它重要指标的局部收益可以在不违反约束时保留或显式纳入当前版本，但这是结果接纳规则，不是搜索优先级：它不能抬高一个低 primary ROI 候选的优先级，不能自动替换当前 Target 的 Champion，也不能把尚未实现的主目标报告为完成。

## 按需读取

只加载当前问题所需的脚本和 reference。

| 任务 | 脚本或资料 |
|---|---|
| 检查测试集、精度校验、driver、目标环境并冻结 Target | `scripts/readiness.py`；`references/environment_readiness.md` |
| 建立 Experiment，执行 baseline、screen、target 或 final audit | `scripts/workload_evaluate.py`；`references/performance_iteration.md` |
| 解析或采集 Nsight Compute 事实 | `scripts/profile_ncu.py`；`references/ncu_metrics_guide.md` |
| 解析或采集 Nsight Systems 事实 | `scripts/profile_nsys.py` |
| 解析或采集 PyTorch Profiler 事实 | `scripts/profile_pytorch.py` |
| 分析冻结的编译产物或显式 binary | `scripts/compiler_evidence.py`；`scripts/sass_check.py` |
| 查询内置离线知识 | `scripts/knowledge_query.py` |
| 查看、选择或恢复当前最佳版本 | `scripts/champion.py` |
| 判断服务测试是否可信 | `references/serving_evidence_protocol.md`；`references/nonstationary_serving_evidence.md` |
| 查阅最新的一手资料或进行外部质证 | `references/research_augmentation.md` |
| 构造各公开操作的封闭 JSON request | `references/request_protocol.md` |

先用对应脚本的 `--help` 确认 operation 和命令行参数，再按 `references/request_protocol.md`
构造封闭 JSON request。不要为确认输入格式而通读脚本源码。

## 优化流程

1. 冻结 Target：一个 primary、最低有效收益、不可退化 constraint/guardrail，以及测量分辨率的估计方法；在候选说明中另行预声明有业务价值的 secondary 及其门槛。同时明确允许修改的文件、风险边界、时间与 GPU 使用范围，以及宿主机是否只给建议。首次 live workload 前简要说明串行服务/GPU 生命周期、可并行只读工作、从 driver mode 推导的预计 live 调用数、当前完成点，以及首次适配成本与可避免重试；该说明不持久化。
2. 执行 readiness。优化 Target 必须冻结原始版本、真实测试集、精度规则、command driver、性能目标、环境身份和统计要求；诊断 Target 可以只绑定已有报告，但不能假装具备 workload。
3. 修改代码前先审计 primary 及每个决策指标的实际计算口径：明确分子、分母、数据来源，以及是否由文本重分词、日志采样或其它重建过程得到；用冻结 workload 的恒等关系检查 driver 输出。口径与 Target 不一致且无法修正时，结果只能作 diagnostic，不能用于阈值判断。完成审计后，修改代码前先测原始业务基线，并用预先声明的 original 重复样本估计当前环境的测量分辨率。精度未通过时，不解释性能样本。
4. 每次新 profiler 事实用于候选判断前，先核对 Target、Variant、case/request slice、并发、phase、coverage 和 claim layer，再完成系统级归因：说明主要 measured time 与未归因部分，比较可行的 subsystem 方向；coverage 已知时界定端到端收益上限，未知时明确标记。按“对 primary 的预期端到端收益、phase/关键路径匹配、coverage 可信度、可行性、风险和实验成本”排序候选；实现容易、局部增幅大或恰好缺一个配置都不能代替这个排序。coverage 判断必须与指标匹配：吞吐和均值可以用完整 workload 的时间占比界定收益上限；p95、p99 等长尾指标要看受影响请求、关键路径和 phase，不能仅因调用次数少或总耗时占比低而否定。候选成本或可行性没有证据时也明确标记，并给出证伪首选方向的最低成本观测。技术栈新颖或版本敏感、关键源码能力未知、存在竞争性解释、下一候选昂贵或连续候选无收益时，按 `references/research_augmentation.md` 完成有界的一手资料核验；只记录改变事实判断、候选排序或实验设计的研究增量。证据不适用时，只保留 diagnostic hypothesis。
5. 创建 Experiment 前完成源码静态审查或独立的最低成本证伪，并预声明失败最多能否定当前实现、集成方式、代理结论还是整个机制；拒绝范围不能越过证据能够支持的范围。已经证伪、primary 收益上限低于 minimum effect，或预期效果低于正式测量分辨率且既不服务于预先声明的重要 secondary、也没有明确合并计划时，不执行昂贵候选。AOT/CUTLASS 模板工作在首次构建前冻结同一机制的有界配置、实验开关、回退方式和构建次数，先用一个实验产物筛选，不逐个 knob 重复完整构建。随后冻结 Experiment，再依次完成精度校验和短版成对初筛。正确性、安全、dispatch 或环境失败只否定相应实现或结果；除非预先声明且实际取得了能够区分机制的证据，局部 proxy 或一次集成回退不能关闭整个机制。高 primary ROI 候选失败而机制仍未被区分时，先解释局部与完整 workload 结果的差额，并按外部研究规则质证；若存在一个不同且最低成本的判别实验，且其结果可能改变决定，才追加一次。重新尝试必须带来新证据、不同实现路径或不同测量设计。只有 profiler 能回答一个明确且尚未解决的问题时才运行它；进入正式 target 前无条件简短复核 Target、Variant、case/request slice、phase、coverage、收益上限、测量分辨率和 ROI，新 profile 事实则重新执行第 4 步的完整判断。
6. 本步只判断已经测出的结果是否值得保留，不改变候选搜索优先级。检查所有已声明且有业务价值的重要指标，不只看 primary。若改动在真实 workload 上稳定改善一项重要指标，例如 p95 或 p99，同时正确性通过且总体吞吐、平均延迟等 guardrail 未超过允许的退化范围，就将它纳入优化结果，记录为适用场景明确的局部结果；没有提升当前 primary 不能单独作为删除改动的理由。ChatGPT 可以依据预先声明的 secondary 和维护成本将其列入交付建议，但它不能成为当前 Target 的 Champion；必须同时记录 primary 未达成，也不能据此停止主目标搜索。测试后才发现的新收益先作为假设；要因该收益选择 Champion，必须重新冻结以该指标为 primary、其它重要指标为约束的 Target，避免从噪声中事后挑结果。最后将实测收益与不确定性同用户的最低有效收益、下一步时间和 GPU 成本比较；继续是否值得由 ChatGPT 判断，命令超时只负责防止工具卡死。
7. 正式结果有效且通过当前 Target 的 primary verdict 后，ChatGPT 才能显式选择候选。secondary-only 收益保留为局部结果；要让它成为 Champion，先以该指标为 primary 冻结新的 Target。需要 workload 或服务层最终结论时，再对当前最佳版本和 original 运行 final audit。
8. 每个终态都留下简短 Handoff，包含结论与证据、claim layer、当前 Target 的 primary 是否达成、Champion 或 Original、另行保留的局部结果、局部结果到端到端目标的解释、各被拒方向实际关闭到实现、集成还是机制、未覆盖风险、skill friction/feedback，以及 workspace 状态和停止原因。若在用户授权的时间或 GPU 上限明显未耗尽时停止，还要记录已用与剩余预算、当前候选族的覆盖边界，并重新做一次跨 subsystem 候选排序；对尚未关闭且预期 primary ROI 最高的方向，检查它是否被失败实现误杀，必要时按外部研究规则补充反例。只有该方向也有证据表明不值得或不能执行时，才形成 Target 的 terminal reason。当前候选族关闭不等于 Target 完成；局部结果存在不等于优化任务完成。

## 使用后反馈

将 Handoff 中的 `skill friction/feedback` 用于记录本次真实使用暴露、且可能改变 skill 指令、工具或知识的具体问题，以及值得保留的有效机制。每条只写观察到的行为、实际影响或成本、期望改动和最小证据；不要复述完整优化过程，不要把基本正确行为当作成果，也不要为填满字段添加通用风险。没有可行动反馈时写 `none`。

Handoff 默认留在用户环境。只有用户明确授权时才向外部仓库提交反馈；移除私有 workload、内部地址、凭据和未公开材料，并将相互独立的问题分别提交。优化结果或上游 PR 可以作为问题来源和证据，但本身不等于 skill feedback。

## 证据规则

- Candidate 和 reference 内容必须冻结，并由每份结果显式引用。
- 精度失败会使相应性能结果无效。
- 变体比较使用同一 Target、同一测试集和环境身份下的成对样本。
- profiler 只返回观测事实；它不返回优化方向、ROI 或下一步。
- 知识查询只返回匹配材料或空结果。空结果不会阻止源码分析、profiling 或 ChatGPT 自行提出假设。
- 未知 profiler 版本、关键字段或单位必须拒绝解析；已知格式中的非关键扩展内容只作为
  `unmodeled` 保留，不能参与语义计算。不能套用相近版本或相近架构。
- `ERR_NVGPUCTRPERM` 表示当前宿主机权限不允许读取 NCU counter。记录限制并改用其他有效证据；宿主机权限调整只给建议，除非用户明确授权。
- 共享宿主机在选卡和正式性能采样前启动确定性、低频、只读的 CPU/GPU 观测，持续到采样结束并保存时间对齐的原始输出。GPU 进程列表为空或显存占用很低都不能单独证明设备空闲；还要在有界窗口检查利用率、功耗和进程可见性，并解释持续活动。观测缺失或中断，或存在与正式样本窗口重叠、达到有依据阈值且未被解释的资源污染时，性能归因 inconclusive；瞬时、非重叠异常不否定 correctness。
- 编译产物结论不能越过其证据阶段。源码、AOT 包或 binary 中的字符串只证明文本存在；fusion、消除或调度等 post-lowering 主张必须来自绑定同一 Experiment 的 lowered/generated code、编译器精确匹配记录或 runtime kernel 证据。
- 使用子智能体时，主智能体是远端代码、GPU 实验和 artifact root 的唯一写者；只并行边界清楚的独立只读任务，使用最低足够能力，并说明模型、思考强度和目标。周期采样由确定性命令完成，正式共享 GPU 实验不并行。
- 保留原始报告、成对样本、环境身份、被拒候选和 terminal reason，使结论可复核。

## 外部资料

外部研究按决策风险有条件触发，不是每个候选的固定阶段。触发时先核验当前版本的一手资料；涉及高成本候选、高潜机制关闭或 Target 终止且本地证据仍有关键不确定性时，再使用至少一个可用的非 OpenAI 模型家族挑战假设和证据遗漏。ChatGPT Pro、Codex 和 OpenAI 子智能体不能互相算作异构质证。外部模型不批准候选，也不接管执行；发送前移除私有内容，并保留不同意见。外网或异构模型不可用不阻塞本地优化，但要记录未完成项和结论强度；最终仍由当前 Target 的本地精度与测量证据决定。

本 skill 不提供操作系统沙箱。只修改用户授权的文件和隔离环境。驱动、GPU 权限、频率、功耗、服务和容器运行时等宿主机变更，默认只提供建议。
