---
name: cuda-kernel-optimizer
description: "Use when optimizing, tuning, diagnosing, or profiling CUDA, CUTLASS, Triton, PyTorch, vLLM, TensorRT-LLM, or another GPU workload; when assessing an NCU, Nsys, or PyTorch Profiler report; or when the test workload, correctness checks, measurement path, or target environment is incomplete."
---

# CUDA Kernel and Workload Optimizer

ChatGPT 负责优化判断：识别瓶颈、提出候选、评估投入产出并决定下一步。随 skill 安装的工具只执行一次明确操作，完成检查、测量、解析或事实记录；工具不选择方向，不判断 ROI，也不生成下一步计划。

优化目标是用户的完整 workload，而不是孤立的 kernel 指标。kernel 测量只能支持 kernel 层结论；要得出 workload 或服务层结论，必须先取得用户提供的真实测试集和精度校验。不得自行编造、下载或替换测试 workload。

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

1. 明确性能目标、最低有效收益、允许修改的文件、风险边界、时间与 GPU 使用范围，以及宿主机是否只给建议。
2. 执行 readiness。优化 Target 必须冻结原始版本、真实测试集、精度规则、command driver、性能目标、环境身份和统计要求；诊断 Target 可以只绑定已有报告，但不能假装具备 workload。
3. 修改代码前先测原始业务基线。精度未通过时，不解释性能样本。
4. 结合源码与已有观测分析 kernel、launch、框架、CPU/数据、传输、通信、I/O、服务和环境因素。ChatGPT 保留竞争性假设，优先选择能以最低成本证伪主要假设的观测。
5. 创建 Experiment 前完成源码静态审查或独立的最低成本证伪；已经证伪时不执行候选。随后冻结 Experiment，再依次完成精度校验和短版成对初筛。只有 profiler 能回答一个明确且尚未解决的问题时才运行它；前序证据仍有效时才进入正式成对测试。
6. 将实测收益与不确定性同用户的最低有效收益、下一步时间和 GPU 成本比较。继续是否值得由 ChatGPT 判断，命令超时只负责防止工具卡死。
7. 正式结果有效后，ChatGPT 可以显式选择候选。需要 workload 或服务层最终结论时，再对当前最佳版本和 original 运行 final audit。

## 证据规则

- Candidate 和 reference 内容必须冻结，并由每份结果显式引用。
- 精度失败会使相应性能结果无效。
- 变体比较使用同一 Target、同一测试集和环境身份下的成对样本。
- profiler 只返回观测事实；它不返回优化方向、ROI 或下一步。
- 知识查询只返回匹配材料或空结果。空结果不会阻止源码分析、profiling 或 ChatGPT 自行提出假设。
- 未知 profiler 版本、关键字段或单位必须拒绝解析；已知格式中的非关键扩展内容只作为
  `unmodeled` 保留，不能参与语义计算。不能套用相近版本或相近架构。
- `ERR_NVGPUCTRPERM` 表示当前宿主机权限不允许读取 NCU counter。记录限制并改用其他有效证据；宿主机权限调整只给建议，除非用户明确授权。
- 保留原始报告、成对样本、环境身份、被拒候选和 terminal reason，使结论可复核。

## 外部资料

外部搜索和第三方 AI 质证是可选的研究手段，适合核实一手资料、困难方向、明显平台期或最终审查。发送前移除私有内容，并保留不同意见。外网不可用不妨碍本地优化；是否接受候选仍由本地精度与测量证据决定。

本 skill 不提供操作系统沙箱。只修改用户授权的文件和隔离环境。驱动、GPU 权限、频率、功耗、服务和容器运行时等宿主机变更，默认只提供建议。
