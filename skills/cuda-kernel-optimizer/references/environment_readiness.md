# 优化前的环境检查

`readiness.py check` 用于确认一次优化是否具备可运行、可校验、可比较的基础条件，并将这些条件冻结为 Target。它不会选择优化方向，也不会自动安装依赖。

## 优化 Target 必需输入

- original：待优化代码、binary 或部署快照；
- 测试集：真实数据集、代表性请求或可重复 replay；
- 精度校验：期望输出、容差或业务精度标准；
- command driver：读取封闭 request、执行一次 workload、写回封闭 result；
- 性能目标：主要指标、方向、最低有效收益和约束；
- 环境要求：目标 GPU 与必需工具；
- 统计要求：最小配对数量、置信度和采样规则；
- smoke：一次低成本 combined 精度与测量检查；
- 扫描上限：冻结文件的数量、总字节数和时间上限。

诊断 Target 可以绑定已有 profiler 报告而不提供 workload，但输出只能是诊断事实，不能形成完整 workload 性能结论。

## Command driver

从 `templates/` 中的协议和示例开始适配。driver 是唯一的 workload 运行接口，它必须：

- 严格校验 request 与 output path；
- 读取指定 Variant、测试集和精度 reference；
- 返回原始精度结果、性能样本、环境身份和清理状态；
- 不自行选择候选、改变测试范围或安装依赖；
- 不在未声明的位置留下后台任务。

长服务 driver 在首次 live workload 前还要核对服务 PID/PGID、ready probe、结果目录和 cleanup
方法，并在当前成本图中说明哪些准备是首次适配、哪些失败会形成可避免重试。readiness 只执行
一次明确 probe；它不接管服务生命周期或建立新的持久流程。

V1.4 optimization readiness 只接受 `execution_mode: "combined"` 和
`smoke.mode: "combined"`。smoke 请求两个样本，以验证 driver 确实执行 sampling；它不用于估计稳定分布。返回的 primary、constraints、unit 和样本数必须与 objective 和 sampling 精确闭合。额外诊断写入已声明 artifact 或日志，不能作为未声明 constraint 返回。separate driver 需要保存两次完整 probe evidence，延期到有版本化 Target contract 时再支持。

## 失败处理

以下问题应在修改代码前解决：

- original 本身无法通过精度校验；
- 测试集不能代表用户要优化的真实目标；
- driver 不能稳定产生原始样本；
- 环境身份或 GPU 资源无法确认；
- 关键工具缺失且没有等价证据路径；
- 测量噪声大到无法识别最低有效收益。

命令失败会在现有错误输出中保留 stop reason、returncode、截断 stdout/stderr 和 cleanup。readiness 成功只证明本次最小 probe 闭合，不证明之后的 baseline 分布稳定，也不证明共享宿主机不会被其他任务污染。

普通 Python 工具可在用户授权的隔离环境中安装。驱动、GPU 权限、时钟、功耗、服务和容器运行时等宿主机变化默认只给建议。修复测试脚本或依赖不是一次性能迭代，应单独说明耗时和剩余风险。

成功的 readiness 会生成 `target.json`、冻结 original object，并完成一次低成本 smoke。之后任何 operation 都必须显式引用同一个 Target。
