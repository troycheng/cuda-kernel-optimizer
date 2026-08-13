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

### 单机多进程启动

在加载模型或占用 GPU 前，用实际容器、shell、环境变量和 launcher 完成一次有界的 rendezvous
probe。单机 `torchrun` 在 hostname 的 IPv4/IPv6 解析和网络接口尚未实测证明时，不默认使用
`--standalone`；使用 `--master-addr=127.0.0.1`、启动前确认空闲并记录的显式
`--master-port`，Gloo 仍尝试解析 hostname 时再显式设置 `GLOO_SOCKET_IFNAME=lo`。多机任务不能
使用 loopback，必须从每个节点验证 master 地址和所选接口可达。

连续出现 `IPv6 network addresses ... cannot be retrieved`、`gai error`、hostname 解析失败或
rendezvous 超时，属于 launcher/readiness failure，不是 kernel、GPU 或性能结果。立即在有界时间内
停止，不用同一启动形式重试；改用已验证的显式 rendezvous 后重新 probe。远程 SSH 客户端退出不
代表远端进程已结束：启动前记录独立 PID/PGID，失败后只终止该进程组，等待退出并核对 launcher、
worker、资源监控和 GPU process 均无残留。最后把可工作的完整命令写回 command driver；后续
Experiment 不再重新探索已知失败的启动形式。

optimization readiness 要求 Driver V2 至少声明 `single_variant_combined`。smoke 请求两个样本，
验证 driver 确实执行 sampling；它不用于估计稳定分布。一次 evidence bundle 同时返回正确性和
性能事实，primary、constraints、unit 和样本数必须与 objective 和 sampling 精确闭合。只有
driver 确实能在同一进程内运行两个 subject 并保持所声明的共享状态时，才声明
`paired_same_process_combined`。V2 不兼容旧 Driver V1 Target。

## 失败处理

以下问题应在修改代码前解决：

- original 本身无法通过精度校验；
- 测试集不能代表用户要优化的真实目标；
- driver 不能稳定产生原始样本；
- 环境身份或 GPU 资源无法确认；
- 关键工具缺失且没有等价证据路径；
- 测量噪声大到无法识别最低有效收益。

共享 GPU 的“空闲”必须由一个短、有界、只读的观测窗口证明，而不是只看 compute PID 或显存。至少同时记录设备身份、利用率、显存、功耗和可见进程；PID 为空但持续高利用率属于未解释污染，必须换卡、等待或标为 inconclusive。低显存本身既不能证明空闲，也不能单独证明污染。多 GPU workload 还要在冻结设备集合前核对 NUMA、PCIe/NVLink 与 P2P 可达性。

命令失败会在现有错误输出中保留 stop reason、returncode、截断 stdout/stderr 和 cleanup。readiness 成功只证明本次最小 probe 闭合，不证明之后的 baseline 分布稳定，也不证明共享宿主机不会被其他任务污染。

普通 Python 工具可在用户授权的隔离环境中安装。驱动、GPU 权限、时钟、功耗、服务和容器运行时等宿主机变化默认只给建议。修复测试脚本或依赖不是一次性能迭代，应单独说明耗时和剩余风险。

成功的 readiness 会生成 `target.json`、冻结 original object，并完成一次低成本 smoke。之后任何 operation 都必须显式引用同一个 Target。
