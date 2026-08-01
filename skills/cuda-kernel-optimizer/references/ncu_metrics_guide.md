# Nsight Compute 指标说明

NCU 用于解释单个或一组 kernel 的硬件行为。它不能单独证明 workload 收益，也不应成为每个候选的固定步骤。

## 使用条件

- 已明确需要回答的问题，例如访存吞吐、计算管线、occupancy、spill 或 launch 配置；
- kernel dispatch 和输入 shape 已冻结；
- NCU 版本、报告来源和指标单位可确认；
- replay 开销与副作用可以接受。

`profile_ncu.py analyze` 只解析已支持的导出格式；`collect` 使用当前 Target 的 command driver 采集并导出 CSV。未知版本、关键字段或单位会拒绝生成语义观测；非关键扩展列和未建模指标会单独保留，不参与语义计算。原始 `.ncu-rep`、导出 CSV、工具身份和解析结果应同时保留。

## 常用观察

| 问题 | 典型观测 | 注意事项 |
|---|---|---|
| 是否受 DRAM 约束 | DRAM throughput、bytes、sector 和 memory pipe 活跃度 | cache 命中和请求合并会影响解释，不能只看一个百分比 |
| 是否受计算管线约束 | tensor、FP、integer 等 pipe utilization | 必须确认实际指令类型和可并行工作量 |
| occupancy 是否限制并发 | active warps、register、shared memory、block 限制原因 | 高 occupancy 不是目标，低 occupancy 也不自动意味着瓶颈 |
| 是否发生 spill | local load/store、register 数和 SASS | local traffic 还需区分参数、栈和真正 spill |
| 是否存在分支或尾块损耗 | warp efficiency、predication、分支、不同 shape 对比 | 首先保证尾块精度，不为整齐 shape 的收益牺牲边界正确性 |
| 是否存在异常访存 | transaction、sector、request、L1/L2/DRAM 关系 | 结论需与访问模式和生成代码对应 |

Roofline 只能描述当前 kernel 在既定假设下的上限关系。算术强度、峰值和吞吐的单位必须一致，且不能把理论上限直接当成候选收益预测。

## 权限失败

出现 `ERR_NVGPUCTRPERM` 时：

1. 记录 NCU 版本、命令、退出码和错误；
2. 将 counter 观测标为不可用，不伪造空值或沿用旧报告；
3. 根据问题改用 timing、Nsys、PyTorch Profiler、编译产物或 SASS；
4. 如确实需要 counter，向用户说明宿主机权限调整方法和风险，但不自行修改。

NCU 结果只有在与当前 Target、Variant、case、工具身份和环境匹配时才可用于本轮判断。
