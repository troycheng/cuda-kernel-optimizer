# 项目演进试运行：Profiler 证据对象校验

状态：评测定义已冻结，尚未重放

## 案例快照

### 挑战视图

- 用户要求：候选 Profiler 采集只能使用与当前 Experiment、正确性结果和不可变证据对象一致的输入。
- 问题类型：确定性工具一致性缺陷。
- 影响范围：`workload_adapter.resolve_profile_collection()` 的候选前置校验。
- 正确行为：错误角色、未知 manifest 字段或被修改的 payload 必须在 Profiler 命令启动前拒绝。
- 隐私：案例全部来自公开 Git 历史和公开测试，不含用户 workload、trace、权重或内部环境。

### 审计视图

- 原始版本：`5211e832b6d5055ed316fe6fc77efa57813f5934`
- 修复版本：`9a3ff596907fcab7dd9abf4615bb080a1a2c2222`
- 历史发布：`v1.4.0`
- 已知限制：本次是历史修复后的回溯重放，不声称历史开发时已经预注册。

## 项目版本

- 比较轴只允许是上述两个 Git commit 的仓库内容。
- Python、操作系统和测试命令在后续记录中作为外部条件保存，不属于项目版本。

## 评测定义

- 类型：一致性评测。
- 比较臂 A：原始版本加同一份冻结测试文件。
- 比较臂 B：修复版本及其原始测试文件。
- 评测器来源：`9a3ff596907fcab7dd9abf4615bb080a1a2c2222:tests/test_workload_adapter.py`。
- 测试一：`test_candidate_collection_rejects_non_candidate_experiment_role`。
- 测试二：`test_candidate_collection_rejects_changed_evidence_payload`。
- 有效结果：A 中两个测试因未拒绝非法输入而失败；B 中两个测试通过。
- 混杂处理：两个 worktree 使用同一个 `python3`，不运行 GPU、不访问网络、不修改宿主机。
- 结论上限：L1 精确案例行为，加上当前分支回归可达到 L2 支持范围回归；不支持性能或跨任务通用性结论。
