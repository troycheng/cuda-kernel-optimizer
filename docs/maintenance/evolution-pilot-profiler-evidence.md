# 项目演进试运行：Profiler 证据对象校验

状态：回溯重放完成；评测定义已由提交 `9c8cdaf6f24dbeccc47526288eae95bc879ff4f6` 冻结

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

## 评测结果

### 实际身份

- 原始版本：`5211e832b6d5055ed316fe6fc77efa57813f5934`
- 修复版本：`9a3ff596907fcab7dd9abf4615bb080a1a2c2222`
- 评测器：`9a3ff596907fcab7dd9abf4615bb080a1a2c2222:tests/test_workload_adapter.py`
- Python：`Python 3.9.6`
- GPU：未使用
- 网络：未使用
- 原始 worktree 的生产源码保持在原始版本，只替换了上述测试文件。

### 事实

两个比较臂使用相同命令：

```bash
python3 -m unittest \
  tests.test_workload_adapter.ProfileCollectionBindingTests.test_candidate_collection_rejects_non_candidate_experiment_role \
  tests.test_workload_adapter.ProfileCollectionBindingTests.test_candidate_collection_rejects_changed_evidence_payload
```

| 比较臂 | 结果 | 实际观测 |
|---|---|---|
| 原始版本 | return code 1，2/2 失败 | 两个测试均报告 `ValueError not raised`；错误角色和被修改的 payload 没有被拒绝 |
| 修复版本 | return code 0，2/2 通过 | 两类非法输入均在候选 Profiler 前置校验中被拒绝 |
| 修复版本完整适配器测试 | return code 0，10/10 通过 | `python3 -m unittest tests.test_workload_adapter` 通过 |

第一次执行请求因临时目录清理使用递归删除而被本地安全策略在进程启动前拒绝。该请求没有创建 worktree，也没有运行评测，不属于有效或失败试验。改为 `git worktree remove` 和空目录删除后，评测定义、源码、测试和比较条件均未改变。

临时 worktree 在命令退出时全部清理，主工作区没有残留评测文件。

### 结论范围

- 可以确认：在这两个公开复现中，原始版本没有拒绝错误候选角色和被修改的证据 payload，修复版本会拒绝；修复版本当时的完整适配器测试同时通过。
- 不能确认：GPU 性能变化、其它证据类型、跨模型行为、未声明环境和通用优化能力。
- 回溯限制：评测定义晚于历史修复，只能证明当前可重现该差异，不能声称历史开发时已经预注册。

## 发布决定

- 历史决定：修复版本随 `v1.4.0` 发布。
- 当前审计：只确认公开复现和回归结果，不重新制造一次发布。
- 维护责任：未来发现反例时形成新案例和新评测，不改写本次结果。
