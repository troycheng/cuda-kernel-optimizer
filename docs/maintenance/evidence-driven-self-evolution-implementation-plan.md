# 证据驱动的项目演进首期实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法跟踪进度。

**目标：** 用一个真实、公开、可复查的确定性工具缺陷走通项目演进流程，再据此发布最小贡献模板和校验门禁。

**架构：** 首期只增加仓库维护材料和公开贡献入口。一次回溯案例先固定案例、比较对象和评测方法，再重放旧版本与修复版本；确认流程可用后，才提取 Markdown 模板，并用一个小型 `unittest` 检查公开材料。V1.4 运行时、安装包、知识库和优化决策方式保持不变。

**技术栈：** Markdown、Git worktree、Python 3 标准库、`unittest`、MkDocs。

---

## 1. 范围和文件职责

### 创建

- `docs/maintenance/evolution-pilot-profiler-evidence.md`：首个回溯案例的冻结定义、实际结果和证据限制；先提交定义，再追加结果。
- `docs/project-evolution.md`：面向用户和贡献者的中文主文档，只说明什么时候值得贡献、需要哪些材料和如何完成一次评测。
- `docs/project-evolution.en.md`：与中文主文档语义一致的英文版。
- `docs/evolution-case-profiler-evidence-validation.md`：从首个回溯案例提炼出的公开示例；明确它是确定性工具修复，不是性能案例。
- `.github/evolution/case-snapshot.md`：案例快照模板。
- `.github/evolution/evaluation-definition.md`：在结果产生前提交的评测定义模板。
- `.github/evolution/evaluation-result.md`：只记录事实和结论上限的结果模板。
- `.github/evolution/release-decision.md`：维护者在发布、拒绝、撤回或回滚时使用的决定模板。
- `tests/test_project_evolution.py`：检查公开文档、模板、案例、隐私边界和 V1.4 职责边界；不解析用户私有材料，不成为新的运行时校验器。

### 修改

- `docs/index.md`：增加项目演进文档入口。
- `mkdocs.yml`：把中英文项目演进文档和公开案例加入导航。
- `README.md`：在“进一步阅读”中加入中文项目演进入口。
- `README.en.md`：在“Further reading”中加入英文项目演进入口。
- `CONTRIBUTING.md`：说明普通修复、项目演进案例和私有实践派生贡献的区别。
- `.github/pull_request_template.md`：增加适用于行为、知识或工具语义变化的证据检查项。
- `tests/test_public_docs.py`：把新增公开页面纳入链接和导航检查。
- `tests/test_standalone_project.py`：检查四个贡献模板及贡献入口存在。

### 明确不修改

- `skills/cuda-kernel-optimizer/SKILL.md`；
- `skills/cuda-kernel-optimizer/scripts/` 下的 17 个生产模块；
- `skills/cuda-kernel-optimizer/references/knowledge/`；
- V1.4 Target、Experiment、Invocation、Champion 和 handoff；
- CI checkout 深度、GPU CI 或自动执行社区附件的逻辑。

首期不增加 JSON schema、数据库、Controller、队列、自动知识准入、自动评测、自动 PR、自动合并或自动发布。若人工案例没有暴露稳定重复的机械错误，不再增加独立 validator。

## 2. 首个案例和结论上限

首个案例使用仓库已经公开的 profiler 证据校验缺陷：

- 比较基线：`5211e832b6d5055ed316fe6fc77efa57813f5934`；
- 候选版本：`9a3ff596907fcab7dd9abf4615bb080a1a2c2222`；
- 发布载体：`v1.4.0`；
- 缺陷一：候选 Experiment 的 Variant 角色没有被强制为 `candidate`；
- 缺陷二：correctness evidence object 只检查 manifest，没有通过对象存储完整物化并核对 payload；
- 评测类型：一致性评测；
- 评测器：候选提交中的两个聚焦回归测试，原样运行在两个源码版本上；
- 允许结论：候选版本在这两个公开复现上拒绝了非法对象，基线没有；
- 禁止结论：不能据此声称提高了 GPU 性能、覆盖全部 evidence object、证明通用行为能力或证明当时已预注册评测。

这是回溯案例。评测定义将在本轮重放前冻结，但晚于历史修复本身，因此只能验证当前能够重现差异，不能把它包装成当时的前瞻性评测。

## 3. 任务拆分

### 任务 1：在看到重放结果前冻结案例和评测定义

**文件：**

- 创建：`docs/maintenance/evolution-pilot-profiler-evidence.md`

- [ ] **步骤 1：写入案例身份和挑战视图**

文档必须先包含以下内容，不写任何测试结果：

```markdown
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
- Python、操作系统和测试命令在评测结果中记录为外部条件，不属于项目版本。

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
```

- [ ] **步骤 2：确认文档没有提前写入结果**

运行：

```bash
rg -n "评测结果|实际通过|实际失败|return code" docs/maintenance/evolution-pilot-profiler-evidence.md
```

预期：没有匹配。

- [ ] **步骤 3：检查格式并提交冻结定义**

运行：

```bash
git diff --check
git add docs/maintenance/evolution-pilot-profiler-evidence.md
git commit -m "docs: 冻结首个项目演进评测"
```

预期：格式检查通过；提交只包含这一份文档。

### 任务 2：用同一评测器重放旧版本和修复版本

**文件：**

- 修改：`docs/maintenance/evolution-pilot-profiler-evidence.md`
- 临时：两个 detached Git worktree，验证结束后删除

- [ ] **步骤 1：创建干净的比较目录**

运行：

```bash
pilot_root="$(mktemp -d)"
git worktree add --detach "$pilot_root/original" 5211e832b6d5055ed316fe6fc77efa57813f5934
git worktree add --detach "$pilot_root/revision" 9a3ff596907fcab7dd9abf4615bb080a1a2c2222
git -C "$pilot_root/original" restore --source 9a3ff596907fcab7dd9abf4615bb080a1a2c2222 -- tests/test_workload_adapter.py
```

预期：两个 worktree 创建成功；原始版本只替换评测器文件，生产源码仍为 `5211e83`。

- [ ] **步骤 2：核对唯一有意变化的代码身份**

运行：

```bash
git -C "$pilot_root/original" rev-parse HEAD
git -C "$pilot_root/revision" rev-parse HEAD
git -C "$pilot_root/original" diff --name-only
python3 --version
```

预期：依次输出 `5211e83...`、`9a3ff59...`；原始 worktree 只有 `tests/test_workload_adapter.py` 被评测器覆盖；记录实际 Python 版本。

- [ ] **步骤 3：运行原始版本**

在 `$pilot_root/original` 运行：

```bash
python3 -m unittest \
  tests.test_workload_adapter.ProfileCollectionBindingTests.test_candidate_collection_rejects_non_candidate_experiment_role \
  tests.test_workload_adapter.ProfileCollectionBindingTests.test_candidate_collection_rejects_changed_evidence_payload
```

预期：FAIL；两个测试至少有一个明确显示原始版本没有拒绝预期的非法输入。若测试因导入、fixture 或其它环境原因失败，结果记为 `invalid`，停止比较，不修改测试适配结果。

- [ ] **步骤 4：运行修复版本**

在 `$pilot_root/revision` 运行相同命令。

预期：`Ran 2 tests` 和 `OK`。若不通过，记录实际失败并停止，不调整评测定义或测试来追求通过。

- [ ] **步骤 5：运行候选版本的完整适配器测试**

在 `$pilot_root/revision` 运行：

```bash
python3 -m unittest tests.test_workload_adapter
```

预期：该提交当时的完整 `test_workload_adapter` 测试通过。实际测试数量写入结果，不预先硬编码。

- [ ] **步骤 6：清理 detached worktree**

运行：

```bash
git worktree remove --force "$pilot_root/original"
git worktree remove --force "$pilot_root/revision"
rmdir "$pilot_root"
git worktree prune
```

预期：临时目录和 worktree 注册均被移除。

- [ ] **步骤 7：只按实际输出追加评测结果和发布决定**

追加以下结构，字段值使用刚才实际观察到的命令、Python 版本、return code 和测试数量：

```markdown
## 评测结果

### 实际身份

- 原始版本：完整 commit
- 修复版本：完整 commit
- 评测器：完整 commit 与测试文件路径
- Python：实际版本
- GPU/网络：未使用

### 事实

| 比较臂 | 命令 | 结果 | 说明 |
|---|---|---|---|
| 原始版本 | 两个聚焦测试 | 实际结果 | 实际失败原因 |
| 修复版本 | 两个聚焦测试 | 实际结果 | 实际通过或失败原因 |
| 修复版本 | 完整适配器测试 | 实际结果 | 实际测试数量 |

### 结论范围

- 可以确认的结论：只写实际证据直接支持的窄结论。
- 不能确认的结论：GPU 性能、全部证据类型、跨模型行为和通用性。
- 回溯限制：评测定义晚于历史修复，不能声称当时已经预注册。

## 发布决定

- 历史决定：该修复随 `v1.4.0` 发布。
- 当前审计：只确认公开复现和回归结果，不重新制造一次发布。
- 维护责任：未来发现反例时形成新案例，不改写本次结果。
```

- [ ] **步骤 8：提交不可变结果**

运行：

```bash
git diff --check
git add docs/maintenance/evolution-pilot-profiler-evidence.md
git commit -m "docs: 记录首个项目演进回放结果"
```

预期：只追加实际结果，没有修改已冻结的评测定义。

### 任务 3：先写公开贡献面和轻量门禁的失败测试

**文件：**

- 创建：`tests/test_project_evolution.py`
- 修改：`tests/test_public_docs.py`
- 修改：`tests/test_standalone_project.py`

- [ ] **步骤 1：创建聚焦测试文件**

写入：

```python
from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "case-snapshot.md",
    "evaluation-definition.md",
    "evaluation-result.md",
    "release-decision.md",
}


class ProjectEvolutionTests(unittest.TestCase):
    def test_public_guides_keep_the_six_concepts_and_v14_boundary(self) -> None:
        chinese = (ROOT / "docs/project-evolution.md").read_text("utf-8")
        english = (ROOT / "docs/project-evolution.en.md").read_text("utf-8")
        for marker in (
            "演进契约", "案例快照", "项目版本", "评测定义", "评测结果", "发布决定",
            "ChatGPT", "确定性工具", "不修改 V1.4 运行时",
        ):
            self.assertIn(marker, chinese)
        for marker in (
            "Evolution Contract", "Case Snapshot", "Project Revision",
            "Evaluation Definition", "Evaluation Result", "Release Decision",
            "ChatGPT", "deterministic tools", "does not modify the V1.4 runtime",
        ):
            self.assertIn(marker, english)
        self.assertIn("不自动合并", chinese)
        self.assertIn("does not automatically merge", english.lower())

    def test_templates_are_small_markdown_records_not_a_new_schema(self) -> None:
        root = ROOT / ".github/evolution"
        self.assertEqual({path.name for path in root.iterdir()}, TEMPLATES)
        joined = "\n".join(
            (root / name).read_text("utf-8") for name in sorted(TEMPLATES)
        )
        for marker in (
            "Private material", "Claim ceiling", "Repository revision",
            "Evaluator identity", "Actual result", "Human decision",
        ):
            self.assertIn(marker, joined)
        self.assertNotIn("next_action", joined)
        self.assertNotIn("current_state", joined)
        self.assertNotIn("knowledge_weight", joined)

    def test_public_case_is_narrow_replayable_and_not_a_performance_claim(self) -> None:
        text = (ROOT / "docs/evolution-case-profiler-evidence-validation.md").read_text(
            "utf-8"
        )
        for marker in (
            "5211e832b6d5055ed316fe6fc77efa57813f5934",
            "9a3ff596907fcab7dd9abf4615bb080a1a2c2222",
            "v1.4.0",
            "retrospective",
            "not a performance case",
            "test_candidate_collection_rejects_changed_evidence_payload",
        ):
            self.assertIn(marker, text)
        self.assertIsNone(re.search(r"\b\d+(?:\.\d+)?%\b", text))

    def test_public_materials_do_not_admit_private_evidence(self) -> None:
        paths = [
            ROOT / "docs/project-evolution.md",
            ROOT / "docs/project-evolution.en.md",
            ROOT / "docs/evolution-case-profiler-evidence-validation.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / ".github/pull_request_template.md",
        ]
        joined = "\n".join(path.read_text("utf-8") for path in paths)
        self.assertIn("private workload", joined.lower())
        self.assertIn("must still stand", joined.lower())
        self.assertIn("human", joined.lower())
        self.assertNotIn("private evidence queue", joined.lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：扩充公共文档清单**

在 `tests/test_public_docs.py` 的 `PUBLIC_PAGES` 中加入：

```python
    "docs/project-evolution.md",
    "docs/project-evolution.en.md",
    "docs/evolution-case-profiler-evidence-validation.md",
```

在 `test_evidence_knowledge_and_validation_boundaries_are_explicit` 之后增加一个方法，检查 `docs/project-evolution.md` 链接公开案例和英文版。

- [ ] **步骤 3：扩充社区文件清单**

在 `tests/test_standalone_project.py::test_community_files_are_present` 的 `relative` 元组中加入：

```python
            ".github/evolution/case-snapshot.md",
            ".github/evolution/evaluation-definition.md",
            ".github/evolution/evaluation-result.md",
            ".github/evolution/release-decision.md",
```

并检查 `CONTRIBUTING.md` 包含 `Project evolution`，Pull Request 模板包含 `private material` 和 `Evaluation Definition`。

- [ ] **步骤 4：运行聚焦测试，确认按预期失败**

运行：

```bash
python3 -m unittest tests.test_project_evolution tests.test_public_docs tests.test_standalone_project
```

预期：FAIL，原因只应是尚未创建公开指南、案例、模板或导航；不能出现现有 V1.4 测试退化。

### 任务 4：从实际案例提取最小公开指南和模板

**文件：**

- 创建：`docs/project-evolution.md`
- 创建：`docs/project-evolution.en.md`
- 创建：`docs/evolution-case-profiler-evidence-validation.md`
- 创建：`.github/evolution/case-snapshot.md`
- 创建：`.github/evolution/evaluation-definition.md`
- 创建：`.github/evolution/evaluation-result.md`
- 创建：`.github/evolution/release-decision.md`
- 修改：`docs/index.md`
- 修改：`mkdocs.yml`

- [ ] **步骤 1：编写中文主文档**

`docs/project-evolution.md` 只保留以下结构：

```markdown
# 项目如何从真实使用中改进

English 链接

## 这套方法解决什么问题
用一段话说明：真实任务暴露问题，但单个案例不能自动成为通用规则。

## 不会发生什么
明确不在运行中自改、不上传私有材料、不自动接纳知识、不自动合并或发布、不改变 V1.4 运行时。

## 六个记录分别做什么
用一张六行表解释演进契约、案例快照、项目版本、评测定义、评测结果和发布决定。

## 一次贡献怎样完成
任务结束 -> 人决定整理 -> 案例快照 -> 项目修改 -> 冻结评测 -> 评测结果 -> 人决定发布。
说明箭头不是状态机，没有对象自动调用下一步。

## 哪些内容适合贡献
列出公开复现、兼容边界、测试、代码、文档和公开性能案例。

## 私有实践怎样帮助项目
明确私有材料留在本地；公开改动删去“内部验证”声明后仍须独立成立。

## 需要多少证据
区分确定性工具的一致性评测与 ChatGPT 行为评测；局部证据只支持局部声明。

## 模板和示例
链接四个模板与公开 profiler 证据校验案例。
```

正文控制在 180 行以内，不复制 367 行维护设计，不引入新术语。

- [ ] **步骤 2：编写语义一致的英文版**

`docs/project-evolution.en.md` 使用相同章节顺序和相同边界。英文版不逐句直译，以自然技术文档语言表达；中英文中的六个概念、隐私边界、自动化边界和案例链接必须一一对应。

- [ ] **步骤 3：从试运行文档提炼公开案例**

`docs/evolution-case-profiler-evidence-validation.md` 只包含：

- `This is a retrospective conformance case, not a performance case.`；
- 两个完整 commit、`v1.4.0` 和评测器身份；
- 两个聚焦测试命令；
- 实际重放结果；
- 能确认与不能确认的结论；
- 私有材料为零；
- 指向详细维护记录的相对链接。

不写加速百分比，不把回溯重放描述成前瞻性验证。

- [ ] **步骤 4：编写四个 Markdown 模板**

模板只使用标题和自然语言字段，不使用 YAML front matter、JSON 或 schema。

`case-snapshot.md` 必须包含：

```markdown
# Case Snapshot
## Original request and required outcome
## Public challenge view
## Audit provenance
## Environment and authorization
## Private material
## Safe public derivative
```

`evaluation-definition.md` 必须包含：

```markdown
# Evaluation Definition
## Repository revision or evaluation arms
## Only intended comparison axis
## Evaluator identity
## Environment and model identity
## Workload, correctness, budget, and repetitions
## Expected outcome envelope
## Claim ceiling
```

`evaluation-result.md` 必须包含：

```markdown
# Evaluation Result
## Bound definition and actual identities
## Actual result
## Valid, failed, interrupted, and unrun trials
## Correctness, performance, and cost facts
## Terminal reasons and uncertainty
## Supported claim scope
```

`release-decision.md` 必须包含：

```markdown
# Release Decision
## Human decision
## Evidence reviewed
## Accepted scope and trade-offs
## Claims not established
## Release, rejection, withdrawal, or rollback carrier
```

每个模板都明确：删除所有填写提示后再提交；不要附加 private workload、trace、权重、源码、图像或内部地址。

- [ ] **步骤 5：加入公开导航**

在 `mkdocs.yml` 中增加：

```yaml
  - Project Evolution: project-evolution.md
  - Project Evolution (English): project-evolution.en.md
  - Evolution Case: evolution-case-profiler-evidence-validation.md
```

在 `docs/index.md` 的 Documentation 列表中增加三条相对链接，并说明案例是确定性工具的一致性修复，不是性能收益样例。

- [ ] **步骤 6：运行聚焦测试**

运行：

```bash
python3 -m unittest tests.test_project_evolution tests.test_public_docs
```

预期：新增指南、模板和案例相关测试通过；若测试要求与实际设计冲突，应从六概念和 V1.4 边界重新判断，不能为绿灯添加无意义文案。

### 任务 5：接入社区贡献入口，不改变普通修复流程

**文件：**

- 修改：`CONTRIBUTING.md`
- 修改：`.github/pull_request_template.md`
- 修改：`README.md`
- 修改：`README.en.md`

- [ ] **步骤 1：更新贡献指南**

在现有本地检查命令之前增加 `## Project evolution`，准确说明：

- 普通、范围明确的代码修复仍可直接提交，不强制填写四个模板；
- 改变 ChatGPT 行为、知识适用范围、评测协议或公开性能结论时，才使用项目演进材料；
- 评测定义必须在结果产生前提交或以不可变 commit 固定；
- 私有实践可以说明问题来源，但公开改动删除该声明后仍必须独立成立；
- 项目不接收 private workload、内部 trace、权重、专有源码、业务图像或内部地址；
- 外来代码和附件不会自动在自托管 GPU 上运行；
- 评测结果不自动合并，维护者通过 release、拒绝、撤回或 rollback 承担最终决定。

加入指向 `docs/project-evolution.en.md` 和四个模板的相对链接。

- [ ] **步骤 2：更新 Pull Request 模板**

在现有 Evidence 清单后增加：

```markdown
## Project evolution, when applicable

- [ ] The public change must still stand if any private validation statement is removed
- [ ] No private material, internal address, workload, trace, weight, source, or image is attached
- [ ] The Evaluation Definition was frozen before the reported result
- [ ] The claim is no broader than the recorded evidence
- [ ] This pull request does not automatically admit knowledge, merge, or release the change
```

在标题下说明：普通小修复不需要创建项目演进记录，只有行为、知识、评测协议或性能声明变化时填写本节。

- [ ] **步骤 3：更新中英文 README 的进一步阅读**

中文增加：

```markdown
- [项目如何从真实使用中改进](docs/project-evolution.md)
```

英文增加：

```markdown
- [How real use improves the project](docs/project-evolution.en.md)
```

不在 README 重新解释完整设计，不修改 V1.4 项目定位或版本说明。

- [ ] **步骤 4：运行社区和 README 聚焦测试**

运行：

```bash
python3 -m unittest \
  tests.test_project_evolution \
  tests.test_standalone_project \
  tests.test_readme_sync
```

预期：全部通过。

- [ ] **步骤 5：提交公开贡献面**

运行：

```bash
git add \
  docs/project-evolution.md \
  docs/project-evolution.en.md \
  docs/evolution-case-profiler-evidence-validation.md \
  docs/index.md mkdocs.yml \
  .github/evolution \
  .github/pull_request_template.md \
  CONTRIBUTING.md README.md README.en.md \
  tests/test_project_evolution.py \
  tests/test_public_docs.py \
  tests/test_standalone_project.py
git commit -m "docs: 增加项目演进贡献流程"
```

预期：提交不包含 `skills/cuda-kernel-optimizer/` 下的任何文件。

### 任务 6：整体验证和范围审计

**文件：**

- 不新增文件；只验证前三个提交

- [ ] **步骤 1：运行完整 CPU 测试**

运行：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

预期：全部通过，不跳过或删除已有测试。

- [ ] **步骤 2：编译并运行安装自检**

运行：

```bash
python3 -m compileall -q skills/cuda-kernel-optimizer/scripts tests
python3 skills/cuda-kernel-optimizer/scripts/self_check.py
```

预期：编译无错误；self-check 返回 passed，生产模块仍为 17 个。

- [ ] **步骤 3：验证 staged installation**

运行：

```bash
stage="$(mktemp -d)"
cp -R skills/cuda-kernel-optimizer "$stage/cuda-kernel-optimizer"
python3 "$stage/cuda-kernel-optimizer/scripts/self_check.py"
rm -rf "$stage"
```

预期：独立安装副本 self-check 通过。这里的复制只用于临时验证，不修改仓库文件。

- [ ] **步骤 4：证明没有扩张生产代码**

实施计划使用独立提交 `docs: 规划证据驱动的项目演进首期` 保存。验证时从提交信息取得实施基线：

```bash
plan_base="$(git rev-list -1 --grep='^docs: 规划证据驱动的项目演进首期$' HEAD)"
test -n "$plan_base"
git diff --name-only "$plan_base"..HEAD -- skills/cuda-kernel-optimizer
git diff --stat "$plan_base"..HEAD
git diff --check "$plan_base"..HEAD
git status --short --branch
```

预期：第一条没有输出；改动只属于已列出的维护文档、公开文档、GitHub 模板和测试；diff check 通过；工作区干净。

- [ ] **步骤 5：按设计规格逐项复核**

对照 `docs/maintenance/evidence-driven-self-evolution-design.md`，确认：

- 六个概念全部能在公开指南和首个案例中找到，但没有增加六个运行时对象；
- 评测定义的提交早于结果提交；
- 回溯限制没有被删掉；
- 私有实践不能直接成为公开性能证据；
- 普通修复没有被迫走复杂流程；
- 没有 Controller、自动下一步、自动知识准入、自动 PR、自动合并或自动发布；
- 没有为了通过测试堆叠同义文案或细碎 case。

若任何一项不成立，回到对应任务整体修正；不得增加旁路、兼容层或第二套流程。

## 4. 完成标准

首期完成时，项目应只新增一种维护能力：真实使用中的项目问题可以被整理、冻结、比较和人工发布。它不改变一次 GPU 优化怎样运行。

必须同时满足：

- 首个案例在结果产生前已有独立的评测定义提交；
- 同一评测器在旧版本和修复版本上形成可解释差异；
- 公开案例不声称性能收益或通用能力；
- 社区能用四个 Markdown 模板提交公开、范围明确的材料；
- 私有材料不进入项目，私有自述不增加证明权重；
- 一个聚焦测试文件覆盖公开材料的必要边界；
- 安装 skill 的生产代码、知识库、17 模块自检和 V1.4 流程完全不变。

只有实际重复使用暴露了明确的机械检查成本，才另行设计 validator 或草稿 Pull Request 适配器；它们不属于本计划。
