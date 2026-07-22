# V1.2 证据边界自适应投入控制实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不编造成功概率和预期收益的前提下，让 Controller 根据本地证据、累计投入和用户授权逐动作选择、重新估算、继续或停止，并为 V1.3 保留稳定的知识接口。

**架构：** 新增两个小型纯模块：`adaptive_investment.py` 负责方向组合、动作价值、支配关系和累计授权判断；`knowledge_adapter.py` 负责将本地卡片、实时搜索和外部 AI 建议规范化为只能生成本地可证伪假设的建议。现有 `diagnostic_decision.py` 和 `workload_controller.py` 只做输入适配、动作执行与结果持久化，候选阶段继续复用 `CandidateGate`，但授权不足改为 `REVIEW_REQUIRED`，并保留真实 paired interval 语义。

**技术栈：** Python 3 标准库、`unittest`、现有 paired statistics、JSON 账本和 workload Controller。

---

## 文件职责

- 创建 `skills/cuda-kernel-optimizer/scripts/adaptive_investment.py`：纯函数决策器，不执行命令、不读写文件。
- 创建 `tests/test_adaptive_investment.py`：方向生命周期、动作支配、累计授权和噪声场景测试。
- 创建 `skills/cuda-kernel-optimizer/scripts/knowledge_adapter.py`：规范化本地和运行期知识建议，隔离外部数值收益。
- 创建 `tests/test_knowledge_adapter.py`：知识缺失降级、影子方向准入、外部收益隔离和去重测试。
- 修改 `skills/cuda-kernel-optimizer/scripts/diagnostic_decision.py`：把 V1.1 工件适配为方向组合并返回自适应投入摘要。
- 修改 `tests/test_diagnostic_decision.py`：验证备选方向保留、累计授权和知识建议不能晋级。
- 修改 `skills/cuda-kernel-optimizer/scripts/budget.py`：候选阶段授权与证据结论分离，输出 blocked action 和累计预测。
- 修改 `skills/cuda-kernel-optimizer/scripts/workload_controller.py`：持久化 `REVIEW_REQUIRED`、恢复未验证改动并保证重放幂等。
- 修改 `skills/cuda-kernel-optimizer/tests/test_time_gates.py`、`tests/test_budget.py` 和 `tests/test_workload_controller.py`：证明昂贵阶段没有被启动。
- 修改 `skills/cuda-kernel-optimizer/scripts/workload_reviewer.py`：加入 GitHub Copilot、模型元数据和运行期请求去重。
- 修改 `tests/test_workload_reviewer.py`：验证供应商顺序、模型记录、并行上限和失败降级。
- 修改 `skills/cuda-kernel-optimizer/scripts/self_check.py`：将两个新运行时模块纳入安装包完整性检查。
- 修改 `skills/cuda-kernel-optimizer/SKILL.md`、`skills/cuda-kernel-optimizer/references/research_augmentation.md` 和 `skills/cuda-kernel-optimizer/references/long_running_control.md`：只描述已实现的行为。

### 任务 1：实现纯自适应决策器

**文件：**
- 创建：`skills/cuda-kernel-optimizer/scripts/adaptive_investment.py`
- 创建：`tests/test_adaptive_investment.py`

- [ ] **步骤 1：为收益边界、决策价值和累计授权编写失败测试**

测试至少覆盖：

```python
def test_external_gain_cannot_change_local_bound():
    direction = direction_fixture(lower=0.0, upper=4.0)
    direction["external_claim_pct"] = 20.0
    result = decide_next_action(
        [direction], [evidence_action("check-z", direction="z")],
        authorization=authorization(max_seconds=100),
        spend=spend(elapsed_seconds=0), minimum_effect=1.0,
    )
    assert result["portfolio"][0]["benefit"]["upper"] == 4.0


def test_cumulative_small_actions_cannot_bypass_authorization():
    result = decide_next_action(
        [direction_fixture(status="supported")],
        [candidate_action("next", p90_seconds=2.0)],
        authorization=authorization(max_seconds=100.0),
        spend=spend(elapsed_seconds=99.0), minimum_effect=1.0,
    )
    assert result["decision"] == "REVIEW_REQUIRED"
    assert result["blocked_action"]["action_id"] == "next"
    assert result["projected_spend"]["p90_seconds"] == 101.0


def test_no_decision_changing_outcome_means_action_is_skipped():
    action = evidence_action("repeat", direction="z")
    action["outcomes"] = [
        {"outcome_id": "same-a", "supports": [], "opposes": []},
        {"outcome_id": "same-b", "supports": [], "opposes": []},
    ]
    result = decide_next_action(
        [direction_fixture()], [action], authorization=authorization(),
        spend=spend(), minimum_effect=1.0,
    )
    assert result["decision"] == "STOP"
    assert result["reason"] == "no_decision_changing_action"
```

另加测试证明：低上界方向关闭、`stale` 方向只允许 refresh、便宜且覆盖更多假设的动作支配昂贵动作、不同 mechanism 的 fallback 不因首选实现失败而删除、同一 checkpoint 重放结果相同。

- [ ] **步骤 2：运行新测试并确认因模块不存在而失败**

运行：

```bash
python3 -m unittest -v tests.test_adaptive_investment
```

预期：`ImportError` 或 `FileNotFoundError` 指向 `adaptive_investment.py`。

- [ ] **步骤 3：实现最小纯函数 API**

实现公开入口：

```python
def decide_next_action(
    directions: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    *,
    authorization: Mapping[str, Any],
    spend: Mapping[str, Any],
    minimum_effect: float,
) -> dict:
    """Return one evidence-authoritative next decision without side effects."""
```

实现必须：

- 严格校验有限数值、稳定 ID、状态和成本等级；
- 只读取 `benefit.lower/upper/basis/identity_digest/stale`，忽略外部收益字段；
- 关闭 `falsified`、精确机制重复和 `upper < minimum_effect` 的方向；
- 将 stale 方向限制为 `refresh` 动作；
- 根据 action outcomes 判断是否真的改变 support/oppose 或 candidate interval；
- 按成本、扰动、风险、覆盖目标和独立证据增量计算确定性支配关系；
- 在动作启动前检查 `spent + action.p90_seconds`；
- 返回 `decision`、`reason`、`portfolio`、`selected_action`、`blocked_action`、`projected_spend`、`skipped_actions` 和 `next_checkpoint`；
- 不读文件、不调用模型、不执行命令。

- [ ] **步骤 4：运行测试并确认通过**

```bash
python3 -m unittest -v tests.test_adaptive_investment
```

预期：全部通过。

- [ ] **步骤 5：提交任务 1**

```bash
git add skills/cuda-kernel-optimizer/scripts/adaptive_investment.py tests/test_adaptive_investment.py
git commit -m "feat: add evidence-bounded investment decisions"
```

### 任务 2：实现稳定知识适配器

**文件：**
- 创建：`skills/cuda-kernel-optimizer/scripts/knowledge_adapter.py`
- 创建：`tests/test_knowledge_adapter.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/self_check.py`

- [ ] **步骤 1：编写失败测试**

```python
def test_external_suggestion_requires_local_scope_and_falsifier():
    suggestion = {
        "source": "github-copilot",
        "mechanism_id": "new-layout",
        "statement": "change the layout",
        "external_gain_pct": 20.0,
        "scope_node_ids": [],
        "unmodeled_interval_id": None,
        "falsification_question": None,
        "evidence_action": None,
    }
    result = recommend(context_fixture(), external=[suggestion])
    assert result["candidates"] == []
    assert result["rejections"][0]["reason"] == "not_locally_falsifiable"


def test_valid_external_shadow_has_no_numeric_gain_or_support():
    result = recommend(context_fixture(), external=[valid_shadow_fixture()])
    shadow = result["candidates"][0]
    assert shadow["origin"] == "external"
    assert shadow["confidence"] == "inconclusive"
    assert "external_gain_pct" not in shadow
    assert shadow["promotion_authority"] == "none"
```

另加：空知识返回 `knowledge_support=unavailable`；同一查询摘要去重；过期/版本不匹配条目标记 unavailable；本地卡和外部建议使用相同规范化接口；未知字段和敏感字段被拒绝。

- [ ] **步骤 2：运行测试确认失败**

```bash
python3 -m unittest -v tests.test_knowledge_adapter
```

预期：缺少 `knowledge_adapter.py`。

- [ ] **步骤 3：实现知识接口**

```python
def recommend(
    context: Mapping[str, Any],
    *,
    bundled: Sequence[Mapping[str, Any]] = (),
    searched: Sequence[Mapping[str, Any]] = (),
    external: Sequence[Mapping[str, Any]] = (),
    prior_query_digests: Sequence[str] = (),
    limit: int = 3,
) -> dict:
    """Normalize bounded suggestions; never create evidence or benefit facts."""
```

结果只包含规范化机制、适用条件、节点或 uncovered interval、证伪问题、evidence action、风险、来源和新鲜度。外部数值收益、成功概率、promotion verdict 和命令回调全部丢弃或拒绝。

- [ ] **步骤 4：将新模块加入安装包自检并运行测试**

```bash
python3 -m unittest -v tests.test_knowledge_adapter tests.test_skill_metadata
```

预期：全部通过。

- [ ] **步骤 5：提交任务 2**

```bash
git add skills/cuda-kernel-optimizer/scripts/knowledge_adapter.py skills/cuda-kernel-optimizer/scripts/self_check.py tests/test_knowledge_adapter.py
git commit -m "feat: add bounded knowledge adapter"
```

### 任务 3：接入 V1.1 诊断决策

**文件：**
- 修改：`skills/cuda-kernel-optimizer/scripts/diagnostic_decision.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/workload_controller.py`
- 修改：`tests/test_diagnostic_decision.py`
- 修改：`tests/test_active_diagnosis_vertical.py`

- [ ] **步骤 1：编写诊断集成失败测试**

新增测试证明：

```python
def test_supported_fallback_is_retained_after_primary_candidate_failure():
    result = decide_with_two_supported_directions(primary_failed=True)
    assert result["decision"] == "PURSUE"
    assert result["next_action"]["hypothesis_id"] == "fallback"


def test_external_shadow_can_only_request_measurement():
    result = decide_with_external_shadow()
    assert result["decision"] == "MEASURE"
    assert result["next_action"]["action_id"] == "shadow-falsifier"
    assert result["primary_diagnosis"]["confidence"] == "inconclusive"


def test_stale_execution_map_blocks_candidate_implementation():
    result = decide_with_stale_bound()
    assert result["decision"] == "MEASURE"
    assert result["next_action"]["kind"] == "refresh"
```

- [ ] **步骤 2：运行目标测试确认新断言失败**

```bash
python3 -m unittest -v tests.test_diagnostic_decision tests.test_active_diagnosis_vertical
```

- [ ] **步骤 3：实现适配层**

在 `diagnostic_decision.py` 中：

- 将 ranked hypotheses 转成 `adaptive_investment` directions；
- 将 selected/rejected evidence requests 转成 actions；
- 从 identity-matched timings 和 Controller state 构造累计 spend；
- 调用一次 `decide_next_action`；
- 继续输出原有四种 diagnosis decision，同时在 `investment_brief` 中加入 portfolio、累计投入、selected/blocked action、bound basis 和 next feedback point；
- 保持原有字段兼容，不能让知识或外部 review 修改 local decision。

在 `workload_controller.py` 中只负责传入当前 elapsed/remaining authorization、候选历史和知识适配结果，并持久化摘要。

- [ ] **步骤 4：运行诊断相关测试**

```bash
python3 -m unittest -v \
  tests.test_performance_model \
  tests.test_evidence_selector \
  tests.test_diagnostic_decision \
  tests.test_active_diagnosis_vertical
```

预期：全部通过。

- [ ] **步骤 5：提交任务 3**

```bash
git add skills/cuda-kernel-optimizer/scripts/diagnostic_decision.py skills/cuda-kernel-optimizer/scripts/workload_controller.py tests/test_diagnostic_decision.py tests/test_active_diagnosis_vertical.py
git commit -m "feat: adapt diagnosis to cumulative investment control"
```

### 任务 4：改造候选阶段和授权终态

**文件：**
- 修改：`skills/cuda-kernel-optimizer/scripts/budget.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/workload_controller.py`
- 修改：`skills/cuda-kernel-optimizer/tests/test_time_gates.py`
- 修改：`tests/test_budget.py`
- 修改：`tests/test_workload_controller.py`

- [ ] **步骤 1：先增加候选阶段失败测试**

测试必须逐项证明：

- 静态失败后 GPU action 计数为 0；
- correctness 失败后 profiler 和 paired action 计数为 0；
- short upper bound 低于 threshold 后 formal 计数为 0；
- short interval 跨 threshold 时只允许一个已声明的 bounded follow-up；
- follow-up P90 使累计投入越界时，在 action 调用前返回 `REVIEW_REQUIRED`；
- `REVIEW_REQUIRED` 返回 `blocked_action`、`projected_spend`、`elapsed_seconds` 和 `skipped_expensive_stages`；
- 重复 evaluate 不重复执行 action；
- authorization exhaustion 不映射成 `budget_expired` 或 rejected performance。

- [ ] **步骤 2：运行测试确认旧行为失败**

```bash
python3 skills/cuda-kernel-optimizer/tests/test_time_gates.py -v
python3 -m unittest -v tests.test_budget tests.test_workload_controller
```

- [ ] **步骤 3：修改 `CandidateGate`**

实现：

- 将 `hard_ceiling_admission_failed` 改为 `REVIEW_REQUIRED` 和 `authorization_insufficient_for_next_action`；
- 结果包含 `next_stage`、`blocked_action`、累计实际和 projected P90；
- correctness/static/performance failure 仍为 evidence `STOP`；
- profiler 只有声明 live uncertainty 时执行；
- short interval 上界低于 threshold 时停止；区间跨 threshold 时允许唯一 bounded paired follow-up；
- formal/service lower bound达到 threshold 才能 `PROMOTE`；
- 所有 stage 仍在每次执行前重新检查授权。

- [ ] **步骤 4：修改 Controller 终态**

新增 `_finish_review_required(...)`，行为为：

- 保存 candidate diff、time gate、blocked action 和 evidence；
- 回退未完成正式验证的工作区到冻结 snapshot；
- 状态设为可继续但非 performance rejection；
- repeated call 返回同一 decision digest；
- external final review 超时不能把已经正式确认的本地 win 改为失败。

- [ ] **步骤 5：运行候选与 Controller 测试**

```bash
python3 skills/cuda-kernel-optimizer/tests/test_time_gates.py -v
python3 -m unittest -v tests.test_budget tests.test_workload_controller tests.test_long_run_recovery
```

预期：全部通过。

- [ ] **步骤 6：提交任务 4**

```bash
git add skills/cuda-kernel-optimizer/scripts/budget.py skills/cuda-kernel-optimizer/scripts/workload_controller.py skills/cuda-kernel-optimizer/tests/test_time_gates.py tests/test_budget.py tests/test_workload_controller.py
git commit -m "feat: adapt candidate stages to cumulative authorization"
```

### 任务 5：接入实时搜索元数据和 GitHub Copilot

**文件：**
- 修改：`skills/cuda-kernel-optimizer/scripts/workload_reviewer.py`
- 修改：`tests/test_workload_reviewer.py`
- 修改：`skills/cuda-kernel-optimizer/references/research_augmentation.md`

- [ ] **步骤 1：编写失败测试**

```python
def test_copilot_is_second_for_repository_review():
    configs = reviewer_configs(
        "gemini", "github-copilot", "glm", "google-ai-mode"
    )
    selected = select_reviewer_configs(configs, "major")
    assert [item["provider"] for item in selected] == [
        "google-ai-mode", "github-copilot"
    ]


def test_provider_surface_and_underlying_model_are_distinct():
    artifact = run_completed_copilot(model="auto")
    review = artifact["reviews"][0]
    assert review["provider"] == "github-copilot"
    assert review["underlying_model"] == "auto"
```

另加：相同 request digest 和 provider 不重复执行；不同 surface 使用同一 underlying model 不计为两个 heterogeneous models；输出仍保留完整 response、失败 provider 和 total wait；总等待不超过 180 秒。

- [ ] **步骤 2：运行 reviewer 测试确认失败**

```bash
python3 -m unittest -v tests.test_workload_reviewer
```

- [ ] **步骤 3：实现供应商与元数据扩展**

- 将 `github-copilot` 放在 Google AI Mode 之后；
- 允许 reviewer config 提供可选 `underlying_model`，值为安全短字符串、`auto` 或 `unknown`；
- aggregate 记录 provider surface、model、completed/failed、完整规范化 response 和总等待；
- 使用 request digest、canonical provider 和 model 形成本轮去重键；
- 不改变外部 review 的 advisory-only verdict 集合。

- [ ] **步骤 4：运行 reviewer 测试**

```bash
python3 -m unittest -v tests.test_workload_reviewer
```

预期：全部通过。

- [ ] **步骤 5：提交任务 5**

```bash
git add skills/cuda-kernel-optimizer/scripts/workload_reviewer.py tests/test_workload_reviewer.py skills/cuda-kernel-optimizer/references/research_augmentation.md
git commit -m "feat: add run-scoped Copilot review metadata"
```

### 任务 6：更新使用说明并完整验证

**文件：**
- 修改：`skills/cuda-kernel-optimizer/SKILL.md`
- 修改：`skills/cuda-kernel-optimizer/references/long_running_control.md`
- 修改：`tests/test_public_docs.py`、`tests/test_skill_eval.py`、`tests/test_readme_sync.py`

- [ ] **步骤 1：先修改文档行为测试**

测试要求文档明确出现：累计授权是启动边界而非目标；收益上限不是预期收益；知识和外部 AI 只能生成本地可证伪方向；Copilot 的代码/仓库专项定位；普通模式最多一次汇总授权；无人值守零次询问。

- [ ] **步骤 2：运行文档测试确认旧文字不满足**

```bash
python3 -m unittest -v tests.test_public_docs tests.test_skill_eval tests.test_readme_sync
```

- [ ] **步骤 3：最小更新技能与长任务说明**

只写已实现入口、决策循环、输出字段和失败语义。不扩充知识目录，不写新的设计论文，不在 README 宣称未发布版本。

- [ ] **步骤 4：运行定向测试**

```bash
python3 -m unittest -v \
  tests.test_adaptive_investment \
  tests.test_knowledge_adapter \
  tests.test_performance_model \
  tests.test_evidence_selector \
  tests.test_diagnostic_decision \
  tests.test_active_diagnosis_vertical \
  tests.test_budget \
  tests.test_workload_controller \
  tests.test_workload_reviewer \
  tests.test_public_docs \
  tests.test_skill_eval \
  tests.test_readme_sync
python3 skills/cuda-kernel-optimizer/tests/test_time_gates.py -v
```

预期：全部通过。

- [ ] **步骤 5：运行完整 CPU 回归**

```bash
python3 -m unittest discover -s tests -p 'test*.py' -v
```

预期：零失败、零错误；只允许仓库既有的明确 GPU 跳过项。

- [ ] **步骤 6：运行安装包和差异检查**

```bash
python3 skills/cuda-kernel-optimizer/scripts/self_check.py
python3 -m unittest discover -s skills/cuda-kernel-optimizer/tests -p 'test*.py' -v
git diff --check
git status --short
```

预期：自检成功、安装包测试全部通过、无 whitespace error，状态只包含本计划的预期文档修改。

- [ ] **步骤 7：逐项核对规格的 20 项验收标准**

将每项映射到具体测试名；缺少测试时先补失败测试，再补最小实现。不得用“完整回归通过”替代逐项映射。

- [ ] **步骤 8：提交任务 6**

```bash
git add skills/cuda-kernel-optimizer/SKILL.md skills/cuda-kernel-optimizer/references/long_running_control.md tests/test_public_docs.py tests/test_skill_eval.py tests/test_readme_sync.py
git commit -m "docs: explain adaptive investment control"
```

## 实现结束条件

只有在以下条件同时满足时才能宣布完成：

- 20 项规格验收均有自动化测试；
- 定向测试、完整 CPU 回归、安装包测试和 self-check 都有本轮新鲜成功输出；
- `git diff --check` 无错误；
- 最终 diff 不包含知识库扩写、无关重构、新 schema 家族或未获授权的宿主机修改；
- 分支尚未推送，除非用户在实现验证后再次明确要求发布。
