# 性能诊断与优化决策引擎实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers-zh:executing-plans` 在当前会话中逐项执行，并使用 `superpowers-zh:test-driven-development` 保证每个行为先出现预期失败。

**目标：** 在现有 workload Controller 中加入一个小而可验证的决策层，用已封存的执行图、假设和证据判断下一步应当 `MEASURE`、`PURSUE`、`REVIEW_REQUIRED` 还是 `STOP`，并尽早给出收益上限、未知项和投入建议。

**架构：** 确定性代码从执行图计算关键路径事实和可缩减上限；模型仍只提出最多三个机制假设；现有证据选择器负责挑选最便宜且能区分假设的动作；新决策模块把这些结果组成投资简报。Controller 只增加两个窄调用点，不改写预算、恢复或执行状态机。

**技术栈：** Python 3 标准库、`unittest`、现有 JSON artifact/ledger、现有 reviewer CLI、可选 RTX 5090/Nsys/NCU 验收。

---

## 任务 1：确定性性能模型

**文件：**

- 新增：`tests/test_performance_model.py`
- 新增：`skills/cuda-kernel-optimizer/scripts/performance_model.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/execution_map.py`

1. 先写失败测试，覆盖：关键路径节点按 layer 聚合；重叠区间不会被重复计算；可缩减上限被 workload window 封顶；低于 `minimum_effect_us` 的方向标记为不合格；缺少同身份历史耗时时 P50/P90 为 `null` 且依据为 `unavailable`。
2. 运行 `python3 -m unittest tests.test_performance_model -v`，确认因模块或行为缺失而失败。
3. 实现 `build_performance_model(execution_map, *, minimum_effect_us, action_timings=()) -> dict`。只接受已验证 execution map；输出 observed/missing layers、关键路径贡献、方向收益上限、矛盾/未知项和 identity-matched action timing range。
4. 在 `execution_map.py` 增加一个纯函数，返回节点的非重复关键路径区间贡献；不改变现有 map schema。
5. 重跑新测试以及 `tests.test_execution_map`，通过后提交：`feat: add deterministic performance model`。

## 任务 2：假设上限、机制去重与定向 NCU 门禁

**文件：**

- 修改：`tests/test_hypothesis_space.py`
- 修改：`tests/test_evidence_selector.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/hypothesis_space.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/evidence_selector.py`

1. 先写失败测试，证明：第四个 active 假设被拒绝；同一 mechanism 仅改大小写、分隔符或名字仍被拒绝；已关闭的 mechanism key 不能重新进入；非 kernel 节点上的假设不能请求 NCU；NCU 必须只绑定一个明确 kernel 节点；已有全局 trace 时优先静态/定向证据而不是重复全局 profile。
2. 运行两个测试文件并确认预期失败。
3. 在 `validate_hypothesis_set(..., closed_mechanism_keys=())` 中加入稳定 mechanism key、最多三个 active 假设和跨轮关闭机制门禁；不增加 schema 文件。
4. 给 `select_evidence_request` 传入已验证 execution map。对 `ncu_kernel` action 强制单一 kernel scope；排序继续先看区分能力，再按 perturbation/risk/cost 递增，禁止模型提供自定义分数。
5. 更新现有调用点和 fixtures，运行两个目标测试及 `tests.test_active_diagnosis_vertical`，通过后提交：`feat: bound hypotheses and targeted profiling`。

## 任务 3：决策状态与投入简报

**文件：**

- 新增：`tests/test_diagnostic_decision.py`
- 新增：`skills/cuda-kernel-optimizer/scripts/diagnostic_decision.py`

1. 先写失败测试，覆盖：有可执行区分动作返回 `MEASURE`；方向证据充分返回 `PURSUE`；收益上限低于 minimum effect 返回 `STOP`；有价值动作仅因授权不足被拒绝时返回 `REVIEW_REQUIRED`；没有合格新方向返回 `STOP`；未知耗时不得伪造数值；外部 reviewer 不可用或意见冲突不改变本地决定。
2. 运行 `python3 -m unittest tests.test_diagnostic_decision -v`，确认缺少实现而失败。
3. 实现 `decide_next_step(performance_model, hypothesis_result, evidence_selection, *, external_review=None) -> dict`，只接受四种状态；每个结果包含 primary diagnosis、benefit ceiling、uncertainty、next action、cost basis、next checkpoint 和 terminal reason。
4. 对 `REVIEW_REQUIRED` 只识别授权/能力/剩余 profile action 等阻断；不能把预算耗尽写成无价值。对 `STOP` 要给出可重放的原因。
5. 运行目标测试并提交：`feat: add diagnostic decision and investment brief`。

## 任务 4：Controller 窄接入与证据更新

**文件：**

- 修改：`tests/test_workload_controller.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/workload_controller.py`

1. 先写失败测试，证明首次 global analysis 后产生 `performance_model.json`；注册假设后产生 `decision.json` 和 `investment_brief.json`；artifact digest 进入 ledger/context；新证据后重新计算而不是复用旧 digest；`STOP` 与 `REVIEW_REQUIRED` 分别映射到稳定的 next action/terminal reason。
2. 运行对应 Controller 测试并确认预期失败。
3. 新增两个 sibling module loader。在 `_build_active_diagnosis_context` 后构建性能模型；在 proposal validation/selection 后构建决策；在 admitted evidence 后重新进入相同计算路径。
4. 只在 `active_diagnosis/` 写新 artifact，并沿用 atomic JSON、digest 和 ledger。不得改 `orchestrate.py`、`budget.py`、`run_control.py`。
5. 运行 `tests.test_workload_controller`、`tests.test_active_diagnosis_vertical` 和新模块测试，通过后提交：`feat: integrate diagnostic decisions into controller`。

## 任务 5：可选外部 AI 方向质证

**文件：**

- 修改：`tests/test_workload_reviewer.py`
- 修改：`tests/test_diagnostic_decision.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/workload_reviewer.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/workload_controller.py`

1. 先写失败测试，覆盖 provider 优先级 `google-ai-mode > glm > kimi > deepseek > gemini`；普通方向取一个、重大方向取两个、平台期/最终审查取三个；并行调用共享 180 秒；失败 provider 和完整有效回答均被记录；所有 provider 不可用时本地结果不变。
2. 运行 reviewer/decision 测试并确认预期失败。
3. 实现纯函数 `select_reviewer_configs(configs, trigger)`，只从用户已经配置且可调用的 reviewer 中选择；不加入账号、cookie、API key 或厂商 SDK。
4. Controller 仅在初始重大方向、冲突/平台期和最终重大变更调用现有 reviewer protocol。发送压缩、脱敏、digest 绑定的 evidence packet；外部意见只进入 review section，由本地证据重新裁决。
5. 运行目标测试并提交：`feat: add bounded external direction challenge`。

## 任务 6：行为回归、5090 验收与技能发布说明

**文件：**

- 修改：`tests/test_active_diagnosis_vertical.py`
- 修改：`tests/gpu/sm120/test_sm120_acceptance.py`
- 修改：`tests/test_skill_metadata.py`
- 修改：`skills/cuda-kernel-optimizer/SKILL.md`
- 修改：`README.md`
- 修改：`README.zh-CN.md`
- 修改：`CHANGELOG.md`

1. 先扩展 CPU vertical tests：launch/CUDA Graph、memory/coalescing、compute GEMM、transfer overlap 四类受控场景；断言首要 layer、top-3 mechanism、允许的 profiler、投资简报和终态，不以“schema 可解析”代替行为验收。
2. 先写 skill contract 失败测试，要求技能入口指导 AI：检查原始业务基线和 readiness；先读投资简报；只执行 decision 指定的一步；保留 complete-service 目标；provider 不可用时本地继续。
3. 补最小 SKILL/README/release note，避免复制代码清单或加入新的参考文档。
4. 在 5090 目标机使用项目副本运行四类真实负载。只安装项目/容器内工具；宿主机权限、驱动或 profiler counter 改动只给建议。记录 time-to-direction、昂贵 profiler 次数、无效 candidate 数、GPU profile 时间和 terminal decision；与 `main` 比较，至少两个场景减少昂贵动作，任何场景不得增加无必要阶段。
5. 运行完整验证：

   ```bash
   python3 -m unittest discover -s tests -p 'test_*.py'
   python3 -m unittest discover -s skills/cuda-kernel-optimizer/tests -p 'test_*.py'
   python3 -m compileall -q tools skills/cuda-kernel-optimizer/scripts tests
   python3 skills/cuda-kernel-optimizer/scripts/self_check.py
   git diff --check
   ```

6. 核对变更范围和复杂度：没有新 Controller、没有 schema 家族、没有 provider SDK、没有 full/workload-wide NCU 默认；新增生产代码必须由行为测试覆盖。提交：`docs: publish performance decision workflow`。

## 执行检查点

- 任务 1—3 后检查一次：纯决策核心是否能脱离 Controller 独立测试，是否出现新 schema 或重复规则。
- 任务 4—5 后检查一次：Controller 改动是否仍是窄调用，外部 AI 是否保持可选且不参与 promotion authority。
- 任务 6 后做独立 code review 和完整验证；在用户要求发布前不推送远端、不合并 `main`。
