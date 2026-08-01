# 性能迭代

每轮只验证一个能够清楚解释的候选机制。ChatGPT 负责选择候选和判断投入是否值得；脚本只执行指定 operation。

## 1. 先测 original

readiness 通过后，先执行 `workload_evaluate.py baseline`。原始业务基线必须来自冻结的测试集、精度规则、driver 和环境身份。没有有效 baseline，不创建候选。

## 2. 冻结 Experiment

调用 `workload_evaluate.py experiment` 前，至少说明：

- 机制与预期影响；
- claim layer；
- 修改范围；
- 最低成本证伪；
- minimum effect；
- 拒绝条件；
- 进入正式测试的条件；
- 使用的测试 case 与成对采样设计。

组合两个已测机制属于新的 Candidate，必须建立新的 Experiment 并重新验证精度和性能。过去两个候选各自通过，不代表组合后仍正确或仍有收益。

## 3. 由低成本到高成本

常见顺序是：

1. 静态审查或独立小测试；
2. 构建与最低精度校验；
3. 短版成对性能初筛；
4. 回答明确问题所需的 profiler；
5. 正式成对 workload 测试；
6. 完整服务测试。

`screen` 执行 Experiment 中声明的低成本路径。前一项已足以拒绝候选时，不启动后续昂贵动作。`conservative_bound` 只有在预先说明它为何约束正式目标，并实际证明收益上限低于 minimum effect 时才能拒绝。`diagnostic_proxy` 只检验声明的局部机制；低代理收益或样本不足不能单独否定完整 workload，ChatGPT 根据该主张、其它证据和正式测试成本决定是否继续。profiler 不是固定阶段；只有它能区分仍然竞争的解释时才值得运行。

涉及越界访问、向量化或异步拷贝时，将 `compute-sanitizer memcheck` 纳入 screen；涉及 shared memory、多阶段 pipeline、barrier、warp specialization、原子操作或跨 stream 同步时，再按风险加入 `racecheck`、`initcheck` 或 `synccheck`。这些检查由 Experiment 显式声明，不由机制名称自动触发，也不能替代业务精度校验。

## 4. 判断收益

收益判断同时考虑点估计、区间、最低有效收益、约束和测量稳定性。可移除时间是“假设该部分完全消失”的上限，不是候选必然获得的收益。

正式 `target` 结果应比较 Candidate 与当前 reference。reference 起初是 original；选择过有效 Candidate 后，后续候选直接与当前 Champion 比较，避免只证明自己优于已经落后的 baseline。

## 5. 选择与最终复测

有效正式结果不会自动更新最佳版本。ChatGPT 复核精度、统计结果、环境身份和适用范围后，显式调用 `champion.py select`。需要回退时，用拒绝当前 Champion 的 final audit 调用 `restore-original`。

在形成 workload 或服务层最终结论前，执行 `final_audit` 重新比较 original 与当前 Champion。kernel 指标改善不能替代这一步。

## 6. 时间与停止

每条外部命令有独立 timeout 和进程组清理，防止构建、测试或 profiler 卡死。是否继续优化不由 timeout 决定，而由现有证据、预期收益、下一步时间和 GPU 成本、风险与用户授权共同决定。

出现以下情况应尽早停止：

- 收益上限低于最低有效收益；
- 精度或 dispatch identity 失败；
- screen 已按预先声明的证伪条件拒绝机制，或 conservative bound 已证明收益上限不足；
- 重复证据已经否定该机制；
- 下一步超出用户授权的时间、GPU、风险或修改范围；
- 没有新的非重复方向。

修复 runner、测试集或依赖不算性能 iteration。若基础环境问题持续消耗时间，应停止优化并单独报告，而不是把环境维护包装成候选实验。
