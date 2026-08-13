# 性能迭代

每轮只验证一个能够清楚解释的候选机制。ChatGPT 负责选择候选和判断投入是否值得；脚本只执行指定 operation。

当前 Target 的 primary 决定研究方向和任务是否完成。secondary 或长尾收益的接纳发生在结果产生之后，不能反向成为低 primary ROI 候选的优先理由。

## 1. 先测 original

readiness 通过后，先执行 `workload_evaluate.py baseline`。原始业务基线必须来自冻结的测试集、精度规则、driver 和环境身份。没有有效 baseline，不创建候选。

在解释 baseline 前先审计每个决策指标的语义：写清实际分子、分母和数据来源，并标出由文本重分词、抽样日志或不同 tokenizer 重建的值。利用冻结 workload 可推出的恒等关系做检查；例如固定每请求输出 token 数时，优先使用合约已知或服务端 generation accounting 的总量，而不是重新 tokenize 生成文本。口径与 Target 不一致且无法修正时，只保留 diagnostic 样本，不在接近阈值的比较中使用该指标。

## 2. 冻结 Experiment

每次新 profiler 事实用于候选判断前，先确认它与 Target、Variant、case/request slice、并发、phase 和环境身份一致，
并区分 steady、capture、warmup、mixed 与 unknown。随后用系统级归因回答：主要 measured time
分布在哪些 subsystem，哪些时间未归因，还有哪些合理方向，为什么当前方向更值得验证。成本或
可行性没有证据时写 unknown，不凭局部热点臆测 ROI。

若候选只覆盖完整 workload 的一部分，令 `f` 为该部分占完整 workload 时间的比例，`r` 为候选
使该部分延迟下降的比例，则理想化吞吐收益上限为 `1 / (1 - f × r) - 1`。局部结果给出相对
吞吐增幅 `g` 时，换算 `r = g / (1 + g)`；例如 `g = 19.3%` 时 `r ≈ 16.2%`。若给出的是
`k` 倍加速，则换算 `r = (k - 1) / k`。该式忽略额外 launch、同步、CPU 和通信成本，只能作为上限。
coverage unknown 时，先选择最低成本 coverage 观测或端到端证伪；上限低于 minimum effect 时
不创建 Experiment。

先按对 primary 的预期端到端收益排序仍可行的 subsystem 和候选，再比较 coverage 可信度、
phase/关键路径匹配、实现风险和取得结论的成本。不要因为某个局部 kernel 容易修改、microbenchmark
容易获得正数或源码中恰好缺少某个 tile，就让它越过 primary ROI 更高的方向。若当前可修改范围内
没有足够 primary ROI，回到系统归因比较 communication、runtime、scheduler 等仍在授权范围内的方向，
而不是继续扫描低覆盖 kernel。

在第一次昂贵 target 前用 original 的成对重复估计各决策指标的测量分辨率。候选的预期 primary
收益低于 minimum effect 或该分辨率，且也不针对一个预先声明、预期超过自身门槛的重要 secondary 时，
不为了证明“整体不回归”而单独启动完整服务测试；保留局部证据，等待与同一 Target 下其它互不冲突的
已测机制组合后再建立新的 Candidate，或直接记录为局部结果。

上面的时间占比和 Amdahl 上限只适用于吞吐、均值等可按时间贡献估算的指标。判断 p95、p99
等长尾指标时，要看改动影响了哪些请求、是否位于关键路径，以及发生在 warmup、steady 还是
request drain；不能仅因调用次数少或总耗时占比低而否定长尾收益。

再完成源码静态审查或独立小测试；已经证伪时不创建候选。调用
`workload_evaluate.py experiment` 前，至少说明：

- 机制与预期影响；
- claim layer；
- 修改范围；
- 最低成本证伪；
- minimum effect；
- 拒绝条件；
- 失败最多能否定当前实现、集成方式、代理结论还是整个机制；
- 进入正式测试的条件；
- reference 与 candidate 的关系；
- Target 规则之外的附加精度 gate，以及只作解释的 diagnostic；
- 使用的测试 case、成对采样设计、隔离或同进程生命周期；
- 会改变候选判断或实验设计的版本化外部前提；未核实时标记为假设。

对 AOT/CUTLASS 模板机制，在第一次构建前同时冻结该机制的有界候选配置、config ID、实验开关、
回退方式和预计构建次数。把可以由同一产物独立选择的配置编进一次 screen artifact，先完成局部筛选；
只为最终胜者再构建一次交付产物。不要把每个 tile、stage 或开关各变成一次完整 wheel 构建。

组合两个已测机制属于新的 Candidate，必须建立新的 Experiment 并重新验证精度和性能。过去两个候选各自通过，不代表组合后仍正确或仍有收益。

## 3. 由低成本到高成本

常见顺序是：

1. 静态审查或独立小测试；
2. 构建与最低精度校验；
3. 短版成对性能初筛；
4. 回答明确问题所需的 profiler；
5. 正式成对 workload 测试；
6. 完整服务测试。

`screen` 执行 Experiment 中声明的低成本证据计划。Driver V2 的一次调用同时返回每个 subject 的正确性和性能观测；工具先验证正确性，再决定这些性能样本能否解释，不为两个判断重复启动完整 workload。隔离比较每个 pair/case 启动 reference 和 candidate 各一次；只有 driver 声明确实支持且 Experiment 冻结了共享状态时，才在同一进程的一次调用内完成两个 subject。前一项已足以按预声明范围拒绝候选时，不启动后续昂贵动作。`conservative_bound` 只有在预先说明它为何约束正式目标，并实际证明收益上限低于 minimum effect 时才能拒绝机制。`diagnostic_proxy` 只检验声明的局部机制；低代理收益或样本不足不能单独否定完整 workload，ChatGPT 根据该主张、其它证据和正式测试成本决定是否继续。正确性、安全、dispatch identity、环境或指标口径失败会使相应性能结果无效，只能关闭受影响的实现或测量；不能据此评价机制。

当 profile 和 coverage 表明候选仍有达到 minimum effect 的可能，或局部机制收益与完整 workload 结果冲突时，关闭机制前先解释差额来自路径未命中、额外 launch/同步/通信、资源竞争、测量分辨率还是机制本身。现有证据不能区分时，按 `research_augmentation.md` 核验一手资料并取得独立反例；若存在一个不同且最低成本的判别实验，且其结果可能改变结论，才追加一次。重新尝试必须有新证据、不同实现路径或不同测量设计；简单重跑、换名或调参不能延长已被同类证据否定的方向。profiler 不是固定阶段；只有它能区分仍然竞争的解释时才值得运行。

每次进入正式 `target` 前，无条件简短复核 Target、Variant、case/request slice、phase、coverage、
收益上限和 ROI。若其间取得新 profiler 事实，重新完成上一节的系统级归因，而不是沿用旧候选理由。

共享宿主机在正式性能采样前启动低频、只读的 CPU/GPU 观测，持续到采样结束，并保存与样本
时间对齐的原始输出。观测缺失、中断，或出现与样本窗口重叠、达到用户或环境规则给出的影响阈值
且无法解释的污染时，性能结果标为 environment-inconclusive；correctness 结果仍独立保留。
瞬时或非重叠异常不限制性能归因。进程列表为空或显存很低不能单独证明 GPU 空闲；选卡时还要在
有界窗口采样利用率和功耗，正式窗口继续保留这些观测。正式共享 GPU 实验保持串行，周期采样交给确定性命令。

涉及越界访问、向量化或异步拷贝时，将 `compute-sanitizer memcheck` 纳入 screen；涉及 shared memory、多阶段 pipeline、barrier、warp specialization、原子操作或跨 stream 同步时，再按风险加入 `racecheck`、`initcheck` 或 `synccheck`。这些检查由 Experiment 显式声明，不由机制名称自动触发，也不能替代业务精度校验。

## 4. 判断收益

收益判断同时考虑点估计、区间、最低有效收益、约束和测量稳定性。可移除时间是“假设该部分完全消失”的上限，不是候选必然获得的收益。

本节是结果接纳规则，不是候选排序规则。一个结果可以值得保留或显式纳入当前版本，但仍未达到当前 Target 的 primary，且不能因此停止主目标搜索。

判断优化结果时检查真实 workload 上所有已声明且有业务价值的重要指标，不只看 primary。若
改动稳定改善一项重要指标，例如 p95 或 p99，同时正确性通过且总体吞吐、平均延迟等关键指标
未超过允许的退化范围，就将它纳入优化结果，记录为局部结果并明确适用场景。没有提升当前主指标，只能说明它
没有达到当前 Target 的主目标，不能单独作为删除改动的理由，也不能据此停止主目标搜索。

若某项收益是在测试后才发现，先把它作为新假设；只有重新冻结以该指标为主目标、以其它重要
指标为约束的 Target 并通过正式验证后，才能据此选择 Champion。这样既保留整体不负向的长尾
优化，也避免从噪声中事后挑选看起来有利的指标。

正式 `target` 结果应比较 Candidate 与当前 reference。reference 起初是 original；选择过有效 Candidate 后，后续候选直接与当前 Champion 比较，避免只证明自己优于已经落后的 baseline。

## 5. 选择与最终复测

有效正式结果不会自动更新最佳版本。只有当前 Target 的 primary verdict 通过后，ChatGPT 才复核精度、统计结果、实际 runtime 身份、比较契约、维护成本和适用范围，并显式决定是否调用 `champion.py select`。容器 lineage 不完整时，结论只能归属于被冻结的最终 runtime，不能归因于未经确认的 upstream base。secondary-only 收益保留在局部结果和交付建议中；要将其选为 Champion，必须以该指标为 primary 建立并验证新的 Target。需要回退时，用拒绝当前 Champion 的 final audit 调用 `restore-original`。

在形成 workload 或服务层最终结论前，执行 `final_audit` 重新比较 original 与当前 Champion。kernel 指标改善不能替代这一步。

## 6. 时间与停止

每条外部命令有独立 timeout 和进程组清理，防止构建、测试或 profiler 卡死。是否继续优化不由 timeout 决定，而由现有证据、预期收益、下一步时间和 GPU 成本、风险与用户授权共同决定。

时间或 GPU 授权是上限，不要求为了耗尽预算而执行低价值实验；但在上限明显未耗尽时，关闭当前候选族不能直接结束 Target。先记录 elapsed/remaining budget、已覆盖的候选空间和残余系统成本，再重新比较至少一轮跨 subsystem 方向，例如 communication、runtime、scheduler、memory、fusion 或更贴近真实请求的 workload。对尚未关闭且预期 primary ROI 最高的方向，先检查它是否只被失败实现或不充分 proxy 连带拒绝；存在关键知识或因果缺口时按 `research_augmentation.md` 质证。只有该方向也有证据表明收益上限不足、成本不值得、不可行或超出授权，才形成全局 terminal reason。

出现以下情况应尽早停止：

- 与当前指标匹配的收益上限低于最低有效收益；
- 精度或 dispatch identity 失败；
- screen 已按预先声明的证伪条件拒绝机制，或 conservative bound 已证明收益上限不足；
- 重复证据已经否定该机制；
- 下一步超出用户授权的时间、GPU、风险或修改范围；
- 重新完成跨 subsystem 候选排序后，没有新的非重复且值得投入的方向。

只有局部结果而 primary 未达成时，不得把任务标记为优化完成。若仍有更高 primary ROI 方向，
继续该方向；若没有或成本已不值得，保持 original 或已有 Champion，并以“主目标未达成”的
terminal reason 停止，同时在 Handoff 中另列局部结果。

修复 runner、测试集或依赖不算性能 iteration。若基础环境问题持续消耗时间，应停止优化并单独报告，而不是把环境维护包装成候选实验。
