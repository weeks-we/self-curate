> 2026-11-11 更新：本文下方是切换 Rebench 前的全框架缺口盘点，涉及5016/固定七技能/旧V3入口的描述不再用于新实验一。当前实现、待用户填写项和真实验收边界以 [Rebench计划](EXP001_REBENCH_PLAN_ZH.md) 与 [新交接](handoffs/EXP001_REBENCH_WORKSTATION_V1.md) 为准。已补Rebench→M1/Teacher装配、Pro V2 pristine regrade、离线资产及FSDP专用恢复预检；技能规则仍待用户确定，真实GPU/容器未在本机验收。

# 当前流程待配置、待实现与待验收清单

核对日期：2026-11-11。依据当前 self-curate 源码、指导书、配置和交接文档做静态核对。本次没有重新执行训练或 GPU/Docker 验收，不把之前 CPU mock 的通过结果当成真实实验通过。

结论：图中的八项覆盖了主要研究决策，但不能理解为“填八个配置就能执行完整 RSI”。还缺部分真实后端装配、实验级控制和服务器验收；同时，外层单次 attempt 的进程恢复已经实现，不能再整体列成未实现。

## 1. 八项逐条对齐

| 项目 | 当前已经有 | 当前真正缺什么 |
|---|---|---|
| 1 数据池与成功/失败准入 | 历史5016读取、轨迹规范化、准入/隔离、隔离区及审计记录 | 最终来源方案、成功的证据标准；全池方案需要改适配；Raw Random另见后文 |
| 2 七技能与Teacher审计 | 七技能定义草案、Teacher prompt、API调用、引用到token映射及审计入口 | 确认定义、边界案例、模型/API预算、复核与争议处理；真实标注尚待执行 |
| 3 三角色业务prompt | 固定协议、JSON结构、共享模型/私有history及反馈槽位 | 通用业务正文为空；EXP001另有已填的一次性prompt，不能替代循环版 |
| 4 Leval目标/NICE配方/回放 | NICE数学内核、autograd、隔离候选SFT、参考契约与历史反馈 | 科研配方、真实回放适配和正式provider装配；不是只填权重 |
| 5 内层停止 | StopCondition/StopDecision、手动step、历史最优与保存恢复 | 正式停止/预算规则及配置到回调的接线；mock固定5次不代表科研标准 |
| 6 外层接受/退回 | DecisionContext、AcceptanceDecision、两阶段判断、补证据等待 | 确定性判断策略实现及阈值；目前没有可直接采用的正式Decider |
| 7 控制轨迹语料 | CurationDataProvider、JSON数据提供器、训练数据契约及第二阶段训练 | 真正的历史→轨迹语料生产器、监督内容、质量门槛与预算 |
| 8 接受后下一轮与恢复 | OuterCycle.restore、原子journal、数据使用/费用恢复、CPU跨进程验收 | 完整真实多轮runner、更新后agent实际加载checkpoint、历史转交和全实验恢复 |

### 1）正式数据池及准入：先确定“成功”指什么

代码在 [historical.py](../src/self_curate/preparation/historical.py)、[pipeline.py](../src/self_curate/preparation/pipeline.py) 和 [sources.json](../configs/experiments/exp001/sources.json)。当前正式路径仍是 historical_exact 的5016条；pipeline 对真实输入检查行数必须为5016。若最终改用更大的轨迹池，需要同时改来源加载、schema、任务映射、隔离和资产身份，不是只改路径或expected_rows。

一个需要特别确认的现状：pipeline排除 resolved=False，但允许 resolved 缺失并记录警告；随后在来源与静态完整性通过后写 quality.status=pass、method=executable，证据中 fresh_execution 仍是 unknown。这是“历史来源＋静态准入”，不能解释成每条轨迹都重新执行并成功。若实验定义要求成功轨迹经过真实执行验证，应补相应过程并对齐quality的语义；Teacher不能代替执行证明。

应冻结：来源revision、数据单位是任务还是轨迹、重复轨迹策略、失败/未知/截断处理、训练与开发/终测隔离、数据池分母以及每轮选样预算。产物应是带原始来源和准入理由的不可变pool manifest，而不是只有一个数据集名称。

### 2）七技能与Teacher：有草案和实现，但没有最终科研定义

[skills.py](../src/self_curate/preparation/skills.py) 已定义 code_localization、failure_reproduction、error_analysis、hypothesis_revision、implementation_edit、test_construction、regression_check，且有现成标注prompt；文件中也保留待确认TODO。

要确定每种技能的正例、反例、完整证据链和不确定状态。例如，调用测试不等于回归检查完成；改变编辑位置不一定是修订假设。Teacher输出必须指向实际消息引用，代码再映射到token位置，不能凭自然语言评价直接造标签。

Teacher配置在 [server_bootstrap.template.json](../configs/experiments/exp001/server_bootstrap.template.json)，调用实现在 [provider.py](../src/self_curate/preparation/provider.py)。模板中的模型名、地址、调用预算和可选review_teacher是待确认配置，不能当成服务已经可用。还需冻结审计样本、复核触发条件、分歧和not_assessed的处理，以及标注版本变化时哪些资产作废。

### 3）三角色prompt：补业务说明，不改数值权力边界

[configs/roles](../configs/roles) 三个instructions均为空；[agents/prompts.py](../src/self_curate/agents/prompts.py) 已有不可变角色协议。EXP001的 [专用prompts](../configs/experiments/exp001/prompts) 中Lopt/Lexec已经有正文，但它明确是一次性静态策略，不是完整动态循环。

通用Lopt需说明如何读能力报告、池摘要和可比历史，如何在固定算子集合内提案；Lexec需说明只执行给定程序并报告缺失/不可行；Leval需说明如何引用代码计算的事实，区分观测与假设，不能改分数或接受决定。

这里的“筛选程序”是 [ProgramSpec与八种算子](../src/self_curate/contracts/program.py)，不是任意Python源码执行。prompt应匹配真实输入槽位和输出schema，避免要求模型调用尚不存在的工具。正文、模板和采样参数都需要版本冻结。

### 4）Leval：目标、配方与真实回放三件事分别完成

[valuation/nice.py](../src/self_curate/valuation/nice.py)、[torch_nice.py](../src/self_curate/valuation/torch_nice.py) 已有确定性NICE风格估值与autograd；[valuation/sft.py](../src/self_curate/valuation/sft.py) 已有可选隔离SFT验证。NICE是单checkpoint的fresh-Adam方向余弦代理，不等于已经测得真实训练收益。

待确定的配置包括：ObjectiveSpec的技能权重与保持约束；NICERecipe的可训练坐标、clip、epsilon、投影维度/seed；开发任务、rollout次数与采样协议。更换方法或量纲须创建新协议，不能把nice分数与reward_delta直接比较。

关键接入缺口：现有 [execution/agent.py](../src/self_curate/execution/agent.py) 保存输入序列和消息，但明确写 native_generated_token_mask_available=False；Harbor路径也注明梯度mask不可用。[ReplayTrajectory](../src/self_curate/valuation/contracts.py) 则要求真实输入前缀及生成动作labels，并与每个rollout、token计数和tokenizer绑定。目前源码没有把任务runner结果自动构造成NICEReference的生产桥接。

因此应实现实际生成token/工具参数/模板边界对齐、回放构建和前向一致性验收，再构造真实NICE provider。动作hash、语义工具动作和API报告的token数都不能替代逐token mask。当前CodingTaskRunner还固定greedy协议；若研究需要随机多次rollout，也要扩展并冻结协议。

### 5）内层停止：已有回调，缺研究策略与实际预算连接

[InnerSearch.run](../src/self_curate/search/inner_loop.py) 没有stop_condition会返回awaiting_stop_condition，避免无限循环。当前CPU联调用固定5次作为验收情景。

正式规则需定义有效评估次数、最大候选尝试数、连续无改进、最小改进幅度、无可行候选和预算耗尽时如何退出。invalid/pending是否占尝试预算，应明确区别于有效评估计数。

还需把token、墙钟、梯度计算和任务调用预算接到停止/派发前检查；只有CostLedger统计，不等于已经有全实验资源预留与扣减。未知花费不能算0。规则落到确定性StopDecision回调，结束时仍交付历史最佳的已评价批次。

### 6）外层接受：有状态机，不存在可直接采用的科研阈值

[outer_loop.py](../src/self_curate/orchestration/outer_loop.py) 支持accept/reject/needs_more_evidence和两阶段判断。标准应由确定性Decider产生，不能靠LLM自由决定。

需分别规定：子集SFT相对哪份基线比较、轨迹SFT相对已接受子集模型还是另一个参照；主指标增益、保持技能退化约束、有效任务覆盖、平分规则、补证据条件和预算。环境失败不能当0分模型失败，也不能根据最终benchmark选择权重或阈值。

当前mock决定是预设序列；EXP001则刻意不做checkpoint接受/退回。应实现独立策略模块和版本化配置，消费DecisionContext输出绑定其hash的AcceptanceDecision。不要把EXP001中禁用评价/接受的协议引用当正式RSI判断规则。

### 7）控制轨迹语料：外层训练器有了，真正的语料生成器没有

已有 [CurationDataProvider](../src/self_curate/training/contracts.py)、[JsonCurationDataProvider](../src/self_curate/orchestration/io.py)，以及外层第二阶段训练。现有CPU mock只提供 input_ids=[1,2,3]、labels=[-100,2,3]，不是真实历史总结。

应明确一个训练样本的输入与目标。例如，输入包含当前能力报告、候选策略与此前可比结果，目标是合规的策略修改和理由；该修改之后的评价可以用于事后质量筛选，但不能倒灌进修改前的输入，造成未来结果泄漏。

需实现历史读取、角色/轮次/模型来源追踪、负面尝试保留或筛选规则、摘要模型选择、去重、质量检查和精确assistant mask。还要冻结数量/token预算及训练配方。若关闭轨迹SFT作为消融，应有明确模式，而非喂占位token绕过第二阶段。

### 8）下一轮总控：单次恢复已完成，多轮真实闭环仍缺装配

[OuterCycle.restore](../src/self_curate/orchestration/outer_loop.py) 与 [outer_recovery.py](../src/self_curate/orchestration/outer_recovery.py) 已实现新格式attempt恢复，完成结果复用、决定续交、M1使用/曝光/成本恢复和未知调用保护。不能继续把它写成未实现；也不能据此宣称全局模型服务切换与整个研究实验已经完成。

目前 [smoke/runner.py](../src/self_curate/smoke/runner.py) 的三轮装配是mock；正式 [EXP001 runner](../src/self_curate/experiments/exp001/runner.py) 是Random/static_m0，明确不执行Leval、后续刷新、控制SFT或接受判断。

真实总控需要在每轮终态后确定accepted ModelRef及checkpoint，重建三角色、scorer、Leval reference、context和历史命名空间，并携带曝光和预算进入下一轮。拒绝时保留正确父模型，不抹掉已训练数据记录。

还有一个具体接线点：[SharedRoleClient](../src/self_curate/experiments/exp001/server.py) 当前给router.bind传checkpoint=None；[AdapterRouter](../src/self_curate/execution/model.py) 只允许M0省略checkpoint。因此，不能只给这个客户端换ModelRef就让新target接管下一轮，必须把已接受adapter接到真实角色调用链。跨轮history应显式转交摘要/引用，而不是改写旧事件的model_hash。

## 2. 原清单外应单独列出的六类事项

这些事项有的与原八项相关，但应独立验收，不能被“总控”或“配数据”一句话盖过去。

### A. Raw Random路径与后来的实验定义尚未对齐

你后来提出静态Random从质量准入前抽。当前EXP001的Random和static仍共享inputs.records与context；Random走同一个prepare_selection质量门禁，并没有独立raw_random_pool输入。仅不用NLL/技能不等于准入前随机。

需区分原始可解析/可编码/长度合法/完成集合隔离的最低结构检查，与研究框架额外的质量选择。然后保留raw与admitted两份来源清单，为Random接独立入口，冻结选样分母与预算。不能让格式损坏数据直接进入训练，也不能悄悄把框架准入后的结果当原始随机基线。

### B. 任务开发集、最终benchmark和执行协议要正式冻结

任务注册、来源锁、环境/判分适配已有实现，但仍需确定实际开发题单、技能分层、题数/seed、最终benchmark子集或完整集、重叠关系以及token/时间/工具预算。当前sources.json登记了多个SWE套件及Terminal来源，不等于它们已经在服务器下载、隔离或成功判分。

同一题多技能的计分规则、套件间重叠和环境失败分类都应固定。最终测试不能进入Lopt输入、NICE目标、策略选择或接受规则。切换更强harness时应新建协议，避免把harness变化当成数据筛选效果。

### C. 定向刷新规则已有，Lopt方向的生产与调度没有

[FocusedRefreshService](../src/self_curate/signals/refresh.py) 实现了首轮全池、后续定向且外层未用数据的刷新。[focus_from_outer_decision](../src/self_curate/signals/refresh_contracts.py) 仍要求调用方传skills；CPU mock的方向也直接来自配置。

需实现“外层结果→Lopt结构化方向→校验SkillFocus→刷新→新context”。还需约定下一轮只在focus池筛选，还是允许使用完整池；后者的范围外样本可能缺当前模型指标，不能用旧版本顶替。若采用percentile算子，要另建匹配模型和参考池的CDF，不能沿用旧CDF。主流程只用raw指标时无需为此提前增加复杂度。

### D. 实验预算与消融定义不能等到运行时临场决定

应固定样本预算或监督token预算、轮数、每轮规模、学习率/seed、轨迹SFT开关，以及额外轨迹训练的计算量如何记账。长轨迹的等样本数并不代表等token/等算力；不能将更多训练量的收益直接归因给筛选。

后续V1–V4还需对应配置和runner分支，不是填EXP001的modes就自动具备。通用初始策略如何产生、初始资产能复用给哪些变体也要冻结；现成的一次性初始策略生成/共享资产能力不等于通用动态runner已接好。

### E. 正式资产接入与八卡FSDP预检需要补齐和真实验收

模型/数据加载、Docker准备、FSDP、checkpoint和日志已有代码；实际路径、来源/镜像锁、API凭据环境、GPU UUID、模型架构/LoRA目标模块仍需服务器实物核验。模板null有些会由bootstrap生成，不应全部误判成需要人手填，也不能当成已有有效值。

一个实际缺口：[server/capacity.py](../src/self_curate/server/capacity.py) 目前仍是每个worker仅见一张卡的反向/重载预检。它不是八rank FSDP collective、分片优化器、分布式保存/恢复和adapter导出的验收。正式preflight现有receipt字段也未单独证明这些FSDP特性。需增加后端专用预检及可核验receipt，然后在8×H200上实际执行；不能把旧单卡sft_backward_passed当新FSDP已验收。

### F. 初始化、资产生产和整个实验的恢复仍有边界

训练恢复、单次OuterCycle恢复、内层恢复不等于整个多轮实验在任意位置都能自动恢复。EXP001的_resume要求已存在冻结resume-plan；初始化和一次性选样尚未发布完整计划时，会明确阻塞并要求核对。

真实总控仍需持久化当前轮号、最后接受模型、技能方向、共享资产消费、总预算及在途操作。Teacher未知调用、未提交的首次Lopt输出、半成品刷新和无法确认的外部任务，都不能盲目重发或计0成本。应提供核验/显式重试路径，而不是新建空history重新开始。

## 3. 哪些不是“还没实现”

- 数据ID/标签/长度/版本化NLL与entropy/梯度引用及高效内存查询已经有；不需要再加持久化operator索引。
- 八算子、精确批次、同模型三角色隔离、Leval反馈槽位、历史最优和单次外层状态机已经有。
- NLL/entropy计算及定向刷新已有模块，主要缺真实运行和上游方向/下游调度连接。
- FSDP训练、训练恢复、OuterCycle恢复、时间编号和日志已有代码与CPU层面的验收；真实GPU通过与否仍是另一个结论。
- 真实数据准备、Teacher调用和harness不是全空，只是最终研究配置、部分桥接和服务器执行尚未完成。

## 4. 建议执行顺序

1. 先冻结实验版本：保持当前5016，还是另做全池版本；修正Raw Random入口；冻结开发/最终测试及预算。
2. 完成数据准入与标签定义、Teacher审计，产出可复用真实资产。
3. 若只跑EXP001 Random/static_m0，可先完成一次性prompt和新FSDP专用预检，做真实小规模试跑；它不依赖内层NICE、循环停止、接受策略或控制轨迹SFT。
4. 要跑完整RSI，再实现真实Replay→NICEReference，冻结目标、通用prompt、停止与接受规则。
5. 实现控制语料生产、Lopt技能方向及真实多轮runner/角色checkpoint切换/实验级恢复。
6. 继续沿用CPU mock验收控制逻辑，再单独验收真实多卡、harness、最大上下文和强制中断恢复。最后才提交正式实验预算。

本次只更新核对文档，不自动更换数据集、指标、阈值或prompt，不宣称已修复上述缺口。
