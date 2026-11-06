# 实现决策

依据：`self_curate_codex_blueprint.md` v0.1，SHA256
`25ba1bd1587f0b5debd9c016b93d54cb3524ee994f55afc776353ebd2f6fcd89`。
日期：2026-11-01。原始蓝图保留原样，不将其中示例当成已冻结实验配置。

## 本次已确定

| ID | 决定 | 理由与影响 |
|---|---|---|
| D01 | 本次范围是第 6 节 M1，不是完整 P0/P1 | 只搭 M1 必要基础，不创建空 Agent/训练/梯度类 |
| D02 | 独立 `src/self_curate` 包，运行时无第三方依赖 | 数据模块可在 CPU 验收，不绑定服务商或旧项目 |
| D03 | JSON/JSONL 严格 schema，未知字段、重复 JSON key、非有限数报错 | 不静默丢失轨迹或证据 |
| D04 | 第一版仅支持文本与显式结构化 tool_calls | 多模态、其他来源格式须独立 adapter，不能猜测转换 |
| D05 | token span 是完整序列中的全局半开区间 | 每条证据绑定消息位置、tokenizer hash 和 template hash |
| D06 | assistant payload 全部监督，头部、分隔符与其他角色不监督 | adapter 必须准确给出 payload spans；需要不同 EOS/mask 规则时升版本 |
| D07 | 工具动作以结构化 tool_calls 计数 | 不通过文本中出现 shell/test 等字样猜工具调用数 |
| D08 | `spans=null/未提供` 表示缺失，`spans=[]` 表示已标注但无 token 证据 | 缺失指标为 null；明确空证据计数为 0；总证据并集为 0 时 share=null |
| D09 | 占比分母为所有技能证据与监督 mask 相交后的并集 | 各技能内部去重，跨技能可重叠，因此 share 总和可大于 1 |
| D10 | complete_events/behaviors 是有来源的外部标注 | M1 保存和校验结构，不用 Teacher 自动推断“真实修复成功” |
| D11 | quality pass 要求 executable/manual 来源与带 hash 的证据引用 | M1 校验声明，尚不运行或认证这些证据；Teacher-only pass 拒绝 |
| D12 | M1 保留 pass/fail/unknown，质量门禁由 M2 固定外壳执行 | 数据规范化不隐式只留下成功样本 |
| D13 | 默认排除 accepted 历史；rejected 的允许/排除必须显式配置；pending 总是排除 | 前向或提取过梯度不等于训练过；拒绝经历不能消失 |
| D14 | 完整轨迹不裁剪；超长策略为显式 reject 或 error | reject 写排除清单；error 不发布数据池 |
| D15 | content_hash 绑定规范化消息，record_hash 另外绑定来源、mask、标签、质量与使用历史 | 同内容跨来源可由 M2 精确去重，来源证据仍独立保留 |
| D16 | fixture 模式显式标 synthetic=true；外部预计算模式的真实性为 unknown | token IDs 本身不能证明由所声明 tokenizer 产生；audit 不声称核验来源真实性 |
| D17 | staging 完整后发布，已存在 output 拒绝覆盖 | 完成 manifest、文件 hash、记录 hash 都需可复核；失败 staging 保留 failure.json |
| D18 | 所有未接入的真实 tokenizer、模型和实验能力明确留待后续 | 不用 fixture 冒充“真实模型已验收” |
| D19 | M1 即定义 SignalRecord/ModelRef/SignalSpec 与外部 provider 接口 | 计算算法可后接，当前数据契约已有明确接入与传递位置 |
| D20 | signals 是模型/范围/计算配置限定的记录列表 | 同一轨迹可保留多个 checkpoint，缺值为 null，不生成假的零分 |
| D21 | 信号绑定实际 tokenization 与有效选择位置 hash | 变更 token、mask、模板、技能证据位置后不可错用旧分数 |
| D22 | 百分位额外绑定参考样本、CDF 版本/文件、模型与 SignalSpec | 同一 raw 值可有不同 CDF 视图；不在 M1 计算 CDF |
| D23 | 同身份互补 NLL/entropy 可合并；相互矛盾的数值报 SIGNAL_CONFLICT | 不覆盖旧计算结果，模型/配置不同的结果分开保留 |
| D24 | 挂接标注默认拒绝令技能范围信号过期；显式 drop_stale_signals 才移除新对象中的过期项 | 全 assistant 范围的兼容信号继续有效；旧对象和前向计算历史均保留 |
| D25 | m1-v2 增加 signals，兼容只读加载 m1-v1 | 不改旧 snapshot/数据池来伪造恢复或迁移 |
| D26 | 按用户澄清，不做持久化 operator 查询索引 | 使用按 ID 和完整版本建立的只读内存 MetricView，不提前实现筛选算法 |
| D27 | 单/多指标批次严格保序并标记缺失与版本 | 拒绝未知/重复 ID，不靠行号联接不同子集，不填零或静默删行 |
| D28 | 新增 GradientSpec/GradientFeature，仅存向量引用 | 绑定数据、模型、坐标、loss、scope、精度、投影、变换、可选优化器、实现和配置 |
| D29 | AnnotationRecord/MetadataRecord 绑定 data_id、content、source 和实际 tokenization | 即使内容相同也不能给另一个 ID 挂错标签；同源版本变化明确失效 |
| D30 | 注册技能名称与实际样本技能标签分离 | evidence_metrics 中缺失的注册技能不等于样本已具有该技能 |
| D31 | 使用历史只追加或由 pending 进入终态 | 不允许删训练尝试、改 receipt/model、将 accept/reject 互相反转 |
| D32 | m1-v3 增加元数据/标签封装/梯度槽位 | 保持 m1-v1/m1-v2 的只读校验、旧文件与旧 record_hash |
| D33 | 2026-11-02 完成本轮 M2/P1 八类算子，算子语义 m2-v1 | 与 M1 分包；不提前接 Agent、真实模型和训练后端 |
| D34 | 配方与系统 ExecutionContext 分开；固定 QualityGate/BudgetPack | Agent 后续只能修改配方，不能放宽系统预算、质量和参考上下文 |
| D35 | 有序单输入链、共享前缀、可选终端 Mix | 拒绝任意代码、环、无效依赖和 Mix 后过滤，保留 JSON 顺序与真实依赖的区别 |
| D36 | 通过 M1 MetricView 读取精确模型/spec/CDF 的信号 | 缺失 error/exclude/request；请求不触发计算，不改候选集参考 CDF |
| D37 | Mix 先唯一归属，再按完整轨迹监督 token 分配 | 重叠分支不重复计数，短缺不静默改变配比；最终按实际份额复查 |
| D38 | ordered-greedy-v1 组批及 batch 整除修整是确定性启发式 | 不裁剪轨迹、不自动放宽；返回 infeasible 不构成穷尽组合的无解证明 |
| D39 | 最终 BatchManifest 同时保存筛选版本和训练 token 身份 | M5/M6/M8 传同一清单；后补梯度允许改变 record_hash，但不能改训练内容、token/mask 或顺序 |
| D40 | 语义去重只接外部版本化成对相似度 | 不计算 embedding；已知内容版本过期报错，缺失对有明确政策 |
| D41 | 稳定 hash 实现选择随机性，系统训练 seed 独立控制最终排列 | 不使用进程全局 RNG；相同快照/程序/上下文字节重放 |
| D42 | 复查版 0.2.1/m2-v2 使用精确十进制预算窗口和分支比例校验 | 消除浮点补偿带来的隐含放宽；拒绝超过 JSON 安全整数范围的 token_target；旧上下文明确失效 |
| D43 | batch 修整维护总量和分支汇总，不反复扫描每个剩余子集 | 保持移除优先级和同分规则；以独立参考算法作变量长度对照 |
| D44 | 最终清单单独报告历史暴露 ID、样本比例和监督 token 比例 | 允许复用时仍可审计；前向、梯度、候选、接受和拒绝训练不混为一谈 |
| D45 | 无活动模型时也禁止上下文混用 tokenizer/template；verify_batch 支持预期模型检查 | 维护训练 token 一致性，并给 M5/M6/M8 提供明确的模型交接校验 |
| D46 | 0.3.0 补齐 P0 五个核心契约及 WorkflowBundle，新增 schema=1 | Task/Rollout/Evaluation/Capability/Round 与既有程序、批次和样本梯度交接；M1/M2 语义不变 |
| D47 | 目标侧另设 TargetGradientSpec/Feature | 明确 reward_ascent、开发任务、rollout、奖励/advantage/聚合版本；不把 SFT 样本梯度直接当作能力目标 |
| D48 | pending/invalid 保留 null，不产生可排名总分；环境失败不计零奖励 | 零目标单独标注，缺梯度和无效目标不能参与最佳可行比较 |
| D49 | 成本事件唯一，CostRef 绑定事件内容；未知计量总数保持 null | 保留失败、复测与拒绝成本；P0 不执行扣预算，P2 接入幂等事件 |
| D50 | 能力统计显式声明 task_mean/rollout_mean，并按技能 ID 匹配 | 重复 rollout 不冒充独立任务；不确定性未知时不造值 |
| D51 | RoundRecord 仅校验模型/批次/版本与证据关系，不执行验收策略 | 实际阈值通过固定 policy artifact 提供，P2/P6 再实现执行和提交 |
| D52 | validate_workflow 区分记录检查、训练输入检查与 artifact 内容核验 | 传入 MetricView 才核对实际 token/mask；不因 schema 通过宣称向量/环境/权重真实可用 |
| D53 | 0.4.0 的 p0-v2 校验支持外部模型、目标与选择上下文锚点 | 内部自洽不等于符合系统固定政策；未提供锚点的回执明确标 false |
| D54 | P0/M2 共用 selection_identity；成本 ID 索引构造一次且不序列化 | 保留 m2-v2 既有清单身份，避免后续 hash 公式漂移与成本引用的反复扫描 |
| D55 | P1 prepare-selection 生成只有真实选择结果、成本和空下游字段的工作流 | 不重抽、不伪造梯度/估值/训练；正式衔接 P0，无需手拼 JSON |
| D56 | attempt_id 与 invocation_id 区分外层尝试和物理选择调用 | 真实重复执行应分开计账；P2 恢复读取已完成结果，不把重复运行当作恢复 |
| D57 | 原始快照用于选择重放；追加特征后的数据池用于消费既定清单 | 训练内容/token/mask 不变即可追加梯度；新选择显式建立新上下文 |
| D58 | 0.4.1/p0-v3 增加独立 expected_batch_id，P1 自动绑定实际结果 | 系统模型/目标/context 相同仍可能对应不同成员或训练顺序；消费历史最佳时必须核对已保存的精确清单 |
| D59 | 独立报告先检查自己的引用与成员完整性 | valid 估值必须有对应目标和样本 cache 引用；能力指标成员并集须等于报告成员；完整跨记录验证仍不可省略 |

## 后续须冻结的研究语义

- 技能体系与人工标注/事件充分性标准（M2/M4）。
- 真实 tokenizer/chat template adapter、特殊 token/EOS 的监督策略（M3 前）。
- 数据源及来源真实性核验，任务家族与近似题面的隔离策略（真实数据导入前）。
- 模型、参数坐标、梯度 loss/reduction、投影与优化器变换（M3/M5）。
- 奖励、rollout、开发集、目标聚合与接受阈值（M4/M8）。
- token/batch、训练配方、重复曝光和被拒绝数据的正式复用规则（真实试点前）。
- 搜索与外层预算、最终测试隔离及确认性比较（对应试点/正式实验前）。

这些不阻碍当前 P0 契约与 M1/M2 的 CPU 实现；不能用示例值替代正式决策。

## 2026-11-03：Leval / Lopt 单步交付

- D60：验证计算暂缓，ValidationRequest/Result 绑定精确模型、程序、批次、目标、上下文、measure 与协议；仅检查数值一致性。
- D61：Leval 确定性保真，修改建议只作待检验假设；无效或待验证结果不以零替代，不触发优化。
- D62：Lopt 提供规则策略和共享 ModelRef 的模型适配器；检查完整 ProgramSpec、实际 diff、谱系和编辑预算，保留失败回复与未知成本。
- D63：候选复用 P1 在冻结 context 下运行，只交出下一次验证请求，不宣称改进、替换历史最佳或训练；单步链路与完整 P2 状态机分开。

## 2026-11-03：全模块复查

- D64：P1 的执行、发布和已存结果读取分开。publish_preparation 不执行算子；load_preparation 保留已存成本并在原始视图验证。反馈输出附带标准候选 handoff，支持下一轮直接消费。
- D65：规则候选与模型候选均保留校验失败和调用证据；调用 hash 绑定实际输入，提案 ID 绑定实际输出。固定 seed 策略同时锁定 seeded operator 的节点命名空间。
- D66：共享模型客户端每次调用前后都须保持原 ModelRef，变化即停止后续调用。Lopt 获得冻结控制配置、context_hash 与 artifact 摘要；不传完整成对特征表，这些配置仍不进入模型可修改字段。
- D67：外部报告转接核对精确 coverage，真实域拒绝已知 mock 目标/参考上下文；M1 读取核对存储配置和审计标签。原文档与历史证据不改写。

## 2026-11-03：三个共享参数角色

- D68：RoleTeam 的执行、验证解释、优化三个角色使用同一个客户端实例和不可变 ModelRef；每次调用前后检查版本，真实服务的权重真实性由后端核验。
- D69：每角色一个固定 PromptTemplate，正文允许为空；自动组装 role-work-v1 输入。角色、session、模型和 prompt hash 绑定各自 history，不共享对话列表。
- D70：History 追加 before/after 事件，保存真实结果和失败；仅把本角色最近工作送入模型，完整记录保留。只通过显式工作产物跨角色传递信息。
- D71：快照显式保存到新目录，恢复校验身份与 hash 链，不重跑已完成工作；未结束事件必须显式核对。实时刷盘和完整搜索恢复仍属于 P2。
- D72：角色层复用原数值检查与算子，模型解释不能覆盖分数，执行角色不能编辑程序。运行器账本只统计物理模型调用，P1 与 attempt 引用避免重复累计。

## 2026-11-03：两遍框架复查

- D73：同一业务会话固定 context/objective/measure/protocol/edit policy，RoleScope 可部分绑定后补齐，但已有条件不可改写；旧业务历史仅从确定性记录推断约束。
- D74：RoleCycle 保留源请求/结果/上下文/策略，绑定实际调用 ID 与 attempt；完整产物通过 publish_role_cycle/load_role_cycle 交接，history save 不冒充候选持久化。
- D75：恢复允许独立 session/attempt/backend/scope 锚点；Team 的运行器与角色绑定只读，整个业务 cycle 串行收集自己的模型调用。
- D76：执行失败通过显式 previous_cycle 摘要交给 Lopt，不共享整段角色历史、不伪造数值验证。自动搜索与接受仍留给 P2。

## 2026-11-04：按代码阅读教程复查

- D77：MetricView.data_ids 改为只读属性，保持既有 tuple 读取接口。候选范围改变必须构建新视图和 ExecutionContext，不允许在同一缓存快照/context 身份下替换公开成员列表。
- D78：数据池、选择、P1、反馈、cycle、角色快照与 demo 统一使用短名称同级 staging。独占 mkdir 并重试名称碰撞，继承工作区 ACL；不重复拼接完整 output 名和 UUID，保留现有发布锁、原子 rename、拒绝覆盖及失败证据。此变更不宣称支持任意长度 Windows 路径。

## 2026-11-04：内层循环

- D79：退出条件由调用方注入 StopCondition，返回 StopDecision；默认 None，自动 run 明确返回 awaiting_stop_condition，不隐式无限运行或冻结示例阈值。手动 step 一次只执行一项提案/筛选或一次验证操作。
- D80：最新已验证候选与历史最佳分离。仅 valid 且满足固定约束的结果可按 total_value 严格比较；平分保留先者，负分不以零基线排除。best 保存同一 PreparedSelection、请求、实际 BatchManifest 与外部结果，不重抽训练成员。
- D81：pending/invalid 验证只等待，不冒充零分或触发优化；失败操作保留证据、已知/未知成本并阻塞，显式 retry 或新外部证据才继续。完整的不可行/blocked cycle 可通过原 previous_cycle 反馈给下一轮 Lopt；没有验证数据的失败候选不生成数值收益。
- D82：搜索成本复用 CostLedger，去重合并 seed/P1、物理角色调用及本次调用验证器的费用；不叠加工具 history span 或逻辑 Lopt attempt 的同一笔模型费用。导入既有验证结果默认只保留原 usage，不再次收取；异步新工作可显式传入绑定当前请求/模型/attempt 的原 CostEvent，按原事件 ID 去重。验证 usage 须为该次独立调用的增量费用，不能重复提交累计总额。调用身份绑定 search/session/外层 attempt/模型。
- D83：搜索快照包含完整 P1、cycle、验证序列、role history、账本与 timeline。恢复重放确定性状态并独立核对初始请求、编辑政策、search/session/attempt/执行模式；不调用模型或验证器。可选 journal 在外部操作前后保存新快照；在途恢复为 interrupted，不能冒称完整 P2/M9 的外部任务完成对账。
- D84：完整搜索快照拥有未发布目录时，cycle 使用内部写入函数填充子目录，整体完成后统一发布，避免无意义的额外嵌套 staging。既有公开 cycle 格式和独立发布行为保持兼容。

## 2026-11-05：全模块审计

- D85：MetricView 构造时统一核验工具调用关系、assistant mask、技能证据派生指标、证据总量和技能唯一性。直接传入 TrajectoryRecord 不能绕过文件入口已有的语义检查；指标批量查询仍保持按 ID 的内存索引，不逐次重算。
- D86：外部操作派发前的检查点写入失败，回滚本次未启动的操作号与 pending 标记；操作已经完成后的保存失败不回滚其结果或费用，只能补保存。真正进行中的调用恢复仍为 interrupted，不自动重发。
- D87：优先解析有效 usage，再处理回复 JSON 序列化和内容校验。坏回复仍失败，已报告的有效费用保持原值；没有合法 JSON 的回复不伪造 response hash。Lopt 耗尽重试时按实际错误概括，不把后端/协议错误一律写成 schema 失败。
- D88：TrajectoryRecord 暴露派生的 known_mock_artifacts，不改变序列化或历史 hash。real P1 拒绝候选上下文中的显式 fixture/mock 轨迹来源；real 工作流消费视图时也核对选中记录的来源、质量证据、特征、标签及使用历史引用。没有已知 mock 标记不等于真实性认证，权重/向量/任务结果仍须由真实后端核验。
- D89：独立 OptimizationAgent 使用 previous_cycle 时也核对 source_policy 与当前 EditPolicy，不能把不同编辑条件下的旧失败作为同条件执行反馈。内层已有的固定政策和历史最佳规则保持不变。

## 2026-11-05：用户确认的外层两阶段训练

- D90：用户新增筛选轨迹 SFT，确认接着子集权重训练；每次验证后判断接收／退回。第一阶段退回结束尝试；第二阶段退回保留第一阶段模型。此要求扩展原蓝图，不改写研究原文。
- D91：训练／任务执行分包；数据／任务和配方显式接入，torch 为可选依赖。微型 CPU 实际更新与 fixture 能力结果分开报告。
- D92：子集消费 best 精确清单及顺序；筛选轨迹独立语料绑定停止搜索 handoff。两阶段各有 P0 TrainingRecord，不强塞进原单训练 RoundRecord。
- D93：每阶段隔离候选、新优化器，仅接续权重；按累积窗口实际监督 token 归一，不截断／重抽，保护冻结参数和调用方 RNG。
- D94：固定开发 suite／seed／采样／协议，验证接收实际 checkpoint 引用；按任务奖励汇总，环境失败不记零；无 baseline 不宣称增益。
- D95：判断阈值待填，无 policy／判断器就等待。决定绑定阶段模型、训练回执、能力报告、policy；补证显式复测并计费。attempt 内指针原子更新，不宣称全局提交或完整恢复。

## 2026-11-05：两遍适配复查

- D96：同一补证决定保持等待；新验证的复测授权须绑定新上下文，run/step 不隐式重调判断器。
- D97：TaskEvaluation 绑定逐次 TaskRunRequest/Result、协议、checkpoint、attempt 与成本。旧报告可读但缺证据不得判断；通过只读目录 loader 核对并补齐，不改历史 artifact。
- D98：外层消费精确 TrainingRequest 和验证 checkpoint；拒绝内部自洽但属于另一份请求／权重的结果，保留已发生费用。DecisionContext 同时绑定 attempt。
- D99：任务派发前的目录／请求写盘错误保留此前调用成本，不给未派发工作制造一次模型事件；TrainingOutcome 导入也必须符合训练预算。
- D100：接续训练的 requires_grad 参数名与已保存 checkpoint 元数据一致；真实 loader 的参数隔离和实际权重真实性仍由 model store 核验。
- D101：训练与任务执行后端返回错误类型时，也先保留有效 ResourceUsage，再报契约错误；与现有角色运行器的失败成本语义一致。

## 2026-11-06：用户指定两套 Leval 实现

- D102：用户新增可选的内层候选 SFT 评价，扩展原蓝图；三个共享角色及参考模型仍冻结。所有候选从共同参考模型隔离训练，不提交、不接上个候选，不触发外层筛选轨迹 SFT。
- D103：method=nice/sft 在构造请求/会话前选择。协议绑定模型、参考数据/任务 suite、配方与实现；分别返回无量纲 NICE-inspired predicted_gain 与任务 reward_delta，不跨方法比较，也不由 Lopt 改协议。
- D104：参考用户 NICE 项目数学定义，样本 token-mean CE；任务 reward*NLL-token-sum/固定 rollout 数；全局裁剪后 fresh-Adam、再 CountSketch；按任务及实际子集成员平均 cosine。零任务留分母，全部零阻塞，不等价于 reward-ascent 目标内积或完整论文复现。
- D105：TorchNICEBackend 实际求导但不创建优化器；隔离模型、固定 trainable 坐标、参数/buffer 前后 hash、RNG 恢复。只为选中成员求梯度，内存缓存绑定完整模型/输入/掩码/配方/代码身份。
- D106：SFT 验证复用外层 TrainingService/TaskValidator/TaskSuite；基线需协议证据，首次计算或显式导入；按技能计算奖励增量。基线状态通过独立 hash/协议锚点恢复，不重算或重复收费；无有效奖励不填零。
- D107：LevalContext 只使用同模型/objective/context/measure/protocol/backend 的已完成数值观察，最多 8 条；提供本次、相关历史、相对上一 valid/窗口最佳可行的代码比较和原 LevalFeedback 槽位。模型解释不能改数值。只向 Lopt 显式 handoff，不读取别的角色私有 history。旧记录缺身份时不猜测。
- D108：评价内部详细 ledger 和未知小计保留，搜索仅计一次增量汇总；验证后端异常带有效 usage 时也保留。候选训练通过 updated_records 记录 pending candidate_sft，不污染原池或伪造 accepted_sft。真实数据/生成重放真实性和服务器目标模型验收继续由后端承担。

## 2026-11-06：Leval 两轮文档与模块复查

- D109：SFT/NICE protocol 绑定内核、后端/服务源码和明确声明的实现身份；真实候选验证适配器必须提供包含实例配置的 implementation_hash。StateDictModelStore 同时绑定模型 factory，不能仅靠相同权重文件假设计算实现相同。
- D110：Torch NICE/SFT 自己禁用 ambient autocast，不改变参数原生 dtype；CE 为 float32。执行证据记录 torch、设备、TF32/确定性等数值条件；条件变化使协议/缓存失效，运行中发生变化则失败并保留费用。
- D111：自动基线有独立 reference_namespace，不能由另一空 provider 重测后冒充原基线。显式相同参考保持稳定；reference schema=2 同时保存候选使用历史，独立 hash/protocol/input 绑定恢复、单调合并，不逆转终态。旧记录只读，新实现不静默续用旧协议。
- D112：服务坏返回或无明细异常保留已报告 usage；无 ledger 的边界调用成本保持未知，不填零或编造 rollout。已有详细/有效空 ledger 按原事实保留，搜索仍只计一次汇总。无效基线保存 attempted_baseline；已知超长在任何模型调用前拒绝。
- D113：NICE real 域 URI 统一 strip/lower 检查；本机反例仅是合约 fixture。评价目录采用 UUID 全 128 位的紧凑编码，缩短 Windows 嵌套路径；不改任务完整请求 hash 或宣称任意长路径支持。
