> Teacher v1已按2026-11-14用户决定启用六技能。正式入口与复用方式见 [Teacher交接](handoffs/TEACHER_V1_WORKSTATION.md)；本文旧的待定技能、重叠任务组和复用标注prompt的描述已被取代。


## 2026-11-14：harness 二次复查与小额 DS 连通测试

发现并修复启动入口 importlib 作用域错误、SDK终态与HTTP usage提交竞争、非法seed返回500、
角色后续checkpoint无法传递、预算停止的已知计量丢失，以及Pro短期token写入trial JSON。
32项CPU适配、7项复查回归、真实SDK0.2.87的模拟transport控制协议检查、3×5联调、
EXP001预演/完成态恢复及ZIP CRC通过。DS真实JSON和两轮工具调用成功，合计成功输出76 token；
首次强制工具请求HTTP400保留记录。key不落盘。DS连通测试不证明Qwen/LoRA/CLI/容器/NICE真实验收。
完整证据：`runs/claude-harness-review-20261114/verification.json`。


## 2026-11-14：统一 Claude Agent SDK harness

三角色JSON会话、SWE/NICE rollout、Pro/Terminal解题改接固定SDK0.2.87+Claude2.1.153，
仍使用AdapterRouter的同一Qwen/adapter；workflow和确定性算子/梯度/训练/官方判分不交给harness。
新增vLLM0.22.0引擎token采集与NICE replay适配、种子middleware、预算网关及工作站SDK任务派生镜像。
旧mini安装/Pro协议和旧harness初测不能直接复用。原公共工作流契约保留。
实现与验收边界见 [harness说明](HARNESS_CLAUDE_SDK_ZH.md) 和
[交接](handoffs/CLAUDE_SDK_HARNESS_V1.md)。本机仅CPU模拟，真实CLI/GPU/Docker/NICE反向仍须服务器验收。

# 实现决策

## 2026-11-08：EXP001 恢复、进度与来源复核

- D139：共同前缀和选样完成后冻结恢复计划。恢复绑定源码/环境/配置/资产根和精确清单，不重调 Agent/评分或重选数据；完整分支只读核验。
- D140：连续 Torch 后端每 optimizer-step 发布权重/optimizer/Python-Torch-CUDA RNG/游标/checksum。部分累积窗口回到最后提交点，待发布目录不作恢复点；保留失败费用和不确定使用历史。
- D141：开发与最终测评按逐题完成证据恢复；未知在途外部调用先对账，不自动重发或择优重试。内核 lease 防止双进程写同一输出。
- D142：阶段进度原子更新、事件追加，ETA 只估计当前阶段；状态查看不加载模型。旧运行证据保持原格式读取，不改 source hash 迁移。
- D143：sources.json 保存官方完整 commit/config/split/Terminal 4.0.0；real 准入绑定 registry hash 和本地文件校验回执。历史 5016、tool 24100 和作者 ID 清单分开审计。
- D144：正式方案是文本 LoRA，base/vision 冻结；实际 Qwen PEFT/ModelStore、SWE-smith/SWE-bench/Harbor adapters 与服务器验收仍缺，不能把通用接口/mock 当真实后端。
- D145：每分支 global batch=1×8×1=8；四分支独立，不合成 batch32。八卡利用率尚未测；Terminal 原生 GPU/多容器/超时要求优先保留。

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

## 2026-11-06：用户要求前向 NLL/entropy 并回填 M1

- D114：SignalService 消费调用方的规范化 TrajectoryRecord/ScoringDataProvider，实际计算 NLL 和完整词表 entropy；实现旧 SignalProvider.compute 协议，M1 的 make_signal/attach_signals 继续只做包装与附加。
- D115：因果目标 p 使用 logits[p-1]，只对 assistant mask 或 skill span 的监督交集统计，重叠取并集，完整上下文保持。mean/sum、nats/bits 分开标识；缺标注/零证据为 missing，位置 0 无因果前驱则拒绝，非有限值不裁剪成假数值。
- D116：Torch 后端使用隔离 Store 的 eval/inference，禁用 ambient autocast，不调用 backward/optimizer；保留参数 dtype、float64 稳定全词表统计、RNG 与前后状态 hash。外部 logits 接口绑定 ModelRef/request/位置/dtype/词表与增量费用，源真实性由供应方负责。
- D117：SignalRecord 绑定完整模型/adapter/tokenizer 版本及输入/选中位置/计算身份；逐 token 内存缓存可以跨 scope 聚合，不能跨模型/输入/计算配置复用。已附同版本 raw 值可复用；不从当前批次临时生成 CDF 百分位。
- D118：前向使用记录只更新 forward_scoring，不改技能/质量/训练 history。新 M1 池按 ID 合并，保留原源 hash/排除记录并写 parent_pool_id/计算回执；消费时重建 MetricView/ExecutionContext，原快照回放保持独立。失败保留部分结果和实际已知费用，输出为新目录不覆写。

## 2026-11-06：NLL/entropy 再次复查

- D119：根评分请求 schema=2 保存原始输入及实际聚合的逐位置统计快照；重载验证 input record_hash、只允许信号/前向历史增量、重聚合数值及来源/状态一致。直接复用完整同版本信号对照原输入，不补造 token 数值。旧 schema=1 按原身份只读兼容，不伪造其缺失证据。
- D120：失败目录保存 partial-records.jsonl 及内容 hash，沿用 read_records 读取；异常仍携带完整部分记录与已知 ledger。纯前向模块拒绝后端明确报告的梯度/训练工作并保留费用，未知计数保持 None；外部源真实性仍由供应方负责。

## 2026-11-06：提交前目录整理

- D121：按用户要求删减冗余文件。11 份阶段复查及旧进度完整合并为 progress.md，更新引用并将 README/progress 聚焦当前接口与状态；研究原文、源码、回归和必要 fixture/配置保留。清理生成构建副本、缓存与测试临时目录，先归档核验；正式验收记录保持原路径。此次只整理文档和生成文件，不改变计算协议或模型版本，不代用户提交/推送 Git。

## 2026-11-07：用户确认先实现共享资产与更新规范

- D122：共享资产以完整输入身份/kind/backend/依赖 key 定位，逐文件 checksum、完成标记和原子发布；生产者源码/环境/配置与可选 commit 留证。不同 Git commit 不自动判同或不同计算，相关实现与输入身份才决定复用。来源输出不改写，已有 key 不覆盖冲突结果。
- D123：优先桥接 M0 开发测评、NLL/entropy 和 Lopt 生成；typed loader 继续验证原契约。只附加当前 M1 signals，不覆盖质量/技能/训练历史；原角色调用只导出当时 before-context 和单次费用，复用程序不回放 history 或复用旧选集。消费回执与原物理 CostLedger 分开，未知费用不补零。
- D124：运行输入锁保存 resolved config 与 AssetRef，读取须提供预期 lock/source/env 身份；实际凭据不得写入配置。该锁不代替真实来源核验、任务排程或外层恢复。数据准备/标注留待第一次实验执行，本轮不运行附件中的流程。
- D125：服务器更新入口与固定 commit 的作业目录/环境分离，输出资产放独立存储；运行中不更新该作业代码或共享环境。Git ignore 不保护已跟踪文件，pull 无冲突不等于数值运行不受影响。不替用户提交/推送，不在本机执行服务器 Git/GPU/API 流程。

## 2026-11-07：用户确认文档归档与逐实验交接

- D126：原递进实验方案、数据准备 spec 和最新 Coding 准备 prompt 原样归档，执行总览连接共享准备与 V0–V4，不改写用户研究原文。附件中的实施指令不因归档而自动执行；动态来源在实际运行前再核查和锁定。
- D127：每个逻辑实验增加版本化 handoff，引用公共模块/协议/资产，区分 draft/ready/retired；运行状态写本地输出。默认先串行单源码目录，结束并保存后再更新，worktree 作为并行/旧版本恢复选项，不要求每实验开分支。
- D128：handoff 负责执行交接，配置固定变量，控制代码落实状态/预算/恢复；不能以文档代替未实现功能。handoff 与代码一起提交时，启动后解析完整 commit 和 handoff hash 并冻结，避免在文档中自引用尚不存在的提交 SHA。刷新策略和首次真实执行配置/代码待下一阶段，不代填研究参数。

## 2026-11-07：首轮全量与后续按外层反馈定向刷新

- D129：用户明确第一轮全池初始化 NLL/entropy；第二轮起仅处理外部 Lopt 方向并集内、从未参与本参数分支外层真实任务子集更新的数据。方向推断留接口，None 等待、空 tuple 明确无工作，不擅自全量刷新。
- D130：candidate_sft 同时包含内层试训，不能据此排除。新增分支内 OuterExposureHistory，从固定 OuterHandoff/完成 TrainingOutcome 登记真实外层曝光；完成即计入，退回不撤销。控制语料不映射成原任务样本；同内容改 ID 仍计入使用。部分更新无法确认则阻止刷新，外层崩溃恢复仍须单独审计和实现。
- D131：复用 SignalService 数值/缓存协议，当前接收 ModelRef 精确查询；新增版本，不覆盖旧指标，不将范围外旧值冒充新值。完整计划、全池前后快照、嵌套评分证据与费用按新目录保存并严格重载。新内层显式重建 context/Team，不修改活跃内层模型或快照；方向生成、实验级排程与真实后端不在此阶段实现。

## EXP001：用户最新定义的三波静态/随机比较

- D132：本实验以 M0 和三波训练后的 M1/M2/M3 为四次观察，覆盖历史两次尝试安排；一次选 floor(实际作者匹配/准入/去重 N×10%)，三个不重叠波次单遍曝光。Random 可以独立且不用 Agent/指标；static_m0 全池初始化、一次 Lopt/Lexec 后固定，不运行 Leval、刷新、控制 SFT 或接收/退回。
- D133：一个 seed42、5e-6/2e-5 两档，四独立分支共用 M0，同模式共用精确成员和顺序；按样本数匹配并报告实际监督 token，不冒称 token 等预算。P1 选择粒度1，训练有效 batch8；尾窗口真实更新，不丢样本。新增连续 optimizer/RNG/权重 checkpoint 后端，不用旧的 fresh-per-stage 三次训练冒充连续训练。
- D134：先冻结全部清单与配方，开发测评只记录，所有训练封存后才执行所有节点的 test benchmark。test 保持独立类型/原 split，不伪装为 development；重叠最终套件只在逐题完整身份相同时复用，保存原请求/回执并分开费用，不按测试选权重。
- D135：三项 M0 公共资产和 prepared inputs 使用已有 AssetStore/输入锁，保留原生产证据，不重计消费费用。新控制器/配置/prompt/handoff 独立组织；四训练子进程与16最终任务槽位不等于 DDP。真实数据/Teacher、目标模型/任务部署和容量 smoke 尚待接入；正式 handoff 保持 draft，中途自动恢复不伪称已实现。
- D136：逻辑实验和物理 run_id 分离，重复运行的费用事件不能重名。连续阶段回执在各参数分支登记 OuterExposureHistory，真实更新在开发结果前计入；未知部分更新留 unresolved，不改公共候选池，供后续刷新/分支继承复用。
- D137：初始策略输入提供实际代码校验的算子参数模板、完整程序示例与分技能池统计，不依赖后续变体定义或 benchmark。大批量指标严格重载按费用事件 ID 建内存映射，保留原校验与数值/缓存身份，避免二次扫描和哈希；不新增持久化 operator 查询索引。
- D138：大批量 signals 复用时一次解析不可变资产引用，再按样本回填；不为每条记录重新解析整份 manifest。保持 StoredAsset.manifest 每次返回独立结构的语义，不引入可被调用方修改的全局缓存；原来源、校验与费用定义不变。

## 0.15.0 / 2026-11-08

- D146：采用历史精选原始5016行，覆盖旧作者ID/tool匹配。原始数量与准入数量独立，全部准入时选501条，三波各167。
- D147：共同初始PEFT文本LoRA，完整超参与目标校验；分块精确loss、驻留冻结指标。CPU数值对照不等于真实模型容量验收。
- D148：任务注册绑定原生环境/镜像与scorer；mini-SWE-agent生成，SWE-smith/SWE-bench官方判分，Harbor保留Terminal原生配置。语义动作不是梯度token mask。
- D149：不可变run完成后在旁目录生成校验ZIP，复制复用M0原始测评证据；排除权重/训练数据/optimizer。真实服务器环境、数据标注与容量仍待验收。


## 2026-11-08：服务器准备与 GPU 分离（0.16.0）

- 原始 5016 固定来源进入静态准入、真实 tokenizer payload mask、七技能证据标注和隔离注册；保留 quarantine 与未知值，Teacher 不独立证明质量。首次准备及标注可复用，API 预算耗尽/在途未知是基础设施阻塞，不删除样本。
- 执行只读取已提交来源与任务镜像；Dockerfile 构建运行依赖，可选来源镜像。无 GPU assets 阶段准备宿主 daemon 的任务镜像及 Harbor 仅安装快照，正式任务不再安装 agent，资源配额保持原任务定义。仍不得宣称所有原生测试脚本能断网运行。
- 全池指标按 token 平衡分到八卡，每卡常驻参考模型；GPU 上做全词表 NLL/entropy 归约；四个独立训练分支与其余卡验证分时协调。长上下文激活可 offload，必须先做完整真实容量验收，不能静默减预算。
- 服务仅管理自己启动的进程；跨进程任务槽和 Terminal GPU 独占池；容器通过仅推理、单模型临时凭据 relay 访问模型，不获得 LoRA 管理接口密钥。
- 服务器环境与来源锁、准备清单、逐题结果、真实 token/时间/利用率和可复用 M0 证据留存。已有训练恢复规则保留；选择之前尚未完成发布的未知调用须人工核对，不承诺任意位置无损自动重试。


## 2026-11-09：冗余文档与测试组织清理

删除退役 V1/V2 handoff、旧审计合集、过时就绪/执行总览及已由接口覆盖的实现前草稿；
原始字节和清单在忽略目录备份。保留用户研究原文、接口、V3、决策和运行证据。
角色两个 pass 测试归为 test_role_team.py；教程回归合入跨模块回归，全部函数与参数化不变。


## 2026-11-09：缩减历史专项测试

用户要求继续删减模块检测文件。移除无其他模块导入的历史专项审计套件，保留核心模块、
数值、数据版本、恢复和服务器集成。确认这是回归范围取舍，不把有效历史反例称为无效；
备份到忽略目录。运行时契约校验和业务任务验证不属于可直接删除的检测脚本。


## 2026-11-09：用户确认仅保留运行所需内容

删除开发测试、模块演示与专用 fixture，不删除未来实验业务模块或其一致性校验。
保留 EXP001 整体 mock 作为部署预演；不提供已删除的模块 demo 命令，不保留 pytest 的失效入口。
LoRA 以 server.template.json 的 adapter_settings.lora 为准。服务器使用 Docker/无 GPU assets 准备，
删除未被该运行流程引用的旧 venv 安装脚本。旧源码完整备份，原始研究文档不改写。


## 2026-11-10：独立的内外层 CPU mock 验收入口

用户要求将整体嵌套流程交给服务器 agent 前先在本机运行。新增 self_curate.smoke 与 configs/smoke/nested_cpu_mock.json，只支持 CPU mock，固定外层 3 轮、每轮 5 次内层候选评估；配置中评分、技能方向和决策均为显式模拟情景，不替代正式策略。复用现有 InnerSearch、OuterCycle、SignalService/FocusedRefreshService、训练/任务验证服务，不复制核心状态机。跨轮新建绑定当前模型的角色会话，旧私有 history 保持原身份，以角色私有摘要引用和外层决策证据输入新会话。旧 runtime-only 决策继续有效，不恢复源码测试树；此次用户明确要求的完整流程验收保留为可移植运行入口。支持内层中途保存恢复与完成证据复核，不宣称通用外层跨进程恢复。交接见 docs/handoffs/NESTED_CPU_MOCK_HANDOFF.md。

## 2026-11-10：正式 SFT 使用 Accelerate FSDP，恢复与日志

用户要求用成熟 FSDP 框架、实现中断续训、完善带时间编号的日志，并按 8×H200（约141GB/卡）安排训练。真实 EXP001 改为一个分支占用八卡 FSDP，四个分支依次执行；训练结束释放显存后再用八卡验证已发布节点。服务管理只关闭本项目拥有的进程。原“四张独立训练卡、四张验证卡”的默认安排不再用于该后端。

新增 training/fsdp.py、fsdp_worker.py 与 runlog.py。采用 Accelerate 1.15.0/PyTorch FSDP1 FULL_SHARD、PEFT 官方自动包装、BF16 LoRA、仅 assistant payload 的精确监督损失。参数分片、分布式反向、模型/优化器/RNG/混合精度状态交给框架，项目保留精确清单和阶段控制。FSDP 包装后才构造优化器，插件创建前绑定 LOCAL_RANK。仍有项目级训练循环，不能说成已使用 HF Trainer。保持有效batch=8、每卡micro=1、累积=1，科研学习率、seed和曝光次数不改。

框架状态默认每4步及阶段结束保存，校验后原子发布；恢复核对请求、数据、模型、源码、环境和GPU拓扑，未提交窗口允许重放并记录未知花费。保留最近两份完整恢复状态，清理旧分片前保留checksum证据；三阶段adapter单独保留。worker持有OS锁，控制器中断后不能对仍运行的worker重复发起训练。通用TrainingService增加显式resume；完成结果复用不新增训练调用。通用OuterCycle的进程级恢复依然是独立缺口。

新运行支持 --output-root 自动分配 YYYYMMDD-HHMMSS-hash-id；原 --output 保留用于精确恢复。控制器、torchrun、每rank事件日志分别落盘，记录时间、token、loss、学习率、吞吐、ETA、显存、GPU利用率、保存/恢复/失败。全局token字段不能跨rank重复求和，物理rank前向token另外记录；缺失GPU时间和未知中断花费不补0。结果包不包含新的job.json、权重或恢复分片。具体路径与服务器步骤见 handoffs/SFT_FSDP_RECOVERY_HANDOFF.md。

本机使用微型随机CPU模型实际执行Accelerate保存、强制中断和恢复，最终权重与连续训练逐张量一致；回归、嵌套mock、EXP001 mock/恢复/ZIP及编号检查的实际记录在忽略的runs目录。没有在本机加载目标模型或启动GPU/Docker。8卡真实FSDP、长上下文容量、真实adapter重载和吞吐必须在服务器重新验收，不能根据CPU结果声称已经通过。


## 2026-11-10：OuterCycle 进程恢复与原 CPU 联调扩展

新增 outer_recovery.py：完整恢复状态、前序hash链与原子journal，冻结输入/源码/后端/独立model和batch锚点核验，attempt OS lease及旧实例拒绝。恢复已完成训练与验证不重复执行，持久化接受/退回决定按同一上下文续交；M1 usage、外层实际训练曝光及费用同时恢复。失败后的显式retry最多两次，无法核验的外部在途调用不盲目重发。FSDP后端接收当前接受模型的checkpoint。仅支持新格式，不改写旧snapshot，不等同于全局服务模型提交。

原self_curate.smoke入口、配置和NESTED_CPU_MOCK_HANDOFF.md保持使用，加入训练结果、验证结果、接受决定三个位置的模拟中断，并由独立Python进程恢复。仍为3×5、5次训练、42次rollout，无GPU和外网。各阶段恢复、错误锚点、损坏日志、失败重试与旧实例检查记录在忽略的runs目录；真实后端仍需服务器验收。


## 2026-11-11：EXP001 Rebench 工作站准备与初始三角色
用户固定Qwen3.8-27B、成功SWE-rebench轨迹、Verified/Pro/Terminal；技能/标注规则与规模待定。新增准备计划、工作站入口、独立Dockerfile与交接，旧5016入口不替换或冒充新数据适配。新prompt opt-in一次初测Leval→Lopt，独立历史和共享反馈资产；旧mock不启用。轻量粗筛接受tokenizer回调，基础设施错误向上传播。工作站负责联网下载/锁与Teacher、选定题单后准备环境；集群只做计算。Pro当前V2/default642与Terminal4.1经官方页核对；数据commit与环境仍待工作站实查。准备层尚非完整Rebench→M1/Teacher新schema装配，Pro干净环境重判与Terminal4.1离线资产验收不可省略。未进行目标权重下载或真实GPU/Docker/Teacher调用。

## 2026-11-11：Rebench 实验一规模与工程接入

本节取代前一节“规模待定、M1桥接/Pro未完成”的状态。用户确认硬筛前成功轨迹总数 N 的10%（floor）作为每种选择的K，候选不足不缩预算；Random与静态策略都从同一硬筛/隔离池选择。开发题每技能50题，独立分层选组，多标签重叠题只执行一次。技能ontology与两类Teacher规则仍由用户确认，不默认旧七技能。

实现独立rebench_workstation六阶段入口（registry/model/tasks/pool/publish/config）、来源锁/Parquet按config和split导出、Teacher请求级缓存/本次预算、5%随机及争议复核、证据标注和M1发布。轨迹原tools完整保存并加入版本化system上下文；技能present/absent/uncertain分开，SkillSelect遵循unknown的缺失策略。原始池流式准入，逐条写M1；最终M1索引仍加载tokens，需工作站RAM验收。硬筛成功依据source resolved与静态完整性，fresh_execution明确not_performed。

Pro V2对接官方locked agent和干净patch replay最终判分；Rebench fork和Verified隔离Python环境。增加冻结题目环境准备、安装快照、离线检查/镜像导入导出/资产seal接口，及独立Dockerfile.rebench。Pro真实Harbor/断网容器未在开发机验收，不能称已正式跑通。

初测Leval→Lopt→Lexec共享M0且history隔离；反馈资产进入冻结清单，summary按实际新增role调用计数，inner_valuation_calls单独为0。旧两角色mock仍可独立运行。新增FSDP服务器专用中断恢复对照，CPU inspect始终不授予GPU通过；微型随机模型用真实Accelerate在CPU执行中断/恢复与逐张量对照。

部署先在工作站下载/安装/Teacher/镜像缓存；集群只导入和核验后计算，不在GPU作业内联网安装。8卡单分支FSDP的micro1、accum1、global8沿用；实际GPU容量/吞吐/重载由集群单独验收。当前正式新方案与交接见EXP001_REBENCH_PLAN_ZH.md及handoffs/EXP001_REBENCH_WORKSTATION_V1.md。


Teacher V1 本地交付验收（2026-11-14）：31 项临时 CPU 检查通过；330 条模拟任务标注选出 300 道互异题，跨准备目录复用为 0 次 Teacher 调用。真实 API 小试完成 3 条任务、2 条完整轨迹的 schema 和逐字证据校验；早期失败保留。最终源码下 3×5 内外层 mock、EXP001 Random/static mock、恢复与 ZIP CRC 通过。真实 Qwen tokenizer 尚未接入本次小试，不能声称真实 token 比例已验收；正式 100+100 试标和语义人工抽检待工作站执行。验收记录：runs/teacher-v1-verification-20261114/summary.json。


2026-11-14 EXP001文档收口：唯一总入口改为Rebench工作站handoff；旧SERVER_V3标为历史恢复，README同步。统一六技能、300互异题、100+100试标及Claude SDK；新增阶段输入/输出/阻塞/恢复/交付清单。仅文档变更，未改运行接口或配置身份，未执行GPU。

2026-11-15：按用户要求固定Rebench任务/轨迹SHA、Flash复核；先官方协议再文献，分离Pro/Terminal预算并将原生agent timeout传入SDK，拒绝复用旧预算注册。推理槽与worker可配置，目标4/卡并以真实并发容量报告门禁；loss分块256保持相同监督目标，global8与10%预算不变。来源与规模/协议证据见EXP001_EVALUATION_AND_SCALE_20261115_ZH.md；真实GPU与容器仍未验收。

2026-11-15最终CPU验收：18项专项、5项任务注册、31项Teacher与300题复用、EXP001 K4完整mock及恢复、3×5嵌套、两份ZIP CRC通过。记录runs/nov15-final-verification/summary.json；没有GPU/容器/新付费Teacher通过证据。另核对Qwen官方Pro256K+采样与当前128K贪心的协议差异，详见本次实验补充，不冒称复现。

2026-11-15用户纠正NICE最新SDK配置：以reevaluation.yaml为对齐源，统一250轮、262K上下文/总输出、单次32K、Verified7200秒/工具1200秒、temperature1/top_p.95/top_k20、thinking/preserve/xhigh。Pro保留官方50分钟、Terminal原生时限，其余沿Verified；移除10000轮/1M输出自定保护。增加逐请求vLLM Anthropic转换策略与引擎采样证据、独立前缀262K容量和新Pro sampling回执。训练131K/全局batch8不变；旧资产/运行不原地迁移。见EXP001_SDK_EVALUATION_PROTOCOL_ZH.md。

250轮SDK对齐最终CPU验收：17项参数/转换/预算检查、7项独立前缀及Pro错误回执拒绝、完整EXP001 K4四支三阶段、3×5嵌套恢复和两份ZIP CRC通过。证据runs/sdk250-verification/summary.json；不代表真实vLLM/262K GPU容量或官方判分已通过。

2026-11-15用户批准一题多轨迹：新增版本task_key、开发题关联资产与M1关系标签、按完整输入/输出预算分组（默认2条）、逐成员证据校验/复核/恢复、pilot与正式流程同规划器；m2-v3新增可选TaskLimitSelect和Lopt任务覆盖统计。NICE/梯度/指标保持原轨迹及模型/上下文身份，不传播为任务分数，Random未强制限额。本轮只CPU模拟，无新增付费API。见TASK_TRAJECTORY_LINKS_ZH.md。

任务关联交付CPU验收：31项既有Teacher契约、27项关联/分组/乱序ID匹配/部分恢复/预算拆组/引用隔离/TaskLimitSelect检查、300题资产零调用复用、EXP001四支三阶段K4及完成后恢复、3×5嵌套跨进程恢复和两份ZIP CRC通过。证据runs/task-links-verification-20261115/summary.json。本轮新增付费API为0；真实合批语义一致性和节费幅度仍待小规模API试标。


2026-11-15 Teacher 复核：新增 off/flagged/sampled/all，新模板默认 off；保留证据/身份/格式校验与逐条审计，兼容旧直接集成。真实单对样本复核两次均 approve，费用受缓存影响；本次未新增付费请求。详见 TEACHER_REVIEW_POLICY_ZH.md。


2026-11-15：官方元数据核验后固定 Pro HF 2d52cb3df914a3fcf80c7f66738b3a88ae37fc50、Pro tooling 66f92766bba642462d4bbe5479e83f91f9211862、Terminal4.1.0 4070d6cef56a7d20cb5dc98d5e10ada88a6b9440。开发集移除固定16worker瓶颈，随可用服务槽调度，8卡×4槽目标32并发；保留长上下文容量门槛和Terminal GPU预留。详见 EXP001_FROZEN_BENCHMARKS_ZH.md，未执行GPU/容器验收。


2026-11-15 EXP001测评调整：保留每分支三个开发测评，最终benchmark从initial＋12阶段权重改为initial＋四个stage3，按固定训练末节点，不按测评分数选模。旧结果不改写，新run执行。


2026-11-16：按用户要求Coding默认task-required标签，成功任务一次标注并关联轨迹；开发取无成功覆盖任务；不生成轨迹证据及占比，missing保留，旧模式兼容。General专门IF方案仅文档待实施。见CODING_TASK_LABELS_ONLY_ZH.md与GENERAL_IF_PLAN_ZH.md。未调用付费API。


2026-11-16：当前EXP001使用任务继承标签与M1v4（不发布技能证据片段/比例字段）；EvidenceSelect及evidence_batch删除，operator版本m2-v4。旧M1类型只作历史读取兼容，不能将旧程序当新算子版本恢复。详见CODING_TASK_LABELS_ONLY_ZH.md。


2026-11-18：SDK结果封存、清理解耦、网关静默读取取消的部分修复；6项CPU及3×5 mock通过。完整逐项状态和未完成项见 SDK_REPAIR_20261118_ZH.md，尚未正式GPU/容器验收。


2026-11-18追加修复：持久SWE判分重试、Pro分阶段、Terminal禁止自动重放、最终全目录seal、压缩时序验证/持久server探针、阶段成本、每槽容量失败证据。25项专项CPU、3×5 mock及EXP001四分支通过。Terminal跨进程verifier-only仍需冻结Harbor适配；真实GPU/SDK/Docker未验收。详见SDK_REPAIR_20261118_ZH.md。


2026-11-18实验安排：EXP002比较V1固定程序动态信号与V3动态程序（3外层×5内层，无控制轨迹SFT，NICE/SFT验证待定）；二者共用Lopt方向驱动、排除真实外层曝光数据的定向刷新。EXP003加入V4控制学习、用户提供的baseline、已训checkpoint补充测试、统一prompt/history的单agent消融。详见EXP002_EXP003_PLAN_ZH.md；本次仅文档，未启动或宣称实现新workflow。


2026-11-18文档组织纠正：实验二与三拆为EXP002_PLAN_ZH.md、EXP003_PLAN_ZH.md，删除上轮新建的合并计划EXP002_EXP003_PLAN_ZH.md。旧记录中的合并文件引用仅为历史；当前分别独立配置/workflow/handoff/run与验收，共享核验后的只读资产。
