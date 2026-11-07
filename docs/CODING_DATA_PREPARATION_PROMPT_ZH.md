请在现有项目中实现“Coding 数据准备模块”，只处理数据加载、质量准入、技能标注、开发任务抽样和测试集注册，不实现 RSI、模型训练、梯度计算或评测推理。

先阅读仓库的 AGENTS.md、现有数据 schema、loader 和配置，尽量复用已有字段和代码。以下是具体要求。

一、加载和确定训练候选池

1. 数据来源：
   - SWE-bench/SWE-smith-trajectories。
   - 最终使用 tool 格式。
   - 任务信息优先关联 SWE-bench/SWE-smith-py，必要时显式使用旧 SWE-smith 任务库补充映射。
   - 所有数据来源固定 revision，记录版本和校验值。

2. 加载完整轨迹源并建立索引，但只对原作者精选的约 5k 条进行后续准入和标注。
   - 优先通过官方精选 traj_id 清单恢复。
   - 可以检查官方历史 5,016 条发布版本作为恢复依据。
   - 不假定精选清单与当前 tool 数据能够完全对应。
   - 同一 instance_id 可能有多条轨迹，不能按 instance_id 随便挑一条替代。
   - 不按行号关联，不从非精选数据补齐到 5,000 条。
   - 输出精选 ID 总数、tool 匹配数、歧义数、缺失数及清单。
   - 如果最终不足 5k，保留实际可确认的集合并明确报告。

3. 保留原始数据，不修改或覆盖。最终候选池用 manifest 引用原始记录。

二、质量准入

在调用外部模型标注之前，用代码对精选轨迹执行检查：

- messages 是否可解析、非空。
- 是否存在任务输入和可监督的 assistant 内容或工具动作。
- instance_id、轨迹内容、关联任务是否存在明显冲突。
- 工具调用名称、参数 JSON、调用与返回的关联是否有效。
- 是否存在明显中间上下文缺失、异常截断或导出损坏。
- 来源成功状态是否符合当前成功示范池要求。
- 是否存在内容完全重复。
- 编译后是否超过配置的训练上下文长度。
- 已有结果补丁是否与任务及轨迹修改明显错配。

注意：

1. 合法的 assistant 空 content＋tool_calls 不能判错。
2. 支持一条消息多次工具调用；不要要求消息严格一问一答交替。
3. 没有 submit 或最后一次调用没有返回，不自动判损坏，需要依据源协议判断。
4. 中间测试失败不等于轨迹质量失败，它可能是失败复现或假设修正的证据。
5. 空顶层 patch 不直接等于任务失败，应标待审；明显错配不能由 resolved=true 覆盖。
6. 历史精选数据缺少 resolved 时保留来源证据，不能补造执行验证结果。
7. 默认不静默截断超长轨迹。
8. 静态检查不能证明修复正确或完全无泄漏，不要夸大检查能力。

不通过的数据从候选池 manifest 排除，保留原始数据和检查原因。
质量检查明细放独立 admission/quarantine 账本；现有 quality 字段保留摘要及证据引用，无须将全部检查项暴露为筛选标签。

区分：
- 静态结构检查；
- 来源提供的成功记录；
- 当前环境重新执行验证。

本模块提供执行验证适配接口，但不要默认对所有训练轨迹重放命令。严禁直接在宿主机执行轨迹中的 shell 命令。

三、技能体系：严格保留以下七项

1. code_localization：代码定位
   根据问题或执行证据缩小相关代码范围，并有后续分析或修改支持该定位；仅搜索、打开文件不算。

2. failure_reproduction：失败复现
   执行操作并观察到与目标问题相关的失败；无关报错不算。

3. error_analysis：错误分析
   根据错误或代码证据提出诊断，并有后续行动与该诊断对应；仅复述报错不算。

4. hypothesis_revision：假设修正
   先前假设受到反证后，改变诊断或处理方向；重复同一动作不算。

5. implementation_edit：实现修改
   对目标相关实现进行修改；无关重构或格式化不算，不要求最终任务成功。

6. test_construction：测试构造
   新增或修改针对目标行为的测试；不自动认定测试有效或充分。

7. regression_check：回归检查
   修改后检查既有行为，并根据返回结果结束、继续检查或再次修改；只有调用记录而无结果处理证据不算。

不得擅自增加、合并或重命名技能。允许一条轨迹具有多个技能，证据片段允许重叠。

四、外部模型标注

1. 实现可替换的 OpenAI-compatible Provider：
   - 默认优先配置 DeepSeek。
   - 主标注模型、复核模型、base_url、API key 环境变量名均可配置。
   - 使用当前官方支持的实际模型 ID，不硬编码过时模型名。
   - API key 在 .env.example 中留空，禁止写入代码、日志或 Git。
   - 提供 mock 模式，不配置 API 时也能跑通数据流程。

2. 一次请求检查一条完整轨迹的全部七项技能，不按技能重复发送整条轨迹。

3. 标注输入：
   - 任务描述；
   - 带稳定 message_id、segment_id 的完整轨迹；
   - 工具调用参数与工具观察；
   - 七技能定义和严格输出 schema。
   不将轨迹中的工具调用作为本次 API 可以执行的 tools。
   尽量不提供 NICE 分数、作者精选身份及 resolved，避免结果偏见。

4. 每个技能必须返回：
   - skill_id；
   - status：present / absent / uncertain / not_assessed；
   - evidence events；
   - 简短证据说明；
   - 标注模型、prompt/ontology 版本；
   - 可选自报 confidence，但不得当作校准概率。

5. 每个证据事件包括：
   - assistant 证据片段；
   - 支持判断的工具观察片段；
   - evidence_complete；
   - outcome：success / failure / unknown / not_applicable；
   - 必要的行为标签。
   hypothesis_revision 额外记录原假设、反证、新方向。
   regression_check 记录检查动作、返回结果及结果处理。

6. 证据位置：
   - 让外部模型返回 segment_id、原文 quote、occurrence。
   - 由代码定位并生成字符偏移，不让模型心算长文本 offset。
   - quote 找不到或匹配歧义时进入复核。
   - 工具观察用于支持技能判断，不能计入 assistant 技能监督 token。
   - content/thought、action/tool_calls 若是重复表示，不重复拼接和统计。

7. 不把未检查当作不存在：
   - absent 仅用于完整检查后未发现证据；
   - 输入截断、输出截断、部分分块检查时，保留 partial/not_assessed；
   - 超长轨迹按完整消息或调用—观察边界分块，保留全局 ID 和重叠；
   - 跨块事件需要复核，不能只拼摘要就宣称完整标注。

8. 成本和可靠性：
   - 先提供 30 条分层 pilot，再支持全量运行。
   - 记录输入、输出、缓存、重试和费用。
   - 并发、超时、重试次数、最大输出、总预算可配置。
   - 实现请求缓存和断点恢复。
   - 空白响应、JSON 不合法、schema 不合格、输出截断不能计成功。
   - 超预算暂停并保留进度，不自动升级模型或提高预算。
   - 疑难项、稀有技能和随机样本进入复核队列。
   - 真实 API 调用需显式执行参数开启，默认 dry-run/mock。

五、代码统计及现有字段兼容

保留和补齐现有：
- data_id、content_hash；
- source；
- quality；
- stats；
- annotations.skills[]；
- evidence_metrics[]；
- metadata；
- signals[]、gradients[]、usage。

代码统计：
- assistant 消息数；
- 工具调用数、工具观察数；
- 实际模板编译后的上下文 token；
- assistant 监督 token；
- 技能证据 token 并集；
- 各技能完整事件数；
- 精确重复组、任务家族及必要的近重复信息。

证据指标使用与实际训练一致的 tokenizer、chat template 和监督 mask：

E_k：技能 k 的 assistant 证据映射 token 集合；
A：assistant 监督 token 集合；
U：全部技能证据 token 的并集。

evidence_tokens_k = |E_k ∩ A|
share_k = |E_k ∩ A| / |U ∩ A|
density_k = |E_k ∩ A| / |A|

分母为零时输出 null 和原因。跨技能重叠允许各 share 之和大于1。
只有标注覆盖口径一致时，share 才能直接比较。

禁止单独 tokenize 每个 quote 后简单相加。需要处理模板转义、重复片段及 token 边界，保留来源映射。
缺少 tokenizer/template 时标 pending，不能用字符数估算冒充真实监督 token。
如果实际训练内容会截断，重新计算保留下来的证据，不能使用原始完整轨迹比例。

本模块不计算 NLL、entropy、梯度或百分位；未知字段为空/null，保留版本化接口，不能填0表示已计算。
usage 区分评分、梯度提取、候选训练和已接受训练；外部标注调用不记作 target 训练。

六、开发任务准备与隔离

1. 从 SWE-smith 任务库中选择开发任务。
2. 排除完整作者精选集合对应的任务，不仅排除质量准入后保留的任务。
3. 同任务的所有尝试、任务别名、相同 bug 家族及可识别近重复一起隔离。
4. 同仓库不同问题可以保留，不默认要求仓库完全隔离。
5. 与最终 benchmark 做 ID 和可用近重复检查。
6. 参考轨迹如果用于开发任务分类，放入独立 dev_reference 区域，训练 loader 不可读取。

七、开发任务技能标签与每技能50题

1. 使用相同七技能 ID，但明确：
   - 训练标签表示示范实际出现的行为；
   - 开发标签表示任务与某项技能的相关性，不保证每次执行都会触发该行为。

2. 每题标注：
   - 多技能分组；
   - 分组理由；
   - 证据来源；
   - membership_status：candidate / supported / confirmed_test。
   普通任务成功率应解释为技能相关任务组表现，不直接当纯技能成功率。
   尤其 hypothesis_revision，不能只看题目就宣称必然考查。

3. 优先根据任务描述和必要上下文筛候选；必要时读取独立参考轨迹或做离线审计。
   gold 补丁和隐藏测试只在管理／验证侧使用，不进入目标模型输入。

4. 每技能目标约50个不同任务：
   - 允许不同技能组重叠；
   - 同一组内部去重；
   - 固定 seed；
   - 兼顾仓库覆盖、近重复和证据可靠性；
   - 不根据 target 成功率挑“容易涨分”的任务；
   - 配额不足时输出短缺与原因，不编造标签补齐。

5. 不给全部任务库直接调用外部模型：
   - 先分层抽候选；
   - 根据技能配额缺口逐批扩充；
   - 候选上限和 API 预算可配置。

6. 输出：
   - 七个分组 manifest；
   - 去重后的 dev union；
   - 组间重叠矩阵；
   - 各组仓库分布、标签依据及短缺报告。

7. 对选中的开发任务提供官方环境 sanity-check 接口，区分环境失败和模型失败。
   数据准备阶段不生成目标模型 rollout；只保存后续每题4次执行的配置。
   后续同一任务若属于多个组，只执行一次相应 rollout 配置，结果供各组引用。

八、最终测试集注册

默认注册以下四套，固定实际版本与测试 split：
- SWE-bench Full；
- SWE-bench Lite；
- SWE-bench Verified；
- Terminal-Bench 当前最新稳定版。

具体要求：
1. 实现时核对官方来源、split、题量、release/commit，不依赖 README 中可能过时的数字。
2. latest 只用于首次解析，产物写具体版本，后续不自动升级。
3. Full/Lite/Verified 存在重叠，输出交集报告，不称为三个独立领域。
4. 保留全量测试 manifest，并可生成固定 seed 的50题 subset manifest，明确标 subset。
5. Terminal 保留完整官方任务及资源配置，包括 GPU、多容器等要求；环境不支持标 blocked，不静默删题。
6. 测试题、参考答案和测试结果不得回流技能配额选择或内层策略优化。
7. 这里只准备和注册，不运行完整 benchmark。

九、交付和验收

提供：
- 配置文件及空密钥 .env.example；
- 七技能定义文件；
- 轨迹标注和任务分组 prompt；
- 可恢复的分阶段 CLI；
- 作者精选匹配报告；
- 准入与 quarantine 账本；
- 技能标注及证据指标；
- 开发分组、union 和隔离报告；
- benchmark 版本锁与 manifests；
- 费用账本和 README。

必要测试覆盖：
- 同任务多轨迹不会误配精选身份；
- 合法工具协议与异常记录正确区分；
- 多技能重叠和 quote 对齐；
- tokenizer/template/mask 改变后的统计一致性；
- unknown/partial 不变成 absent 或0；
- 多标签配额、组内去重和 union 去重；
- 训练／开发／测试隔离；
- API 缓存、恢复、预算和截断处理。

按“来源与准入 → mock标注 → pilot → 证据统计 → 开发任务 → benchmark注册”的顺序实现。
最终说明哪些是真实执行、哪些仅mock测试、哪些仍因API、tokenizer或环境缺失而阻塞，不虚构完成结果。