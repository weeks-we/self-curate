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

## 后续须冻结的研究语义

- 技能体系与人工标注/事件充分性标准（M2/M4）。
- 真实 tokenizer/chat template adapter、特殊 token/EOS 的监督策略（M3 前）。
- 数据源及来源真实性核验，任务家族与近似题面的隔离策略（真实数据导入前）。
- 模型、参数坐标、梯度 loss/reduction、投影与优化器变换（M3/M5）。
- 奖励、rollout、开发集、目标聚合与接受阈值（M4/M8）。
- token/batch、训练配方、重复曝光和被拒绝数据的正式复用规则（真实试点前）。
- 搜索与外层预算、最终测试隔离及确认性比较（对应试点/正式实验前）。

这些不阻碍当前 M1 的 CPU 数据实现；不能用示例值替代正式决策。
