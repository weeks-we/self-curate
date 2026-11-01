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

## 后续须冻结的研究语义

- 技能体系与人工标注/事件充分性标准（M2/M4）。
- 真实 tokenizer/chat template adapter、特殊 token/EOS 的监督策略（M3 前）。
- 数据源及来源真实性核验，任务家族与近似题面的隔离策略（真实数据导入前）。
- 模型、参数坐标、梯度 loss/reduction、投影与优化器变换（M3/M5）。
- 奖励、rollout、开发集、目标聚合与接受阈值（M4/M8）。
- token/batch、训练配方、重复曝光和被拒绝数据的正式复用规则（真实试点前）。
- 搜索与外层预算、最终测试隔离及确认性比较（对应试点/正式实验前）。

这些不阻碍当前 M1 的 CPU 数据实现；不能用示例值替代正式决策。
