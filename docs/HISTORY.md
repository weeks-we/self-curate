# 历史复查与进度归档

本页合并原来的 11 份阶段复查报告和旧 progress.md，保留历史发现、修复及验收边界。其中“当前”“下一步”和“未实现”均指报告当时的状态，不作为当前开发入口。当前状态见 [progress.md](progress.md)，接口以各模块接口文档为准。

原报告文件名在各节保留，原始字节另有本地清理备份；正式验收产物保留在忽略的 runs/ 中。

## 索引

- [M1 三轮复核与框架对齐（0.1.2）](#m1-review)
- [M2 第二次复查与修正](#m2-review)
- [P0 引入后的适配复查与 P1 完整交接](#p0-p1-integration)
- [P0 / P1：按原始文档重新核对](#p0-p1-alignment)
- [全模块连贯性与接口复查（0.5.1）](#coherence-review)
- [两遍框架对齐与接口复查（0.6.1）](#roles-review)
- [按代码阅读教程复查：0.6.2](#tutorial-review)
- [全模块实现边界、逻辑与连通性审计：0.7.1](#full-module-audit)
- [两遍逻辑与接口复查：0.8.1](#outer-review)
- [Leval 文档与模块两轮复查（0.9.1）](#leval-review)
- [NLL / entropy 复查（0.10.1）](#signal-review)
- [清理前完整进度记录](#previous-progress)

<a id="m1-review"></a>

原文件：`M1_THREE_PASS_REVIEW_ZH.md`。

# M1 三轮复核与框架对齐（0.1.2）

依据：用户提供的 `data_centric_rsi_architecture.md` 第 4、5、6、8、12 节，
以及 `self_curate_codex_blueprint.md` 的 M1、数据契约、梯度缓存与分阶段实施约定。
参考文件保留原文；其中的演示数字不构成正式实验参数。

## 第一轮：源码与框架逐项对齐

发现并补齐的缺口：旧版仅有序列内的单条信号查找；缺少供 operator 使用的统一 ID 保序批次接口；
梯度没有专用容器；外部标签缺少强绑定封装；使用历史有结构但缺少安全更新入口。

按用户澄清，取消持久化查询索引设计，实际实现只使用进程内字典与不可变记录快照。

| 框架要求 | M1 当前落点 | 处理 |
|---|---|---|
| 一条数据与标签一一关联 | data_id、content_hash、source、AnnotationRecord/MetadataRecord | 跨 ID/内容/来源/token 版本拒绝挂接 |
| 长度与成本字段独立 | stats、ComputeCost | 从消息/mask 重算长度；梯度成本字段仅为外部声明 |
| 技能与证据 | annotations/evidence_metrics | 标签与注册表区分；交集/并集分母、missing 保持正确 |
| 任务、来源、重复信息 | DataLabels、SourceRef | 预留任务、仓库、重复组与其他键值标签；不计算语义去重 |
| NLL/entropy 版本 | SignalRecord/ModelRef/SignalSpec/PercentileReference | 模型、实际输入、mask、scope、单位、配置与 CDF 对齐 |
| 梯度版本 | GradientSpec/GradientFeature | 数据、模型、参数坐标、loss、归一化、精度、投影、变换/优化器、配置与实现 |
| 按需取值 | MetricView/MetricBatch | 按 ID 和版本查询；字段可单独或组合读取，不实现筛选算法 |
| 数据使用状态 | UsageHistory + 安全更新 helper | 四类经历分开；pending、reject、accept 不混淆；终态不反转 |
| 接受/拒绝后的缓存版本 | 明确 ModelRef/spec 的精确读取 | 新模型无对应结果即 missing；不退回旧模型 |
| 原证据保留 | m1-v1/v2/v3 兼容 | 旧数据只读校验，原 checksum 不改 |

M4 的目标任务梯度、M5 的真实反向与向量物理缓存验证、M6 的影响估值和 M2 的筛选算法仍由对应模块实现。

## 第二轮：边界与错误路径

完整测试覆盖：重复/未知 ID、乱序子集、单列/多列、缺失/0、不同 checkpoint、统计范围/单位、
不同 CDF、梯度参数坐标/投影 seed/精度/优化器/计算配置、标签来源版本、重标注后的过期信号，
以及训练 pending/accepted/rejected 转换和历史不可丢失。

额外核对：

- 长度标签必须与实际消息与 mask 统计一致，不能由调用方随意填数。
- 同 raw 计算身份的不同 CDF 视图不能含有矛盾数值。
- 梯度向量维度、dtype、norm 与 spec 一致；零向量与缺失分开。
- 相同缓存身份的冲突向量/已知成本明确报错，不覆盖旧证据。
- label registry 不会把每个样本自动标成“具备所有技能”。
- 更新记录会生成新对象，已经构建的 view 保持原快照。

## 第三轮：组合、旧证据与规模检查

使用独立的 ID→预期值对照表验证：两种模型、两个标量、部分缺失、部分梯度、不同使用状态；
随机打乱源记录和请求子集，逐列对比返回值，不能靠行号恰巧一致通过。

另以 1,000/5,000 条显式合成记录，重复读取 500 个乱序 ID 的 NLL/entropy，并进行单条随机读取。
建立 view 的成本与查询成本分别计时；检查每个返回值及缺失位置。
本机初测 500-ID 双指标批次中位数约 1 毫秒；最终实测以 `runs/m1-three-pass/benchmark.json` 为准。
这是 CPU 合成工程测试，不是大模型吞吐、任意数据规模的保证或研究结果。

旧版实际已发布的三个数据池分别只读载入，核对记录版本及所有文件 checksum 前后一致。
最终还检查 CLI、完整 pytest、格式/静态检查、wheel 构建/独立导入及交付 zip。

## 验收与限制

新仓库最终实现已通过 **96 项不同测试、0 失败、0 跳过**；验收结果记录于 `runs/m1-three-pass/verification.json`。
测试重复执行不算新增用例。

- 没有运行 GPU、真实模型、NLL/entropy 计算、梯度反向、自动标注或训练。
- 梯度文件为有 hash 的外部引用，尚不代表物理文件可用或真实梯度已算对；后端仍须验证。
- 实际标签/质量证据真实性依赖可信来源；M1 检查其结构与一致性。
- MetricView 是内存快照；数据更新后显式重建，且目前没有持久化查询数据库。
- 通过这三轮检查说明已覆盖上述行为与边界，不承诺所有未来数据源和计算后端都不会出错。

<a id="m2-review"></a>

原文件：`M2_REVIEW_ZH.md`。

# M2 第二次复查与修正

日期：2026-11-02。范围是 M2 的逻辑、异常输入、性能与 M1/M5/M6/M8 交接，不启动真实模型或训练。

## 实际发现并修正

| 问题 | 影响 | 修正 |
|---|---|---|
| BudgetPack 为每种删样本方案反复复制、扫描剩余清单 | 每次修整近似二次开销，候选较多时明显变慢 | 一次维护总 token 和分支 token；每种方案只计算受影响汇总，保留原有移除优先级 |
| 允许复用历史训练数据时，没有独立的最终批次复用报告 | 无法直接核对最终成员中被拒绝/已接受历史的比例 | 增加 usage_exposure，分别报告六类历史的 ID、样本数/比例和监督 token 数/比例 |
| 未指定 ModelRef 时，上下文可混合不同 tokenizer/template | Python 公共入口可能组成不可交给单个模型训练的 token 清单 | 构建上下文时检查全体 token 身份一致；verify_batch 另支持预期 ModelRef 校验 |
| 预算取整使用固定浮点补偿量 | 极窄容差下可能把不满足声明边界的 1 个 token 放过 | 按声明的十进制值使用有理数精确计算上下界和最终分支份额；token_target 限定在 JSON 安全整数范围 |
| 中间分支短缺未统一声明“不是全局无解证明” | 可能误读贪心分配失败的数学含义 | 所有算子不可行诊断统一标注 global_infeasibility_proven=false |

预算边界的可重现例子：target=100、tolerance=0.00999999999999999，其严格窗口应为 [100,100]；旧的补偿会放宽至 [99,101]。
复查后 token 与份额容差均不额外暗加宽限。三等分等无法用有限十进制精确表示的目标，需要显式合理容差；不能一面要求零容差，一面暗中接受近似比例。

精确内容去重、语义相似度查询也改用上下文已冻结的内容 hash，避免在内层比较中重复序列化轨迹。
公开信号仍通过 M1 的 ID、ModelRef、SignalSpec、CDF 视图读取，没有添加持久化索引。

## 独立验证

- 对无分支、两分支、三分支分别作 80 组可重现变量长度场景，使用逐项枚举移除方案的慢速参考算法核对最终成员；共 240 组。
- 数值步骤的显式 stub 测量：4003 条、有效 batch=8、需修整 3 条时，复查前约 2.46 秒，修正后约 0.008 秒，选择 hash 相同。该测量只包含打包步骤，不代表端到端或真实实验性能。
- 3468 组嵌套字段类型变异，覆盖程序、系统上下文、批次清单的解析；未出现未封装的 Python 异常。该检查验证异常路径，不等于所有非法输入均被穷尽。
- 新增技能范围 NLL/entropy 与整条轨迹分数隔离、截断换序的真实成员差异、seeded Mix 唯一归属与重放、复用比例、tokenizer/模型身份和预算边界的回归用例。
- 全量测试、CLI 重放、旧数据池只读兼容、源码检查、wheel 构建与独立执行在目标目录重新验收；具体统计见 `runs/m2-review/verification.json`。

## 版本与后续使用

包版本为 0.2.1，算子语义为 m2-v2，JSON schema 仍为 1。旧批次仍可只读解析、核对训练输入；旧 m2-v1 冻结上下文不能在新语义下静默执行。
原 0.2.0 的产物、测试记录、wheel 和交付包保持不变。当前合成配置已更新到 m2-v2。

M5/M6/M8 接清单时调用 `verify_batch(manifest, view, model_ref=current_model)`，继续使用原 training_order。
若模型改变，应在新上下文重新筛选/评估；补梯度等特征则不改变既定训练 token 身份。
M2 仍使用确定性贪心，不保证找出全部离散组合中的可行解；NLL/entropy、CDF、梯度、模型服务和训练计算仍属后续模块。

<a id="p0-p1-integration"></a>

原文件：`P0_P1_INTEGRATION_REVIEW_ZH.md`。

# P0 引入后的适配复查与 P1 完整交接

版本：0.4.0；P0 校验语义 p0-v2；M2 算子仍为 m2-v2。复查开始于 2026-11-02 23:35（北京时间），跨日交付。
本轮范围依据蓝图 P0/P1：从接口、实现、调用方到测试和发布证据；不进入 P2 状态机或真实模型实验。

## 发现与处理

| 发现 | 影响 | 修正 |
|---|---|---|
| 工作流仅检查包内模型/目标/上下文的自洽 | 修改目标及全部引用后，内部仍可能合法；消费方还需与系统固定配置核对 | validate_workflow/load_workflow 增加 expected_model、expected_objective、expected_selection_context_hash；回执分别说明是否进行了外部锚点核对 |
| M2 与 P0 各自拼接程序/执行 hash | 后续修改可能只更新一边，造成跨模块失配 | 共用 selection_identity，保持既有 hash 公式不变，旧 m2-v2 批次可精确复用 |
| CostLedger.reference 每次扫描所有事件建索引 | 多 rollout/成本引用时产生不必要的反复扫描 | 构造一次只读内存索引；不进入 JSON、record_hash 或解析字段，不新增持久化数据库 |
| 模型计算成本事件可省略 model_hash | 成本引用无法可靠归属 checkpoint | 除 selection 外的模型操作成本必须声明模型身份；原 fixture 已满足，历史证据保留 |
| real 域的 mock URI 检查只匹配小写前缀 | 大小写或空白可掩盖显式 mock 标记 | 对 fixture/mock 前缀统一规范化检查；仍不冒称核验了实际文件 |
| 能力 task_mean 每个任务重复扫描 rollout | 报告较大时增加校验开销 | 按 task_id 一次分组，聚合口径不变 |
| RoundRecord 的决策名与 M1 历史状态名不同 | 直接传 accept/reject 会与 accepted/rejected 不匹配，证据不足也不应变成拒绝 | 提供 training_use_decision 显式映射，needs_more_evidence 保持 pending；不自动更新曝光 |
| P1 只有裸选择输出，P0 完整样例需要手工构造 | 后续模块容易重建批次或误填未执行结果 | 新增 prepare-selection：从真实执行的算子结果生成只有精确批次和本次成本的未估值工作流；下游字段保持为空 |

这里的“外部锚点”是调用方提供的系统配置，不是输入包自带字段的另一份拷贝。P2/P5 的工具层应掌管这些值和路径，提案角色只能修改 ProgramSpec。

## P1 验收对照

| 蓝图要求 | 当前交付 |
|---|---|
| M1 规范化、证据、质量、使用历史 | 保留既有实现与回归检查 |
| 八算子 + 固定 QualityGate/BudgetPack | 使用原 M2 执行器，不重复建设 |
| 预算、重叠、缺失、顺序和 seed | 既有独立参考/回归测试继续执行，配合新交接场景 |
| 精确 BatchManifest | P1 输出与 P0 工作流中的 batch 为同一清单，训练顺序/哈希不重抽 |
| 不可行和缺分数不伪造批次 | 保存结果、请求、已完成 trace 和选择成本；无 workflow/batch 文件 |
| 相同输入与 seed 重放 | 原 select 保持确定性；prepare 的 batch/outcome 可精确重放，实际耗时与调用成本另记 |
| 后续按需梯度和估值 | P1 不调用 provider、不生成梯度/估值；验证后追加声明式 mock 下游记录仍使用同一批次 |

## 本轮检查的模块连接

1. **M1→M2**：ID 保序、缺失原位、内容/token/标签版本、固定 ModelRef/spec/CDF；无持久化 operator 索引。
2. **M2→P0**：共享执行身份公式；prepare 的可行批次直接进入 WorkflowBundle；已完成 trace 与成本都有引用。
3. **P1→M3**：MISSING_SIGNAL 请求保留 metric/scale、数据 IDs、模型、spec、CDF；P1 不调用前向。新模型的旧分数不能替代缺失新分数。
4. **P1→M5/M6**：样本特征按完整 cache_key 传递，目标按 target_version；追加特征不改变选定训练 token/mask。P0 继续核对坐标、计算覆盖和固定目标。
5. **P1→M8/P2**：既定 training_order 与 training_data_hash 保留；接受/拒绝和训练曝露仍由之后的实际执行事件驱动。
6. **P0→成本/恢复**：attempt_id 标识外层尝试，invocation_id 标识一次物理选择调用；恢复应读取已发布产物，不能重跑同一调用冒充恢复。

## 后续接口的明确使用约定

- WorkflowBundle 描述一个候选/选中批次的交接；搜索历史用多个不可变记录保存，不向一个 bundle 塞入所有历史模型和所有缓存特征。
- P1 选择结果的原始 context/snapshot 用于选择重放。M5 追加特征后，使用 load_workflow(view=...) 消费原清单；重新选择则显式创建新 ExecutionContext。
- 运行时应传系统的 expected_model/objective/context_hash。只读历史文件可仅作内部关系检查，回执会显示外部锚点尚未核对。
- M1 的 AnnotationProvider/SignalProvider/GradientProvider 与 make/attach/get 接口保持不变。raw scoring 与固定 CDF 的请求消费服务由 M3 接入；本轮不把它们标作已计算。
- 实际训练记账时，TrainingUse.model_id 使用起始 ModelRef.model_hash；receipt 应是稳定的训练尝试/任务身份，不能用会随状态变化的整个 TrainingRecord.record_hash 替代。RoundRecord.training_use_decision 将 accept/reject 映射为 accepted/rejected，pending/needs_more_evidence 映射为 pending。终态通过 M1 原有更新接口记录。P1 不写训练历史。
- 真实来源、任务家族与近似题面的隔离、向量内容、任务环境、模型权重、验收阈值和提交行为仍需后续模块核验。schema 的 split/版本检查不能代替这些实际检查。

## 交付与证据

新增 API/CLI 见 [p1_handoff_interfaces.md](p1_handoff_interfaces.md)。全量测试、解析变异、原数据只读、2000 条规模场景、独立 wheel 与压缩包检验见 `runs/p01-integration/verification.json`。
旧 M1/M2/P0 验收目录和交付包保留；不通过改写旧快照修复新协议。没有执行 Git 提交、推送或合并。

本轮可交付边界是 P0/P1 的本地确定性数据与交接链路；完整模拟搜索、接受提交与恢复属于 P2，尚未执行。

<a id="p0-p1-alignment"></a>

原文件：`P0_P1_REQUIREMENTS_ALIGNMENT_ZH.md`。

# P0 / P1：按原始文档重新核对

日期：2026-11-03。当前包 0.4.1，校验语义 p0-v3，M2 算子语义仍为 m2-v2。
本轮以原始技术蓝图第 5、6、7、16 节和 data_centric_rsi_architecture.md 为依据；两份原始文档保持不变。
范围是接口与本地确定性实现，不修改研究目标或把后续算法提前声称完成。

## 实际发现与修复

| 问题 | 可重现场景 | 修正 |
|---|---|---|
| 外部模型/目标/context 锚点不足以指认历史最佳清单 | 同时反转 training_order 并更新批次、估值、训练和 round 的包内引用，仍是内部自洽的另一份清单 | 新增 expected_batch_id，核对系统保存的完整 BatchManifest.batch_id；CLI --expected-batch 从外部清单取 ID；P1 交接自动核对实际执行结果 |
| 单独解析估值报告时，引用完整性弱于工作流检查 | valid 报告可声明全覆盖却没有 sample_cache_keys 或 target refs；仅凭 feasible 属性容易误用 | computed IDs 与 cache refs 数量一致；valid 必须有目标 refs，且与估值技能集合一致。跨记录真实性仍由 validate_workflow 检查 |
| 能力报告内部成员集合只在完整包中检查 | 独立 report 的指标可额外引用未声明任务/rollout，或漏掉报告级成员 | 指标 task_ids/rollout_ids 的并集必须分别等于报告级集合；跨技能重叠仍允许 |

这些是接口边界检查缺口；原 M1/M2 的八算子逻辑和预算算法未改动。
旧的合法 fixture 和 m2-v2 批次继续可读，不通过改写旧结果消除差异。

## 文档—实现—验证对照

| 文档要求 | 实现对应 | 验证与状态 |
|---|---|---|
| P0 核心 schema、配置、CLI、AGENTS、待决策项 | contracts、ImportConfig、ExecutionContext、CLI 与 docs/decisions | 已实现，保留显式待决策项 |
| ModelRef 与 artifact 可序列化、内容身份完整 | ModelRef、ArtifactRef、严格解析与标准 JSON | 既有往返/缺字段/未知字段检查继续运行 |
| P0 程序、批次、梯度、报告和 round 接口一致 | WorkflowBundle / validate_workflow，含 12 个核心类型与辅助契约 | 逐层检查 ID、模型、token/mask、cache_key、target_version、成本与状态 |
| 精确历史最佳清单交给训练，不重新抽样 | BatchManifest.training_order/training_data_hash；新增独立 expected_batch_id 锚点 | 对完整重封装的训练顺序替换作回归验证 |
| M1 保留上下文，监督 assistant，证据统计不造零 | normalize/tokenization/evidence、外部标注封装与 MetricView | 手工 token/并集参考、缺失与零值区分、旧池读取检查 |
| P1 八算子与固定 QualityGate/BudgetPack | operators 与严格 ProgramSpec，系统 context 不在可编辑配方中 | 全八算子、图校验、门禁与预算检查 |
| 根据当前 checkpoint 的版本指标筛选 | MetricView 按 ID、ModelRef、spec、CDF 精确取数 | 切模型缺失请求、跨 scope/CDF 错配和不替用旧分数 |
| 多技能样本唯一入选，按分支监督 token 记账 | MixAllocate 唯一归属；最终实际份额、残差与语义曝光报告 | 重叠、seeded 归属、离散预算与独立数值参考 |
| 固定过滤可交换，截断可能产生顺序差异 | 保序过滤与显式排序/截断/抽样 | 分别验证交集交换与真实截断差异 |
| 缺失和不可行明确报告，不自动放宽 | 显式 missing_policy、结构化 request、infeasible 结果 | 无假 batch/workflow，无隐式模型调用，保留 trace/成本 |
| 同一程序与固定输入得同一精确清单 | select / prepare / inspect-preparation | 选择结果与批次可重放；实测时间另记，不宣称计时文件字节一致 |
| 拒绝仍保留训练和成本历史 | M1 使用历史、CostLedger、RoundRecord 与状态名映射 | 成本不可篡改、终态不可逆、needs_more_evidence 不提交 |
| 冻结内层与先筛选再补梯度 | P1 只读取已存指标，输出未估值工作流；下游字段为空 | 不调用模型、梯度或训练 provider；声明式追加记录保持原批次 |

## 必须正确理解的验证层级

1. parse_evaluation_report / parse_capability_report 检查本记录内部结构，不会加载外部目标、任务、向量或奖励环境。
2. EvaluationReport.feasible 是声明状态与约束结果的派生属性；搜索比较前仍必须执行完整 validate_workflow，并核验它依赖的后端产物。
3. validate_workflow 的系统 expected_* 值由调用方掌管。不能从同一个待验包中取值再宣称已核对外部固定规则。
4. expected_batch_id 应来自搜索保存的历史最佳清单。它同时绑定成员、原内容/token 身份、分支、seed 和训练顺序。追加特征时保留原 BatchManifest，不重建其 record_hash。
5. 传入 view 才会核对实际训练 token/mask。artifact_contents_verified 仍为 false，schema 合法不等于向量、环境或 checkpoint 已经验真。

## 仍属于后续阶段

- P2：真实运行的 mock 搜索/训练/接受拒绝事件、历史最佳选择、预算扣减、原子状态与 resume。
- P3/P4：模型 tokenizer 验收、NLL/entropy/CDF 与样本/目标梯度的实际计算、代理方向及真实收益校准。
- P5/P6：共享权重角色与工具权限、真实训练、接受阈值执行和权重提交。
- 真实数据的来源权限、家族与近似题面隔离、模型/向量文件核验仍需相应后端和预检。

以上不是本地 P0/P1 fixture 验收缺陷，不能用当前 CPU/mock 测试替代它们。
本轮测试与交付证据位于忽略的 `runs/p01-document-alignment/`，旧验收产物保持不变。

<a id="coherence-review"></a>

原文件：`COHERENCE_REVIEW_ZH.md`。

# 全模块连贯性与接口复查（0.5.1）

日期：2026-11-03。范围是当前已实现的 M1、M2/P1、P0、外部验证交接、Leval、Lopt、CLI、文件产物与安装包。

原来的单步示例能运行，但不能据此认定全部接口已衔接正确。本轮新增反例复现并修复下列问题；最终测试数量与运行结果以 `runs/coherence-review-final/verification.json` 为准。`runs/coherence-review/` 保留首轮验收记录。

## 发现与修复

| 问题 | 原行为与影响 | 修复 |
|---|---|---|
| 下一轮文件交接不完整 | 候选 workflow 不能直接作为下一轮的 --prepared 输入 | 发布标准 candidate_prepared/，复用已执行的 P1 结果，不重筛、不重记成本 |
| 旧报告转接遗漏覆盖核对 | batch_id 正确但 coverage.required_ids 来自其他数据时仍可转接 | 核对覆盖集合等于请求的精确 batch ID 集合 |
| 规则候选拒绝后记录丢失 | 编辑策略更严格时直接抛错，没有已发布调用证据 | 返回 blocked，保存原候选回复、错误和实际 CPU 成本 |
| 规则调用 hash 不一致 | request_hash 与保存的 request 内容不一致 | 保存完整规则输入，统一绑定请求/回复的内容摘要 |
| 节点改名绕过 seed 锁 | 仅检查同 ID 节点，改名后可换 seed | 比较整个 seeded operator 命名空间，新增/删除/改名也需显式放开 |
| 客户端调用中切换模型 | 首次调用前检查通过后，后续状态变化仍可能接受提案 | 每次调用前后检查；模型改变则留存失败并停止后续调用 |
| 不同输出共用提案 ID | 相同输入的不同回复可能具有相同 proposal_id | ID 同时绑定完整候选、解释、状态与角色命名空间 |
| 真实域检查遗漏 | 请求的目标 artifact、P1 固定 CDF/语义上下文未统一拒绝 mock 引用 | 在已有边界补齐 fixture/mock URI 检查；不冒称来源真实性已获证明 |
| M1 部分审计标签未重算 | quality_counts、缺失标注统计、backend 等矛盾时可能通过读取 | 核对存储配置、tokenizer 和所有现有审计字段；保持旧错误类型和历史池兼容 |

同时补齐 FeedbackStep 的类型与聚合检查：反馈、父程序、实际 diff、候选、成本、独立角色命名空间和下一次请求必须一致。服务传给 Lopt 的上下文现在包含冻结的预算、质量门禁、信号与 CDF 配置。

通过 `ExecutionContext.role_summary()` 提供角色输入：控制配置完整保留，语义相似度只传 artifact 身份与 pair_count，不序列化完整 pairs 表；context_hash 仍绑定执行器使用的完整冻结上下文。原 `to_dict()` 和选择身份计算不变。

## 逐段适配核对

| 链路 | 已检查的不变量 | 当前边界 |
|---|---|---|
| 原轨迹 → M1 | assistant-only mask、工具调用、证据并集、缺失值、静态标签、使用历史 | 外部标注与分词来源真实性由上游负责 |
| M1 → MetricView | ID 保序；模型/spec/CDF/梯度按版本取值，缺失保持原位置 | 只使用内存索引，不新增持久化 operator 查询索引 |
| MetricView → M2 | 八算子注册、严格 DAG、固定门禁、整条轨迹与监督 token 配额 | 缺信号显式请求，不隐式运行模型 |
| M2 → P1/P0 | 同一 BatchManifest、训练顺序、模型、目标与上下文，空下游字段 | 特征追加后消费原 workflow；选择回放仍需原始快照 |
| P1 → 验证请求 | 完整程序、实际批次、measure、protocol 和精确请求摘要 | ValidationProvider 只有协议，没有验证数值内核 |
| 外部结果 → Leval | 请求/backend 匹配、固定技能覆盖、加权和与约束一致 | 区分观测与假设，不声称一致性检查等于模型验证 |
| Leval → Lopt | 反馈不被改写、共享 ModelRef、角色命名空间隔离、真实程序差异 | 规则可直接运行；真实模型客户端需后续接入 |
| Lopt → P1 → 下一轮 | 修改预算、随机配置锁、再次组批、标准候选交接 | 只返回待验证候选，保留 blocked/失败/不可行状态 |
| CLI/文件/打包 | 拒绝覆盖、原子目录发布、递归 checksum、错误码、独立 wheel 导入 | CPU synthetic 验收；没有真实 GPU、训练或容器实验 |

## 可复用的 P1 接口

```python
from self_curate.operators import publish_preparation, load_preparation

# prepared 已由 prepare_selection 执行完成；发布不运行算子，不新增事件。
publish_preparation(prepared, "runs/example/prepared")
restored = load_preparation(
    "runs/example/prepared", original_view,
    expected_model=model, expected_objective=objective,
    expected_batch_id=known_batch_id,
)
```

load_preparation 校验文件、原始快照、CPU 重放与独立锚点，返回 PreparedSelection，保留已有 ledger。verify_preparation 仍只返回验证回执；prepare_to_directory 仍只执行一次选择。

## 连续步骤

第一轮用法见 [Leval/Lopt 接口](feedback_lopt_interfaces.md)。获得下一次请求对应的外部结果后直接执行：

```powershell
python -m self_curate feedback-step --prepared runs/feedback-demo/step/candidate_prepared --pool runs/feedback-demo/pool --request runs/feedback-demo/step/next_validation_request.json --result path/to/next_external_result.json --policy runs/feedback-demo/step/edit_policy.json --attempt-id demo --invocation-id feedback-002 --output runs/feedback-demo/step2
```

下一份结果的 request_hash 必须指向 next request。上一份结果不能换路径重复使用。合成验收中的 mock 数值不能用于研究结论。

候选 handoff 的 ledger 只包含原选择事件，父步骤 ledger 另含 Leval/Lopt。后续 P2 必须沿历史收集事件并按 event_id 去重，不能只读一个候选 workflow 就认定涵盖所有搜索成本。

## 保持不变与剩余边界

- 原始蓝图、架构文档、旧数据池与验收证据不改写；M1 m1-v3、M2 m2-v2、P0 p0-v3 保留，包版本更新为 0.5.1。
- 不增加梯度、NLL、entropy 或 reward 计算算法，不初始化真实模型。
- 当前可由人工或外部调度器逐步迭代；P2 的自动 best/current 比较、预算扣减、恢复状态机、最终训练接受仍未实现。
- 测试通过证明已覆盖的本地接口与控制流符合预期，不等于真实模型效果已改善，也不能保证所有潜在问题都已消除。

<a id="roles-review"></a>

原文件：`ROLES_TWO_PASS_REVIEW_ZH.md`。

# 两遍框架对齐与接口复查（0.6.1）

2026-11-03。原始蓝图和数据架构文档保持原样。本轮不是重复执行同一套测试：第一遍检查框架不变量与职责，第二遍由保存恢复、下游消费和失败分支验证接口。最终统计以 `runs/roles-two-pass/verification.json` 为准。

## 第一遍：定义与代码对齐

最初的 8 个新增反例全部失败，确认以下缺口，已修复：

| 原始要求 | 0.6.0 的问题 | 本轮处理 |
|---|---|---|
| 内层模型和评价口径固定 | 会话只绑定模型、prompt 和 attempt；同一 session 可以换 objective、context、measure、protocol | 新增不可变 RoleScope，固定选择上下文、目标、口径、协议内容 hash 与编辑策略 hash；冲突在新模型调用前拒绝 |
| 三角色始终使用同一运行器 | RoleAgent 自身 runtime 只读，但 RoleTeam 的公开属性仍可被替换成另一个对象 | Team 的 runtime 和三个角色绑定改为只读；新 checkpoint 使用新 Team |
| 完整工作流必须保持身份一致 | RoleCycle 对 proposal backend、原始 source context、下一次完整请求等检查不足 | 保留原请求、外部结果、原上下文和编辑策略；校验反馈、父子谱系、实际 diff、候选与下一次请求 |
| 独立外部锚点不能只靠产物内部自洽 | restore 可核对模型/prompt，但不能直接指定期望 session/attempt/backend | 新增 expected_session_id、expected_attempt_id、expected_backend、expected_scope |
| 下游拿到完整实际子集 | save 只保存角色历史，RoleCycle.to_dict 原来只是一份摘要 | 新增完整 cycle 发布/读取接口，附标准 candidate_prepared，原 save 继续只保存角色历史 |

执行角色保持薄工具控制层，未变成训练器。验证角色仍是“确定性核对外部数值 + 模型解释”，模型不能覆盖数值。优化角色仍只生成受校验的 ProgramSpec，没有 evaluator、质量门禁或训练提交权限。

`feedback_step` 从上一份外部验证报告继续，所以顺序为 Leval → Lopt → Lexec → next request；这表示循环中间的一步，并不改变蓝图完整循环的 Lopt → 执行 → 验证顺序。

## 第二遍：消费者、异常与恢复

第二遍验证下列链路，并补齐失败反馈入口：

| 消费者或分支 | 验证内容 |
|---|---|
| P1 消费者 | 发布/读取完整候选；发布不重新筛选，不新增一次选择调用 |
| P0 成本与工作流 | 每个 cycle 只引用自己的物理模型调用和自己的 P1 选择事件；两个 cycle 的事件 ID 可直接去重合并，attempt/model/backend 一致 |
| 后续梯度模块 | 使用选定 BatchManifest 的精确 ID/order，追加梯度后通过原 workflow 消费；不拿新特征快照冒充旧选择回放 |
| 后续验证模块 | next request 绑定实际候选；通过 ValidationProvider 或外部服务回传该 request_hash 的结果 |
| blocked / pending / invalid | 不伪造候选、总分或新验证请求；调用和失败证据保留 |
| candidate_failed / candidate_infeasible | 保存 P1 错误与实际成本，可显式把执行摘要交给下一次 Lopt，不能编一份新数值报告代替失败 |
| 角色历史恢复 | 旧 0.6.0 的业务记录可推断已有 scope；旧文件不改写，已混用条件的历史拒绝继续 |
| 文件与逻辑证据 | 校验递归 checksum、源请求、候选 batch、固定编辑策略、实际 diff、逻辑尝试与物理调用数，拒绝删除尝试或替换来源 |

一整个 Team.feedback_step 在同一运行器锁下执行，保证当前 cycle 收集的 model_call_ids 不会混入另一并发 cycle 的调用。

## 新接口与实际用法

```python
from self_curate.agents import RoleTeam, load_role_cycle, request_scope

cycle = team.feedback_step(
    request, external_result, original_context, edit_policy,
    attempt_id="outer-001", invocation_id="child-001",
)
team.publish_cycle(cycle, "runs/cycle-001")  # 完整候选与工作流产物，不重跑
team.save("runs/history-001")               # 独立的 prompt/history 快照

loaded = load_role_cycle(
    "runs/cycle-001", original_context.view,
    expected_request=request,               # 独立系统锚点，必传
    expected_policy=edit_policy,
    expected_candidate_batch_id=known_best_batch_id,  # 若消费历史最佳，独立传入
)

restored = RoleTeam.restore(
    "runs/history-001", shared_client,
    expected_model=current_model,
    expected_prompts=prompts,
    expected_session_id="outer-001-inner-001",
    expected_attempt_id="outer-001",
    expected_backend="mock",
    expected_scope=request_scope(request, edit_policy),
)
```

Cycle 目录保存原请求、原外部结果、上下文、编辑策略、Leval 产物、完整程序提案、尝试记录、本 cycle 的模型账本、合并账本、候选 P1 目录及可行时的 next request。role history 与完整 prompt 不自动传给其他角色；完整模型调用证据仍由对应的 history 快照保留。

如果候选执行失败，后续调度器可以从原已验证父请求出发修订程序：

```python
repair = team.feedback_step(
    loaded.source_request, loaded.source_result, original_context, edit_policy,
    attempt_id="outer-001", invocation_id="repair-002",
    previous_cycle=loaded,
)
```

该入口只显式传递前一次程序和执行摘要，不传整个 Lexec history，也不把失败候选标成“已完成新验证”。如果前一候选已收到新验证，则输入其 next request 和实际新结果。自动选父程序、best/current 比较与停止条件仍由未来 P2 负责。

## 旧快照与范围说明

新快照保存 RoleScope。恢复旧 0.6.0 快照时，只从确定性业务记录推断已有锚点，不从模型文字猜测；缺少的字段不会被声称已核验，可由后续系统输入补齐。旧记录如果包含相互冲突的条件，会拒绝继续使用，应建立新会话。

当前 RoleScope 固定的是一个原始选择快照。追加梯度后消费已有 batch 不需要新选样；若补齐 NLL/entropy 等特征后要重新选择，应按已有 P1 约定显式建立新 context/session，保留同一 ModelRef 和旧记录。自动特征回填后继续原搜索、跨快照条件化记忆检索仍由后续调度/缓存模块设计，不把旧 context_hash 偷换成新快照。

`load_role_cycle` 要求原始 MetricView，以验证选择回放；追加梯度后，使用 candidate_prepared/workflow.json 加 `load_workflow(..., view=enriched_view, expected_batch_id=...)` 消费精确批次。

成本账本覆盖本次物理模型调用与本次选择，不因导入旧 external_result 重复收费；外部验证的 usage/证据保留在源结果中，真实验证器执行时的记账仍由验证后端/调度器负责。

目前仍未实现真实共享权重加载/服务端核验、数值估值与梯度内核、实时事件刷盘、自动 best/current 搜索、训练和接受状态机。这些是已声明的后续模块，不是本轮测试已经完成的内容。最终测试隔离仍需数据/验证后端保证，通用 generate(work) 不提供任意 JSON 的自动泄漏识别。

<a id="tutorial-review"></a>

原文件：`TUTORIAL_FLOW_REVIEW_ZH.md`。

# 按代码阅读教程复查：0.6.2

日期：2026-11-04。范围是独立 Self-Curate 项目当前已实现的数据流与接口。原始架构文档和历史运行证据保持不变；本轮不扩展真实验证、梯度、训练或 P2 搜索功能。

## 发现与修复

### 1. 固定上下文中的数据范围可以漂移

原 `MetricView.data_ids` 虽然是 tuple，公开属性仍可重新赋值。在一个已创建的 ExecutionContext 上，把它从两条数据改成其中一条，再执行同一个 ranked 筛选程序，实际成员会从 `data-00` 变为 `data-01`，但 snapshot_hash、context_hash、execution_hash 都不变。

这会破坏“同一冻结上下文对应同一候选数据池”的承诺。修复为只读属性；改变候选范围必须显式创建新 MetricView 和 ExecutionContext。查询顺序、缺失值语义、ID/版本索引与持久化格式不变。回归测试同时检查赋值被拒绝，以及重复执行结果保持一致。

### 2. Windows 嵌套发布的临时路径过长

原发布层逐层使用完整 output 名加 32 字符 UUID 作为 staging 名。即使最终候选文件路径低于 Windows 路径限制，cycle/feedback 内再发布 P1 时，临时路径仍可能先超限，报 FileNotFoundError。

本轮在较深的隔离检查目录跑原有全量测试，得到 386 通过、2 个路径相关失败；另增加一例最终文件路径小于 250 字符的 cycle 发布测试，修改前也能独立复现失败。

数据池、选择、P1、feedback、cycle、history 与 demo 现统一使用短名称同级 staging，通过独占 mkdir 创建并重试名称碰撞，沿用原发布锁和原子 rename，拒绝覆盖既有输出。失败数据池仍留下 failure.json。采用 mkdir 以保留托管 Windows 工作区的 ACL 继承。此修复减少临时目录造成的路径膨胀，不承诺任意超长路径都可用。

## 按教程核对的链路

| 阅读阶段 | 核对内容 | 结论/边界 |
|---|---|---|
| M1 一条轨迹 | 内容/token/mask、外部标注封装、信号/梯度版本、使用历史 | 既有 ID 保序、缺失值和跨版本拒绝继续有效；修复公开成员列表重绑定 |
| M2 一次筛选 | 固定门禁、DAG 执行、八算子、trace、完整轨迹和确定性组批 | 程序与实际批次分离；贪心 infeasible 不冒充全局无解 |
| P0/P1 | 实际批次交接、原始视图重放、追加特征后的精确清单消费、成本 | 保留 expected model/objective/context/batch 锚点；未执行下游字段仍为空 |
| Leval/Lopt/Lexec | 外部结果校验、反馈保真、完整子程序与实际 diff、候选重跑 | 可行候选只产生下一次验证请求，不捏造收益或训练 |
| 共享与隔离 | 同一客户端/模型、各自 prompt/history、固定 scope、显式跨角色产物 | 仍由确定性代码校验角色输出；真实服务加载身份须由后端核验 |
| 发布与恢复 | history save 与完整 cycle 发布、checksum、成本引用、只读加载 | 修复嵌套临时路径；历史格式保持兼容，不把快照等同完整 P2 恢复 |

## 验证与证据

- 新增 `tests/test_tutorial_review.py`，两个案例在修改前均失败，修改后通过；修改前复现日志保留在验收目录。
- 针对性回归 48 项通过；修复实现的隔离全量测试 390 项通过。最终交付副本的全量结果和按教程运行的新 mock 验收以 `runs/tutorial-flow-review/verification.json` 为准。
- 新 mock 验收检查两次反馈、角色历史保存/恢复、共享客户端与隔离记忆、精确成员及训练顺序、完整 cycle 的发布/加载和文件 checksum。
- 历史数据池和角色快照只读加载；旧证据校验、源码交付包及本轮检查记录放在忽略的 `runs/tutorial-flow-review/`。

未发现其他阻塞当前已实现主链路的问题。此结论是源码复查、CPU 测试和显式 mock 的结果，不替代真实模型、数值验证器、梯度向量、训练后端或研究结论验收。

<a id="full-module-audit"></a>

原文件：`FULL_MODULE_AUDIT_ZH.md`。

# 全模块实现边界、逻辑与连通性审计：0.7.1

日期：2026-11-05。依据原始蓝图、数据架构、Motivation 讨论稿的基本过程、当前代码与接口说明。本轮只修复已实现模块，不新增真实计算、训练、研究停止阈值或外层闭环。

## 结论范围

当前可以运行的是合成/外部结果驱动的内层调度：数据 → 算子 → P1 精确批次 → 外部验证接口 → Leval/Lopt/Lexec → 下一次验证 → 历史最佳与交接。共享模型服务本身、真实评分与梯度、任务执行、训练和外层晋级不在已实现范围。原有 420 项通过仍不能覆盖全部异常入口；本轮额外实际复现了五类问题。

## 发现并修复的五类问题

| 问题 | 修改前的实际表现 | 修复 |
|---|---|---|
| 内存数据入口语义检查不足 | 直接构造/replace TrajectoryRecord 后，可把 density 改成 0.99、篡改 share/events、重复技能行，或把 user token 放进监督 mask，再交给 MetricView；文件入口会重算，内存入口不会 | 视图构造时核验工具关系、监督位置、技能证据派生值/总量及唯一性；查询时仍使用索引 |
| 派发前保存失败误标在途 | 写入预操作检查点失败、验证器一次都未调用，搜索却变成 interrupted，正常重试受阻 | 只回滚未派发的操作号与 pending 标记；保留事后保存失败时已经完成的结果和费用 |
| 异常回复丢失已知成本 | 回复包含不可 JSON 序列化值时，Runtime 与独立 Lopt 在解析 usage 前先序列化整包，导致有效 token/GPU 计量丢失 | 先验证并保留 usage，再校验回复；坏内容继续失败。错误说明不再把后端/协议失败统一归为 schema 问题 |
| 已知 mock 数据进入 real 交接 | 把 tokenizer 声明换成 precomputed，并替换系统侧 artifact 后，轨迹里仍带 fixture/mock 来源和质量证据却能通过 real P1/P0 | 检查实际轨迹的来源、质量、标签、信号/梯度及使用历史 ArtifactRef；已知 mock 引用进入 real 路径时明确拒绝 |
| 独立优化接口混用历史编辑条件 | 直接调用 OptimizationAgent.optimize 时，previous_cycle 可来自另一套 EditPolicy；模型会拿到缺少原政策条件的旧执行反馈 | 历史 cycle 的 source_policy 必须等于本次固定 policy。内层调度原本已有固定政策，该缺口影响独立角色入口 |

新增 13 项回归；其中 11 项在修复前实际执行并失败，另 2 项补充证据总量与“已完成工作仅保存失败”的边界。三个旧的 tokenizer/技能缺失测试原先直接删 annotations 却留下旧证据指标，现改用正式 attach_annotations 清除标注，使输入保持一致；原 tokenizer 错误码、缺失政策和技能曝光断言全部保留。

## 单模块实现矩阵

| 模块 | 已实现并核对的范围 | 尚不能宣称完成的部分 |
|---|---|---|
| 核心契约/P0 | 严格解析、版本身份、模型/批次/目标/任务/报告/成本的关联与独立预期锚点 | 实际向量计算、外部环境执行、权重加载、训练或接受规则执行 |
| M1 数据 | 文本/工具轨迹、assistant mask、证据统计、外部标签/信号/梯度引用挂接、按 ID/版本取值、使用状态转换 | 自动标注、真实 tokenizer 编译、NLL/entropy/CDF 与梯度生产、数据来源真实性认证 |
| M2 算子 | 八类算子、固定质量/预算外壳、完整轨迹、trace、精确成员与训练顺序 | 外部语义相似度计算；确定性贪心不保证找到全局可行或最优组合 |
| P1 交接 | 执行、发布、读取、原始快照 CPU 重放、未执行下游字段为空、成本保留 | 消费异步缺失指标请求并自动重跑的新后端 |
| Leval | 绑定检查、固定加权与约束一致性、事实/假设区分、语言解释不能覆盖数值 | 从两侧梯度计算收益的数值内核及其校准；真实能力验证 |
| Lopt | 完整程序、实际 diff、父子谱系、编辑/节点/seed 政策、有限修复、调用证据 | 策略有效性证明；规则生成与 mock 提案不是学习收益 |
| 三角色 | 共用客户端/ModelRef，独立 prompt/history，固定 RoleScope，显式交接，调用前后身份检查 | 实际模型服务、服务端权重真实性、具体业务 prompt 正文、传输层超时实施 |
| 内层搜索 | current/best 分离，有效可行结果排名，失败/等待，重复投递去重，精确最佳交接，稳定边界恢复 | 用户待填退出条件、真实验证后端、资源预留/扣减政策、在途外部任务完成对账 |
| 外层/M8/M9 | 已有记录契约、模型/批次/费用引用和使用历史接口 | 模拟或真实 SFT、配对测评、接受/拒绝、原子晋级、完整外层恢复和全局基础设施费用 |

## 跨模块重点检查

1. M1 → M2：请求 ID 保序，缺失值留在原单元格；模型/spec/CDF/梯度坐标版本不匹配时拒绝。新增内存入口核验堵住派生指标绕过。
2. M2 → P1 → P0：ProgramSpec 与实际 BatchManifest 分开；发布已有结果不重选，加载的 CPU 重放不新增选择费用。最佳消费需要独立 batch/request 锚点。
3. P1 → 验证接口：请求绑定模型、程序、精确批次、目标、measure、protocol 和 context。验证器负责实际计算，调度器只核对返回结果的绑定与算术。
4. Leval → Lopt → Lexec：只通过明确产物共享反馈，保持各自记忆；解释不能改数字，提案不能改 evaluator/门禁/预算，执行器只执行授权程序。
5. 搜索 → 最佳交接：只比较 valid 且固定约束通过的结果，平分保留先者，负值不与虚构零基线比较；训练清单取原实际成员和顺序。
6. 交接 → 后续梯度/训练：固定清单可在追加特征后的视图消费；选择重放必须使用原始视图。已有梯度引用不证明文件存在或向量有效。
7. 恢复与费用：timeline/角色 history/最佳/成本一起核对；导入旧结果不重收费用，新增外部工作用原 CostEvent ID 去重。预操作失败和事后保存失败分开处理。

## 已知约束与后续接入陷阱

- 退出条件按用户要求保留 None。自动 run 会返回 awaiting_stop_condition；条件未填时只能显式 step。demo 的三次验证结束规则不属于研究配置。
- NLL/entropy 是外部预计算输入。只有某个 CDF 视图的 SignalRecord 时，默认 raw 查询不会跨上下文猜取；生产器同时使用 raw/percentile 算子时应保存无 CDF 的 raw 记录和对应 CDF 视图，二者可引用同一次实际前向证据，不重复计算。
- 选样依赖的信号缺失会返回明确请求，但当前循环不会自动计算它们。启动冻结搜索前应补齐所需选样指标；更新标签/分数快照需显式建立新 context/session。选中后的梯度可由外部验证器在独立缓存中补算。
- 同步外部调用只能在操作边界检查 stop callback；真实后端必须实际实施超时。回调应只读状态，不在回调里再执行搜索工作。
- 启用 journal 才会在操作边界自动存完整快照；快照会复制既有 history/cycles，长搜索的累计存储与 I/O 会增长，当前业务账本不包含全部控制器/文件系统开销。
- 在途恢复为 interrupted 后不自动重发，尚无通用外部完成对账；这不是完整 M9。损坏的最近完整快照会报错，不静默回退。
- 外部结果一致性、ArtifactRef/hash、ModelRef 都不是实际计算真实性证明。拒绝已知 mock 标记也不等于认证其他来源。
- 角色低层 generate(work) 是可信调用方接口；最终测试任务的契约标签检查不替代真实来源家族、近似题面隔离与服务端数据挂载隔离。
- 确定性质量门禁使用外部质量证据，代码无法证明人工/外部标注结论真实；重算密度只保证派生统计一致。
- 模型 commentary 只做结构和引用检查，不能保证每句解释正确；系统排名不读取解释文本，不能把自然语言总结当作新的实验事实。
- 排名按已核验 total_value 严格比较，极小浮点差异也可算改善。正式实验需在验证/比较协议中明确数值精度、最小改善和噪声处理；本轮没有擅自加入研究阈值。

## 防止把 mock 与声明误写成结果

FixtureTokenizer 是合成分词器；FixtureRoleClient 是显式脚本客户端；FixtureValidationProvider 的 0.1/0.3/0.2 是声明值。`make_signal` / `make_gradient` 是包装已计算结果，`parse_contract` / `validate_workflow` 检查记录与声明关联，均不替代真实算法执行。模型“建议改善某技能”属于待检验假设。

本轮保留原研究文档和旧证据，不把合成通过写成自改进成立。最终全量测试、内层与独立串行参考、检查点/最佳清单、旧池/旧历史兼容及源码包检验见 `runs/full-module-audit/verification.json`。审计结论仅覆盖实际复查和测试范围，不承诺已证明不存在任何潜在错误。

最终审计专项 13 项、原目录全量 433 项、格式与静态检查通过。源代码中的 8 处省略号均位于明确的 Protocol 接口，包含待填退出条件与外部后端；未把这些接口算作真实算法实现。Motivation 中估值校准、程序搜索相对固定规则的收益、更新模型是否更会选择数据、记忆/缓存收益等研究问题，仍需要实际实验，不能由这些工程测试推导结论。

<a id="outer-review"></a>

原文件：`OUTER_TWO_PASS_REVIEW_ZH.md`。

# 两遍逻辑与接口复查：0.8.1

2026-11-05。对照 AGENTS、原蓝图、既有 P0/P1 和本次用户确认的两阶段流程检查。
原蓝图单阶段描述与新增筛选轨迹训练的差异已由用户要求覆盖；没有改写原文。

## 第一遍：契约、状态与执行边界

| 发现 | 影响 | 修复 |
|---|---|---|
| 补证判断后再次 run 会重新调用判断器；新验证能继续沿用旧补证决定 | 可能未经显式新证据推进，或多次复测 | 已判断的相同上下文等待；新报告须有自己的补证决定才能复测 |
| TaskEvaluation 只核对套件元数据，缺少逐次请求／返回绑定 | protocol 或 checkpoint 字段可能被替换而未报错 | 保存完整 execution_requests/results，核对模型、任务、seed、协议、checkpoint、结果、费用与 attempt |
| 外层仅信任回执内部自洽，未完整比对原训练请求及验证 checkpoint | 可消费另一份学习率配方或另一份权重的有效回执 | 外层比对精确请求和权重引用；不匹配立即 blocked，已执行成本保留 |
| 第二个任务的 mkdir／请求写盘在异常保护外 | 前一任务已发生的费用可能未传给外层 | 派发前写盘纳入保护；保留已完成事件，不给未派发任务造一次模型费用 |
| 训练／验证后端返回错误类型时先报错、后读取 usage | 已报告的合法费用可能丢失 | 先保留有效 ResourceUsage，再拒绝坏结果，仍然记录失败 |
| TrainingOutcome 解码只检查回执字段相互一致，未复核计划预算 | 可整体改写处理量或 optimizer steps，绕过执行服务已有核对 | 构造／JSON 边界也核对 epochs、监督 token、累积窗口及精确步数 |
| 接续 checkpoint 未核对 requires_grad 参数集合 | 相同 ModelRef 下可能意外解冻基座 | 与上一 checkpoint 保存的 trainable_parameters 核对，范围变化在更新前拒绝 |

另外补齐 DecisionContext 的 attempt 绑定，防止来自不同外层尝试的记录混用。
第一批 10 项案例在旧实现均失败，其中 9 项直接复现旧边界问题、1 项定义旧报告迁移接口。
第二遍又实际复现了接续参数范围变化和两种坏返回 usage 丢失问题；失败日志随交付保留。新增专项共 18 项。

## 第二遍：完整链路与历史产物

复查路线为：M1 记录／mask → P1 精确清单 → 内层历史 best → 子集 SFT → 任务验证 → 判断 →
筛选轨迹 SFT → 同协议任务验证 → 再判断。没有改变原清单、重抽训练数据或填入研究阈值。

专项覆盖等待状态不重复派发、两处退回、精确 recipe/checkpoint、费用保留、协议和 reward 文件改写、
严格 JSON 往返、旧报告补齐、CPU 梯度参考、实际 checkpoint 接续、adapter 基座冻结与参数范围漂移。

### 旧 0.8.0 报告怎么接入

旧 TaskEvaluation JSON 仍可只读解码，原 CapabilityReport、rollout 和成本保持原值。
缺少逐次请求／返回时 `protocol_verified=False`，不能直接作为接收或 baseline 证据。
旧运行目录实际已保存这些文件，因此可只读补齐：

```python
from self_curate.orchestration import load_task_evaluation

report = load_task_evaluation(
    old_validation_directory,
    expected_model=system_saved_model,
    expected_suite=system_saved_task_suite,
    expected_checkpoint=system_saved_checkpoint,
)
```

loader 将目录名、请求／结果、report、rollout、费用与调用方锚点一起核对。
返回新的不可变组合对象，不改写旧 evaluation.json 或任何历史文件。
新目录的报告自身也保留执行证据，文件与报告不一致会拒绝。

`protocol_verified` 只表示所提供记录的执行协议绑定自洽，不证明远程模型或环境真实执行。
真实模型／任务适配器仍负责来源和实际计算的核验。

## 保留的边界

- 内层退出条件和外层接收标准仍待填；未默认接收或安装数字阈值。
- 真实模型加载、任务执行／判分、自动筛选轨迹生产和数值梯度仍由外部模块接入。
- attempt 内 accepted 指针与全局提交分开；完整 M9 崩溃恢复没有实现。
- ModelStore 仍必须隔离可变参数；接续范围检查不能证明自定义 loader 没有共享底层 tensor。
- 无 GPU／目标模型／真实 benchmark 执行，mock 数值不代表效果提升。

最终全量测试、两遍专项、原 0.8.0 报告只读兼容、mock 闭环和历史证据保护见
`runs/outer-two-pass-review/verification.json`。

<a id="leval-review"></a>

原文件：`LEVAL_REVIEW_ZH.md`。

# Leval 文档与模块两轮复查（0.9.1）

2026-11-06。本轮核对原蓝图、D102–D108 用户扩展、P0/P1 契约、M1 使用历史、两套 Leval、共享角色、内层搜索及外层训练/验证。

结论：发现了版本、数值条件、恢复与费用边界的问题，已补回归并修复。原蓝图保持原文，候选 SFT 仍按用户新增授权运行在隔离分支；解释角色与共享模型不提交参数。

## 第一轮：要求、数值与模块接口

| 问题 | 原行为 | 修复 |
|---|---|---|
| SFT 协议未绑定实现 | 换训练器/任务执行器仍可能保持同一 protocol，历史被当成同口径 | protocol 绑定数值内核、服务/后端类的源码版本、声明的 implementation_hash；真实适配器须明确声明实现与配置身份 |
| 调用方 autocast 泄入 | 相同输入/配方在 AMP 上下文里得到另一份梯度或 SFT 更新，缓存 key 未区分 | Torch NICE/SFT 在自身计算范围禁用外部 autocast；保留参数原生 dtype，CE 用 float32；记录 torch/设备/TF32/确定性等数值条件，变化则失效或拒绝 |
| 模型 factory 未绑定 | 同 state_dict 与不同模型构造实现可能拥有相同后端身份 | StateDictModelStore 把 factory 源码/声明身份加入 implementation_hash；GPU/远端 store 仍负责核验实际权重与隔离 |
| 恢复丢失 M1 使用记录 | restore_reference 恢复基线，却没带回先前候选的 candidate_sft | 保存紧凑的 usage_records，按 ID/content/tokenization 绑定恢复，单调合并；旧 pending 不覆盖现有 accepted/rejected |
| 坏返回/异常丢费用 | 服务无 ledger 时丢弃有效 usage；空集合汇总甚至使未知费用显示为 0 | 保留服务详细 ledger；缺 ledger 则记录边界调用及有效 usage，其余未知为 null；不编造 rollout 数，也不把已报告的空 ledger 当成一次物理执行 |
| 预算预检时机太晚 | 已知轨迹过长，仍先计算基线后才拒绝 TrainingRequest | 在任何基线/训练调用前检查完整轨迹长度，不裁剪、不浪费已可避免的验证费用 |
| 无效基线证据缺项 | provider.baseline 只接受有效报告，导致失败 evidence 中看不到本次无效报告 | 额外保存 attempted_baseline，保留真实失败报告而不把它缓存为有效基线 |

最初 10 个专项在修复前全部实际失败。其中 CPU 微型模型的环境 autocast 泄入使原始梯度最大差约 0.00210、SFT 权重最大差约 0.00228；修复后与默认数值上下文一致。失败证据保留在本次运行目录，不把测试奖励当研究结果。

## 第二轮：恢复、历史、边界与旧产物

1. **自动基线不能冒充已存在参考。** 原先两个新 provider 在相同配置下会拥有相同 protocol，即使各自重新测了一次基线。现在自动基线有独立不可变 reference_namespace；旧搜索只能使用同一 provider 或恢复其已保存基线。显式导入同一份基线仍拥有稳定身份。namespace 是引用身份，不是采样 seed，不改变训练/任务随机规则。
2. **真实域中的 mock 识别一致。** NICEReference 与 NICE 分数证据在检查 URI 前统一 strip/lower，禁止空白或大小写掩盖的 fixture/mock。元数据反例仅用于本机单测，不代表真实模型或任务执行。
3. **Windows 嵌套证据路径。** 深目录复查实际遇到 WinError 206。评价 run 名改用完整 128 位 UUID 的 URL-safe 紧凑编码，减少路径占用；不削减身份位数，不声称支持任意长路径。任务执行的完整 request hash 目录格式保持兼容。
4. **重叠批次只复用兼容梯度。** 测试同模型/同配方的较小重叠批次只命中既有样本；修改投影配置重新计算。原有 source/model/mask/坐标身份继续保留。
5. **恢复不逆转历史。** 检查使用记录内容绑定、终态单调性、无重复收费、原始池不修改，以及两种内核到 Leval→Lopt→P1→内层快照的连通性。原有外层“子集→验证/判断→接续轨迹训练→验证/判断”继续沿用原模块。

另三项 factory/真实域反例和两项参考身份/真实适配器身份反例也在原 0.9.0 源码下实际复现失败。专项共新增 20 项，首次失败证据共覆盖 15 项；其余案例核验重叠缓存、稳定导入和终态恢复等边界。

## 接口与兼容

- 既有 ValidationRequest/Result、RoleEvaluation、RoleScope、P0/P1、TaskSuite/TaskEvaluation、TrainingRequest/Outcome 不改格式。Leval/Lopt prompt 继续用现有结构化槽位，正文待填。
- `SFTValidator.save_reference()` 输出 reference schema=2，新增 reference_namespace 和 usage_records；restore 核验独立文件 hash、protocol、模型/suite/recipe，以及逐条训练输入身份。它不恢复在途服务或把旧费用重新收取。
- 0.9.0 的搜索/cycle/报告仍可只读核验，不改写旧文件。新实现改变了执行协议身份，因此旧计算会话不能静默接着新版数值内核运行；需要新建会话，或显式核验旧 TaskEvaluation 后作为新会话的固定参考导入。旧 schema=1 reference 文件可解析字段，但不伪造新版 namespace/实现身份来绕过 protocol 检查。
- Python 本地 mock 适配器可使用类/源码自动身份；真实候选验证的 TrainingBackend 和 TaskRunner 必须提供 SHA256 字符串 implementation_hash，覆盖远端实现和实例配置。数字身份不是远端执行真实性的证明，加载权重/环境/判分仍由真实 adapter 与服务器验收负责。
- NICE 的梯度矩阵及版本证据属于 valuation；M1 仍通过既有外部 GradientFeature 接口接入指标，本轮不把矩阵行伪装成已经回填的 M1 特征或持久化 M5 缓存。候选训练使用历史则通过 updated_records 显式返回，原始数据视图不改写。

全量验收、两种闭环、旧产物恢复、原 NICE 数值串行对照、旧证据 checksum 及源码包证据，以 `runs/leval-review/verification.json` 为准。真实 GPU、目标模型长上下文容量、任务容器/判分、提示词正文和研究阈值继续待接入；本机测试不证明模型能力提升。

<a id="signal-review"></a>

原文件：`SIGNAL_REVIEW_ZH.md`。

# NLL / entropy 复查（0.10.1）

2026-11-06：沿“供应数据 → 前向统计 → M1 标签 → 保存/重载 → 筛选 operator”复查。原有数学公式、p-1 因果位移、assistant/技能监督范围、完整上下文和模型版本隔离保持正确。本轮修复持久化和外部后端边界。

## 实际发现与修复

1. 原 loader 校验文件 checksum 和结构，但不会独立重聚合数值。标签与结果文件同时写错并重算 checksum 时可能被接收。新增逐位置聚合输入快照，重载重新计算 NLL/entropy；跨调用缓存同样覆盖。
2. 原根请求只保存原记录 hash，却缺少原记录快照，无法完整核对评分过程中质量/技能/历史是否被改动。新增 inputs.jsonl，按原 hash 核对并只接受 signal/forward_scoring 增量。
3. 后续样本失败时，异常对象包含已完成标签，磁盘只留下部分标签 hash。新增 partial-records.jsonl 及 failure.json 中的 checksum，进程结束后仍可用既有 read_records 恢复这些记录。
4. 外部后端明确报告梯度或训练工作时仍可能被当作纯前向统计接收。服务和成功批次边界现在拒绝已知非零相应计数，保留原始实际费用；不把未知数改为零。

首批 8 个负例在 0.10.0 下实际失败，日志保留在验收目录 negative-before.txt。新增测试同时覆盖完整指标直接复用与新技能范围交叠、两个前向调用的缓存重聚合、bits 换底、缺失快照和错误 computed/cached 声明。

## 接口与兼容

公共 SignalService/SignalProvider/ModelStore/LogitsProvider 和 M1/P0/P1 契约保持原样。新根评分请求 schema=2，其他数值/数据契约版本不变。旧 schema=1 产物只读兼容；新计算实现 hash 与旧版本不同，不能把旧值冒充新口径继续计算。旧结果按其原模型/computation hash 显式加载，不重写旧证据。

标签与证据的内部一致性校验不等于对外部源真实性的认证。参考 Torch 后端逐样本加载隔离分支并核对状态，尚未验收目标大模型/GPU 的容量和吞吐；固定参考 CDF 百分位仍待接入，缺失值不填零。

## 验收

最终全量 pytest、独立数学对照、缓存和旧产物加载、完整 mock 衔接、历史证据保护与源码包，以 runs/signal-review/verification.json 为准。本机只运行微型 CPU 模型与显式合成 logits；没有真实任务能力提升结论。

<a id="previous-progress"></a>

## 清理前完整进度记录

# 当前进度：NLL / entropy 持久化与接口复查（0.10.1）

2026-11-06：补齐原始 M1 输入快照和逐位置聚合输入，重载时重新聚合并核对质量/技能/历史未被评分改动；跨调用缓存同样检查。失败目录保存可直接 read_records 的部分标签；明确报告过梯度/训练的外部后端不作为纯前向结果发布，实际费用保留。

首批 8 项反例已在旧实现上实际复现。接口和旧 schema=1 产物保持只读兼容，计算实现变化进入新 computation hash。详见 [SIGNAL_REVIEW_ZH.md](HISTORY.md#signal-review)，最终验收和源码包以 runs/signal-review/verification.json 为准。

# 历史：NLL / entropy 前向计算与 M1 回填（0.10.0）

2026-11-06：新增 signals 包，复用 SignalProvider/SignalRecord/SignalSpec/SignalBinding，不把数值塞进静态字符串标签。数据供应接口接受规范化轨迹，TorchSignalBackend 实际冻结前向，ProvidedLogitsBackend 接入调用方 raw 全词表 logits。按全局 p-1 对齐，仅统计 assistant 或技能 span 的监督交集，支持均值/和、自然底/2 底；技能缺失保留 missing。

同模型/同输入/同计算身份复用逐 token 统计，新增范围只补缺位置；模型/adapter/tokenizer/mask/计算配置变更不覆盖旧版本。输出完整证据、结果与费用，返回 M1 enriched records；新增 publish_signal_enrichment 发布新数据池，不改原始池。已有两个 raw operator 可直接消费，固定 CDF 百分位保持待接入。

详见 [SIGNAL_COMPUTATION_ZH.md](SIGNAL_COMPUTATION_ZH.md)。本机只执行 CPU 微型模型/显式 logits fixture，不下载目标权重。最终回归、数学独立对照、M1 池/算子接续和源码包以 runs/signal-computation/verification.json 为准。

# 历史：Leval 两轮文档/模块复查（0.9.1）

2026-11-06：补齐 SFT 实现版本与自动基线命名空间、Torch 数值上下文隔离/身份、模型 factory 版本、基线恢复后的 M1 候选使用历史、未知/失败费用、预算预检、无效基线报告与真实域 mock 识别；压缩评价 UUID 路径但保留完整 128 位身份。

新增 20 项专项，15 项在 0.9.0 下实际复现失败。两轮发现、兼容接口与实际边界见 [LEVAL_REVIEW_ZH.md](HISTORY.md#leval-review)。最终原目录全量测试、两种闭环、旧产物只读恢复、数值对照和源码包以 runs/leval-review/verification.json 为准。真实模型/GPU/任务评分未执行，prompt 正文和研究阈值继续待填。

# 历史：可切换 Leval 内核与历史反馈（0.9.0）

2026-11-06：增加 NICEValidator 和 SFTValidator。前者实现选中样本 assistant CE 梯度、实际 rollout 动作 NLL 重放、fresh-Adam/CountSketch/固定任务分母 cosine；后者复用外层训练与任务验证接口，从冻结参考模型训练隔离候选，再计算固定开发 suite 相对基线奖励增量。通过 configs/leval.json、create_validator 选择，不在同一搜索混排。

新增 LevalContext，代码核对本次/同条件历史观察并生成比较槽位；Leval 与 Lopt prompt 输入明确消费，私有 history 不互读。cycle、搜索快照兼容旧格式；候选失败费用保留。基线可校验保存/恢复，避免续跑重算基线。

接口、示例与边界见 [LEVAL_BACKENDS_ZH.md](LEVAL_BACKENDS_ZH.md)。两个 CLI demo 是合成接线；专项还用微型 CPU 模型实际求梯度和 SFT，不代表任务能力提升。真实模型/任务环境、原始数据真实性、具体 prompt、停止/接收阈值待后续。最终验收证据位于 runs/leval-backends/verification.json。

# 历史：两遍逻辑与接口复查（0.8.1）

2026-11-05：第一遍复现并修复补证状态、验证协议证明、外层精确请求／权重消费、派发前 IO 异常费用、TrainingOutcome 预算和 DecisionContext attempt 绑定；第二遍核对历史报告、执行文件、数值回执和模型接续，补齐可训练参数范围及坏返回的有效 usage 保留。新增 18 项专项，第一批 10 项、参数范围及两个 usage 案例先实际复现失败。详见 [OUTER_TWO_PASS_REVIEW_ZH.md](HISTORY.md#outer-review)。

旧报告仍只读加载；无执行证据时不得直接判断，可通过 load_task_evaluation 核对原目录中的实际请求／结果后接入，不改写旧文件。研究阈值和真实执行边界保持原样。最终证据位于 runs/outer-two-pass-review/verification.json。

## 历史：两阶段外层 SFT、任务验证与接收／退回（0.8.0）

2026-11-05：按用户确认实现“子集训练→验证→判断→接收后继续筛选轨迹训练→验证→判断”。
训练模块提供 best 精确承接、配方、可选 torch 实际优化、隔离权重／adapter 与检查点。
任务验证模块提供固定开发任务／seed／采样协议、可注入执行器、逐任务记录和能力汇总。
标准未填写时停在 awaiting_decision，不默认接收；第二阶段退回保留第一阶段模型。
详见 [OUTER_TRAINING_ZH.md](OUTER_TRAINING_ZH.md)。

数据生产、目标模型加载适配、真实任务执行／评分、研究阈值、全局提交与完整 M9 恢复仍待后续。
微型 CPU 模型实际更新与完整接续已测试；mock 奖励无研究效力。最终证据以 runs/outer-training/verification.json 为准。

## 历史：全模块实现边界与连通性审计（0.7.1）

2026-11-05：重新检查数据/算子/P0/P1/角色/反馈/内层循环/存储交接。原有 420 项通过后，用额外边界案例复现并修复五类问题：直接 MetricView 入口未重核 mask/派生证据；未启动操作在预写检查点失败后被误标 interrupted；不可序列化回复导致已知 usage 丢失及错误诊断过窄；已知 mock 轨迹可借 precomputed 声明进入 real 交接；独立 Lopt 入口未拒绝不同 EditPolicy 的 previous_cycle。新增 13 项回归，其中 11 项先实际复现失败。详见 [FULL_MODULE_AUDIT_ZH.md](HISTORY.md#full-module-audit)。

本轮不实现真实模型、数值验证、NLL/entropy/梯度计算或外层训练。内层退出条件仍待调用方配置；原技术文档与历史证据保持原样。最终验证记录位于忽略的 runs/full-module-audit/verification.json。

最终审计专项共 13 项，原目录全量 433 项、格式与静态检查通过。独立串行对照、旧池/旧角色历史/0.7.0 搜索检查点兼容、证据保留及源码包检查以该 verification.json 为准。

## 历史：内层搜索闭环，退出条件待配置（0.7.0）

2026-11-04：新增 search 模块，复用 RoleTeam、P1 和 ValidationProvider，完成外部验证、反馈、提案、筛选、再验证的多轮接续。分别维护 current/best/history/停滞次数/成本；只有 valid 且满足固定约束的实际批次可成为 best，平分保留更早项。退出条件按用户要求保留 StopCondition 回调，不代填候选数、预算或停滞阈值；未配置时 run 返回 awaiting_stop_condition，step 可手动推进。支持完整检查点、稳定边界恢复和独立最佳清单交接；未确认完成的在途操作恢复为 interrupted，不自动重发。接口与运行方法见 [INNER_SEARCH_ZH.md](INNER_SEARCH_ZH.md)。

当前交付覆盖 P2 的内层搜索部分；外层模拟训练/接受/拒绝、真实数值验证、梯度、真实模型服务仍未实现。新证据位于忽略的 runs/inner-search/，原研究文档与旧证据保持原样。

内层专项 30 项、隔离副本全量 420 项通过；最终原目录回归、独立串行对照、历史兼容与源码包检查以 runs/inner-search/verification.json 为准。

## 历史：按阅读教程复查数据流与交接（0.6.2）

2026-11-04：沿数据记录、指标视图、筛选、P0/P1 交接、三角色反馈及持久化逐段复查。修复 MetricView 的公开成员列表可重绑定、导致冻结上下文实际候选范围漂移的问题；统一短名称同级 staging，避免最终路径合法时嵌套临时路径先超过 Windows 限制。保留原子发布、拒绝覆盖、失败证据和历史格式。新增两个回归案例；检查与交付证据见 [教程链路复查](HISTORY.md#tutorial-review)。

## 历史：三角色两遍框架与接口复查（0.6.1）

2026-11-03：第一遍核对框架不变量，修复会话未固定评价条件、Team 可半替换、cycle 身份检查与独立恢复锚点遗漏；第二遍验证完整产物交接、失败反馈、P0 成本、P1/梯度消费者和旧历史兼容。新增 RoleScope、cycle 发布/加载、previous_cycle 执行反馈接口。详细说明见 [两遍复查](HISTORY.md#roles-review)，最终证据与源码包位于 `runs/roles-two-pass/`。

## 历史：三角色共享参数与独立上下文（0.6.0）

2026-11-03：新增共享客户端 RoleRuntime 和执行/验证解释/优化三个 RoleAgent，绑定同一不可变 ModelRef；各自固定 prompt 配置、独立 before/after history、快照与版本检查。Prompt 正文为空，填写 configs/roles 下对应 JSON 的 instructions 即可。

复用现有 P1、Leval 数值事实整理与 Lopt 程序检查；不实现真实模型服务、数值验证内核或完整搜索状态机。使用方式与边界见 [三角色接口](THREE_ROLE_AGENTS_ZH.md)。模型与 P1 成本绑定同一外层 attempt_id，会话 ID 独立。最终证据与源码包位于忽略的 `runs/three-roles-final/`；首轮结果保留在 `runs/three-roles/`。

## 历史：全模块连贯性与接口复查（0.5.1）

2026-11-03：从 M1/P0/M2/P1 到外部验证交接和 Leval/Lopt 重新核对，修复连续候选交接、报告覆盖、失败记录、调用摘要、随机配置锁、模型状态变化、提案 ID、真实域和审计统计边界。候选现在可以直接作为下一轮的标准 P1 输入，发布不重跑、不重记成本。

详细发现、适配矩阵、运行方式和未实现边界见 [全模块复查](HISTORY.md#coherence-review)。最终验收与源码包位于忽略的 `runs/coherence-review-final/`，`runs/coherence-review/` 保留首轮结果。角色输入使用控制配置与 artifact 摘要，避免传完整成对特征表。原始蓝图与旧证据保持不变，P2 搜索状态机、数值验证内核和真实模型服务继续留待后续。

## 历史：Leval 反馈与 Lopt 单步优化（0.5.0）

2026-11-03：按用户最新优先级，跳过验证算法实现，新增精确绑定的外部结果接口、确定性 Leval 反馈、规则与可注入模型 Lopt、实际程序差异、有限重试、完整调用记录，以及 P1 候选重跑和下一次验证请求。

当前可运行 synthetic fixture 链路。真实模型客户端与验证器仅留接口，完整 P2 搜索、历史最佳、恢复和训练接受仍未实现。接口与接续任务见 [feedback_lopt_interfaces.md](feedback_lopt_interfaces.md)。本轮证据位于忽略的 `runs/feedback-lopt/verification.json`。

以下保留历史记录，其中旧“下一轮”安排由本段最新优先级替代。

# 当前进度：P0/P1 文档对齐复查（0.4.1）

2026-11-03：重新对照原蓝图 P0/P1 与数据架构文档，保持原始需求不变。
修复历史最佳清单缺少独立外部锚点、独立估值报告引用不全、独立能力报告成员不自洽三个边界问题。
复查矩阵见 [P0_P1_REQUIREMENTS_ALIGNMENT_ZH.md](HISTORY.md#p0-p1-alignment)。校验语义 p0-v3；M1 数据和 m2-v2 算子语义不变。
当前证据与交付包位于忽略的 `runs/p01-document-alignment/`；实际统计以 verification.json 为准。
本轮不进入 P2；真实计算、实验与提交仍未执行。

## P0/P1 正式交接历史（0.4.0）

2026-11-03：完成 P0 引入后的模块适配复查，并在原有八算子基础上提供 P1 的 prepare-selection / inspect-preparation 正式入口。
复查记录见 [P0_P1_INTEGRATION_REVIEW_ZH.md](HISTORY.md#p0-p1-integration)，接口与命令见 [p1_handoff_interfaces.md](p1_handoff_interfaces.md)。
新增系统模型/目标/上下文锚点校验、共享选择身份计算、只读成本 ID 索引及选择成本发布。
可行结果直接交出同一 BatchManifest 的未估值工作流；不可行/缺分数保留原因、请求、trace 和本次选择成本。
本轮不接真实模型，也不执行 P2 搜索、候选训练或提交。证据与更新交付包在忽略的 `runs/p01-integration/`，以 verification.json 记录为准。

阶段边界：P0/P1 本地契约、八算子和交接已具备；后续为 P2 的模拟搜索、历史最佳、接受/拒绝与恢复。

## P0 补齐历史（0.3.0）

2026-11-02：按 P0/P1 审计补齐 P0 接口。12 项核心契约均已有类型，另新增目标梯度、成本引用和 WorkflowBundle 的跨记录校验。
公共接口见 [p0_interfaces.md](p0_interfaces.md)，实现前设计见 [p0_contract_design.md](p0_contract_design.md)。

当前验收边界：严格解析、序列化往返、模型/批次/目标/成本绑定、待执行与无效状态、任务/rollout 统计口径，以及现有 M1/M2 回归。
样例 accept/reject 是明确的 mock 记录，不是已经执行的 P2 外层尝试。真实算法、原子提交与恢复均未在本轮实现。
新证据与交付包保存在忽略的 `runs/p0-contracts/`；具体测试统计、CLI、旧数据只读和打包检查以其中 verification.json 为准。

阶段状态：P0 本地契约链完成；P1 本地八算子已验收；下一步 P2 使用这些接口实现模拟业务与状态迁移。

## M2/P1 交付历史（0.2.1）

日期：2026-11-02。本轮按用户要求先分析依赖，再完成 M2 八类算子与公共交接接口。

随后按用户要求复查，修正批次修整性能、历史暴露报告、混合 tokenizer 边界和预算数值边界，补充预期模型交接检查。
本轮说明见 [M2_REVIEW_ZH.md](HISTORY.md#m2-review)，当前算子语义为 m2-v2。
新验收与交付包位于 `runs/m2-review/`；旧 `runs/m2-acceptance/` 保留为 0.2.0 的历史证据。

- 实现前设计：[m2_design.md](m2_design.md)；公共入口和完整参数：[m2_interfaces.md](m2_interfaces.md)。
- 上游复用 M1 的 ID 保序、版本化内存 MetricView，不增加持久化 operator 索引。
- ProgramSpec 严格 DAG、固定 QualityGate/BudgetPack、八类算子、精确 BatchManifest 与节点追踪均已实现。
- NLL/entropy 读取外部预计算结果；语义去重读取外部成对相似度；本轮不运行模型、不生成梯度或训练曝光。
- CPU 全量测试和合成 CLI 验收覆盖 M1 回归、八算子、图校验、版本匹配、缺失请求、混合唯一归属、token/batch 约束和精确清单交接。
- 合成示例：12 条输入 → 8 条完整轨迹/64 个监督 token，两个分支各 32 token；有效 batch=8。
- 组批为版本化确定性贪心，不能将未找到合法批次解释为所有组合均无解。

该版证据与源码交付包位于忽略的 `runs/m2-review/`，不纳入 Git；实际测试统计和各项检查以其中 verification.json 为准。

## 后续按这份计划执行

| 轮次 | 具体交付 | 验收重点 |
|---|---|---|
| 下一轮 P2 | 显式 mock 内外层搜索循环、历史最优清单、恢复状态 | 上升后下降仍保留真实最佳 manifest；拒绝保留证据；恢复不重复训练/接受；只消费本轮 M2 公共接口 |
| 随后真实输入后端 | 专用 tokenizer/template 与外部 NLL/entropy 服务 | 独立 token/mask 对照；模型/spec/CDF 精确绑定；缺失请求与回填闭环；禁止隐式 fixture 回退 |
| M5/M6 | 最终清单上的按需梯度与确定性估值 | 完整缓存身份、仅计算缺失最终成员、同一批次评估、零/无效梯度处理 |
| M8 | 候选训练、验证、接受/拒绝及曝光记账 | verify_batch 后按原 training_order 训练；拒绝也保留曝光；不按测试集调参 |

可直接交给下一轮执行的任务：

```text
阅读 AGENTS.md、docs/self_curate_codex_blueprint.md、docs/decisions.md、
docs/p0_interfaces.md、docs/p1_handoff_interfaces.md、docs/progress.md。
实现 P2 的显式 mock 搜索循环，复用现有核心契约、WorkflowBundle、CostLedger 及 prepare_selection。
先定义运行状态，不重写 P0/M1/M2，不引入真实模型或训练后端。
测试历史最优与当前候选分离、精确清单传递、拒绝保留证据、接受后的模型身份切换、恢复幂等。
mock 指标必须标注 synthetic；训练曝光只在模拟训练事件中通过 M1 usage 接口记录。
模拟执行记录必须 backend=mock；阶段完成后使用 validate_workflow(bundle, view, expected_model=..., expected_objective=..., expected_selection_context_hash=..., expected_batch_id=best_batch_id) 校验跨模块身份和历史最佳清单。
完成后跑全量测试、更新公共接口/决策/进度，交付可运行命令和证据，不执行 Git 提交。
```

## M1 已交付历史

| 版本 | 完成内容 | 当时验收 |
|---|---|---|
| 0.1.0 / m1-v1 | 文本与工具轨迹规范化、assistant-only mask、技能证据统计、使用历史、不可覆盖数据池 | 44 passed；证据位于 runs/acceptance/ |
| 0.1.1 / m1-v2 | 外部标签和 NLL/entropy 接入、精确模型/scope/spec/CDF 绑定与信号合并 | 67 passed；证据位于 runs/acceptance-signals/ |
| 0.1.2 / m1-v3 | ID 保序内存 MetricView、外部标签/元数据封装、版本化梯度引用、无损使用状态更新及三轮复核 | 96 passed；证据位于 runs/m1-three-pass/ |

旧 m1-v1/m1-v2 数据池继续只读兼容，不改写旧文件恢复。旧产物、交付包与研究文档保留。
M1 数据取值见 [m1_operator_data_contract.md](m1_operator_data_contract.md)，外部计算结果接入见 [enrichment_interfaces.md](enrichment_interfaces.md)，三轮复核见 [M1_THREE_PASS_REVIEW_ZH.md](HISTORY.md#m1-review)。

真实 tokenizer、自动标注、NLL/entropy/CDF 计算、模型后端、梯度计算和训练尚未实现；fixture 验收不能作为研究结论。
