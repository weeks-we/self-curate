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
