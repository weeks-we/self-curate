# 当前进度：两遍逻辑与接口复查（0.8.1）

2026-11-05：第一遍复现并修复补证状态、验证协议证明、外层精确请求／权重消费、派发前 IO 异常费用、TrainingOutcome 预算和 DecisionContext attempt 绑定；第二遍核对历史报告、执行文件、数值回执和模型接续，补齐可训练参数范围及坏返回的有效 usage 保留。新增 18 项专项，第一批 10 项、参数范围及两个 usage 案例先实际复现失败。详见 [OUTER_TWO_PASS_REVIEW_ZH.md](OUTER_TWO_PASS_REVIEW_ZH.md)。

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

2026-11-05：重新检查数据/算子/P0/P1/角色/反馈/内层循环/存储交接。原有 420 项通过后，用额外边界案例复现并修复五类问题：直接 MetricView 入口未重核 mask/派生证据；未启动操作在预写检查点失败后被误标 interrupted；不可序列化回复导致已知 usage 丢失及错误诊断过窄；已知 mock 轨迹可借 precomputed 声明进入 real 交接；独立 Lopt 入口未拒绝不同 EditPolicy 的 previous_cycle。新增 13 项回归，其中 11 项先实际复现失败。详见 [FULL_MODULE_AUDIT_ZH.md](FULL_MODULE_AUDIT_ZH.md)。

本轮不实现真实模型、数值验证、NLL/entropy/梯度计算或外层训练。内层退出条件仍待调用方配置；原技术文档与历史证据保持原样。最终验证记录位于忽略的 runs/full-module-audit/verification.json。

最终审计专项共 13 项，原目录全量 433 项、格式与静态检查通过。独立串行对照、旧池/旧角色历史/0.7.0 搜索检查点兼容、证据保留及源码包检查以该 verification.json 为准。

## 历史：内层搜索闭环，退出条件待配置（0.7.0）

2026-11-04：新增 search 模块，复用 RoleTeam、P1 和 ValidationProvider，完成外部验证、反馈、提案、筛选、再验证的多轮接续。分别维护 current/best/history/停滞次数/成本；只有 valid 且满足固定约束的实际批次可成为 best，平分保留更早项。退出条件按用户要求保留 StopCondition 回调，不代填候选数、预算或停滞阈值；未配置时 run 返回 awaiting_stop_condition，step 可手动推进。支持完整检查点、稳定边界恢复和独立最佳清单交接；未确认完成的在途操作恢复为 interrupted，不自动重发。接口与运行方法见 [INNER_SEARCH_ZH.md](INNER_SEARCH_ZH.md)。

当前交付覆盖 P2 的内层搜索部分；外层模拟训练/接受/拒绝、真实数值验证、梯度、真实模型服务仍未实现。新证据位于忽略的 runs/inner-search/，原研究文档与旧证据保持原样。

内层专项 30 项、隔离副本全量 420 项通过；最终原目录回归、独立串行对照、历史兼容与源码包检查以 runs/inner-search/verification.json 为准。

## 历史：按阅读教程复查数据流与交接（0.6.2）

2026-11-04：沿数据记录、指标视图、筛选、P0/P1 交接、三角色反馈及持久化逐段复查。修复 MetricView 的公开成员列表可重绑定、导致冻结上下文实际候选范围漂移的问题；统一短名称同级 staging，避免最终路径合法时嵌套临时路径先超过 Windows 限制。保留原子发布、拒绝覆盖、失败证据和历史格式。新增两个回归案例；检查与交付证据见 [教程链路复查](TUTORIAL_FLOW_REVIEW_ZH.md)。

## 历史：三角色两遍框架与接口复查（0.6.1）

2026-11-03：第一遍核对框架不变量，修复会话未固定评价条件、Team 可半替换、cycle 身份检查与独立恢复锚点遗漏；第二遍验证完整产物交接、失败反馈、P0 成本、P1/梯度消费者和旧历史兼容。新增 RoleScope、cycle 发布/加载、previous_cycle 执行反馈接口。详细说明见 [两遍复查](ROLES_TWO_PASS_REVIEW_ZH.md)，最终证据与源码包位于 `runs/roles-two-pass/`。

## 历史：三角色共享参数与独立上下文（0.6.0）

2026-11-03：新增共享客户端 RoleRuntime 和执行/验证解释/优化三个 RoleAgent，绑定同一不可变 ModelRef；各自固定 prompt 配置、独立 before/after history、快照与版本检查。Prompt 正文为空，填写 configs/roles 下对应 JSON 的 instructions 即可。

复用现有 P1、Leval 数值事实整理与 Lopt 程序检查；不实现真实模型服务、数值验证内核或完整搜索状态机。使用方式与边界见 [三角色接口](THREE_ROLE_AGENTS_ZH.md)。模型与 P1 成本绑定同一外层 attempt_id，会话 ID 独立。最终证据与源码包位于忽略的 `runs/three-roles-final/`；首轮结果保留在 `runs/three-roles/`。

## 历史：全模块连贯性与接口复查（0.5.1）

2026-11-03：从 M1/P0/M2/P1 到外部验证交接和 Leval/Lopt 重新核对，修复连续候选交接、报告覆盖、失败记录、调用摘要、随机配置锁、模型状态变化、提案 ID、真实域和审计统计边界。候选现在可以直接作为下一轮的标准 P1 输入，发布不重跑、不重记成本。

详细发现、适配矩阵、运行方式和未实现边界见 [全模块复查](COHERENCE_REVIEW_ZH.md)。最终验收与源码包位于忽略的 `runs/coherence-review-final/`，`runs/coherence-review/` 保留首轮结果。角色输入使用控制配置与 artifact 摘要，避免传完整成对特征表。原始蓝图与旧证据保持不变，P2 搜索状态机、数值验证内核和真实模型服务继续留待后续。

## 历史：Leval 反馈与 Lopt 单步优化（0.5.0）

2026-11-03：按用户最新优先级，跳过验证算法实现，新增精确绑定的外部结果接口、确定性 Leval 反馈、规则与可注入模型 Lopt、实际程序差异、有限重试、完整调用记录，以及 P1 候选重跑和下一次验证请求。

当前可运行 synthetic fixture 链路。真实模型客户端与验证器仅留接口，完整 P2 搜索、历史最佳、恢复和训练接受仍未实现。接口与接续任务见 [feedback_lopt_interfaces.md](feedback_lopt_interfaces.md)。本轮证据位于忽略的 `runs/feedback-lopt/verification.json`。

以下保留历史记录，其中旧“下一轮”安排由本段最新优先级替代。

# 当前进度：P0/P1 文档对齐复查（0.4.1）

2026-11-03：重新对照原蓝图 P0/P1 与数据架构文档，保持原始需求不变。
修复历史最佳清单缺少独立外部锚点、独立估值报告引用不全、独立能力报告成员不自洽三个边界问题。
复查矩阵见 [P0_P1_REQUIREMENTS_ALIGNMENT_ZH.md](P0_P1_REQUIREMENTS_ALIGNMENT_ZH.md)。校验语义 p0-v3；M1 数据和 m2-v2 算子语义不变。
当前证据与交付包位于忽略的 `runs/p01-document-alignment/`；实际统计以 verification.json 为准。
本轮不进入 P2；真实计算、实验与提交仍未执行。

## P0/P1 正式交接历史（0.4.0）

2026-11-03：完成 P0 引入后的模块适配复查，并在原有八算子基础上提供 P1 的 prepare-selection / inspect-preparation 正式入口。
复查记录见 [P0_P1_INTEGRATION_REVIEW_ZH.md](P0_P1_INTEGRATION_REVIEW_ZH.md)，接口与命令见 [p1_handoff_interfaces.md](p1_handoff_interfaces.md)。
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
本轮说明见 [M2_REVIEW_ZH.md](M2_REVIEW_ZH.md)，当前算子语义为 m2-v2。
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
M1 数据取值见 [m1_operator_data_contract.md](m1_operator_data_contract.md)，外部计算结果接入见 [enrichment_interfaces.md](enrichment_interfaces.md)，三轮复核见 [M1_THREE_PASS_REVIEW_ZH.md](M1_THREE_PASS_REVIEW_ZH.md)。

真实 tokenizer、自动标注、NLL/entropy/CDF 计算、模型后端、梯度计算和训练尚未实现；fixture 验收不能作为研究结论。
