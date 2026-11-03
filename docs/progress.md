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
