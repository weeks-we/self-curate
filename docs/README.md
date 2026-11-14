> EXP001 Rebench (2026-11-11): [workstation plan](EXP001_REBENCH_PLAN_ZH.md), [handoff](handoffs/EXP001_REBENCH_WORKSTATION_V1.md). New preparation contract; not a ready legacy server config.

# 文档入口

- [当前流程待配置、待实现与待验收清单](IMPLEMENTATION_GAPS_ZH.md) (2026-11-11)

Rebench 实验一入口：[工作站/集群交接](handoffs/EXP001_REBENCH_WORKSTATION_V1.md)。旧5016实验的[V3交接](handoffs/EXP001_SERVER_V3.md)仅保留历史兼容，不作为本次入口。
实际部署见 [容器部署](SERVER_CONTAINER_DEPLOYMENT_ZH.md)，已实现与待办见 [progress.md](progress.md)。

## 实验与原始需求

- [EXP001 定义](EXP001_RANDOM_STATIC_M0_ZH.md)：Random/static_m0、两档学习率、三波训练和最终测试。
- [共享资产](EXPERIMENT_ASSETS_ZH.md)：初始能力、指标、策略的保存与复用。
- [交接规则](handoffs/README.md)、[下一实验模板](handoffs/TEMPLATE.md)。
- [递进研究计划](RSI_Progressive_Experiment_Plan.md)：后续 V0–V4 与消融顺序。
- [数据准备规范](RSI_Data_Preparation_Codex_Spec.md)、[用户准备 prompt](CODING_DATA_PREPARATION_PROMPT_ZH.md)。
- [技术蓝图](self_curate_codex_blueprint.md)、[数据中心架构](data_centric_rsi_architecture.md)、[实现决策](decisions.md)。

研究原文保留，不把其中历史示例命令、阶段数和“未实现”当成当前执行定义。

## 模块接口

按 [代码阅读教程](CODE_READING_TUTORIAL_ZH.md) 阅读：

| 模块 | 文档 |
|---|---|
| M1 数据与标签 | [数据接口](m1_interfaces.md)、[算子取值](m1_operator_data_contract.md)、[外部标注](enrichment_interfaces.md) |
| NLL/entropy | [计算](SIGNAL_COMPUTATION_ZH.md)、[定向刷新](FOCUSED_SIGNAL_REFRESH_ZH.md) |
| M2/P1 | [算子接口](m2_interfaces.md)、[精确清单交接](p1_handoff_interfaces.md) |
| P0 | [契约接口](p0_interfaces.md) |
| Agent 与反馈 | [三角色](THREE_ROLE_AGENTS_ZH.md)、[Leval→Lopt](feedback_lopt_interfaces.md) |
| 内层与验证 | [搜索循环](INNER_SEARCH_ZH.md)、[两类 Leval](LEVAL_BACKENDS_ZH.md) |
| 通用外层 | [训练、验证和判断](OUTER_TRAINING_ZH.md)；与 EXP001 专用流程区分 |

旧审计合集、旧就绪/执行总览、退役 handoff 和实现前草稿已从正式目录删除。
原始字节与清单保存在本地被忽略的 `runs/redundancy-cleanup-20261109/`，不作为执行入口。
