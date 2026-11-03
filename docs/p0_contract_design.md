# P0 契约补齐：实现前设计

范围：补第 5 节缺失的五个核心契约及目标侧/成本引用；与已有 ProgramSpec、BatchManifest、ModelRef、样本 GradientFeature 对接。
不实现搜索、resume、训练、模型调用、奖励计算或梯度估值算法。原 M1/M2 schema 与 m2-v2 算子语义保持不变。

## 类型与依赖

- 任务/执行：TaskBudget、TaskRecord、RolloutRecord。任务绑定环境、评分器、split、预算；大 token/工具数据用 ArtifactRef。
- 成本：ResourceUsage、CostEvent、CostRef、CostLedger。事件 ID 唯一，引用同时绑定选中事件的内容 hash；缺失计量不补零，失败事件保留。
- 目标侧：TargetGradientSpec/Feature。引用现有 GradientSpec 坐标，另绑定技能组、任务集合、rollout 集合、奖励/advantage/聚合配置和 reward_ascent 方向；pending/invalid/zero 显式区分。
- 报告：ObjectiveSpec、SkillWeight/Constraint、GradientCoverage、EvaluationReport、CapabilityReport。固定目标有 hash，估值与能力报告分离，样本特征按 cache_key 引用，统计口径显式声明。
- 外层交接：TrainingRecord、RoundRecord。只保存训练/决策记录，约束候选与起始模型、精确 training_data_hash、版本及配对开发报告；阈值由固定 policy artifact 引用，不代填。
- WorkflowBundle + validate_workflow：只读一致性检查，贯通 program→batch→样本/目标梯度引用→报告→round。可另传 M1 MetricView 核验真实 token/mask。

所有新根契约含 schema_version 与 backend=mock/real；mock 不可伪装成 real。最终测试任务禁止进入优化 bundle，独立 TaskRecord/CapabilityReport 可以描述测试评估。
配置和序列化采用严格 JSON，拒绝未知字段和错误嵌套类型。保留旧模型/数据池/清单证据；不为本轮模拟声明真实算法通过。

## 验证边界

结构校验只核对记录关系、确定的聚合口径和固定目标加权/约束一致性，不从向量 artifact 计算 dot、不执行验收阈值。
pending/invalid 报告不能参与最佳可行选择；零目标梯度不可冒充有效目标；环境失败 reward=null。
P2 再实现历史最佳、成本扣减、恢复和原子提交。真实向量、任务环境和 artifact 文件内容由后续后端核验。
