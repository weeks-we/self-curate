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
