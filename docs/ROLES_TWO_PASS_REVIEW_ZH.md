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
