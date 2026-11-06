# 内层闭环与接口：0.9.1

0.9.1 的 [Leval 两轮复查](LEVAL_REVIEW_ZH.md) 固定真实执行身份和自动基线 namespace。恢复数值运行时使用同一参考状态；旧搜索/角色产物只读核验保持兼容，不能把新建 provider 当作旧基线恢复。

search 模块负责内层调度，退出条件仍待填。复用 M1/M2、P0/P1、三角色和 ValidationProvider 接口；0.9.0 的 [valuation](LEVAL_BACKENDS_ZH.md) 实现可切换 NICE 梯度估值/候选 SFT 任务增量。停止后的两阶段 SFT／任务验证／接收退回由外层模块承接，见 [OUTER_TRAINING_ZH.md](OUTER_TRAINING_ZH.md)。

## 现在的实际调用链

```text
初始 P1 批次 + ValidationRequest
  → 外部 ValidationProvider / submit_validation
  → 校验绑定、记录结果与成本、更新 current 和 best
  → Leval 事实与解释 → Lopt 完整子程序 → Lexec 执行原 P1
  → 新验证请求 → 外部验证 → ……
  → 调用方 StopCondition 决定结束
  → 历史最佳程序 + 已评价过的精确批次 + 原验证结果
```

初始程序必须已经通过 P1 产生可行批次。初始批次也会验证并参与最佳比较，不能直接假定它得分为零。最后一个结果停止时不再为提出新候选额外调用模型；其数值绑定已由确定性代码核验。

## 文件与依赖

| 文件 | 职责 |
|---|---|
| `search/contracts.py` | StopCondition/StopDecision、EvaluatedCandidate、SearchState/Result、验证尝试 |
| `search/inner_loop.py` | 多轮调度、当前/最佳、失败与成本、手动推进与自动循环 |
| `search/storage.py` | 完整快照、timeline 重放、检查点恢复、最佳精确清单交接 |
| `search/demo.py` | 显式 mock 值与两次角色 cycle、恢复和历史最佳演示 |

验证器仍使用 `agents.ValidationProvider.evaluate(request) -> ValidationResult`。后续可在其内部组合“核验精确批次 → 查询缓存 → 只补最终成员缺失梯度 → 固定估值器”，无需改写搜索主循环。返回值须对应请求的模型、batch、objective、measure、protocol 和 backend。内部任何缺失或失败不得填零。

选择仍使用冻结原始 MetricView/context；验证器可持有独立特征/缓存服务，不能原位改写该选择快照。重新选择所依赖的 NLL/标签快照发生变化时，须显式创建新 context/session，而非在当前搜索偷偷换版本。

## Python 接入

```python
from self_curate.agents import RoleTeam
from self_curate.search import InnerSearch

# initial: 已完成且可行的 PreparedSelection
# request: 与 initial 精确匹配的 ValidationRequest
# edit_policy: 已有固定 EditPolicy
# shared_client: 三角色共用、版本固定且每次调用无共享对话的客户端
team = RoleTeam(
    shared_client, model=request.model_ref,
    session_id="outer-001-inner-001", attempt_id=initial.attempt_id,
    backend=request.backend,
)
search = InnerSearch(team, initial, request, edit_policy,
                     search_id="outer-001-inner-001")

result = search.run(validator)  # 默认返回 awaiting_stop_condition，不产生新调用

# 退出条件尚未确定时，手动推进一项操作：
search.step(validator)         # 若在等待验证，调用验证器一次
search.step()                  # 若已有有效父结果，反馈/提案/执行一次
pending = search.state.pending_request
# 也可以由外部异步流程交回结果：
# search.submit_validation(external_result)
```

新搜索要求新的、尚无工作历史的 RoleTeam；已有搜索通过 restore 接续，不能在旧会话上重新开始计数。`use_model=False` 为显式规则消融模式，仍复用角色工具和 P1，不伪造模型调用。

## 退出条件的待填位置

`run(validator, *, stop_condition=None)` 的参数就是预留接口。没有默认最大轮数、停滞阈值或资源上限。未配置时返回 `awaiting_stop_condition`，内部待验证/ready 状态保持原样；`step()` 始终可用于有限的手动推进。

后续只需要实现并传入这个回调，不必修改循环：

```python
from self_curate.search import SearchState, StopDecision

def your_stop_condition(state: SearchState) -> StopDecision:
    # 待填写研究规则。
    # 满足时：return StopDecision(True, "记录实际退出原因")
    # 继续时：return StopDecision(False)
    raise NotImplementedError("请先填写退出条件")

# 填完后：
# result = search.run(validator, stop_condition=your_stop_condition)
```

回调可读取 evaluated_count（含初始批次的有效结果数）、candidate_attempts（完整 cycle 加失败角色操作）、validation_attempts、failed_operations、operation_count、stagnation_count、current、best、pending_request 和 ledger。零值与未知费用不能混淆；使用预算条件时需明确如何处理未知计量。

最佳比较使用严格更大的固定 total_value，平分保留先者；stagnation_count 在新最佳时归零，在非改善的有效结果或未形成可验证候选的完整 cycle 后增加。角色传输失败另计 failed_operations 并阻塞，不自动重试。以上是状态统计，不是默认退出阈值。

回调在操作边界检查，不能中途取消正在执行的同步后端；传输超时与单工具预算由真实后端实施。已有 EditPolicy 的有限 JSON 修复次数继续有效，它不等于内层搜索的停止规则。退出回调不交给 LLM 编辑，不从 JSON 执行 Python；恢复后由调用方再次提供。

## 验证、失败与最佳

- `valid` 且约束通过：可以更新 best；valid 但约束失败：可作为 current 给 Lopt 反馈，不能成为 best。
- `pending` / `invalid`：保留原结果，返回 awaiting_validation，不反复轮询，不触发基于假分数的提案。再次 step/run 或提交新结果是调用方的显式动作。
- 验证异常、错绑定或角色调用异常：保留证据与成本，返回 blocked。检查后调用 `retry(reason=...)` 产生新的物理尝试，或给待验证请求提交有效外部结果。
- 候选不可行、算子失败或提案 blocked 但 RoleCycle 完整：记录其状态，保留原已验证父批次，下一轮通过 previous_cycle 把执行信息交给 Lopt；不调用验证器给不存在的批次评分。
- 相同外部结果重复投递不重复计账；一个请求的有效最终结果不可被另一份结果覆盖。验证器应在内部完成其协议要求的重复测量，再交出最终结果。
- 搜索结束没有可行结果时返回 no_feasible_result，`require_best()` 明确失败，不伪造空批次。

`SearchResult.best` 保留原 PreparedSelection、ValidationRequest 和 ValidationResult，训练顺序与已评价批次完全一致。返回最后候选和返回历史最佳是两件事。

## 成本范围

账本合并初始 P1、各候选 P1、物理角色调用与本次调用验证器的费用。失败、无效 JSON 和验证失败保留费用；未知计量保留 null。外部 `ValidationResult.usage` 必须描述本次调用的增量费用，不应把同一个远程任务的累计费用反复交回。

`submit_validation(result)` 默认导入既有证据，只保留其原始 usage，不把读取结果当成一次新模型计算。若外部异步 worker 为本次搜索实际完成了新工作，调用 `submit_validation(result, cost_event=original_event)`，传入原始 CostEvent；它必须绑定当前 request_hash、ModelRef、外层 attempt、backend 和同一 usage。按原事件 ID 去重，相同结果再次投递不新增账，也不能换一份费用身份。validation_attempts 统计结果接入/验证尝试；实际已记账物理工作以 ledger 为准。

不重复叠加工具 history 的嵌套耗时、Lopt attempt 所引用的同一模型调用。checkpoint I/O 和控制器开销不包含在这些业务事件中，完整基础设施成本仍待后续 M9 汇总。

## 检查点与恢复

```python
search.save("runs/search/snapshot-001")

restored = InnerSearch.restore(
    "runs/search/snapshot-001", shared_client, original_view,
    expected_request=initial_request, expected_policy=edit_policy,
    expected_search_id="outer-001-inner-001",
    expected_session_id="outer-001-inner-001",
    expected_attempt_id=initial.attempt_id,
    expected_prompts=the_same_prompts,
    expected_use_model=True,
)
```

快照包含初始 P1、完整 cycles、验证尝试、失败记录、角色 history、timeline 和账本。恢复核验 checksum，再重放确定性状态，重新确定 best/current；不调用模型、验证器或新增业务成本。加载 P1 时会进行既有 CPU 重放核验，需要原始选择视图。

恢复省略 expected_prompts 时核对内置默认模板；使用自定义 prompt 时必须显式传入原模板。cycle 的解释和模型提案也会与实际 role history 对照，不能只改 cycle 文件后重新计算 checksum。

可选 `search.enable_checkpoints(new_directory)` 会在外部操作之前和之后写入新的完整快照。使用 `InnerSearch.restore_latest(directory, client, original_view, **上述锚点)` 读取最近完整版本；损坏的最近快照不会静默回退。恢复后若继续自动写检查点，请显式选择新的目录。

预写检查点失败时尚未派发外部工作，0.7.1 会回滚本次未启动操作的标记，保留原状态并抛出保存异常。若外部工作已经完成、仅事后保存失败，内存结果和费用继续保留，应补保存，不能重新调用后端冒充恢复；旧磁盘检查点仍可能保守标记为 interrupted。

稳定边界可以继续运行；操作进行中的快照恢复为 interrupted，不自动重发调用，也不能标成已经完成。必须独立核对外部任务和产物，后续再接完成对账协议。这一状态保留已知最佳，但不承诺失联外部任务的费用已完整结算。未启用自动检查点时，进程崩溃仅能保留最后一次显式保存的状态。此能力不是完整外层 M9 恢复。

## 给后续训练模块交接

调用方 stop 决策结束搜索后：

```python
result = search.result()
best = result.require_best()
search.publish_best("runs/search/best")

from self_curate.search import load_best
loaded = load_best("runs/search/best", original_view,
                   expected_request=best.request)
```

交接含 `prepared/` 标准 P1、request/validation、best_attempt、result、完整搜索账本及 checksum。用系统独立保存的最佳请求作锚点，不能仅信目录内部自洽。训练实际输入使用原 `prepared/workflow.json` 和保存的 training_order。追加梯度后消费清单沿用 P0 的 load_workflow(view=enriched_view, expected_model=..., expected_objective=..., expected_selection_context_hash=..., expected_batch_id=...)；不要用新特征快照冒充原选择重放。

本轮外部验证结果不被伪装为已计算两侧梯度的 EvaluationReport；正式梯度验证器之后需提供可追溯的真实报告和向量证据。

## 可运行 mock 示例

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -B -m self_curate inner-demo --project-root . --output runs/inner-demo
```

输出目录必须不存在。示例明确声明三个外部 mock 值 0.1 → 0.3 → 0.2；两个三角色 cycle，中途保存/恢复，最后交出 0.3 对应的原批次，而非最后的 0.2 批次。为让示例有限结束，demo 自己传入“三个声明值后结束”的回调；生产退出条件仍为 None。所有数值仅为工程验收，无真实模型、梯度或学习收益证明。

查看 summary.json、result.json、checkpoint-1/、checkpoint-final/ 和 best/。测试入口为 `tests/test_inner_search.py`，全量测试与验收记录位于忽略的 `runs/inner-search/`。

本轮内层专项 30 项和隔离副本全量 420 项通过。最终交付验证另覆盖自动循环与独立手动串行参考一致、稳定边界恢复不重复调用、退出条件缺省不发起工作、历史池/角色快照兼容、旧证据未改写和源码包 checksum；以 verification.json 实际记录为准。
