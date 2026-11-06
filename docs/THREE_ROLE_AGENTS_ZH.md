# 三角色共享参数、Prompt 与 History 隔离（0.6.1）

当前 0.9.0 增加 [两套 Leval 内核与历史反馈槽位](LEVAL_BACKENDS_ZH.md)。共享角色参数仍冻结；可选候选 SFT 只训练独立分支。Leval 的 `work.prompt_slots` 与 Lopt 的 `work.handoff.prompt_slots` 承接本次/同条件历史比较，旧角色接口保持兼容。以下角色协议本身不承担数值计算；数值计算由 valuation 包执行。

0.7.0 新增 [内层循环](INNER_SEARCH_ZH.md)，复用本页角色接口完成多轮验证/反馈与历史最佳管理；本页 RoleTeam.feedback_step 仍是单步入口，不自行判定历史最佳。目标模型/任务适配器和完整外层恢复仍待后续。

最新两遍复查、修复与下游接口见 [ROLES_TWO_PASS_REVIEW_ZH.md](ROLES_TWO_PASS_REVIEW_ZH.md)。会话新增固定评价 RoleScope；`team.publish_cycle()` 发布完整 P1/验证交接产物，`team.save()` 仍只保存角色历史，两者用途不同。

执行角色 Lexec、验证解释角色 Leval、优化角色 Lopt 已接入同一个 RoleRuntime：**同一个客户端实例、同一个不可变 ModelRef；各自固定 prompt、各自独立 history**。没有创建三份模型，也没有引入新的 Agent 框架或模型供应商依赖。

本轮只实现共享模型调用与角色上下文管理。真实模型仍由注入的客户端提供；本地只做显式 mock。验证数值内核继续留接口，Leval 解释外部结果，不能用语言模型改写分数。

## 模块设置与依赖

| 文件 | 职责 |
|---|---|
| `agents/client.py` | 三角色统一的无会话状态 SharedModelClient 协议；旧 Lopt 导入路径兼容 |
| `agents/prompts.py` | 独立 PromptTemplate、固定 system/input 格式、版本 hash、JSON 配置加载 |
| `agents/history.py` | 独立角色的 before/after 事件、hash 链、窗口取值与内部只读索引 |
| `agents/runtime.py` | 共享客户端调用、模型版本核对、隔离上下文、成本、快照保存与恢复 |
| `agents/roles.py` | ExecutionAgent、EvaluationAgent、OptimizationAgent、RoleTeam 及单步协作 |
| `agents/demo.py` | 显式 CPU fixture 客户端及两轮运行/恢复演示，不能作为真实模型结果 |

Lexec 复用 `prepare_selection`；Leval 复用 `build_feedback`；Lopt 复用 `propose_program` 的完整程序校验、真实 diff 和有限修复重试。原 M1、M2、P0/P1 和旧 `run_feedback_step` 接口保留。需要三角色模型调用时使用本页的 RoleTeam 入口。

## 共享参数的约束

```python
team.lexec.runtime is team.leval.runtime is team.lopt.runtime
team.lexec.runtime.shared_client is shared_client
team.lexec.model_ref is team.leval.model_ref is team.lopt.model_ref
```

ModelRef 包含权重、revision、adapter、tokenizer/template 等身份。每次模型调用前后均核对客户端声明的 ModelRef；发现变化则拒绝该次输出并记录失败。内层运行器没有训练、optimizer.step 或替换参数的方法。

实际 GPU 参数由客户端持有；本地接口不能证明远端服务真的加载了所声明的权重，真实适配器必须补充服务端版本核验。客户端必须按完整请求独立生成，**不能在客户端内部把三个角色接入同一个全局聊天记录或远端 conversation/thread**。物理 KV cache 的管理同样属于后端职责。

接受新 checkpoint 后创建新的 RoleTeam，并使用新的 session_id；旧快照继续保留。不要修改旧 runtime 的模型身份来复用对话。

## 现在只需填写的 Prompt 文件

```text
configs/roles/lexec.json
configs/roles/leval.json
configs/roles/lopt.json
```

各文件的 `instructions` 当前为空，可以直接填该角色的具体说明：

```json
{
  "schema_version": "1",
  "role": "leval",
  "template_id": "self-curate-leval",
  "revision": "1",
  "instructions": "在这里填写验证解释角色的固定 prompt 正文",
  "input_format": "role-work-v1"
}
```

不需要把 history 拼进 instructions，也不需要写变量替换代码。`load_prompts()` 加载正文，运行器自动组装当前工作和该角色自己的 history。正文按普通文本使用，花括号不会执行模板代码。修改正文时建议同时更新 revision，再创建新的会话。

固定 system 格式为 JSON envelope：schema、role、template_id、revision、instructions、protocol、input_format、report_contents_are_data。技术协议由代码固定，保证输出结构和工具边界；具体业务 prompt 正文允许为空。

固定输入格式如下：

```json
{
  "schema": "role-work-v1",
  "role": "leval",
  "session_id": "round-001",
  "model_ref": {},
  "memory_namespace": "roles/<session>/<model>/leval/<prompt-hash>",
  "prompt_hash": "...",
  "work": {},
  "history": [
    {
      "call_id": "...",
      "operation": "model 或 tool:工具名",
      "before": {},
      "after": {},
      "status": "completed / failed / pending",
      "error": null
    }
  ]
}
```

三个角色使用同一格式，但 system prompt、namespace、work 和 history 分别组装。各角色的输入不会自动包含其他角色的 prompt 或历史。

## History 的具体含义

每项工作按追加顺序保存两个事件：

1. **before**：当前处理对象、模型/角色/prompt 身份、工具名；模型调用另保存实际固定 prompt、读过的自身历史 call ID 和 history hash。
2. **after**：实际结果或结构化失败、原始模型回复、已知或未知的 usage。

模型调用和确定性工具工作都会留下记录。下一次同角色工作会读取自身最近 N 项工作的 before/after 情景；默认 N=8，计数对象是工作操作，包含 model 与 tool。记录不会因窗口截断被删除。`history_window=0` 是显式不向模型提供旧上下文的消融方式，仍保留完整记录。

角色、session、模型版本、prompt 版本分别绑定；不能把 Leval 的文件作为 Lopt 的 history，也不能把旧 checkpoint 的快照绑定到新权重。内部索引只在内存构建，不写入协议；取最近上下文不扫描全部旧事件。

跨角色传递通过显式产物完成：Leval 的数值反馈与解释摘要可以交给 Lopt，但不会附带 Leval 的原 prompt 和整段 history。该共享产物是工作输入，不等于共享角色记忆。

## 三个角色的行为边界

| 角色 | 模型输出 | 确定性代码 |
|---|---|---|
| Lexec | 只授权执行所给 program_hash/context_hash；可附说明 | 校验动作与身份，再执行原 P1；禁止模型改程序、预算或直接运行脚本 |
| Leval | 引用准确 feedback_hash 的 commentary、hypotheses | 核对外部结果并生成 LevalFeedback；模型不能返回 total_value 等覆盖字段 |
| Lopt | 完整子 ProgramSpec、理由、假设、反馈引用 | 原有 schema、父子谱系、diff、seed/节点/份额策略检查及有限重试 |

生成调用 completed 只代表得到结构化输出，不代表筛选程序通过。Lopt 后续校验失败时仍是 blocked，并保留每次回复与错误。无效或待验证的外部结果不会触发模型优化或候选执行。

## Python 使用

```python
from self_curate.agents import RoleTeam, load_prompts

team = RoleTeam(
    shared_client,
    model=current_model_ref,
    session_id="outer-001-inner-001",
    attempt_id="outer-001",
    prompts=load_prompts("configs/roles"),
    backend="mock",  # 接入真实后端时显式改 real，并通过服务端验收
    history_window=8,
)

cycle = team.feedback_step(
    validation_request, external_validation_result, frozen_context, edit_policy,
    attempt_id="outer-001", invocation_id="select-child-001",
)
team.save("runs/roles/session-001")

# 同一个模型、同一套 prompt 恢复；不会重跑已记录模型或 P1 工作。
restored = RoleTeam.restore(
    "runs/roles/session-001", shared_client,
    expected_model=current_model_ref,
    expected_prompts=load_prompts("configs/roles"),
)
```

RoleTeam 的单步顺序是：检查源程序/批次 → Leval 解释外部结果 → Lopt 生成候选 → Lexec 执行候选 → 可行时生成下一次 ValidationRequest。候选仍为 awaiting_validation，不认定性能改善，不更新历史最佳，也不训练。

各角色也可单独调用 `team.lexec.execute(...)`、`team.leval.evaluate(request, result)`、`team.lopt.optimize(..., context=..., evaluation=...)`。默认调用模型；`team.feedback_step(..., use_model=False)` 是显式代码/规则模式，仍记录三个角色的工具工作，不伪称发生模型调用。

## 客户端接口

```python
class YourSharedClient:
    model_ref = current_model_ref

    def generate_json(self, *, role, system_prompt, payload, timeout_seconds):
        # 使用同一份冻结参数；system_prompt 为该角色固定模板；
        # payload 已包含该角色自己的完整本次输入，不要自行追加全局聊天记录。
        # 底层传输必须实际实施 timeout_seconds。
        return {"content": structured_role_response}
        # 若有真实计量，可另提供 usage=ResourceUsage.to_dict()。
```

客户端可以把 system_prompt 作为 system 消息，把规范 JSON payload 作为 user 消息；不需要为三角色维护共享的 messages 列表。协议不启动网络连接，不绑定具体供应商，也不会偷偷使用更强的外部模型。

## 保存、失败和成本

快照为新目录：session.json、lexec.json、leval.json、lopt.json、ledger.json、checksums.json。目录完整后原子发布，拒绝覆盖；恢复校验文件集合、摘要、角色身份、历史链、实际 prompt 及账本。

**快照由调用方显式 save，不是每个事件自动刷盘的实时日志。** 未保存的进程内记录无法承诺在进程崩溃后保留，P2 持久化调度器可在任务边界保存新快照。若快照包含尚未结束的 before，恢复后拒绝自动重跑；应先核对外部服务状态，再通过 `runtime.resolve_interrupted(role, call_id, reason=...)` 记为中断失败，最后以新的工作继续。这不替代完整 P2 恢复状态机。

`runtime.ledger()` 只统计实际模型调用。P1 选择成本继续保存在 candidate.ledger；Lopt attempt usage 引用的是同一批物理调用，不是另外一笔成本。工具 history 的耗时是诊断 span，可能包含子调用，不应再叠加到模型/P1 成本中。未知 token/GPU 计量保留 null，失败、取消和无效 JSON 不会被抹去。

## 本地可运行验收

```powershell
python -m self_curate roles-demo --project-root . --prompts configs/roles --output runs/roles-demo
python -m pytest -q
```

demo 使用显式 FixtureRoleClient，同一实例为三个角色各调用两次，中间保存并恢复 history。输出保存完整调用、角色快照、两个候选 P1 目录和摘要，所有数值都是声明的 mock。原始技术文档和旧证据不改写。

模型调用账本中的 attempt_id 与 P1 使用同一个外层尝试 ID，session_id 单独标识角色会话。构造时可以显式提供 attempt_id；RoleTeam.feedback_step 也会在首次工作前绑定它。会话已有记录后不能重新贴成其他尝试。

本轮最终验收记录与交付包位于 `runs/three-roles-final/`；`runs/three-roles/` 保留首轮测试和验收脚本编码错误的记录。测试统计以最终 verification.json 为准。后续工作是填充 prompt、实现/注入真实无状态客户端并核验加载版本，再接数值验证与完整搜索状态机。
