# NLL / entropy：首轮初始化与后续定向刷新

当前实现：0.12.0，2026-11-07。根据用户最新定义实现精确刷新，不在这里替 Lopt 推断技能方向。

## 1. 轮次与范围

| 时机 | 输入 | 行为 |
|---|---|---|
| 外层第 1 轮开始 | 初始模型、完整候选池、空的外层使用历史 | 全池初始化 NLL/entropy；缺标签也计算默认 assistant 指标 |
| 外层第 2 轮及以后开始 | 当前已接收模型、Lopt 技能方向、累计外层使用历史 | 只处理技能方向并集内、从未参与外层真实任务子集 SFT 更新的数据 |
| 后续轮尚未提供方向 | `focus=None` | 返回 `awaiting_focus`，不调用模型，不擅自全量刷新 |
| 明确提供空方向 | `skills=()` | 成功的无工作结果，不调用模型 |
| 外层存在无法确认的部分更新 | 历史含 `unresolved` | 返回 `blocked_outer_exposure`，保留输入，不调用模型 |

“完整候选池”是已经由数据准备/质量准入模块提供的池，不在本模块下载、标注或补齐数据。输入仍须满足 M1 轨迹/tokenizer 契约。

**未使用 = 从未参与本实验参数分支的外层真实任务子集更新。** 内层筛选、NICE、内层候选 SFT 不算。外层子集训练完成但尚未判断、或随后退回，都算已使用。筛选轨迹 SFT 的控制语料不冒充原任务池样本。相同内容改 ID 仍算已使用；同一 ID 换内容直接报错，不混淆版本。

方向是“筛选哪些样本来刷新”，默认指标仍是这些样本完整上下文中 assistant 监督位置的 NLL/entropy，不会自动变成仅对某技能片段计算。可显式传 skill_evidence spec；缺证据仍返回缺失，不能填零。

## 2. 公开接口

从 `self_curate.signals` 导入：

- `SkillFocus(outer_round, model_ref, skills, feedback, rationale)`：后续流程/Lopt 提供的方向输入；技能必须属于传入的 ontology。这里只校验，不生成方向、不添加 LLM 调用。
- `FocusedRefreshService(scorer, *, branch_id, known_skills, output, specs=None, max_refresh_records=None, max_context_tokens=None)`：复用已有 `SignalService` 和外部数据/前向后端。
- `.initialize(records, *, history)`：首轮全量初始化。
- `.run_round(records, *, outer_round, history, focus=None)`：后续刷新。
- `.plan(...)` 与 `.execute(plan, records, *, history, focus=None)`：先审阅计划再执行；输入、版本、方向或使用历史改变后旧计划不能执行。
- `load_refresh_result(path, *, expected_model, expected_computation_hash, expected_plan_hash, expected_history_hash)`：固定预期身份，核对完整输出与精确增量。
- `focus_from_outer_decision(context, decision, *, outer_round, skills, rationale)`：可选的外层反馈桥接；终态接收选新参数，退回选父参数。技能仍由调用方给出。不能从待补证据判断生成方向。

`max_context_tokens` 按本次需要刷新的完整输入上下文 token 总数预检，`max_refresh_records` 按需处理行数预检；不通过时整批停止，不截断轨迹或偷偷抽样。这是保守上界；实际前向费用以 ledger 为准。两项为可选保护，不代填研究预算。

从 `self_curate.data` 导入：

- `OuterExposureHistory(branch_id, backend)`：分支内累计外层使用历史；按 ID/内容内存查询，不是持久化 operator 索引。
- `.record(outer_handoff, training_outcome)`：从固定外层清单和已完成训练结果登记；验证精确成员、顺序、模型、batch、token/step 数及回执。
- `.was_used(record)`：按 ID 和内容判断；`.fork(new_branch_id)` 继承公共历史，之后独立累积。
- `.persist(root)` / `load_outer_exposures(path, expected_hash=..., expected_branch=...)`：只增版本的 checksum 快照，旧版本不覆盖。

已有 `OuterCycle(..., outer_exposures=history)` 可自动登记已完成的子集 SFT，先于验证/接收判断；通过 `cycle.outer_exposures` 取得更新历史。所有轮次必须传递这份累计历史，不能第二轮重新构造空历史。旧调用方可以不启用，但那样不能宣称具有自动外层使用追踪。

## 3. 装配示例

以下是 Python 接口示意，变量由之后的实验控制入口提供，不是当前 CLI 命令：

```python
from self_curate.data import OuterExposureHistory
from self_curate.signals import FocusedRefreshService, SkillFocus

history = OuterExposureHistory(branch_id, backend)  # 仅全新分支首次创建
initial_refresh = FocusedRefreshService(
    initial_scorer, branch_id=branch_id, known_skills=skill_names,
    output=run_root / "initial-signals",
)
initial = initial_refresh.initialize(candidate_records, history=history)
# initial.records：全池 M1 记录，已经附加版本化信号；原记录不变。

# 新内层搜索 -> 固定历史最佳 -> 外层：
# outer = OuterCycle(..., outer_exposures=history)
# 外层执行与接收判断结束后：
history = outer.outer_exposures

# 外层结果反馈后，未来 Lopt 流程提供方向，accepted_scorer 绑定最终接收参数。
direction = SkillFocus(
    outer_round=2, model_ref=outer.accepted_model,
    skills=tuple(lopt_skill_names), feedback=outer_feedback_ref,
    rationale=lopt_rationale,
)
next_refresh = FocusedRefreshService(
    accepted_scorer, branch_id=branch_id, known_skills=skill_names,
    output=next_run_root / "focused-signals",
)
result = next_refresh.run_round(
    initial.records, outer_round=2, history=history, focus=direction,
)
if result.status == "completed":
    all_records = result.records
    # 明确需要限制下轮候选范围时，用绑定当前模型的新模板创建新 context：
    new_context = result.focused_context(current_model_template)
```

多轮时传递最新的全池 `result.records` 和累计 history。`current_model_template` 的 ModelRef 必须已经是当前接收参数；质量门禁/预算由控制层给出。`focused_context` 保留这些策略，只把 view 限制为当前 eligible 样本并使用当前 raw SignalSpec；不会自动重建 CDF。若需要完整池，使用 `result.metric_view`，此时范围外样本仍可能没有当前模型指标，operator 必须按缺失处理。

初始化 M0 也可先通过共享资产模块复用；同一模型/输入/mask/spec/计算身份的完整指标直接命中，不重复前向。新 checkpoint 不使用旧版本替代。既有资产桥接仍有效，本模块没有新增自动的实验资产注册排程。

## 4. 版本、落盘与费用

每次输出独立目录：`plan.json`、`history.json`、原始全池 `inputs.jsonl`、回填后的全池 `records.jsonl`、`result.json`、`checksums.json`；实际执行时另含可严格重载的 `scoring/` 快照。只在完整写出后发布目录。

计划包含 branch/round、模型和 adapter 身份、pool/历史/focus hash、spec、计算身份、ontology、预算及 eligible/refresh/cached/used/outside/missing_skill 的 ID。结果保留旧版本，只对本次请求样本增加指标及其 forward 使用回执；质量、技能、梯度、原训练历史不改写。范围外和外层已使用样本保持原样。

完全命中缓存时不记新物理模型费用；实际调用的资源/失败费用保留在 CostLedger。失败保留部分记录与 `failure.json`，不会发布完整成功目录，也不会用零覆盖缺失。可审阅后将部分记录作为同一模型/方向/历史的下一次输入，已完成指标按精确身份复用；这不等于自动调度恢复。

外层训练报告部分消耗或消耗未知时，`OuterCycle` 留下未决使用状态，禁止后续刷新。需要通过可验证训练回执确定实际更新范围；不从 token 总数猜受影响的 ID。当前外层尚无崩溃自动恢复：进程异常退出时必须审计 `training_started` 等日志与训练产物，不能从旧的空使用快照直接推进下一轮。

## 5. 与后续模块的边界

- 不修改正在运行的内层 context、Team 或 RoleRuntime。外层参数改变后，控制层重建这些对象和目标评价命名空间，再开始新内层循环。
- 不实现 Lopt 技能推荐、近似刷新、全池强制更新、目标模型加载或数据准备。
- 不改变 V0 固定教材定义。后续配置由实验控制层显式决定是否调用刷新；V0/Random 等关闭路径仍需实现。
- 尚需统一配置 runner、跨轮装配/恢复、真实模型和任务适配，以及正式配置/handoff。当前模块通过 CPU/mock 验收不表示已有真实实验结果。

相关契约：[指标计算](SIGNAL_COMPUTATION_ZH.md)、[共享资产](EXPERIMENT_ASSETS_ZH.md)、[外层训练](OUTER_TRAINING_ZH.md)、[执行总览](README.md)。
