# Self-Curate 代码阅读教程

适用版本：0.6.2；核对日期：2026-11-04。本文针对独立项目 `D:\py project\self-curate`，按当前实际实现讲解。原始蓝图描述研究终态；了解当前进度先看 [progress.md](progress.md) 顶部，再读 [THREE_ROLE_AGENTS_ZH.md](THREE_ROLE_AGENTS_ZH.md) 和 [ROLES_TWO_PASS_REVIEW_ZH.md](ROLES_TWO_PASS_REVIEW_ZH.md)。沿本教程开展的代码复查与修复见 [TUTORIAL_FLOW_REVIEW_ZH.md](TUTORIAL_FLOW_REVIEW_ZH.md)。

## 1. 先记住系统在处理什么

项目要让模型根据反馈，修改“下一批训练数据怎么选”的程序。内层筛选与比较时冻结模型；外层最终才考虑对最佳实际批次训练。

目前已经实现数据管理、确定性筛选、跨模块契约、外部验证结果转反馈，以及三个角色共享客户端和隔离上下文。真实数值验证、梯度计算、SFT、完整搜索与接受/拒绝状态机仍待后续实现。

先沿下面这条已实现的链阅读：

```text
原始轨迹 → M1 规范化数据池 → MetricView 按 ID/版本取指标
    → ProgramSpec + ExecutionContext → M2 确定性筛选
    → BatchManifest → P1 PreparedSelection → 外部验证接口
```

再阅读反馈接续：

```text
已选父批次的 ValidationRequest + 外部 ValidationResult
    → Leval 整理事实与解释
    → Lopt 提出完整子 ProgramSpec
    → Lexec 调用已有 P1 执行子程序
    → 可行候选批次 + 下一次 ValidationRequest
    → 等待外部验证
```

第二条链从“父批次已经有验证结果”处开始，所以调用顺序是 Leval → Lopt → Lexec。它与蓝图里的 Lopt → Lexec → Leval 循环一致，只是入口位置不同。生成候选尚不能证明改进。

## 2. 目录只需要先认识这几块

| 目录/入口 | 负责什么 | 第一遍怎么读 |
|---|---|---|
| `configs/` | 数据导入、筛选程序、冻结上下文、目标、编辑政策、角色 prompt | 先读 fixture JSON，看到真实输入长什么样 |
| `src/self_curate/contracts/` | 对象结构、身份和严格校验 | 按下面的对象顺序查，不必整目录通读 |
| `src/self_curate/data/` | M1：规范化、数据池、指标视图、外部特征挂接、使用历史 | 先 `pool.py`、`view.py`，再看规范化细节 |
| `src/self_curate/operators/` | M2 筛选和 P1 正式交接 | 先 `runtime.py`，再 `handlers.py`、`preparation.py` |
| `src/self_curate/agents/` | 反馈、提案、三角色编排、共享客户端、独立记忆和发布 | 先 `demo.py`、`roles.py`，再看 runtime/history |
| `src/self_curate/cli.py` | 命令分发 | 用命令名定位函数，不必先通读全部分支 |
| `tests/` | 正常行为、缺失数据、过期版本及非法跨模块交接的例子 | 每读一个模块，跟读对应测试 |
| `runs/` | 本地运行产物和历史验收证据 | 对照 JSON 理解真实输入输出；不用提交 Git |

M1/M2 是数据与算子模块名；P0/P1 是契约与交接阶段名。它们不是四套独立运行系统：P0 为各模块规定交接格式，P1 把 M2 的实际结果包装成下游能核验的输入。

## 3. 第一步：运行一个例子，再读代码

在 VS Code 打开 `self-curate` 项目，用终端执行。当前核对使用的默认 Python 尚未安装本包，因此先指定本地 `src`，无需为了看示例下载依赖或权重：

```powershell
Set-Location 'D:\py project\self-curate'
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -B -m self_curate --help
python -B -m self_curate roles-demo --project-root . --prompts configs/roles --output runs/learn-roles-01
```

输出目录必须不存在。第二次运行换成 `runs/learn-roles-02`，不要覆盖历史产物。若已经在项目虚拟环境完成 editable install，可直接运行命令，无需设置 `PYTHONPATH`。

这个 demo 使用显式 `FixtureRoleClient`，无真实模型调用。两次反馈接续之间保存并恢复角色 history，正常路径共六次模拟模型调用。第二次外部验证结果也是程序明确构造的 fixture，不能用于推断科研效果。

按顺序查看输出：

1. `summary.json`：看到 `backend=mock`、`real_model_called=false`、`improvement_verified=false`，以及两次 `awaiting_validation`。
2. `parent/program.json` 与 `candidate-1/program.json`：筛选策略发生了什么修改。
3. `parent/batch.json` 与 `candidate-1/batch.json`：实际成员、分支和训练顺序怎样变化。
4. `evaluation-1.json`、`proposal-1.json`、`next-validation-1.json`：反馈、修改建议和下一次验证请求如何传递。
5. `client_calls.json`：同一客户端如何按角色接收不同 prompt、work 和 history。
6. `history-1/` 与 `history-2/`：各角色分别留下哪些工作记录。

已经有一份可直接阅读的历史示例：`runs/roles-two-pass/roles-demo/`。只想理解结构时，可以先读它，不必重新运行。

**这一阶段的自测：你能否指着文件说清楚“策略改了”“成员改了”“效果已改善”分别需要什么证据？目前最后一项并未由 demo 证明。**

## 4. 第二步：搞清楚六类核心对象

| 对象 | 直观含义 | 源码 |
|---|---|---|
| `TrajectoryRecord` | 一条完整轨迹及其来源、mask、统计、标签、信号、梯度引用和使用历史 | [contracts/types.py](../src/self_curate/contracts/types.py) |
| `MetricView` / `MetricBatch` | 给定 ID 和版本，按原请求顺序取一个或多个指标 | [data/view.py](../src/self_curate/data/view.py) |
| `ProgramSpec` | 筛选配方：DAG 节点、算子、输入依赖、参数、seed | [contracts/program.py](../src/self_curate/contracts/program.py) |
| `ExecutionContext` | 固定的执行条件：数据快照、模型、预算、质量政策和信号配置等 | [operators/context.py](../src/self_curate/operators/context.py) |
| `BatchManifest` | 实际选中哪些数据，以及选择顺序、训练顺序和对应身份 | [contracts/selection.py](../src/self_curate/contracts/selection.py) |
| `PreparedSelection` | 一次实际选择的交接包：状态、结果、成本及可用时的未估值工作流 | [operators/preparation.py](../src/self_curate/operators/preparation.py) |

例子：“筛 recovery 和 regression，按监督 token 各占一半”属于程序；“选中 data-00、data-01 等八条，按某顺序训练”属于批次。以后训练应该消费已经选中并评价过的精确 `BatchManifest`，不能拿同一个程序重新抽一遍替代。

`ModelRef` 是模型版本身份，实际权重由客户端持有。`ArtifactRef` 是文件位置与 checksum 引用。看到这两个对象，不代表模型已经加载或梯度已经计算。

`content_hash` 回答“数据内容是否相同”；模型/spec/快照/context 等身份回答“是否还在同一套计算条件下”。读校验代码时先问它在保护哪种身份，不必先研究 hash 算法。

## 5. 第三步：追踪一条数据，理解 M1

阅读顺序：

1. `tests/fixtures/m2_trajectories.jsonl`：只展开第一条，看 messages、技能与外部信号。
2. `contracts/types.py` 的 `TrajectoryRecord`：逐一对应规范化后的字段。
3. `data/normalize.py` 的 `normalize_trajectory`：看原始轨迹怎样变成规范记录。
4. `data/pool.py` 的 `build_pool` / `read_pool`：看数据怎样发布和重新核验。
5. `data/view.py` 的 `static_batch` / `signal_batch` / `gradient_batch`：看 operator 如何取值。

先记住三个规则：保留上下文，只监督 assistant payload；标签与指标按 ID 和内容/计算版本匹配；缺失值不当成 0。

可以在项目终端运行 Python，然后做这个只读练习：

```python
from self_curate.data import read_pool

view = read_pool('runs/roles-two-pass/roles-demo/pool').metric_view()
ids = ('data-02', 'data-00')
batch = view.static_batch(ids, ('supervised_tokens', 'skills'))
print(batch.data_ids)
print(batch.column('supervised_tokens'))
print(batch.column('skills'))
```

确认结果仍按 `data-02`、`data-00` 的请求顺序返回。以后 operator 按多列联合筛选，就依赖这个对齐保证。视图在内存里建立 ID/版本索引；批量查询遍历请求子集，不为每次查询扫描整个数据池。`view.data_ids` 只读；改变候选范围时构建新视图和上下文，不能原位重绑成员。

NLL/entropy 和梯度属于有版本的特征。信号读取需要匹配 `ModelRef`、`SignalSpec`，百分位还需要对应参考分布。梯度有更完整的坐标、loss、投影等身份。fixture 里的数值已预置；M1 负责保存、绑定、查询，不负责运行真实模型来算出它们。

使用历史也不是简单的 used 布尔值：前向评分、提取梯度、候选训练、接受/拒绝训练要分别理解。筛选出一条数据，不等于它已经接受训练。

详细查询例子见 [m1_operator_data_contract.md](m1_operator_data_contract.md)。对应测试先读 `test_m1.py`、`test_signal_interfaces.py`。

**自测：把两个 ID 交换顺序，指标会不会错位？模型版本变了，为什么不能直接沿用旧 NLL？只有“有梯度引用”，是否等于引用文件已经验真？**

## 6. 第四步：看一次筛选怎样执行，理解 M2

先打开 `configs/programs/m2.fixture.json`，使用 VS Code 格式化 JSON，再对照 `operators/runtime.py` 的 `execute_program`。

```text
固定 QualityGate
  → 按 DAG 依赖顺序执行各节点
  → 每个节点记录输出 ID、排除原因和 trace
  → 固定 BudgetPack
  → SelectionResult + 可行时的 BatchManifest
```

八类可编排算子是 `SkillSelect`、`EvidenceSelect`、`NLLSelect`、`EntropySelect`、`LengthSelect`、`DedupSelect`、`SampleSelect`、`MixAllocate`。第一遍只精读 `LengthSelect` 和 `SampleSelect` 对应 handler，理解“输入 ID → 读取指标 → 排序/过滤 → 输出 ID”，再看分支混合。

重点区分程序可修改的筛选参数与固定 context 中的约束。`MixAllocate` 的配额按监督 token 计算，不能普遍解释为样本条数占比。示例每条长度相同时，二者才刚好对应。

这份合成 fixture 的基准示例是 12 条输入，得到 8 条完整轨迹、64 个监督 fixture token。不能为了凑预算裁掉轨迹末尾。组批采用确定性贪心；返回 infeasible 表示这次执行未得到合法批次，不能自动推断所有组合都无解。

检查结果时优先读 `trace.json` 与 `batch.json`。它们分别回答“为什么这样选”和“最终选了谁”。对应测试读 `test_operators.py`、`test_m2_review.py`。

**自测：某条数据没被选中，你能在哪个节点找到原因？程序合法但结果不可行，为什么不矛盾？**

## 7. 第五步：理解 P0/P1，别把校验当执行

读 `operators/preparation.py` 的 `prepare_selection`，再读 [p1_handoff_interfaces.md](p1_handoff_interfaces.md)。

P1 复用 M2 的实际执行结果，加入选择成本和身份关系，形成下游可以核验的交接。还没运行的梯度、估值、训练等字段保持为空，不填一套看似完整的模拟结果。

P0 在 `contracts/workflow.py`、`workflow_validation.py` 中定义并校验模型、目标、任务、报告、批次、成本与轮次关系。`validate_workflow` 通过，只说明这些记录满足所检查的约束；没有执行真实估值、训练或验收。

重要接口区分：

- `prepare_selection` / `prepare_to_directory`：实际执行一次选择。
- `publish_preparation`：发布已经得到的结果，不重新选择。
- `load_preparation`：读取已有结果并核验交接，保留原成本。
- `verify_preparation`：校验回执；阅读时留意它与完整加载的区别。

以后消费历史最佳，除了模型/目标/context，还要用系统保存的 `expected_batch_id` 锁定精确成员与训练顺序。选择重放用原始快照；后续追加梯度后的视图可用于消费固定清单，不能把快照变化当作同一次新选择。

对应测试读 `test_p0_contracts.py`、`test_p01_adaptation.py`。

## 8. 第六步：最后再读三个 Agent

先读 `agents/demo.py` 的 `run_role_demo`，看到真实对象怎样组装；接着读 `agents/roles.py` 的 `RoleTeam.feedback_step` 和 `_feedback_step`。再向下追三个角色：

| 角色 | 主要入口 | 代码负责把关什么 |
|---|---|---|
| Leval | `EvaluationAgent.evaluate` → `leval.build_feedback` | 校验外部结果，整理确定性数值事实；模型只增加解释和假设 |
| Lopt | `OptimizationAgent.optimize` → `lopt.propose_program` | 生成完整子程序；核验引用、父子关系、实际 diff 和编辑政策 |
| Lexec | `ExecutionAgent.execute` → `prepare_selection` | 核对给定 program/context 身份，再执行原筛选代码 |

这里的 Agent 是模型决策与工具调用的角色封装。语言模型不直接替代 operator，不改写外部数值分数，也不能随意运行生成脚本来绕过约束。

随后再读共享与隔离：

1. `agents/client.py`：三个角色共用 `SharedModelClient.generate_json` 协议。
2. `agents/runtime.py`：同一客户端、同一 `ModelRef`，逐次调用检查身份并记录成本。
3. `agents/prompts.py` 与 `configs/roles/*.json`：固定格式自动组装，具体 `instructions` 正文当前可为空。
4. `agents/history.py`：各角色自己的 before/after 事件；默认最近 8 项工作进入上下文，完整记录仍保留。这不是 8 个完整 cycle，也不是 token 限额。
5. `agents/scope.py`：一个业务会话内固定 context、objective、measure、protocol 与编辑政策，避免边做边换评价条件。

跨角色通过明确产物传递。例如 Leval 的反馈和解释可交给 Lopt，但不自动复制 Leval 的整段 prompt/history。真实客户端也要按每次完整请求生成，不能自己维护一个三角色混在一起的全局对话。

对应测试按 `test_role_runtime.py` → `test_role_boundaries.py` → `test_roles_pass2.py` 阅读。先看正常测试的输入与断言，再看防止越权修改和版本错配的失败测试。

## 9. 保存与交接有两个入口

`team.save(...)` 保存角色会话和 history；`team.publish_cycle(cycle, ...)` 发布完整单步业务产物，包括父请求/结果、固定政策、反馈、提案、实际候选 P1 目录及可行时的下一次验证请求。后者通过 `load_role_cycle(...)` 加载。

demo 自己发布了候选目录和摘要；完整 cycle 的较新接口示例可直接阅读 `runs/roles-two-pass/typed-cycle/`，对应会话在 `typed-history/`。不要把 demo 的 `cycle-1.json` 摘要误认为整个可移植业务交接目录。

`runtime.ledger()` 统计会话里的物理模型调用；发布 cycle 的 `ledger.json` 汇总该 cycle 模型调用和本次候选选择成本。它们均不是完整外层科研实验的总成本。工具 history 的 span 可能包含子调用，不能再次全部相加。

会话快照需要显式保存，不是每条事件自动刷盘；这套能力也还不等于完整 P2 搜索恢复。

## 10. 建议按三个学习阶段推进

1. **先掌握一条数据和一次筛选。** 跑 mock，读数据记录、程序、trace、batch；用自己的话解释成员怎样产生。
2. **再掌握一次反馈接续。** 读父请求/结果、反馈、提案、候选与 next request；沿 `RoleTeam._feedback_step` 单步调试。
3. **最后掌握跨模块约束。** 读对应失败测试，理解缺失、版本变化、精确批次、独立 history、scope 和成本为什么要校验；再讨论 P2 调度与真实后端接入。

调试时建议依次给 `run_role_demo`、`RoleTeam._feedback_step`、`build_feedback`、`propose_program`、`execute_program`、`prepare_selection` 下断点。前两个帮你看整体；后四个按执行进度逐个启用，避免一开始停得太碎。

每读一个函数，只记四件事：输入对象、读取/校验的身份、实际做了什么、返回什么。先沿正常路径走通，再看异常分支。独立学习过程中修改配置请使用副本和新输出目录；修改研究代码后才需要按项目要求跑全量测试。

最终应能讲清楚：一条轨迹如何变成可查询数据；程序如何选出精确批次；外部结果如何变成反馈；Lopt 怎样产生受约束的下一版程序；三个角色怎样共享模型却各自记忆；候选为何仍需等待验证。掌握这些，就已经理解当前实现的主干。
