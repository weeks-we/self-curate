# P0 核心契约与跨模块交接

当前包 0.4.1；新增契约 schema_version=1，校验语义 p0-v3；M1 数据格式和 M2 的 m2-v2 语义保持不变。
实现前设计见 [p0_contract_design.md](p0_contract_design.md)。此前审计中的五个缺失核心契约均已补齐。

## 能力与边界

P0 冻结可序列化的数据结构、严格 JSON 边界和跨记录身份关系。
本轮不执行模型、训练、梯度计算、代理 dot product、接受阈值或原子提交；样例中的奖励、向量、估值和 accept/reject 均是**声明的 mock 记录**。
P2 仍需实现搜索状态、历史最佳、预算扣减、两次真实的模拟尝试及 resume。

## 公共入口

```python
from self_curate.contracts import (
    TaskRecord, RolloutRecord, EvaluationReport, CapabilityReport, RoundRecord,
    TargetGradientSpec, TargetGradientFeature, CostEvent, CostLedger,
    WorkflowBundle, parse_contract, load_workflow, validate_workflow,
    task_set_hash, rollout_set_hash,
)

bundle = load_workflow("configs/p0.fixture.json")
receipt = validate_workflow(bundle)  # 检查记录身份和声明的一致性
```

`parse_contract(kind, dict)` 只接受固定注册的类名，不解析代码或外部 Python 模块。
另提供 parse_task_record、parse_rollout_record、parse_evaluation_report、parse_capability_report、parse_round_record、parse_workflow_bundle。
所有新类型均可通过 `.to_dict()` 序列化；`.record_hash` 绑定规范化内容。JSON 严格拒绝未知字段、错误类型、缺必填项和不支持的 schema_version。
显式 JSON 中每个新契约的 schema_version 必填；构造 Python 对象时默认值为 1。数值缺失用 None/null，不填零。

## 任务与执行记录

| 类型 | 主要字段 |
|---|---|
| TaskBudget | max_steps、max_output_tokens、max_context_tokens、max_wall_seconds，均为显式正预算 |
| TaskRecord | task_id/version、skills、payload/environment/scorer 的 ArtifactRef、split、budget、backend、family_ids |
| RolloutRecord | rollout_id、ModelRef、task_id/task_hash、seed/sampling、actions/observations、tokenization_hash/action_mask_hash、generated_tokens、reward、status/stop_reason、scorer、cost、backend |

TaskRecord.task_hash 包含任务、环境、评分器、split 和预算。RolloutRecord 必须匹配这份任务版本和评分器。
actions/observations 引用应分别保存真实动作 token/生成位置及完整工具上下文；P0 只核对引用与 hash 声明，不读取这些大文件。
status 支持 completed/task_failed/budget_exhausted/environment_failed/cancelled。
completed/task_failed 必须有有限 reward；环境失败/取消必须 reward=null；预算耗尽允许保留已评分或尚未评分的真实状态，并保存 stop_reason。

TaskRecord 可独立描述 train/development/test。优化用 WorkflowBundle 禁止任何 test 任务；目标梯度仅绑定 development 技能任务组。
这项契约检查不代替真实来源家族、近似题面和原始训练数据的隔离审计。

## 成本与缺失计量

| 类型 | 用途 |
|---|---|
| ResourceUsage | 复用 ComputeCost 的墙钟/GPU 秒/处理 token；另分 Agent 输入输出、评分 token、新增梯度样本/token、缓存命中、工具调用和训练监督 token |
| CostEvent | 唯一 event_id、attempt_id、operation、subject_id、可选 model_hash、执行状态、usage、receipt、backend |
| CostLedger | 某个记录范围的事件集合；禁止重复 event_id，失败/取消事件不会因结果被拒绝而消失 |
| CostRef | event_ids 加选中事件内容的 events_hash，避免事件 ID 相同而计量被替换 |

`ledger.reference(tuple_or_list_of_ids)` 生成 CostRef；`ledger.summary()` 按维度给出 known_subtotal、missing_events、total。
任一事件该维度未知时，total 为 null；显式已知零可保留为 0。空账本表示尚无事件，累计为 0，不代表执行过某项计算。
多份报告可引用同一事件，账本按事件唯一累计，不对报告引用重复相加。
P0 不执行预算预留、扣减或重试调度。P2 应复用事件 ID 做幂等记录；新的物理尝试用新的事件 ID。

## 样本侧与目标侧梯度

现有 GradientSpec/GradientFeature 继续保存样本级向量引用，缓存身份不变。
TargetGradientSpec 另声明：skill、生成动作的 GradientSpec、reward_config、advantage_config、aggregation_config，以及明确的 reward_ascent 方向。
TargetGradientFeature 保存 ModelRef、task_ids/task_set_hash、rollout_ids/rollouts_hash、vector、dimension/dtype/norm、validity、reason、cost、backend。

- `task_set_hash(tasks)`、`rollout_set_hash(rollouts)` 对唯一 ID 和完整版本作稳定绑定。
- target_version 绑定目标配置、模型、任务/rollout、向量与有效性；不能跨 checkpoint 或奖励/聚合版本复用。
- valid 目标必须有非零有限 norm、向量和计算记录；zero 必须明确 norm=0 且带理由；invalid/pending 的 norm 保持 null。
- pending 可以没有 rollout/vector，须显式 reason 和空/已有成本引用；无效或零目标不能形成有效的可排名估值。
- 样本与目标两侧核对参数坐标、投影、计算精度、变换和可选优化器状态；样本之间还必须采用相同 loss/reduction/normalization 配置。目标与 SFT loss 定义分别记录，不强行等同。

本轮不加载向量，不验证投影误差或梯度正负号的真实计算；这些属于 P3/P4。

## 固定目标与两类报告

ObjectiveSpec 保存唯一目标、显式 SkillWeight、SkillConstraint、estimator 与 acceptance_policy 的 ArtifactRef，派生 objective_hash。
当前聚合结构为显式 weighted_sum；实际技能权重、保持边界、估值器和接受阈值由配置提供，fixture 数值不作为真实实验默认。

EvaluationReport 引用 batch_id、objective_hash、每技能 TargetRef.target_version、sample_cache_keys、GradientCoverage、SkillValue、ConstraintResult、diagnostics、状态及成本。
GradientCoverage 按 ID 严格划分 required/computed/missing；有效估值必须全覆盖，目标有效，技能与约束完整。
校验器核对声明的总分是否等于固定加权和、约束结果是否满足配置的边界；不从向量计算估值。
单独解析时也要求每个 computed ID 有一个 cache 引用；valid 的目标引用与估值技能必须完整对应。
invalid/pending 可以保留部分字段，但 total_value 必须 null，不能参与 best-feasible 比较。valid 数值报告仍可能因保持约束失败而 feasible=false。
feasible 是声明状态的派生属性，不能替代完整工作流与依赖产物验证。
diagnostics 引用应包含 norm/异常、配比、来源覆盖、重复和与 best 的差异等详细报告；实际诊断的产生由后续模块负责。

CapabilityReport 引用模型、任务集合、逐任务 rollout、split 和按技能的 CapabilityMetric。
每个 metric 显式选择 task_mean 或 rollout_mean；sample_count 始终是 rollout 数，task_ids 单独保留。
校验器核对成员集合与已声明的聚合均值；有效报告要求完整的已评分结果。环境失败不能暗计为 0 后混入均值。
独立报告的指标任务/rollout 并集也必须等于报告级成员，跨技能重复出现仍允许。
uncertainty 未估计时保持 null；提供数值时必须同时提供 uncertainty_method。P0 不计算不确定性。

## 训练与轮次记录

TrainingRecord 保存 starting_model、精确 batch_id/training_data_hash、固定 recipe、status、候选 ModelRef/checkpoint、实际 processed_supervised_tokens、receipt 与 cost。
选择了多少独特 token 与实际训练处理多少 token 分开记，后者可能包含重复曝光；真实 recipe 与执行日志核验由 trainer 后端完成。
pending 不得携带伪造训练产物；completed 必须有候选/checkpoint、正训练 token 与成本；failed 保留成本和 receipt。

RoundRecord 保存 attempt_id、accepted_version 前后值、起始/最终已接受模型、目标/决策 policy hash、选中批次/估值、TrainingRecord、前后能力报告、decision/reason/evidence 与完整 attempt 成本引用。

- accept：必须引用已训练候选和有效的配对开发报告，最终模型恰为该候选，accepted_version 加 1。
- reject / needs_more_evidence / pending：已接受模型与 accepted_version 保持原值，候选训练和成本仍可保留。
- needs_more_evidence 需要已有候选及配对记录，但不会提交；证据不足的报告可显式 invalid。
- 前后能力报告须使用相同任务版本、评分器、按任务采样配置/次数及技能聚合口径；seed 原样保存，不以 rollout 数替代独立任务数。

这是记录关系校验，不执行 policy 阈值、不写 accepted_model 指针、不恢复进程状态。P2 实现这些动作后仍须产出符合契约的记录。
`round.training_use_decision` 提供 M1 状态名映射：accept→accepted、reject→rejected、pending/needs_more_evidence→pending。
该属性不写历史，不证明训练已发生；控制器必须在实际执行事件发生时，用稳定的训练 receipt 和起始模型 hash 调用 M1 使用状态接口。

## WorkflowBundle 的检查范围

Bundle 保存一个候选程序/批次、活动模型、selection_context_hash、固定目标、相关任务/rollout/梯度/报告、账本及可选 round。
按现有 m2-v2 的 hash 关系验证程序、系统上下文和 BatchManifest；所有联接使用 ID、cache_key 或内容版本，不按数组位置匹配。
同时检查成本内容引用、mock/real 域、目标任务隔离、精确训练数据/顺序和轮次模型关系。
它用于已有候选清单的跨模块交接；尚未选样或无可行清单的轮次可单独保存 RoundRecord，不需要伪造 BatchManifest。

`validate_workflow(bundle, view=None, *, expected_model=None, expected_objective=None, expected_selection_context_hash=None, expected_batch_id=None)` 返回：

- validation_scope=record_identities；只验证声明的结构和身份。
- training_inputs_verified：是否实际传入 M1 MetricView 并核对训练 token/mask 和样本梯度位置。
- artifact_contents_verified=false：不会因为 schema 合法就宣称外部向量、环境或 checkpoint 文件真实可用。
- expected_model_verified / expected_objective_verified / selection_context_verified：只有传入系统预期值并匹配时才为 true；不能仅凭包内自洽宣称符合外部固定政策。
- expected_batch_verified：只有传入并匹配系统保存的精确最佳 batch_id 时才为 true，避免完整重封装的另一份清单/训练顺序被误作历史选择。

P2 的执行交接应传入系统活动 ModelRef、固定 ObjectiveSpec、原选择 context_hash、系统保存的精确 batch_id 及 M1 MetricView，再交给对应后端核验 artifact 文件。
load_workflow 接受同名 expected 参数。P1 直接生成此包的入口见 [p1_handoff_interfaces.md](p1_handoff_interfaces.md)。
CostLedger 的内存 ID 索引不进入 JSON/hash；模型操作成本必须带 model_hash，纯选择事件允许不绑定模型。

## 本地验收命令

安装本包后，在项目根目录运行：

```powershell
python -m self_curate validate-workflow --workflow configs/p0.fixture.json
python -m self_curate normalize --config configs/m2.data.fixture.json --output runs/p0-demo/pool
python -m self_curate validate-workflow --workflow configs/p0.fixture.json --pool runs/p0-demo/pool
python -m pytest -q
```

fixture 明确 backend=mock，包含声明的 3 任务/8 rollout、8 个样本梯度引用、2 个目标引用和完整记录链。
CLI 退出码 0 表示契约验证通过，2 表示契约或 IO 错误。它不启动 GPU，也不表示已跑通 P2 的接受/拒绝状态机。
