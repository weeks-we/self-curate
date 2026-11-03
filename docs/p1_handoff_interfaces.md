# P1 正式入口：筛选并交接给 P0

包版本 0.5.1。P1 执行沿用 m2-v2；PreparedSelection 和发布格式 schema_version=1。
原 `execute_program` / `select` 保留。需要后续工作流消费时使用本接口，无需手工拼装批次。

## Python API

```python
from self_curate.contracts import parse_contract, validate_workflow
from self_curate.operators import prepare_selection, prepare_to_directory, verify_preparation

# program/context/objective 由提案和系统配置分别提供；context 已持有 M1 MetricView。
prepared = prepare_selection(
    program, context, objective,
    backend="mock", attempt_id="attempt-1", invocation_id="select-001",
)
if prepared.status == "feasible":
    bundle = prepared.workflow
    receipt = prepared.validate_workflow()
    assert bundle.batch == prepared.selection.batch
```

| API | 返回/约束 |
|---|---|
| prepare_selection(program, context, objective, *, backend, attempt_id, invocation_id) | PreparedSelection；要求明确的活动 ModelRef 和 mock/real 域，不接模型/GPU后端 |
| prepare_to_directory(pool, program, context, objective, output, *, backend, attempt_id, invocation_id) | 加载四个路径、执行一次候选、完整后发布新目录；拒绝覆盖 |
| verify_preparation(path, original_view, *, expected_model=None, expected_objective=None, expected_batch_id=None) | 核验固定文件/checksum，并在原始视图上重放 CPU 算子；可独立核对已保存清单；只返回验证回执 |
| prepared.validate_workflow() | 用实际传入的 context/model/objective 与视图核对 P0 包；无可行包时报 INSUFFICIENT_POOL |
| load_preparation(path, original_view, *, expected_model=None, expected_objective=None, expected_batch_id=None) | 完成文件、锚点和重放检查后返回 PreparedSelection，沿用已保存的执行成本 |
| publish_preparation(prepared, output) | 发布已经执行的 PreparedSelection；不重跑、不新增选择事件，拒绝覆盖 |

PreparedSelection 是本次调用的只读内存对象；持久化边界为下列 JSON 文件。
status 为 feasible/infeasible/failed。捕获的算子 ContractError 在 prepared.error 中保留结构化 details/requests；不转换为零分或空批次。

## 输出

所有状态发布 prepared.json、program.json、context.json、objective.json、outcome.json、ledger.json、trace.json、checksums.json。
只有 feasible 发布 batch.json、workflow.json。

- workflow 含原 ProgramSpec、同一 BatchManifest、系统 ObjectiveSpec、活动模型、原 context_hash 和选择成本。
- tasks/rollouts/target_gradients/sample_gradients/evaluations/capabilities 为空，round_record=null；这些输出尚未执行。
- 不可行报告保留分支/预算缺口。缺 NLL/entropy 时保留计算请求与完整模型/spec/CDF 身份，P1 不计算它们。
- 选择成本仅记录 CPU selector 的实测墙钟。GPU、模型处理、梯度、训练 token 明确为 0，因为本阶段没有调用这些能力；配置加载、发布及 inspect 的开销不包含在此项中，P2 工具调用账本可单独覆盖。
- CostEvent.receipt 的 record://selection-outcome/<hash> 是规范 JSON 记录的逻辑身份，不冒充某个外部物理文件的验证结果。文件本身另由 checksums.json 校验。

## 重放与调用身份

attempt_id 是外层尝试，invocation_id 是本次物理调用。真正重试或再次运行使用新的 invocation_id；恢复已完成调用应先读取、校验产物，不再次执行并重复计账。
同一程序、快照、系统政策、模型/信号/CDF 和 seed 得到相同 selection/batch；实际运行的墙钟、调用 ID 和相应 ledger/workflow hash 可不同。
原 select 的无计时产物仍可字节一致重放。

verify_preparation 需要**原始选择快照**，校验 context 的 snapshot_hash。若之后追加梯度、信号或使用历史，则改用：

```python
from self_curate.contracts import load_workflow

bundle = load_workflow(
    "runs/p1-demo/prepared/workflow.json", view=enriched_view,
    expected_model=current_model,
    expected_objective=fixed_objective,
    expected_selection_context_hash=original_selection_context_hash,
    expected_batch_id=historical_best_batch_id,
)
```

这会核对原批次的内容与 token/mask，允许追加特征。重新选样时应显式建立新上下文，不能把新快照冒充旧选择重放。

## CLI

在项目根目录安装包后：

```powershell
python -m self_curate normalize --config configs/m2.data.fixture.json --output runs/p1-demo/pool
python -m self_curate prepare-selection --pool runs/p1-demo/pool --program configs/programs/m2.fixture.json --context configs/m2.fixture.json --objective configs/p1.objective.fixture.json --backend mock --attempt-id demo-attempt --invocation-id select-001 --output runs/p1-demo/prepared
python -m self_curate inspect-preparation --prepared runs/p1-demo/prepared --pool runs/p1-demo/pool --expected-model configs/p1.model.fixture.json --expected-objective configs/p1.objective.fixture.json
```

prepare-selection：退出码 0 可行、3 不可行、2 算子/契约/IO 失败。
程序/输入配置校验失败发生在执行前，不发布执行产物；执行中的结构化失败会发布 failed 结果及本次选择成本。
inspect-preparation：0 验证通过，2 产物/版本/配置错误；不会因发现缺分数而调用模型。

工作流也可单独核对系统配置：

```powershell
python -m self_curate validate-workflow --workflow runs/p1-demo/prepared/workflow.json --pool runs/p1-demo/pool --expected-model configs/p1.model.fixture.json --expected-objective configs/p1.objective.fixture.json --expected-context configs/m2.fixture.json --expected-batch runs/p1-demo/prepared/batch.json
```

expected 参数来自系统受控配置和系统已保存的清单；--expected-context 要求 --pool。
回执分别列出模型、目标、上下文、精确批次和训练输入是否已核对。--expected-batch 也可传给 inspect-preparation。
执行历史最佳清单时应使用搜索保存的那份 BatchManifest，不能只依赖包内自洽。未提供的检查不会伪标完成。

## 之后如何接模块

M3 消费缺分数请求并回填新快照；M5 仅补最终批次缺失梯度；M6 引用精确 cache_key/target_version 形成有效报告。
P2 搜索维护历史最佳 prepared.workflow.batch 和报告，M8 使用同一 training_order/training_data_hash；P1 不做这些执行步骤。
backend=mock 的 fixture 仅验证工程语义。真实向量、模型、任务环境、接受/拒绝和恢复必须由后续后端验收。
