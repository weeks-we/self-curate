# M2 算子与批次交接接口

版本：self-curate 0.2.1；程序/上下文/清单 schema 为 `1`；算子语义为 `m2-v2`。
复查发现与修正见 [M2_REVIEW_ZH.md](M2_REVIEW_ZH.md)。旧的 m2-v1 冻结上下文不在新语义下静默重放；原交付证据保留。
设计依据见 [m2_design.md](m2_design.md)，上游字段见 [m1_operator_data_contract.md](m1_operator_data_contract.md)。

## 依赖和执行顺序

```text
M1 read_pool → MetricView（ID、内容版本、标签、长度、使用历史、版本化信号）
                           ↓
系统 ExecutionContext + ProgramSpec
                           ↓
QualityGate → 单输入节点链/共享前缀 → 可选终端 MixAllocate → BudgetPack
                           ↓
SelectionResult + BatchManifest + 节点追踪
                           ↓
M5 仅为最终成员补梯度 → M6 评估同一 manifest → M7 决策 → M8 按原清单训练
```

M2 直接使用 M1 的只读内存视图，不创建持久化查询索引，也不重新计算标签、mask、证据分母、NLL、entropy 或梯度。
每次执行固定一个视图、活动 ModelRef、SignalSpec 和参考 CDF。模型变更后必须提供新上下文；不能拿旧 checkpoint 的分数代替。
M2 不记录新的训练曝光；后续训练模块在训练真正发生时通过 M1 的 usage 接口追加历史。

## Python 公共入口

```python
from self_curate.contracts import parse_program, verify_batch
from self_curate.data import read_pool
from self_curate.operators import load_program, load_context, execute_program, load_batch

pool = read_pool("runs/m2-demo/pool")
view = pool.metric_view()
program = load_program("configs/programs/m2.fixture.json")
context = load_context("configs/m2.fixture.json", view)
result = execute_program(program, context)
if result.status == "feasible":
    exact_training_order = verify_batch(result.batch, view, model_ref=context.model_ref)
```

| 接口 | 输入 → 输出 | 约束 |
|---|---|---|
| `parse_program(dict)` / `load_program(path)` | JSON → `ProgramSpec` | 严格字段与参数；拒绝未知算子、环、未知依赖、不可达节点 |
| `parse_context(dict, view)` / `load_context(path, view)` | 系统配置+视图 → `ExecutionContext` | 冻结政策、模型、信号/CDF、训练 seed；可验证已保存快照 |
| `execute_program(program, context)` | 固定程序+上下文 → `SelectionResult` | 不修改输入，不调用模型，不写文件 |
| `select_to_directory(pool, program, context, output)` | 四个路径 → `SelectionResult` | 输出新目录；拒绝覆盖；完成后发布 |
| `load_batch(path)` / `parse_batch_manifest(dict)` | 清单 → `BatchManifest` | 校验派生 token/分支统计与 hash；不是任意 JSON 直通 |
| `verify_batch(manifest, view, *, model_ref=None)` | 清单+视图+可选预期模型 → 原 `training_order` | 核对每条内容、实际 token/mask 和监督长度；允许之后追加梯度等特征；指定模型时必须匹配清单 |

`ProgramSpec`、`NodeSpec`、`BatchManifest`、`BatchMember`、`SelectionResult`、`NodeTrace` 从 `self_curate.contracts` 导出。
`QualityPolicy`、`BudgetPolicy`、`SignalContext`、`SimilarityPair`、`SimilarityTable`、`ExecutionContext` 从 `self_curate.operators` 导出。

## ProgramSpec：可编辑配方

必填：`schema_version, program_id, nodes, output, seed`；可选 `parent_id`。
节点固定为 `id, op, inputs, params`。节点 ID 长度 1–128，只允许字母/数字/下划线/点/横线。
`eligible_pool`、`final_batch` 为系统保留 ID。单输入算子恰好一个依赖；Mix 至少两个依赖且必须为最终节点。
允许共享前缀；不允许 Mix 后再过滤或嵌套 Mix。JSON 节点数组的排列不影响依赖顺序；Mix 的输入列表顺序决定固定归属优先级。
节点最多 256，系统可通过 `max_nodes` 收紧（默认 128）。

```json
{
  "schema_version": "1", "program_id": "recovery-only", "seed": 11,
  "nodes": [
    {"id": "skill", "op": "SkillSelect", "inputs": ["eligible_pool"],
     "params": {"skills": ["recovery"], "match": "any"}},
    {"id": "sample", "op": "SampleSelect", "inputs": ["skill"],
     "params": {"method": "uniform", "size": 4, "seed": 22}}
  ], "output": "sample"
}
```

`QualityGate`、`BudgetPack` 不是可编辑算子。`program_id/parent_id` 是谱系信息，不参与语义 `recipe_hash`；seed、参数、依赖顺序和算子版本参与。
`program_hash` 进一步绑定系统政策与执行上下文；`execution_hash` 绑定此次快照。

## 八算子参数

除明确要求排序/抽样的算子外，筛选保留输入顺序。区间均为闭区间，`null` 表示该端无界，不表示缺失值可通过。

| 算子 | 参数与规则 |
|---|---|
| `SkillSelect` | 必填非空 `skills`；`match=any/all`（默认 any）。读取真实标签，而不是技能注册表。`missing_policy=exclude/error/include`（默认 exclude）。 |
| `EvidenceSelect` | 必填 `target_skill`；可选 `min_share/min_density`∈[0,1]、非负整数 `min_tokens/min_events`、`required_behaviors` 列表。所有已指定条件须同时满足。缺失默认 exclude，可 error/include；已知不满足的条件不能由 include 绕过。 |
| `NLLSelect` / `EntropySelect` | `scope=assistant/skill_evidence`（后者必填 skill）；`scale=raw/percentile`（后者必填 reference_group）。至少指定 `interval`、`keep_fraction` 或 `rank_direction` 之一。先区间，再按 ascending/descending 排序，同分按 ID，再取 ceil(剩余数×keep_fraction)。只有区间时不排序；比例默认升序。`missing_policy=error/exclude/request`（默认 error）。 |
| `LengthSelect` | 至少一个范围：`assistant_turn_range/tool_call_range/supervised_token_range/context_token_range`。长度口径独立，多条件取交集。 |
| `DedupSelect` | 必填 `mode=exact/annotated_group/semantic`；`representative_rule=first/lowest_id/shortest`。组标注取传递闭包，不拿来源家族当重复组。semantic 必填 threshold∈[-1,1]，按代表优先级贪心比较。`missing_policy=error/exclude/include`，默认 error。 |
| `SampleSelect` | 必填 `method=uniform/ranked/stratified`；`size` 与 `token_target` 恰选一个正整数；seed 默认 0。ranked 使用输入顺序；uniform 用稳定 hash 无放回抽样；stratified 必填 strata，支持 source_id/repository/task_type/skills 的组合，skills 作为整组标签分层。按条数使用与分层样本数成比例的最大余数配额；按 token 使用与分层可用监督 token 成比例的轮转。缺失分层默认 error，可 exclude/include，include 将缺失作为单独层。 |
| `MixAllocate` | 必填 `branch_token_shares`，键与输入节点精确对应、正数且和为 1。`overlap_rule=fixed_priority/seeded_priority`（默认前者），seed 默认 0。先唯一分支归属，再按监督 token 贪心分配。只支持 `shortfall_policy=report_infeasible`，不会自动回填另一分支改变目标比例。 |

Sample 的 `token_target` 是**预筛选上限**，会跳过装不下的整条轨迹并报告短缺；最终训练预算由系统 BudgetPack 硬性检查。
节点随机顺序绑定程序 seed、节点 ID 和节点 seed；最终训练顺序另由系统 training_seed 决定，不使用全局随机数状态。
改变只做固定过滤的节点顺序通常不影响交集；排序/比例截取/去重/抽样涉及顺序，不能假定交换等价。

## ExecutionContext：系统固定的外壳

必填 `schema_version, quality_gate, budget_pack`；可选 `model_ref, signal_contexts, similarities, training_seed, training_order, max_nodes`。
保存后的配置还包含 `snapshot_hash, operator_version`，可直接重放；两者不匹配明确报过期。

- QualityPolicy 默认仅 pass，排除 pending、accepted、rejected 训练历史；系统可显式允许 unknown/accepted/rejected，设置 excluded_ids/families。程序无法更改这些值。
- BudgetPolicy 必填正整数 token_target（不超过 JSON 安全整数上限 2^53−1）、token_tolerance∈[0,1)、正整数 effective_batch；可选 share_tolerance∈[0,1]（默认 0.1）、max_samples（不小于 effective_batch）、deduplicate_content（默认 true）。
- token 最终窗口为 `[ceil(target×(1-tolerance)), floor(target×(1+tolerance))]`，按配置值的十进制表示作精确有理数计算，不额外加入隐含宽限；至少 1 个 token。样本数必须整除 effective_batch。
- Mix 分支预配额为 `target×share ± target×share_tolerance`（取整），保留完整轨迹；BudgetPack 最后按**实际总监督 token**重新检查每个分支实际份额与目标份额的绝对差，不以样本条数代替。
- `training_order=seeded/selection`；默认 seeded。`training_seed` 只控制最终训练排列，不改变已选成员。
- `signal_contexts` 为 `{spec, references}` 数组；同一个 scope/skill 只允许一个固定 spec，参考组唯一且必须匹配活动模型。百分位仅读已保存的参考 CDF 结果，不从当前候选集重算。
- 一个执行上下文中的所有记录必须使用同一 tokenizer/template 身份，即使暂未提供 ModelRef；避免把不同 token 编码组成同一批次。
- `similarities` 为 `{identity: ArtifactRef, pairs: [...]}`；每项包含 left_id/right_id、两个 content_hash、value。缺失比较对按算子政策处理；内容过期不能静默接受。实现不加载 embedding 模型。

`ordered-greedy-v1` 依次装完整轨迹，最后修整样本数以满足 batch 整除，优先保留 token 下限和分支比例，再保留输入中更早成员。
不保证求出所有离散组合中的可行解；找不到符合固定约束的结果返回 infeasible，并明确 `global_infeasibility_proven=false`。

## 输出和后续模块责任

`SelectionResult.status` 为 feasible/infeasible；只有 feasible 带 batch。batch 保存：

- 每条 data_id、content_hash、原 record_hash、实际 tokenization_hash、supervised_tokens 和唯一 branch。
- 排序后的 members、明确 selection_order 和 training_order，训练 seed。
- 目标 branch_shares、实际分支 token/份额、模型身份、synthetic 标志。
- program/execution/snapshot/policy hash、batch_id、training_data_hash。

M5/M6 应以最终 members 为输入，在这个 manifest 上计算和估值，不能先算一批、最终训练另一批。
后续新增梯度导致 record_hash 变化是正常的：verify_batch 校验内容与训练 token/mask，允许特征补全；原记录 hash 仍留作筛选时证据。
真实 M5/M6/M8 交接时须传入系统当前的 ModelRef 作预期模型检查；模型改变后不能把旧清单误作当前模型上的选择结果。
M8 必须校验并使用清单中的 training_order；此清单尚不是训练超参数或 optimizer 合约。
M2 保证交接数据可核验；后续模块仍需真正实现并遵守这些接口。

最终 BudgetPack 追踪另外报告 token_target_residual、branch_share_residuals 和 semantic_skill_exposure。
语义技能曝光按真实标签统计样本数、整条轨迹监督 token、已知证据 token 与缺失计数；多技能样本可在多个技能中出现，不能把这些数相加冒充唯一训练 token。
分支归属仍严格唯一，和语义标签统计分开；全部证据缺失时证据总量保持 null。

`usage_exposure` 按最终打包成员报告 used_forward_scoring、used_gradient_extraction、used_candidate_sft、used_accepted_sft、used_rejected_sft、pending_sft：每类包含 data_ids、sample_count、sample_fraction、supervised_tokens、token_fraction。
前向/梯度暴露与训练暴露分别统计；类别可以重叠，不能相加当作唯一成员数。空批次的比例为 null；本模块只读历史，不产生新的训练事件。

`MISSING_SIGNAL` 且策略 request 时，ContractError.details.requests 含 metric、scale、data_ids、ModelRef、SignalSpec、可选参考 CDF。
同时带 program/execution hash、失败节点与 completed_traces；后续计算服务用这些身份补齐指标，然后创建新上下文重跑，不会现场隐式调用模型。

## CLI 与可运行验收

在项目根目录安装 `python -m pip install -e ".[dev]"` 后：

```powershell
python -m self_curate normalize --config configs/m2.data.fixture.json --output runs/m2-demo/pool
python -m self_curate validate-program --program configs/programs/m2.fixture.json
python -m self_curate select --pool runs/m2-demo/pool --program configs/programs/m2.fixture.json --context configs/m2.fixture.json --output runs/m2-demo/selection
```

示例贯通全部八类算子：12 条合成轨迹，每条 8 个监督 fixture token，最终 8 条/64 token，两个分支各 32 token，effective_batch=8。
其中分数、标签、判分证据均显式为 fixture，不代表真实 NLL/entropy/质量验证。
若不安装，可以在当前终端临时设置 `$env:PYTHONPATH='src'` 后运行；不需要目标模型或第三方运行时依赖。

selection 新目录内含 program.json、context.json、result.json、trace.json、batch.json、checksums.json。
退出码：0 为可行；3 为预算/配额不可行（不生成 batch）；2 为契约/IO 错误。
节点执行失败会发布 failed 的 result.json、上下文、程序与 checksums，保留计算请求和已完成追踪；程序/上下文校验阶段失败不会发布选择结果。
所有已有输出拒绝覆盖；相同输入和 seed 重放可得到字节相同的选择产物。M1 数据池不被修改。

## 下一步

先接入 P2 的显式 mock 搜索循环：消费本模块的精确清单，测试历史最优、拒绝保留证据和恢复幂等。
随后接真实 tokenizer/外部分数服务，再做 M5 样本梯度、M6 估值和 M8 候选训练；这些不能以本轮 fixture 验收代替。
