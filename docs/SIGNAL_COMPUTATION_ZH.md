# NLL / entropy 计算与数据标签回填（0.10.1）

入口是 `self_curate.signals.SignalService`。数据由调用方提供，模块负责前向指标计算、结果输出和 M1 标签回填；不获取数据、不生成技能标注、不训练模型。

## 计算定义

输入是一条完整、无 padding 的 M1 `TrajectoryRecord`，包含实际 input_ids、assistant_mask、tokenizer/template 身份、消息边界，以及可选技能证据 span。原始 JSONL/Parquet 等数据清洗和 tokenize 仍由供应方或既有 M1 正规化模块完成。

对于选中的目标 token 位置 p：

```text
distribution = softmax(raw_logits_at_position[p - 1])
NLL[p]       = -log distribution[input_ids[p]]
entropy[p]   = -sum_v distribution[v] * log distribution[v]
```

这里 entropy 是整个输出词表的条件分布熵，与目标 token 的 NLL 不同；不能用一个 token 的 logprob 代替 entropy。不应用 temperature/top-k/top-p，不截断词表，不平滑或裁剪目标概率。

- `scope=assistant`：只统计 assistant payload 的监督位置。system/user/tool 和模板头尾保留为上下文，不进入分子或分母。
- `scope=skill_evidence, skill=...`：统计该技能 span 与 assistant mask 的交集，重叠 span 先取并集；前向仍使用完整上下文。
- `reduction=token_mean/token_sum`：按选中 token 数取均值或求和。
- `log_base=e/2`：分别输出 nats/bits。内部每 token 的 NLL 和 entropy 先以 float64/nats 计算，再统一聚合/换底。
- 未标注、空证据、无监督交集返回 `status=missing, signal=None`，不填 0。NaN/inf、错误词表、错行或错版本失败并保存费用；真实数值为 0 可以合法输出。
- 如果选中位置为 0，则没有 preceding-token 的因果预测。模块拒绝该输入，要求数据供应方提供先前上下文/BOS；不静默忽略这一项导致 token_count 与标签口径不一致。

使用稳定 log-sum-exp，按行分块计算完整词表熵。极小概率不先转成概率再取 log，因此可保留大而有限的 NLL；不能表示的非有限结果明确失败。

## 数据与模型接口

可以直接传 M1 对象列表/迭代器，也可以实现 `ScoringDataProvider.records()`。已有数据池使用 `read_pool(path).records`；已有正规化 JSONL 可用 `signals.read_records(path)`，它会重核 M1 mask、派生字段和版本。

```python
from self_curate.signals import SignalService
from self_curate.signals.torch_backend import TorchSignalBackend

# current_model: 调用方已冻结的完整 ModelRef。
# model_store: 已有 ModelStore 接口，load_branch(ModelRef) -> LoadedModel。
# data_provider.records(): 供应方返回正规化的 TrajectoryRecord。
scorer = SignalService(
    model=current_model,
    backend=TorchSignalBackend(model_store, backend="real", device=chosen_device),
    output="runs/new-forward-signals",
    max_input_tokens=chosen_complete_context_limit,
)
assistant_spec = scorer.spec()
skill_spec = scorer.spec(scope="skill_evidence", skill="recovery")
result = scorer.score_provider(data_provider, specs=(assistant_spec, skill_spec))
# 等价：result = scorer.score(records, specs=(assistant_spec, skill_spec))
updated_records = result.updated_records
view = result.metric_view()
values = view.signal_batch(requested_ids, current_model, assistant_spec)
```

`TorchSignalBackend` 已实现实际因果模型前向、p-1 对齐、完整词表 NLL/entropy。复用已有隔离 `ModelStore`，核对加载模型身份；eval/inference mode，无 backward/optimizer.step，不改变 requires_grad 标志；参数和 buffer 前后 hash 相同，恢复调用方 Python/torch RNG。禁用外部 autocast，保留参数原生 dtype，统计运算为 float64。torch/设备/数值条件/模型 factory/实现版本进入计算身份。

没有模型路径或模型名硬编码，不自动下载权重。实际模型类、权重真实性、上下文容量与 GPU 加载由供应方 ModelStore 和服务器验收负责。普通 Torch 参考实现输出完整 logits；大上下文的内存优化可通过下述后端协议接入，不在本机冒称容量已验收。

## 只提供 logits 的接口

若调用方自己执行模型或接远端服务，可实现：

```python
class MyLogitsProvider:
    model_ref = current_model
    implementation_hash = pinned_producer_config_and_code_sha256

    def forward(self, request):
        # request.input_ids：完整输入，不能只拿技能片段独立前向。
        # request.positions：本次缺少统计值的全局目标 token 位置。
        # rows[j] 对应模型位置 request.positions[j] - 1 的 raw 全词表 logits。
        return LogitsOutput(
            request.record_hash, current_model, request.positions,
            rows, source_evidence, incremental_usage,
        )

backend = ProvidedLogitsBackend(
    MyLogitsProvider(), backend=execution_domain,
    vocab_size=full_output_vocabulary_size,
    logits_dtype="float32",  # 必须与实际数组 dtype 一致
    chunk_size=128,
)
```

rows 形状必须是 `[len(request.positions), vocab_size]`，由 provider 按 p-1 从原始因果输出抽取。不是生成后的 top-k 表、不是只有目标 token 的 logprob。返回值绑定 request hash、ModelRef 和位置顺序，调用前后检查 producer.model_ref；不能把另一条数据或另一套权重的输出按行混入。

provider 负责实际权重/raw-logit 语义和源证据真实性，hash/计数检查不等于远端真实性认证。usage 是本次调用新增费用；导入已经计算好的 logits 时不重复申报旧模型推理费用，使用 source_evidence 引用原始计算。新产生的本地统计耗时单独写入证据。

还可以实现更底层的 `TokenMetricBackend.score(ForwardRequest, output) -> TokenMetricResult`，直接提供按位置绑定的两列 per-token/nats 数组。此接口保留给分布式/流式模型后端；须声明 implementation_hash、精确请求绑定、完整位置覆盖、证据和增量费用。

## M1 标签与版本

复用原契约，不把动态数值塞进静态字符串标签：

| 位置 | 内容 |
|---|---|
| record.signals | 真实计算的 SignalRecord：NLL、entropy 与 SignalSpec |
| signal.model_ref | base_model_id、revision、weights_hash、adapter_hash、parameterization、tokenizer/template 完整身份 |
| signal.binding | data_id、content_hash、tokenization_hash、选中位置 hash 和 token_count |
| signal.spec | scope/skill、聚合、log 底与计算实现/配置 hash |
| signal.provenance | request/实际逐 token 数值/模型版本的内容 hash 证据 |
| record.usage.forward_scoring | 前向统计使用记录；不混进 gradient_extraction、candidate_sft 或 accepted_sft |

更新模型、adapter、tokenizer、mask 或计算方式不会覆盖旧版本。查询与 operator 必须显式传当前 ModelRef 和相应 spec；不匹配返回 missing/stale，不拿旧 checkpoint 数值补位。原有 `make_signal` 仍只封装已算好的值，新 `SignalService.compute(record, model_ref, spec)` 实现既有 M1 `SignalProvider` 协议。

同一数据的多个 scope/reduction/log-base 尽量复用一次前向。内存缓存按数据/模型/输入/计算身份保存逐位置统计，后来扩展技能范围只补缺失位置；换 scope 会重新绑定 selection_hash，不能把原技能聚合值当新技能值。来源数据记录中的已完成同版本结果也可直接复用，不重复记物理前向费用。

本轮计算 raw NLL/entropy；nll_percentile、entropy_percentile、percentile_reference 保留 None。不能把当前候选批次的临时排序当成固定 CDF 百分位。原参考组接口继续保留，后续可按固定参考分布单独补齐。

## 输出与数据池发布

每次 score 写入新 `score-<id>`：request.json、逐次前向 request/result、逐 token 统计与证据、结果元数据 result.json、带新标签的 records.jsonl、费用账本与 checksums.json。原始对象/池不原地修改。

0.10.1 的根 request 使用 schema_version=2，新增 inputs.jsonl 原始输入快照及 token-statistics.json 聚合输入快照。重新加载时核对原 record_hash，确认只增加 signals/forward_scoring；使用保存的逐位置 NLL/entropy 重新聚合、核对标签值、provenance 和 computed/cached 状态。跨调用缓存、均值/和、换底也按同样规则检查。直接复用原记录中完整同版本指标时，对照输入快照，不要求重新前向或伪造逐 token 数组。ForwardRequest、SignalRecord、M1/P0/P1 的既有契约版本不变。

旧 schema=1 产物仍可只读加载，并尽可能核对目录中已有前向数值；不为它们补造缺失的原始输入或跨调用统计证据。原始 schema=1 的完整校验能力较弱，不能据此宣称与新快照具有同等证据。checksum 和互相一致的文件不能证明远端数据/权重真实，供应方真实性职责保持不变。

```python
from self_curate.signals import load_scoring_batch
from self_curate.signals.backends import local_artifact
from self_curate.data import publish_signal_enrichment

folder = scorer.output / ("score-" + result.batch_id)
loaded = load_scoring_batch(
    folder, expected_model=current_model,
    expected_computation_hash=scorer.computation_hash,
)
publish_signal_enrichment(
    original_pool_path, loaded.updated_records, new_pool_path,
    receipt=local_artifact(folder / "result.json"),
)
```

发布按 ID 合并，只允许 signal/forward_scoring 增量，拒绝偷偷改消息、监督 mask、技能标注、质量或训练使用历史。产生新的 pool_id，记录 parent_pool_id 和计算回执；excluded/原输入来源 hash 保留。新池可直接由既有 read_pool/MetricView/NLLSelect/EntropySelect 消费，未提供的样本保持原样。

标签更新后重建 MetricView/ExecutionContext；既有选择的原快照回放继续使用原视图，不能拿新快照冒充原 selection context。已固定批次消费可继续按既有 verify_batch 检查训练输入未变。

ScoringFailure 保留此前已完成记录/结果和全部已知 ledger；失败目录另保存 partial-records.jsonl，failure.json 记录其文件名/hash，可通过 read_records 读取已完成标签与其余未完成记录。失败目录没有成功发布标记，不能当完整批次加载；物理写入故障仍应以异常中的 records/ledger 为准。失败/缺失不会制造数值 0。

外部后端明确报告非零梯度或训练计数时，纯前向服务拒绝发布标签，并保留该调用的实际费用。未知计数继续是 None，不据此证明外部服务从未更新权重；原生 Torch 后端另有参数/buffer 前后核对。

模型或数据供应、数据划分/来源真实性仍通过外部接口接入。NLL 下降或 entropy 下降不等于任务能力进步，也不代替 Leval 或外层任务验收。

## 本机验收

```powershell
python -m pytest -q
python -m self_curate signals-demo --output runs/new-signals-demo
```

demo 使用显式合成 logits，执行实际 NLL/entropy 数学计算、M1 回填/发布/重载和两个 raw operator；不加载目标模型。专项测试另外用微型 CPU 模型执行真实前向并对照独立 log_softmax 结果，检查 mask/因果位移/多版本/数值稳定性/无训练/费用和失败边界。真实 GPU 和目标模型容量未在本机运行。
