# M1 给 operator 的数据契约（0.1.2）

按用户明确边界，**没有引入 SQLite、持久化 operator 查询索引或筛选算法**。
M1 负责数据与指标的匹配、版本标识、外部结果挂接和按 ID 取值。
operator 根据一个或多个字段自行筛选；Model/梯度/标注的计算仍由后续后端实现。

## 一条数据有哪些位置

| 位置 | 内容 | 版本/匹配方式 |
|---|---|---|
| data_id / content_hash | ID、规范化消息内容 | ID 唯一，内容 hash 独立于标签和训练历史 |
| source | 原始来源、revision、家族 ID、证据 | source hash；不能把来源家族自动当作重复组 |
| stats | assistant 轮数、工具调用/观察数、监督/上下文 token 数 | 从实际消息/token/mask 重算，拒绝不一致统计 |
| annotations / evidence_metrics | 技能名、证据位置、占比/密度/绝对量、行为 | tokenizer/template 与内容绑定；缺失不补 0 |
| annotation_record | 外部标注封装 | data_id、内容、来源、tokenization、监督位置与来源证据 |
| metadata / metadata_record | 任务类型、仓库、重复组、其他键值标签 | 同样使用绑定数据的外部封装；未知字段保留 missing |
| signals | NLL、entropy、可选百分位 | 模型/adapter + 实际输入与 mask + scope/单位/配置 + CDF |
| gradients | 样本级梯度向量引用及 norm/维度/精度/成本 | 模型 + 数据绑定 + 参数坐标/loss/归一化/投影/变换/优化器/实现与配置 |
| usage | 前向评分、梯度提取、候选训练、已接受训练 | 带证据引用，pending/rejected/accepted 分开，终态不可反转 |

梯度仅存 `ArtifactRef(uri, sha256)` 和元数据，不内嵌大向量。M1 不自动读取向量或证明它真实计算过；
后续梯度后端在作为有效缓存使用前，还须检查实际文件、格式和 checksum。

## operator 怎么读取一个或几个字段

```python
from self_curate.data import MetricView, read_pool

pool = read_pool("runs/my-pool")
view = pool.metric_view()  # 也可 MetricView(records)；每个不可变快照构建一次
ids = ("data-b", "data-a")  # 当前 operator 子集的顺序

lengths = view.static_batch(ids, ("supervised_tokens", "context_tokens", "skills"))
scores = view.signal_batch(ids, model_ref, signal_spec,
                           fields=("nll", "entropy"), missing="none")
evidence = view.evidence_batch(ids, "failure_recovery", fields=("share", "density"))
gradients = view.gradient_batch(ids, model_ref, gradient_spec, missing="none")

assert scores.data_ids == ids
for data_id, nll, entropy in zip(scores.data_ids, scores.column("nll"), scores.column("entropy")):
    pass  # operator 自己决定范围、排序或缺失政策
```

所有批量结果返回同一组明确的 data_ids、content_hashes 和等长列，**严格保持请求顺序**。
缺失数据不会删掉一行或移动后面的指标；`missing="none"` 在原位返回 None，并列出 missing cells。
默认 `missing="error"` 的信号/梯度读取在缺失时明确失败。0 是有效指标，不能当成缺失。
未知 ID、重复请求 ID、同一视图里重复数据 ID 均拒绝，避免覆盖或重复计账。

`MetricBatch.context` 保存模型、spec、CDF 身份，以及与 ID 对齐的完整 feature_keys；
证据批次额外标注 annotation_versions。静态列可读取 annotation_version、metadata_version、
source_version、usage_version，便于留痕。

可取静态列：assistant_turns、tool_calls、tool_observations、context_tokens、supervised_tokens、
quality_status、skills、task_type、repository、duplicate_groups、tags、source_id、family_ids，
以及 used_forward_scoring、used_gradient_extraction、used_candidate_sft、used_accepted_sft、
used_rejected_sft、pending_sft 和上述版本列。

skills 来自**实际提供的标注**，不会把配置中的所有技能名都标成这条数据具有的技能。
没有标注为 None；明确提供空技能集合为 ()；已有技能标签但缺 token 证据时，技能存在、证据指标仍为 missing。

MetricView 仅使用进程内字典：构建一次后，单条查询为字典取值，批量读取为请求子集大小的遍历；
不会反复扫描全部轨迹、解析 JSON、重算 token hash 或读取向量。构建时进行绑定校验。
更新标签、信号或使用状态返回新记录；需要新状态时显式构建新 view，旧 view 保持原快照。

## 外部标注与元数据的一对一匹配

```python
from self_curate.data import (
    make_annotation_record, attach_annotation_record,
    make_metadata_record, attach_metadata_record,
)

# 外部工作者应返回绑定具体数据的对象，不能只返回无 ID 的标签列表。
result = make_annotation_record(record, externally_created_annotations, provenance=(receipt,))
record = attach_annotation_record(record, result)
metadata_result = make_metadata_record(record, externally_created_metadata, provenance=(receipt,))
record = attach_metadata_record(record, metadata_result)
```

即使两条数据内容相同，ID 不同也不能交叉挂接；同 ID 的内容、来源或 token 版本改变也会拒绝。
原始 JSONL 中内嵌的 annotations/metadata 依赖所在数据行的明确关联；保留旧的低层
`attach_annotations(record, AnnotationSet)` 接口供已知关联场景使用。跨工作者传递优先使用上述绑定封装。

## 梯度的版本与挂接

GradientSpec 明确保存：参数坐标 hash/数量、loss/reduction、scope/skill、归一化版本、
计算和存储 dtype、投影方法/维度/seed/配置、梯度变换、可选优化器状态、实现与计算配置 hash。
后端必须把 dropout/RNG、截断和实际计算设置纳入 computation_hash；M1 不替后端推断这些身份。

```python
from self_curate.data import make_gradient, attach_gradients

feature = make_gradient(record, model_ref, gradient_spec,
                        vector=vector_artifact, dimension=vector_dim,
                        dtype=storage_dtype, norm=vector_norm,
                        provenance=(receipt,), cost=compute_cost)
record = attach_gradients(record, [feature])
```

缺失为 gradients=[]；已算出的零向量保留 norm=0、validity=zero；非法 norm/维度不伪装成零。
不同 checkpoint、参数坐标、投影 seed、精度、归一化、优化器状态或配置产生不同 cache_key。
同一身份下的向量/数值/已知成本冲突报错，不覆盖旧证据。成本字段不是多次重算尝试的总账；
重复尝试的资源汇总由后续账本根据 provenance 处理。
改变技能证据位置后，对应技能范围的 NLL/entropy 和梯度默认报过期；显式失效策略才移出新对象，
原对象和历史计算经历仍保留。全 assistant 范围的兼容信号继续有效。

## 使用状态怎么更新

```python
from self_curate.data import record_candidate_training, resolve_training

record = record_candidate_training(record, pending_training_use)
record = resolve_training(record, attempt_id, "accepted")  # 或 rejected
```

训练 attempt_id 应由外层使用运行命名空间生成，避免不同运行重复编号。
同 attempt 的模型和 receipt 不能改变，已接受/已拒绝终态不能反转，已有经历不能被删除。
拒绝训练仍保留 candidate_sft；前向或梯度登记不会自动变成训练使用。

## 序列化与兼容性

新规范化版本为 m1-v3：新增 metadata、gradients、annotation_record、metadata_record 的可选位置。
新的 JSONL 输入与输出保留这些字段，CLI/audit 汇总梯度引用数量。
m1-v1、m1-v2 仍可只读校验和加载，旧 record_hash 和文件 checksum 保持不变；不改旧数据补字段。
这里存储的是**样本侧**梯度引用；开发任务的目标梯度/rollout 版本由后续 M4/M5 管理。
