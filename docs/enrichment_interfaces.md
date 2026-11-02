# 外部标注与模型信号的接入接口

本次补齐数据层的位置、公共类型、校验、挂接与查询。自动标注、NLL/entropy 前向计算、
CDF 计算均未实现，也不会因调用数据接口而自动发生。

0.1.2 补充了绑定 data_id/content/source/tokenization 的 AnnotationRecord、MetadataRecord，
以及样本梯度引用和 operator 的 MetricView。跨工作者标注应使用绑定封装；下文的 AnnotationSet
低层挂接保留给已经知道数据关联的场景。详见 [当前 operator 数据契约](m1_operator_data_contract.md)。

## 1. 数据现在保存在哪里

```text
TrajectoryRecord
  annotations     外部技能标注；未提供为 null
  evidence_metrics  M1 根据标注与监督 mask 计算的确定性统计
  signals         外部模型信号列表；未计算为 []
    SignalRecord
      binding     数据 ID、内容、实际 tokenization、选中位置与 token 数
      model_ref   基座、revision、权重/adapter、tokenizer/template、参数表示
      spec        scope、reduction、log base、计算实现与配置 hash
      nll         已计算值或 null
      entropy     已计算值或 null
      nll_percentile / entropy_percentile   可选，未计算为 null
      percentile_reference                 可选，参考组及 CDF 身份
      provenance  计算证据引用，挂接后登记为前向计算历史
```

NLL 与 entropy 都允许暂时缺一个；0 是合法数值，null 才表示缺失。
不建立 nll/entropy 全空的“已计算”记录。模型信号与 content_hash 分离，挂接不会改变轨迹内容身份。

## 2. 外部标注器接口

```python
from self_curate.data import AnnotationProvider, attach_annotations
from self_curate.contracts import parse_annotations

# AnnotationProvider 当前返回绑定数据的 AnnotationRecord 或 None。
# 以下是已知关联时可继续使用的低层 AnnotationSet 导入接口。
# 若结果来自 JSON，可通过严格解析器导入：
labels = parse_annotations(external_json_object)
enriched = attach_annotations(record, labels)
```

该操作生成新对象，保留原记录，重算 evidence_metrics 和证据总量。
tokenizer/template hash 不兼容或 span 越界时拒绝挂接。

如记录已有 skill_evidence 范围分数，改变相关证据位置默认报 STALE_CACHE。
需要使它们失效时显式调用 `attach_annotations(record, labels, drop_stale_signals=True)`：
只从新对象移除过期信号，原对象与历史计算证据继续保留。

## 3. 后续计算模块返回什么

```python
class SignalProvider(Protocol):
    def compute(self, record: TrajectoryRecord,
                model_ref: ModelRef, spec: SignalSpec) -> SignalRecord: ...
```

模型加载与计算由后续 provider 实现。数据模块不实例化 provider，也没有默认的伪计算后端。
模型服务可以直接返回 `SignalRecord`；跨进程 JSON 通过 `parse_signal_record()` 解析。

调用方式如下。`model_ref` 与 `receipt` 由外部模型/证据管理器提供，标量值由实际计算模块返回：

```python
from self_curate.contracts import SignalSpec
from self_curate.data import make_signal, attach_signals, get_signal

spec = SignalSpec(
    scope="assistant",
    reduction="token_mean",
    log_base="e",
    computation_id=forward_implementation_version,
    computation_hash=forward_config_and_implementation_hash,
)

signal = make_signal(
    record, model_ref, spec,
    provenance=(receipt,),
    nll=externally_computed_nll,
    entropy=externally_computed_entropy,
)
enriched = attach_signals(record, [signal])
value = get_signal(enriched, model_ref, spec).require_value("nll")
```

`make_signal()` 仅包装传入结果，自动绑定当前记录的内容/token/mask/位置身份。
如果后续计算模块已经实现同一接口，也可以直接：

```python
signal = external_provider.compute(record, model_ref, spec)
enriched = attach_signals(record, [signal])
```

### 身份与范围

- ModelRef：base_model_id、revision、weights_hash、adapter_hash、tokenizer、parameterization；以内容构成 model_hash。
- SignalSpec.scope：assistant 或 skill_evidence；后者必须指定 skill，只选择该技能证据与监督 mask 的交集。
- reduction：token_mean 或 token_sum；它们的信号不能混用。
- log_base：e 或 2，记录实际计算的单位，不自动转换数值。
- computation_hash：覆盖真实计算配置与实现，包括数值精度、上下文处理、模型模式等；调用方负责提供可信身份。
- SignalBinding：由 M1 生成。缺少技能标注或范围内没有监督 token 时，报 MISSING_SIGNAL，不产生零分。

同样的 token 内容可以拥有多个模型版本的 signal。查询必须传入精确 ModelRef/SignalSpec，
不会退回另一个 checkpoint 或另一种统计范围。

同上下文先返回 NLL、后返回 entropy 时可以再次 attach：互补值合并；相同值幂等；
同身份出现不同值时报 SIGNAL_CONFLICT，需核对计算身份，不允许默默覆盖。
provenance 与 usage.forward_scoring 取证据并集；不会伪造梯度或训练经历。

## 4. 可选百分位的位置

SignalRecord 预留 nll_percentile、entropy_percentile 与 PercentileReference：

```text
PercentileReference
  reference_group       参考组名称
  reference_ids_hash    固定参考样本 ID 集合身份
  cdf_version          CDF 版本
  artifact             CDF 文件 URI + SHA256
  model_hash           产生参考分数的模型
  spec_hash            对应统计范围/单位/计算配置
```

百分位必须在 [0,1]，并提供对应 raw 值与 CDF 引用。
查询该视图时向 `get_signal(..., percentile_reference=reference)` 提供同一引用，
再调用 `.require_value("entropy", "percentile")`。不同 CDF 视图独立存储，raw 值仍须一致。
M1 只存储和验证这些字段，不估算百分位。

## 5. JSONL、数据池与兼容性

原始输入 JSONL 新增可选 `signals` 数组。每个元素是 `SignalRecord` 的 JSON 表示。
外部程序可以把 `signal.to_dict()` 经 JSON 序列化后加入原始记录，再执行现有 `normalize/build_pool`。
规范化输出持久化 signals；read_pool 会重新核验绑定、数值、参考版本与文件 checksum。
audit/inspect-pool 另外给出 signal_records 数量，不把未计算数据填零。

当前规范化版本是 m1-v3；输入 schema 仍为向后兼容的 1（新增可选字段）。
旧 m1-v1/m1-v2 数据池可只读加载，旧 record_hash 保持不变。显式挂接生成新版本对象，
新的落盘输出必须使用新目录；不要修改原池来补写分数。

以上校验只能确认外部声明的格式和一致性，无法证明计算真实执行或权重来源可信；
真实模型与计算证据的验真在对应后端阶段完成。
