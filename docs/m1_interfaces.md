# M1：公共接口与输入输出契约

本模块对应蓝图第 6 节。运行时不需要模型，不调用训练、梯度、网络或容器。

## 1. Python 接口

```python
from self_curate.data import (
    TokenizationBackend, FixtureTokenizer, PrecomputedTokenizer,
    normalize_trajectory, load_config, build_pool, read_pool,
)

record = normalize_trajectory(raw_record, backend, declared_skills=("failure_recovery",))
record.data_id
record.content_hash                 # 规范化消息内容；不混入来源或标注
record.record_hash                  # 包括 token、mask、指标、质量、使用历史
record.stats.supervised_tokens
record.tokenization.assistant_mask
record.labels                       # 非监督位置为 -100
record.evidence_metrics             # tuple[EvidenceMetrics, ...]

config = load_config("configs/m1.fixture.json")
manifest = build_pool(config, "runs/a-new-pool")
snapshot = read_pool("runs/a-new-pool")
snapshot.records                    # tuple[TrajectoryRecord, ...]，按 data_id 排序
snapshot.excluded                   # 精确排除原因和记录身份
```

公共数据契约均来自 `self_curate.contracts`，可通过 `.to_dict()` 导出 JSON 对象。
不可变 dataclass 用 tuple 保存嵌套序列。对象与大文件按内容 hash 关联，不以路径充当身份。
来源、质量、标注和使用历史通过 record_hash 单独绑定；同一内容的多个来源仍可由 M2 发现重复。

当前提供的是 M1 的完整公共契约；ProgramSpec、BatchManifest、ModelRef 等属于后续模块，
本次不发布只有占位字段的假接口。

## 2. 原始输入：一行一个对象

必填：`schema_version="1"`、`data_id`、`source`、`messages`、`quality`。
可选：`annotations`、`usage`、`tokenization`。具体小样本见 `tests/fixtures/m1_trajectories.jsonl`。

| 字段 | 约定 |
|---|---|
| source | source_id、revision、record_id、artifact={uri,sha256}；可选 family_ids |
| messages | role 为 system/user/assistant/tool；content 必须为字符串，不静默丢文本块或图片 |
| assistant.tool_calls | [{call_id,name,arguments}]；arguments 可为 JSON object 或其 JSON 字符串，规范化为 canonical JSON 文本 |
| tool.tool_call_id | 必须对应待返回的 call_id；多个工具可按任意顺序返回，之后才能出现下一条非 tool 消息 |
| quality | status 为 pass/fail/unknown；method 为 executable/manual/teacher/unverified；pass 要求 verifier_id 与 evidence 引用 |
| annotations | tokenizer_hash、template_hash、skills；缺失整个字段表示未提供技能证据 |
| usage | 四类曝光历史，见下节；省略表示源记录声明无使用历史，并非已经核验它从未使用 |
| tokenization | precomputed 模式必填；fixture 模式禁止携带，防止覆盖真实 token |

缺字段、未知字段、重复 JSON key、错误类型、NaN/Infinity、坏 span、未配对工具调用均明确失败。
空行也视为 JSONL 数据错误，错误携带输入行号。

## 3. 分词与监督 adapter

```python
class TokenizationBackend(Protocol):
    @property
    def identity(self) -> TokenizerRef: ...
    def encode(self, messages: tuple[Message, ...], provided: dict | None = None
               ) -> TokenizedConversation: ...
```

`TokenizerRef` 包含 tokenizer_id、revision、tokenizer_hash、template_id、template_hash、backend、schema_version。
`TokenizedConversation` 包含 identity、input_ids、assistant_mask、message_spans、schema_version。

- input_ids 与 mask 等长，mask 必须为布尔值；0/1 整数不会冒充 bool。
- 每条消息对应一个 payload token span，使用整个序列中的 `[start,end)`，不是消息内偏移。
- span 不含角色 header、模板分隔符；可以覆盖完整 assistant 工具动作序列。
- 本版本所有 assistant payload token 为 true，其他位置全部 false。
- EOS/特殊 token 若需监督，必须先定义 adapter/span/mask 的新版本，不能悄悄改规则。
- annotations 中的 span 必须在所声明消息的 payload span 内，可标到工具观察，但统计时只有监督交集计数。

`FixtureTokenizer` 为一字符一 token 的显式合成实现，另带角色 header/分隔符，便于手工检查。
`PrecomputedTokenizer(identity)` 读取外部 tokenizer 已产生的 token/alignment，不加载权重；
缺失 token、身份不同或 mask 对不齐就报错，绝不退回 fixture。

本模块只能验证外部对齐声明的结构及一致性，不能仅凭 token 数组证明真实 tokenizer 版本或源证据真实性。

## 4. 技能标注与充分性

每个技能条目含 `skill`、`annotator_id`；可选 `spans`、`complete_events`、`behaviors`、`confidence`。
span 为 `{message_index,start,end}`，全部 token 坐标绑定 annotations 的 tokenizer/template hashes。

```text
技能监督证据 tokens = union(该技能所有 span) 与 assistant_mask 的交集大小
证据总分母          = union(所有技能 span) 与 assistant_mask 的交集大小
share               = 技能监督证据 tokens / 证据总分母
density             = 技能监督证据 tokens / 全部 assistant 监督 tokens
```

同一技能重复或相交 span 不重复计数；跨技能证据可共享，share 之和可以超过 1。
`spans=null`/未提供：status=missing，计数/share/density 均为 null。
`spans=[]`：status=annotated，计数为 0，density=0；若总分母非零，share=0，否则 share=null。
事件数和行为标注保持 unknown/null，不从文本关键词自动推断测试成功。

## 5. 四类使用历史

```json
{
  "forward_scoring": [{"uri": "...", "sha256": "<64 lowercase hex>"}],
  "gradient_extraction": [],
  "candidate_sft": [{
    "attempt_id": "attempt-1", "model_id": "candidate-1",
    "receipt": {"uri": "...", "sha256": "<64 lowercase hex>"},
    "decision": "rejected"
  }],
  "accepted_sft": []
}
```

candidate_sft 的 decision 为 pending/rejected/accepted。accepted_sft 是已接受 attempt_id 集合，
必须与 candidate_sft 中 decision=accepted 的集合完全一致，不能只填“已接受”而没有训练经历。

默认排除 accepted；rejected_history_policy 必填 allow/exclude；pending 固定排除。
forward/gradient 历史不自动构成训练排除。允许被拒绝的数据再次进入候选池时，原历史仍完整保留。

## 6. 配置与数据池

配置使用 JSON，路径相对于配置文件所在目录解析。fixture 示例位于 `configs/m1.fixture.json`。
precomputed 模式额外必填 tokenizer identity；本版本不内置任何目标模型或真实实验预算。

`max_context_tokens` 是完整模板化序列长度。oversize_policy=reject 写入排除清单，
oversize_policy=error 整体失败；两者都不截断输入。QualityGate 不在 M1 执行，fail/unknown 原样保留。

发布目录包含：

| 文件 | 内容 |
|---|---|
| trajectories.jsonl | 规范化完整记录、消息、mask、输入 token、指标、历史 |
| excluded.jsonl | 每条政策排除的 ID、内容/记录 hash、理由、上下文长度 |
| audit.json | 数量、质量分布、token 总数、缺失标注统计；仅证明结构校验通过 |
| manifest.json | 输入 SHA256、实现 SHA256、配置、tokenizer、记录/文件 hash 和 pool_id |

`read_pool` 校验文件 checksum、重算记录的 mask/统计/证据指标、校验清单和审计总量。
它不重写文件；重复调用同一份产物不变。build_pool 不隐式续跑、不覆盖已发布目录。

写入先放到同级隐藏 staging，再完整发布；失败留下 `failure.json` 且不发布完成目录。
发布锁记录 PID 和 staging 路径；中断遗留锁需先检查拥有进程，不能盲删。也可使用新的输出目录。

## 7. 错误和下游约定

`ContractError.code` 是 `ErrorCode` 枚举，`.to_dict()` 给出 code/message/details。
CLI 成功返回 0；业务/输入错误返回 2，并在 stderr 输出 JSON，不拿空池或 0 分掩盖错误。
不存在的文件、权限与编码问题报告 ENVIRONMENT_FAILURE；缺失/坏完成产物报告 ARTIFACT_MISMATCH。

当前导入器按小规模数据池设计，排序和结果在内存中保留；大规模分片/外排序应另作扩展并测容量，
不能从本地小样本通过推断可处理任意大小的轨迹库。

M2 可直接读取 records 的 quality、stats、evidence_metrics、usage 与 source.family_ids；
不要再计算一份有不同分母的 share，也不要修改原规范化记录。
