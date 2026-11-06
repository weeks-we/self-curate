# Leval 反馈与 Lopt 筛选程序优化接口（当前 0.9.0）

0.9.0 已接入两套 [Leval 数值后端与历史 prompt 槽位](LEVAL_BACKENDS_ZH.md)：NICE 风格梯度 cosine，以及候选子集 SFT 后的任务奖励增量。原 `ValidationRequest/Result`、`build_feedback` 和 Lopt 单步契约保持兼容；以下 0.5.1 的外部结果服务说明描述该服务本身，不再代表整个项目尚无验证/训练实现。

0.7.0 的 [InnerSearch](INNER_SEARCH_ZH.md) 已将本页验证接口与三角色单步串成内层循环，管理历史最佳、成本和检查点；本页服务本身仍保持单步语义。外层训练/接受见 [外层接口](OUTER_TRAINING_ZH.md)，目标模型和任务适配器由调用方接入。

本轮全模块复查见 [COHERENCE_REVIEW_ZH.md](COHERENCE_REVIEW_ZH.md)。可行候选附带标准 `candidate_prepared/`，可在获得下一次外部验证结果后直接交给下一轮 feedback-step，无需重复执行 prepare-selection。

传给模型的 `selection_context` 来自 `ExecutionContext.role_summary()`：包含冻结控制配置与 context_hash，语义表仅包含 artifact 身份及 pair_count，不传完整 pairs 特征。执行器仍使用完整上下文，原持久化格式和 context_hash 计算不变。

0.5.1 补充：`propose_program(..., context=frozen_context)` 可接收完整冻结配置，服务会自动传入；直接调用时若省略，不能假设客户端已经知道系统预算。`allow_seed_change=false` 同时锁定含 seed 参数的算子节点命名空间，因此新增、删除或重命名这些节点也需显式放开。每次模型调用前后均核对 ModelRef，变化时停止后续调用。规则候选被编辑策略拒绝时也以 blocked 保存证据。

本轮按用户要求，先实现一条可执行的纵向链路：**已有 P1 筛选结果 → 外部验证结果 → Leval 事实与建议 → Lopt 完整子程序 → P1 重新筛选 → 下一次验证请求**。验证数值来自外部；本模块不计算梯度、NLL、entropy、预测收益或 rollout 奖励，也不训练模型。原始蓝图和架构文档保持原样。

## 模块与边界

| 位置 | 职责 | 与前后模块关系 |
|---|---|---|
| `contracts/feedback.py` | 8 个严格、不可变、schema=1 的记录类型 | 复用现有 ModelRef、ProgramSpec、BatchManifest、ObjectiveSpec、ResourceUsage |
| `agents/leval.py` | 检查外部结果的一致性，确定性生成事实、失败约束与待检验假设 | 输入外部验证；不替代验证算法 |
| `agents/lopt.py` | 规则优化、共享模型适配器、真实程序差异和有限修复重试 | 只输出合法 ProgramSpec，复用 M2 八算子 |
| `agents/service.py` | 执行一次反馈优化步骤并原子发布证据 | 复用 P1 `prepare_selection`；交出下一次验证请求 |

`ValidationProvider.evaluate(request) -> ValidationResult` 定义验证后端协议；0.9.0 的 NICEValidator/SFTValidator 实现此协议，无隐式 mock 回退。原 fixture 仍是人为声明的外部 mock 数值，不能用作实验结论。

本轮实现 Leval 的确定性反馈逻辑，Lopt 提供可直接运行的 `method="rule"` 和可注入的 `method="model"`。尚未接入真实共享 LLM 服务，因此不宣称完整 M7/P5 或完整 P2 内外循环已经完成。

## 公开契约

- `ValidationRequest`：request_id、精确 ModelRef、完整父程序、实际 BatchManifest、固定 ObjectiveSpec、selection_context_hash、measure、protocol ArtifactRef、backend。`request_hash` 绑定全部字段。measure 必须显式选 `predicted_gain` 或 `reward_delta`，两者不能混用。
- `ValidationResult`：request_hash、valid/invalid/pending、逐技能值、总值、逐约束结果、原因、外部 evidence、实际或未知 usage。valid 必须完整覆盖固定目标；总值和约束标记必须与固定聚合规则一致。invalid/pending 不得有可排名总值。
- `LevalFeedback`：完整版本锚点、原始结果 hash、数值事实、失败约束、EditHint 假设、独立记忆命名空间。不会把“指标较低”写成“已证明样本不足”。
- `EditPolicy`：规则的配额步长、所有候选配额的上下界、证据 token 步长、最大修改节点数/总节点数、最多 3 次模型调用、超时参数、seed 是否可改。上下文系统节点上限仍然有效。share_step 是规则策略步长，模型候选受份额上下界约束。
- `ProgramChange`：由代码计算 added/removed/modified 节点 ID、output_changed、seed_changed；不接受模型自报差异。
- `LoptProposal`：完整候选程序、父 recipe hash、反馈 hash、实际差异、rationale、hypothesis、proposed/blocked、rule/model、角色命名空间。

使用 `parse_feedback_contract(kind, dict)` 做严格解析，`record.to_dict()` 序列化。旧 P0 的 22 个公开记录和 `parse_contract` 保持兼容；M1/M2 语义版本和 `p0-v3` 校验版本未改。

## 反馈与修改逻辑

Leval 首先验证 request/result 完全匹配，核对技能覆盖、固定加权和、约束结果。约束未达下界时给出提高相应技能曝光的**探索假设**；超过上界时给出降低曝光假设。约束均满足时，在正权重技能中选择本轮数值较低的一项探索。这里不是因果推断，不能保证改变曝光就能改善指标。

规则 Lopt 沿 Mix 分支向上查找 SkillSelect/EvidenceSelect，只有技能能唯一定位到一个分支时才转移监督 token 份额，总和保持不变。无法修改份额时，可以提高相应 EvidenceSelect 的最低技能 token 数。无法定位就 blocked，不任意更改 seed。修改后重新经过严格 DAG/schema、父子谱系、非空实际策略差异、节点预算和份额边界检查。

然后 P1 在**原来的 M1 快照与冻结 context** 下重新执行。质量门禁、系统预算、模型、目标、验证协议和训练排序配置都由调用方固定，不在模型输出 schema 内。新的程序可能组批不可行；这时保留候选、错误/trace 和成本，不放宽预算，不产生下一次验证请求。

对于可行候选，只产生 `next_validation_request.json`。`improvement_verified=false`，不会填入新分数、替换历史最佳、自动训练或接受权重。后续调度器必须拿到该新请求对应的验证结果，再决定比较与接受。当前模块是单步服务，不是历史最佳/恢复状态机。

## Python 接口

```python
from self_curate.agents import (
    request_from_preparation, check_external_result, build_feedback,
    propose_program, run_feedback_step, feedback_step_to_directory,
    result_from_evaluation, ValidationProvider, SharedModelClient,
)

# prepared_dir/pool_dir 指向已存在且校验通过的 P1 产物。
request = request_from_preparation(
    prepared_dir, pool_dir, request_id="validation-001",
    measure="predicted_gain", protocol=protocol_artifact,
)
# result 由外部验证器提供，不能在这里用随机构造值补位。
check_external_result(request, result)
feedback = build_feedback(request, result, memory_namespace="round-1/leval")
proposal, attempts = propose_program(
    request, result, feedback, edit_policy,
    method="rule", memory_namespace="round-1/lopt",
)
step = run_feedback_step(
    request, result, frozen_context, edit_policy,
    attempt_id="outer-001", invocation_id="feedback-001",
)
```

已有 P0 `EvaluationReport` 可通过 `result_from_evaluation(request, report, evidence, usage=...)` 转接，目前只接受 `measure=predicted_gain`，并核对模型、batch、objective。报告转接方必须确认报告来自 request 所声明的 protocol；旧报告本身没有 protocol 字段，本适配器不声称能证明该来源关系。协议摘要是身份锚点，不是 artifact 内容真实性证明。

## LLM 适配器

客户端实现 `model_ref` 属性以及：

```python
def generate_json(*, role, system_prompt, payload, timeout_seconds) -> dict:
    # 底层传输必须实施 timeout_seconds 超时；同步协议本身不会终止线程。
    return {
        "content": {
            "program": complete_child_program_dict,
            "rationale": "为何尝试该修改",
            "hypothesis": "需要下一轮验证的假设",
            "cited_feedback_id": payload["feedback"]["feedback_id"],
        },
        # 可选 usage 为既有 ResourceUsage 的完整 JSON；没有计量就省略。
    }
```

通过 `propose_program(..., method="model", client=client)` 或服务的相同参数接入。客户端 ModelRef 必须等于 request 的精确版本，Leval/Lopt 命名空间不能相同。payload 含父程序、已核对的反馈、固定目标与协议、原 batch、编辑策略和前次错误。模型不能输出 evaluator、质量门禁或系统预算字段。

每次调用保留完整 system prompt/payload、原始回复（包括无效 JSON 字符串）、错误、hash 和成本。错误回复只能在固定次数内修复，耗尽则 blocked，不回退到规则结果伪装成功。未知 token/GPU 开销仍是 null，仅补记录本机实际测得的调用墙钟。传输异常同样保留；非 JSON 可序列化的客户端返回属于协议错误，不能作为有效模型产物。

## 可运行的本地交接示例

以下所有数值与 tokenizer 都是显式 synthetic fixture，输出目录必须不存在：

```powershell
python -m self_curate normalize --config configs/m2.data.fixture.json --output runs/feedback-demo/pool
python -m self_curate prepare-selection --pool runs/feedback-demo/pool --program configs/programs/feedback.parent.fixture.json --context configs/m2.fixture.json --objective configs/p1.objective.fixture.json --backend mock --attempt-id demo --invocation-id parent --output runs/feedback-demo/prepared
python -m self_curate request-validation --prepared runs/feedback-demo/prepared --pool runs/feedback-demo/pool --protocol configs/feedback.protocol.fixture.json --measure predicted_gain --request-id external-fixture-validation --output runs/feedback-demo/request.json
python -m self_curate validate-feedback --request runs/feedback-demo/request.json --result configs/feedback.result.fixture.json
python -m self_curate feedback-step --prepared runs/feedback-demo/prepared --pool runs/feedback-demo/pool --request runs/feedback-demo/request.json --result configs/feedback.result.fixture.json --policy configs/feedback.policy.fixture.json --attempt-id demo --invocation-id feedback-001 --output runs/feedback-demo/step
```

初始两分支各 50%，外部 fixture 声明 regression=-0.02 未达 -0.01 下界；规则候选把 regression 分支改到 62.5%，recovery 为 37.5%。完整轨迹仍为 8 条、64 个监督 token，对应两分支 5/3 条；这是配额与交接验收，不是学习效果验证。

`feedback-step` 发布 `step.json`、原请求/外部结果/编辑策略、Leval feedback、完整 Lopt proposal、调用与重试记录、独立角色命名空间、当前步骤 ledger、候选程序与 P1 结果；可行时另有候选未估值 workflow 和 next request。全部文件由 checksums.json 覆盖，目录原子发布且拒绝覆盖已有输出。外部验证 usage 原样保存在 external_validation.json，当前步骤 ledger 只计本次 Leval/Lopt/P1，避免重复收费。

退出码：`feedback-step` 的 0 表示候选可行、等待验证；3 表示需要证据、提案 blocked 或候选不可行；2 表示契约、IO 或执行错误。`validate-feedback` 的 0 只说明绑定与数值一致性通过，**不代表模型验证已经执行或性能通过**。

## 近期可按模块继续

1. 实现一个验证后端，返回 ValidationResult，并验证模型/数据/协议来源及真实计算成本。
2. 接入统一的真实 SharedModelClient，实施传输超时与角色上下文隔离；对比规则和模型提案。
3. 单独实现 P2 调度器：有限候选搜索、真实估值比较、历史最佳 manifest、恢复和成本扣账；不把本轮单步候选直接当最佳。
4. 最后接训练与接受模块，只训练已验证的历史最佳精确 batch，保持最终测试隔离。
