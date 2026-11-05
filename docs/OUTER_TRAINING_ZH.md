# 外层两阶段训练、任务验证与接收／退回

版本 0.8.1，2026-11-05。用户确认：筛选轨迹 SFT 接着子集 SFT 的权重继续训练；每次验证后做接收／退回判断，标准待填。这是对原蓝图单次候选 SFT 的新增要求，原研究文档保留。最新检查与旧报告接入见 [两遍复查](OUTER_TWO_PASS_REVIEW_ZH.md)。

## 流程和退回语义

```text
正常停止的内层搜索 → 历史最佳精确子集
→ 隔离分支 SFT → 固定开发任务验证 → 判断
  接收 → 子集权重成为本次 attempt 的 accepted 模型
       → 接入外部筛选轨迹 → 从子集权重创建隔离分支 SFT
       → 同一开发任务／采样／seed／协议验证 → 再判断
```

- 第一阶段退回：本次尝试结束，保留内层冻结模型，不执行第二阶段。
- 第二阶段退回：保留第一阶段已接收模型，保留第二阶段产物和成本。
- 标准或判断器未配置：返回 `awaiting_decision`，不自动接收。
- `needs_more_evidence`：等待；可显式复测，不自动重训或无限重试。
- 两阶段接收：`completed`；退回：`rejected`；执行异常：`blocked`。
- 未添加内层退出条件、最终 test 执行或三个角色的自动权重切换。

## 模块与公共接口

| 模块 | 入口 | 职责 |
|---|---|---|
| training/contracts.py | TrainingExample/Data/Request、SFTRecipe | 版本化、不可变的训练输入／预算 |
| training/data.py | prepare_subset | P0 锚点核验和 best 原清单精确导出 |
| training/service.py | TrainingService、TrainingFailure | 单次物理训练、P0 TrainingRecord、成功／失败成本 |
| training/torch_backend.py | TorchSFTBackend、ModelStore、StateDictModelStore | 实际参数优化、独立权重与 checkpoint 接续 |
| training/artifacts.py | verify_local_checkpoint | 本地 checkpoint 文件树校验 |
| validation/contracts.py | TaskSuite、TaskRunner、TaskRunRequest/Result、TaskEvaluation | 外部任务执行接口和报告身份 |
| validation/service.py | TaskValidator | 逐任务调度／落盘、分技能汇总 |
| orchestration/contracts.py | DecisionProvider、DecisionContext、AcceptanceDecision | 接收标准接口，具体阈值不预填 |
| orchestration/outer_loop.py | OuterCycle | 两次训练→验证→判断的逻辑连接 |
| orchestration/io.py | parse/load_outer_contract、JsonCurationDataProvider | 严格 JSON 接入，不执行任意类型或代码 |

基础包不自动导入 torch。可选安装：`python -m pip install -e '.[training]'`。
本机实际测试 torch 2.5.1+cpu，无目标权重下载、GPU 或真实任务环境执行。

## 如何接入

```python
from self_curate.orchestration import OuterCycle
from self_curate.training import TrainingService
from self_curate.training.torch_backend import TorchSFTBackend
from self_curate.validation import TaskValidator

cycle = OuterCycle(
    stopped_search_result, immutable_or_enriched_view,
    expected_model=system_saved_model_ref,
    expected_batch_id=system_saved_best_batch_id,
    trainer=TrainingService(TorchSFTBackend(model_store, backend="real", device=device)),
    validator=TaskValidator(task_runner), suite=development_suite,
    subset_recipe=subset_recipe, curation_recipe=curation_recipe,
    curation_provider=external_curation_provider,  # 可暂缺，届时等待数据
    acceptance_policy=None,                    # 标准待填
    baseline=compatible_baseline_or_none,
    history_artifacts=published_inner_history_refs,
    output=new_attempt_directory,
)
state = cycle.run()  # 第一阶段训练／验证完成后等待判断
cycle.configure_acceptance(policy_artifact)  # 仅能配置一次
state = cycle.run(decider=my_decision_provider)
final_model = cycle.accepted_model
```

系统锚点须来自调用方保存的 ModelRef 和最佳 BatchManifest。输入必须正常停止，无在途操作且 best 可行；不能用最后候选替代 best。后补特征允许，但内容、token、mask 和训练顺序不能变。`step()` 一次推进一项工作；`run()` 遇缺依赖／证据不足／终态返回。

## 数据接口

`TrainingData(kind, source_hash, tokenizer, examples, provenance, backend)` 中 examples 为按训练顺序排列的 `tuple[TrainingExample]`。每例含 ID、content_hash、完整 input_ids 和 labels。labels 只允许原 token 或 -100，首 token 必须忽略；拒绝空监督、重复 ID、错 tokenizer/template、超长和未知字段，不裁剪或重抽。

子集由 `prepare_subset` 从 M1 导出，原样复用 assistant mask。source_hash 为原 batch_id，P0 TrainingRecord.training_data_hash 复用原 BatchManifest.training_data_hash。

筛选轨迹由 `CurationDataProvider.prepare(handoff)` 提供：kind 为 curation，source_hash 必须等于 handoff.record_hash。handoff 绑定停止搜索、最佳请求、原模型、原批次与历史 artifact。外部生产者负责总结、质量和 assistant 掩码；本轮不生成训练数据。仅有 token/labels 不能证明其来源和角色对齐真实。

也可在 `awaiting_curation_data` 时 `cycle.supply_curation_data(data)`；文件接入用 `JsonCurationDataProvider(path)`。`load_outer_contract("SFTRecipe"/"TaskSuite"/"TrainingData", path)` 支持严格解码。

第二阶段有自己的数据身份和 P0 TrainingRecord，不将其伪装成原任务子集。旧 WorkflowBundle/RoundRecord 是单训练结构，保持兼容，不把第二阶段塞入原单阶段工作流。

## 训练语义与模型适配

SFTRecipe 显式填写 learning_rate、epochs、accumulation_steps、max_sequence_tokens、seed；支持 AdamW/SGD、weight_decay、可选 max_grad_norm。当前 Torch 后端：

1. 从 ModelStore 加载隔离可变分支，只优化 requires_grad=True 参数。
2. micro-batch=1，按原顺序重复完整 epoch，不重抽、不混接轨迹。
3. causal shift 后按累积窗口所有监督 token 求 CE 总和，再除以窗口监督 token 总量；尾窗口按实际量归一，不做序列等权平均。
4. 两阶段各建新优化器；第二阶段接续权重，不接续第一阶段 optimizer 动量。
5. 保存实际权重、ModelRef、checksum、训练配方／参数名、optimizer／RNG 状态。后续加载实际 checkpoint，不用改 ModelRef 冒充训练。
6. 保护调用方 Python／torch RNG。训练后、验证前、接收前核验本地 checkpoint；改写文件不能继续使用。远程对象由 adapter 验真。

ModelStore 实现 load_branch、save_branch、load_checkpoint；必须加载确切权重和训练参数范围，不得共享 accepted 模型的可变参数。具体 Transformer／PEFT loader 尚需接入。

接续已训练 checkpoint 时，Torch 后端核对上次保存的可训练参数名；不能在相同参数身份下意外解冻基座。

通用 StateDictModelStore 接受 model_factory、initial_model、initial_state，加载现有本地 torch state_dict；初始 ModelRef.weights_hash 必须是初始文件 SHA256。factory 必须返回新实例。adapter store 声明 checkpoint_kind=adapter，起始模型必须有 adapter_hash；更新实际 adapter hash，保持基座 weights_hash。full store 处理无 adapter_hash 的完整模型。

训练 loss 仅为拟合诊断，不用于声称能力提升。本轮未实现分布式、AMP、scheduler、特定目标模型 loader 或训练中途恢复。

## 任务验证与判断

TaskSuite 固定 development TaskRecord、seeds、sampling/protocol artifact、task_mean/rollout_mean；拒绝 test。两次使用同一 suite。

TaskRunner.execute(request, output) 必须加载 request.model_ref 和 request.checkpoint，实际自由生成、调用工具、落实 TaskBudget 和协议、评分并保存完整轨迹。返回结果绑定 request.record_hash，含 actions/observations、token/mask hash、reward、stop_reason 和增量 usage。

TaskValidator 保存逐次请求／返回／RolloutRecord／CostEvent，产出现有 P0 CapabilityReport。TaskEvaluation 再核对 task/seed 完整覆盖、模型、费用与实际均值。task_failed 可有真实零奖励；environment_failed/cancelled 必须 reward=None。任一任务未评分则整份报告 invalid，均值缺失，不能接收。预算耗尽只有后端实际评分后才能带奖励。

0.8.1 的 TaskEvaluation 还保存 execution_requests/results，核对精确协议、checkpoint、attempt、输出和成本。旧 JSON 无这些字段时只读可加载，protocol_verified=False；须使用 load_task_evaluation 配合可信模型／suite／checkpoint 锚点核对原请求与结果文件，才能作为判断或 baseline 证据。该属性不认证远程实际执行真实性。

真实任务、模型生成、工具与评分器留在 TaskRunner；本轮没有真实 benchmark 成绩。初始 baseline 可缺，缺失时不宣称测得增益；传入时须精确匹配当前模型与 suite。第二阶段自动用第一阶段已接收报告作为 baseline。

判断器返回 `AcceptanceDecision(context.record_hash, outcome, reason, evidence)`，outcome 为 accept/reject/needs_more_evidence。context 绑定当前父模型、该阶段训练、确切能力报告、baseline 和固定 policy。手动接入可用 submit_decision。重复同一决定幂等，旧上下文不能应用或逆转终态。

待填：目标增益、保持能力退化、最少有效任务数、统计证据与预算等。无默认阈值、无默认接收。证据不足后可显式 `request_revalidation(reason=...)`；新复测独立计费，不重训。

同一补证上下文再次 run/step 不重复调用判断器。新报告必须得到自己的补证决定，不能用上一份报告的决定继续派发复测；调用方也可以显式 submit_decision 提供补证后的终局判断。

## 持久化与实现边界

每次新目录保存 handoff、数据、配方、suite、训练与验证目录、判断上下文、追加 journal 和 accepted.json。accepted 指针原子替换，只属于本次 attempt；不自动提交全局服务／优化器／角色版本。下一次内层需用 accepted_model 创建新 Team/context/信号命名空间。

cycle.updated_records 交出带 M1 训练历史的新记录：子集训练完成标 pending，判断后 accepted/rejected；原池不改写。外部可用已有发布接口保存。筛选轨迹曝光由独立请求／回执／判断保留。失败时仅保留实际报告的处理 token，不猜哪些样本已优化。

cycle.ledger 合并内层、训练、任务、汇总、失败和复测物理费用，不重复计账，未知量为 None。全部 journal／完整性校验 I/O 尚未纳入统一成本。

本轮有证据持久化，没有 OuterCycle 自动崩溃恢复；不是完整 M9。同名输出拒绝覆盖，失败不自动重发；进程在途退出须先人工核对已有调用和权重。

## 本地验证与后续

`python -m self_curate outer-demo --output runs/outer-demo` 运行显式 mock，不训练目标模型、不测能力；输出 training_executed=false 和 improvement_verified=false。`python -m pytest -q` 包含独立梯度参考、冻结参数／RNG、本地 checkpoint 接续、完整内层到两次实际微型 CPU SFT、adapter 基座不变、接收／退回／等待／失败与 JSON 边界。

后续按项填入 model store、筛选轨迹生产者、真实 TaskRunner、固定判断标准；真实权重和任务环境需独立验收。
