# Leval 两种验证方式与历史反馈（0.9.1）

0.9.1 的 [两轮复查](HISTORY.md#leval-review) 补齐执行版本/数值条件、自动基线身份、M1 使用历史恢复、失败费用与预算预检。0.9.0 的旧产物保持只读；执行协议发生变化时新建会话，不静默混算。

2026-11-06。入口是 `self_curate.valuation`，沿用已有 `ValidationProvider.evaluate(request) -> ValidationResult`。`InnerSearch`、Leval→Lopt 和外层训练不需要换一套状态机。

本次用户明确增加候选子集 SFT 验证。这扩展了原蓝图“内层不为候选 SFT”的约定：**三个角色和搜索参考模型仍冻结，只有隔离的候选分支允许训练**。候选分支不提交、不作为下一个候选的训练起点。原研究文档保留原文。

## 1. 两种方式的含义

| 配置 method | 类 | request.measure | 输出含义 | 更新哪些参数 |
|---|---|---|---|---|
| `nice` | `NICEValidator` | `predicted_gain` | 单 checkpoint 的 NICE 风格 fresh-Adam 余弦代理，无量纲 | 不更新；只在隔离模型上求梯度 |
| `sft` | `SFTValidator` | `reward_delta` | 候选子集 SFT 后，每技能任务平均奖励减冻结基线平均奖励 | 只更新候选分支 |

`configs/leval.json` 只保存 `method`。调用方用 `load_method()` 读取后，把该方法的依赖交给 `create_validator()`。无自动 mock 回退。`backend=mock/real` 表示执行域，与 `method` 是两个维度。

方法、参考模型、任务/参考 rollout、配方和实现身份绑定到 `provider.protocol`。**先选方法，再构造 ValidationRequest 和 RoleTeam/InnerSearch**。一个搜索会话内不可切换方法、模型或评价协议；换方法时新建会话和输出目录。两种分数不可混排，约束边界也应按各自单位明确配置；Leval/Lopt 不改这些边界。

Torch NICE/SFT 显式禁用调用方 autocast，按参数原生 dtype 前向，CE 用 float32；这不把 BF16 参数强制转为 FP32。后端身份还绑定 torch/设备、模型 factory、数值条件与实现源码。真实 SFT 训练器和任务执行器需提供包含配置版本的 `implementation_hash`。所有身份变化都会阻止沿用旧 protocol。

## 2. NICE 的具体算法

参考用户项目的 [scoring.py](https://github.com/weeks-we/NICE-Subset-Validation/blob/main/nice_validation/scoring.py) 与 [METHOD.md](https://github.com/weeks-we/NICE-Subset-Validation/blob/main/docs/METHOD.md)，在独立项目中实现数学内核，不导入邻项目代码和实验配置，不继承其模型名、任务数、硬件或抽样规则。

1. 对清单中每条完整样本计算 assistant 监督位置的 **token 平均 CE 梯度** `g_i`。
2. 对固定开发任务 t 的每次实际 rollout，按真实输入和生成动作 mask 重放，计算动作 NLL 的 **token 和**。任务方向 `h_t = sum(reward * grad(NLL_sum)) / 固定 rollout 数`。奖励零的模型失败仍占分母；环境失败/未评分不能冒充零。
3. 对样本原坐标梯度做全局范数裁剪，`c=min(1, max_grad_norm/(||g_i||+1e-6))`，再做 fresh-Adam 首步方向 `a_i=(c*g_i)/(abs(c*g_i)+adam_epsilon)`。不创建优化器，不调用 `step()`。
4. 可选 CountSketch-v1：先变换再投影，样本与任务共享参数顺序、全局坐标、维数和 seed。不能对已经投影的向量再做逐坐标 Adam 变换。
5. 每样本、每任务求 cosine，零方向记该任务的 0，保留固定任务分母。每技能先对其任务平均，再对全部选中成员平均，与参考项目“子集成员分数平均”的口径一致。不是聚合 batch 梯度的 cosine，也不是按监督 token 加权成员分数。
6. 原始梯度 cosine 同时落盘作诊断，排名使用 fresh-Adam cosine。所有任务方向均零时返回 invalid，`total_value=null`，搜索不能据此优化。

这里 h 使用奖励加权 **NLL 梯度**，与已有 `TargetGradientSpec.objective=reward_ascent` 的目标梯度符号不同，不能直接混用。对成功动作训练时 cosine 正，SGD 沿负 CE 梯度更新可提升成功概率；测试用可解析 Bernoulli 策略的有限差分核对正负号。

这是一种单 checkpoint 的 NICE-inspired 适配。fresh Adam 的方向代理不等于多步 AdamW 的实际奖励增益；没有复现论文的 warmup、多 checkpoint、已有 Adam moments。这里 `predicted_gain` 是沿用旧请求契约的字段名，evidence 额外标注 `unit=dimensionless_proxy`，不解释为奖励单位。

### 接入目标轨迹与梯度后端

`NICEReference(evaluation, replays)` 消费现有 `TaskEvaluation`，要求 development suite、完整任务/seed、有效任务结果及逐请求协议证据。`ReplayTrajectory` 按**完整 RolloutRecord hash**连接，不按行号。每次重放输入为 `TrainingExample` 的真实 input_ids/assistant labels；标签数须等于该 rollout 的 generated_tokens，非生成上下文不监督。工具结果仅作输入。

实际任务 runner/重放适配器负责提供真实输入前缀、生成位置及来源 evidence；计数和 hash 绑定不代替生成 logprob 对齐或源文件真实性验收。本轮不下载任务、启动容器或实现目标模型专用 loader。

`TorchNICEBackend(store, backend=..., device=...)` 已实现实际 autograd。复用外层 `ModelStore.load_branch()`，要求隔离分支、准确 ModelRef/可训练坐标，只对 `requires_grad` 参数求导；LoRA store 应冻结基座并记录 adapter 身份。`parameter_coordinates(model)` 提供参数名/形状/dtype 的 hash 和总参数数，须与 `NICERecipe` 一致。模型 eval 关闭 dropout，autograd 保持开启；调用方 RNG 恢复，参数及 buffer 前后 hash 相同。

样本缓存只计算选中成员，key 包含模型/adapter/tokenizer 身份、完整样本 tokens/labels/content、配方/坐标/投影和代码实现 hash；任务缓存另含完整 reference hash。它们是可重建的内存缓存，不是 operator 持久索引。每次 run 保存实际 vectors.npz、gradient_identity.json、request/result/evidence 和 checksum。前置 reference 的原成本作为导入证据保留，不在每个候选重复收费。

```python
from self_curate.valuation import NICERecipe, NICEReference, create_validator
from self_curate.valuation.torch_nice import TorchNICEBackend, parameter_coordinates

# evaluation 和 replays 来自同一冻结模型、固定开发任务的真实执行/重放适配器。
reference = NICEReference(evaluation, replays)
# coordinates/count 由服务端已核验模型的 trainable 参数取得。
recipe = NICERecipe(coordinates_hash, parameter_count,
                    adam_epsilon=chosen_epsilon, max_grad_norm=chosen_clip,
                    projection_dimension=chosen_dimension,
                    projection_seed=chosen_seed)
provider = create_validator(
    "nice", model=frozen_model, view=original_view, reference=reference,
    recipe=recipe, engine=TorchNICEBackend(store, backend=backend),
    output="runs/new-nice-evaluations",
)
```

## 3. 候选 SFT → 任务验证

`SFTValidator` 复用外层 `TrainingService`、可选 `TorchSFTBackend`、`TaskValidator` 和同一份 `TaskSuite`。任务自由执行/环境/评分仍通过现有 `TaskRunner.execute(TaskRunRequest, output)` 接入；不能用 teacher loss 代替任务奖励。

顺序是：先得到冻结模型的同条件任务基线 → 精确候选子集 SFT → 候选 checkpoint 在同 suite 验证 → 每技能做差 → 固定 ObjectiveSpec 汇总/检查约束 → Leval 反馈 → Lopt。基线已有时可传入有协议证据的 `TaskEvaluation`；没有时首次调用计算一次。基线无效时不花费候选训练；候选任务验证无效时保留训练与费用，结果不可排名。

候选数据取 `BatchManifest.training_order` 中的 ID/content/tokenization/mask；不重新采样、不裁剪、不接上个候选的权重。每次新建输出和训练/任务 attempt。所有候选从当前冻结 ModelRef 出发，各自新优化器；不训练筛选轨迹（筛选轨迹属于外层第二阶段）。

```python
from self_curate.training import TrainingService
from self_curate.training.torch_backend import TorchSFTBackend
from self_curate.validation import TaskValidator
from self_curate.valuation import create_validator

provider = create_validator(
    "sft", model=frozen_model, view=original_view,
    trainer=TrainingService(TorchSFTBackend(store, backend=backend)),
    validator=TaskValidator(task_runner), suite=fixed_development_suite,
    recipe=explicit_sft_recipe, output="runs/new-sft-evaluations",
    baseline=optional_verified_baseline,
)
```

从现有 P1 接入时，构造绑定该协议的请求：

```python
from dataclasses import replace
from self_curate.valuation.io import load_method

method = load_method("configs/leval.json")
# settings_by_method 由调用方提供各方法的模型/数据/后端依赖。
provider = create_validator(method, **settings_by_method[method])
request = replace(unvalued_request, protocol=provider.protocol,
                  measure="predicted_gain" if method == "nice" else "reward_delta")
# 仅限尚未绑定 RoleTeam/InnerSearch 的原始 P1 请求。
# 如需修改 objective 的指标单位/边界，须在 prepare_selection 之前配置。
search = InnerSearch(team, initial, request, edit_policy, search_id=search_id)
search.step(provider)  # 或 run(provider, stop_condition=你提供的退出条件)
```

completed SFT 通过 `provider.updated_records` 返回独立的 M1 candidate_sft 记录，decision 仍 pending；不污染原始池、不伪造 accepted_sft，也不把任务奖励高低自动解释为模型接收/退回。外层正式训练/判断流程保持独立。

保存基线：`anchor = provider.save_reference(new_directory)`，将 anchor 与该搜索的 protocol 一起保存。稳定边界续跑时用 `SFTValidator.restore_reference(reference_json, expected_hash=anchor.sha256, expected_protocol=saved_request.protocol, **原依赖)`。它核对模型/suite/配方、原基线及协议并只读恢复，不重跑/重复收费。搜索检查点仍由 InnerSearch 保存；不宣称实现外部在途作业的崩溃对账。新建一个空 provider 不等于恢复旧基线。

schema=2 reference 同时保存自动基线 namespace 和紧凑 candidate_sft 使用记录，按训练输入 ID/content/tokenization 恢复并单调合并。自动基线的新 provider 拥有新 protocol；不能凭同配置重新测基线后接入旧搜索。显式导入同一 TaskEvaluation 的参考身份保持稳定。候选长度在基线调用前预检；无效基线通过 attempted_baseline 保存，不能进入缓存。

## 4. Leval history 与预设 prompt

ReplayTrajectory 还必须显式携带 tokenizer/template 身份，并与评测模型一致。旧 RoleEvaluation 的 `validation_implemented=false` 字段为兼容旧 artifact 而保留，指解释角色自身不执行验证；实际数值执行证据在 `ValidationResult.evidence` 和新后端的运行文件中。

固定 prompt 仍在 `configs/roles/leval.json` 与 `lopt.json` 的 `instructions`，正文继续允许为空。本轮不擅自确定研究提示词文案。模型调用的结构化输入增加稳定槽位 `leval-feedback-slots-v1`：

| 槽位 | 内容 |
|---|---|
| current_result | 当前实际程序/批次 ID、成员数、监督 token、每技能值、总值、约束、证据、有效状态 |
| related_history | 同 ModelRef、ObjectiveSpec、selection context、measure、protocol、backend 的已完成 Leval 观察；最多 8 条 |
| comparison | 相对上一条 valid 观察、窗口内最佳可行观察的总值变化及每技能变化；缺依据为 null |
| feedback | 原代码产生的 LevalFeedback：观测事实、约束失败和待验证的修改假设 |

Leval 从 `work.prompt_slots` 读取，Lopt 从 `work.handoff.prompt_slots` 读取；三角色仍共享权重、独立固定 prompt 和私有 history，跨角色只传这份显式 handoff。未来填写 instructions 时按槽位引用即可，无需修改数值内核或搜索循环。模板和结果均通过 JSON 数据封装传递，不把历史内容执行为系统指令。

数值比较由代码完成；模型解释仍只能返回 commentary/hypotheses，不能改分数或约束。invalid/pending 不产生可排名的“改进”。同一结果去重；旧 history 缺完整 request 时不猜其可比性。`RoleEvaluation.context` 保存可重核的观察和 hash，cycle 发布/加载及搜索保存/恢复均保留；旧 cycle 无此字段仍兼容。窗口最佳不是整个搜索的全局最佳，后者继续由 InnerSearch 精确维护。

## 5. 证据与费用

每次评价写新目录，拒绝覆盖。保存 request、训练/任务原回执、baseline、梯度身份/向量、结果、上下文和 checksum；失败保留 failure.json。验证失败也将可用 ResourceUsage 传回 InnerSearch，不能因抛异常抹掉已发生的计算。

SFT 内部物理训练/任务 ledger 全量保留，InnerSearch 仅计一次本次评价的汇总成本，避免双计。ResourceUsage 某项存在未知就保留 null；`evidence.usage_summary` 同时保存已知小计和未知事件数，不能把其他后端的未知费用补成零。基线在第一次实际调用收费，之后只引用；外部导入的旧基线不再次收费。

如果注入服务没有返回 ledger，保留其有效 usage 和一条缺少明细的边界调用，其他费用保持未知；不从缺失报告猜测 rollout 数。若服务明确返回有效的空 ledger，则保留“没有派发物理工作”的语义。

直接拒绝选中样本与开发任务的已声明 ID/来源 family 重叠。原始数据来源闭包、近似题面去重、预训练污染、真实任务评分和模型加载真实性仍需数据/服务器适配器验收；最终 test 任务不可接入优化 suite。

## 6. 本机验收与后续接入

```powershell
python -m pytest -q
python -m self_curate leval-demo --method nice --output runs/leval-nice-new --project-root .
python -m self_curate leval-demo --method sft --output runs/leval-sft-new --project-root .
```

两个 demo 使用显式合成向量、权重回执和任务奖励，用于验证接线、三轮循环、历史比较及 Lopt prompt 消费，不代表模型能力提升。专项测试另外使用微型 CPU 随机模型，实际执行 autograd、监督 mask 对照、SFT 更新和冻结起点隔离。

真实接入只需提供：当前模型的隔离 ModelStore、固定 development TaskSuite/TaskRunner；NICE 另需该模型的任务执行及真实 ReplayTrajectory；显式 SFT/NICE 配方。数据标注、真实 GPU/模型容量和环境判分未在本机执行。内层停止条件、外层接受阈值、具体 prompt 正文继续由后续配置填写。
