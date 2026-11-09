# EXP001：Random 与固定 M0 策略

版本 0.16.0。唯一交接：[EXP001_SERVER_V3.md](handoffs/EXP001_SERVER_V3.md)。
部署、八卡服务、容量和恢复边界见 [服务器部署](SERVER_CONTAINER_DEPLOYMENT_ZH.md)。
来源见 [sources.json](../configs/experiments/exp001/sources.json)。全部 5016 条准入时选 501 条，
三波各 167；真实服务器验收仍待执行。

## 本次定义

采用用户此次说明覆盖旧递进文档中“两次尝试”的阶段数：**一次初始观察 + 三波任务数据训练 = 四个观察点**。原研究文档保留，避免改写历史。

共同候选池是历史精选原始5016行经过质量准入、技能标注并去重后的池，保留原始messages，不与较新tool split拼接替代。总选样数 `K=floor(N×0.1)`，不从普通轨迹补齐。原始完整任务ID集仍参与开发/测试隔离，不能只隔离最终10%。

| 路径 | 选择 | 训练 |
|---|---|---|
| Random | 使用一个预设 seed，一次均匀无放回选 K 条 | 按固定顺序分三波，连续训练一遍 |
| static_m0 | M0 开发测评 → 全候选池 NLL/entropy → Lopt 一次提案 → Lexec 一次执行 | 同样三波；程序、阈值、M0 信号以后不变 |

Random 可独立运行，`modes=["random"]` 时不依赖 Lopt、Lexec 或指标计算。联合矩阵中的三项初始资产属于 static_m0 的公共前缀，不能把这些费用说成 Random 算法必需成本。

两个学习率、同一 seed，得到 Random×2、static_m0×2 四个分支。每个分支从同一初始模型/adapter 开始；同一模式的两个学习率使用同一份精确 BatchManifest 和顺序。模型参数正常学习，“静态”指选择策略与数据指标不动态更新。

本次不运行 Leval、内层搜索、NICE、梯度选样、控制轨迹 SFT、接收/退回或验证集选权重。初始模型和每个训练节点都在同一开发 suite 上记录能力；所有训练结束并封存后，才执行各 checkpoint 的最终 benchmark。M0 只测一次、共同引用。

## 数据波次、batch 与训练语义

整条轨迹是最小单位，assistant-only 监督，完整系统/用户/工具上下文保留。例如 K=500，三波为 167/167/166；训练只遍历每条一次，不把三个节点分别再跑完整数据。

`TorchContinuousBackend` 使用同一个 optimizer、同一参数分支与连续 RNG。阶段保存权重、optimizer、RNG、数据游标、监督 token、step 数、训练配方与文件 checksum。保存/验证回调不会改变后续训练 RNG。波次末尾不足有效 batch 的窗口明确完成一次较小更新，并记录实际曝光。

选择侧 P1 使用 `effective_batch=1`，用于保持准确样本数；训练侧有效 batch 独立固定为 8。不会因 P1 整除规则把 500 条悄悄变成 496 条。`count_context()` 只建立容纳 K 条完整轨迹的 token 包络，不宣称两模式严格 token 等预算。实际监督 token、输入 token 和成本分别保存。

通用 Torch 参考后端物理 micro-batch=1，gradient accumulation=8，单分支 world_size=1，因此有效 batch=8。四分支并行不会让某个模型的 global batch 变成 32，也不是四卡 DDP。AdamW、weight_decay=0、max_grad_norm=1、恒定学习率、无 warmup，epochs=1；这是本次小规模方案，不冒称论文原配方。

## 参数来源与经验选择

- 模型：`Qwen/Qwen3.8-27B`，本次查到的固定 revision 为 `37561816e66b5b9153b1b64c10968c18c1e6f4a0`。官方配置是带视觉组件的模型，目标适配须使用对应多模态加载类并核对实际结构，不能随意换成普通 CausalLM 自动类。[官方模型元数据](https://huggingface.co/api/models/Qwen/Qwen3.8-27B)
- 学习率：`5e-6 / 2e-5`，作为当前小数据、单遍 LoRA 的两档工程试验值。Qwen 旧版 LoRA 教程示例采用 `5e-6`；它没有证明这一数值适合本模型。[Qwen 官方教程](https://qwen.readthedocs.io/en/v2.5/training/SFT/llama_factory.html)
- SWE-smith 原论文对旧 Qwen 做全参数微调，给出 `5e-5`、最多 3 epochs、32768 上下文；没有提供可直接照搬为本模型 LoRA 最优值的证据。这里保留较保守的两档，不引用论文冒充匹配实验。[论文 F.1](https://arxiv.org/html/2504.21798v1#A6.SS1)
- seed=42 是预先固定的单次可行性 seed，沿用常见工具默认；没有对多个 seed 试跑择优，也不能凭它报告稳定增益。[Transformers 参数文档](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- 计划采用 BF16 底座、文本线性层 LoRA，rank=16、alpha=32、dropout=0；视觉与底座冻结。真实目标模型的 adapter 创建/保存/重载需要 ModelStore 适配与服务器实测。

训练规模由 10% 样本和一次遍历固定；“尽量大”用于上下文容量与吞吐调优，不能扩大监督 mask、重复曝光或偷偷改训练轮数。131072 是待服务器容量验收的上下文方案，不是已测可用值。

## 八卡组织

1. 初始化阶段建议由服务器评分适配器在八卡分片执行固定 M0 推理，避免同一 27B 模型每条记录重新装载。当前 CPU 参考 scorer 不提供这套八卡服务部署。
2. 训练阶段，四个独立子进程分别绑定四张授权 UUID，另一半 GPU 提供固定模型身份的任务推理服务。任务环境/容器并行不与训练进程共享可变模型对象。
3. 通用控制器的开发测评逐题调用既有 TaskValidator；四分支可并行。服务器 TaskRunner 须把请求路由到对应 checkpoint，并管理模型加载/工具执行/服务槽位。不能只改显示 model ID 而实际用错参数。
4. 训练全部封存后，benchmark 执行器最多 16 个并发任务；服务器适配器可利用全部八卡、每卡两个任务槽。当前 checkpoint 任务完成后再进入下一个模型身份。
5. Full/Lite/Verified 允许最终测试间重叠；完全一致的模型/任务版本/seed/协议可复用逐题执行，保存原请求和消费回执，不重记费用。协议或输入不同不复用，缺失/失败不补零、不按分数重试。

micro-batch=1 是当前参考实现；真实长轨迹需要 gradient checkpointing、受支持的内存优化损失和驻留评分实现。普通全词表长序列 logits 张量可能无法容纳，不应直接用“有八张 H200”推断单卡训练可行。真实适配若不能满足容量/速度验收应 blocked；优化方案在冻结前确定，不在正式过程中静默缩上下文或改样本。

## 公共资产与产物

`AssetStore` 管理：准备好的候选/标签/任务清单、M0 开发集能力、全池 M0 NLL/entropy、一次 Lopt 程序及其真实 before-context。后续相同初始条件的实验读取消费回执，原物理成本保留，当前不重复调用/计费；改变模型、输入、模板、计算身份、prompt、可见反馈或策略约束时重新核对 key。

初始 Lopt 包含由代码校验过的八类算子参数模板、完整 ProgramSpec 示例、各技能可用数量及 M0 指标分布；不要求模型猜内部接口。初始输入不透露后续实验变体或测试结果。随机 seed 必须与系统固定值一致，反馈引用必须对应实际 M0 开发报告。

复用程序仍须执行当前数据清单的筛选，不复用旧选集；Lexec 的独立角色记忆保持隔离。后续 V1/V2 控制器须使用相同公共前缀构造方式，不能仅因为都叫“M0”就强行命中。

```text
shared-assets/                  # 多实验共用，独立存储；不提交 Git
run-id/
  run.json                     # 实际配置、源码 hash、环境、可用 commit、preflight
  source-checksums.json
  handoff.md                   # 提供 --handoff 时复制的冻结文本
  preparation-reuse.json
  frozen-inputs.json            # 精确 AssetRef
  initial/                     # 三类初始资产消费与原生成上下文
  selections/{random,static_m0}/
  training-plans.json          # 全部清单、顺序、波次与四份固定请求
  worker-logs/                 # 并行子进程日志
  branches/<mode>-lr<1|2>/
    training/checkpoints/stage-<1|2|3>/
    development-<1|2|3>/
    branch-result.json
    branch-complete.json
  training-sealed.json          # 所有训练完成后才生成
  benchmark/<checkpoint>/<suite>/
  controller-ledger.json        # 新的初始化、选样、训练、开发调用
  checkpoints.csv
  summary.json                 # benchmark 分列各自实际/复用费用
  completion.json              # 完整输出文件集及 checksum
```

上游数据标注成本保存在 preparation asset 的 origin ledger；最终测试成本保存在各 benchmark report，不与优化费用混在一起。`all_reports_valid=false` 时不能据“流程结束”宣称得到有效研究结果。

每次执行自动生成独立 run_id，逻辑实验 ID 不充当物理费用事件 ID。每分支 checkpoint 另存 `exposure-index.json` 和版本化 `exposure-history/`，登记真实外层任务使用，供未来刷新模块读取或 fork；不污染公共池和其他分支。部分更新范围不明时标 unresolved；缺少 branch-complete 的在途运行不能只凭旧 index 推进下一轮。

全池评分重载中的费用事件改为一次建立 ID→event 内存映射，避免每条前向反复扫描全账本；原校验、数值协议与计算缓存身份不变，不增加持久化 operator 索引。

复用回填一次解析整批的不可变 AssetRef，避免每条记录重解析完整 manifest；不缓存可被调用方修改的 manifest 字典。总墙钟使用 summary.seconds；并行 ledger 各事件的墙钟加总不是作业历时。上游准备、公共前缀和最终测试成本保持单列。

## 执行与恢复边界

已实现本地入口：

```powershell
python -m self_curate.experiments.exp001 --config configs/experiments/exp001/mock.json --output runs/exp001-new --assets runs/exp001-assets --handoff docs/handoffs/EXP001_SERVER_V3.md --parallel-training
```

串行 CPU 对照省略 `--parallel-training`。`modes` 可以只留 Random 或 static_m0。`--resume` 可从冻结选样后的原运行自动恢复：读取已提交的 optimizer-step 权重/optimizer/RNG/游标，跳过完整分支和已完成的逐题验证/benchmark；已完成整场运行仍只读核验。已派发但缺少回执的外部调用和未冻结的初始化/选样须先对账，不能悄悄重复。具体边界、进度查看命令与真实后端责任见 [运行核对](SERVER_CONTAINER_DEPLOYMENT_ZH.md)。

正式模板 [server.template.json](../configs/experiments/exp001/server.template.json) **不能直接执行**：该文件含待解析字段；须由 bootstrap 结合真实资产、GPU UUID 和容量回执生成最终配置，不能把模板当成已验收配置。资源和来源建议见 [server_bootstrap.template.json](../configs/experiments/exp001/server_bootstrap.template.json)，填写持久存储、Teacher 与执行参数后由 bootstrap 解析。

## 一次性的服务器接口

`ExperimentPorts` 见 [ports.py](../src/self_curate/experiments/exp001/ports.py)：

- `preflight(config)`：真实返回 backend、passed、全部授权 gpu_uuids 与通过验收的 max_context_tokens；补来源/环境/监督 mask/任务容器/模型装载与 checkpoint smoke 证据。
- `prepare(config, output)`：产出 `PreparedInputs`；可通过 `load_prepared_manifest()` 消费已完成的数据准备产物。它验证 M1、标签、版本和显式 ID/家族隔离，不代替 Teacher 标注、作者身份恢复或近重复审计算法。
- `scorer`、`validator`、`client`、`training_backend(branch)` 和 `benchmark_runner`：对接已有计算/角色/训练/任务协议。scorer 的 `score()` 结果及输出须能由既有严格 loader 重载。
- `context()` 可直接复用 `count_context()`；objective、edit_policy 固定在适配器/配置，不让 Lopt 改写。
- `attach_training(config, model)`：子进程仅绑定已锁定 M0 与后端，不能重新下载/标注/评分或生成策略。
- `release_initial()`：启动训练进程前释放初始化阶段驻留资源；推理服务可按已冻结资源计划继续提供任务执行。
- `benchmark_runner` 须支持配置的线程并发，按完整 ModelRef/checkpoint 装载；不能持有一个会被其他请求偷偷改参的共享模型。

prepared manifest schema=1，包含 files（records/model/development/benchmarks 四个相对文件）、checksums、skills、source_lock、isolation_evidence 和 origin_ledger。records 为已有 M1 JSONL；model 为 ModelRef JSON；development 为 TaskSuite；benchmarks 为 BenchmarkSuite JSON 数组且任务保持 test split。原始下载、外部标注和来源审计要求继续按 [数据准备规范](RSI_Data_Preparation_Codex_Spec.md) 完成。来源下载、准入/隔离、Teacher 标注、任务注册已在 preparation/server 模块接入；实际数据与容器仍需服务器验收，未支持的原生环境配置会阻塞而非静默跳过。

改代码只在开发目录进行；实验运行时冻结代码/环境，输出、数据、权重和服务配置放 Git 忽略/仓库外目录。不在服务器为每个实验复制算法，不由 Agent 提交/推送源码。状态和研究原文不改写。

## Batch 与学习率依据


| 项目 | 当前方案 |
|---|---|
| 每个训练进程的 physical micro-batch | 1 条完整轨迹 |
| gradient accumulation | 8 |
| 单分支 data-parallel world size | 1 |
| 单分支 global/effective batch | `1 × 8 × 1 = 8` 条 |
| 并行分支 | 4 个独立实验，不是 DDP；不能把每分支 batch 写成 32 或 64 |
| 完整训练遍历 | 1 遍，无重复采样、无截断、无 drop_last |
| 501 条的阶段曝光 | 167、334、501 |
| 501 条的累计 optimizer steps | 21、42、63；波尾各按 7 条的小窗口更新并记录 |
| loss 权重 | 每个累积窗口按实际 assistant 监督 token 归一 |
| 学习率 | 5e-6、2e-5；恒定 LR，无 warmup |
| seed | 42；一个预设 seed，不据此报告多 seed 稳定性 |

`5e-6` 可在 [Qwen 官方旧版 LoRA 教程](https://qwen.readthedocs.io/en/v3.0/training/llama_factory.html) 找到，但该页明确仍以 Qwen2.5 为例，**不是 Qwen3.8-27B 在本数据上的最佳参数证据**。`2e-5` 是本实验为小数据、单遍训练预先设定的较高一档工程探索值。SWE-smith 论文中的全参微调配方也不应直接冒充当前 LoRA 的验证结果。两档均保留所有节点，不利用 benchmark 选学习率或权重。

监督 token 总量来自所选完整轨迹及 assistant mask，不能任意调大。131072 是待容量验收的上下文上限；吞吐优化不能扩大监督 mask、重复数据或悄悄截短轨迹。
