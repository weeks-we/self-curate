# Self-Curate：整体配置与后续实验计划

更新：2026-11-15。适用当前独立项目 `self-curate` 的 EXP001；不继承相邻 NICE 项目的训练、抽样和预算配置。本页是配置快照；真实执行以本次生成并冻结的配置、来源锁和验收回执为准。

## 1. 实验一的完整流程

实验一比较 **Random** 与 **Static-M0（初始模型制定一次策略后固定）**。两种模式各跑两个学习率，共四个独立训练分支，都从共同 M0 adapter 出发。

```text
联网工作站：固定来源 → 下载模型、任务、轨迹和 benchmark
  → 少量 Teacher 试标 → 抽候选任务并标注 → 每主技能选 50 题
  → 保存开发集引用 → 排除这些任务对应的全部训练轨迹
  → 轻量硬筛 → Teacher 标注轨迹和证据 → 发布 M1 数据资产
  → 准备选定任务/benchmark 环境、缓存、镜像与环境锁
  → 集群离线和资源预检

GPU 集群：M0 开发集初测 + 全候选 M0 NLL/entropy
  → Leval 解读初测 → Lopt 制定一次策略 → Lexec 执行并记录
  → Random / Static-M0 各确定完整 K 条训练数据
  → 各学习率连续 SFT 一遍，在约 1/3、2/3、全部曝光处保存 checkpoint
  → 各节点跑开发集 → 训练结束后对 M0 与各分支最终节点跑三个 benchmark
  → 汇总日志、成本、结果并打包
```

实验一不调用 NICE，不做内层候选搜索，不动态刷新 NLL/entropy，不加入控制轨迹 SFT，不用开发集执行接受/回退。三段 checkpoint 是一次连续训练的阶段节点，不是重新初始化的三次训练。

## 2. 模型与软件环境

| 项目 | 配置 |
| --- | --- |
| Target | `Qwen/Qwen3.8-27B`，约 27B 参数 |
| Model revision | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| 正式加载类 | `AutoModelForImageTextToText` |
| 调整方式 | LoRA；基座冻结 |
| Agent harness | Claude Agent SDK `0.2.87` + Claude Code CLI `2.1.153` |
| SDK Python / CLI | `/opt/self-curate-sdk/bin/python` / `/opt/self-curate-sdk/bin/claude-pinned` |
| 模型推理服务 | vLLM `0.22.0`，本地服务与协议网关 |
| 训练框架 | Accelerate `1.15.0`，FSDP version 1，BF16 |
| 其他依赖声明 | torch≥2.1、transformers≥5、peft≥0.17、pyarrow≥15、huggingface_hub≥0.35、docker≥7 |
| 独立官方判分环境 | Rebench fork、SWE-bench Verified、Pro V2 tooling、Harbor |

上述带 ≥ 的是依赖范围，不是最终环境锁。工作站需生成准确 pip freeze、基础镜像 digest 和 harness commit 回执；Harbor、Rebench/Verified 判分工具完整 commit、Claude 二进制 checksum、基础镜像 digest 仍需在实际构建时落地。Dockerfile：`docker/Dockerfile.rebench`；SDK 依赖：`docker/requirements-sdk.txt`。

Claude Code 负责把模型调用组织为可执行 agent；vLLM 负责模型推理；Harbor/官方判分工具负责任务环境和结果判断。框架 workflow 负责串联这些模块。上下文管理使用 Claude 原生机制；旧字段 `recent_tool_observations=5` 不代表实际只保留最近五条观察。

## 3. 数据来源、规模与选样预算

| 用途 | 数据源 / split | 规模与使用方法 |
| --- | --- | --- |
| 任务来源 | `nebius/SWE-rebench` / default / test | 固定数据卡标明 test 共 21,336 题；另有 filtered 6,542 题，本实验没有改用 filtered |
| 轨迹来源 | `nebius/SWE-rebench-openhands-trajectories` / default / train | 作者数据卡报告 67,074 条轨迹，其中 32,161 条成功；成功任务数 3,792。实际导出后复核统计 |
| 任务候选范围 | 与成功轨迹关联的任务 | 再执行 benchmark/家族/近似题面等隔离，不是从全部 21,336 题无条件抽样 |
| 初抽任务数 | `min(eligible_tasks, max(1000, 200×skill_count))` | 六技能时目标先抽最多 1,200 题，seed=42 |
| 开发集 | 每个唯一主技能 50 题 | 目标 300 个不同任务；类别不足报告缺口，不复制题目凑数 |
| 训练总预算 | `K=floor(N_raw_success/10)` | 分母是硬筛前原始成功轨迹数，开发集隔离/硬筛/Teacher 排除不缩小分母 |
| 条件示例 | 若导出成功数确认为 32,161 | K=3,216，三段各 1,072；这不是绕过实际统计的硬编码 |
| Random | 准入后可用候选池随机抽 K 条 | 与 Static-M0 使用同一可用数据资产；候选不足停止，不偷换分母 |

来源 SHA：

| 来源 | revision |
| --- | --- |
| Rebench 任务 | `89cdfbab4ab1bd8f5a658bb212d1b63624f4f881` |
| Rebench 轨迹 | `35455389ab51bf5e2306bfd436ef72d0f98bf882` |

任务与轨迹按 `instance_id` 关联，并保存版本化 task_key、trajectory_id 和来源证据。同一任务可以有多条训练轨迹；默认不自动限制每任务条数，策略可显式使用 TaskLimitSelect。开发任务一旦选定，排除它对应的全部轨迹，同时执行来源家族和近重复隔离。

轻量硬筛：结构完整、工具调用对应、有效 assistant 监督内容、长度检查、完全去重、集合隔离；成功状态采用可信来源 resolved=1，不在准备阶段重放所有轨迹。整轨迹超长明确排除，不静默截断。样本数相同不意味着监督 token 相同，需报告两种模式的实际 token、长度和唯一任务数。

官方数据卡：[任务](https://huggingface.co/datasets/nebius/SWE-rebench/blob/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881/README.md)、[轨迹](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/blob/35455389ab51bf5e2306bfd436ef72d0f98bf882/README.md)。以上统计是作者发布口径，发布本次资产时保存实际导出统计与 checksum。

## 4. Teacher、技能与三个角色

技能采用 `coding-skills-v1`：控制逻辑与状态一致性、数据表示与变换、接口契约与组件协同、平台/依赖/构建兼容、资源生命周期与并发时序、算法/数学/计算效率。

任务标“需要的能力”：core / supporting / not_required / uncertain，另保存唯一 primary_skill、风险与原文证据。轨迹标“实际展示的能力”：demonstrated / not_observed / uncertain；保存上下文、assistant 动作、结果观察的证据引用。两者独立判断，不把任务标签直接复制给轨迹。

Teacher 返回结构化 JSON；本地按 ID、schema、引用原文校验并持久化，tokenizer 将 assistant 证据映射为监督 token 区间，再计算覆盖数量/比例。无法确定的样本不等于无技能；只有材料完整、全技能 not_observed 的样本才可按 exclude_no_skill 排除，并保留审计记录。

| Teacher 参数 | 正式准备 | 100任务＋100轨迹试标模板 |
| --- | --- | --- |
| 模型 | 主标注与复核均 deepseek-flash | 同左 |
| endpoint / key | https://api.deepseek.com / 环境变量 DEEPSEEK_API_KEY | 同左 |
| Thinking / effort | enabled / high | 同左 |
| 最大输出 / 超时 | 16,384 tokens / 900 秒 | 同左 |
| 输入保护 | 900,000 bytes，不等于 token 容量 | 同左 |
| 主标注调用数 | 按规模自动计算；规则 ceil(1.25×(任务候选+准入轨迹数)) | 最多 300 次 |
| 主标注费用保护 | ¥3,000 | ¥50 |
| 复核调用数 | 配置自动预算；关闭时不调用 | 最多 30 次，关闭时不调用 |
| 复核费用保护 | ¥1,500 | ¥30 |
| 费用保护费率 | 输入¥2/百万、输出¥8/百万；保守上限，不是实时账单报价 | 同左 |
| review_mode | **off**；可选 flagged / sampled / all | off |
| 抽检比例 | sampled 时沿用5%＋异常项规则；微型试标抽首条 | 受模式控制 |
| 同任务轨迹分组 | 开启，最多2条；组输入600,000 bytes；输出每成员6,000＋公共2,048 tokens | 同左 |

正式调用预算的实现见 teacher-budget.json；组请求、缓存命中、失败与重试会使实际调用数不同于轨迹数。不得把费用保护上限当成消费预测。试标配置 tokenizer/model 默认 null；要验收真实证据 token 比例，必须绑定已下载的锁定 tokenizer/model 引用。

三个角色共享 target 参数，prompt/history 隔离：

| 角色 | 实验一工作 | prompt |
| --- | --- | --- |
| Leval | 解读 M0 开发集测评，指出技能表现和证据限制；不编造 NICE 分值 | prompts-rebench/leval.json |
| Lopt | 根据初测、Leval 反馈、M0 NLL/entropy、标签和池统计，一次生成固定筛选程序 | prompts-rebench/lopt.json |
| Lexec | 监督执行组合筛选程序，检查每步子集及约束，保存执行记录 | prompts-rebench/lexec.json |

上述 prompt 均位于 `configs/experiments/exp001/`；实际 operator、数值计算和约束校验由代码执行。角色程序生成采样由准备流程生成：temperature=0、seed=42、max_tokens=8192；Lopt 超时600秒。与下面解题测评的采样不同。

## 5. 训练参数和 checkpoint

| 参数 | 当前值 |
| --- | --- |
| 分支 | Random / Static-M0 × 5e-6 / 2e-5，共4分支 |
| 起点 | 同一 M0 adapter；不同学习率独立训练，不互相接续 |
| 同模式成员与顺序 | 两个学习率共用同一固定选择和顺序 |
| GPU | 8×H200，141GB/卡；FSDP world_size=8 |
| 精度 | BF16 |
| Micro batch | 每卡1；全局有效batch=8，完整窗口每rank累积1 |
| 轮数 / seed | 1 epoch / 42 |
| 最大完整序列长度 | 131,072 tokens |
| 优化器 | AdamW，weight_decay=0，max_grad_norm=1.0；当前未配置 warmup/scheduler |
| Loss | supervised_token_mean，仅 assistant payload；不对技能 token 额外加权 |
| LoRA | rank16、alpha32、dropout0、bias=none、use_rslora=false |
| Target modules | q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj |
| Language path | model.language_model |
| 显存相关 | gradient_checkpointing=true，loss_chunk_tokens=256，cpu_activation_offload=false |
| 阶段节点 | 约 K/3、2K/3、K 曝光处；不能整除时以保存的实际边界为准 |
| 恢复检查点 | 每4 optimizer steps保存，保留最近2份；阶段 adapter 另存 |

各分支连续训练；每个阶段节点跑同一开发集。训练结束后，对 M0 和四分支各自最终节点进行三个 benchmark 测评，即通常共5份模型状态（1个M0＋4个最终权重）；若实际执行数量变化以manifest为准。benchmark 不反馈给 Lopt、不用于调参或换数据。实验一没有接受/回退门槛。

恢复须使用相同 run_id、源码/环境/数据/模型身份及冻结配置，恢复 optimizer、步数、RNG、数据位置等；不能拿 adapter 权重重新启动就称为精确续跑。CPU mock 恢复已验收，真实 FSDP 跨进程中断恢复仍须服务器验证。

## 6. 开发验证与三个 benchmark

| 集合 | 规模 | 环境/判分 |
| --- | --- | --- |
| Rebench 开发集 | 300题目标，每主技能50题 | Rebench官方fork，任务配套环境与测试；整体及分技能统计 |
| SWE-bench Verified | 500题 | 官方SWE-bench环境及判分 |
| SWE-bench Pro V2 | 642题 | 官方Harbor V2工具；捕获补丁后在全新沙箱权威重判分 |
| Terminal-Bench 4.1.0 | 66题，完整版本 | Harbor原生任务环境/验证器；含需要GPU的任务 |
| 三个benchmark合计 | 1,208题/模型状态 | 不含开发集 |

| Benchmark 来源 | 固定 revision |
| --- | --- |
| SWE-bench/SWE-bench_Verified | 78f471bf655a3137b2e8a75af1501690ec009ec3 |
| ScaleAI/SWE-bench_Pro | 2d52cb3df914a3fcf80c7f66738b3a88ae37fc50 |
| scaleapi/SWE-bench_Pro-os 官方任务/工具 | 66f92766bba642462d4bbe5479e83f91f9211862 |
| harbor-framework/terminal-bench v4.1.0 | 4070d6cef56a7d20cb5dc98d5e10ada88a6b9440 |

Terminal标识 `terminal-bench/terminal-bench@4.1.0`；registry entry5不代表版本5。上述源码固定不代替实际任务镜像digest固定。

统一解题预算（含开发集，具体以注册的 TaskRecord 为准）：

| 参数 | 当前值 |
| --- | --- |
| 最大agent轮数 | 250 |
| 单次最大输出 | 32,768 tokens |
| 每题累计最大输出 | 262,144 tokens |
| 上下文窗口 | 262,144 tokens；含输入和本次输出，不是另加32K |
| Thinking | 开启，preserve_thinking=true，reasoning_effort=xhigh |
| 采样 | temperature1、top_p .95、top_k20、min_p0、presence_penalty0、repetition_penalty1 |
| 单工具超时 / 模型请求超时 | 1,200秒 / 7,200秒；受任务总时限约束 |
| 解题重复 | 当前suite登记一个seed=42；不是官方Terminal示例的5次采样 |

时间预算不同：Verified/开发集使用7,200秒工程预算；Pro遵循官方50分钟协议，SDK agent_wall_seconds=2940以预留收尾；Terminal遵循每题task.toml的原生时限。Pro/Terminal bindings中的172,800秒是外层保护值，不能理解成agent获准跑48小时。Pro需要关闭外网工具、隔离隐藏测试，并执行fresh-sandbox regrade。

这些统一token/轮数属于本项目实验协议，不声称三个benchmark官方共同采用。完整重判分和模型失败/基础设施失败须分别记录，不按最终结果选择性重试。

## 7. 推理并发与算力安排

每卡独立vLLM BF16服务；配置每卡4任务槽、max_num_seqs=4，8卡目标32任务并发。gpu_memory_utilization=.90，max_num_batched_tokens=8192，prefix caching和chunked prefill开启。开发集和benchmark均按可用服务槽调度，已移除开发集固定16worker瓶颈。

训练阶段FSDP占用8卡；测评阶段切换为独立推理服务。Terminal原生GPU任务按既有调度预留卡，模型侧并发随剩余服务缩减，不能重复分配同一张GPU。

**32槽尚未真实GPU验收。**独立长前缀262K容量检查必须通过；服务器可比较每卡4/6/8槽，以任务/小时、tokens/s、KV抢占、OOM和CPU/容器资源选稳定吞吐点，再冻结slots和max_num_seqs。不得降低科研预算来让容量验收通过，也不能承诺已充分利用8卡。

## 8. 部署、资产复用与结果交付

网络工作全部在工作站：下载、pip/镜像构建、Teacher调用、选定题目环境准备、镜像/依赖缓存和离线核验。集群仅加载已准备资产执行计算；有缺依赖立即阻塞，不临时联网补齐。基础存储根模板为 `/storage/self-curate-rebench`，两端必须正确共享/映射。

复用资产：固定数据/模型、Teacher标注、开发集及环境、benchmark环境、初始adapter、M0开发集测评、全候选M0 NLL/entropy及首次策略。复用必须匹配完整身份、prompt/schema、协议与checksum；协议或模型变化时不能沿用旧测评结果。

每次实验使用独立workflow/config/handoff、唯一run_id和输出目录；attempt日志使用时间+hash标识。记录原始模型请求、工具观察、token usage（含缓存/推理信息但避免重复计数）、wall time、调用/训练/判分阶段、恢复状态和成本，GPU阶段记录显存/利用率/吞吐。只提交代码和正式配置，不提交数据、权重、密钥、缓存、运行状态。

交付目录通常为 `runs/deliverables/<run_id>/`：先看 summary.json、results.csv、evidence/checkpoints.csv，再看 results.zip、BUNDLE_MANIFEST.json、SHA256SUMS.txt。结果包包含结果及对应证据/原生日志；基座、原始训练数据、adapter权重和optimizer/RNG恢复状态留服务器，不打入普通结果包。复用资产的原始证据按打包实现校验后纳入。

## 9. 当前仍需现场落地的配置

1. 工作站实际路径/挂载、基础镜像digest、完整依赖锁、Claude二进制checksum、Harbor/Rebench/Verified grader commit与构建回执。
2. 实际成功轨迹/任务覆盖统计、可用候选数、六技能300题覆盖、tokenizer/model绑定、开发集发布引用。
3. 所有选定题目镜像digest、环境安装缓存、Pro协议回执、网络隔离及离线可执行验收。
4. 八张GPU UUID、container gateway、服务端点、CPU/RAM/磁盘容量，以及FSDP恢复/长上下文并发容量回执。

自动生成项不手造：training_count、正式Teacher max_calls、模型fingerprint、objective、角色sampling、serving pool、source registry和development_asset。正式必须通过Rebench流程生成server.preflight.json，再完成容量验收形成server.resolved.json；**server.template.json仍含历史SWE-smith默认，不能直接启动。**

## 10. 后续实验计划（简版）

| 阶段 | 增加的机制 | 主要比较 |
| --- | --- | --- |
| EXP001 / V0 | Random vs 固定M0策略，三段训练 | 建立基线，验证数据/训练/测评链条 |
| V1 动态评分 | 模型更新后刷新指标，程序和参数固定 | 当前模型信号是否带来增益 |
| V2 动态参数 | Lopt根据开发反馈改阈值/权重/配比 | 自主调参是否有效 |
| V3 动态程序 | 开放算子组合、顺序和分支修改 | 程序结构搜索是否有效 |
| V4 控制学习 | 加入合格筛选/策略修订轨迹SFT | 更新后的target是否更会选数据 |
| 成本与稳健性 | 精确/方向性刷新、重要对照多seed、外部DS基线 | 质量—成本、随机性与优势来源 |
| 扩展 | 冻结方法后开展跨数据/General验证 | 结论是否可迁移 |

NICE是否作为V2/V3内层估值器依独立验证结果决定；实验一不依赖它。后续内层停止、外层接受/回退、控制语料准入与多轮预算须在对应实验启动前另行冻结，不把这些待定规则混入当前实验一。旧递进计划中的SWE-smith和“两次尝试”保留作历史思路，当前执行以本页Rebench和EXP001三段协议为准；后续阶段次数需各自新配置明确。

## 11. 文件入口

- 唯一当前交接：`docs/handoffs/EXP001_REBENCH_WORKSTATION_V1.md`。
- 总体计划：`configs/experiments/exp001/rebench.workstation.json`。
- 机器bindings：`configs/experiments/exp001/workstation.bindings.template.json`、`environments.bindings.template.json`。
- 技能/Teacher：`configs/preparation/coding-skills.v1.json`、`teacher-v1/`、`teacher-pilot.template.json`。
- 测评协议：`configs/experiments/exp001/{swe,pro,terminal}-task-protocol.json`。
- 来源与并发细节：`docs/EXP001_FROZEN_BENCHMARKS_ZH.md`。
- 复核开关：`docs/TEACHER_REVIEW_POLICY_ZH.md`。

当前通过的是CPU mock、恢复、接口及打包检查；不等于真实27B训练、32并发或三个官方容器判分已通过。此次仅整理文档，没有新增付费Teacher调用。


2026-11-15 测评收口：1/3、2/3、全部训练节点均测开发集，只有各分支 stage3 测最终 benchmark；不按开发集或 benchmark 选择更早 checkpoint，两个学习率分支均保留最终测评。M0 初测资产可复用；不同阶段的中间权重仍保存，默认不跑 benchmark。旧13节点计划不再采用。


## 2026-11-16 当前标注变更（覆盖前文旧Teacher流程）

Coding改为[按任务标注并继承](CODING_TASK_LABELS_ONLY_ZH.md)：六技能不变，成功任务先标一次，开发候选取success_uncovered；不再做轨迹证据/占比或合批Teacher。新profile为coding-task-labels.v1.json，Flash4096输出/effort low/复核off，先小规模验收。证据指标缺失不填0。General先按[专门IF数据方案](GENERAL_IF_PLAN_ZH.md)记录，暂不配置或执行。


补充：按用户后续要求，证据字段从正式M1v4数据发布中删除，EvidenceSelect与evidence_batch删除，算子版本m2-v4；不再以“保留空比例字段”作为当前方案。NLL/entropy及梯度整assistant接口不变。内部旧类型仅用于历史兼容。
