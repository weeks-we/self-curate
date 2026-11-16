> 当前测评协议已按用户要求重新对齐最新NICE SDK：250轮、262K、thinking/xhigh及采样。见 [唯一当前测评表](../EXP001_SDK_EVALUATION_PROTOCOL_ZH.md)；下文旧75轮/128K/贪心/10000轮说明不再作为执行参数。

> 2026-11-16：当前默认改为[轻量任务标注](../CODING_TASK_LABELS_ONLY_ZH.md)，该页覆盖下文旧的逐轨迹证据/合批标注及success_covered开发集描述。使用新profile、新准备根；先人工验收task-only试标，不自动全量。

> 2026-11-15：先读 [来源/官方测评/算力/规模补充](../EXP001_EVALUATION_AND_SCALE_20261115_ZH.md)。Rebench任务/轨迹已固定完整SHA，Teacher复核改Flash；Verified沿NICE数值，Pro与Terminal独立预算。官方文档优先，工程保护须单独披露。

> Teacher v1已按2026-11-14用户决定启用六技能。正式入口与复用方式见 [Teacher交接](TEACHER_V1_WORKSTATION.md)；本文旧的待定技能、重叠任务组和复用标注prompt的描述已被取代。

> 2026-11-14 更新：本页旧 mini-SWE 安装/调用描述已被 [Claude SDK harness 交接](CLAUDE_SDK_HARNESS_V1.md) 取代；其余研究定义不因此修改。

﻿# EXP001_REBENCH_WORKSTATION_V1

给服务器 agent：只在独立 self-curate checkout 操作，先读根 AGENTS、`docs/EXP001_REBENCH_PLAN_ZH.md`。本交接用于 Rebench 新实验；不要运行旧 bootstrap 的 `stage all`，不要继承相邻 NICE 的样本数、seed 或学习率。不 commit/push，不修改旧运行快照。

## 0. 当前边界

已实现 Random / 静态策略、Rebench→M1、三角色初始反馈、环境准备和 Pro V2 重判接口。六技能和任务/轨迹/复核三类规则已批准，正式ontology为 configs/preparation/coding-skills.v1.json。真实Teacher公开样例3题+2轨迹已通过格式和原文证据校验；尚无代表性100+100、真实token比例、Docker、Harbor和八卡验收。先完成下面每道门禁，失败时报告具体阻塞，不把 mock 回执改成 real。

固定 K=floor(硬筛前全部 resolved=1 轨迹数/10)，Random 和静态策略均选 K；从共同硬筛/隔离池选择，池不足停止。按唯一主技能每类50题，共300道互异开发任务；保留次技能但不重复凑主组。只在训练集选择中使用任务开发报告，最终 benchmark 不进入 agent 输入。

## 1. 本地 CPU 可复核

在项目依赖环境中设置 `PYTHONPATH=src`，不安装目标权重、不启动 GPU 或 Docker：

```bash
python -m self_curate.experiments.exp001 --config configs/experiments/exp001/mock-rebench-budget.json --output-root runs/rebench-cpu --assets runs/rebench-cpu-assets --package-results
python -m self_curate.smoke --config configs/smoke/nested_cpu_mock.json --output runs/nested-cpu-new
```

EXP001 的 mock 原始成功数显式设 40、合格池 30，故两组都选 4，而非 3；四分支、三阶段、开发题结果和最终模拟 benchmark、ZIP 必须完整。记录自动生成的输出路径，恢复用原路径加 `--resume`，不能换输出根冒充恢复。嵌套 mock 是通用模块回归（3 外层×5 内层），不是本次 EXP001 科研算法。

CPU FSDP 检查入口：`python -m self_curate.server.fsdp_preflight --request <隔离的TrainingRequest.json> --output <新目录> --mode inspect`。training_id 以 preflight- 开头、有效 batch8；输出 passed=false/GPU=false 是正确行为。

## 2. 联网工作站：来源与 Docker

使用新持久根 `/storage/self-curate-rebench`（R）。不同机器挂载同一绝对路径，避免 ArtifactRef 指向旧地址。复制并填写两个 bindings 模板到 R，不提交机器路径/凭据。固定本次源码 commit 和完整环境锁。

```bash
python -m self_curate.experiments.exp001.workstation --location workstation resolve --plan configs/experiments/exp001/rebench.workstation.json --output /storage/self-curate-rebench/source-lock.json
python -m self_curate.experiments.exp001.workstation --location workstation download --lock /storage/self-curate-rebench/source-lock.json --storage /storage/self-curate-rebench/sources --output /storage/self-curate-rebench/download-receipt.json
```

Rebench任务/轨迹使用计划中已固定完整SHA；其他仍为null的来源仅允许联网解析一次，此后使用source-lock完整SHA。对 tasks、trajectories、verified、pro 分别执行：

```bash
python -m self_curate.experiments.exp001.workstation --location workstation export-table --lock /storage/self-curate-rebench/source-lock.json --storage /storage/self-curate-rebench/sources --role tasks --output /storage/self-curate-rebench/exports/tasks.jsonl
```

其余三次替换 role 和输出文件名即可。命令按冻结 config/split 导出 Parquet，保存 `.source.json` 侧车，不覆盖已有导出。任务 V1 与轨迹 instance_id 连接要核验；不能拿 V2 同名任务替代。

在工作站从官方 registry 核验并下载 Terminal-Bench 4.1 的真实任务内容，保存 `terminal-source.json`，含 release=4.1、repo_id、40 位 revision、root 及来源 URL；官网指向 registry entry5，不代表 CLI 一定接受 @4.1。不要只凭文件名宣布已下载正确版本。

构建 `docker/Dockerfile.rebench`：传 BASE_IMAGE（不可变 digest）、REBENCH_COMMIT、PRO_COMMIT、VERIFIED_COMMIT、HARBOR_COMMIT（均完整 SHA）。main 环境、Rebench fork、Verified 分别安装，Pro 使用官方 v2 Harbor tooling。固定版本选择由工作站 agent 核验后写构建证据，不允许猜 SHA。镜像内保存 pip freeze；镜像构建不等于任务容器准备完成。CPU 模型准备需要完整 27B 基座的主存及额外余量，先核验工作站 RAM，不可在开发者本机自动尝试。

## 3. 先选开发题，再配题目环境

1. 先导出全部最终 benchmark 的任务元信息，用 environments bindings 执行 catalog。此阶段不需要加载全部 Rebench 执行环境。
2. 直接使用已批准的 coding-skills.v1.json 及其引用的 task/trajectory/review prompt/schema；不要启用 ontology.pending.json 或旧七技能。任务 prompt 依据题面/参考补丁/测试补丁标“要求什么能力”，轨迹 prompt 依据实际行为及消息引用标“展示什么能力”；二者共用 ID。
3. 用 preparation bindings 按顺序运行 registry、model、tasks。Teacher key 只放环境变量 DEEPSEEK_API_KEY，不写 JSON。

```bash
python -m self_curate.server.rebench_environments --location workstation --bindings /storage/self-curate-rebench/environments.bindings.json --stage catalog
python -m self_curate.preparation.rebench_workstation --location workstation --bindings /storage/self-curate-rebench/workstation.bindings.json --stage registry
python -m self_curate.preparation.rebench_workstation --location workstation --bindings /storage/self-curate-rebench/workstation.bindings.json --stage model
python -m self_curate.preparation.rebench_workstation --location workstation --bindings /storage/self-curate-rebench/workstation.bindings.json --stage tasks
```

先依 Teacher交接在独立pilot目录逐步完成至多100题+100轨迹及人工抽检，再执行正式候选；不改已消费的 prompt 或样本前缀。正式候选数默认 min(可用数,max(1000,200×技能数))；每技能抽 50 后冻结 groups/assignments 和300个互异ID。短缺停止；扩大候选需新准备身份，已有内容相同的 Teacher 缓存可复用。候选标注及参考答案只能在准备侧使用，不向 target 的任务推理泄露参考解。

Teacher 主模型 deepseek-flash，复核 deepseek-flash，输出上限 16384、等待 900 秒。调用量由真实规模导出，非金额预算；5% 固定随机复核+争议复核，共受10%复核调用上限约束。输入超服务容量不能截断，记录阻塞；试标时确认模型确实接收完整长轨迹。未知在途调用要查服务记录后协调，不盲目重复。

4. 运行 pool：计算硬筛前分母、隔离开发题与终测、轻量硬筛、轨迹标注和 M1 对齐。它不真实重放全部成功轨迹。

```bash
python -m self_curate.preparation.rebench_workstation --location workstation --bindings /storage/self-curate-rebench/workstation.bindings.json --stage pool
```

## 4. 环境、断网验收、发布

按 environments bindings 顺序运行 images、install。只准备冻结开发题和三个正式 benchmark；Pro 既有 agent 安装镜像，也有独立 verifier-only 依赖快照，不把隐藏测试缓存注入 agent 镜像。

```bash
python -m self_curate.server.rebench_environments --location workstation --bindings /storage/self-curate-rebench/environments.bindings.json --stage images
python -m self_curate.server.rebench_environments --location workstation --bindings /storage/self-curate-rebench/environments.bindings.json --stage install
python -m self_curate.execution.runtime --help
```

用 runtime 命令为主环境与单独 Rebench grader 环境生成实际 ArtifactRef；bindings 的 runtime_lock/rebench_runtime_lock 填真实 URI+sha256，不写空壳通过标记。

Pro 使用当前 Claude SDK agent 产补丁，再用独立干净沙箱的 patch replay/verifier 重判；不再使用旧 locked_mini_swe 作为agent。最终 reward 仅来自官方重判。镜像映射、路由与新协议回执见 Claude SDK交接。所有正式 Pro 任务逐题 oracle=1、nop=0、断网依赖检查，加上网络只允许模型服务、冻结采样/seed/thinking 的真实调用证据；使用 shell-free 检查 manifest 的 argv/cwd/timeout_seconds/evidence。调用：

```bash
python -m self_curate.server.offline --location workstation checks --manifest <已审核的checks.json> --output <checks目录>
python -m self_curate.server.offline --location workstation pro-receipt --manifest <同一checks.json> --checks-root <checks目录> --tooling /opt/graders/pro/v2/tooling --tasks /opt/graders/pro/v2/tasks --prepared /storage/self-curate-rebench/environments/pro-prepared-environments.json --output <pro-protocol.json>
```

检查 manifest 必须给 model_endpoints、network_overlay ArtifactRef；逐项 evidence 使用文件名→JSON字段路径→期望值，network_probe 需真实 allow/deny，cli_probe 需实际 temperature=1/top_p=0.95/top_k=20/seed/thinking=xhigh，不能自写成功 JSON 替代运行证据。main 模块 API 在 `server/offline.py`，部署相关命令由 agent 按实际 Docker/Harbor 版本核验。若平台不支持 Docker socket、嵌套容器、模型服务回连或原生任务 GPU，先报告；不改测试协议掩盖问题。

然后填 Pro protocol ArtifactRef，运行 register；之后 preparation 执行 publish、config。config 还需八个获配 GPU UUID、真实 container_gateway、installed_environments、runtime_lock；自动生成服务池/relay/目标/编辑协议，沿用现有公开模块接口。缺真实容量回执时只生成 server.preflight.json，不能假装正式通过。

```bash
python -m self_curate.server.rebench_environments --location workstation --bindings /storage/self-curate-rebench/environments.bindings.json --stage register
python -m self_curate.preparation.rebench_workstation --location workstation --bindings /storage/self-curate-rebench/workstation.bindings.json --stage publish
python -m self_curate.preparation.rebench_workstation --location workstation --bindings /storage/self-curate-rebench/workstation.bindings.json --stage config
```

使用 offline export-images 导出 environment image-names.json 中的镜像；主运行镜像也必须加入清单。用 seal 冻结只读资产子树（不含 private、API key、可变运行状态；receipt 放树外）。工作站执行完整断网加载和官方判分验收。需要 target 推理/GPU 的检查留到集群，不能因此在集群补装环境。

## 5. 集群：导入、核验、计算

先 `offline --location cluster import-images ...` 与 `verify ...`，核对归档 hash、加载后 image ID 和全部文件。设置四个 OFFLINE/NO_INDEX 环境开关；外网不开，缺依赖立即阻塞。保持共享路径、源码、环境和来源身份。

八卡单分支 FSDP：micro1×8卡×累积1=global8；BF16、LoRA rank16/alpha32/dropout0，四支顺序训练。先跑隔离的 preflight- 请求：真实长上下文容量、框架强制中断/恢复与连续对照、adapter 导出/重载、任务容器 smoke，核验 GPU UUID 和实际峰值显存。CPU inspect 不构成这几项通过证据。检查 Token/时间、工具、GPU原生任务预算与配置一致，冻结后不静默降预算。

正式入口是生成的 server.resolved.json，沿用 `python -m self_curate.experiments.exp001 --config ... --output-root ... --assets ... --package-results`。EXP001 没有内层搜索/接受回退/筛选轨迹 SFT。它运行 M0开发测评→全集初始NLL/entropy→Leval→Lopt→Lexec→两种固定子集→四分支三阶段训练及开发验证→全部阶段最终benchmark。

## 6. 存储、恢复、交付

- R/sources、annotations、environments、preparation/prepared：冻结来源、原始标注和可复用资产。
- `--assets`：初测、初始NLL/entropy、反馈、策略等带身份缓存；原始费用和本次复用记录分开。
- `--output-root` 下 YYYYMMDD-HHMMSS-hash：实际实验配置、角色私有history、算子trace、训练与任务逐条结果、费用/时间/token日志。
- 同级 controller-logs：每次启动/恢复的时间编号日志；训练目录保留恢复 checkpoint/优化器/RNG及三个阶段adapter。
- `deliverables/<run_id>/results.zip`：可分享报告、CSV/PNG、配置、身份/校验和及必要日志；大模型权重与完整恢复分片留持久盘，不进入小结果包。

恢复必须使用原 output 和 `--resume`；不改冻结配置、数据、源码或快照。已完成结果只核验复用；无法核实的外部调用先对账。同类基础设施错误最多两次重试。代码更新用固定 commit 的独立 worktree，不在运行源码目录直接 pull。最后交付 run_id、results.zip、GPU/离线验收回执和未完成项，禁止用 mock 指标写正式结论。


## 7. 执行agent的阶段验收清单

本页是总流程；Teacher和Claude SDK交接补充本阶段细节，不能回退旧V3研究定义。
输入计划 rebench.workstation.json 已同步六技能和100条试标标记；
正式采用 bindings 的 coding-skills-v1 与 Teacher pilot模板。来源revision=null仍需联网解析成SHA。
不要修改已消费配置或快照；每阶段输出和阻塞记录写共享存储，不改本交接记进度。

|阶段|实际输入/动作|通过后保留的证据；失败则停止|
|---|---|---|
|0 固定源码|独立checkout、源码commit、PYTHONPATH=src；复制bindings到共享存储|源码身份、运行位置、工作站RAM/磁盘；不继承NICE配置|
|1 来源|resolve→download→export-table；Terminal官方registry核验|source-lock、download-receipt、JSONL及.source.json、Terminal真实版本和SHA|
|2 环境与目录|构建主镜像及SDK运行层；benchmark catalog；registry/model|镜像digest、pip freeze、完整基座/tokenizer/初始adapter身份、benchmark题目元信息|
|3 Teacher试标|依TEACHER_V1_WORKSTATION；少量后100+100，真实tokenizer|原始请求/回复、引用校验、费用/usage、各技能分布、人工语义抽检；格式通过不等于标签正确|
|4 开发题|tasks；六主技能各50，次技能保留|selected-development.json、300唯一ID、development_asset和checksum；缺组停止|
|5 候选池|pool；隔离、硬筛、轨迹标注、M1|硬筛前成功数N、K=N//10、全部排除原因、record/skill-features；池不足K停止|
|6 所选题目环境|images/install、SDK镜像映射、官方grader及离线检查|300开发题与三个benchmark的环境覆盖、网络allow/deny、官方oracle/nop和Pro新协议；无需准备全Rebench任务环境|
|7 发布|register→publish→config；填写UUID/网关/运行时锁|PreparedInputs、server.preflight.json或server.resolved.json；只有preflight不表示正式可跑|
|8 搬运|export-images/seal→集群import-images/verify|归档hash、image ID、共享绝对路径一致；离线缺依赖立即停止|
|9 集群验收|独立工程smoke、真实长上下文FSDP及SDK、强制中断恢复对照|容量/吞吐/恢复/重载/token路由和官方判分证据；禁止用CPU mock替代|
|10 正式执行|冻结resolved配置→EXP001；原output恢复加--resume|M0开发初测和全池指标、Leval/Lopt/Lexec、Random/static两学习率共四支三阶段、M0＋四个最终节点benchmark|
|11 交付|核验完成摘要、逐题结果、CRC及日志|run_id、results.zip路径和sha256、权重/恢复分片引用、未完成/失败清单|

pool可以在任务环境制备前执行，因为不重放训练轨迹；环境准备必须在register/publish前完成。
Teacher子交接中建议先配环境再pool也可，但不改变选题、隔离和训练预算。

正式运行（仅集群验收通过后；路径按生成结果填写）：
```bash
export PYTHONPATH=src
python -m self_curate.experiments.exp001 --config /storage/self-curate-rebench/preparation/server.resolved.json --output-root /storage/self-curate-rebench/runs --assets /storage/self-curate-rebench/assets --package-results
```
resolved文件实际位置以config阶段回执为准。恢复必须使用上次实际run目录：
```bash
python -m self_curate.experiments.exp001 --config <原resolved配置> --output <原run目录> --assets <原assets目录> --resume --package-results
```
交付时区分completed、blocked、failed；任何标注、测评或训练缺项都不得写全流程通过。
全量benchmark开销为5个模型节点乘三个冻结题单；工作站冻结实际题数，不能现场抽小或因失败删题。


## 2026-11-15追加执行要求

1. 两份Rebench SHA见计划，不再解析为main；真实下载与导出仍需内容hash及ID连接验收。
2. 使用新的environment bindings：swe_budget、pro_budget、terminal_budget分开。TaskBudget外层watchdog不是Harbor agent时限；SDK实际读取原生agent时限，Pro额外2940秒收尾保护。旧已注册任务不能换预算继续复用。
3. preparation bindings的benchmark_protocol_files按suite发布三个不同protocol；publish intent校验文件内容，修改必须新准备目录。Verified对齐NICE数值；SDK原生context管理不冒称旧最近5观察裁剪。
4. Flash复核独立prompt；旧异模型缓存不冒用。同模型复核仍需人工抽检。
5. 8卡训练保持global8，loss_chunk_tokens=256已版本化；推理目标4槽/卡、32任务、max_num_seqs4及显存0.90。Terminal原生GPU预留会减少可用推理卡和worker。
6. 运行server.capacity.check_inference时传service_settings=正式resolved中adapter_settings.services，并将实际新报告发布ArtifactRef，主capacity receipt填inference_capacity_report；必须覆盖全部声明槽位、262144上下文及32768输出，且各槽独立前缀。旧两路回执不授权四路运行。
7. 先做2/4槽吞吐与完整长度容量对照；若4不合适，显式使用新配置新预检，不在正式运行自动降档。观察CPU/磁盘/容器并发，不能只追GPU占用。
8. 默认10%不变；若真实N=32161则K3216，1072/2144/3216三个累计节点。以导出统计为准，不预填32161通过标记。

## 任务关联与批量标注补充（2026-11-15）

先读[任务—轨迹接口](../TASK_TRAJECTORY_LINKS_ZH.md)。保持先选定300题、排除其全部轨迹再处理剩余池；使用更新bindings和pilot的trajectory_grouping。先对同任务2条做组/单条一致性检查，再扩大试标，不能用未实测组模式直接全量收费。任务/轨迹都有独立证据，group失败保留成功成员；检查selected-task-relations、每条task-link、task-relations、grouping-summary与provider usage。m2-v3新增TaskLimitSelect，旧运行不原地恢复；不新增NICE任务分数或自动限制Random。


复核开关更新（2026-11-15）：新模板默认 `review_mode: off`；可选 flagged / sampled / all。关闭不跳过程序校验。请先读 [Teacher 复核策略](../TEACHER_REVIEW_POLICY_ZH.md)，旧的默认复核描述以该页为准。


来源冻结与并发更新：先读 [固定来源与并发](../EXP001_FROZEN_BENCHMARKS_ZH.md)。Pro HF、官方工具与 Terminal SHA 已填写，Terminal CLI 为 @4.1.0；旧“revision 待联网解析”描述被该页替代。开发集和 benchmark 均按可用服务槽调度，目标 32 并发，实际 GPU 容量回执仍需服务器生成。


2026-11-15：用户确认阶段开发测评＋最终benchmark。四分支各stage1/2/3仍测开发集；benchmark仅initial及各stage3，共5个模型状态、15份suite报告。stage1/2不跑benchmark，不引入学习率或checkpoint选模。旧run不能原地迁移协议，使用新run_id；精确一致的M0资产按现有身份规则复用。
