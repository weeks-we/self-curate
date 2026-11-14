> 2026-11-14 更新：本页旧 mini-SWE 安装/调用描述已被 [Claude SDK harness 交接](CLAUDE_SDK_HARNESS_V1.md) 取代；其余研究定义不因此修改。

﻿# EXP001_REBENCH_WORKSTATION_V1

给服务器 agent：只在独立 self-curate checkout 操作，先读根 AGENTS、`docs/EXP001_REBENCH_PLAN_ZH.md`。本交接用于 Rebench 新实验；不要运行旧 bootstrap 的 `stage all`，不要继承相邻 NICE 的样本数、seed 或学习率。不 commit/push，不修改旧运行快照。

## 0. 当前边界

已实现 Random / 静态策略、Rebench→M1、三角色初始反馈、环境准备和 Pro V2 重判接口。技能定义及两类 Teacher 规则仍由用户确定，`ontology.pending.json` 故意不可执行。真实网络、Teacher、Docker、Harbor、八卡训练都没有在开发者本机验收。先完成下面每道门禁，失败时报告具体阻塞，不把 mock 回执改成 real。

固定 K=floor(硬筛前全部 resolved=1 轨迹数/10)，Random 和静态策略均选 K；从共同硬筛/隔离池选择，池不足停止。每技能 50 个开发任务，组间可重叠，执行并集一次。只在训练集选择中使用任务开发报告，最终 benchmark 不进入 agent 输入。

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

null revision 仅允许联网时解析一次，此后使用 source-lock 中完整 SHA。对 tasks、trajectories、verified、pro 分别执行：

```bash
python -m self_curate.experiments.exp001.workstation --location workstation export-table --lock /storage/self-curate-rebench/source-lock.json --storage /storage/self-curate-rebench/sources --role tasks --output /storage/self-curate-rebench/exports/tasks.jsonl
```

其余三次替换 role 和输出文件名即可。命令按冻结 config/split 导出 Parquet，保存 `.source.json` 侧车，不覆盖已有导出。任务 V1 与轨迹 instance_id 连接要核验；不能拿 V2 同名任务替代。

在工作站从官方 registry 核验并下载 Terminal-Bench 4.1 的真实任务内容，保存 `terminal-source.json`，含 release=4.1、repo_id、40 位 revision、root 及来源 URL；官网指向 registry entry5，不代表 CLI 一定接受 @4.1。不要只凭文件名宣布已下载正确版本。

构建 `docker/Dockerfile.rebench`：传 BASE_IMAGE（不可变 digest）、REBENCH_COMMIT、PRO_COMMIT、VERIFIED_COMMIT、HARBOR_COMMIT（均完整 SHA）。main 环境、Rebench fork、Verified 分别安装，Pro 使用官方 v2 Harbor tooling。固定版本选择由工作站 agent 核验后写构建证据，不允许猜 SHA。镜像内保存 pip freeze；镜像构建不等于任务容器准备完成。CPU 模型准备需要完整 27B 基座的主存及额外余量，先核验工作站 RAM，不可在开发者本机自动尝试。

## 3. 先选开发题，再配题目环境

1. 先导出全部最终 benchmark 的任务元信息，用 environments bindings 执行 catalog。此阶段不需要加载全部 Rebench 执行环境。
2. 用户确定 ontology 后填写 ontology 文件的 version、skills、task_prompt、trajectory_prompt；不要替用户启用旧七技能。任务 prompt 依据题面/参考补丁/测试补丁标“要求什么能力”，轨迹 prompt 依据实际行为及消息引用标“展示什么能力”；二者共用 ID。
3. 用 preparation bindings 按顺序运行 registry、model、tasks。Teacher key 只放环境变量 DEEPSEEK_API_KEY，不写 JSON。

```bash
python -m self_curate.server.rebench_environments --location workstation --bindings /storage/self-curate-rebench/environments.bindings.json --stage catalog
python -m self_curate.preparation.rebench_workstation --location workstation --bindings /storage/self-curate-rebench/workstation.bindings.json --stage registry
python -m self_curate.preparation.rebench_workstation --location workstation --bindings /storage/self-curate-rebench/workstation.bindings.json --stage model
python -m self_curate.preparation.rebench_workstation --location workstation --bindings /storage/self-curate-rebench/workstation.bindings.json --stage tasks
```

先在独立 pilot 输出根做 64 条标注检查，再执行正式候选；不改已消费的 prompt 或样本前缀。正式候选数默认 min(可用数,max(1000,200×技能数))；每技能抽 50 后冻结 groups/assignments/union。短缺停止；扩大候选需新准备身份，已有内容相同的 Teacher 缓存可复用。候选标注及参考答案只能在准备侧使用，不向 target 的任务推理泄露参考解。

Teacher 主模型 deepseek-flash，复核 deepseek-v4-pro，输出上限 16384、等待 900 秒。调用量由真实规模导出，非金额预算；5% 固定随机复核+争议复核，共受10%复核调用上限约束。输入超服务容量不能截断，记录阻塞；试标时确认模型确实接收完整长轨迹。未知在途调用要查服务记录后协调，不盲目重复。

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

Pro 使用官方 `locked_mini_swe:LockedMiniSwe` 产补丁，再用 `patch_replay:PatchReplayAgent` 在干净沙箱重判，最终 reward 仅来自后者。所有正式 Pro 任务逐题 oracle=1、nop=0、断网依赖检查，加上网络只允许模型服务、greedy/seed 的真实调用证据；使用 shell-free 检查 manifest 的 argv/cwd/timeout_seconds/evidence。调用：

```bash
python -m self_curate.server.offline --location workstation checks --manifest <已审核的checks.json> --output <checks目录>
python -m self_curate.server.offline --location workstation pro-receipt --manifest <同一checks.json> --checks-root <checks目录> --tooling /opt/graders/pro/v2/tooling --tasks /opt/graders/pro/v2/tasks --prepared /storage/self-curate-rebench/environments/pro-prepared-environments.json --output <pro-protocol.json>
```

检查 manifest 必须给 model_endpoints、network_overlay ArtifactRef；逐项 evidence 使用文件名→JSON字段路径→期望值，network_probe 需真实 allow/deny，cli_probe 需实际 temperature=0/seed，不能自写成功 JSON 替代运行证据。main 模块 API 在 `server/offline.py`，部署相关命令由 agent 按实际 Docker/Harbor 版本核验。若平台不支持 Docker socket、嵌套容器、模型服务回连或原生任务 GPU，先报告；不改测试协议掩盖问题。

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
