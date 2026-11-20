# 师兄B：普通任务环境与短训验收交接

本次只验证服务器能否完成“下载准备→数学抽样→三领域LoRA短训→checkpoint恢复/加载→全部benchmark判分”。不启动正式EXP001的Full/Random矩阵，不运行NICE或三个筛选agent，不付费调用teacher。短训分数只作环境诊断，不能作为正式实验结果。

请让执行agent先读本文件、AGENTS.md、docs/BENCHMARK_ENV_AUDIT_ZH.md。正式实验另见docs/HANDOFF.md。本交接针对self-curate普通任务项目，与邻近NICE项目分开。

## 1. 工作站准备，集群离线运行

统一使用configs/ordinary/sources.json的固定revision，不临时替换为最新版；记录源码commit及未提交补丁hash。创建独立环境、共享资产根和本次UTC时间-ID输出目录，数据、模型、运行结果均不进Git。模型和数据准备好后后续复用。

| 资产 | 准备内容 |
|---|---|
| 三个模型 | Qwen/Qwen3.5-9B、Qwen/Qwen3.5-4B、meta-llama/Llama-3.1-8B-Instruct；三个都完整下载并核对config/tokenizer/权重清单。Llama需已有HF访问权限；失败明确记录，不能换模型冒充完成。本次仅9B训练 |
| 代码数据 | BAAI/TACO train；test如下载只作保留测试来源，不入短训或开发题 |
| 数学数据 | project-numina/NuminaMath-1.5原集，保留problem/solution/answer/source/problem_type等字段 |
| IF数据 | allenai/tulu-3-sft-personas-instruction-following，实际repo/config以来源锁为准 |
| 代码benchmark（4） | LiveCodeBench release_v6、HumanEval+、MBPP+、BigCodeBench instruct/full |
| 数学benchmark（5） | MATH500、AIME2025、AIME2026、OlympiadBench英文纯文本开放题、GSM-Plus |
| IF benchmark（3） | IFEval、IFBench、Multi-IF英文三轮 |

下载模型/原始数据/12个benchmark以及官方checker源码、增强测试、依赖资源。EvalPlus增强测试、NLTK/其他checker资源需提前缓存。冻结所有导出题单、grader和镜像hash，不能到集群联网补装。

已有入口（从项目根执行；以下是准备步骤，不会启动正式矩阵）：

```bash
python -m self_curate.experiments.ordinary lock --output /shared/self-curate/assets/source-lock.json
python -m self_curate.experiments.ordinary download --lock /shared/self-curate/assets/source-lock.json --storage /shared/self-curate/assets --include-models
```

路径是示例，先创建父目录并替换为真实共享路径。download会生成source-bindings.json；后续export、benchmarks、evalplus-cache、seal、server-bindings具体参数通过`python -m self_curate.experiments.ordinary --help`及各子命令`--help`核对。

环境构建使用docker/Dockerfile的prepare、gpu、grader三个target；BigCodeBench另用docker/Dockerfile.bigcodebench，不把其旧numpy/numba依赖混进通用grader。联网工作站构建、pip check和freeze后，导出镜像供集群导入；记录最终image digest。构建失败要修复并记录，当前Dockerfile尚未经过本地真实构建验收。

先确认公司平台如何启动CPU判分容器。当前ServerBackend调用Docker；只有容器内安装docker命令并不等于具备daemon/平台执行权限。如平台只能提交CPU容器作业，agent需补部署适配，不得跳过沙箱或把官方判分替换成假分数。

## 2. 数学50k抽样与三领域数据准备

从固定原集导出数学raw JSONL及来源sidecar，然后：

```bash
python -m self_curate.experiments.ordinary sample-math --raw /shared/self-curate/raw/numina.jsonl --count 50000 --seed 42 --output /shared/self-curate/assets/math50k
```

先按source比例分配50,000名额，再按每个source内部problem_type比例分配；整数余数修正保证恰好50,000。保存配额、抽中ID、字段及hash，核对两层分布；不要改成直接全池随机50k。之后轻量准入可能减少合格数，两个数分开报告。

使用prepare生成三个领域的标准池，真实9B tokenizer计算长度；完整输入8192上限，超长整条排除，不截断。保留原生技能标签及answer_tokens/完整length。证明题可SFT，禁止将证明题当自动判分开发题；本轮不要求补齐全池teacher标签。

从每个合格训练池各用seed42固定抽128条短训样本，保存ID和内容hash；三领域各自训练，不能混池。按ID、来源及题面与最终benchmark隔离。样本不足或没有有效答案时明确报错，不复制凑数。

## 3. Qwen3.5-9B三次独立LoRA短训

本次短训参数作为独立环境验收配置保存；不能把正式configs/ordinary/exp001.json改成已批准，也不能让现有plan默认启动18分支。

| 项目 | 本次验收设置 |
|---|---|
| 分支 | coding / math / if，各128条；每次从共同初始基座和初始adapter开始 |
| 模型 | 固定revision的Qwen3.5-9B，AutoModelForImageTextToText；thinking关闭 |
| 训练 | HF Trainer + Accelerate FSDP2 + PEFT，BF16；底座冻结 |
| LoRA | r32、alpha64、dropout0；目标投影沿用现有training配置，包含Qwen3.5线性注意力、排除视觉模块 |
| LR/epoch | LR2e-5，3 epoch，warmup3%、weight_decay0；只为环境验收，不定稿正式科研超参 |
| Batch | 8×H200（141G），每卡micro4×累积1，global32；每epoch约4次optimizer更新，共约12次/领域 |
| 长度 | 完整输入≤8192，答案位置监督，不监督题面/模板/padding |
| 保存 | 每epoch原生checkpoint，共3份/领域、9份；保留optimizer/scheduler/RNG，不只存LoRA |

先用真实授权GPU UUID进行容量预检；无GPU时不执行，不允许静默CPU/mock替代。使用已有Qwen3.5优化kernel。若micro4不可行，保留报错和显存证据，再将本次验收改成micro2×累积2保持global32，记录新配置；不能暗改正式实验参数。

至少一个分支在epoch1保存后、epoch2期间模拟中断，再从同一run_id恢复，确认global_step/optimizer/scheduler/RNG/累计token与曝光继续衔接。避免主动中断checkpoint写入过程；另外验证已完成评测恢复时不重复生成。所有恢复都保留原始日志。

## 4. 全部benchmark测评

先对12个benchmark各取少量题做真实生成+官方checker烟测：代码正解能过、明显错误/超时能拒绝；数学/IF验证解析和判分，不能只验证数据能读。

然后三个领域只加载最后一份epoch3 checkpoint，跑**对应领域完整固定题单**：代码4、数学5、IF3，共12组测评。epoch1/2只保存和统计成本，不测benchmark。不是让代码checkpoint跑数学/IF，不是只导入12个数据集，本次只测最终一份。Initial全量复测和Full/Random矩阵本次不要求。

普通任务直接生成答案，不需要Claude Code/SWE仓库环境。测评建议8个单卡vLLM副本、TP1、每副本并发32起测、GPU memory0.9；temperature0、top_p1、thinking关闭、context16384；代码/IF输出4096、数学8192。先确认adapter加载及多轮IF交互正确，再测吞吐。不因判分慢放开不安全执行权限。

代码必须无网络、无凭据、无宿主home/GPU挂载的CPU沙箱，限制进程/内存/时间。BigCodeBench绑定独立grader镜像digest。数学/IF用各自官方checker；Multi-IF保留三轮上下文，不能扁平化成三道独立题。

本次耗时主要可能在完整benchmark生成与CPU判分，不是128条短训。逐阶段报进度和ETA，不预先保证固定小时数。

## 5. 当前代码边界与执行agent职责

已有：ordinary CLI资源准备/抽样、training/ordinary.py原生训练、backend.py生成/判分、evaluation.py断点缓存、reporting.py及CPU smoke。

**目前没有本次“3×128短训+12测评”的一键入口。** 请执行agent在独立验收目录补一个薄控制脚本，调用上述模块并记录状态，禁止复制训练引擎或修改通用接口；本交接不是宣称该runner已实现。正式workflow.validate_config固定EXP001矩阵，不能硬套本次小矩阵。

ServerBackend要求真实server_smoke_receipt及asset seal：先独立完成真实模型加载、反向、checkpoint恢复和grader验证，再生成可核查凭据并绑定，不能只手填server_smoke_passed=true绕过。平台适配修复应交付补丁和说明，不能把本次验收配置混进正式EXP001入口。

## 6. 返回给我的验收包

输出统一放`runs/b-env-<UTC日期-时间-随机ID>/`（示例目录），逐领域和checkpoint隔离。打包以下内容，不装入模型/原始数据/大权重：

- B_ACCEPTANCE.md：每项通过/失败/未执行、具体命令、机器/源码/环境信息与待修复问题。
- 固定配置、来源锁、镜像digest、pip freeze、GPU UUID与显存/吞吐实测。
- 数学50k两层抽样配额/分布、三份128样本manifest与排除统计。
- 9份checkpoint的epoch/step/路径/hash、保存/恢复证据；实际输入/监督token、唯一样本/曝光次数、训练与GPU时长。
- 12组按约定协议的成绩、逐题预测/判分与生成token、生成/判分时间，summary.csv、training_costs.csv和可读报告。
- 薄控制脚本/必要修复补丁、日志、完成状态和ZIP校验清单。大权重留服务器，报告可访问路径与checksum。

只有三个模型资产完整、三领域短训/恢复成功、9个adapter可加载、12个benchmark真实判分及12组结果齐全，才能说本次验收完成；有未执行项就如实列出。不要据这些短训分数讨论数据筛选效果，也不要自动继续正式实验。

2026-11-21更新：只测最终checkpoint；官方判分修正及八卡优化见 [复核记录](BENCHMARK_OFFICIAL_CHECK_20261121_ZH.md)。
