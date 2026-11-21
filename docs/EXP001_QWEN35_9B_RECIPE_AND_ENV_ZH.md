# Qwen3.5-9B：实验一与 NICE 的训练、并行和环境配置

> 最新长度与逐benchmark预算见[长度/预算核查](TRAINING_LENGTH_AND_BENCHMARK_BUDGETS_ZH.md)。原文统一4096/8192输出仅为旧设置；当前执行exp001.evaluation.benchmark_budgets。三个epoch测评及A/B范围遵循HANDOFF。


更新：2026-11-21。仅适用于 EXP001 和它附带的 NICE 验证，不作为 EXP002–004 的默认超参。用户授权本轮选择并直接修改配方；科研参数已选定。`approved=false`保留为服务器环境验收门，不能理解为还在等待用户批准学习率。本地 CPU 检查不证明真实 GPU 容量或容器构建成功。

## 1. 选择依据与限度

| 一手来源 | 实际设置与适用范围 | 本项目如何采用 |
|---|---|---|
| [LLaMA-Factory Qwen LoRA 示例](https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/examples/train_lora/qwen3_lora_sft.yaml) | Qwen3-4B-Instruct，非 thinking，LoRA r8/all，LR1e-4，3epoch，cosine；是示例而非三领域最优结论 | 支持把1e-4作为 LoRA 上档，不能照搬其2048长度 |
| [TACO 作者仓库](https://github.com/FlagOpen/TACO) | 训练示例5e-5，旧 CodeLlama/StarCoder 系列；任务同属算法题代码监督 | 选5e-5下档保守对照；不冒称作者测过 Qwen3.5-9B |
| [Qwen3.5 微调指南](https://unsloth.ai/docs/models/qwen3.5/fine-tune) | 支持9B的BF16 LoRA、文本注意力与MLP投影，提醒线性注意力内核；其显存数字依赖自己的实现 | 8×141GB不需要为省显存引入4bit量化误差；实际仍用本项目 HF/PEFT/FSDP2，不迁移其优化后显存估计 |
| [LKValues，AACL 2026](https://aclanthology.org/2026.aacl-long.118.pdf) | 包含Qwen3.5-9B-Base的LoRA，LR1e-4、1epoch、1024长度、cosine/warmup3%；价值对齐任务 | 给同家族量级参考；Base和我们的已后训练模型、任务不同，不能当作直接验证 |
| [LESS，ICML 2024](https://arxiv.org/html/2402.04333v2) | Llama/Mistral7B，LR2e-5、batch128、4epoch，adapter设置不同 | 表明DS中epoch不必为1，也说明全量/LoRA/adapter规模不同不能只抄LR |

数学和IF也核对了原作者配方：[Numina作者训练报告](https://huggingface.co/blog/winning-aimo-progress-prize)在DeepSeekMath7B全参训练中使用LR2e-5、global32、3/4epoch、cosine；[Tulu3官方8B复现](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)为Llama3.1-8B全参混合数据，LR5e-6、global128、2epoch、4096长度、warmup3%。它们都不是本次“后训练Qwen9B＋单领域LoRA”，低学习率不能未经区分直接搬过来。数学沿用global32并保存原生推导监督，8192用于减少长解排除；IF以低档5e-5与高档1e-4对照检查约束保持，不额外用IFBench挑学习率。最终同时报告两档，不能只留下benchmark较好的一档。

没有找到同时精确覆盖“Qwen3.5-9B后训练版＋TACO/Numina1.5分层50k/Tulu IF＋本项目FSDP2”的公开最优配方。因此选相邻经验中 **5e-5、1e-4** 两档，而不是声称已经调优。三领域共用这两档和global32，减少用领域专属调参解释结果的空间。关闭推理模式不意味着删除数学原生参考解中的推导。

主实验r32/alpha64：三个领域的知识、算法和约束适配需要一定容量，缩放比2，避免一开始扩到r128。NICE r16/alpha32：每组只有1,000条、125个更新步，保持较小适配器，并确保评分和目标SFT使用同一LoRA坐标。两个实验的rank不同，不能直接把其成绩差异解释为NICE选样收益；NICE内部High/Low四分支配方完全相同。warmup3%在NICE约4步，主实验按自身步数计算；cosine随后衰减。

## 2. 冻结配方

| 参数 | 实验一 Full / Random | NICE High / Low |
|---|---|---|
| 模型 | Qwen3.5-9B，共同初始模型；thinking关闭 | 相同基础模型，独立初始adapter |
| LR | **5e-5 / 1e-4**，独立训练 | 相同两档，High/Low各两分支 |
| epoch | **3** | **1** |
| LoRA | **r32 / alpha64 / dropout0** | **r16 / alpha32 / dropout0** |
| 每条分支GPU数 | **4**，原生Trainer/Accelerate FSDP2 | **4**，相同训练框架 |
| 每卡micro batch | **4** | **2** |
| 梯度累积 | **2** | **1** |
| global batch | **4×4×2 = 32** | **4×2×1 = 8** |
| BF16 | 底座冻结，训练LoRA；不使用QLoRA | 同左；评分不做optimizer.step |
| 优化器 | fused AdamW；beta=.9/.999，eps1e-8，clip1，WD0 | 同左；eps/clip与fresh-Adam评分一致 |
| 调度 | cosine，warmup ratio .03 | 同左 |
| 最大长度 | **完整prompt＋answer 8192 tokens**，超长排除，不截断答案 | 相同；rollout单独检查context |
| 损失 | assistant有效token监督；HF默认有效token聚合 | 每条样本答案token均值，再对样本平均，与候选评分一致 |
| 加速 | 长度分组、梯度checkpointing、padding倍数8、workers4/prefetch2、FSDP2 forward后不重分片 | 同左 |
| packing | false | false |
| 保存 | 每epoch完整checkpoint，共3份 | optimizer step42/84/125，共3份；曝光336/672/1000 |
| benchmark节点 | **测epoch1/2/3，epoch3为主报告**；前两份保留成本与恢复信息 | **三节点都测**，用户指定的独立例外 |

LoRA仅作用文本层：q/k/v/o_proj、gate/up/down_proj，以及Qwen3.5线性注意力的in_proj_qkv/z/b/a、out_proj；视觉层、embedding和lm_head不训练。实际模块匹配和可训练参数清单随receipt保存。FSDP2需真实版本内核验收，不以“命令能启动”替代真实反向和恢复。

主实验种子42；Initial、Full、Random5/10/20/30%，无Random50%。各随机比例嵌套，两个LR使用同一组成员与顺序，分支均从共同起点开始。三个领域共33分支（其中3个Initial）、30次SFT、90个epoch checkpoint、372组分阶段benchmark。NICE另4次SFT、12 checkpoint、60组测评（48 benchmark＋12原20题诊断）。总计34次SFT、102个训练checkpoint、420 benchmark组＋12诊断组。不能根据测试集选较好的epoch。

## 3. 8×H200 的实际调度

新实现 `parallel_run.py` 将一个独立8卡单节点分为两路，不让两个分支共享同一GPU。每路内部是4rank FSDP2，不是8rank训练；不同路的梯度和优化器相互独立。

| 路 | GPU清单 | vLLM端口 | 行为 |
|---|---|---|---|
| lane0 | 已授权UUID列表前4个 | 8000–8003 | 自己的分支训练→4副本生成→后续分支 |
| lane1 | 后4个 | 8100–8103 | 同时运行另一分支；可以与lane0的训练或生成重叠 |
| CPU判分 | 最多2个独立容器 | 无GPU | 每个8CPU/48GB；生成完即可入队，不阻塞GPU继续下一项 |

同一路训练前停止自己拥有的vLLM；生成前训练进程已经退出。不在一张卡上叠加训练和显存利用率90%的vLLM。任务按固定index%2分配，恢复时保留GPU归属。两路尾部可能有空闲，并非承诺每秒八卡100%利用率；长Full和短Random任务交错及异步CPU判分用于减少空转。NICE梯度/rollout阶段先独占8卡，样本独立分片，不与训练交叉，也不把候选梯度跨卡平均。

vLLM **TP1，4副本/路，64个并发序列/卡**，两路最多512条在途序列；这不是训练batch或实测吞吐。context16384，max_num_batched_tokens8192，chunked prefill，显存比例.90，最大LoRA rank64。输出上限与有效上下文按exp001.evaluation.benchmark_budgets逐项执行，见长度/预算核查；物理服务窗口仍16384；greedy，top_p1，thinking=false。官方判分器与项目生成预算是两件事，不能把这些预算称为benchmark统一官方标准。

同一路同一底座的NICE多个checkpoint使用vLLM官方load/unload LoRA接口，避免反复加载底座。[官方接口](https://docs.vllm.ai/en/latest/features/lora/)；服务仅绑定127.0.0.1，动态加载不能暴露外网。版本对Qwen3.5混合注意力LoRA的真实支持必须B验证；不支持时明确报错，不悄悄用基础模型代测。CPU判分容器都有独立名称，取消只清理本次拥有的进程组与容器。

## 4. NICE 的额外参数

TACO train抽20题，每题4次rollout；题目及显式同源、同ID、同题面参考解从3000候选排除。正式rollout是本地HF精确token ID的随机生成：temperature1/top_p1/top_k0，最多4096输出，context16384，不用API模型代替。20题原生参考解须先通过锁定的TACO官方checker；不通过停止，不按模型表现换题。

候选CE按assistant token平均；奖励加权目标方向按生成token和计算、每题除4；20题分母固定，失败零方向保留。fresh-Adam后CountSketch8192维，seed42，保留原始余弦等诊断；不optimizer.step。分数排序最低和最高各1000条，中间1000不训练。原20题复测是诊断，非独立benchmark，不据此挑节点。B仅2题×1个64token rollout、8候选反向probe，不训练、不跑NICE正式测评。

## 5. 模型、数据与 benchmark 来源

以下固定revision均通过公开HF元数据查询核对；这只验证来源存在，不代表权重/测试资源已经在本地下载。Llama需要访问授权。规模是原始来源口径，最终准入池/测试题单另记manifest。

| key | 官方/HF链接 | 固定revision | split/config |
|---|---|---|---|
| qwen9b | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |   |
| qwen4b | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |   |
| llama8b | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `0e9e39f249a16976918f6564b8830bc894c89659` |   |
| taco_train | [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO) | `d593ed0a2becbbc952230bb89be09189bf1056dc` | train default |
| numina | [project-numina/NuminaMath-1.5](https://huggingface.co/datasets/project-numina/NuminaMath-1.5) | `1b05109f9e5c1ad06c0663519502416c30b300f8` | train default |
| personas_if | [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | `fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e` | train default |
| livecodebench | [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` | test release_v6 |
| humaneval_plus | [evalplus/humanevalplus](https://huggingface.co/datasets/evalplus/humanevalplus) | `d32357cf319e50e9c8d8dab5ea876c72b0fd321b` | test  |
| mbpp_plus | [evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus) | `b2d74c91837c3f2a20c1299ae98133cbe7cfa077` | test  |
| bigcodebench | [bigcode/bigcodebench](https://huggingface.co/datasets/bigcode/bigcodebench) | `b74c0d0bf70d2c0bc459be537895cca163007f1a` | v0.1.4  |
| math500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | test  |
| aime25 | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) | `563bb8404243c5f09de6ec262f2db674fe5bce9b` | test  |
| aime26 | [math-ai/aime26](https://huggingface.co/datasets/math-ai/aime26) | `79037aebdb6580008fb960d17cb21fd3099083e3` | test  |
| olympiad | [Hothan/OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | `91184b52131e7fc9455fef848035173aea8cc01a` | train OE_TO_maths_en_COMP |
| gsm_plus | [qintongli/GSM-Plus](https://huggingface.co/datasets/qintongli/GSM-Plus) | `3b708db57b96a16e8e3368ed2956990c0809440e` |   |
| ifeval | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | `966cd89545d6b6acfd7638bc708b98261ca58e84` | train  |
| ifbench | [allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test) | `ed7c12452594b7028b15807b86d3938150b663ae` | train  |
| multi_if | [facebook/Multi-IF](https://huggingface.co/datasets/facebook/Multi-IF) | `0ab97ce0b45c7f57772e8ba2ac1616f4b00bd3aa` |   |
| evalplus_grader | [evalplus_grader](https://github.com/evalplus/evalplus.git) | `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` |   |
| lcb_grader | [lcb_grader](https://github.com/LiveCodeBench/LiveCodeBench.git) | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` |   |
| bcb_grader | [bcb_grader](https://github.com/bigcode-project/bigcodebench.git) | `09dd993f46c3fbf3a799465bb96d524edcb0b199` |   |
| taco_grader | [taco_grader](https://github.com/FlagOpen/TACO.git) | `245eba3beb2d23a07082307de303de2589e4321a` |   |
| ifeval_grader | [ifeval_grader](https://github.com/google-research/google-research.git) | `c878a53ec1764bcb91a6178dc5c315202a3b5fa2` |   |
| ifbench_grader | [ifbench_grader](https://github.com/allenai/IFBench.git) | `488918665d53f867fa908a293769ec6e71e74aec` |   |
| multi_if_grader | [multi_if_grader](https://github.com/facebookresearch/Multi-IF.git) | `1cdb53ed18499ad729e0766e5d3099dd5344406f` |   |
| olympiad_grader | [olympiad_grader](https://github.com/OpenBMB/OlympiadBench.git) | `ba5b26a7e2849940b598a9159c1190daa2b9175f` |   |
| taco_test | [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO) | `d593ed0a2becbbc952230bb89be09189bf1056dc` | test default |

数据：TACO train25,443题（test1,000仅保留测试来源，不作开发集）；NuminaMath1.5原896,215题，**先按source、再按source内problem_type同比例抽50,000原题**，seed42、最大余数补齐精确总额，之后准入，因此合格池可能少于50k；Tulu IF原29,980条。证明题可SFT，不能进入自动判分开发题。三域末尾完成原生标签映射、长度标签与分布、每技能50题的未来开发集；EXP001 Full可能见过这些题，因此不能当EXP001独立验证，后续从基础模型重启并隔离。

12个benchmark：LCB release_v6、HumanEval+、MBPP+、BCB instruct/full；MATH500、AIME25/26、Olympiad英文纯文本开放数学、GSM-Plus；IFEval、IFBench、Multi-IF英文三轮。EvalPlus完整增强测试不能只用HF镜像顶替。LCB/EvalPlus/BCB/TACO生成代码只在无网络受限CPU容器执行，LCB官方多进程8路；IF官方规则判分，Multi-IF只报英文三轮对应口径。MathVerify用于MATH/AIME/GSM答案等价是项目协议；Olympiad采用其单独依赖的官方checker。

## 6. 软件与镜像

工作站联网完成模型、数据、源码checker、增强测试、NLTK等资源下载和镜像构建，再seal文件checksum并导出docker save/load给离线集群。GPU镜像和CPU通用判分、BCB专用判分分开；不在集群pip install或下载。当前base image已锁linux/amd64 digest；最终项目镜像须在工作站构建后再记录自己的digest，不能把base digest冒充项目镜像digest。

| 镜像用途 | 已锁定的base image |
|---|---|
| gpu | `vllm/vllm-openai@sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484` |
| prepare_grader | `python@sha256:05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177` |
| bigcodebench | `bigcodebench/bigcodebench-evaluate@sha256:1327bddf60be9bc241648c59e6060cac4ca50248a0588ab735cd0200b17cc8c2` |

对应标签为vllm/vllm-openai:v0.34.0、python:3.12-slim、bigcodebench/bigcodebench-evaluate的本次解析版本；Dockerfile实际用摘要，不再追随latest。来源检查记录在configs/ordinary/environment-lock.json。

GPU项目显式版本：Transformers5.22.0、Accelerate1.16.0、PEFT0.22.0、datasets5.2.0、PyArrow26.0.1、smolagents1.26.0、OpenAI SDK3.57.0、flash-linear-attention0.6.0、causal-conv1d1.7.0；vLLM0.34.0来自固定base。Torch/CUDA/NCCL/Triton由该base提供，不自行另装一个Torch破坏vLLM配套。jsonschema/nvidia-ml-py/PyYAML仍有允许范围，最终pip freeze和pip check作为环境锁证据，不声称当前所有传递依赖已静态完全锁死。

官方入口：[vLLM镜像](https://hub.docker.com/r/vllm/vllm-openai)、[BCB镜像](https://hub.docker.com/r/bigcodebench/bigcodebench-evaluate)、[Transformers](https://huggingface.co/docs/transformers)、[Accelerate FSDP](https://huggingface.co/docs/accelerate/usage_guides/fsdp)、[PEFT](https://huggingface.co/docs/peft)、[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)。这些Python包的选定版本已查公开发布元数据；**没有在CPU本机真实构建CUDA镜像**，版本存在不保证组合兼容。

## 7. 日志、恢复、交付

运行根按UTC时间-ID独立命名；计划保存源码/配置/模型/数据身份。每checkpoint记录实际有效输入token、监督token、样本曝光、更新步数、墙钟、GPU时长和吞吐；生成记录输入/输出token和耗时；判分另计CPU墙钟。各训练session与恢复偏移保留，不用mock零值补真实成本；丢失、未持久化的成本明确记为未知，不能声称完整核算了所有中断前GPU工作。`summary.csv / training_costs.csv / scores_wide.csv / REPORT.md`与NICE子目录分别导出；打包交付保留checksum，权重和原始大资产按既有bundle策略不塞进结果zip。

完整native checkpoint保留adapter、optimizer、scheduler及**4个rank的RNG**；只有adapter不能声称原训练断点续跑。不完整节点移开保留证据，回到最近完整节点；已完成的生成/判分按身份复用。中断后必须同源码/配置/数据/UUID归属，使用--resume；本轮变更源码后不能复用旧计划或篡改旧hash，B的新receipt也要重做。

## 8. 放行前仍需服务器完成

1. 填8个独占GPU UUID、共享绝对路径、CPU资源和Docker daemon/公司容器作业能力；本实现的CPU判分入口使用Docker CLI，需要实际可用的daemon，而非只有命令行。
2. 联网工作站构建三个target和BCB镜像，pip check/freeze，测试离线导入和官方checker，然后封存**最终镜像digest**；核实H200驱动支持base内CUDA。
3. 4rank完整8192长度反向、保存、强制中断恢复；每卡micro4×累积2是当前配方。可测micro2×4、4×2、8×1，global固定32；正式运行前根据实测有效token吞吐冻结一组，不能自动OOM降预算。NICE另验micro2/global8。
4. 两路并行训练/生成、每卡64序列、4副本热切adapter、CPU判分取消均真实smoke；确认适配器模块兼容、显存不溢出、进程互不清理。CPU至少留2×8核/2×48GB给判分，并给训练dataloader和服务额外余量。
5. NICE八卡真实反向和原生TACO参考解自检；不得把单位奖励probe当成成功率验收。

A在自己的环境完成下载、构建和预检后，填齐A的bindings/receipt并放行approved，再启动冻结后的新计划。B通过仅提供提前测试参考。现有本地CPU结果证明流程及数值实现，不证明真实8卡速度。本轮不承诺未测得的运行小时数：记录预热后有效token/s和生成token/s，用总实际token预算换算，再加官方判分和加载时间。避免以GPU显存大小推导虚假的完成时间。

依赖核查实际发现并修复：vLLM0.34.0声明Transformers>=5.16.1,<5.23.0，旧pin5.23.0越界；已统一改5.22.0，亦满足FLA>=5.6.2,<6。vLLM要求Torch2.14.0，实际Torch仍由base提供并在工作站pip check确认。

八套官方checker固定Git commit和smolagents/OpenAI/EvalPlus的选定版本也已逐项查询公开API，全部来源存在；环境锁内保存核查URL。

## 配置原文件索引

- [实验一全部参数](../configs/ordinary/exp001.json)
- [NICE专用覆盖参数](../configs/ordinary/exp001_nice.json)
- [全部来源、revision、分割与导出设置](../configs/ordinary/sources.json)
- [环境锁及逐条来源核查](../configs/ordinary/environment-lock.json)
- [GPU/准备/通用判分Dockerfile](../docker/Dockerfile)
- [BCB独立Dockerfile](../docker/Dockerfile.bigcodebench)
- [GPU Python依赖](../docker/ordinary-runtime.txt)

本轮本地验收：Transformers5.22.0环境下33项流程smoke、22项完整双LR＋NICE/中断恢复、26项配置负例、5项调度/热切HTTP模拟通过；另有真实微型CPU反向/有限差分/流式投影及4/8rank成本张量归约检查。凭据在被忽略的runs/recipe-acceptance-20261121/，CPU验收zip已做CRC检查，未进行GPU或Docker验收。

2026-11-21最新确认：EXP001普通基线epoch1/2/3全部跑benchmark；Initial仍只测一次。epoch3为主要报告，不按benchmark挑权重。两LR正式矩阵372组benchmark；NICE保持48组benchmark＋12组原题诊断，总432组测评记录（420 benchmark＋12诊断）。师兄B仍保持独立短测：每个benchmark只测一次，不扩大验收规模。此前相反节点说明以本条为准。

2026-11-21用户确认标签：保留8/7/10类；未映射样本先不标、保留训练池、不自动调用teacher；按技能开发集暂不使用未映射记录，缺额如实报告。原生对应关系见[NATIVE_SKILL_MAPPING_ZH.md](NATIVE_SKILL_MAPPING_ZH.md)。

2026-11-21部署口径：A/B异地独立。A从零下载资产、构建镜像并本机验收；B仅提前测试反馈，B的目录、缓存、镜像或凭据不是A的运行依赖。A内工作站与集群之间可共享／迁移A自己的资产。
