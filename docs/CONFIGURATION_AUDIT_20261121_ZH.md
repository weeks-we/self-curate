# 最终配置核查（2026-11-21）

> 本轮生效配置与研究依据见[训练和环境总表](EXP001_QWEN35_9B_RECIPE_AND_ENV_ZH.md)。LR已选5e-5/1e-4；两路各4卡，主实验global32、NICE global8；异步CPU判分。approved=false仅是服务器验收门。后文历史“草案/待用户check”不再作为科研参数待办。


权威：configs/ordinary/exp001.json、exp001_nice.json、sources.json；A/B入口见HANDOFF.md。配置核对通过不等于服务器环境已经通过。approved仍为false，两档学习率已依据调研选定，服务器验收仍待完成。

## 核对与修正

1. roadmap补上Random30和内置NICE，普通报告节点改为epoch3；保存节点仍为epoch1/2/3。
2. NICE实际1epoch/三进度节点，不再把普通实验的evaluation.epochs=[3]带入其生成协议；单独记录progress_fractions和primary_progress。
3. 主实验和NICE只共用training.learning_rates=[5e-5,1e-4]，移除无执行作用的历史三档候选字段。learning_rate=1e-4标量只保留给单job/B短训默认，正式矩阵由两档列表逐分支覆写。
4. 新增configuration.py，在prepare/plan前验证实际支持的训练、保存、分片、采样和测评设置；NICE的SFT epsilon/clip必须与fresh-Adam一致，两处保存比例必须相同，B不得开启训练或测评。冲突直接报错，不静默忽略字段。
5. throughput_preflight明确标为服务器agent人工实测要求，当前没有自动调参器；不会自行改变global batch或学习率。

## 当前执行配置

| 项目 | 普通Full/Random | NICE |
|---|---|---|
| 目标模型 | Qwen3.5-9B，关闭thinking | 同一基础模型，关闭thinking |
| 数据 | TACO train；NuminaMath1.5分层50k后准入；Tulu Personas IF | TACO train持出20任务，排除后随机3000候选 |
| 抽样seed | 42 | 42；projection seed42 |
| 学习率 | 5e-5、1e-4，本轮已选 | 唯一读取同一列表 |
| LoRA | r32/alpha64/dropout0 | r16/alpha32/dropout0，参考旧NICE |
| 有效batch | 4卡×micro4×累积2=32 | 4卡×micro2×累积1=8 |
| 框架/精度 | 原生HF Trainer/Accelerate FSDP2/PEFT，BF16 | 训练相同框架；梯度为8卡独立样本分片，不跨卡平均候选梯度 |
| Epoch | 3 | 1 |
| 完整SFT长度 | 8192，整条超长排除 | 8192；回放还核对生成context预算 |
| 优化器 | fused AdamW；weight_decay0；warmup3% | 同上，epsilon1e-8、clip1与打分绑定 |
| 保存 | 全部3个epoch，完整恢复状态 | 42/84/125步；曝光336/672/1000，完整恢复状态 |
| 测评 | 测epoch1/2/3，epoch3为主报告 | 三个进度节点各测原20题＋四coding benchmark |
| checkpoint选择 | 不按最终bench择优 | 不按最终bench择优，原20题为诊断 |

NICE随机rollout是20×4=80次，temperature1/top_p1/top_k0、max_new_tokens4096/context16384；实际token IDs由本地HF保存，不依赖重新分词。失败奖励0保留固定4分母，20题零方向保留；所有方向为0则停止，不能生成任意高低排名。

最终测评为greedy单答案，temperature0/top_p1，代码/IF输出4096、数学8192、context16384；8个单卡vLLM副本、TP1、每副本并发64起测、memory0.9。不能把这些工程预算说成12个benchmark统一官方默认。B负责真实容量/吞吐/逐题长度验收。

优化项保留长度分桶、8对齐padding、pinned memory/persistent worker、FLA/causal-conv kernel检查和梯度checkpointing；FSDP2 reshard_after_forward=false增加显存占用以减少重复通信，需要服务器最长batch确认。HF默认Adam beta0.9/0.999与linear scheduler未单独暴露为JSON参数，随锁定transformers版本执行。

## 数量和A/B边界

- 普通：33分支（含3个Initial）、30次SFT、90训练checkpoint、372组分阶段benchmark。
- NICE：4次SFT、12训练checkpoint、48组benchmark＋12组原20题诊断。
- 合计：34次SFT、102训练checkpoint、432组统计。Initial没有训练checkpoint，不加入训练曝光。
- B原环境短训：仅9B三个领域各128条×3epoch，最终各领域bench共12组；另外NICE仅2题×1次64token回放＋8候选反向probe，不判分、不筛选、不SFT、不额外测评。
- A正式run先NICE后常规基线。B的梯度凭据需与模型/LoRA/候选池/源码/资产一致；本轮代码变更后旧plan/旧凭据不能直接作为新版验收，需生成新的计划和B记录。

## 固定来源

三个模型均在sources.json固定revision；本轮核对字段，不重新下载权重。12个benchmark的固定revision与文件存在性在前一轮已在线通过核查，本轮没有把它当作容器执行成功。

| 模型 | HF | revision |
|---|---|---|
| qwen9b | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| qwen4b | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| llama8b | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `0e9e39f249a16976918f6564b8830bc894c89659` |

| 领域 | benchmark来源 | 环境 |
|---|---|---|
| coding | [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) | 通用自建grader，固定官方checker或约定Math-Verify |
| coding | [evalplus/humanevalplus](https://huggingface.co/datasets/evalplus/humanevalplus) | 通用自建grader；EvalPlus额外完整增强缓存 |
| coding | [evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus) | 通用自建grader；EvalPlus额外完整增强缓存 |
| coding | [bigcode/bigcodebench](https://huggingface.co/datasets/bigcode/bigcodebench) | 独立BigCodeBench官方候选镜像构建并锁digest |
| math | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 通用自建grader，固定官方checker或约定Math-Verify |
| math | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) | 通用自建grader，固定官方checker或约定Math-Verify |
| math | [math-ai/aime26](https://huggingface.co/datasets/math-ai/aime26) | 通用自建grader，固定官方checker或约定Math-Verify |
| math | [Hothan/OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | 通用自建grader，Olympiad另有独立Python venv |
| math | [qintongli/GSM-Plus](https://huggingface.co/datasets/qintongli/GSM-Plus) | 通用自建grader，固定官方checker或约定Math-Verify |
| if | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | 通用自建grader，固定官方checker或约定Math-Verify |
| if | [allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test) | 通用自建grader，固定官方checker或约定Math-Verify |
| if | [facebook/Multi-IF](https://huggingface.co/datasets/facebook/Multi-IF) | 通用自建grader，固定官方checker或约定Math-Verify |

具体官方模式/项目判分差异仍以BENCHMARK_OFFICIAL_CHECK_20261121_ZH.md为准。尤其IFBench用loose主分，Multi-IF只报告English四指标均值，数学Math-Verify不冒称完全复现所有原论文协议。TACO开发任务用新接入的固定官方checker，A先跑20参考解自检，B的梯度probe不覆盖其判分。

## 尚需确认/填写

| 项目 | 当前状态 | 由谁处理 |
|---|---|---|
| 两档LR的最终数值 | 5e-5/1e-4为草案，approved=false | 用户check科研参数后，再冻结正式计划 |
| GPU UUID | 主配置空数组；不能在approved=true下留空 | B/A填写各自8个真实UUID |
| 资产/共享路径、模型下载权限 | 工作站准备，Llama需HF权限 | B配置并提供清单；A核对复用 |
| 通用/BCB镜像digest、GPU镜像及环境freeze | 构建候选存在，真实digest由工作站产出；不能只用latest放行 | B构建并验收 |
| server_smoke_receipt及nice_gradient_smoke_receipt | 需真实文件、hash和身份匹配 | B执行后生成；A复核 |
| 八卡最长batch、实际吞吐、主存/磁盘、平台CPU判分作业能力 | 本机无法验证；throughput预检不是已执行的自动调参 | B实测记录；A使用冻结配置 |
| 每技能50题的真实覆盖 | 按完整数据标签分布验收；不足不复制，不自动合并 | EXP001末尾报告缺口再处理 |

本轮未新增付费teacher调用或要求填写API key。无真实GPU/容器凭据前，只能说配置/CPU控制流通过，不能说正式实验已具备全部实测证据。

本轮最终验收：26项配置一致性/非法覆写检查、22项NICE整合集成、33项原基线回归通过；CPU数值和恢复凭据校验复查通过。配置快照和凭据位于忽略的runs/final-configuration-audit-20261121。仍未在本机执行真实9B/GPU/Docker；科研超参整体保持draft/approved=false。当前launcher为单机8卡，多节点部署需单独适配。

2026-11-21最新确认：EXP001普通基线epoch1/2/3全部跑benchmark；Initial仍只测一次。epoch3为主要报告，不按benchmark挑权重。两LR正式矩阵372组benchmark；NICE保持48组benchmark＋12组原题诊断，总432组测评记录（420 benchmark＋12诊断）。师兄B仍保持独立短测：每个benchmark只测一次，不扩大验收规模。此前相反节点说明以本条为准。
