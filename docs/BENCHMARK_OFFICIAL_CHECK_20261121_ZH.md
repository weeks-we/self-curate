# Benchmark与八卡配置复核（2026-11-21）

> 最新长度与逐benchmark预算见[长度/预算核查](TRAINING_LENGTH_AND_BENCHMARK_BUDGETS_ZH.md)。原文统一4096/8192输出仅为旧设置；当前执行exp001.evaluation.benchmark_budgets。三个epoch测评及A/B范围遵循HANDOFF。


## 当前测评次数

每个训练分支只用第3 epoch权重，对其领域每个benchmark生成一次、判分一次。仍保存三个epoch checkpoint和每份训练成本，不再测epoch1/2。Initial仍按每个benchmark一次。正式EXP001为18分支、15次训练、45训练checkpoint、72组最终测评；师兄B的三领域短训只需12组最终测评，不要求Initial。换协议须新plan/run_id，不能改旧run。

## 官方判分与下载

12个HF固定revision和实际文件已逐项通过公开API访问检查；这验证来源存在，不等于容器已跑通。sources.json是下载与版本权威，不能仅按下面简称拉最新版本。

| Benchmark / HF来源 | 当前判分与模式 | 环境与核查结论 |
|---|---|---|
| [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite) release_v6 | 官方LCB code-generation测试执行器、单答案pass@1；非self-repair/code-execution模式 | HF的test至test6文件；官方源码依赖+通用自建CPU grader镜像；6秒参数为项目冻结配置，不冒称所有发布版本默认 |
| [HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus) | 官方EvalPlus base+增强测试、单答案pass@1 | HF只作来源参考；正式题单必须用EvalPlus完整增强release缓存，不用HF镜像代替；可参考官方ganler/evalplus，本项目统一自建镜像 |
| [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) | 同上 | 注意HF镜像与EvalPlus release题数可能不同，冻结实际ID、完整测试与hash |
| [BigCodeBench](https://huggingface.co/datasets/bigcode/bigcodebench) v0.1.4 | 官方evaluate，instruct/full、calibrated=true、local、pass@1；canonical解先执行，按官方流程获得题目时间预算 | 独立Dockerfile.bigcodebench，官方候选bigcodebench/bigcodebench-evaluate；latest必须在工作站解析并锁定实际digest。已去掉旧的直接check_correctness+固定2/60秒调用；从冻结HF行注入官方evaluate，避免集群下载题单 |
| [MATH500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Math-Verify最终答案等价 | 这是项目选择的公开判分协议，非统一官方生成/判分配方；纯CPU通用镜像 |
| [AIME2025](https://huggingface.co/datasets/math-ai/aime25) | Math-Verify数值答案等价 | HF托管题单，不意味着数学竞赛官方模型评测协议；单greedy答案，不冒称原论文avg@32等统计 |
| [AIME2026](https://huggingface.co/datasets/math-ai/aime26) | 同上 | 完整题单gold解析仍需工作站验收；不评证明过程 |
| [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | 官方MathJudger；OE_TO_maths_en_COMP英文纯文本开放数学题，采用题目error精度 | HF为托管副本；核对官方OpenBMB来源字段；独立SymPy/ANTLR venv已在通用grader镜像定义，不下载证明/图片题充数 |
| [GSM-Plus](https://huggingface.co/datasets/qintongli/GSM-Plus) | 当前Math-Verify数值/表达式最终答案等价 | 官方支持HF下载test/testmini；本项目固定test。Math-Verify不是该仓库原始抽取脚本的逐字复现，需完整gold可解析审计；特殊答案不能误算0分 |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | 官方strict/loose × prompt/instruction四项；strict prompt主列 | 官方google-research checker；通用CPU镜像预置NLTK资源，题单字段名虽为train仍只作最终评测 |
| [IFBench](https://huggingface.co/datasets/allenai/IFBench_test) | 官方checker；主列改为论文采用的loose prompt，strict及instruction分数保留 | 原来strict主列已修正；官方README明确temp0，非thinking适用；pip/Git固定提交安装 |
| [Multi-IF](https://huggingface.co/datasets/facebook/Multi-IF) | 官方三轮上下文与checker；每轮四指标均值、再跨轮平均；另保留各轮四指标和全三轮同时成功诊断 | 原来把全三轮全对作为唯一主分数已修正。当前仅English，不能声称复现全语言总分；官方Git+HF CSV，通用CPU镜像 |

“官方checker相同”不等于“所有模型生成参数统一官方默认”。我们固定temperature0、top_p1、每题一个答案、关闭thinking；代码/IF输出4096、数学8192、context16384。这些是本项目的共同预算。HumanEval等采用统一chat包装也要随template hash公布，不能拿本项目分数无条件等同论文原始配方。

### 官方依据

- [LiveCodeBench官方仓库](https://github.com/LiveCodeBench/LiveCodeBench)：任务模式、release与评分；[EvalPlus官方](https://github.com/evalplus/evalplus)：增强测试、vLLM生成、Docker判分。
- [BigCodeBench官方指南](https://github.com/bigcode-project/bigcodebench/blob/main/ADVANCED_USAGE.md)提供镜像；实际参数按我们锁定的[官方evaluate源码](https://github.com/bigcode-project/bigcodebench/blob/09dd993f46c3fbf3a799465bb96d524edcb0b199/bigcodebench/evaluate.py)执行，不能把新版文档默认值误套旧提交。
- [IFEval官方](https://github.com/google-research/google-research/tree/master/instruction_following_eval)、[IFBench官方README](https://github.com/allenai/IFBench)、[Multi-IF论文4.1节](https://arxiv.org/html/2410.15553v1)与[官方仓库](https://github.com/facebookresearch/Multi-IF)定义各自指标。
- [OlympiadBench官方](https://github.com/OpenBMB/OlympiadBench)、[GSM-Plus官方](https://github.com/qtli/GSM-Plus)、[Math-Verify](https://github.com/huggingface/Math-Verify)。数学集合不存在统一必用的容器镜像，按实际checker构建依赖即可。

## 8×H200（141G）训练与测评

Qwen3.5-9B、三领域独立训练；HF Trainer/Accelerate原生FSDP2+PEFT，BF16，LoRA r32/alpha64/dropout0、LR2e-5、3epoch、warmup3%、weight_decay0、完整长度8192；每卡micro4×累积1×8=global32。底座冻结；监督assistant有效答案token。其他科研参数仍处于草案批准状态。

现有优化：八卡训练；Qwen3.5 FLA/causal-conv优化kernel强制检查；长度分桶；gradient checkpointing；8对齐padding；pinned-memory与persistent worker；fused AdamW；FSDP2 reshard_after_forward=false利用141G显存保留前向参数，减少反向重新all-gather。后者会增加显存，必须真实最长batch预检；如失败可在新验收配置改true，不静默更改正式协议。

不要为了占显存盲目把global32提高到256：这会改变optimizer更新数。服务器预检固定global32，比较micro1×累积4、micro2×累积2、micro4×累积1；每种预热2步、计量5步，记录有效输入/监督token每秒、step时间、每卡峰值显存，采用通过真实反向和恢复检查的最高吞吐组合并统一冻结。全长8192容量测试另行执行，小样本预热不能替代。

测评8个单卡vLLM副本、TP1、每副本并发32（总上限256）、memory0.9，按真实吞吐和截断率验收。CPU判分独立资源池，不能挤占GPU生成进程CPU导致卡空等；当前自动后端仍需平台Docker或CPU作业适配。

本机只有CPU，无法声明上述设置达到最佳GPU利用率。师兄B需返回真实吞吐/显存/恢复/容器证据；新吞吐参数未验证前不得称正式环境全部通过。

本轮验收：12/12 HF固定revision/文件在线存在；CPU全流程33项通过（72测评、45训练成本行、断点与幂等）；IFBench/IFEval/Multi-IF指标数值fixture及BCB完整evaluate接入参数、离线注入、异常后恢复的mock检查通过。BCB真实执行、官方容器构建、GPU吞吐与FSDP2恢复仍待服务器。凭据：runs/benchmark-official-audit-20261121。
