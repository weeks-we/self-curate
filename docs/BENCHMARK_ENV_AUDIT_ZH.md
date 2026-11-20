> Current proof policy: valid reference proofs are retained for SFT; proof examples carry evaluation_eligible=false and are excluded from automatically graded development tasks. Earlier blanket proof-exclusion statements below are superseded. Full input length is prompt + answer + chat-template tokens, not answer tokens alone.

# Benchmark环境与数学题型核查（2026-11-20）

## 结论与验收边界

不能宣称所有正式环境已加载通过。本机没有Docker命令，也未启动真实GPU、容器或执行生成代码。本轮完成了12个数据加载路由的公开样例检查、官方IF检查器CPU正反例、四个数学集合真实样例标准答案解析，以及OlympiadBench官方MathJudger独立依赖正反例。

这些是接口与小样验收，不代替完整题单、Linux容器、资源上限、断网和全部约束类型验收。EvalPlus缓存加载回归使用本地fixture，不是完整增强题单验收。

| Benchmark | 判分与环境 | 本轮/当前状态 |
|---|---|---|
| LiveCodeBench release_v6 | 官方LCB测试执行器；Python/Linux，多进程、signal、超时；生成代码需要沙箱 | 样例加载通过；官方执行器未在容器运行 |
| HumanEval+ | 官方EvalPlus、完整增强测试，CPU沙箱 | 缓存加载分支已修复并回归；完整容器执行待验收 |
| MBPP+ | 同上，冻结真实release题单 | 同上，不能用HF简化镜像题数代替 |
| BigCodeBench instruct/full | 官方runner、第三方函数库、系统库；独立CPU沙箱 | 原Python3.12通用镜像不兼容其旧依赖，已拆出独立构建候选；尚未build |
| MATH-500 | 本项目选用Math-Verify最终答案等价判定 | 真实坐标答案样例通过，修复部分数字误解析；不判断证明过程 |
| AIME2025 | 数值最终答案判定 | 真实标准答案样例解析通过；完整题单待验收 |
| AIME2026 | 同上 | 同上 |
| OlympiadBench OE_TO_maths_en_COMP | 官方MathJudger，独立SymPy/ANTLR环境 | 纯文本样例加载及官方判定器正反例通过；容器待验收 |
| GSM-Plus | 当前选用Math-Verify；须核对全体gold/特殊题型 | 真实标准答案样例通过；完整gold解析审计待验收 |
| IFEval | 官方规则checker、NLTK等CPU依赖 | 官方关键词正反例通过；全部规则类型待验收 |
| IFBench | 固定官方package/规则，预置NLTK资源 | 官方正反例通过；已修正只读挂载下的import/cache风险 |
| Multi-IF英文 | 官方metrics/ifeval，保留模型自己的三轮上下文；即使英文也导入Thai依赖 | 官方三轮正反例通过；完整数据和断网镜像待验收 |

TACO test仅保留最终测评用途，不参与开发或调参。其完整官方特殊判分适配仍未完成，当前相关路径明确拒绝运行，不能复用LCB当作完整TACO判分。

## 本轮修复的实际问题

1. benchmarks.py使用read但漏导入，EvalPlus缓存分支会NameError；已修复并实际走该分支。
2. MATH-500样例标准答案为坐标，直接按裸字符串解析会只提取其中数字3。现将gold作为完整LaTeX答案解析，验证“完整坐标正确、仅回答3错误”；gold无法解析时报告协议错误，不计成模型答错。
3. BigCodeBench官方固定requirements含NumPy1.21.2、Numba0.55等旧依赖，不应混入Python3.12数学/IF环境。通用grader已移除BCB，新增docker/Dockerfile.bigcodebench，基于其官方评测镜像提供构建候选；ServerBackend要求单独的digest-pinned绑定，不静默使用通用环境。
4. IFBench导入会创建模块旁.nltk_data。现在使用镜像中固定安装的package、构建时导入创建目录，不在只读源码挂载内写入。
5. NLTK除punkt/punkt_tab之外还需stopwords、averaged_perceptron_tagger_eng。已加入预置；Multi-IF的PyThaiNLP缓存也改为镜像预置目录，并配置离线只读运行。

本机CPU测试遇到代理下NLTK动态下载失败、PyThaiNLP默认写用户目录的问题，不能忽略这些报错。正式环境必须预置资源，并在断网只读模式复测，不依赖运行时下载。

## 官方与其他项目的做法

- [EvalPlus官方README](https://github.com/evalplus/evalplus)提供Magicoder示例：先本地/vLLM生成，再在ganler/evalplus Docker中执行测试。这支持将GPU生成和CPU判分分开，而不是让模型拥有宿主shell。
- [LiveCodeBench官方README](https://github.com/LiveCodeBench/LiveCodeBench)提供独立评测及超时/CPU并发说明；多进程超时执行器不等同安全沙箱，我们仍在外部加隔离。
- [BigCodeBench官方高级指南](https://github.com/bigcode-project/bigcodebench/blob/main/ADVANCED_USAGE.md)提供预构建评测镜像与独立requirements，并推荐沙箱；不能只装runner就认定题目函数库齐全。
- [EvalPlus官方列出的采用者](https://github.com/evalplus/evalplus)包括Magicoder、StarCoder2、Qwen、Tulu等。采用相同评测工具不意味着每个工作的镜像版本、超时与prompt完全相同，不据此臆测其具体部署。

公司平台建议：工作站准备并冻结通用grader和BCB独立镜像；GPU作业输出答案到共享目录；CPU容器作业按官方规则判分。当前自动后端仍是Docker；平台CPU作业/Apptainer调度尚需适配。Apptainer默认共享宿主部分资源，不可把普通exec等同沙箱。

通用镜像构建：docker build --target grader -t self-curate-grader -f docker/Dockerfile .

BCB镜像构建候选：docker build -t self-curate-bcb-grader -f docker/Dockerfile.bigcodebench .

构建成功后冻结实际digest，填入server-bindings的grader_image（通用）与grader_images.bigcodebench（BCB）。还要保存环境freeze、官方样例成功/失败/超时记录、离线和只读运行证据。默认latest只是待解析的构建候选，不是正式来源锁。

## 数学：题型与证明正确性

[NuminaMath-1.5官方数据卡](https://huggingface.co/datasets/project-numina/NuminaMath-1.5)明确区分：question_type为MCQ、proof、math-word-problem；problem_type为数学领域，两者不是同一维度。answer='proof'表示证明题，'notfound'表示未提取到答案。

| 题型 | 如何判定 | 当前项目政策 |
|---|---|---|
| 数值/表达式输出 | 提取最终答案，数值/符号等价检查 | 保留；它不验证每一步推导 |
| MCQ | 必须统一选项字母/序号与选项内容，再比较 | 训练准入可保留；后续开发题的选项判分适配需专项检查 |
| 自然语言证明 | 人工审阅；模型裁判只能作近似判断 | 当前数学准入排除，不用最终答案匹配冒充证明验证 |
| 形式化证明 | Lean/Isabelle等检查器验证形式化代码 | 不是当前自然语言题自动具备的环境，需要另行形式化与配置 |

证明题可以用于SFT，但“有参考证明可训练”和“模型的新证明能自动严格判定”是两回事。我们当前选择排除，是为了统一低成本的开发集自动测评，不是认为证明题没有训练价值。

当前50k原始分层抽样可能抽到proof，随后硬筛会排除，所以最终合格数学池可能不足50k，且准入后分布会变化。

五个数学benchmark的当前测评路径主要检查最终答案。OlympiadBench整个项目包含OE开放答案题和TP证明题；我们固定的是OE_TO_maths_en_COMP，不含TP子集。[官方说明](https://github.com/OpenBMB/OlympiadBench)对开放答案提供自动判分，对证明题采用抽样评估。不能宣称当前框架已经解决证明正确性评估。


?????????NLTK?????PyThaiNLP???????????socket???nltk.download?????IF checker???????????????????????????????????????
