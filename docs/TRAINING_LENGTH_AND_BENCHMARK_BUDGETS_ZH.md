# 训练长度与 benchmark 生成预算核查

2026-11-21。训练长度、单次生成输出上限、服务端上下文窗口是三个不同参数。本次使用现有固定Qwen3.5-9B tokenizer和关闭thinking的实际聊天模板，核对完整token序列与训练renderer一致；没有下载模型权重或运行GPU。

## 1. 三个训练集的实测结果

完整长度＝模型模板＋user题面/指令＋assistant参考解，**不是仅题目，也不是仅answer字段**。表中P99表示99%的已测样本不超过该值。答案列单独分词，可能与上下文中assistant掩码计数有轻微边界差异；完整长度为实际训练模板精确计数。

| 领域/测量范围 | 有效测量数 | 完整最短 | P50 | P95 | P99 | 完整最长 | 超8192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TACO train全池，按同一固定参考解规则 | 18,834 | 57 | 618 | 1515 | 2337 | 617279 | 15（0.0796%） |
| Numina1.5分散窗口抽查，非最终50k | 1,624 | 69 | 412 | 1092 | 1586 | 2463 | 0（0.0000%） |
| Tulu Personas IF全池 | 29,980 | 30 | 336 | 841 | 1596 | 9896 | 1（0.0033%） |

| 领域 | 题目token P50 / P99 / max | 参考答案token P50 / P99 / max |
|---|---|---|
| coding | 424 / 1300 / 13616 | 146 / 1452 / 617109 |
| math | 81 / 290 / 724 | 313 / 1451 / 2351 |
| if | 54 / 168 / 607 | 262 / 1508 / 9810 |

测量边界：

- TACO读取固定revision全部25,443条train的题面、参考解、starter_code和图片标记；参考解使用现有稳定hash选择，执行同样的非空/图片/语法筛查。为测长度不下载约2GB测试字段，因此这里**未做原生tests合法性、题面去重及benchmark隔离**，不是最终训练池人数。最终准入池仍由正式prepare决定。
- Tulu IF覆盖全部29,980条原始记录的两消息模板；未进行最终去重/测试隔离。
- Numina按seed42选20个分散窗口×100条，共2000条；现有质量准入后1624条可测，无viewer截断行。184条source_invalid、49条缺可验证答案、143条缺监督文本被排除。不是全896,215条或最终分层50k的随机独立样本，**不能把抽查0条超长解释为全池0条超长**。
- Numina官方viewer统计是字符数：problem最大45,303字符，solution最大26,844字符，两者最大值不必来自同一题，也不能当作token数。SFT使用solution整段推导，不是平均很短的最终answer字段。[官方数据](https://huggingface.co/datasets/project-numina/NuminaMath-1.5)

本次不因少数长尾自动修改8192训练上限。正式prepare现在会在长度排除之前，写入manifest.length_audit_before_length_filter：min/P50/P95/P99/max，以及超过4096/8192/12288/16384/32768的条数。必须对最终Numina50k和三领域实际候选池看此表后冻结；如要提高上限，须重新准备共同候选池、重新做最长batch容量验收、创建新计划。不能先按8192筛掉长样本后，再从幸存样本宣称8192足够。

原始样本、列裁剪Parquet、逐项数值证据保存在被忽略的runs目录，不提交Git。正式实验仍整条排除超长样本，不截断答案，不从多个参考解中为了长度改挑更短的解。

## 2. 逐 benchmark 预算

用户要求按官方文档。落实方式：**优先采用锁定官方runner明确写出的默认值；只有模型示例时注明示例；官方没有统一数值时保留明确标记的项目预算**。默认值不等于benchmark对所有模型的强制规定，也不代表本项目完整复现官方的prompt/采样/判分协议。

| Benchmark | 每次输出上限 | 有效上下文上限 | 依据 |
|---|---:|---:|---|
| livecodebench | 2000 | 16384（项目容量，官方未统一指定） | [official_locked_runner_default](https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/runner/parser.py)；Official output default2000; no common context limit specified. Context16384 is project capacity, not official requirement. |
| humaneval_plus | 768 | 2048（官方runner默认） | [official_locked_vllm_default](https://raw.githubusercontent.com/evalplus/evalplus/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/provider/base.py)；Output768; context2048 from the same commit provider/vllm.py. Historical runner defaults, not a benchmark-wide mandatory rule. |
| mbpp_plus | 768 | 2048（官方runner默认） | [official_locked_vllm_default](https://raw.githubusercontent.com/evalplus/evalplus/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/provider/base.py)；Same EvalPlus vLLM context2048 default. |
| bigcodebench | 1280 | 12800（官方runner默认） | [official_locked_runner_default](https://raw.githubusercontent.com/bigcode-project/bigcodebench/09dd993f46c3fbf3a799465bb96d524edcb0b199/bigcodebench/generate.py)；Official run_codegen output1280/context12800. |
| math500 | 8192 | 16384（项目容量，官方未统一指定） | [project_fallback_no_universal_official_budget](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)；Data split does not prescribe one universal LLM generation/context budget. |
| aime25 | 8192 | 16384（项目容量，官方未统一指定） | [project_fallback_no_universal_official_budget](https://huggingface.co/datasets/math-ai/aime25)；Competition questions do not define LLM token budgets. |
| aime26 | 8192 | 16384（项目容量，官方未统一指定） | [project_fallback_no_universal_official_budget](https://huggingface.co/datasets/math-ai/aime26)；Competition questions do not define LLM token budgets. |
| olympiad | 2048 | 16384（项目容量，官方未统一指定） | [official_locked_text_model_example](https://raw.githubusercontent.com/OpenBMB/OlympiadBench/ba5b26a7e2849940b598a9159c1190daa2b9175f/inference/code/evaluators/deepseek_evaluator.py)；Author DeepSeekMath text example and GPT4 text example use2048; not mandatory for every model. No universal context cap;16384 is project choice. |
| gsm_plus | 2048 | 16384（项目容量，官方未统一指定） | [official_local_runner_default](https://raw.githubusercontent.com/qtli/GSM-Plus/3474129ec12fcd3e8ac08cb037aca1928efca98c/scripts/general_model_inference.py)；Author local vLLM CLI default2048; project uses own zero-shot prompt/MathVerify, not exact few-shot paper reproduction. |
| ifeval | 4096 | 16384（项目容量，官方未统一指定） | [project_fallback_official_checker_only](https://github.com/google-research/google-research/tree/c878a53ec1764bcb91a6178dc5c315202a3b5fa2/instruction_following_eval)；Official checker consumes already-generated responses; no universal generation/context budget. |
| ifbench | 4096 | 16384（项目容量，官方未统一指定） | [project_fallback_official_model_dependent](https://github.com/allenai/IFBench/tree/488918665d53f867fa908a293769ec6e71e74aec)；Authors specify temp0 but vary generated-token cap by model;4096/16384 are explicit nonthinking project choices. |
| multi_if | 1024 | 16384（项目容量，官方未统一指定） | [official_locked_local_vllm_default](https://raw.githubusercontent.com/facebookresearch/Multi-IF/1cdb53ed18499ad729e0766e5d3099dd5344406f/multi_turn_instruct_following_eval_vllm.py)；1024 per turn, cumulative user+assistant history must fit context. Keep project greedy protocol; author example sampling differs. |

关键说明：

1. HumanEval+/MBPP+：锁定EvalPlus provider/base.py输出768；同提交provider/vllm.py的服务窗口2048。采用的是这个实际路径的默认值，不能把其他工具的4096说成EvalPlus统一官方默认。
2. BigCodeBench：锁定run_codegen输出1280、上下文12800。LiveCodeBench：锁定parser输出2000，但没有统一上下文值。
3. OlympiadBench：官方DeepSeekMath文本示例和GPT4文本示例均使用2048输出；是作者模型示例，不能冒称所有模型必须2048。
4. GSM-Plus：官方本地vLLM CLI默认2048。我们仍用既有零样本包装/MathVerify，非其few-shot论文完整复现。
5. Multi-IF：采用实际main函数传入的GenerationSetting **1024/轮**，不是utils类未被采用的4096默认，也不是thinking API示例25000。本项目仍按既有greedy单答案测评；官方示例temperature=.6/top_p=.9，与此不同。
6. MATH500/AIME25/AIME26是题目集合，没有针对所有LLM统一规定的上下文/输出预算；保留8192输出、16384上下文，明确记为项目预算。IFEval仅提供checker；IFBench作者明确输出上限随模型类型调整，因此其4096/16384也不能叫官方固定预算。

## 3. 运行中的约束和日志

服务端vLLM保留16384物理窗口，以复用同一路底座和adapter热切；各benchmark的**逻辑请求窗口**在生成端用同一真实tokenizer执行检查。因此EvalPlus仍遵守2048有效窗口、BCB遵守12800，不能借较大的物理服务窗口偷偷多用上下文。

每轮检查“已有全部user/assistant历史＋生成模板＋预留输出”不超过该benchmark有效窗口。Multi-IF保留前两轮真实回答后再检查第三轮。超限明确报错，不截断题目、不删历史、不丢测试题、不悄悄减少输出。记录实际输入/输出token、请求上限、有效窗口、本地prompt token、finish_reason和预算来源；改变预算使缓存身份失效。

以上预算只改变benchmark及其checkpoint生成；NICE原20题诊断仍为4096输出/16384上下文，NICE评分的随机HF rollout也保留4096/16384。训练max_length保持8192。三个epoch全部测benchmark的安排不变。

## 4. 验收与交接

CPU：9项预算/上下文/真实长样本准入检查通过，包括第三轮累积上下文超限、缓存身份失效、真实9896-token IF样本计入排除前统计。普通33项联通和完整22项双LR/NICE/恢复检查另行重跑；实际GPU容量、vLLM模板一致性与容器判分仍由B验收。

真实抽查还发现Numina缺solution会触发AttributeError；适配器已改为empty_supervision明确排除并计数。本轮没有增加teacher/API费用。

当前权威配置：[exp001.json](../configs/ordinary/exp001.json)。本轮改变了源码/评测协议，必须生成新plan和执行地点自身的验收receipt，不能改旧结果或复用旧预测混入新预算。

补充核实：TACO最长参考解含巨大的硬编码整数列表，最长一行616,978字符，完整输入617,279 tokens；不适合为此把普通训练上限扩到数十万。8192在本次已测TACO结构合格行上覆盖99.9204%，IF覆盖99.9967%。这是观测覆盖率，不代表重新执行测试后的最终池比例。

EvalPlus上下文另用真实Qwen模板检查固定HF镜像题面：HumanEval+164题，最大输入429 tokens；MBPP+378题，最大118 tokens；加768输出都可容纳于2048。正式增强测试release和最终题单仍由工作站封存，不以HF镜像替代增强测试。

最新CPU验收：33项流程smoke、26项配置负例、22项完整双LR＋NICE／恢复、9项预算与真实长度专项检查全部通过。无真实GPU推理、GPU训练或容器判分验收。

2026-11-21部署口径：A/B异地独立。A从零下载资产、构建镜像并本机验收；B仅提前测试反馈，B的目录、缓存、镜像或凭据不是A的运行依赖。A内工作站与集群之间可共享／迁移A自己的资产。
