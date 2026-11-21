> 最新2026-11-21：NICE已并入EXP001；A完整交接见docs/HANDOFF_A_EXP001_ZH.md，B测试交接见docs/HANDOFF_B_ENV_ACCEPTANCE_ZH.md。普通两LR草案33分支/132结果，NICE另加4分支/60结果；旧单LR/旧单独NICE目录文字由新交接覆盖。

> 2026-11-21：每个benchmark只测最终epoch3一次；仍保存3份训练checkpoint。EXP001共72组测评。最新执行见docs/HANDOFF.md与docs/BENCHMARK_OFFICIAL_CHECK_20261121_ZH.md。

> 2026-11-20最新确认：Random50取消；Initial/Full/Random5/10/20/30，统一3 epoch，18分支/15训练/45 checkpoint/192测评。当前执行请读 docs/HANDOFF.md 与 docs/CHECKPOINT_ACCOUNTING_ZH.md；此前相反数字为历史说明。

> Latest training/evaluation protocol: [3 epochs, save and evaluate every epoch checkpoint](THREE_EPOCH_PROTOCOL_ZH.md). 18 training branches, 54 epoch checkpoints, 228 benchmark results; all prior 1-epoch/final-only estimates are superseded. Remaining hyperparameters are unapproved.

> Latest TACO split policy: development tasks come from train only, 50 per skill; test is reserved exclusively for final evaluation. Historical test-as-development instructions below are superseded.

> Latest math source: NuminaMath-1.5, hierarchical source/problem_type proportional sample of 50,000. See [current sampling protocol](MATH50K_STRATIFIED_ZH.md). The earlier external20k/enrichment description below is superseded.

> Current update: see [EXP001 update](EXP001_UPDATE_20261120_ZH.md). It supersedes the earlier math pool, random ratios, and development split timing below. Random now includes 30% and 50% (21 branches / 18 SFT / 84 evaluations). Math uses the fixed 20k question pool but needs reference-solution enrichment. Development splits are published AFTER EXP001; they are not independent validation for already exposed EXP001 checkpoints. Hyperparameters remain unapproved.

# 普通任务实验：选型、阶段与边界

2026-11-20 重新定义。此前 SWE 交互轨迹实验的方案及交接不再作为执行入口。个人调研材料保留在被 Git 忽略的 `docs/meetings/`。

## 固定对象

| 对象 | 本轮选择 | 用法 |
|---|---|---|
| 主模型 | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 实验一、二与主要消融；关闭 thinking |
| 小模型 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 后续跨规模实验；关闭 thinking |
| 跨家族模型 | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 后续跨家族实验，工作站需先获 HF 访问权限 |
| Agent harness | [smolagents ToolCallingAgent](https://huggingface.co/docs/smolagents/reference/agents) | 支撑 Lexec、Leval、Lopt；共享模型引用，各自 prompt/history/tool 白名单 |
| 训练 | HF Trainer + Accelerate 原生 FSDP2 + PEFT LoRA | 不自行实现分布式优化器、参数分片或 RNG 恢复 |
| 普通评测 | vLLM 直接生成 + 对应官方判分 | 不通过 agent，不给模型工具或标准答案；代码判分单独 CPU 容器 |

模型、数据、判分仓库的当前 commit 已写入 `configs/ordinary/sources.json`。工作站 `lock` 命令生成不可变来源锁，下载时按 SHA 取文件，不使用浮动 `main`。本地未下载三个模型权重。

## 数据与标签

| 领域 | 第一数据集、原始规模 | 标签及准入 | 最终 benchmark |
|---|---|---|---|
| 代码 | [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO)，train 25,443、test 1,000 | train 一题一份固定参考解；原生 `skill_types`，保留 tags/difficulty；解析代码及输入输出测试结构，不重新执行所有答案 | LiveCodeBench release_v6、HumanEval+、MBPP+、BigCodeBench instruct |
| 数学 | [NuminaMath-1.5](https://huggingface.co/datasets/project-numina/NuminaMath-1.5)，896,215 | 原生 `problem_type`；排除来源标记无效、证明题、无可验证答案；不暗设 50k 上限 | MATH-500、AIME 2025、AIME 2026、OlympiadBench 英文纯文本数学、GSM-Plus |
| IF | [Tulu-3 Personas IF](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following)，29,980 | 原生 constraints 映射为词汇、格式、长度等能力组；保留原字段；不把标签名称冒充完整可执行 checker | IFEval、IFBench、Multi-IF 英文三轮 |

这些是来源规模，**不是准入后的训练规模**。训练分母是：去重、隔离最终测试和未来开发题、排除无效格式及超长样本后的唯一问题池。所有方法共用同一池。缺原生技能标签的样本仍可进入全量/随机基线，缺失状态保留；不能补成零或伪造标签。当前三池不需要付费 teacher 才能启动实验一。

最新用户决定：数学先抽一部分题作共同候选池，具体规模/抽法暂不实现，后续商讨；Full 是候选池全量，不是原始 Numina 全集。训练参数同样待最终确认。

TACO test 的 1,000 题从实验二起用于开发，不再同时充当最终 benchmark。每技能 50 题允许任务同时属于多组，但总表按唯一 ID 只执行一次；不足 50 显式报告，不能重复填充。数学/IF 后续开发题必须在实验一训练前持出并冻结；原生能力组是否合并、每组数量需确认，不能先训练后划开发集。

隔离实现了规范化题面完全匹配和去重；不能据此声称已排除 Numina 与数学 benchmark 的所有改写题污染。来源家族、近似题面与长 n-gram 复查仍属于正式数据验收，须保留清单。

## 四个实验独立管理

1. **EXP001：校准基线。** 仅 9B；每领域免训、全量、随机 5%/10%/20%。12 次独立 SFT，15 个评测分支；不调用三个 agent、NICE 或筛选轨迹学习。详见 `EXP001_ORDINARY_ZH.md`。
2. **EXP002：动态性的消融。** 仅 9B、代码领域；固定同一任务开发集。V0 固定初始指标与程序；V1 程序固定、更新下一轮关注技能方向上尚未用于真实参数更新的数据指标；V3 同时允许改程序。保留原有循环/反馈接口，具体 NICE/SFT 评估后端和每轮预算另行冻结。宏观安排不是已验收的真实实验。
3. **EXP003：控制学习与比较方法。** 三领域三模型；实现控制轨迹训练定义及选定 baseline；补齐另两模型的全量和指定随机比例对照。未确定的比较方法和轨迹损失不预填。
4. **EXP004：补齐完整框架主表。** 9B 的数学/IF，以及另两模型的三领域完整框架。用户提到的“Qwen3.5-8B”按已选定的 **9B** 处理。每个领域的正式筛选比例跨模型保持一致，不能按某个模型最终测试分数挑比例。单 agent 与筛选能力提升验证留待后续设计。

实验一的 5/10/20% 是预先声明的规模曲线，不能看测试成绩后挑一个作为唯一“公平”基线。后续域内预算在 `roadmap.json` 中仍为待确认。

## 模块入口

| 文件 | 职责 |
|---|---|
| `preparation/ordinary.py` | 三池字段适配、轻量准入、去重、原生标签、排除账本 |
| `preparation/ordinary_m1.py` | 普通样本接入既有 M1，保留稳定 ID/技能和任务关联 |
| `preparation/ordinary_tokenization.py` | 与 M1 相同的 assistant payload 掩码，关闭 thinking |
| `agents/smol_client.py`、`agents/ordinary.py` | 新 harness 与既有 RoleRuntime 的适配、工厂 |
| `configs/ordinary/prompts/` | 三角色固定业务 prompt，仍遵循原有 JSON 协议 |
| `training/ordinary.py` | FSDP2/LoRA、HF 原生断点、loss/token/时间记录 |
| `experiments/ordinary/workflow.py` | 不可变子集、15 分支、训练/评测状态与恢复 |
| `experiments/ordinary/sources.py`、`workstation.py` | 工作站联网下载、锁、资产校验、开发题持出、服务器绑定 |
| `experiments/ordinary/benchmarks.py`、`evaluation.py`、`grade.py` | 题单、直接生成、正式判分分离 |
| `experiments/ordinary/smoke.py` | 无 GPU 的可重复连通性验收，所有分数显式 mock |

已有 data/operators/signals/feedback/inner/outer 接口保留。历史 SWE 适配代码暂不暴力删除，避免破坏模块导入；它们没有出现在新入口里，也不代表普通任务新实验已经具备真实 NICE 验收。

## 已核实的适配差异

- LiveCodeBench `release_v6` 是累计六批 JSONL，不是只取 `test6.jsonl`。
- EvalPlus 的完整增强测试从固定版本 EvalPlus 官方缓存准备，保存实际 IDs/hash。HF 同名简化镜像的字段或题数不能代替正式 release。
- OlympiadBench 使用自己的 MathJudger 与独立 ANTLR 环境；其他数学题使用 math-verify。IF 三个集合各用其官方 checker，不调用付费裁判。
- Multi-IF 保留真实的三轮模型回答，不能用数据中的参考回答替代上文。
- Transformers 5.23 的 FSDP 参数与旧版不同：本实现使用 `fsdp=True`、`version=2` 和新的 warmup 参数。代码检查不等于 H200 容量/恢复验收。

本地验收与服务器验收分开记录，入口见 `HANDOFF.md`。
