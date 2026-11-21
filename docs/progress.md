> Latest training/evaluation protocol: [3 epochs, save and evaluate every epoch checkpoint](THREE_EPOCH_PROTOCOL_ZH.md). 18 training branches, 54 epoch checkpoints, 228 benchmark results; all prior 1-epoch/final-only estimates are superseded. Remaining hyperparameters are unapproved.

> Current proof policy: valid reference proofs are retained for SFT; proof examples carry evaluation_eligible=false and are excluded from automatically graded development tasks. Earlier blanket proof-exclusion statements below are superseded. Full input length is prompt + answer + chat-template tokens, not answer tokens alone.

> Latest TACO split policy: development tasks come from train only, 50 per skill; test is reserved exclusively for final evaluation. Historical test-as-development instructions below are superseded.

> Latest math source: NuminaMath-1.5, hierarchical source/problem_type proportional sample of 50,000. See [current sampling protocol](MATH50K_STRATIFIED_ZH.md). The earlier external20k/enrichment description below is superseded.

> Current update: see [EXP001 update](EXP001_UPDATE_20261120_ZH.md). It supersedes the earlier math pool, random ratios, and development split timing below. Random now includes 30% and 50% (21 branches / 18 SFT / 84 evaluations). Math uses the fixed 20k question pool but needs reference-solution enrichment. Development splits are published AFTER EXP001; they are not independent validation for already exposed EXP001 checkpoints. Hyperparameters remain unapproved.

# 状态：普通任务迁移

## 本轮实现

- smolagents ToolCallingAgent 适配既有共享模型调用协议，三角色独立工具/历史/日志；固定业务 prompt 和工厂接口。
- 三领域原生数据/标签适配、静态准入、重复及题面隔离、超长排除账本、M1 适配。
- 固定 27 项官方来源 revision，工作站下载/导出、benchmark 清单、完整 EvalPlus 缓存、资产 seal 与服务器绑定。
- 普通任务 FSDP2/LoRA 后端、HF checkpoint 恢复、input/supervised token、loss、时间和显存日志。
- EXP001 15 分支的不可变选择、显式 mock/real、状态恢复与逐题/逐轮生成缓存；直接生成与官方判分分开。
- 新 Docker 分阶段构建及唯一 HANDOFF；旧 SWE 实验文档/交接退出。

## 本地验收边界

真实 TACO、Numina、Tulu 少量公开记录已取得并检查，数据不提交 Git。临时 CPU 检查验证原生标签到 M1、工具调用、费用统计、随机嵌套子集、中断恢复、重复启动与技能覆盖缺口。smoke 命令可在师兄机器重复执行，使用合成模型回复，不消耗 API。

已执行：34 项临时 CPU 检查；新 ordinary smoke 的 10 项验收；原有三外层/每层五内层 mock 回归（含独立进程恢复和 zip 校验）。另使用固定 revision 的真实 Qwen3.5-9B tokenizer，验证代码/数学/IF 各一条真实样本：总 token 1130/403/536，监督 token 390/302/486，答案正文掩码完全一致，thinking 为空。没有下载模型权重。

**尚不能宣称正式实验可直接放行：** 没有目标权重/GPU/Docker 正式运行；8 卡 FSDP2 保存/恢复、vLLM 线性注意力 LoRA、12 grader 容器及镜像依赖解算须实际验收。正式配置 approved=false。

## 科研/平台待确认

学习率与 batch 等训练参数；数学/IF 开发题的分组和持出数量；TACO 每技能 50 题的真实覆盖是否充分；题面改写污染复查；集群可否调用 Docker CPU grader。原生标签不代表 teacher 已重标，也不代表完整质量重执行。

用户后续明确数学先抽一部分：抽样规模/方法待商讨实现，不启动 Numina 全池训练。训练超参也未确认。

EXP002—004 目前为分开的宏观路线；真实动态实验、控制轨迹学习科研定义、baseline 选择及完整主表仍需按阶段实现和验收。


Latest implementation: 21-branch smoke passed after adding Random30/50. Fixed 20k source schema checked with two real records. Exact reference join, prompt-only SFT rejection, real-tokenizer answer length fields, and synthetic seven-skill 50-per-skill split/disjoint remainder passed CPU checks. No full real pool label distributions, paid teacher run, GPU training or container build performed.


Benchmark audit: see BENCHMARK_ENV_AUDIT_ZH.md. Fixed EvalPlus read import, full LaTeX gold parsing, IFBench read-only import path, preloaded IF resources, and separated BigCodeBench image binding. Docker is unavailable locally; full container acceptance remains pending. Twelve sample loading routes passed, four math gold samples and the official Olympiad judge passed CPU checks; IF keyword positive/negative tests passed but full resource/type coverage is not a completed container acceptance.


Proof-policy CPU acceptance: seven synthetic math skills, 60 evaluable + 5 proof examples per skill. All 350 development examples are non-proof; all 35 proofs remain in the future training pool. Real Qwen tokenizer character/token previews: coding n=1 ratio3.16; math n=10 ratio2.53; IF n=3 ratio5.41. These are small-sample estimates, not corpus statistics.

2026-11-20：取消Random50，统一3epoch；新增原生checkpoint训练曝光/有效输入token/监督token/墙钟计量，8rank求和并绑定恢复。测评单列生成与CPU判分计量；自动输出summary/training_costs/scores_wide/REPORT。NICE重做指导书是忽略的个人交接，使用用户确认Qwen3.5-9B，不修改邻居NICE代码。

本轮CPU验收：33项全流程smoke通过，含18分支/192结果/45成本行、恢复与幂等；生产计数器使用真实CPU tensor检查监督mask、模拟8rank归约、恢复offset，另核对45行epoch增量/48行宽表。结果zip含表格并通过CRC。验收凭据位于忽略的runs/checkpoint-accounting-acceptance-20261120。真实FSDP2/9B/vLLM/判分容器未在本机验证。

2026-11-21：benchmark只测最终epoch3，保存3份训练成本，EXP001为72结果，B短训12结果。官方审计修正IFBench loose主分、Multi-IF逐轮四指标均值、BCB调用完整官方evaluate并使用冻结本地题单。训练保留global32、启用fused AdamW/8对齐/固定global吞吐预检与FSDP2少重分片；真实GPU与容器待服务器验证。

本轮验收：12/12 HF固定revision/文件在线存在；CPU全流程33项通过（72测评、45训练成本行、断点与幂等）；IFBench/IFEval/Multi-IF指标数值fixture及BCB完整evaluate接入参数、离线注入、异常后恢复的mock检查通过。BCB真实执行、官方容器构建、GPU吞吐与FSDP2恢复仍待服务器。凭据：runs/benchmark-official-audit-20261121。

2026-11-21新增用户授权：NICE并入EXP001，20×4/3000/top-bottom1000、两档共享LR、1epoch、42/84/125保存。参考旧NICE r16/alpha32/batch8；通用SFT仍r32/alpha64/global32/3epoch。新增真实HF回放/独立GPU梯度、官方TACO沙箱适配、四分支全链路及B梯度probe凭据。A执行完整实验，B仅环境/短训和NICE梯度检查。主LR草案1e-5/2e-5统一配置，未批准。


## 内置NICE本轮验收（2026-11-21）

22项完整CPU显式mock集成检查通过：共享两档LR、20/3000隔离、高低各1000、4分支/12节点/60评测、B只反向、普通132+NICE60、90+12训练成本行、断点/幂等与选样篡改拒绝。保留的普通基线/三角色隔离33项回归通过。

真实CPU小模型完成反向/有限差分、sum-vs-mean、流式投影对照已有NICE数学实现、base/LoRA权重不变、19零题仍计入20分母、sample-mean Trainer扩展与42/84/125保存回调。恢复/凭据用明确的文件fixture验证，非真实FSDP恢复。

发现并修复Windows短时文件占用造成原子替换失败：只对指定Win错误限次重试，故障注入验证旧状态不损坏，随后全流程重跑通过。结果zip含普通与NICE的报告/表格并通过CRC（247文件）。凭据见忽略的runs/integrated-nice-acceptance-20261121。

本机没有下载目标权重或运行GPU/Docker/真实TACO代码判分。B负责真实9B梯度和既有环境验收，A负责正式TACO参考解自检、80rollout与全量梯度、FSDP保存/恢复和所有节点真实测评；不能拿本地mock成绩作研究结论。

2026-11-21最终配置核查：[当前配置与待填项](CONFIGURATION_AUDIT_20261121_ZH.md)。修正roadmap Random30/NICE、NICE进度协议元数据和配置一致性校验；当前仍为未批准草案，真实容量/镜像/凭据由B/A验收。

本轮最终验收：26项配置一致性/非法覆写检查、22项NICE整合集成、33项原基线回归通过；CPU数值和恢复凭据校验复查通过。配置快照和凭据位于忽略的runs/final-configuration-audit-20261121。仍未在本机执行真实9B/GPU/Docker；科研超参整体保持draft/approved=false。当前launcher为单机8卡，多节点部署需单独适配。


2026-11-21：用户授权按经验选择EXP001/NICE专用配方：LR5e-5/1e-4，两路各4卡FSDP2；global32/8不变；异步CPU判分、LoRA热切与owned进程取消。三个base镜像锁linux/amd64 digest。CPU33项smoke、22项全矩阵/恢复、26项配置检查和真实微型CPU反向通过；真实GPU/Docker仍需B验收。详细依据和全参数见EXP001_QWEN35_9B_RECIPE_AND_ENV_ZH.md。

2026-11-21用户更新：EXP001三个epoch checkpoint均测benchmark，epoch3主报告；不改变训练超参或NICE节点。常规372＋NICE60＝432测评记录。

2026-11-21长度与预算核查：真实Qwen tokenizer测TACO/IF全池与Numina2000窗口样本；8192先保留，prepare新增长度排除前精确分布。12个benchmark分项预算，区分官方runner默认、模型示例和项目fallback；真实生成检查每轮全历史窗口。缺Numina监督文本明确排除。见TRAINING_LENGTH_AND_BENCHMARK_BUDGETS_ZH.md。

2026-11-21再次配置核对：18项必需HF来源有URL/revision。修复finalize对三epoch矩阵的数量适配；修复原生IF约束空格／别名、复合标签与未知标签显式复核。IF29980全量映射九类、无缺标；math probability／IF style及少量数学Other补标仍需口径确定，尚未修改taxonomy提案。见EXP001_READINESS_CHECKLIST_ZH.md。

2026-11-21用户最终标签口径：保留预设Coding8／数学7／IF10，未映射先不补标。配置native_only、teacher禁用；finalize把未映射排除出按技能开发集并保留训练池，零覆盖技能保留、50题不足仍显式报告。9项专项CPU检查通过；对应关系见NATIVE_SKILL_MAPPING_ZH.md。
