# 普通任务实验一：服务器交接

先读 [实验设计](ORDINARY_EXPERIMENTS_ZH.md)、[数学50k抽样](MATH50K_STRATIFIED_ZH.md)、[训练协议](THREE_EPOCH_PROTOCOL_ZH.md)、[资源统计与汇总](CHECKPOINT_ACCOUNTING_ZH.md)、[判分环境](BENCHMARK_ENV_AUDIT_ZH.md)。唯一配置 configs/ordinary/exp001.json，来源锁 configs/ordinary/sources.json；训练参数未全面批准，approved=false，不自动启动真实计算。

## 准备与冻结

工作站准备固定revision的Qwen3.5-9B、TACO train、原始NuminaMath-1.5、Tulu-3 Personas IF及12个benchmark和官方checker。关闭thinking。集群离线，不在GPU作业中下载环境。TACO test只作最终评测保留；所有开发任务从train抽。

数学先从原集按source比例、再按每个source内部problem_type比例抽原始50,000题，seed42，整数余数修正保证总数；然后轻量准入，最终合格数可少于50k。保留原始解答和字段；证明题允许SFT，禁止自动判分开发集。

```bash
python -m self_curate.experiments.ordinary sample-math --raw raw/numina.jsonl --count 50000 --seed 42 --output assets/math50k
```

按CLI prepare生成三领域池：真实tokenizer、完整长度8192草案、超长整条排除，不截断。answer_tokens用于答案长度标签；length为题面+答案+模板的完整输入。来源、内容、模板、tokenizer都须冻结，标签按ID连接。

## 执行矩阵

三领域各Initial、Full、Random5/10/20/30%，随机子集嵌套seed42，从相同初始模型独立训练。共18分支、15次SFT。Full与Random均3 epoch，原生HF/Accelerate FSDP2/PEFT，保存全部3个epoch checkpoint，共45份。每分支训练完只测epoch3对应benchmark，共72组结果；epoch1/2仍保存并记训练成本，不测benchmark。epoch3是主表，禁止按最终benchmark挑最好节点。Random50已取消。

```bash
python -m self_curate.experiments.ordinary plan --pools pools.json --output runs/new-plan
python -m self_curate.experiments.ordinary estimate --plan runs/new-plan/plan.json
python -m self_curate.experiments.ordinary run --plan runs/new-plan/plan.json --bindings bindings.json
# 中断后，保持同一来源/代码/配置和输出身份
python -m self_curate.experiments.ordinary run --plan runs/new-plan/plan.json --bindings bindings.json --resume
```

CPU联调：python -m self_curate.experiments.ordinary smoke --output runs/FRESH_DIRECTORY 。mock不产生真实模型成绩或真实GPU成本。

正式运行先确认学习率/LoRA/batch/长度、8个GPU UUID、环境seal和服务器smoke。现配置8×H200、FSDP2，测评8个单卡vLLM副本；实际吞吐/显存由服务器测。Docker CPU判分需平台支持；BigCodeBench使用独立固定digest镜像。不能把接口检查当作真实判分验收。

## 结果与恢复

run_id为UTC时间+随机ID，results/<run_id>/下保留state.json、frozen-plan.json、complete.json、REPORT.md、summary.csv、training_costs.csv、scores_wide.csv。每分支training/checkpoint-STEP内training-cost.json绑定实际曝光/token/时间。epoch-03下按benchmark保存预测、判分、evaluation-cost.json。

恢复用原生optimizer/scheduler/RNG checkpoint；保存粒度为epoch，轮内中断可能重做本轮，第一轮无checkpoint需重跑。跨中断累计checkpoint统计不包含尚未落盘的丢失工作；rank日志保留各session的消耗证据，不能把遗漏费用填零。

```bash
python -m self_curate.experiments.ordinary bundle --run PATH_TO_RUN --output results.zip
```

打包包含表格、配置、逐题结果和日志，不包含training目录的大权重；汇总表中已带checkpoint成本。训练权重和完整训练日志留服务器，按需另传。

## 标签和后续开发集

实验一末尾完成三个领域标签，输出distribution、development、future-training及manifest。每技能从train抽50题，组间可重叠但相同ID只执行一次；不足不能复制凑数。数学证明题留在训练端。标签缺失先补齐，不伪造能力。

```bash
python -m self_curate.experiments.ordinary apply-labels --pool math-pool --annotations math-labels.jsonl --output math-labeled-pool
python -m self_curate.experiments.ordinary finalize-labels --pool math-labeled-pool --completed-run RUN --output assets/math-split
python -m self_curate.experiments.ordinary finalize-labels --pool if-pool --completed-run RUN --output assets/if-split
python -m self_curate.experiments.ordinary finalize-labels --pool taco-train-pool --completed-run RUN --output assets/coding-split --coding
```

这些开发题可能已进入EXP001 Full训练，不能作为EXP001独立泛化分数；后续实验从原始基座重新开始、排除开发题。complete.json只表示训练/评测完成，不能代替标签阶段验收。

最新官方配置与GPU核查：[2026-11-21复核](BENCHMARK_OFFICIAL_CHECK_20261121_ZH.md)。
