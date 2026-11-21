# 师兄A：实验一正式任务交接

> 最新长度与逐benchmark预算见[长度/预算核查](TRAINING_LENGTH_AND_BENCHMARK_BUDGETS_ZH.md)。原文统一4096/8192输出仅为旧设置；当前执行exp001.evaluation.benchmark_budgets。三个epoch测评及A/B范围遵循HANDOFF。


> 本轮生效配置与研究依据见[训练和环境总表](EXP001_QWEN35_9B_RECIPE_AND_ENV_ZH.md)。LR已选5e-5/1e-4；两路各4卡，主实验global32、NICE global8；异步CPU判分。approved=false仅是服务器验收门。后文历史“草案/待用户check”不再作为科研参数待办。


本文件给A执行完整实验；B仅按HANDOFF_B_ENV_ACCEPTANCE_ZH.md做环境/短训与NICE梯度验收。唯一主配置configs/ordinary/exp001.json；NICE附加配置configs/ordinary/exp001_nice.json，详细定义见EXP001_NICE_ZH.md。不再进入相邻NICE目录运行。

## 开始前

复用B核验过的固定三个模型、三领域数据、数学50k分层抽样、12个benchmark和官方checker/镜像。工作站联网准备，集群离线计算；真实8个GPU UUID、共享挂载、镜像digest及server-bindings需核实。B的server_smoke_receipt与nice_gradient_smoke_receipt都必须有文件、checksum和匹配身份，不能只手填passed。

本轮依据研究经验选定的两档LR为5e-5/1e-4；main.training.learning_rates是唯一列表，learning_rate标量保留给单任务/B短训默认。其余常规LoRA为r32/alpha64、global32、3epoch；NICE参考旧设置r16/alpha32、global8、1epoch。服务器验收通过前approved=false。确认后先改配置再生成新plan，不修改冻结计划。

## 正式执行

1. 三领域池均完成轻量准入、最终测试隔离与真实tokenizer长度。数学原集按source→problem_type抽50,000原始题后准入；证明题允许SFT但不作自动判分开发题。TACO只从train取训练与开发任务。
2. 常规基线：Qwen3.5-9B，代码/数学/IF各Initial、Full、Random5/10/20/30%。Full/Random两个LR独立训练3epoch，测epoch1/2/3，epoch3为主报告；仍保留三个epoch权重和成本。随机子集嵌套，两个LR的同组成员/顺序一致。
3. 自动执行NICE子计划：20个train任务、每题4rollout；剩余随机3,000候选算梯度，最低/最高各1,000，两个共享LR各跑1epoch，四分支保存42/84/125步，并分别测原20题及四个coding benchmark。NICE全部三节点测评是本附加实验特例，不能改常规基线的all-three-epochs规则。
4. 实验末尾完成三领域技能标签/分布和每技能50题的未来开发集发布。它们可能已进入EXP001 Full，不能当EXP001独立开发验证；后续实验从初始模型重新训练并隔离这些题。

```bash
python -m self_curate.experiments.ordinary plan --config configs/ordinary/exp001.json --pools /shared/pools.json --output /shared/runs/exp001-plan
python -m self_curate.experiments.ordinary run --plan /shared/runs/exp001-plan/plan.json --bindings /shared/server-bindings.json
# 中断后：源码、配置、数据与输出身份不变
python -m self_curate.experiments.ordinary run --plan /shared/runs/exp001-plan/plan.json --bindings /shared/server-bindings.json --resume
```

pools.json是coding/math/if三个完整prepared-pool目录的映射。不要传B的128条短训文件；不要用--mock或--synthetic完成正式任务。当前run先执行NICE，再执行常规基线；NICE只依赖已准备环境，不依赖Full训练。这样可在大规模基线消耗GPU前发现TACO参考判分失败或全部零方向；运行前即检查NICE的B验收凭据。只执行上述统一入口，不再单独重复启动NICE。

按当前两LR配方：常规33分支/30次SFT/90 checkpoint/372组测评，NICE4次SFT/12 checkpoint/60组统计。合计34次SFT、102训练checkpoint、432组统计（420 benchmark＋12原20题诊断）。所有分支从相同起点独立训练，不能串联各LR或高低组的参数。

## 交付

运行根results/<UTC时间-ID>/下保存冻结计划、state、普通summary/training_costs/scores_wide/REPORT；nice/下保存附加实验分组、梯度/rollout元数据、四分支各节点结果及成本表。根complete.json同时有普通结果数和nice完成状态；任一必需阶段失败就不宣称全部完成。

```bash
python -m self_curate.experiments.ordinary bundle --run /shared/runs/exp001-plan/results/ACTUAL_RUN_ID --output /shared/exp001-results.zip
```

包内含配置、分组/题单引用、逐题结果、表格、日志与checksum；权重、完整原始数据和大梯度留服务器，返回路径/checksum，不打进小结果包。请附A_REPORT.md列出每项完成/失败/未执行、GPU吞吐/显存、时间/token/曝光量和所有基础设施修复。

TACO官方checker的20题参考解自检、真实9B最大输入反向、FSDP2保存/恢复与所有容器仍需服务器实际执行。尤其不能把B单位权重梯度probe解释为20题有成功奖励，或把原20题分数解释为独立泛化提升。

2026-11-21最新确认：EXP001普通基线epoch1/2/3全部跑benchmark；Initial仍只测一次。epoch3为主要报告，不按benchmark挑权重。两LR正式矩阵372组benchmark；NICE保持48组benchmark＋12组原题诊断，总432组测评记录（420 benchmark＋12诊断）。师兄B仍保持独立短测：每个benchmark只测一次，不扩大验收规模。此前相反节点说明以本条为准。

2026-11-21用户确认标签：保留8/7/10类；未映射样本先不标、保留训练池、不自动调用teacher；按技能开发集暂不使用未映射记录，缺额如实报告。原生对应关系见[NATIVE_SKILL_MAPPING_ZH.md](NATIVE_SKILL_MAPPING_ZH.md)。
