# 实验一附加：普通代码任务NICE验证

> 本轮生效配置与研究依据见[训练和环境总表](EXP001_QWEN35_9B_RECIPE_AND_ENV_ZH.md)。LR已选5e-5/1e-4；两路各4卡，主实验global32、NICE global8；异步CPU判分。approved=false仅是服务器验收门。后文历史“草案/待用户check”不再作为科研参数待办。


2026-11-21用户授权；现已在self-curate内实现，不依赖相邻NICE项目的Python包、环境或运行结果。模型/原始数据/benchmark/镜像沿用普通任务实验一。

## 固定研究流程

1. Qwen3.5-9B，关闭thinking，从共同初始参数开始。先对轻量准入、去重、最终benchmark隔离后的TACO **train** 池固定随机抽20题，seed42。保存任务题面、原生tests、参考解、ID和来源hash，候选端排除同ID/题面/显式URL家族。
2. 从余下训练池随机抽3,000条，固定一个参考解/任务；20题及其任何参考解都不得入候选。缺少3,020个可隔离任务就报错，不复制凑数。
3. 原20题每题4次独立rollout，共80次。使用本地HF generate，temperature1/top_p1/top_k0/repetition_penalty1、输出最多4096、context16384；每次保留实际prompt/generated token IDs和完整生成文本、tokenizer/template/model/initial-adapter身份。没有外部API费用。
4. 生成代码在已有CPU沙箱中由锁定TACO官方testing_util判分，全部tests通过奖励1，否则0。正式前先执行20题固定参考解自检；失败即停止，不按目标模型结果换题。沿用官方测试比较，不用LiveCodeBench判分器替代TACO，也不声称额外实现了所有特殊裁判。
5. 只对初始LoRA求梯度。每题验证方向h_v=(1/4)Σ r_vm·∇NLL(generated tokens)，NLL取生成token之和；失败的0奖励保留在固定分母4。候选参考答案的CE按有效assistant token平均。题面/模板/padding不监督，梯度阶段不optimizer.step。
6. 全局裁剪后fresh-Adam候选方向，再做共享CountSketch8192维；每题验证方向独立归一化，在全部20题上平均余弦，零方向保留为0，不能再归一化20题方向的平均。记录原始梯度余弦诊断、投影/坐标身份、梯度范数与零题数。这是旧项目的单初始checkpoint、fresh-Adam NICE风格变体，不宣称复现原论文所有warmup/多checkpoint设置。
7. 按分数及稳定ID排序，最低1,000和最高1,000直接入Low/High；中间1,000不训练。不是各半池再随机抽。组内顺序固定，两个LR复用完全相同成员/顺序。
8. High/Low × 两个共享学习率，共4条独立LoRA分支，均从共同初始adapter开始，训练1 epoch。保存约1/3、2/3、最终三个节点，并对**每个节点**测原20题和四个coding benchmark。普通Full/Random仍测epoch1/2/3；NICE是用户明确指定的三节点例外。

## 训练和测评配置

主配置configs/ordinary/exp001.json；NICE定义configs/ordinary/exp001_nice.json。

| 项目 | NICE配置 |
|---|---|
| 学习率 | 唯一读取exp001.training.learning_rates，已选定5e-5/1e-4；不能在两处维护不同数组 |
| LoRA | 参考旧NICE的r16/alpha32/dropout0；使用现有Qwen3.5文本投影白名单以适配混合注意力，排除视觉层 |
| Batch | 参考旧NICE有效batch8：4卡×micro2×累积1；主实验仍global32 |
| 优化器／损失 | 原生HF Trainer/Accelerate FSDP2/PEFT、fused AdamW、epsilon1e-8、clip1、weight_decay0、warmup3%；NICE分支为逐样本答案token平均、再batch平均，与候选打分损失一致 |
| 长度／精度 | 完整SFT输入8192、BF16、梯度checkpointing，超长不截断 |
| 保存 | 原生optimizer/scheduler/RNG＋成本receipt；1,000条／batch8共125步，取ceil节点42、84、125，即336、672、1,000次实际曝光 |
| 推理 | 原20题及coding benchmark为greedy单答案，输出4096/context16384、thinking关闭；两路各4个单卡vLLM副本，每副本并发64起测 |

LoRA rank/batch/epoch与主Full/Random不同，因此NICE是独立的高低价值验证，不拿它的短训分数当统一训练预算的主方法对比。20题参与构造NICE方向，其节点得分只作诊断，不是独立泛化。所有三节点保留，不能按最终benchmark选最好节点。

NICE共12份训练checkpoint、12组20题诊断＋48组coding benchmark＝60组结果。正式两档LR下，常规实验一为33分支（30次SFT＋3个Initial）、90训练checkpoint、372组结果；加NICE共34次SFT、102训练checkpoint、432组统计，其中420组为benchmark，12组为原20题诊断。80次随机rollout的奖励不能冒充greedy基线成绩。

## 模块和入口

| 文件 | 职责 |
|---|---|
| experiments/ordinary/nice_experiment.py | 冻结20/3000任务、分片调度、排名选样、4分支控制、恢复和报告 |
| experiments/ordinary/nice_worker.py | 八卡独立分片的真实HF rollout/LoRA backward，实际token回放、投影与权重不变检查 |
| training/lora.py | 与普通SFT共享LoRA构建，保证模型层/目标名/初始化约定一致 |
| training/ordinary.py、epoch_checkpoints.py | 复用原生FSDP2训练；增加sample-mean loss及42/84/125原生保存，保留旧按epoch模式 |
| experiments/ordinary/taco_grade.py、nice_sandbox.py | TACO官方测试适配、Python新版inspect兼容、进程墙钟保护、复用已有CPU容器与vLLM生成 |
| experiments/ordinary/workflow.py | 同一两档LR生成常规分支；自动生成并执行NICE子计划，不另建项目 |

以上src/目录均在self_curate下。配置先由用户确认，再冻结新计划；不得修改旧run身份来恢复。

单独准备和仅梯度验收：

```bash
python -m self_curate.experiments.ordinary.nice_experiment prepare --config configs/ordinary/exp001.json --pool /shared/assets/coding-pool --output /shared/runs/B-nice-plan
python -m self_curate.experiments.ordinary.nice_experiment gradient-smoke --plan /shared/runs/B-nice-plan/plan.json --bindings /shared/server-bindings.json --output /shared/runs/B-nice-gradient
```

B使用完整合格coding池准备20/3000的冻结清单，不用128条短训池。实际只对2道任务各生成1次最多64token，单位权重检查sum-NLL backward，并对8条候选（优先覆盖最长样本）检查mean-CE backward；不判分、不排名、不SFT、不跑NICE benchmark。单位权重不是模型成功奖励，不发布科学分数。全部成功后命令自动把receipt路径和SHA写入server-bindings，A正式运行会验证同模型/池/LoRA/打分方法/源码/资产身份。

正式全流程通过普通EXP001的plan/run自动进入NICE；也支持nice_experiment run单独执行，但**不要独立跑完后再另起一份重复任务**。A统一入口见HANDOFF_A_EXP001_ZH.md。B检查不能代替A正式80次rollout、TACO参考解判分及最长生成回放容量验收。

## 输出和恢复

正式运行的results/<run_id>/nice/下有frozen-plan、state、rollouts/gradients各rank日志及投影索引、selection/scores.json和high/low.jsonl、branches/<group-lr>/training/checkpoint-STEP、progress-1/2/3的20题/benchmark结果、summary.csv、training_costs.csv、REPORT.md和complete.json。

每个训练节点记录实际输入/监督token、曝光、时间、GPU时长及增量；每个测评记录输入/输出token与生成/判分时间。梯度记录候选mean-CE、输入/监督token、范数、原始/Adam投影、时间和坐标身份；完整梯度不写入结果zip。累计训练成本不能在3节点或4个benchmark上重复相加。

--resume只复用身份和checksum一致的记录；完整checkpoint才恢复optimizer/scheduler/8rank RNG，半写checkpoint另存incomplete目录，已发布checkpoint缺状态则拒绝续训。所有20个目标方向为0时停止，不凭任意排序声称NICE有效；出现非有限值、来源/坐标/adapter变化也停止。

代码/模型/模板/方法变化需新plan；已有B凭据会因身份不符失效。CPU mock/小模型数值验证与真实9B、GPU容量、官方容器判分严格区分。

执行顺序：环境验收后，统一run先执行NICE，再跑常规基线；避免TACO新适配或零方向问题到大规模Full训练完成后才暴露。各分支起点独立，执行顺序不改变比较定义。


## 内置NICE本轮验收（2026-11-21）

22项完整CPU显式mock集成检查通过：共享两档LR、20/3000隔离、高低各1000、4分支/12节点/60评测、B只反向、普通372+NICE60、90+12训练成本行、断点/幂等与选样篡改拒绝。保留的普通基线/三角色隔离33项回归通过。

真实CPU小模型完成反向/有限差分、sum-vs-mean、流式投影对照已有NICE数学实现、base/LoRA权重不变、19零题仍计入20分母、sample-mean Trainer扩展与42/84/125保存回调。恢复/凭据用明确的文件fixture验证，非真实FSDP恢复。

发现并修复Windows短时文件占用造成原子替换失败：只对指定Win错误限次重试，故障注入验证旧状态不损坏，随后全流程重跑通过。结果zip含普通与NICE的报告/表格并通过CRC（247文件）。凭据见忽略的runs/integrated-nice-acceptance-20261121。

本机没有下载目标权重或运行GPU/Docker/真实TACO代码判分。B负责真实9B梯度和既有环境验收，A负责正式TACO参考解自检、80rollout与全量梯度、FSDP保存/恢复和所有节点真实测评；不能拿本地mock成绩作研究结论。

2026-11-21最新确认：EXP001普通基线epoch1/2/3全部跑benchmark；Initial仍只测一次。epoch3为主要报告，不按benchmark挑权重。两LR正式矩阵372组benchmark；NICE保持48组benchmark＋12组原题诊断，总432组测评记录（420 benchmark＋12诊断）。师兄B仍保持独立短测：每个benchmark只测一次，不扩大验收规模。此前相反节点说明以本条为准。
