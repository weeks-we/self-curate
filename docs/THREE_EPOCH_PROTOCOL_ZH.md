> 2026-11-21用户变更：保存全部3个epoch，只测epoch3；72组测评。以下旧逐epoch测评叙述由 [最新核查](BENCHMARK_OFFICIAL_CHECK_20261121_ZH.md) 覆盖。

# 实验一：统一3 epoch及逐checkpoint测评

用户最新确认，覆盖此前1 epoch、只测final的草案。

## 当前协议

- Full、Random5/10/20/30，以及后续用于主表公平比较的筛选子集，目标模型SFT统一3 epoch。每条分支从相同初始权重开始，不分别重训三个独立epoch分支。
- 使用HF Trainer原生save_strategy=epoch、save_only_model=false、save_total_limit=3，保存第1/2/3轮完整checkpoint。保留模型/adapter、optimizer、scheduler、RNG等原生训练恢复状态，不能只加载LoRA权重后假称续训。
- 每条训练分支完成3轮后，分别加载三份checkpoint运行其领域的所有benchmark，输出不覆盖。初始模型每个benchmark只测一次。
- 主结果预先指定epoch3，epoch1/2作为学习曲线一并报告。不以最终benchmark最大值挑checkpoint；后续改方案使用新的配置和run_id。
- 当前恢复保存粒度是epoch：轮内中断可从上一完整epoch checkpoint恢复，可能重做本轮；第一轮尚无checkpoint时需要重新开始该分支。更细步级保存未在本次配置启用。

三领域仍有18个实验分支，其中15条训练分支，共45份epoch checkpoint。按代码4、数学5、IF3个benchmark计算，结果数为 `(5条训练分支×3 checkpoint+1初始模型)×(4+5+3)=192`，而非此前84。

训练checkpoint位于各分支training/checkpoint-<global_step>，与epoch的映射写入training/complete.json的epoch_checkpoints。评测位于各分支epoch-01、epoch-02、epoch-03子目录。state.json按epoch与benchmark记录恢复状态，summary.csv包含checkpoint列。

学习率、LoRA、长度、batch等其他参数仍待最后确认；approved仍为false，不启动正式训练。

## 文献中的训练预算

| 工作 | 论文中的实际设置 | 对本项目的启发 |
|---|---|---|
| [LESS，ICML2024，附录A.2](https://arxiv.org/html/2402.04333v2) | 对选定数据集统一4 epoch的LoRA；曾尝试增加小子集epoch对齐训练步数，未获得提升，最后统一epoch | 固定epoch是合理协议，不必强迫小子集与全量跑相同步数；不要混淆其打分warmup与最终SFT |
| [AlpaGasus，ICLR2024，附录D](https://arxiv.org/html/2307.08701v3) | 7B的精选9k与全量52k均3 epoch；13B两组均5 epoch | 同一模型/比较内统一遍数，可以同时衡量数据与训练成本节约 |
| [DEITA，ICLR2024，附录A](https://proceedings.iclr.cc/paper_files/paper/2024/file/6091f2bb355e960600f62566ac0e2862-Paper-Conference.pdf) | 精选SFT数据训练6 epoch；其DPO为另一阶段，不能混为SFT epoch | 多epoch允许，但不能直接套用其全部模型/数据配方 |

## 主表与原论文复现

主表比较的是筛选数据的效果：固定同一目标模型、领域池、持出题单、比例、epoch、训练方式与共同超参。Full与子集的数据量自然不同，因此相同3 epoch并不代表相同训练step或token；同时报告唯一样本数、监督token、时间和GPU成本。

对比方法内部为筛选服务的warmup、scorer训练或搜索步骤，应保留该方法的定义并单列成本，不能一概改成3 epoch而改变算法。统一3 epoch约束的是选好数据之后的目标模型SFT。

如需要忠实复现原论文的4/6等epoch配方，可以单列附加实验，不将它和统一3 epoch的主表结果混合排序。我们动态多轮实验还需统一总选样预算与累计曝光量，不能只看每轮epoch相同。

## 验收与耗时

已完成CPU显式mock验收：18分支、192份checkpoint/benchmark结果、分目录隔离、在第二份checkpoint评测途中中断后恢复、完成后幂等重放；另检查缺失epoch adapter时拒绝发布。真实8卡FSDP2 checkpoint保存/恢复与vLLM装载尚需服务器验收。

estimate命令现将训练输入token乘以3，避免沿用1 epoch估计。相较之前1 epoch且只测final，训练量约3倍，测评组数由72增至192，约2.67倍。数学/代码答案长度及CPU判分仍影响实际耗时，先前1—4天的规划区间不能直接作为本协议的承诺，应重新按精确token和服务器吞吐估算。

最新成本表见 [统计协议](CHECKPOINT_ACCOUNTING_ZH.md)。Random50已取消，独立样本加权系数每epoch为1.65，3轮为4.95。
