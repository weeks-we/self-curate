> Latest training/evaluation protocol: [3 epochs, save and evaluate every epoch checkpoint](THREE_EPOCH_PROTOCOL_ZH.md). 18 training branches, 54 epoch checkpoints, 228 benchmark results; all prior 1-epoch/final-only estimates are superseded. Remaining hyperparameters are unapproved.

> Current proof policy: valid reference proofs are retained for SFT; proof examples carry evaluation_eligible=false and are excluded from automatically graded development tasks. Earlier blanket proof-exclusion statements below are superseded. Full input length is prompt + answer + chat-template tokens, not answer tokens alone.

# 数学候选池：NuminaMath-1.5 两层比例抽样50k

本方案取代之前的外部20k题单、参考解补齐和待定抽样方案。直接从 [project-numina/NuminaMath-1.5](https://huggingface.co/datasets/project-numina/NuminaMath-1.5) 的固定revision `1b05109f9e5c1ad06c0663519502416c30b300f8` 读取原始训练数据，保留problem、solution、answer、source、problem_type及其他原字段。

## 抽样定义

1. 对原集计数，source为第一层、同一source内部problem_type为第二层。null值保留为独立分层，缺少字段则报错，不悄悄丢样本。
2. source配额理想值为 `50000 × N_source / N_total`。
3. source内部配额理想值为 `Q_source × N_source,type / N_source`。
4. 使用最大余数法：先向下取整，将剩余名额依次给小数部分最大的组；相同余数用固定字段顺序打破平局。两层均校正，所以总数恰好50,000，每个source内子配额之和也严格等于source配额；整数与理想配额的误差小于1条。
5. 每个子层按seed=42的哈希优先级无放回抽样。相同原始文件字节、seed、数量生成相同成员。按原始行序发布，所有Random分支之后再用实验seed统一打乱。

抽样单位是原始数据行，不是解答展开后的多份样本。这里保证原集分布，不按难度均衡，不强制每技能相同数量，不让benchmark分数决定抽取成员。

## 工作站执行

先按sources.lock固定来源，使用已有download/export接口生成原始JSONL及`.source.json`。本轮仅实现与CPU验收，不完整下载原集。

```bash
python -m self_curate.experiments.ordinary export --lock sources.lock.json --bindings source-bindings.json --key numina --output raw/numina.jsonl
python -m self_curate.experiments.ordinary sample-math --raw raw/numina.jsonl --count 50000 --seed 42 --output assets/math50k
```

输出：

- sample.jsonl：完整原始记录，恰好50,000行。
- sample.jsonl.source.json：兼容既有prepare接口，绑定原集SHA、抽样文件SHA、seed和sampling_hash。
- manifest.json：原始计数、source配额、联合层配额、原始/抽样比例、被抽行索引和内容hash。

然后对sample.jsonl调用原有 `prepare --domain math`，使用固定tokenizer和benchmark隔离输入。**先按原集比例抽5万，再做既有准入**；证明题/无可验证答案、重复、隔离冲突和超长排除可能使最终合格池少于5万。Full和所有Random分支共用准入后的同一池，不再重新抽一份分母。报告必须同时列出抽样数量、各类排除以及最终合格数量。若以后要求准入后仍严格5万，应另行定义准入后总体和抽样顺序，不能静默补抽改变原始分布定义。

## 标签与后续阶段

直接继承原始problem_type，并按已定义能力映射审计。未映射值保留并报告；这不意味着每条都必须付费teacher重标。原数据自带solution，不再依赖20k题单的参考解关联流程。

实验一末尾完成三领域标签和长度分布审计，每技能50题开发集及后续训练池发布。数学/IF已有训练曝光关系继续明确记录。Random5/10/20/30/50、18次独立SFT、21个评测分支、84组benchmark保持当前设计。超参仍待用户最后确认。

三池原始名义总量变为105,423条，训练累计约226,659条样本曝光（2.15倍；实际因准入和取整减少）。若平均输入1500—3000 token、8卡合计5k—20k input token/s，纯训练算术约4.7—37.8小时，另加padding、装载、保存及84组测评。最终以prepare后的精确token量和服务器吞吐重新估时；此前1—4天只是量级规划，非实测承诺。

## CPU验收

已用6万条显式合成原始记录实际抽取5万，验证精确数量、唯一行、两层配额、null类别、原字段保留、相同seed重放与样本不足时报错。没有下载完整Numina原集，没有生成正式研究子集或GPU结果。
