# 当前交接入口

> 最新长度与逐benchmark预算见[长度/预算核查](TRAINING_LENGTH_AND_BENCHMARK_BUDGETS_ZH.md)。原文统一4096/8192输出仅为旧设置；当前执行exp001.evaluation.benchmark_budgets。三个epoch测评及A/B范围遵循HANDOFF。


- **师兄A：完整正式实验一**：[HANDOFF_A_EXP001_ZH.md](HANDOFF_A_EXP001_ZH.md)。包括常规三领域Initial/Full/Random以及内置NICE附加实验。
- **师兄B：测试验收**：[HANDOFF_B_ENV_ACCEPTANCE_ZH.md](HANDOFF_B_ENV_ACCEPTANCE_ZH.md)。准备环境、三领域短训/12个benchmark；NICE仅加梯度probe，不跑其SFT/测评。
- **NICE具体定义与接口**：[EXP001_NICE_ZH.md](EXP001_NICE_ZH.md)。20×4、3,000候选、高低各1,000、两LR、1epoch、三进度节点。

主配置configs/ordinary/exp001.json，NICE定义configs/ordinary/exp001_nice.json，固定来源configs/ordinary/sources.json。两LR已选5e-5/1e-4；approved=false仅保留为服务器验收门。A/B异地独立环境和资产；A从零准备并本机验收，B仅提前测试反馈；各自独立run_id及输出；用户自行Git提交推送。

判分与GPU配置见BENCHMARK_OFFICIAL_CHECK_20261121_ZH.md，数学50k抽样见MATH50K_STRATIFIED_ZH.md，成本口径见CHECKPOINT_ACCOUNTING_ZH.md。此前72结果的单LR常规矩阵、单独目录NICE的旧交接均不作为本次入口。

2026-11-21最终配置核查：[当前配置与待填项](CONFIGURATION_AUDIT_20261121_ZH.md)。修正roadmap Random30/NICE、NICE进度协议元数据和配置一致性校验；科研配方已选定，真实容量/镜像/凭据由B/A验收。

最新训练与环境完整参数：[EXP001_QWEN35_9B_RECIPE_AND_ENV_ZH.md](EXP001_QWEN35_9B_RECIPE_AND_ENV_ZH.md)。两档LR已选；B须重做4卡分组和并发/恢复验收，旧源码receipt不能复用；A另在自己的环境生成凭据。

2026-11-21最新确认：EXP001普通基线epoch1/2/3全部跑benchmark；Initial仍只测一次。epoch3为主要报告，不按benchmark挑权重。两LR正式矩阵372组benchmark；NICE保持48组benchmark＋12组原题诊断，总432组测评记录（420 benchmark＋12诊断）。师兄B仍保持独立短测：每个benchmark只测一次，不扩大验收规模。此前相反节点说明以本条为准。

当前仍需确定及服务器待办：[实验一／NICE就绪清单](EXP001_READINESS_CHECKLIST_ZH.md)。包含末尾标签口径及本轮修复，不要把技能表proposal当作已完成覆盖验收。

2026-11-21用户确认标签：保留8/7/10类；未映射样本先不标、保留训练池、不自动调用teacher；按技能开发集暂不使用未映射记录，缺额如实报告。原生对应关系见[NATIVE_SKILL_MAPPING_ZH.md](NATIVE_SKILL_MAPPING_ZH.md)。
