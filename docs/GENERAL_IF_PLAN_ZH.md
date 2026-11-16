# General 第一版方案（待实施）

2026-11-16。本页只记录方案，不创建运行配置或启动标注/训练。

- 研究定位：指令遵循领域内的数据筛选，不能据此声称全面General能力。
- 候选池：allenai/tulu-3-sft-personas-instruction-following，当前29,980条，只有train split；冻结revision并去重、隔离后公布实际数量。它含用户请求、assistant参考回答、constraints。
- 标签：优先复用constraints，代码映射成约束类别；仅歧义项可调用Teacher，不全量重标，不生成证据片段和占比。已有参考回答不等于一定满足约束，可执行checker的部分须检查。
- 训练预算：暂议10%，原始规模对应2,998条；分母、模型、学习率、轮数及多阶段协议启动前冻结，不自动继承Coding配置。
- 开发集：从allenai/IF_multi_constraints_upto5（当前95,373条训练侧prompt任务）独立抽300–500题；核对instruction_id/kwargs与checker支持。它不是完整SFT问答集；模型生成后按约束程序判分，不能据此判断所有知识内容都正确。
- 隔离：开发题与训练问答做ID/规范化文本/近重复检查，排除共享上游题目；IFEval与IFBench最终题不得进入优化。
- 最终测试：IFEval541题，IFBench当前官方300题；MMLU-Pro12,032题检查知识/推理保持，validation70题不充当自建开发集；AlpacaEval2的805题暂缓。
- 训练和开发分别使用assistant监督与自由生成，复用现有模块；General接口和真实执行尚未配置、未验收。
- 标签复用本身无Teacher费用；歧义试标可先设20–50元预算，训练/测评GPU成本另算，不冒称已测定。

官方来源：[personas IF](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following)、[开发任务来源](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5)、[IFEval](https://huggingface.co/datasets/google/IFEval)、[IFBench](https://huggingface.co/datasets/allenai/IFBench_test)、[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)。规模以最终锁定导出复核。
