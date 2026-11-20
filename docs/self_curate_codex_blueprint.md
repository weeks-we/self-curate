> 2026-11-21：每个benchmark只测最终epoch3一次；仍保存3份训练checkpoint。EXP001共72组测评。最新执行见docs/HANDOFF.md与docs/BENCHMARK_OFFICIAL_CHECK_20261121_ZH.md。

> 2026-11-20最新确认：Random50取消；Initial/Full/Random5/10/20/30，统一3 epoch，18分支/15训练/45 checkpoint/192测评。当前执行请读 docs/HANDOFF.md 与 docs/CHECKPOINT_ACCOUNTING_ZH.md；此前相反数字为历史说明。

# 当前开发指导书

## 固定原则

当前研究对象改为普通任务数据选择。代码、数学、IF 分领域；每领域从自己的固定池生成唯一 subset；不同模型在同领域比较时固定筛选比例。数据与最终 benchmark 来源见 `ORDINARY_EXPERIMENTS_ZH.md`，实验一参数见 `configs/ordinary/exp001.json`。旧 SWE 轨迹方案已撤销，不得照旧启动。

保留 M1 标准化、版本化标签/指标、稳定数据 ID、任务关联、使用历史；保留 deterministic operators、反馈、内外循环接口。新普通任务样本通过 `ordinary_m1.py` 适配，不以行号连接标签，不把缺失指标填零。

三个 agent 共享同一个 ModelRef，通过 `agents/ordinary.build_runtime` 构造 smolagents ToolCallingAgent。每个角色自己的 prompt、history、工具白名单；模型提出动作/程序，数值事实和程序校验归代码。最终 benchmark 不进入 agent 上下文。仅注册确定性受限工具，不开放任意 shell/Python 执行器。

内层冻结目标/reference 权重；外层才做真实参数更新。NICE 和候选 SFT 评估接口保留，普通任务的真实 NICE 尚待验收。内层具体停止/外层接受标准不在 EXP001 中擅自定稿。

## 执行顺序

先完成 EXP001 的准备、原生标签、免训/全量/随机基线，才进入 EXP002 的动态信号/动态程序实验。EXP003 和 EXP004 独立管理，不将宏观路线写成已实现算法。

工作站下载/构建/冻结全部模型、原始数据、题单、grader、镜像与环境；集群只计算。不能静默联网补依赖，不能用 API 代理结果冒充目标模型分数。

研究参数用户确认，硬件参数服务器实测；新配置默认 approved=false。本地 CPU 接口验收、真实 GPU 容量/恢复验收、正式研究结论三者分别报告。
