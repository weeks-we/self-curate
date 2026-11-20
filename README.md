> 2026-11-21：每个benchmark只测最终epoch3一次；仍保存3份训练checkpoint。EXP001共72组测评。最新执行见docs/HANDOFF.md与docs/BENCHMARK_OFFICIAL_CHECK_20261121_ZH.md。

> 2026-11-20最新确认：Random50取消；Initial/Full/Random5/10/20/30，统一3 epoch，18分支/15训练/45 checkpoint/192测评。当前执行请读 docs/HANDOFF.md 与 docs/CHECKPOINT_ACCOUNTING_ZH.md；此前相反数字为历史说明。

# Self-Curate

普通任务上的反馈驱动数据筛选框架。当前主模型 Qwen3.5-9B，扩展模型 Qwen3.5-4B、Llama-3.1-8B-Instruct；thinking 关闭。

当前唯一实验方案：[普通任务选型与路线](docs/ORDINARY_EXPERIMENTS_ZH.md)。先做三领域免训、全量及 Random 5%/10%/20% 基线，详见 [EXP001](docs/EXP001_ORDINARY_ZH.md)；服务器 agent 从 [HANDOFF](docs/HANDOFF.md) 开始。

```bash
python -m pip install -e '.[ordinary]'
python -m self_curate.experiments.ordinary smoke --output runs/ordinary-smoke-new
```

该验收不用 GPU、目标权重或付费 API。正式配置为 `configs/ordinary/exp001.json`，尚待用户确认参数，默认禁止正式执行。来源 SHA 在 `configs/ordinary/sources.json`。

数据原生标签接入 M1，三个角色使用 smolagents ToolCallingAgent；普通 benchmark 直接生成后由官方 checker 判分。工作站准备下载与镜像，集群断网训练/推理。旧 SWE 实验配置、交接及设计文档已退出，通用模块契约保留。

源码、配置进入 Git；数据、权重、缓存、运行结果和个人会议材料不进入 Git。项目结构见 [阅读顺序](docs/CODE_READING_TUTORIAL_ZH.md)。
