# EXP001 来源冻结与测评并发（2026-11-15）

本次只读取官方发布说明及 HF/GitHub 元数据，不下载目标权重或执行容器/GPU。

| 来源 | 冻结版本 | 完整 revision |
| --- | --- | --- |
| ScaleAI/SWE-bench_Pro | default/test，V2，642 题 | `2d52cb3df914a3fcf80c7f66738b3a88ae37fc50` |
| scaleapi/SWE-bench_Pro-os | v2.0.0，官方任务和判分工具 | `66f92766bba642462d4bbe5479e83f91f9211862` |
| harbor-framework/terminal-bench | 完整版 v4.1.0 | `4070d6cef56a7d20cb5dc98d5e10ada88a6b9440` |

来源：[Pro 数据卡](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/tree/2d52cb3df914a3fcf80c7f66738b3a88ae37fc50)、[Pro 官方工具](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2)、[Terminal 发布](https://github.com/harbor-framework/terminal-bench/releases/tag/v4.1.0)、[官方运行命令](https://www.tbench.ai/run)。HF SHA 由官方 API 返回；两个 GitHub SHA 由 release tag 对应 commit API 返回。

计划已填 SHA；Dockerfile.rebench 的 PRO_COMMIT 默认固定上述工具 SHA。工作站仍须把真实 checkout、文件 checksum、镜像 digest 和环境 lock 保存为下载/安装回执；源码 SHA 不等于镜像 digest。不要改写已有旧 source-lock，使用新输出根重新 resolve。

## 工作站准备 Terminal

以下 clone 只在联网工作站执行；先确认目标目录尚未存在：

```bash
git clone https://github.com/harbor-framework/terminal-bench.git /storage/self-curate-rebench/terminal
git -C /storage/self-curate-rebench/terminal checkout --detach 4070d6cef56a7d20cb5dc98d5e10ada88a6b9440
git -C /storage/self-curate-rebench/terminal rev-parse HEAD
cp configs/experiments/exp001/terminal-source.template.json /storage/self-curate-rebench/terminal-source.json
```

核对 rev-parse 后再发布 registry；任务目录就是仓库 tasks/。完整 release 保留 GPU 题，不换成 63 题 CPU 子集。registry 会计算实际任务文件 checksum。Pro 任务位于镜像 /opt/graders/pro/v2/tasks；遵守官方 pristine-sandbox regrade、禁止 agent 联网和 50 分钟约束。

## Claude Code + vLLM 并发

保持 Claude Agent SDK/固定 Claude Code CLI 作为 harness；模型由本地 vLLM 提供。每张 H200 独立一份模型服务，不使用 8 卡 tensor parallel。每卡 4 槽、max_num_seqs=4、gpu_memory_utilization=.90、max_num_batched_tokens=8192；已启用 prefix caching 和 chunked prefill。8 卡正常测评目标共 32 个任务并发。

本次修复开发集 validator 原来硬编码 16 worker 的瓶颈，改为与当前服务槽数一致；benchmark 本来已按槽求和。阶段切换后同步更新 worker；Terminal 原生任务所需 GPU 按既有调度预留，因此模型可用卡数减少时并发相应减少。CPU、RAM、Docker 容器和文件描述符也需满足并发要求。

32 槽是配置目标，尚未真实验收；不是承诺 32 条同时占满 262K KV 都能驻留。既有独立长前缀容量门槛继续生效。服务器可试每卡 4/6/8 槽并比较 token/s、任务/小时、KV preemption 和 OOM；冻结最高稳定吞吐配置，不能只看 GPU 利用率。更改槽数要同步 max_num_seqs 并重做匹配的容量回执；不得降低题目 token/时间预算来伪装通过。

vLLM 参数依据：[官方性能调优](https://docs.vllm.ai/en/latest/configuration/optimization/)。本次没有 GPU 吞吐实测。
