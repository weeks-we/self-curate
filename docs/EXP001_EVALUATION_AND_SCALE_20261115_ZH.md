> 当前测评协议已按用户要求重新对齐最新NICE SDK：250轮、262K、thinking/xhigh及采样。见 [唯一当前测评表](EXP001_SDK_EVALUATION_PROTOCOL_ZH.md)；下文旧75轮/128K/贪心/10000轮说明不再作为执行参数。

# EXP001：来源锁、官方测评协议与算力安排

更新：2026-11-15。本文及当前模板补充 EXP001 总交接；旧运行保留原配置，不改写恢复。

## 来源固定

|角色|数据|revision|
|---|---|---|
|任务|nebius/SWE-rebench / default / test|89cdfbab4ab1bd8f5a658bb212d1b63624f4f881|
|轨迹|nebius/SWE-rebench-openhands-trajectories / default / train|35455389ab51bf5e2306bfd436ef72d0f98bf882|

依据是官方仓库的完整commit页面：[任务](https://huggingface.co/datasets/nebius/SWE-rebench/commit/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881)、[轨迹](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/commit/35455389ab51bf5e2306bfd436ef72d0f98bf882)。
已写入 rebench.workstation.json，不能再取main。工作站仍须下载校验并核对instance_id连接、真实行数及resolved分布；本地没有下载全池或目标权重。HF API直连在本机失败，完整SHA由浏览工具读取官方commit页面获取；不把这当成数据内容验收。

## 先看官方协议，再看其他工作

### Verified与开发题

用户明确对齐NICE数值：75轮、单次输出32768、整题输出262144、上下文131072、agent墙钟3600秒、单工具300秒。贪心、seed42，每题一次。
来自本地NICE configs/main.yaml的预算，不冒称Verified统一官方标准。TaskBudget与swe-task-protocol.json均保留。
当前上下文管理是Claude原生；旧协议字段recent_tool_observations=5不等于SDK实际裁剪到5条，不能宣称两项目harness语义完全相同。正式smoke应核对实际请求；SDK不支持的旧剪裁行为不得伪报已执行。

### Pro V2

优先依据[官方V2说明](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md)：642题、每题50分钟，agent只能访问模型服务、禁WebFetch/WebSearch，捕获补丁后在干净镜像重判；报告重判结果并保留原始判分。
[官方Claude封装](https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/v2/tooling/locked_claude_code.py)只是原生Claude Code加补丁捕获，并没有统一75轮要求。

项目设置：Harbor原生agent时限不被改写，SDK在min(原生时限,2940秒)停止，留约60秒收尾。2940是本项目的捕获余量，不是将官方50分钟改称49分钟；50分钟总窗口内的有效生成时间差须报告。
上下文131072、单次32768、累计输出1048576、工具300秒；10000轮是契约要求有限正整数而设的宽松工程保护，不是官方轮数标准，也不是目标运行长度。贪心seed42每题一次。
pro-task-protocol.json单独发布，已接入SDK gate；agent不能误拿172800秒外层watchdog当推理时长。
172800秒是外层环境/验证进程安全上限，注册时必须覆盖原生build+agent+verifier，不代表给agent两天。
如轮数/token保护先于官方时间到达，必须保留stop_reason并将结果标明本项目受限协议，不能直接冒称官方无额外上限成绩。

### Terminal-Bench 4.1

[官方运行指南](https://www.tbench.ai/run)给出 terminal-bench/terminal-bench@4.1.0、Harbor、Claude Code和-k5，并提醒部分任务需要GPU。[任务定义论文](https://arxiv.org/html/2601.11868)明确每题有自己的环境、时限和最终状态测试；旧版论文不能代替4.1任务文件。

项目保留冻结task.toml中的agent/verifier/build时限、CPU/RAM/GPU需求、timeout multiplier=1；SDK gate读取该题agent.timeout_sec，不能把外层watchdog当agent预算。未知默认值通过冻结Harbor TaskConfig解析，不猜值。Terminal有独立protocol和budget，不与Pro共用75轮限制。
模型侧工程上限同Pro：131072上下文、32768单次、1048576总输出、10000轮；若提前触发必须披露。正式实验当前是一题一次贪心seed42，不是官方示例5次采样均值，报告不得标作avg@5或直接复刻榜单成绩。
Terminal所需原生GPU独立预留，不能与推理抢同一张卡；实际推理worker随剩余卡数缩放。某任务所需GPU+至少一张推理卡不满足时停止，不删题。

其他工作仅作补充：SWE-Prime使用100轮、131072上下文，且测的是旧731题Pro，不能覆盖当前V2官方协议；MetaCurate使用mini-swe-agent，不宜复制其设置到Claude SDK后声称等价。

## Teacher复核

主标注和复核都使用deepseek-flash。复核仍是独立prompt/会话/缓存身份；保留5%抽检和争议复核，但同模型错误可能相关，不能称为异模型交叉验证。100+100后仍要人工抽检语义。
review价格保护复制Flash费率； max_calls、输出16384、900秒等待与费用保护保留。旧v4-pro复核缓存身份不同，不伪装复用成Flash；已有旧运行不能原地变更。

## 8×H200 141GB的可执行安排

- 训练：一次一支八卡FSDP1 FULL_SHARD，BF16、LoRA16/32、梯度检查点；global8=8卡×micro1×累积1。保持样本顺序、监督mask和学习率，避免把改变batch当成纯工程加速。
- loss词表投影分块从32提高至256个监督位置，减少大量小算子调用；仍为完整assistant交叉熵，没有截断、改变token权重或重复曝光。新分块进入LoRA配置hash；CPU已比较loss与梯度，实际显存/吞吐由服务器验证。不是训练microbatch256。
- 推理目标：每卡一份BF16模型，4任务槽，八卡最多32任务；vLLM max_num_seqs=4、显存比例0.90、prefix cache。模型服务和任务容器的CPU/内存/磁盘也必须足够，GPU利用率不等于端到端效率。
- check_inference按声明槽数发起全上下文容量测试；共同barrier确保全部槽已占用再发送请求，要求所有GPU/槽覆盖。32路请求不是32次串行短prompt测试。
- 新并发正式运行必须有匹配model/context/output/slots的inference_capacity_report ArtifactRef，放入主capacity receipt；server preflight读取真实回执，不凭布尔标志放行。
- 每次调slots、显存比例、chunk或模型服务参数都新建预检目录并重做真实smoke；若4路吞吐更差或OOM，显式冻结新2路配置并重新验收，不能运行中悄改协议。CPU只能证明门禁和调度，没有证明4路一定装得下。
- 记录每卡显存、利用率、tokens/s、排队/墙钟、任务吞吐和训练步耗时。先保证长上下文+恢复通过，再从2/4槽工程负载对照选择吞吐较好的配置；终测结果不能用于挑配置。

## Rebench数据选择规模调研

以下按公开论文/作者发布物逐项核对；不是系统综述频率统计，也不把任务数量当训练轨迹数量。

|工作|池与选择规模|与本实验的关系|
|---|---|---|
|[SWE-Prime](https://arxiv.org/html/2608.27449)|同一OpenHands原始67074轨迹，成功32161为池，选10%（约3216）；有Random-10%和全成功池对照|最直接的规模依据；它还筛监督片段，我们监督完整assistant，不能保证效果相同|
|[MetaCurate](https://arxiv.org/html/2610.23134)|自采混合池4286，包含Rebench V1 2100、V2 1599、SWE-Gym587；默认选1000；另一Nemotron池17220也选1000|混合池而非我们原始OpenHands池；不能说通常都选10%|
|[SI2CA作者数据](https://huggingface.co/datasets/Self-Improving-Coding-Agents/SI2CA-Training-Trajectories)|10780任务=V2 6146+SWE-smith4634；三种生成策略各10780条，共32340；各配置保留正常退出的成功和失败轨迹|生成时策划，不是对同一成功池做固定比例子集选择，不拿它的任务数直接确定我们的K|
|[Trajectory-Aware Benchmark Subset Selection](https://github.com/SAILResearch/swe-agent-subset-selection)|Rebench3188任务、25279条轨迹作为分析数据|选择便宜评测题单，不是训练数据DS；不能算作25279条SFT预算|
|[公开swe-trajectory-subsets数据卡](https://huggingface.co/datasets/davongluck/swe-trajectory-subsets)|发布500/1000等Rebench子集|仅数据卡证据，未据此确认受控论文效果，证据级别低于前述论文|

建议本次先维持用户已批准的K=floor(N_success_before_admission/10)，与最直接SWE-Prime对齐。官方数据卡也报告32161成功轨迹；若锁定导出核验一致则K=3216，三波各1072。这个数目前是条件计算，不硬编码实际N。开发题/去重/Teacher排除不改分母，候选不足停止。
先保存Random/static实际监督token数、轨迹长度和唯一任务数；等样本数并不等于等token预算，结论应报告此差别。后续可另建1000/3000或比例消融，不能因正式benchmark结果不理想临时改K。

## 当前验收边界

本地只做CPU、合成网络响应、保留的完整mock和恢复/ZIP校验。没有运行真实GPU、目标权重、Docker或新付费Teacher。本次固定来源并不等于全量内容验收；四路推理、长轨迹FSDP和官方容器判分必须由服务器执行。


## Qwen模型官方说明与当前受控配置的差异

另核对[Qwen3.8-27B官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)：Pro使用Claude Code、temperature=1、top_p=0.95及256K上下文；表中Terminal为2.1版/Terminus，不是4.1/Claude SDK。通用thinking建议top_k20、reasoning_effort默认xhigh。
本项目当前仍为128K、贪心一次、high（既有研究协议），没有擅自将不同模型状态的一部分改用256K。它是受控对照配置，不是Qwen官方成绩复现。若决定采用官方模型采样/256K，应在任何M0正式初测前单独冻结新协议：服务max_model_len、各suite预算、SDK sampling、Pro实际采样probe及容量回执须一起更新；不能只改JSON的context数字，不能沿用本次128K四槽回执。此项推荐有明确依据，但尚未在服务器验收，不承诺官方成绩。

本次CPU验收：18项预算/配置/调度/loss梯度等价检查、5项真实注册产物的合成检查、31项Teacher回归及300题资产零调用复用通过。最终源码EXP001原始成功40→两组K4、四支三阶段、完成后恢复、3×5嵌套和两份ZIP CRC通过。机器记录：runs/nov15-final-verification/summary.json。未运行GPU、模型权重、Docker或新付费API。
