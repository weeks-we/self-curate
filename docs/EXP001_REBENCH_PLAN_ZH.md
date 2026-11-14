# 实验一：成功轨迹池上的 Random 与静态策略

本方案使用独立的 self-curate 项目，不继承相邻 NICE 项目的配置。技能定义和 Teacher 标注规则由用户另行确认；缺失时准备流程明确停止，不沿用旧七技能。

## 已固定的比较

- 数据：`nebius/SWE-rebench-openhands-trajectories` 中 `resolved=1` 的轨迹；按 `instance_id` 关联 `nebius/SWE-rebench` V1 任务。任务与轨迹分别保存。
- 预算：N 为**硬筛前全部成功轨迹数**，K=floor(N×10%)。开发题隔离、长度和去重不改变这个分母。Random 与静态策略各选 K 条，候选不足直接报错，不补抽、不缩预算。
- 两组都从同一个硬筛、隔离后的池中选择。Random 固定 seed=42；静态策略使用 M0 制定的一次程序，训练过程中不更新程序或 NLL/entropy。
- 开发题每技能独立选 50 题；多标签题可以属于多个组，实际执行并集一次，再按固定组成员统计。缺少 50 题时停止，不用重复题凑数。
- 训练：两种选择 × 两个学习率（5e-6 / 2e-5），四个独立 LoRA 分支均从同一个初始 adapter 开始；每支一遍，将固定顺序近等分为三波，连续训练并保留三个阶段。
- 初始模型及每个阶段在开发集验证。所有训练完成后，初始模型与 12 个阶段权重在冻结 benchmark 上测评，不依据 benchmark 调参或选权重。

模型为固定 revision 的 `Qwen/Qwen3.8-27B`，加载类 `AutoModelForImageTextToText`。最终 benchmark 为 SWE-bench Verified、SWE-bench Pro V2、Terminal-Bench 4.1。Verified 500、Pro V2/default 642 题；Terminal 的官网版本与 registry 条目数字不同，须核验真实下载内容和来源 commit，不能猜 CLI 版本。来源锁在工作站生成，此后不跟随 latest。

## 工作站先准备，集群只计算

1. 锁定来源、下载模型/tokenizer、任务和轨迹、三个 benchmark；按冻结 config/split 导出 JSONL，保留来源侧车回执。
2. 生成 benchmark 隔离目录；从非冲突任务中固定随机候选前缀，Teacher 标注后每技能选 50 题。默认候选数量为 min(可用题数,max(1000,200×技能数))，不是最终题数。
3. 固定开发题后，只准备这些开发题及最终 benchmark 的执行环境，不准备全部 Rebench 任务容器。开发题与轨迹按 ID、来源家族、近似题面隔离。
4. 成功轨迹做轻量硬筛：结构、工具对应、有效 assistant 监督、真实 tokenizer 长度、完全去重和集合隔离。成功依据源数据可信记录，不重新执行每条轨迹，不把 Teacher 评价当作执行证明。
5. Teacher 标注粗筛后的轨迹，引用真实消息证据并映射 token 位置。`present/absent/uncertain` 保留，只有 present 进入正技能标签；uncertain 按算子缺失规则处理。原工具定义纳入训练上下文，原记录另存。
6. 发布 M1、开发题与 benchmark 注册、模型引用、来源锁和 PreparedInputs。任务候选、标注响应、请求费用和审计证据可复用；schema、prompt、内容或模型改变会改变资产身份。
7. 工作站完成 Python 安装、任务镜像构建、官方 oracle/nop/依赖与断网检查，再导出镜像和文件清单。集群只导入、核验、加载和计算，缺资源时停止，不下载或安装。

轨迹粗筛和标签装配按行处理，避免复制全池原始对话；最终 M1 视图仍会加载完整 token 数组。工作站必须有足够 RAM/磁盘，不能把“CPU 可准备”理解成普通笔记本能装下 27B 权重和全池 tokens。初始 adapter 的准备目前会在 CPU 加载完整基座；本机验收不执行这一步。

## 三角色与训练

首次开发题结果 → Leval 一次事实汇报 → Lopt 一次筛选程序 → Lexec 授权 → 确定性算子逐节点执行、记录数量/成员与排除理由。三角色共享 M0，prompt、history 隔离。Leval 本轮不做 NICE 或候选 SFT。基线能力、全集初始 NLL/entropy、初始反馈和初始程序均按完整身份缓存；复用不伪装成新计算。

八卡使用 Accelerate FSDP1 FULL_SHARD，一次运行一个分支：global batch=8卡×micro 1×累积1=8；BF16、LoRA rank16/alpha32/dropout0。已有精确监督 mask、框架 checkpoint 与 RNG 恢复。训练结束释放显存再切推理。此配置是可验收起点，不能根据 CPU mock 声称八卡容量或吞吐已充分利用；真实长上下文、恢复、重载及利用率由集群 GPU 验收。

## 执行入口与状态

- 来源下载、导出：`experiments/exp001/workstation.py`。
- 任务标注、轨迹准入、M1 发布、正式配置生成：`preparation/rebench_workstation.py`、`rebench.py`。
- 环境目录/镜像/安装快照/注册：`server/rebench_environments.py`。
- Pro 官方 agent 后以干净环境重判：`execution/pro_v2.py`。
- 离线证据与镜像导入导出：`server/offline.py`。
- FSDP 专用 CPU 合约检查/服务器中断恢复对照：`server/fsdp_preflight.py`。
- 三角色模板：`configs/experiments/exp001/prompts-rebench/`。

Teacher 主模型 deepseek-flash，复核 deepseek-v4-pro，temperature=0，单次输出上限 16384，等待 900 秒；不是输入容量或金额上限。初始调用上限分别 ceil(1.25×任务候选数)、ceil(0.10×任务候选数)，轨迹数量确定后扩为任务候选+准入轨迹的同一公式。固定 5% 随机复核，并对不确定/格式问题复核；总复核上限 10%，超限停止。缓存按本次消费请求记账；未知在途请求需核对，不盲目重发。先小样本试标，完整上下文过长时停止，不能截断证据后继续。

详细操作见 [工作站交接](handoffs/EXP001_REBENCH_WORKSTATION_V1.md)。本次 CPU 检查只能证明合约和 mock 流程；真实数据下载、Teacher、Docker/Harbor、离线官方判分和八卡容量仍需对应机器验收。尚待用户填写技能定义/两类标注规则，部署侧填写实际 GPU UUID、容器网关、完整 harness commit 与验收回执。

官方来源：[轨迹](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)、[任务](https://huggingface.co/datasets/nebius/SWE-rebench)、[模型](https://huggingface.co/Qwen/Qwen3.8-27B)、[Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)、[Pro V2](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md)、[Terminal](https://www.tbench.ai/)。
