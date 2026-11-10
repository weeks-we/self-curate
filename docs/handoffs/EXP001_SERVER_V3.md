> SFT execution updated 2026-11-10: see [FSDP handoff](SFT_FSDP_RECOVERY_HANDOFF.md). Earlier four independent single-GPU worker instructions are superseded.

# EXP001_SERVER_V3：联网准备与 GPU 执行分离

状态：代码交接；正式结果必须通过服务器容量、环境和评测 smoke。不得把本机 mock 当研究结果。

执行项目为 self-curate。禁止引用相邻 NICE 项目的参数或结果。先读 AGENTS、README、
`docs/progress.md`、`docs/SERVER_CONTAINER_DEPLOYMENT_ZH.md`。

## 实验范围

Random 与 static_m0；共同 M0，两个学习率 5e-6/2e-5，seed=42，四个独立 LoRA 分支。
原始历史来源固定 5016 条；准入、去重及隔离后实际 N 决定 floor(N×10%)，三波单次曝光。
不能把被隔离/拒绝数据补回来凑 501 条。有效 batch=8，保留尾窗，连续 optimizer/RNG。
M0 开发集测一次，static 全池 NLL/entropy 一次、Lopt/Lexec 一次并冻结；不运行 Leval、
指标刷新、控制轨迹训练或接受/退回。每个训练节点立即测开发集；所有节点训练结束后
才做最终测试。SWE 三套分别报告，Terminal 使用其原生环境和资源，不相加冒称独立题数。

## 交接执行顺序

1. 冻结用户提交的源码。配置镜像节点联网、无 GPU：按部署文档构建并记录运行镜像 digest，
   同时锁三个 harness 完整 commit、依赖 freeze。禁止在 GPU 作业内 pip install/升级。
2. CPU 联网节点运行 `--stage download`、`--stage assets`。模型/数据下载按固定 revision；
   任务镜像需准备在 GPU 执行节点会使用的 Docker daemon，或按清单 save/load/私有镜像仓库迁移。
   不要误以为运行镜像里面包含宿主 Docker 缓存。
3. 填好本地计划文件（模板 `configs/experiments/exp001/server_bootstrap.template.json`），
   storage_root 用平台持久路径，保持容器内外绝对路径一致；run_id 不含路径分隔符。
   Teacher 的 base_url/model/API key 环境变量按真实账户填写，不假定示例模型可用。
4. 申请 8 个明确授权的 H200 UUID，再运行 `--stage prepare`；检查准入、隔离、技能覆盖、
   Teacher 费用和抽样审计报告。未标注/基础设施失败不能变成质量不合格后悄悄删数据。
5. `--stage preflight`：每张卡完整上下文反向/LoRA 梯度、优化器内存和权重重载；
   并发推理容量、SWE-smith 官方判分、SWE 三套官方参考补丁环境检查、独立工程 Harbor 任务。
   这不允许提前用模型跑 benchmark。容量不足不能静默截断、降上下文或改研究参数。
6. `--stage run`；中断后同一代码、环境、计划和 run_id 使用 `--resume`。
   检查未知在途调用与失败报告，不得为省事重置快照或重试模型失败。新代码使用新 run_id。
7. 检查状态、报告有效性、结果 ZIP 校验后交付。人只需处理明确阻塞项；不要猜测来源或伪造通过。

## 算力安排

初始化指标全 8 卡按 token 长度平衡分片，每卡常驻一份冻结参考模型。训练四分支各一张卡，
其余卡服务节点验证；训练结束释放训练显存，全卡推理。每个服务两个任务槽，跨进程文件锁
避免过订阅。Terminal 原生需要 GPU 时从独占池分配，其余卡做模型服务。这里不是 DDP。
LoRA CPU activation offload 仅用于长上下文，容量验收先检查主机 RAM，GPU 利用率落真实日志；
未测量前不得声称八卡满载或最优吞吐。

## 保存与交付

`<storage_root>/sources`：核验的原始来源；`task-images`：任务镜像和预安装清单；
`models`：共同初始 adapter；`annotations`：Teacher 原始请求/回复和费用；
`prepared`：M1 数据、技能证据、隔离/准入报告、开发集与 benchmark manifest；
`shared-assets`：绑定完整身份的 M0 能力、全池指标、首次策略及消费回执。
`executions/<run_id>`：环境、容量验收、服务日志/利用率、解析后配置、准备进度；
`runs/<run_id>`：checkpoint、训练/评测逐题证据、token/时间/费用和结果；
`runs/deliverables/<run_id>/results.zip`：供师兄打包发送，包含清单/校验、配置和源码身份、
指标与 CSV、开发/测试报告、轨迹/判分/日志及复用证据，不含原始数据、基座权重、optimizer。
大 adapter/完整恢复状态留在服务器持久存储，不能删后宣称可自动续跑。

## 阻塞处理

缺依赖/来源/镜像/Teacher 配置、覆盖不足、容器不通、容量不够均停止，保留原因及成本。
Docker socket、相同路径挂载、host networking 与 NVIDIA runtime 必须由平台允许。
Terminal 任务原生脚本可能本身联网；本交接预下载和预安装不等于允许修改任务协议或保证
任务运行完全断网。需要断网平台时先做网络封锁的工程 smoke，失败即报告，不改测试脚本。
不得提交、推送、合并 Git；由用户自行操作。
