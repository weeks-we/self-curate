# 当前实现状态与待办

更新：2026-11-09。版本 0.16.0。EXP001 唯一交接为 [V3 Handoff](handoffs/EXP001_SERVER_V3.md)，
部署见 [容器部署](SERVER_CONTAINER_DEPLOYMENT_ZH.md)，文档导航见 [README.md](README.md)。

## 已实现

- M1 完整轨迹、assistant payload mask、质量/技能/证据、使用历史及版本化 NLL/entropy/梯度引用；ID 保序内存查询。
- M2 八算子、固定门禁与预算、精确 BatchManifest；P0/P1 外部锚点校验和不可变交接。
- 三角色共享 ModelRef、独立 prompt/history；Leval→Lopt→Lexec；NICE 和隔离候选 SFT 两类评价；内层历史最佳和恢复接口。
- 通用外层：子集 SFT→验证/判断→接收权重上的轨迹 SFT→验证/判断；判断标准由调用方提供。
- 首轮全池指标；后续按外部技能方向刷新外层未使用数据，保留旧版本。
- 共享资产与消费回执；EXP001 Random/static_m0、四分支三波、节点开发测评、训练后最终测试、optimizer-step 恢复、结果 ZIP。
- 固定历史 5016 读取、准入/隔离、七技能证据标注与任务注册；真实 tokenizer/PEFT 后端、分块 loss、GPU 指标归约和八卡分片。
- 无 GPU download/assets、Dockerfile、任务镜像和 agent 预安装；阶段服务管理、跨进程槽、Terminal GPU 独占池、推理 relay 和真实容量/smoke 代码。

## 服务器仍需执行

本机未下载目标权重、构建真实镜像或运行 GPU/任务容器，代码实现不等于服务器验收。
需固定运行镜像、来源、共享路径、Teacher/API 预算及授权 UUID，执行准备与标注、隔离检查、
完整上下文容量、checkpoint 重载及官方判分 smoke。缺资产、权限或容量不足会阻塞，不用 mock 或删题代替。

## 后续研究

- 通用 V1–V4 实验级调度、预算和确定性停止/接收规则；EXP001 不能冒充所有变体已完成。
- Lopt 技能方向输入流程、控制轨迹语料生产/质量门禁及相应训练配置。
- NICE 校准及真实任务能力结论，须按预设真实实验获得。
- 未发布的初始化/选样和未知在途调用仍需对账，不承诺任意位置自动重复。

## 验收与清理

清理前全量测试 795 passed；5016 条合成数据完整 mock 通过：选 501 条、三波各 167、
13 个模型节点、每分支 63 optimizer steps、恢复不变、ZIP 校验。这不是研究效果证据。
记录在 `runs/server-readiness-0.16.0/verification.json`。

本轮删除退役/重复文档，合并零散测试文件，保留全部测试函数及参数化。
最终清单和验证在本地 `runs/redundancy-cleanup-20261109/verification.json`。


### 2026-11-09：进一步减少日常测试范围

在上一轮仅合并文件之后，按用户要求移除 7 份独立历史专项审计测试。模块主测试、
独立数值参考、数据版本、EXP001 恢复与服务器集成仍保留；生产代码未改。
这次有意减少历史边界回归覆盖，不再沿用上一轮“全部用例保留”的描述。
备份、实际测试数量和验收记录在 runs/lean-tests-20261109/。

精简后已确认：25 个测试文件，674 项全量 CPU 测试通过；原 795 项中的 121 项历史专项不再日常执行。生产代码、运行时校验、业务任务验证及原始研究文档未变。


### 2026-11-09：正式运行目录整理

用户明确要求删除与之后运行无关的内容。剩余测试工程、五个模块 demo、专用 fixture 配置、
旧主机 venv 安装脚本及重复/过时配置已归档并从正式目录删除。保留 EXP001 整体预演、
正式后端和运行时校验。之前的 674 项是删除前验收记录，不表示删除后仍有该套测试。
本轮改为执行临时 CPU 运行检查、EXP001 预演/恢复/ZIP 验证；真实 GPU/Docker 仍需服务器验收。
清单和验证见本地 runs/runtime-only-20261109/verification.json。


### 2026-11-10：内外层 3×5 CPU mock 联调

新增专用 smoke 入口、64 条合成候选配置与 NESTED_CPU_MOCK_HANDOFF.md。三轮实际调用现有内层、外层、指标刷新、训练和验证服务；验收历史最优批次、角色隔离、跨轮模型、接受/拒绝/证据不足、实际训练曝光和指标缓存。每轮内层中途保存恢复；结果生成 checksum、CSV 与可回传 ZIP。此项不扩展正式研究控制器，也不实现通用外层进程恢复。实际本机通过记录和次数见被忽略 runs 中的本轮验收报告；真实数据、权重、GPU 和判分环境仍需另行验收。

## 2026-11-10：正式 SFT 使用 Accelerate FSDP，恢复与日志

用户要求用成熟 FSDP 框架、实现中断续训、完善带时间编号的日志，并按 8×H200（约141GB/卡）安排训练。真实 EXP001 改为一个分支占用八卡 FSDP，四个分支依次执行；训练结束释放显存后再用八卡验证已发布节点。服务管理只关闭本项目拥有的进程。原“四张独立训练卡、四张验证卡”的默认安排不再用于该后端。

新增 training/fsdp.py、fsdp_worker.py 与 runlog.py。采用 Accelerate 1.15.0/PyTorch FSDP1 FULL_SHARD、PEFT 官方自动包装、BF16 LoRA、仅 assistant payload 的精确监督损失。参数分片、分布式反向、模型/优化器/RNG/混合精度状态交给框架，项目保留精确清单和阶段控制。FSDP 包装后才构造优化器，插件创建前绑定 LOCAL_RANK。仍有项目级训练循环，不能说成已使用 HF Trainer。保持有效batch=8、每卡micro=1、累积=1，科研学习率、seed和曝光次数不改。

框架状态默认每4步及阶段结束保存，校验后原子发布；恢复核对请求、数据、模型、源码、环境和GPU拓扑，未提交窗口允许重放并记录未知花费。保留最近两份完整恢复状态，清理旧分片前保留checksum证据；三阶段adapter单独保留。worker持有OS锁，控制器中断后不能对仍运行的worker重复发起训练。通用TrainingService增加显式resume；完成结果复用不新增训练调用。通用OuterCycle的进程级恢复依然是独立缺口。

新运行支持 --output-root 自动分配 YYYYMMDD-HHMMSS-hash-id；原 --output 保留用于精确恢复。控制器、torchrun、每rank事件日志分别落盘，记录时间、token、loss、学习率、吞吐、ETA、显存、GPU利用率、保存/恢复/失败。全局token字段不能跨rank重复求和，物理rank前向token另外记录；缺失GPU时间和未知中断花费不补0。结果包不包含新的job.json、权重或恢复分片。具体路径与服务器步骤见 handoffs/SFT_FSDP_RECOVERY_HANDOFF.md。

本机使用微型随机CPU模型实际执行Accelerate保存、强制中断和恢复，最终权重与连续训练逐张量一致；回归、嵌套mock、EXP001 mock/恢复/ZIP及编号检查的实际记录在忽略的runs目录。没有在本机加载目标模型或启动GPU/Docker。8卡真实FSDP、长上下文容量、真实adapter重载和吞吐必须在服务器重新验收，不能根据CPU结果声称已经通过。


## 2026-11-10：OuterCycle 进程恢复与原 CPU 联调扩展

新增 outer_recovery.py：完整恢复状态、前序hash链与原子journal，冻结输入/源码/后端/独立model和batch锚点核验，attempt OS lease及旧实例拒绝。恢复已完成训练与验证不重复执行，持久化接受/退回决定按同一上下文续交；M1 usage、外层实际训练曝光及费用同时恢复。失败后的显式retry最多两次，无法核验的外部在途调用不盲目重发。FSDP后端接收当前接受模型的checkpoint。仅支持新格式，不改写旧snapshot，不等同于全局服务模型提交。

原self_curate.smoke入口、配置和NESTED_CPU_MOCK_HANDOFF.md保持使用，加入训练结果、验证结果、接受决定三个位置的模拟中断，并由独立Python进程恢复。仍为3×5、5次训练、42次rollout，无GPU和外网。各阶段恢复、错误锚点、损坏日志、失败重试与旧实例检查记录在忽略的runs目录；真实后端仍需服务器验收。
