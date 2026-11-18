> Teacher v1已按2026-11-14用户决定启用六技能。正式入口与复用方式见 [Teacher交接](handoffs/TEACHER_V1_WORKSTATION.md)；本文旧的待定技能、重叠任务组和复用标注prompt的描述已被取代。


## 2026-11-14：harness 二次复查与小额 DS 连通测试

发现并修复启动入口 importlib 作用域错误、SDK终态与HTTP usage提交竞争、非法seed返回500、
角色后续checkpoint无法传递、预算停止的已知计量丢失，以及Pro短期token写入trial JSON。
32项CPU适配、7项复查回归、真实SDK0.2.87的模拟transport控制协议检查、3×5联调、
EXP001预演/完成态恢复及ZIP CRC通过。DS真实JSON和两轮工具调用成功，合计成功输出76 token；
首次强制工具请求HTTP400保留记录。key不落盘。DS连通测试不证明Qwen/LoRA/CLI/容器/NICE真实验收。
完整证据：`runs/claude-harness-review-20261114/verification.json`。

本轮实际本机验收：32项显式CPU替身检查、最终版本3×5嵌套联调（45次角色/42次任务，含跨进程恢复）、
EXP001 Random/static四分支三节点预演、完成态恢复、两份ZIP CRC及原嵌套文件checksum均通过。
结果见 `runs/claude-harness-migration-20261114/verification.json`。没有运行真实SDK二进制、目标模型、Docker或GPU。


## 2026-11-14：统一 Claude Agent SDK harness

三角色JSON会话、SWE/NICE rollout、Pro/Terminal解题改接固定SDK0.2.87+Claude2.1.153，
仍使用AdapterRouter的同一Qwen/adapter；workflow和确定性算子/梯度/训练/官方判分不交给harness。
新增vLLM0.22.0引擎token采集与NICE replay适配、种子middleware、预算网关及工作站SDK任务派生镜像。
旧mini安装/Pro协议和旧harness初测不能直接复用。原公共工作流契约保留。
实现与验收边界见 [harness说明](HARNESS_CLAUDE_SDK_ZH.md) 和
[交接](handoffs/CLAUDE_SDK_HARNESS_V1.md)。本机仅CPU模拟，真实CLI/GPU/Docker/NICE反向仍须服务器验收。

# 当前实现状态与待办

[当前流程待配置、待实现与待验收清单](IMPLEMENTATION_GAPS_ZH.md) (2026-11-11)

更新：2026-11-11。版本 0.16.0。EXP001 唯一交接为 [V3 Handoff](handoffs/EXP001_SERVER_V3.md)，
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


## 2026-11-11：EXP001 Rebench 工作站准备与初始三角色
用户固定Qwen3.8-27B、成功SWE-rebench轨迹、Verified/Pro/Terminal；技能/标注规则与规模待定。新增准备计划、工作站入口、独立Dockerfile与交接，旧5016入口不替换或冒充新数据适配。新prompt opt-in一次初测Leval→Lopt，独立历史和共享反馈资产；旧mock不启用。轻量粗筛接受tokenizer回调，基础设施错误向上传播。工作站负责联网下载/锁与Teacher、选定题单后准备环境；集群只做计算。Pro当前V2/default642与Terminal4.1经官方页核对；数据commit与环境仍待工作站实查。准备层尚非完整Rebench→M1/Teacher新schema装配，Pro干净环境重判与Terminal4.1离线资产验收不可省略。未进行目标权重下载或真实GPU/Docker/Teacher调用。


### 本轮本机验证（2026-11-11）
新增三角色mock完整完成：runs/rebench-final-20261111；all_reports_valid=true，四分支三阶段、逐阶段开发集、最终mock benchmark及ZIP。完成后--resume只复核通过。旧mock完整完成：runs/rebench-legacy-check-20261111，保持不调用Leval。直接CPU检查通过：任务抽样可复现/排除/重复ID拒绝，成功门禁/去重/工具对应/监督/长度/基础设施错误传播，Teacher预算，prompt加载、Python AST与JSON解析，ZIP CRC；下载器以假传输验证固定revision/不重复下载/篡改拒绝。真实HTTP直连受本机网络限制，未做真实下载；未构建Docker、调用Teacher或使用GPU。新来源完整M1装配与Pro V2 clean-sandbox regrade仍未完成，不把本轮准备入口描述成正式实验可启动。

## 2026-11-11：Rebench 实验一工程实现与CPU验收

已实现用户此次要求的准备/桥接、Pro V2官方两阶段判分调用、离线环境与交接接口；固定硬筛前成功轨迹10%和每技能50开发题，Random未遗漏。详见最新计划和交接；之前“尚无M1桥接/Pro实现”的描述仅代表当时状态。

本轮已实际通过：
- runs/rebench-stage-check-ac46ca89：模拟Teacher任务选择与真实本地Parquet按split导出；流式pool阶段写140条左右M1（确数见extended-cpu.json）、200条成功的K=20、技能absent/uncertain筛选语义、工具定义保留、共享Teacher缓存的本次预算、export-table CLI不覆盖JSONL、FSDP CPU inspect不伪报GPU通过。
- runs/rebench-v2-check-b9602eb4：固定分母/短缺拒绝、每技能50及并集、M1/PreparedInputs往返、假传输下Pro agent→pristine replay且仅后者reward生效、离线文件篡改与镜像传输校验。
- runs/rebench-nested-final2-20261111：既有3外层×5内层CPU联调通过，含跨进程恢复与zip。首次未设置可供子进程继承的PYTHONPATH而失败；显式设置后新输出根重跑通过，不改模块接口。
- runs/rebench-legacy-final2-20261111：旧EXP001四分支三阶段及ZIP通过。
- runs/rebench-fsdp-cpu-9a391e47：微型随机模型用实际Accelerate CPU训练，新增preflight中断钩子在提交第1步后退出，再恢复；最终参数与连续对照逐张量相同，8 rank曝光划分无重叠遗漏。真实8卡FSDP未执行。
- 新预算完整mock及完成后恢复通过；最终源码的发布验收目录与后续复用/ZIP复核另记于下方。

未下载目标权重，未调用付费Teacher，未构建/运行真实Docker，未使用GPU。真实来源revision解析/全池数据兼容、Teacher上下文与标注质量、Pro/Terminal官方离线容器、八卡容量/吞吐/重载等仍由工作站和集群验收。用户技能规则是明确的阻塞项，不擅自填入。

### 最终发布验收（同日）

最终源码完整mock：runs/rebench-budget-release-20261111，all_reports_valid=true，两种模式均K=4，四分支三阶段，13个模型节点的最终模拟benchmark、CSV/PNG与results.zip均生成；完成后 --resume 只复核通过。summary真实记录leval_calls=1、initial_feedback_available=true、inner_valuation_calls=0，反馈和初测/指标/策略均列入shared_assets。扩展准备验收实际写出143条M1，硬筛前200条成功→K20。详细机器相关证据都在被忽略runs，不提交原始运行输出。

Final reuse check: runs/rebench-budget-release-reuse-20261111 completed; all five shared asset references identical to first run, new Leval calls=0 vs first run=1, K4 unchanged, both ZIP CRC checks passed. No target/GPU/Docker/Teacher acceptance implied.


Teacher V1 本地交付验收（2026-11-14）：31 项临时 CPU 检查通过；330 条模拟任务标注选出 300 道互异题，跨准备目录复用为 0 次 Teacher 调用。真实 API 小试完成 3 条任务、2 条完整轨迹的 schema 和逐字证据校验；早期失败保留。最终源码下 3×5 内外层 mock、EXP001 Random/static mock、恢复与 ZIP CRC 通过。真实 Qwen tokenizer 尚未接入本次小试，不能声称真实 token 比例已验收；正式 100+100 试标和语义人工抽检待工作站执行。验收记录：runs/teacher-v1-verification-20261114/summary.json。


2026-11-14 EXP001文档收口：唯一总入口改为Rebench工作站handoff；旧SERVER_V3标为历史恢复，README同步。统一六技能、300互异题、100+100试标及Claude SDK；新增阶段输入/输出/阻塞/恢复/交付清单。仅文档变更，未改运行接口或配置身份，未执行GPU。

2026-11-15：按用户要求固定Rebench任务/轨迹SHA、Flash复核；先官方协议再文献，分离Pro/Terminal预算并将原生agent timeout传入SDK，拒绝复用旧预算注册。推理槽与worker可配置，目标4/卡并以真实并发容量报告门禁；loss分块256保持相同监督目标，global8与10%预算不变。来源与规模/协议证据见EXP001_EVALUATION_AND_SCALE_20261115_ZH.md；真实GPU与容器仍未验收。

2026-11-15最终CPU验收：18项专项、5项任务注册、31项Teacher与300题复用、EXP001 K4完整mock及恢复、3×5嵌套、两份ZIP CRC通过。记录runs/nov15-final-verification/summary.json；没有GPU/容器/新付费Teacher通过证据。另核对Qwen官方Pro256K+采样与当前128K贪心的协议差异，详见本次实验补充，不冒称复现。

2026-11-15用户纠正NICE最新SDK配置：以reevaluation.yaml为对齐源，统一250轮、262K上下文/总输出、单次32K、Verified7200秒/工具1200秒、temperature1/top_p.95/top_k20、thinking/preserve/xhigh。Pro保留官方50分钟、Terminal原生时限，其余沿Verified；移除10000轮/1M输出自定保护。增加逐请求vLLM Anthropic转换策略与引擎采样证据、独立前缀262K容量和新Pro sampling回执。训练131K/全局batch8不变；旧资产/运行不原地迁移。见EXP001_SDK_EVALUATION_PROTOCOL_ZH.md。

250轮SDK对齐最终CPU验收：17项参数/转换/预算检查、7项独立前缀及Pro错误回执拒绝、完整EXP001 K4四支三阶段、3×5嵌套恢复和两份ZIP CRC通过。证据runs/sdk250-verification/summary.json；不代表真实vLLM/262K GPU容量或官方判分已通过。

2026-11-15用户批准一题多轨迹：新增版本task_key、开发题关联资产与M1关系标签、按完整输入/输出预算分组（默认2条）、逐成员证据校验/复核/恢复、pilot与正式流程同规划器；m2-v3新增可选TaskLimitSelect和Lopt任务覆盖统计。NICE/梯度/指标保持原轨迹及模型/上下文身份，不传播为任务分数，Random未强制限额。本轮只CPU模拟，无新增付费API。见TASK_TRAJECTORY_LINKS_ZH.md。

任务关联交付CPU验收：31项既有Teacher契约、27项关联/分组/乱序ID匹配/部分恢复/预算拆组/引用隔离/TaskLimitSelect检查、300题资产零调用复用、EXP001四支三阶段K4及完成后恢复、3×5嵌套跨进程恢复和两份ZIP CRC通过。证据runs/task-links-verification-20261115/summary.json。本轮新增付费API为0；真实合批语义一致性和节费幅度仍待小规模API试标。


2026-11-15 Teacher 复核：新增 off/flagged/sampled/all，新模板默认 off；保留证据/身份/格式校验与逐条审计，兼容旧直接集成。真实单对样本复核两次均 approve，费用受缓存影响；本次未新增付费请求。详见 TEACHER_REVIEW_POLICY_ZH.md。


2026-11-15：官方元数据核验后固定 Pro HF 2d52cb3df914a3fcf80c7f66738b3a88ae37fc50、Pro tooling 66f92766bba642462d4bbe5479e83f91f9211862、Terminal4.1.0 4070d6cef56a7d20cb5dc98d5e10ada88a6b9440。开发集移除固定16worker瓶颈，随可用服务槽调度，8卡×4槽目标32并发；保留长上下文容量门槛和Terminal GPU预留。详见 EXP001_FROZEN_BENCHMARKS_ZH.md，未执行GPU/容器验收。


2026-11-15 EXP001测评调整：保留每分支三个开发测评，最终benchmark从initial＋12阶段权重改为initial＋四个stage3，按固定训练末节点，不按测评分数选模。旧结果不改写，新run执行。


2026-11-16：按用户要求Coding默认task-required标签，成功任务一次标注并关联轨迹；开发取无成功覆盖任务；不生成轨迹证据及占比，missing保留，旧模式兼容。General专门IF方案仅文档待实施。见CODING_TASK_LABELS_ONLY_ZH.md与GENERAL_IF_PLAN_ZH.md。未调用付费API。


2026-11-16：当前EXP001使用任务继承标签与M1v4（不发布技能证据片段/比例字段）；EvidenceSelect及evidence_batch删除，operator版本m2-v4。旧M1类型只作历史读取兼容，不能将旧程序当新算子版本恢复。详见CODING_TASK_LABELS_ONLY_ZH.md。


2026-11-18：SDK结果封存、清理解耦、网关静默读取取消的部分修复；6项CPU及3×5 mock通过。完整逐项状态和未完成项见 SDK_REPAIR_20261118_ZH.md，尚未正式GPU/容器验收。


2026-11-18追加修复：持久SWE判分重试、Pro分阶段、Terminal禁止自动重放、最终全目录seal、压缩时序验证/持久server探针、阶段成本、每槽容量失败证据。25项专项CPU、3×5 mock及EXP001四分支通过。Terminal跨进程verifier-only仍需冻结Harbor适配；真实GPU/SDK/Docker未验收。详见SDK_REPAIR_20261118_ZH.md。


2026-11-18实验安排：EXP002比较V1固定程序动态信号与V3动态程序（3外层×5内层，无控制轨迹SFT，NICE/SFT验证待定）；二者共用Lopt方向驱动、排除真实外层曝光数据的定向刷新。EXP003加入V4控制学习、用户提供的baseline、已训checkpoint补充测试、统一prompt/history的单agent消融。详见EXP002_EXP003_PLAN_ZH.md；本次仅文档，未启动或宣称实现新workflow。


2026-11-18文档组织纠正：实验二与三拆为EXP002_PLAN_ZH.md、EXP003_PLAN_ZH.md，删除上轮新建的合并计划EXP002_EXP003_PLAN_ZH.md。旧记录中的合并文件引用仅为历史；当前分别独立配置/workflow/handoff/run与验收，共享核验后的只读资产。
