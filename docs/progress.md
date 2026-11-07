# 当前进度（代码 0.12.0）

更新日期：2026-11-07。本页只描述当前已实现的功能和后续事项；完整阶段记录、复查发现与旧计划统一保存在 [HISTORY.md](HISTORY.md)。原始研究蓝图与架构文档保持原样。

## 定向刷新（2026-11-07）

新增 [FocusedRefreshService / SkillFocus / 外层使用历史](FOCUSED_SIGNAL_REFRESH_ZH.md)。第一轮全量初始化；第二轮起仅处理 Lopt 外部传入技能方向内、从未用于外层真实子集参数更新的数据。内层选择/试训不计入，外层退回仍计入。精确缓存、版本保留、完整落盘、失败费用和严格重载已接通；活跃内层不被修改。方向推断、实验自动排程、完整外层恢复和真实后端仍待实现。

## 共享资产与更新规范（2026-11-07）

用户原实验计划、数据准备详细规范与最新执行 prompt 已原样归档到 docs；连接顺序和当前缺口见 [EXPERIMENT_EXECUTION_PLAN_ZH.md](EXPERIMENT_EXECUTION_PLAN_ZH.md)。已有 [逐实验 handoff 约定与模板](handoffs/README.md)，当前无 ready 的真实执行交接。刷新接口已实现，接下来实现首次实验所缺模块/控制入口并冻结配置和 handoff；不将“文档已写好”当“代码已实现”。

用户确认先落实资产复用，数据准备/标注留待第一次实验执行。新增 AssetStore/AssetSpec/Producer、M0 NLL/entropy/开发测评及初始 Lopt 调用桥接、独立消费回执和冻结运行输入。生产快照校验 hash 后原子发布，不覆盖旧结果；消费不重记原模型/训练费用。原始 Lopt before-context 可从已有 RoleRuntime 导出，程序和选集分别管理。

接口与服务器 main/固定 worktree/环境/外部存储的更新约定见 [EXPERIMENT_ASSETS_ZH.md](EXPERIMENT_ASSETS_ZH.md)。本轮没有执行服务器 Git 更新、目标模型、API、数据准备或真实环境。全量测试与交付证据以本地 runs/shared-assets/verification.json 为准。实验控制 runner、自动重算/复用排程、NICE 特征重组和完整外层恢复仍待实现。

## 渐进实验就绪审计（2026-11-07）

V0–V4 的实际就绪性、执行语义差异、可复用资产与最小推进顺序见 [EXPERIMENT_READINESS_ZH.md](EXPERIMENT_READINESS_ZH.md)。当前核心模块可复用，但尚无统一配置化实验入口；仍需两次外层尝试接续、控制训练关闭路径、V2 结构冻结、跨进程资产恢复、真实执行后端和 V4 对照支持。该页是待办审计，不表示这些能力已实现；本次只补管理文档，没有修改运行代码或启动实验。


## 已实现

- M1：完整轨迹正规化、assistant-only mask、质量与技能证据、版本化 NLL/entropy/梯度引用、分开的使用历史、按 ID 保序的内存 MetricView。未标注/未计算与真实零值明确区分，不增加持久化 operator 索引。
- NLL/entropy：实际隔离 Torch 冻结前向或外部全词表 raw logits，按 p-1 计算，支持 assistant/技能范围、mean/sum、nats/bits。完整模型/adapter/tokenizer/输入/计算身份绑定，逐 token 缓存复用，结果回填 M1 并发布新池。
- 指标持久化：根请求 schema=2 保存原输入与聚合统计；重载核对数值、来源和标签增量，失败时保存部分 M1 记录与费用。旧 schema=1 只读兼容，保留其证据能力边界。
- M2/P1/P0：八类算子、固定 QualityGate/BudgetPack、严格 DAG、精确成员与训练顺序、版本化任务/报告/轮次及成本契约，支持标准交接与原快照重放。
- 三角色与内层：共享固定 ModelRef，prompt/history 隔离；Leval 根据当前与兼容历史提供结构化反馈，Lopt 输出完整筛选程序；多轮搜索保留历史最佳精确清单并支持稳定边界恢复。
- Leval：可切换 NICE 风格梯度代理或隔离候选 SFT/开发任务评价。候选分支从固定参考开始，不晋级到共享角色或串接其他候选。
- 外层：历史最佳子集 SFT → 任务验证 → 判断；接收后从该权重继续筛选轨迹 SFT → 同协议验证 → 判断。退回保留相应已接受参数、尝试和成本。

接口入口见 [README](../README.md) 的模块表。原生数值后端已用微型 CPU 模型检验；真实模型、数据和环境仍需实际接入。

## 最近验收

0.12.0 本地全量 **686 项测试通过**，新增 32 项定向刷新/外层使用交接检查。交付 checksum 与重载示例见 `runs/focused-refresh/verification.json`；仅 CPU/mock，不是目标模型与真实实验验证。

0.11.0 的本地全量 **654 项测试通过**，共享资产新增 49 项；三类共同 M0 产物供多个配置消费且不新增生产调用的 mock 示例、旧记录/外层兼容、格式/静态检查和源码包验收通过。证据见本地 `runs/shared-assets/verification.json`。本次文档归档和 handoff 模板未修改运行代码，未重跑训练/API/真实环境。

此前 0.10.1 全量 605 项及 60 组独立 NLL/entropy 数学对照等证据仍保留在 `runs/signal-review/verification.json`；历史记录不覆盖。

本次整理将 11 份历史复查报告合并到 HISTORY.md，去掉旧 progress.md 中已经失效的“下一轮”入口，并更新文档链接。源码、605 项回归、必要 fixture 和配置保留。构建副本、缓存和测试临时目录属于生成内容；清理时先保存备份、归档与 checksum。实际清理统计和验证以本地 `runs/repository-cleanup/verification.json` 为准。

## 近期计划

1. 接入实际数据供应、tokenizer/chat template 和 ModelStore，确认来源、监督 mask、模型版本与完整上下文容量。
2. 接入真实任务执行、动作重放与官方/任务自定义评分器，固定开发任务协议和基线；最终测试继续隔离。
3. 填写三个角色的 prompt 正文、内层停止回调和逐阶段接收标准；未配置时继续等待，不代填研究阈值。
4. 如实验需要，补固定参考分布的 NLL/entropy 百分位；不把当前候选批次临时排序当固定 CDF。
5. 完成外层崩溃恢复、全局提交与真实大模型服务器 preflight/smoke，再开展正式研究实验。

本机不下载目标权重、不运行 GPU 或真实任务容器。mock、微型 CPU 和合成 logits 的验收不构成任务能力提升证据。
