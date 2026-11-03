# Self-Curate

Model-driven, feedback-guided data curation across learning stages.

这是独立的新项目，不依赖相邻项目的代码、配置或实验流程。架构依据为用户提供的
[技术蓝图](docs/self_curate_codex_blueprint.md)。目前实现 **M1：数据与标注** 和
**M2：八算子与确定性批次选择**，以及 **P0 契约与 P1 正式交接**（0.5.1）。

最新全模块检查、修复和接口适配矩阵见 [连贯性复查](docs/COHERENCE_REVIEW_ZH.md)。Leval/Lopt 的候选现在能直接通过标准 P1 目录承接下一次外部验证结果。

## 本次可使用的能力

- 规范化文本轨迹与结构化工具调用，校验调用/返回关系。
- 接入显式 fixture 分词器，或外部已计算的真实 token/对齐信息。
- 保留全部上下文，仅监督 assistant 内容与动作，导出训练 labels。
- 计算技能证据占比、密度与绝对量；处理重叠、缺失与零分母。
- 保存质量验证来源和四类使用历史；显式处理超长与已训练数据。
- 预留外部标签与 NLL/entropy 的类型化接入；按模型、token、scope、计算配置与 CDF 保存信号。
- 提供按 ID 保序的内存 MetricView，供 operator 读取一个或多个指标；缺失值留在原位置。
- 保存版本化样本梯度 artifact 引用，记录参数坐标、loss、投影、精度、优化器与配置。
- 外部标签封装绑定数据/来源版本；训练使用状态支持可审计的 pending→accept/reject 更新。
- 发布带 checksum 的数据池，下游读取时重新验证 mask、指标与清单。
- 执行严格 JSON DAG，支持八类算子、系统固定质量门禁与完整轨迹组批。
- 按监督 token 混合分支，保存精确成员、训练顺序、版本和排除原因；缺失或不可行明确返回。

## 本地运行

Python 3.11+。运行代码仅使用标准库；测试需要 pytest。

```powershell
python -m pip install -e '.[dev]'
python -m self_curate --help
python -m self_curate validate-config --config configs/m1.fixture.json
python -m self_curate normalize --config configs/m1.fixture.json --output runs/m1-demo
python -m self_curate inspect-pool --pool runs/m1-demo
python -m pytest -q
```

`runs/m1-demo` 必须是新目录；已有结果不会覆盖。fixture 数据和 token 仅验证工程语义，
不是模型 tokenizer、真实测试报告或研究成绩。`inspect-pool` 可重复执行且不改写产物。

输出为 `trajectories.jsonl`、`excluded.jsonl`、`audit.json`、`manifest.json`。
M2 从 `read_pool()` 读取完整数据与指标，不重新分词或修改原轨迹。

八算子端到端示例：

```powershell
python -m self_curate normalize --config configs/m2.data.fixture.json --output runs/m2-demo/pool
python -m self_curate validate-program --program configs/programs/m2.fixture.json
python -m self_curate select --pool runs/m2-demo/pool --program configs/programs/m2.fixture.json --context configs/m2.fixture.json --output runs/m2-demo/selection
```

12 条合成轨迹中选出 8 条、64 个监督 fixture token，两分支各 32 token。
退出码 0/3/2 分别表示可行、预算或配额不可行、契约或 IO 错误。
选择结果保存在新目录内，输入池不变；详细接口与组批边界见 [M2 接口](docs/m2_interfaces.md)。
M2 最近语义复查为 0.2.1/m2-v2，修正与验证说明见 [M2 第二次复查](docs/M2_REVIEW_ZH.md)。

P0 的任务、rollout、目标梯度、两类报告、成本与轮次记录见 [P0 接口](docs/p0_interfaces.md)。
可验证完整的声明式 mock 契约链：

```powershell
python -m self_curate validate-workflow --workflow configs/p0.fixture.json
```

该命令只验证记录关系；P2 的模拟搜索、训练事件、接受/拒绝和恢复尚未执行。

P1 已提供从数据池直接产生未估值工作流的入口，保留原批次和选择成本：

```powershell
python -m self_curate prepare-selection --pool runs/m2-demo/pool --program configs/programs/m2.fixture.json --context configs/m2.fixture.json --objective configs/p1.objective.fixture.json --backend mock --attempt-id demo-attempt --invocation-id select-001 --output runs/p1-demo/prepared
python -m self_curate inspect-preparation --prepared runs/p1-demo/prepared --pool runs/m2-demo/pool
```

接口与重放约定见 [P1 交接说明](docs/p1_handoff_interfaces.md)，本轮发现与修复见 [P0/P1 适配复查](docs/P0_P1_INTEGRATION_REVIEW_ZH.md)。
最新逐条需求核对与边界修正见 [P0/P1 文档对齐复查](docs/P0_P1_REQUIREMENTS_ALIGNMENT_ZH.md)。

## 接下来怎么模块化开发

先读 [M1 接口与数据格式](docs/m1_interfaces.md)、[实现决策](docs/decisions.md)
和 [交付与近期计划](docs/progress.md)。M2 的依赖分析见 [实现前设计](docs/m2_design.md)，
下一步是 P2 显式 mock 搜索循环，使用本模块的精确批次清单。

外部结果接入示例见 [标注与信号接口](docs/enrichment_interfaces.md)。当前提供存储、校验、查询与
provider 契约，NLL/entropy、CDF 和自动标注算法在后续实现；未计算的值保持缺失。

**operator 应优先使用 [M1 数据取值契约](docs/m1_operator_data_contract.md)**。
这次按 [框架文档](docs/data_centric_rsi_architecture.md) 做了
[三轮复核](docs/M1_THREE_PASS_REVIEW_ZH.md)；筛选算法现由 M2 实现，仍使用 M1 内存指标视图。

无需先搭一整套 Agent 基础框架。每轮实现一个可运行、可测试的垂直模块，
只补齐它依赖的公共契约；确定性选择器稳定后，再接模拟循环、梯度与真实角色。

## Leval / Lopt 本轮交付

已实现外部验证结果到反馈、完整候选程序、P1 重跑和下一次验证请求的单步链路。支持本地规则优化和可注入的共享模型客户端；验证算法与真实 LLM 服务留待后续。接口、命令和近期计划见 [Leval / Lopt 接口](docs/feedback_lopt_interfaces.md)。候选明确标记 `improvement_verified=false`。
