# Self-Curate

Model-driven, feedback-guided data curation across learning stages.

这是独立的新项目，不依赖相邻项目的代码、配置或实验流程。架构依据为用户提供的
[技术蓝图](docs/self_curate_codex_blueprint.md)。本次实现其第 6 节 **M1：数据与标注**，
同时补齐 M1 所需的最小包、配置、CLI 和测试；没有提前实现 Agent、梯度或训练模块。

## 本次可使用的能力

- 规范化文本轨迹与结构化工具调用，校验调用/返回关系。
- 接入显式 fixture 分词器，或外部已计算的真实 token/对齐信息。
- 保留全部上下文，仅监督 assistant 内容与动作，导出训练 labels。
- 计算技能证据占比、密度与绝对量；处理重叠、缺失与零分母。
- 保存质量验证来源和四类使用历史；显式处理超长与已训练数据。
- 发布带 checksum 的数据池，下游读取时重新验证 mask、指标与清单。

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
下一模块从 `read_pool()` 读取完整数据与指标，不重新分词或修改原轨迹。

## 接下来怎么模块化开发

先读 [M1 接口与数据格式](docs/m1_interfaces.md)、[实现决策](docs/decisions.md)
和 [交付与近期计划](docs/progress.md)。下一步是 M2/P1 算子执行器。

无需先搭一整套 Agent 基础框架。每轮实现一个可运行、可测试的垂直模块，
只补齐它依赖的公共契约；确定性选择器稳定后，再接模拟循环、梯度与真实角色。
