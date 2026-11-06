# Self-Curate

Model-driven, feedback-guided data curation across learning stages.

独立的数据筛选与自改进研究项目，当前代码版本 **0.10.1**。内层使用固定模型执行筛选、数值评价和反馈优化；外层消费历史最佳的精确样本清单，进行子集 SFT、验证与判断，再从接收后的参数继续筛选轨迹 SFT、验证与判断。

## 从哪里开始

先看 [当前进度](docs/progress.md)，再按 [代码阅读教程](docs/CODE_READING_TUTORIAL_ZH.md) 理解数据流。研究定义见 [原始蓝图](docs/self_curate_codex_blueprint.md)、[数据中心架构](docs/data_centric_rsi_architecture.md) 和 [实现决策](docs/decisions.md)。

历史复查与旧进度已合并到 [HISTORY.md](docs/HISTORY.md)。其中的“下一步”和“未实现”指各报告当时的状态；当前模块状态以下表和 progress.md 为准。

## 模块与接口

| 模块 | 当前职责 | 接口说明 |
|---|---|---|
| M1 数据 | 完整轨迹、assistant mask、质量/技能/使用历史、版本化指标与 ID 保序查询 | [数据格式](docs/m1_interfaces.md)、[算子取值](docs/m1_operator_data_contract.md)、[外部标注](docs/enrichment_interfaces.md) |
| NLL / entropy | 实际冻结前向或外部 raw logits，按 assistant/技能范围计算并回填新数据池 | [计算与回填](docs/SIGNAL_COMPUTATION_ZH.md) |
| M2 / P1 | 八类筛选算子、固定质量门禁、确定性组批与精确清单交接 | [算子接口](docs/m2_interfaces.md)、[P1 交接](docs/p1_handoff_interfaces.md) |
| P0 契约 | 模型、任务、梯度引用、评价、成本和轮次身份核验 | [P0 接口](docs/p0_interfaces.md) |
| 三角色 / 反馈 | 同模型、独立 prompt/history；Leval 数值事实传给 Lopt | [角色接口](docs/THREE_ROLE_AGENTS_ZH.md)、[反馈接口](docs/feedback_lopt_interfaces.md) |
| 内层搜索 | 候选循环、历史最佳、成本与检查点；停止条件由调用方配置 | [内层接口](docs/INNER_SEARCH_ZH.md) |
| Leval | 可切换 NICE 梯度代理或隔离候选 SFT/开发任务评价 | [两套后端](docs/LEVAL_BACKENDS_ZH.md) |
| 外层训练 / 验证 | 两阶段 SFT、任务执行接口与逐阶段接收/退回 | [外层接口](docs/OUTER_TRAINING_ZH.md) |

数据、模型、tokenizer 和任务执行器由调用方通过接口提供。具体 prompt 正文、研究停止条件和接收阈值仍待配置；不默认接收候选。最终测试集不进入筛选或优化。

## 本地运行

Python 3.11+。数据契约与基础 CLI 只使用标准库；指标计算需要 NumPy，实际 Torch 后端与相关专项需要 PyTorch。

```powershell
python -m pip install -e ".[dev,signals,training]"
python -m self_curate --help
python -m pytest -q
```

显式合成示例，输出必须使用新目录：

```powershell
python -m self_curate normalize --config configs/m2.data.fixture.json --output runs/m2-demo/pool
python -m self_curate select --pool runs/m2-demo/pool --program configs/programs/m2.fixture.json --context configs/m2.fixture.json --output runs/m2-demo/selection
python -m self_curate signals-demo --output runs/signals-demo
python -m self_curate outer-demo --output runs/outer-demo
```

完整交接与 API 用法见上表对应文档。示例使用 fixture 数据、合成 logits 或 mock 任务结果；专项测试另有微型 CPU 模型真实前向/梯度/SFT。它们验证工程语义，不代表目标模型的任务能力或 GPU 容量已通过验收。

## 提交前检查

源码在 `src/`，回归测试与必要 fixture 在 `tests/`，可复现实例和角色配置在 `configs/`，架构与接口在 `docs/`。这些都属于项目所需内容。新增的 signals 包、数据池发布模块及其测试也应加入 Git。

`runs/`、`build/`、`dist/`、`*.egg-info/`、缓存、数据、模型、运行状态与凭据由 `.gitignore` 排除，不上传。正式验收产物在本地 runs/ 中保留；历史测试临时文件可以归档后清理。删除已跟踪的历史文档会在 Git 中显示为删除，其内容已归入 HISTORY.md。

由用户检查暂存内容后自行提交和推送：

```powershell
git add -A
git diff --cached --stat
git diff --cached --check
git status --short
```

`git add -A` 同时记录新增、修改和删除。测试通过说明本地工程回归通过；远端认证、分支保护或并发提交仍可能影响后续 push。
