# 两遍逻辑与接口复查：0.8.1

2026-11-05。对照 AGENTS、原蓝图、既有 P0/P1 和本次用户确认的两阶段流程检查。
原蓝图单阶段描述与新增筛选轨迹训练的差异已由用户要求覆盖；没有改写原文。

## 第一遍：契约、状态与执行边界

| 发现 | 影响 | 修复 |
|---|---|---|
| 补证判断后再次 run 会重新调用判断器；新验证能继续沿用旧补证决定 | 可能未经显式新证据推进，或多次复测 | 已判断的相同上下文等待；新报告须有自己的补证决定才能复测 |
| TaskEvaluation 只核对套件元数据，缺少逐次请求／返回绑定 | protocol 或 checkpoint 字段可能被替换而未报错 | 保存完整 execution_requests/results，核对模型、任务、seed、协议、checkpoint、结果、费用与 attempt |
| 外层仅信任回执内部自洽，未完整比对原训练请求及验证 checkpoint | 可消费另一份学习率配方或另一份权重的有效回执 | 外层比对精确请求和权重引用；不匹配立即 blocked，已执行成本保留 |
| 第二个任务的 mkdir／请求写盘在异常保护外 | 前一任务已发生的费用可能未传给外层 | 派发前写盘纳入保护；保留已完成事件，不给未派发任务造一次模型费用 |
| 训练／验证后端返回错误类型时先报错、后读取 usage | 已报告的合法费用可能丢失 | 先保留有效 ResourceUsage，再拒绝坏结果，仍然记录失败 |
| TrainingOutcome 解码只检查回执字段相互一致，未复核计划预算 | 可整体改写处理量或 optimizer steps，绕过执行服务已有核对 | 构造／JSON 边界也核对 epochs、监督 token、累积窗口及精确步数 |
| 接续 checkpoint 未核对 requires_grad 参数集合 | 相同 ModelRef 下可能意外解冻基座 | 与上一 checkpoint 保存的 trainable_parameters 核对，范围变化在更新前拒绝 |

另外补齐 DecisionContext 的 attempt 绑定，防止来自不同外层尝试的记录混用。
第一批 10 项案例在旧实现均失败，其中 9 项直接复现旧边界问题、1 项定义旧报告迁移接口。
第二遍又实际复现了接续参数范围变化和两种坏返回 usage 丢失问题；失败日志随交付保留。新增专项共 18 项。

## 第二遍：完整链路与历史产物

复查路线为：M1 记录／mask → P1 精确清单 → 内层历史 best → 子集 SFT → 任务验证 → 判断 →
筛选轨迹 SFT → 同协议任务验证 → 再判断。没有改变原清单、重抽训练数据或填入研究阈值。

专项覆盖等待状态不重复派发、两处退回、精确 recipe/checkpoint、费用保留、协议和 reward 文件改写、
严格 JSON 往返、旧报告补齐、CPU 梯度参考、实际 checkpoint 接续、adapter 基座冻结与参数范围漂移。

### 旧 0.8.0 报告怎么接入

旧 TaskEvaluation JSON 仍可只读解码，原 CapabilityReport、rollout 和成本保持原值。
缺少逐次请求／返回时 `protocol_verified=False`，不能直接作为接收或 baseline 证据。
旧运行目录实际已保存这些文件，因此可只读补齐：

```python
from self_curate.orchestration import load_task_evaluation

report = load_task_evaluation(
    old_validation_directory,
    expected_model=system_saved_model,
    expected_suite=system_saved_task_suite,
    expected_checkpoint=system_saved_checkpoint,
)
```

loader 将目录名、请求／结果、report、rollout、费用与调用方锚点一起核对。
返回新的不可变组合对象，不改写旧 evaluation.json 或任何历史文件。
新目录的报告自身也保留执行证据，文件与报告不一致会拒绝。

`protocol_verified` 只表示所提供记录的执行协议绑定自洽，不证明远程模型或环境真实执行。
真实模型／任务适配器仍负责来源和实际计算的核验。

## 保留的边界

- 内层退出条件和外层接收标准仍待填；未默认接收或安装数字阈值。
- 真实模型加载、任务执行／判分、自动筛选轨迹生产和数值梯度仍由外部模块接入。
- attempt 内 accepted 指针与全局提交分开；完整 M9 崩溃恢复没有实现。
- ModelStore 仍必须隔离可变参数；接续范围检查不能证明自定义 loader 没有共享底层 tensor。
- 无 GPU／目标模型／真实 benchmark 执行，mock 数值不代表效果提升。

最终全量测试、两遍专项、原 0.8.0 报告只读兼容、mock 闭环和历史证据保护见
`runs/outer-two-pass-review/verification.json`。
