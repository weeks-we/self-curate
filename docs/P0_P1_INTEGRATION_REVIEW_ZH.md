# P0 引入后的适配复查与 P1 完整交接

版本：0.4.0；P0 校验语义 p0-v2；M2 算子仍为 m2-v2。复查开始于 2026-11-02 23:35（北京时间），跨日交付。
本轮范围依据蓝图 P0/P1：从接口、实现、调用方到测试和发布证据；不进入 P2 状态机或真实模型实验。

## 发现与处理

| 发现 | 影响 | 修正 |
|---|---|---|
| 工作流仅检查包内模型/目标/上下文的自洽 | 修改目标及全部引用后，内部仍可能合法；消费方还需与系统固定配置核对 | validate_workflow/load_workflow 增加 expected_model、expected_objective、expected_selection_context_hash；回执分别说明是否进行了外部锚点核对 |
| M2 与 P0 各自拼接程序/执行 hash | 后续修改可能只更新一边，造成跨模块失配 | 共用 selection_identity，保持既有 hash 公式不变，旧 m2-v2 批次可精确复用 |
| CostLedger.reference 每次扫描所有事件建索引 | 多 rollout/成本引用时产生不必要的反复扫描 | 构造一次只读内存索引；不进入 JSON、record_hash 或解析字段，不新增持久化数据库 |
| 模型计算成本事件可省略 model_hash | 成本引用无法可靠归属 checkpoint | 除 selection 外的模型操作成本必须声明模型身份；原 fixture 已满足，历史证据保留 |
| real 域的 mock URI 检查只匹配小写前缀 | 大小写或空白可掩盖显式 mock 标记 | 对 fixture/mock 前缀统一规范化检查；仍不冒称核验了实际文件 |
| 能力 task_mean 每个任务重复扫描 rollout | 报告较大时增加校验开销 | 按 task_id 一次分组，聚合口径不变 |
| RoundRecord 的决策名与 M1 历史状态名不同 | 直接传 accept/reject 会与 accepted/rejected 不匹配，证据不足也不应变成拒绝 | 提供 training_use_decision 显式映射，needs_more_evidence 保持 pending；不自动更新曝光 |
| P1 只有裸选择输出，P0 完整样例需要手工构造 | 后续模块容易重建批次或误填未执行结果 | 新增 prepare-selection：从真实执行的算子结果生成只有精确批次和本次成本的未估值工作流；下游字段保持为空 |

这里的“外部锚点”是调用方提供的系统配置，不是输入包自带字段的另一份拷贝。P2/P5 的工具层应掌管这些值和路径，提案角色只能修改 ProgramSpec。

## P1 验收对照

| 蓝图要求 | 当前交付 |
|---|---|
| M1 规范化、证据、质量、使用历史 | 保留既有实现与回归检查 |
| 八算子 + 固定 QualityGate/BudgetPack | 使用原 M2 执行器，不重复建设 |
| 预算、重叠、缺失、顺序和 seed | 既有独立参考/回归测试继续执行，配合新交接场景 |
| 精确 BatchManifest | P1 输出与 P0 工作流中的 batch 为同一清单，训练顺序/哈希不重抽 |
| 不可行和缺分数不伪造批次 | 保存结果、请求、已完成 trace 和选择成本；无 workflow/batch 文件 |
| 相同输入与 seed 重放 | 原 select 保持确定性；prepare 的 batch/outcome 可精确重放，实际耗时与调用成本另记 |
| 后续按需梯度和估值 | P1 不调用 provider、不生成梯度/估值；验证后追加声明式 mock 下游记录仍使用同一批次 |

## 本轮检查的模块连接

1. **M1→M2**：ID 保序、缺失原位、内容/token/标签版本、固定 ModelRef/spec/CDF；无持久化 operator 索引。
2. **M2→P0**：共享执行身份公式；prepare 的可行批次直接进入 WorkflowBundle；已完成 trace 与成本都有引用。
3. **P1→M3**：MISSING_SIGNAL 请求保留 metric/scale、数据 IDs、模型、spec、CDF；P1 不调用前向。新模型的旧分数不能替代缺失新分数。
4. **P1→M5/M6**：样本特征按完整 cache_key 传递，目标按 target_version；追加特征不改变选定训练 token/mask。P0 继续核对坐标、计算覆盖和固定目标。
5. **P1→M8/P2**：既定 training_order 与 training_data_hash 保留；接受/拒绝和训练曝露仍由之后的实际执行事件驱动。
6. **P0→成本/恢复**：attempt_id 标识外层尝试，invocation_id 标识一次物理选择调用；恢复应读取已发布产物，不能重跑同一调用冒充恢复。

## 后续接口的明确使用约定

- WorkflowBundle 描述一个候选/选中批次的交接；搜索历史用多个不可变记录保存，不向一个 bundle 塞入所有历史模型和所有缓存特征。
- P1 选择结果的原始 context/snapshot 用于选择重放。M5 追加特征后，使用 load_workflow(view=...) 消费原清单；重新选择则显式创建新 ExecutionContext。
- 运行时应传系统的 expected_model/objective/context_hash。只读历史文件可仅作内部关系检查，回执会显示外部锚点尚未核对。
- M1 的 AnnotationProvider/SignalProvider/GradientProvider 与 make/attach/get 接口保持不变。raw scoring 与固定 CDF 的请求消费服务由 M3 接入；本轮不把它们标作已计算。
- 实际训练记账时，TrainingUse.model_id 使用起始 ModelRef.model_hash；receipt 应是稳定的训练尝试/任务身份，不能用会随状态变化的整个 TrainingRecord.record_hash 替代。RoundRecord.training_use_decision 将 accept/reject 映射为 accepted/rejected，pending/needs_more_evidence 映射为 pending。终态通过 M1 原有更新接口记录。P1 不写训练历史。
- 真实来源、任务家族与近似题面的隔离、向量内容、任务环境、模型权重、验收阈值和提交行为仍需后续模块核验。schema 的 split/版本检查不能代替这些实际检查。

## 交付与证据

新增 API/CLI 见 [p1_handoff_interfaces.md](p1_handoff_interfaces.md)。全量测试、解析变异、原数据只读、2000 条规模场景、独立 wheel 与压缩包检验见 `runs/p01-integration/verification.json`。
旧 M1/M2/P0 验收目录和交付包保留；不通过改写旧快照修复新协议。没有执行 Git 提交、推送或合并。

本轮可交付边界是 P0/P1 的本地确定性数据与交接链路；完整模拟搜索、接受提交与恢复属于 P2，尚未执行。
