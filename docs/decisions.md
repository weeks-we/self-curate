> Latest training/evaluation protocol: [3 epochs, save and evaluate every epoch checkpoint](THREE_EPOCH_PROTOCOL_ZH.md). 18 training branches, 54 epoch checkpoints, 228 benchmark results; all prior 1-epoch/final-only estimates are superseded. Remaining hyperparameters are unapproved.

> Latest TACO split policy: development tasks come from train only, 50 per skill; test is reserved exclusively for final evaluation. Historical test-as-development instructions below are superseded.

> Latest math source: NuminaMath-1.5, hierarchical source/problem_type proportional sample of 50,000. See [current sampling protocol](MATH50K_STRATIFIED_ZH.md). The earlier external20k/enrichment description below is superseded.

> Current update: see [EXP001 update](EXP001_UPDATE_20261120_ZH.md). It supersedes the earlier math pool, random ratios, and development split timing below. Random now includes 30% and 50% (21 branches / 18 SFT / 84 evaluations). Math uses the fixed 20k question pool but needs reference-solution enrichment. Development splits are published AFTER EXP001; they are not independent validation for already exposed EXP001 checkpoints. Hyperparameters remain unapproved.

# 当前决策

2026-11-20，用户授权普通任务迁移，取代旧 SWE 交互轨迹实验配置与交接。

- 模型：Qwen3.5-9B 为主，Qwen3.5-4B、Llama-3.1-8B-Instruct 扩展；关闭 thinking。未引入所谓 Qwen3.5-8B。
- 第一池：TACO、NuminaMath-1.5、Tulu-3 Personas IF；利用真实原生标签，不自动付费 teacher。
- harness：smolagents ToolCallingAgent；RoleRuntime/三个角色协议保留，业务 prompt 单独配置。
- EXP001：三领域，各 initial/full/random5/random10/random20；12 次独立训练，不通过 agent/NICE，不按测试分数选权重。
- EXP002/003/004 各自保留独立阶段边界；未来统一域内比例待确认，不伪造 baseline/控制学习实现。
- HF 当前版本采用 FSDP2 原生参数；LoRA 目标补充 Qwen3.5 线性注意力投影。batch/LR/长度等仅草案，正式启动前确认并实测。
- 工作站联网准备、集群离线计算；正式代码判分要求隔离 CPU 容器。集群 Docker 调用能力尚需核实。
- 数据、权重、缓存、生成结果、私人会议文档不提交 Git；用户自行 commit/push。

已有 M1/P0/P1、operators/signals/inner/outer 通用接口继续保留。旧 SWE 运行代码作为未激活兼容实现暂留，旧配置/方案从当前树清除；备份在忽略的 runs 中，不能作为执行入口。

2026-11-20：取消Random50，统一3epoch；新增原生checkpoint训练曝光/有效输入token/监督token/墙钟计量，8rank求和并绑定恢复。测评单列生成与CPU判分计量；自动输出summary/training_costs/scores_wide/REPORT。NICE重做指导书是忽略的个人交接，使用用户确认Qwen3.5-9B，不修改邻居NICE代码。

2026-11-21：benchmark只测最终epoch3，保存3份训练成本，EXP001为72结果，B短训12结果。官方审计修正IFBench loose主分、Multi-IF逐轮四指标均值、BCB调用完整官方evaluate并使用冻结本地题单。训练保留global32、启用fused AdamW/8对齐/固定global吞吐预检与FSDP2少重分片；真实GPU与容器待服务器验证。

2026-11-21新增用户授权：NICE并入EXP001，20×4/3000/top-bottom1000、两档共享LR、1epoch、42/84/125保存。参考旧NICE r16/alpha32/batch8；通用SFT仍r32/alpha64/global32/3epoch。新增真实HF回放/独立GPU梯度、官方TACO沙箱适配、四分支全链路及B梯度probe凭据。A执行完整实验，B仅环境/短训和NICE梯度检查。主LR草案1e-5/2e-5统一配置，未批准。

2026-11-21最终配置核查：[当前配置与待填项](CONFIGURATION_AUDIT_20261121_ZH.md)。修正roadmap Random30/NICE、NICE进度协议元数据和配置一致性校验；当前仍为未批准草案，真实容量/镜像/凭据由B/A验收。


2026-11-21：用户授权按经验选择EXP001/NICE专用配方：LR5e-5/1e-4，两路各4卡FSDP2；global32/8不变；异步CPU判分、LoRA热切与owned进程取消。三个base镜像锁linux/amd64 digest。CPU33项smoke、22项全矩阵/恢复、26项配置检查和真实微型CPU反向通过；真实GPU/Docker仍需B验收。详细依据和全参数见EXP001_QWEN35_9B_RECIPE_AND_ENV_ZH.md。

2026-11-21用户更新：EXP001三个epoch checkpoint均测benchmark，epoch3主报告；不改变训练超参或NICE节点。常规372＋NICE60＝432测评记录。

2026-11-21长度与预算核查：真实Qwen tokenizer测TACO/IF全池与Numina2000窗口样本；8192先保留，prepare新增长度排除前精确分布。12个benchmark分项预算，区分官方runner默认、模型示例和项目fallback；真实生成检查每轮全历史窗口。缺Numina监督文本明确排除。见TRAINING_LENGTH_AND_BENCHMARK_BUDGETS_ZH.md。

2026-11-21再次配置核对：18项必需HF来源有URL/revision。修复finalize对三epoch矩阵的数量适配；修复原生IF约束空格／别名、复合标签与未知标签显式复核。IF29980全量映射九类、无缺标；math probability／IF style及少量数学Other补标仍需口径确定，尚未修改taxonomy提案。见EXP001_READINESS_CHECKLIST_ZH.md。

2026-11-21用户最终标签口径：保留预设Coding8／数学7／IF10，未映射先不补标。配置native_only、teacher禁用；finalize把未映射排除出按技能开发集并保留训练池，零覆盖技能保留、50题不足仍显式报告。9项专项CPU检查通过；对应关系见NATIVE_SKILL_MAPPING_ZH.md。

2026-11-21用户纠正部署：A与B异地；A独立从零准备工作站／集群环境及本机验收，B只提前测试／反馈。重写A完整交接，移除跨地点共享资产／B凭据前置要求；实际验收门继续保留但由运行地点自身生成，接口字段不变。
