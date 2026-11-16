# Teacher 六技能标注与可复用资产

用户于2026-11-14批准六技能及三类标注prompt。正式配置：
`configs/preparation/coding-skills.v1.json`，三份prompt/schema在同目录teacher-v1/。
Teacher直接调用外部API，在工作站执行，不使用解题harness，也不加载训练权重。

## 数据流与职责

锁定任务/成功轨迹 → 同instance_id匹配 → 候选任务Teacher标注 → 主技能各50题 →
冻结任务资产 → 排除全部对应轨迹与来源家族/近似题面 → 轻量硬筛 → 轨迹Teacher标注 →
排除完整审阅后无技能倾向的样本 → M1证据与占比 → 发布实验输入。

主技能：behavior_state / data_representation / interface_integration / platform_build /
resource_concurrency / algorithm_math。完整定义、正反例和消歧规则在ontology中。
每题保留全部次技能；开发主分组只按一个明确primary，每组50、六组共300个不同ID。
composite、uncovered、不确定或有未消除题目风险的任务不强行凑主分组。
成功轨迹覆盖的任务是默认抽样总体；不能据此宣称代表全部Rebench任务分布。
候选采用固定seed的洗牌前缀，补候选时新建准备身份、复用相同请求缓存，不按模型成绩补题。

## 三份prompt与接口

- task.md：任务要求的技能，材料含题面、gold/test patch；同ID成功轨迹可辅助理解。
- trajectory.md：该轨迹实际展示的技能，必须引用assistant动作与结果，不复制任务标签。
- review.md：独立复核，输出approve/revise/needs_context/reject_annotation；修订须为完整原schema对象。

每个请求携带冻结ontology、正式output_schema、数据身份和source_catalog。
复核另外携带原标注规则、原schema及程序错误；corrected_annotation保留原请求ID。
证据使用 source / message_id / field(JSON Pointer) / quote，要求在指定字段中逐字唯一定位。
API返回JSON文本，由程序校验保存；不让Teacher生成token下标或比例。
任务参考答案只在准备/判分侧，solver-tasks.json不含参考答案，不能给Lopt或解题模型泄漏。

## M1写入与比例

正式prepare_records使用锁定的真实fast tokenizer/chat template定位全局半开token区间。
只将assistant_refs映射到监督位置；context/outcome原文和引用完整留在teacher-annotation.json。
AnnotationRecord绑定data_id、content、source、tokenizer/template，原始response和完整标注作为provenance。
MetadataRecord附skill_status标签，算子沿用现有ID查询，无新增持久化operator索引。

每条保留轨迹有record.json和skill-features.json：
- supervised_evidence_tokens：技能证据与assistant mask相交、去重并集后的数量；
- share：该数量/全部技能监督证据并集；
- density：该数量/整条轨迹assistant监督token；
- complete_events：该技能已记录事件数。

重复片段不重复计数；跨技能重叠保留，因此share之和可能超过1。
uncertain对应spans=null、指标null；not_observed对应空证据，存在其他证据时其share=0。
不把定性的confidence冒充数值概率，不给证据token额外训练权重。

## 无技能倾向与失败

完整审阅、六类全部not_observed、无缺失上下文时，Teacher返回exclude_no_skill和原因。
程序校验一致后排除，记录trajectory_id、instance_id、原始标注/复核证明与原因。
多项技能没有主次不构成no_skill；只要有demonstrated就retain。
缺材料、API错误、输入超限、截断、引用错误或无法复核属于blocked，保留证据，不作为无技能删除。
未知在途不自动重发；已完成请求按精确hash缓存恢复。

训练K仍为floor(硬筛前原始成功轨迹数/10)，不因Teacher排除再缩小分母。
Random/static使用同一隔离/准入/技能可标注池；Random不按具体技能或指标选样。
最终池不足K则停止并保留审计，不放宽规则、不复制样本。

## 开发任务资产复用

tasks阶段发布development-asset/，含：
manifest与COMMITTED、原始任务快照、solver可见题目、技能分组、任务标签与证据、ontology和checksums。
selected-development.json保存development_asset ArtifactRef(uri+sha256)。
后续准备可在bindings.development_asset填写这份引用，直接校验复用，不重抽、不重标。
加载核验文件、任务ID、ontology与来源身份，以及与新benchmark的隔离关系；篡改明确拒绝。
这是题单和引用资产，任务环境/官方判分smoke仍是独立工作站步骤。

## 小规模试标与预算

入口：`python -m self_curate.preparation.teacher_pilot --config <pilot.json>`。
模板teacher-pilot.template.json为最多100题+100轨迹，先小样例再扩；不自动全量运行。
有本地锁定tokenizer_path/model引用时计算M1证据和比例；否则仅验证原文引用，
明确标token_metrics=awaiting_locked_tokenizer，不发布伪造token统计。
公开样例试标不代表全池技能分布；100+100后再检查每类是否可组成50题及标签混淆。

主模型deepseek-flash，复核deepseek-flash；显式thinking enabled/high、输出16384、等待900秒。
有独立max_calls、max_input_bytes和人民币预算上限。价格上限使用冻结的高峰未缓存价，
已知usage按保守价估计，未知调用保留最坏预留；不是平台真实账单。
缓存命中不重发请求；不记录key。超字节保护阻塞，不静默截断。
默认5%固定抽检加争议复核；有技能uncertain、低confidence或程序校验问题进入复核。
复核调用与人民币上限耗尽会停止，不能静默略过待复核样本。

## 核心文件

teacher.py：packet/schema/引用/复核/M1转换。
provider.py：API、缓存、输入/调用/费用保护和未知在途。
rebench.py：任务与轨迹标注、无技能排除、M1和审计。
development_asset.py：不可变开发题单与复用校验。
rebench_workstation.py：tasks/pool/publish阶段，来源锁和配置生成。
teacher_pilot.py：独立小规模API试标，不训练、不运行Docker。


比例解释：share/density 衡量的是 Teacher 引用证据覆盖的监督 token，不是样本全部内容的真实潜在技能占比。证据可跨技能重叠，share 之和可能超过 1；同技能重叠区间先取并集。


Teacher V1 本地交付验收（2026-11-14）：31 项临时 CPU 检查通过；330 条模拟任务标注选出 300 道互异题，跨准备目录复用为 0 次 Teacher 调用。真实 API 小试完成 3 条任务、2 条完整轨迹的 schema 和逐字证据校验；早期失败保留。最终源码下 3×5 内外层 mock、EXP001 Random/static mock、恢复与 ZIP CRC 通过。真实 Qwen tokenizer 尚未接入本次小试，不能声称真实 token 比例已验收；正式 100+100 试标和语义人工抽检待工作站执行。验收记录：runs/teacher-v1-verification-20261114/summary.json。

2026-11-15：主标注与复核均用Flash，分离prompt/会话/缓存，但不是异模型交叉验证。现有历史试标包含旧复核模型，不能冒称新Flash复核已实测；新工作站100+100按更新模板执行。

2026-11-15新增：[任务—轨迹关联、分组Teacher及TaskLimitSelect](TASK_TRAJECTORY_LINKS_ZH.md)。M1核心schema不变，关系为版本绑定tags；算子语义版本m2-v3，旧m2-v2上下文不能直接重放。


2026-11-15：复核改为显式可选，新模板默认关闭。配置与费用说明见 [Teacher 复核策略](TEACHER_REVIEW_POLICY_ZH.md)。旧默认抽检描述由该页替代。
