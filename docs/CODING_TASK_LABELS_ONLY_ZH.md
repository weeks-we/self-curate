# Coding 轻量任务标注（当前默认）

2026-11-16，替代EXP001原逐轨迹证据标注方案。六技能定义不变，标签解释改为task_required。

1. 读取成功轨迹，按instance_id关联任务并核验repo/base_commit；成功轨迹必须能找到任务。
2. tasks阶段先对每个成功任务只标一次，材料为题面、参考补丁、测试补丁；不发送完整轨迹。保存training-task-labels。官方发布口径32,161成功轨迹对应3,792任务，正式仍按导出计数。
3. 从没有成功轨迹覆盖的任务中抽开发候选，完成同一轻量标注后每主技能选50题，冻结300题和引用；保持来源家族/近重复隔离。
4. pool阶段轻量硬筛轨迹，再继承各自任务标签；同任务共享标签，轨迹ID、质量、长度、NLL/entropy及使用记录仍各自独立。
5. 任务被可靠判定uncovered、所有技能not_required且无风险/缺失上下文时可排除对应轨迹；不确定/失败不算无技能。

新profile：configs/preparation/coding-task-labels.v1.json；prompt：teacher-v1/task-labels-only.md；返回schema_version=teacher-task-labels-v1。旧coding-skills.v1证据模式保留用于历史兼容，不再是EXP001默认。

Teacher默认Flash、thinking开启/effort low、最大输出4096、review_mode off。单任务输出短JSON，保留各技能状态、主技能和简短理由；不返回技能evidence字段。少量试标确认4096及low不会导致截断/准确率问题后再扩大，不静默截断或自动增加预算。

M1 v4 metadata含skill_label_basis=task_required和skill_status:<id>；正式数据删除annotations、annotation_record、evidence_metrics和stats.evidence_union_tokens，不写skill-features.json，改写task-skills.json。SkillSelect/配额/暴露统计使用任务技能。EvidenceSelect和evidence_batch公开入口已删除，operator升级m2-v4；旧程序不得当新程序恢复。内部历史类型与解码兼容仍保留用于旧资产读取，不会向v4发布证据。

任务标签逐题保存packet、response路径/hash、任务内容hash、prompt/ontology及模型身份，精确缓存重复运行不发请求。变更profile/prompt/输入后使用新准备目录；旧运行、标签和资产不原地修改。正式调用预算按唯一任务而非轨迹数。任务材料和参考解只留准备侧，不泄露给解题agent。

试标模板trajectory_limit=0，最多100任务，结束后ready_for_full_annotation=false，不自动进入正式阶段。先人工检查技能和预算；正式tasks入口仍是独立命令，目前没有签字式代码放行回执，不得让工作站agent跳过人工验收。推荐先10–20题，确认后再扩大。

流程入口仍是docs/handoffs/EXP001_REBENCH_WORKSTATION_V1.md；工作站必须使用新profile和新准备根。General只记录在GENERAL_IF_PLAN_ZH.md，尚不执行。


CPU验收：40任务/41轨迹仅40次合成调用，重复零新增；v4逐条JSON往返、标签查询及旧算子拒绝通过。旧数据域EXP001 K4/恢复、3×5内外层及独立v4继承标签完整EXP001均通过。未执行真实Flash新prompt语义或GPU验收。
