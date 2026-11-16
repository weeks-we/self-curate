# TEACHER_V1_WORKSTATION

> 2026-11-16：当前默认改为[轻量任务标注](../CODING_TASK_LABELS_ONLY_ZH.md)，该页覆盖下文旧的逐轨迹证据/合批标注及success_covered开发集描述。使用新profile、新准备根；先人工验收task-only试标，不自动全量。

先读AGENTS及docs/TEACHER_ANNOTATION_V1_ZH.md。用户已批准coding-skills-v1，不再等待技能定义，
不要恢复旧七技能或ontology.pending.json。所有准备/API调用在工作站，集群不下载安装。
key只设DEEPSEEK_API_KEY环境变量，不写配置、prompt、日志或Git。

1. 使用锁定来源导出tasks/trajectories，按instance_id建立对应关系。保留source sidecar和checksum。
2. 复制configs/preparation/teacher-pilot.template.json到共享存储，填写试标输入和新输出目录。
   先少量工程样例，再至多100任务+100轨迹。冻结ID、来源、schema/prompt和费用上限。
3. `python -m self_curate.preparation.teacher_pilot --config <pilot.json>`。
   若有锁定Qwen tokenizer，填tokenizer_path及model.json；仅加载tokenizer，不加载训练权重。
   未提供时只验证原文引用，不能把字符长度当真实token或声称M1比例已实测。
4. 检查逐条packet/annotation/provenance/blocked、主技能数量、模型分歧、引用错误和真实usage。
   所有blocked都不得计作exclude_no_skill。只修改工程问题，科研定义变更升版本并开新输出。
5. 确认小试标通过且每类覆盖可行后，使用EXP001 workstation bindings模板的新ontology路径，
   按原tasks/pool/publish阶段运行。固定每主技能50题；不足报告缺口，不复制或看M0成绩挑题。
6. tasks完成保存selected-development.json中的development_asset引用，之后实验直接复用。
   开发题的原始参考材料保留在private文件，solver仅接收公开题面和初始仓库。
7. 准备并验收选中任务环境，再运行pool：隔离、轻量硬筛、Teacher、M1证据/比例、无技能排除审计。
   annotation-summary.json、annotation-decisions.json、每条record/skill-features/Teacher回执均需完整。
8. 检查保留池不少于原始成功数10%的K。若不足停止，不缩分母。Random/static共用该池。
9. 发布prepared manifest和共享资产，再按原EXP001与Claude SDK handoff准备服务器计算。

复用条件是ID、内容、来源、ontology/prompt/schema和模型调用身份一致。未知在途先查平台记录，
不删除dispatched.json强制重发；不能覆盖旧快照。新运行目录保留时间/唯一ID。
正式代码和配置提交由用户操作；不提交数据、API返回、标签产物或凭据。

2026-11-15：主标注与复核均用Flash，分离prompt/会话/缓存，但不是异模型交叉验证。现有历史试标包含旧复核模型，不能冒称新Flash复核已实测；新工作站100+100按更新模板执行。

## 任务关联与批量标注补充（2026-11-15）

先读[任务—轨迹接口](../TASK_TRAJECTORY_LINKS_ZH.md)。保持先选定300题、排除其全部轨迹再处理剩余池；使用更新bindings和pilot的trajectory_grouping。先对同任务2条做组/单条一致性检查，再扩大试标，不能用未实测组模式直接全量收费。任务/轨迹都有独立证据，group失败保留成功成员；检查selected-task-relations、每条task-link、task-relations、grouping-summary与provider usage。m2-v3新增TaskLimitSelect，旧运行不原地恢复；不新增NICE任务分数或自动限制Random。


复核开关更新（2026-11-15）：新模板默认 `review_mode: off`；可选 flagged / sampled / all。关闭不跳过程序校验。请先读 [Teacher 复核策略](../TEACHER_REVIEW_POLICY_ZH.md)，旧的默认复核描述以该页为准。
