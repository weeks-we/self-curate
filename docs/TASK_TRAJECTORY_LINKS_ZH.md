# 任务—轨迹关联与同任务分组标注

2026-11-15用户批准。六技能定义不变；传输封装新增teacher-trajectory-group-v1，算子版本升级m2-v3。NICE估值仍按具体轨迹和原模型/上下文绑定，不引入任务统一分数。

## 正式流程

锁定任务/轨迹来源 → 按instance_id核对仓库/可用base_commit → 抽候选任务 → Teacher看题目/参考补丁/测试补丁及同ID辅助成功轨迹 → 每主技能50题冻结300互异开发题 → 排除这些题对应的全部训练轨迹、来源家族及近重复 → 轻量硬筛 → 剩余轨迹按task_key小组标注 → 逐轨迹证据校验和M1发布。

只排除最终选中的开发题对应轨迹，不因题目曾进入候选标注列表就排除全部候选。正式流程只处理原始成功轨迹；开发题关联记录可包含同ID全部源轨迹，隔离不只针对选作helper的一条。

## 身份与保存

task_key由task_source_id、task_source_revision、instance_id、repository、base_commit计算；另外保存完整task_content_hash。不能仅用跨来源的裸instance_id混合V1/V2或不同版本任务。
每条M1保持data_id=trajectory_id；source.record_id仍是instance_id，新关系字段通过版本绑定MetadataRecord/DataTag保存，核心数据schema不变。task-link.json可直接检查；标签/证据仍逐条保存，不继承任务技能。
selected-development.json和development-asset的selection.json保存task_links，selected-task-relations.json提供便于查看的副本。池阶段task-relations.json只是来源关联证据，不是持久化operator查询数据库。
旧开发资产没有新关系字段时不原地改写；必要时在新准备根重新发布，内容相同的Teacher请求缓存可以复用。旧m2-v2执行上下文不能冒充新m2-v3恢复。

## Teacher分组

正式准备与teacher_pilot使用相同bounded_packet_groups规划器，默认enabled=true、最多2条，同一任务材料只提供一次。3条可显式配置但仍须先试标。
分组同时受max_input_bytes=600000、每成员输出预留6000与公共输出预留2048限制，实际provider仍有独立输入/输出/费用保护。超组预算拆成单条，不截断完整轨迹；单条本身超provider容量会阻塞。
trajectory-group.md是可编辑封装prompt；trajectory.md仍定义每条技能标准。批量返回items，每项携带原trajectory_id和完整独立annotation；request_id、instance_id、ontology_version也逐项原样核对。消息ID只属于该成员，不能引用兄弟轨迹；任务gold/test材料只用于理解目标，不能成为动作或结果证据。
主返回与复核都用Flash。只复核抽检和争议成员，复核输入为该成员完整packet。组内部分失败先保存其他已通过结果；恢复复用已完成成员和精确group请求缓存，不整批重付费。API/上下文/schema/证据错误均不算无技能排除。重复/未知成员ID属于歧义响应，整组阻塞，保留原始回复。
每条都有独立annotation-result引用、Teacher annotation、状态和M1 span/share/density；不能按行位置拼接，也不能把任务技能复制给所有轨迹。
grouping-summary.json记逻辑组数/轨迹数；真实调用和费用以provider原始request/response/usage为准。合批减少重复材料和请求数，不保证token或费用按成员数等比例降低。真实Flash合批语义仍须小规模试标；CPU合成成功不代表模型标注正确。

## 查询与策略接口

```python
from self_curate.data import MetricView
view = MetricView(records)
siblings = view.task_relations.related(data_id)  # 包含自身，返回轨迹ID
subset = view.task_relations.related(data_id, within=requested_ids)  # 保留请求顺序
columns = view.static_batch(requested_ids, ('task_key', 'instance_id'))
coverage = view.task_relations.summary(requested_ids)
```

关系视图O(N)构建一次、字典查key/同任务成员；重新构造快照后重建。旧记录没有显式task_key时使用source_id/revision/record_id隔离的legacy-source-scoped身份，不跨来源猜关联。
拿到siblings后由调用方按各data_id及原ModelRef/估值上下文查询NLL、entropy、梯度或NICE结果；本模块不合并、不补零、不传播分数。未来可基于这些关系设计策略，本次不指定NICE任务级策略。

## TaskLimitSelect

```json
{"id":"task_cap","op":"TaskLimitSelect","inputs":["ranked_pool"],
 "params":{"max_per_task":2,"minimum_size":3216,"representative_rule":"first"}}
```

输入已按想要的指标排序时，first保留每任务前两条，并保留输入相对顺序；seeded按固定程序seed挑代表，输出仍保留原输入顺序。minimum_size是显式可行性下限，不改变原始10%预算。不足返回TASK_LIMIT_SHORTFALL，不放宽、不补抽、不复制。
再接SampleSelect选固定K。算子可用但不强制所有实验应用：Random仍均匀抽样，固定K不变。Lopt获得全池/分技能的唯一任务数、每任务轨迹数直方图和cap1/2/3可容纳数量，再决定是否用此算子。
MixAllocate仍是终端算子；要约束混合后的全局重复度，应在所有分支共用的祖先上限额。分别在各分支限额不能保证最终全局限额，不改变原DAG接口来掩盖这一点。

## 文件

- data/task_links.py：版本身份和内存关系视图。
- preparation/teacher_grouped.py：完整输入分组、批量调用、逐成员校验/复核/恢复。
- preparation/rebench.py、rebench_workstation.py、teacher_pilot.py：正式准备和试标接入。
- contracts/program.py、operators/handlers.py：TaskLimitSelect契约与执行。
- experiments/exp001/policy_input.py：向Lopt提供关系覆盖统计和算子用法。

任务关联交付CPU验收：31项既有Teacher契约、27项关联/分组/乱序ID匹配/部分恢复/预算拆组/引用隔离/TaskLimitSelect检查、300题资产零调用复用、EXP001四支三阶段K4及完成后恢复、3×5嵌套跨进程恢复和两份ZIP CRC通过。证据runs/task-links-verification-20261115/summary.json。本轮新增付费API为0；真实合批语义一致性和节费幅度仍待小规模API试标。
