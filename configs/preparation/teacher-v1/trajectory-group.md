# 同任务多轨迹标注封装 · v1

批量模式：输入的 task 和 ontology 由同一任务的若干轨迹共享，成员 packet 中未重复这些字段。
对每个成员分别返回完整 teacher-trajectory-v1 annotation，保留其 request_id、instance_id、trajectory_id。
不能给整个任务一个共用技能标签；成功与否、质量及展示技能均逐条判断。
每个成员的 message_id 命名空间独立，只能引用该成员 trajectory 的分析/动作/观察。
任务/gold/test材料仅帮助理解目标，不构成任何成员的实际动作证据；不能借用其他成员的证据。
输出严格为 {schema_version:teacher-trajectory-group-v1,group_id:原值,items:[
 {trajectory_id:原值,annotation:完整annotation或null,error:null或具体无法标注原因}]}。
每个请求成员恰好一项。无法标注的成员返回error，不要把失败或缺材料标成exclude_no_skill。
