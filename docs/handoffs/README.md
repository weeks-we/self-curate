当前 agent harness 已统一迁移到 Claude Agent SDK，见 [接入与验收说明](../HARNESS_CLAUDE_SDK_ZH.md)。

# 实验交接

当前 Rebench EXP001 唯一总入口：[EXP001_REBENCH_WORKSTATION_V1.md](EXP001_REBENCH_WORKSTATION_V1.md)。
按该文档依次调用 [Teacher交接](TEACHER_V1_WORKSTATION.md)、[Claude SDK交接](CLAUDE_SDK_HARNESS_V1.md) 和 FSDP验收。
EXP001_SERVER_V3.md 是旧 SWE-smith 方案，仅供历史恢复；不得启动当前实验。
状态为代码交接；真实来源、任务环境、GPU容量和吞吐仍需工作站/集群验收。

每个新逻辑实验新增一份版本化 handoff，使用 [TEMPLATE.md](TEMPLATE.md)，调用已有模块。
恢复原运行沿用原 handoff、源码、环境、配置和 run_id；改变协议时新建版本与运行。

## 必须固定

目标/范围、代码入口、来源与环境、授权资源、预算/seed/停止判断、资产身份、输出目录、
恢复条件、失败处理和结果位置。缺科学定义不能临时猜测；缺权限或容量则报告阻塞。

## 执行顺序

1. 读 AGENTS、指定 handoff 和配置，核验固定源码及执行授权。
2. 联网准备依赖、模型、数据和任务镜像；GPU 阶段不临时补大文件。
3. 通过服务器 preflight/smoke，冻结来源、环境、配置、handoff hash 和 run_id。
4. 执行控制代码，保留失败/费用/状态；不重复未知在途调用或模型失败。
5. 校验完成摘要、逐题结果和 ZIP，返回结果路径，不 commit/push/merge。

状态写输出目录，不修改仓库 handoff 记进度。旧作业运行中不更新其源码或环境；结束保存后
再更新。并行或旧版恢复使用固定 commit 的隔离目录。数据、权重、缓存、凭据和结果不进 Git。
