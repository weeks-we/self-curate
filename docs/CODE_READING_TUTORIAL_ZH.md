# 正式实验运行链路

从 [V3 Handoff](handoffs/EXP001_SERVER_V3.md) 和 [服务器部署](SERVER_CONTAINER_DEPLOYMENT_ZH.md) 开始。
旧的依赖测试目录和模块 demo 的开发教程已移除。以下对应实际运行代码。

1. `server/bootstrap.py`：解析计划，核对来源与资源，连接数据准备、preflight 和实验入口。
2. `server/assets.py`、`preparation/`：CPU 联网下载、任务镜像准备、准入/隔离、技能证据和数据清单。
3. `experiments/exp001/initial.py`：共同 M0 测评、全池指标与一次固定策略；`assets/` 负责按身份复用。
4. `operators/`、`data/`、`contracts/`：读取标签、确定性筛选、精确批次与版本校验。
5. `experiments/exp001/runner.py`、`training/continuous.py`：四个独立分支、连续三波、checkpoint 和恢复。
6. `execution/`、`validation/`：模型服务、工具/容器执行、官方判分、任务结果及费用。
7. `experiments/exp001/delivery.py`：检查完成状态、导出和验证结果 ZIP。

运行观察：`executions/<run_id>/progress` 与 services 保存阶段进度和服务利用率，
`runs/<run_id>` 保存训练/逐题结果，`runs/deliverables/<run_id>/results.zip` 用于交付。

EXP001 关闭 Leval、刷新和接收/退回；这些通用模块仍服务后续实验，接口见
[内层搜索](INNER_SEARCH_ZH.md)、[Leval](LEVAL_BACKENDS_ZH.md)、[通用外层](OUTER_TRAINING_ZH.md)。
业务任务验证和运行时校验属于运行依赖，不因名字含 validate/check 而删除。
