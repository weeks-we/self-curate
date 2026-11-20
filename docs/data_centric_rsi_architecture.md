# 数据策划框架：普通任务适配

数据准备 → 原生标签/版本指标 M1 → Lopt 提出程序 → Lexec 监督确定性算子执行 → Leval 解释开发集反馈 → 内层继续搜索 → 固定子集 SFT → 开发集验证 → 下一轮。

这是通用框架。**EXP001 只调用数据准备、独立分支 SFT、最终 benchmark**，不经过 agent 搜索。EXP002 才接入 V0/V1/V3；后续控制轨迹学习不等于当前已实现其科研定义。

- 三角色 harness：smolagents ToolCallingAgent；相同共享参数，各自 prompt/history。
- 工作流：Python 状态机，负责依赖、版本绑定、恢复、日志；它自身不需要 harness。
- 筛选执行：注册算子与 M1 ID/版本化列；不让模型直接改样本集合绕过记录。
- 普通题测评：直接生成答案；数学/IF checker、代码隔离测试。不存在 SWE 仓库状态和交互工具环境。
- 训练：HF Trainer/Accelerate FSDP2/PEFT LoRA，原生 optimizer/RNG checkpoint；不以 adapter 权重加载替代训练恢复。
- 资产：固定来源 SHA，共享只读原始资产；每次实验独立 plan、run_id、state、结果目录。

接口保持兼容不代表原来的长轨迹数据后端适用于新基准。新适配的具体边界、未完成的服务器验收见 `progress.md` 与 `HANDOFF.md`。
