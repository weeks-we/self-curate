> EXP001当前预算、thinking及采样证据以 [最新SDK测评协议](EXP001_SDK_EVALUATION_PROTOCOL_ZH.md) 为准，旧greedy/v1回执不授权新正式实验。

# Claude Code harness：角色、自由执行与测评

更新：2026-11-14。独立 Self-Curate 项目。参考 NICE-Subset-Validation 当前
`sdk_agent.py` / `reevaluation_gateway.py`，移植独立实现，不在运行时 import 邻居仓库。
正式交接见 [harness 交接](handoffs/CLAUDE_SDK_HARNESS_V1.md)。

## 边界

workflow 是确定性的模块调度：角色 → 算子 → 估值 → 反馈 → 训练 → 测评。
模型需要对话状态、工具、执行反馈时由 harness 组织。
三角色仍由原 RoleSession 管理各自的 prompt/history，不让 Claude Code 接管外层调度。
它们当前输出结构化 JSON，SDK tools=[]；Lexec 监督的算子仍在 Python runtime 中执行。
这不等于给三个角色任意 Bash 权限，也不是 Claude Code 的三个子 agent。
任务解题会话允许 Bash/Read/Edit/Write/Glob/Grep，禁止额外子 agent 和 Web 工具。
全部主请求和辅助模型别名锁到 AdapterRouter 当前同一 adapter，模型仍是 Qwen。
Teacher 是准备阶段独立外部 API，不随本次 harness 迁移变成 target agent。

## 调用路径

| 场景 | 新执行层 | 保留的确定性部分 |
|---|---|---|
| Lopt/Lexec/Leval | agents/claude_client.py → execution/claude_sdk.py | 角色上下文、JSON 校验、算子、数值反馈 |
| Rebench/Verified/Smith 开发测评、NICE rollout | CodingTaskRunner → claude_task.py → 容器 claude_worker.py | TaskRunRequest/TaskRunResult、官方 SWEJudge |
| Pro V2 | ProV2Judge → Harbor ClaudeSDKAgent | 原 pristine patch_replay 与官方 verifier |
| Terminal | HarborJudge → Harbor ClaudeSDKAgent | 原生任务容器、服务/GPU资源和 verifier |
| NICE 梯度 | vllm_capture.py → valuation/claude_replay.py | NICEReference、TorchNICEBackend；不执行 optimizer.step |

新正式默认不再调用 mini-SWE。`execution/agent.py` 是保留的旧实现，生产 builder 不使用；
runner 的 mini 注入参数暂保留为兼容既有调用方的名称，默认函数已改为 run_claude。
不改已有记录来冒充新 harness 跑过。

## 版本与上下文

- Claude Agent SDK 0.2.87，Claude Code 2.1.153；显式 cli_path，禁自动更新和云端 fallback。
- vLLM 0.22.0。SDK 不直接调用 OpenAI chat API，走本地 Anthropic Messages 接口。
- Claude 原生上下文管理，不再叠加旧的“最近五条工具观察”截断。
- 轮数/墙钟/累计输出/每次输出/上下文继续读取当前 Self-Curate 配置，不继承 NICE 的250轮预算。
- 网关强制模型和采样参数，并记录实际请求。Qwen thinking 参数在网关设置，不假设 Claude effort 自动映射。
- 贪心与随机 rollout 都可声明。Messages API 本身无 seed 字段，故用专用 header + SeedMiddleware
  把固定 seed 送到引擎 SamplingParams；第 n 次请求 seed=episode_seed+n。不能声称固定 seed 保证GPU完全确定。

## NICE 的原始 token

SDK 日志的文本、工具 JSON、thinking 块不是原始生成 token。
新的服务启动器在 AsyncLLM.generate 边界保留真正 prompt_token_ids/generated_token_ids、adapter、seed、request_id。
每个网关请求通过独立 request ID 关联对应引擎文件；缺失或 token 数不一致立即停止。
native-capture.json 绑定 ModelRef；replay_from_rollout(rollout) 输出现有 ReplayTrajectory：
每次真实输入全部 mask=-100，后接真实生成 token，只监督这些生成位置。
包含模型生成的 thinking/tool 参数及终止位置；并非仅对 SDK 显示的工具参数求梯度。
NICEReference 继续核对生成位置总数，不能拿新文本重新 tokenize 代替。
仅估值反向需要这条链路。普通 benchmark 不因“SDK能跑”就宣称 NICE 反向已验收。
vLLM 原始 token 采集钩子、流式/非流式与自定义模型必须服务器实测；本机仅测合成引擎对象。

## 运行隔离与日志

解题 SDK 仅在预构建任务派生镜像中运行。原 grader 镜像、base_commit、testbed 解释器不变。
不挂模型目录、宿主 HOME、Docker socket 或参考答案。任务只访问受控本地推理 relay。
独立 Docker 网络名称不等于限制外网：平台仍需实际防火墙/NetworkPolicy，冻结前做 allow/deny smoke。
新会话不 resume 其他角色；角色历史由上层显式传入。日志 exclusive creation 防止覆盖。
sdk-events.jsonl / sdk-result.json / inference-usage.json / request-*.json / response-*.raw /
native-capture.json / patch.json 与官方判分结果分开保存。
凭据通过私密短期配置传入，读取后删除，不写 command.json 或普通运行回执。
预算耗尽时仍提取补丁交官方判分；基础设施失败保留日志，不能当作普通0分重试择优。
未知在途 usage 要先对账，不能按0记录并重发。

## 旧资产与接口

SharedModelClient.generate_json、ModelRef、TaskRunRequest/Result、NICEReference、TrainingRequest 未变。
新增 deployment harness 配置、SDK镜像映射、Pro回执版本和NICE replay helper。
旧 mini 安装回执不能授权新SDK；Pro回执升级为 pro-v2-claude-sdk-offline-v1。
旧开发测评、旧 NICE 目标和首次筛选策略不能跨 harness 直接复用；新运行重新发布。
原始任务/训练轨迹/锁定权重可以校验复用；前向NLL/entropy只有其完整计算身份相同才能复用。
源码变化后的旧实验恢复按原规则拒绝，另开新 run，不改旧 snapshot 消除校验。

## 本地与服务器验收

本机只运行显式 CPU 替身检查和既有嵌套/EXP001 mock；不执行真实SDK二进制、GPU、目标权重或Docker。
真实验收必须覆盖：SDK版本/CLI checksum、三角色路由及会话隔离、SWE工具编辑、
原始token逐请求对应与NICE反向、预算停止后patch判分、Pro干净重判、Terminal原生任务资源、
网络隔离与断网加载。任何一项未过，不发布真实实验通过结论。

参考：
- https://github.com/weeks-we/NICE-Subset-Validation
- https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.87/src/claude_agent_sdk/types.py
- https://docs.vllm.ai/en/v0.22.0/serving/integrations/claude_code/


## 二次复查补充（2026-11-14）

原始生成结束与网关usage落盘存在时序差：现在SDK返回后等待handler提交计量再读取，不误判完整请求。
vLLM启动作用域错误已修复，并实际执行了替身启动入口回归；非法seed在进入推理前返回400。
更新模型后的角色可用 `ClaudeSharedRoleClient(..., checkpoint=accepted_checkpoint)`；
由原AdapterRouter校验它与ModelRef一致，不再只能通过省略checkpoint绑定M0。
Pro短期访问令牌只经子进程环境和私密传输提供，不出现在发布的trial JSON。
真实SDK类型/初始化/钩子/解析已用SDK0.2.87+合成transport验证；三次DS成功短调用仅验API协议。
真实CLI与本地Qwen、NICE反向、官方容器判分仍按交接进行服务器验收。
