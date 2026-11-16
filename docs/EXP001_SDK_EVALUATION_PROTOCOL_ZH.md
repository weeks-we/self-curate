# EXP001当前SDK测评协议

2026-11-15用户纠正后，以此文覆盖旧75轮、128K、贪心以及10000轮工程保护说明。
对齐来源是NICE最新configs/reevaluation.yaml及docs/reevaluation/DEPLOYMENT_SDK.md，不是旧configs/main.yaml。
这里只对齐用户指定的测评协议，不继承NICE的数据、选样、学习率或训练分支。

|设置|Rebench开发题 / Verified|Pro V2|Terminal 4.1|
|---|---|---|---|
|最大SDK轮数|250|250（无统一官方轮数，沿Verified）|250（无统一官方轮数，沿Verified）|
|上下文含输入输出|262144|262144|262144|
|单次输出含thinking|32768|32768|32768|
|整题累计输出|262144|262144|262144|
|agent时间|7200秒|官方50分钟窗口；SDK min(原生时限,2940秒)，留补丁捕获时间|task.toml原生agent.timeout_sec|
|单工具时间|1200秒|1200秒|1200秒；最终仍受该题agent原生时限限制|
|采样|temperature1 / top_p0.95 / top_k20|同左|同左|
|Qwen推理|thinking=true / preserve_thinking=true / effort=xhigh|同左|同左|

min_p=0，presence_penalty=0，repetition_penalty=1；seed42，每题一次采样，不叫greedy，不叫avg@5。
Verified官方grader进程上限仍为3600秒，与7200秒agent解题时间分开。Harbor的172800秒TaskBudget只是外层watchdog，注册须覆盖原生build+agent+verifier，不是给agent两天。

[Pro官方说明](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md)要求50分钟、限制网络及干净环境重判；保留该协议。
[Terminal官方指南](https://www.tbench.ai/run)使用Harbor，保留下载任务的逐题时间/CPU/RAM/GPU要求；官方示例-k5不等于本项目单次结果。本项目250轮和token限制属于统一研究预算，提前触发要报告stop_reason。
[Qwen官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)的Pro采用Claude Code、256K和temperature1/top_p0.95；当前在这些参数上已对齐，任务版本/执行实现/重复次数仍不能冒称官方复现。

## 实际调用与thinking验收

三个task protocol与environment bindings共同决定预算。policy_config把thinking、preserve_thinking、xhigh及采样写入每会话gate。网关将同一受控policy发送到计数和生成接口，经relay白名单头传到服务ASGI ContextVar；vLLM0.22.0的Anthropic _build_base_request在转换时显式填ChatCompletionRequest字段，避免扩展参数被丢弃。
engine native capture记录effective_generation及实际SamplingParams；正式smoke核对thinking历史和模板实际生效，不只看配置布尔值。角色会话也开启thinking/xhigh，但结构化程序生成仍用独立角色采样策略和8192输出预算，不把角色工具禁用会话变成250轮解题。
Claude原生压缩显式设置窗口262144与80%阈值，不再宣称只保留最近5个工具观察；真实compact_boundary和压缩后输入变化须服务器验收。

## 工作站与服务器操作

1. 使用更新的swe/pro/terminal budget及protocol重新注册和publish，旧75轮/10000轮注册不得复用。准备、配置、初测和策略的身份已改变，新建运行根；数据下载与内容一致的Teacher缓存可继续按原身份核验复用。
2. 训练/候选准入的max_context_tokens仍为131072；这是独立SFT设置。推理服务max_model_len从开发题/benchmark预算计算为262144，不得被旧131072覆盖。
3. 每卡4槽，八卡32任务；max_num_seqs4、chunked prefill、max_num_batched_tokens8192。每槽使用不同token前缀完成229376输入+32768输出，禁止同前缀共享缓存掩盖容量。capacity.check_inference传service_settings，报告绑定服务hash和independent-slot-token-v1；旧128K或同前缀回执不能授权。
4. Pro offline checks manifest须提供generation_policy：temperature1、top_p0.95、top_k20、thinking/preserve_thinking=true、reasoning_effort=xhigh、min_p0、presence_penalty0、repetition_penalty1。cli_probe从engine证据逐字段验证effective_generation.*，并验证effective_sampling.temperature/top_p/top_k/min_p/presence_penalty/repetition_penalty及harness_backend。不能自写成功JSON。
5. 新Pro回执schema为pro-v2-claude-sdk-offline-v2，sampling_verified与generation_policy绑定运行时实际配置；旧greedy_sampling_verified回执不能用于新采样。
6. 真实GPU容量、路由、长轨迹FSDP、SDK原生压缩和官方判分由服务器验收；CPU fake类只验证参数路径与规则。不得因4槽262K放不下而静默改上下文，必须在正式实验前显式重冻资源配置。

当前总入口仍为handoffs/EXP001_REBENCH_WORKSTATION_V1.md；执行者先读本页再读旧历史说明。

250轮SDK对齐最终CPU验收：17项参数/转换/预算检查、7项独立前缀及Pro错误回执拒绝、完整EXP001 K4四支三阶段、3×5嵌套恢复和两份ZIP CRC通过。证据runs/sdk250-verification/summary.json；不代表真实vLLM/262K GPU容量或官方判分已通过。
