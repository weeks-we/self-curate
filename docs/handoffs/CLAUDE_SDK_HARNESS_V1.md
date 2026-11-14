# CLAUDE_SDK_HARNESS_V1

本交接覆盖所有解题/角色 harness，优先于旧文档中的 mini-SWE 安装和调用说明。
先读 AGENTS、HARNESS_CLAUDE_SDK_ZH.md 和 EXP001_REBENCH_WORKSTATION_V1.md。
不在 NICE 项目改代码，不 commit/push，不下载新基座替代已锁定版本，不启动未授权GPU。

## 工作站准备

1. 新checkout/源码固定commit。复制 configs/experiments/exp001/harness.claude-sdk.template.json
   到共享存储填写。正式 server bindings 的 task_execution.harness 使用同一内容。
2. 构建 docker/Dockerfile.rebench 时另传 CLAUDE_SHA256：官方 Claude 2.1.153 linux-x64 独立发布校验值。
   不得下载后自己算同一文件当可信预期值。SDK0.2.87和vLLM0.22.0固定；完整环境再保存pip freeze/digest。
3. 为每个选定 SWE/Harbor agent 镜像制备 docker/Dockerfile.task-sdk 派生层：

```bash
docker build -f docker/Dockerfile.task-sdk \
  --build-arg TASK_IMAGE="$ORIGINAL_TASK_IMAGE_DIGEST" \
  --build-arg SDK_PYTHON="$SEPARATE_PYTHON_311_OR_NEWER" \
  --build-arg CLAUDE_SHA256="$OFFICIAL_CLAUDE_SHA256" \
  --build-arg TASK_UID="$VERIFIED_WORKSPACE_UID" \
  --build-arg TASK_GID="$VERIFIED_WORKSPACE_GID" \
  -t "$SDK_TASK_IMAGE" .
```

   SDK_PYTHON 必须是独立于 testbed 的解释器；没有则在工作站制备受控运行层，不能升级原测试解释器。
   USER与仓库/日志挂载权限要smoke。容器必须支持原生CLI运行库、Git、Bash、ripgrep；缺项在工作站补。
   将派生镜像推内部registry得到digest；task_images 按原grader镜像digest映射，harbor_task_images
   按完整任务目录 checksum 映射。原 grader/verifier 镜像不装SDK，不混入gold或隐藏测试。
4. 平台准备专用限制网络和relay，只放行模型服务。填写 network、task_user，并保留实际网络验收证据；
   network_policy_verified 仅在真实allow/deny成功后设置true，不把自写布尔值当验收。
5. export SELF_CURATE_HARNESS_CONFIG=/共享目录/harness.json，运行原 environments install。
   它用预构建SDK派生镜像只读验收setup，生成含harness_backend的新installed回执；旧mini回执必须另目录重做。
   Pro继续准备独立verifier-only快照，oracle/nop、依赖和原生判分检查不省略。
6. 运行时锁分属主环境和官方grader环境：主环境可记录 claude_sdk/harbor，Rebench grader只锁swe_rebench；
   不为生成锁而向原testbed安装SDK。Pro真实probe必须记录新harness实际请求采样/路由，
   pro-receipt 新版本 pro-v2-claude-sdk-offline-v1；旧版不能复用。
7. 完整下载SDK/CLI/模型/数据/任务镜像/依赖在工作站完成，封存检查和内部digest。

## 服务器 smoke（正式实验前）

使用Self-Curate的TaskBudget，不继承邻居项目研究参数。通过新build_task_runner/ServerPorts调用：
- 三角色分别一次，核验相同sc-<model_hash>及各自history，禁止外部模型/隐式fallback；
- 一道独立工程SWE任务，真实读/改/测试，捕获patch，再由原官方grader判分；
- 贪心和采样各一轮，核对引擎token、seed和本地usage逐请求一致；replay_from_rollout接NICE反向，不更新参数；
- 缩小工程任务预算触发停止，确认非空patch仍判分，且token计量无未知在途；
- Pro工程任务验证SDK产patch和独立pristine重判，Terminal验证原生环境和资源；
- GPU UUID、长上下文容量、LoRA重新加载与正式预算由服务器preflight验收，不静默降预算。

新服务通过 python -m self_curate.execution.vllm_capture 启动，ServiceManager已接入；
必须保留 SeedMiddleware 和 SELF_CURATE_NATIVE_CAPTURE_ROOT。外部手动服务也要使用同启动方式，
把 native_capture_root 加入serving_pool条目，保证控制进程读到同一目录；只接受可核验共享路径。
流式请求也要逐条验证 request_id→引擎文件，不能只验证API返回200。

## CPU与恢复

保留原嵌套3×5和EXP001 CPU mock命令。它们验证调度连通性，不能替代真实SDK/CLI/模型验收。
角色/任务新增harness身份意味着重新创建run及初测/策略资产，不能改旧snapshot强行resume。
已完成的新run可按原--resume流程核验复用。中断的SDK会话保留日志，不默认复用session_id或自动重发未知调用。
返回环境锁、镜像映射、路由/网络/token/容量smoke证据、官方结果及results.zip。
