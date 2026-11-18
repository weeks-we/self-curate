# SDK 修复与服务器验收交接（2026-11-18）

本轮针对 self-curate。NICE 交接包和 AGENT_REPAIR_HANDOFF 是问题依据，不继承其五模型/600题、路径或GPU安排。禁止热改正在运行的目录。新源码、新冻结身份、新输出根；旧证据只读，不编辑旧snapshot来恢复。

## 修改落点

| 项目 | 实现与行为 |
|---|---|
| 答案先封存再清理 | execution/claude_task.py、sealed.py：AGENT_SEALED 固定必需文件、request hash、补丁与完整日志hash；cleanup.json独立，清理失败不覆盖答案。runner遇pending或未知清理状态阻塞，不再调用模型。清理未知时须先核实自有容器，不把阻塞当正常完成。 |
| SWE判分恢复 | graders.py、phases.py：判分独立持久阶段，最多初次+2次；失败和成功尝试全部保留。成功零分直接复用；已有成功结果验证全文件集合及checksum。缺结束记录的尝试属于状态不明，拒绝自动重试。 |
| 完成结果恢复 | sealed.py、runner.py：RESULT_SEALED全目录清单；恢复验证真实TaskRunResult契约，不以JSON可读替代校验；删除/添加/篡改证据拒绝。旧无seal结果不补造。 |
| Pro生命周期 | pro_v2.py：agent-phase与regrade-phase分开。agent阶段禁止自动重放，regrade最多额外2次；官方零分保留。重判分复用已封存agent_job/patch，继续原官方pristine replay。CLI退出码不能单独推翻存在且有效的官方结果。 |
| Terminal生命周期 | graders.py：原生trial整体持久封存，已完成结果复用；失败/不明确状态不自动再运行模型。不能从丢失的工作区凭空恢复；版本未证明支持verifier-only重入时明确阻塞，不调用整题resume冒充只判分。此分支没有实现任意Terminal任务的跨进程verifier-only恢复；真实工作区恢复能力需要与冻结Harbor版本进一步适配。 |
| 原生压缩时序 | claude_sdk.py、claude_gateway.py：事件monotonic、gateway session_id、请求序号及起止；context_acceptance.py要求先长后短、下降至少1000、原生boundary之后有工具返回；拒绝递增/错序/跨session。context_probe.run_probe是server-only持久ClaudeSDKClient入口，两个warmup→完整pressure_prompt→工具续行，不手工删历史，不写正式acceptance。 |
| 网络故障 | supervised_http.py、claude_gateway.py：count_tokens及生成走可取消缓冲读取；socket监督覆盖无响应头/无响应体。正常慢输出保留，不用轮询时间缩短研究预算；半截SSE保留incomplete_usage，不补零。DNS/connect、TLS以及SDK工具子进程的完整Linux取消验收仍是服务器要求。 |
| 成本 | phases.py、runner.py：每段独立started及finished文件，UTC时间戳和进程内perf_counter；lifecycle-costs.json列阶段、活动耗时、失败成本、墙钟跨度、不完整段和时钟异常，不推算旧缺失成本。 |
| 容量可观测性 | server/capacity.py：每槽started/finished/failure，失败不删除其他槽成功证据，并abort barrier避免无谓等待。不缩并发、上下文或输出预算。 |

SDK/CLI/vLLM版本、模型、prompt、采样和研究预算未因修复更换。原Docker隔离和任务镜像映射沿用。本项目 atomic_json 已有UUID临时文件，不复制NICE同名.tmp逻辑。NICE特定overrides/FLA不能未经本项目数值验收直接复制。

## 本地已执行

临时检查放在NICE工作区被忽略的 runs/sdk-repair-20261118/，没有重新建立项目tests目录。28项专项检查通过：封存/损坏拒绝6项；持久重试/阶段成本/压缩时序9项；真实run_claude替身的清理失败4项；网关完整/慢速/半截/静默SSE4项；最终TaskRunResult封存与篡改2项；容量失败证据3项。

过程中曾发现HTTP/1.0响应提前close socket、仅shutdown取消不及时，均修复后复测；失败记录不当作通过。EXP001初次从错误工作目录启动的相对prompt路径错误已保留，改从项目根新目录运行后通过。

三外层×五内层：runs/sdk-repair-20261118-complete-nested，通过及verify-only打包。EXP001 Random/static四分支三阶段：runs/sdk-repair-20261118-exp001-v2，通过并打包。mock不能验证真实SDK、GPU、Docker判分。没有执行已删除的全量pytest。

## 服务器必须验收

1. 冻结真实依赖与源码。单服务跑SDK→工作镜像→非空/空patch→官方grader，确认cleanup故障后不第二次模型作答。Pro重判分失败恢复时记录模型调用数不变；Terminal失败不得直接重放原生trial。
2. 调用execution.context_probe.run_probe，提供原完整压力prompt、真实binding/gate和minimum_long_input；结果仅diagnostic，不替代正式容量报告。要求同一个SDK会话、真实工具结果及完整usage。工程probe不改正式worker。
3. Linux上补充真实SDK取消、进程树/连接/GPU槽释放、TLS、DNS/connect及cancel/final-usage竞态检查。发现状态不明则停止派发并保留现场，不解除锁伪装释放。
4. 原预算全并发容量、HF/vLLM数值对照及原官方判分，记录每槽真实usage/错误。未测试的不写PASS，不修改已冻结运行层来绕过容量或数值问题。
5. 源码变化不能直接复用旧输出。跨source迁移不在本轮实现；需要另做只读来源链验证，不能改旧hash。

本地交付仍有明确的代码适配缺口：Terminal任意任务的跨进程verifier-only恢复及其冻结版本绑定，尚未实现；其余真实平台项目尚待服务器验收。不能将本文件解释为全部原始要求已完成。


最终源码验收：28项CPU原始日志和JSON见 runs/sdk-repair-20261118-cpu-evidence-v2/test_results.json。新跑 runs/sdk-repair-20261118-final-nested 与 runs/sdk-repair-20261118-final-exp001，均通过；随后verify-only/--resume通过，两份ZIP的testzip均返回None。首次CPU矩阵夹具取消发生在count_tokens阶段，不能当作生成中断，保留失败回执后以generation-entered事件同步取消时机重测。Terminal实际Harbor commit已向用户询问；未取得前不声称原生跨进程verifier-only已完成。
