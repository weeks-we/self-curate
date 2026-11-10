# 内外层嵌套 CPU 联调交接

目的：在服务器验证同一份 self-curate 源码能够完成外层 3 轮、每轮内层 5 次候选评估。只使用合成数据与显式 mock，不做正式实验，不占 GPU。通过只说明本次覆盖的模块连接、状态传递和落盘可运行，不代表真实模型、NICE 有效性或容器环境已验收。

## 给执行 agent 的任务

在 self-curate 仓库执行以下命令。不要在 NICE 项目中执行；不要调用正式 EXP001 server 配置。不要下载模型/数据、调用外部 API、启动 Docker、安装 GPU 依赖或修改正在运行的 NICE 实验。代码有问题先保存错误与输出，不改评分、决策或验收要求来凑通过。

使用 Python 3.11+。本入口仅用标准库，不需要 pytest、torch、numpy 或真实权重。若服务器当前 Python 已满足要求，可直接运行；需要隔离时创建独立 venv，不变更 NICE 环境。源码通过用户推送后获取，不需要本机 runs 归档。

Linux，从仓库根目录：

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN="runs/nested-cpu-$(date +%Y%m%d-%H%M%S)"
python -m self_curate.smoke --config configs/smoke/nested_cpu_mock.json --output "$RUN"
python -m self_curate.smoke --output "$RUN" --verify-only
```

两条命令均应退出 0，最后 JSON 中 status=passed、backend=mock。新运行拒绝覆盖已有目录；失败后保留原目录，修复问题后换一个新的 RUN。`--verify-only` 只核对已完成证据及 ZIP，不继续训练，不表示外层断点续跑。通用 OuterCycle 的跨进程恢复仍未实现。

## 实际覆盖的流程

64 条合成轨迹，两个合成技能。第一轮全池初始化 NLL/entropy；第二轮按显式技能方向、当前模型版本和实际外层训练曝光刷新；第三轮模型因第二轮拒绝而不变，复用匹配版本的缓存，同时排除被拒绝分支实际训练过的数据。

每轮执行初始候选加 4 次修改，共 5 次评估，不是 5 次修改。每轮在第 2 次评估后保存和恢复 InnerSearch，再继续到第 5 次。内层分数是固定 mock 序列，第二次最佳、最后一次不是最佳，用来验收历史最优批次交接。

三角色使用同一 mock ModelRef；每个角色有固定独立 mock prompt 和私有历史。跨外层轮次创建新会话，旧模型 history 保持原命名空间，通过显式的角色私有摘要引用及共享外层反馈进入下一轮，不把旧 history 冒充新模型历史。真实 prompt 不受影响。

| 外层轮次 | 子集阶段 | 轨迹阶段 | 下一轮模型 |
|---|---|---|---|
| 1 | 训练、验证、接受 | 接着子集权重训练、验证、接受 | 轨迹阶段模型 |
| 2 | 训练、验证、证据不足；追加验证后拒绝 | 不执行 | 第一轮已接受模型 |
| 3 | 训练、验证、接受 | 接着子集权重训练、验证、拒绝 | 第三轮子集阶段模型 |

轨迹语料由 mock provider 提供两枚监督 token，只测试接口；不宣称已经实现真实筛选轨迹总结。mock 指标计算后走现有 SignalService/M1 回填与 FocusedRefreshService；不是实际模型 logits。Leval 分数和接受策略同样为明确的验收情景。

## 自动验收要求

- 3 轮、15 次内层候选评估，45 次 mock 角色调用（含每轮三角色各一次跨轮背景输入）。
- 5 次 mock 训练调用，初始验证加各阶段验证及追加验证共 7 次任务评估，每次 3 题×2 seeds，总计 42 次 mock rollout。
- 各轮外层训练成员、顺序与历史最优 BatchManifest 完全一致；轨迹训练从该轮子集权重继续。
- 接受和拒绝保留正确模型；证据不足只追加验证，不重复训练。
- 模型、prompt、history 命名空间一致；每次输入只引用该角色自己的历史。
- 首轮全量初始化；后续只在显式方向刷新外层未使用样本；保留旧版本指标；拒绝分支的实际训练曝光不丢失。
- 同一源码运行期间保持一致；所有证据落盘并进行 SHA256 校验；ZIP 内容逐字节与输出一致。

这些要求由入口自动检查。任何失败退出 2，不报告 passed。默认约一分钟左右，以实际 CPU/磁盘情况为准；不把 mock ETA 当真实实验耗时估计。

## 回传内容

成功后回传 `$RUN.results.zip`，以及最后 JSON 和执行的 Git commit ID。ZIP 包括 config、environment、源码 checksum、summary、rounds.csv、三轮角色输入、内层快照/best、刷新输入输出、外层训练/验证/决策和曝光历史。不包含真实模型、真实数据或凭据。

快速看 `$RUN/summary.json` 和 `$RUN/rounds.csv`；逐轮证据在 `round-01/` 至 `round-03/`。失败时回传完整终端错误与该 RUN 目录，注明哪条命令退出非零。不要把 mock 分数发成正式实验成绩。

本机验收记录放在被 Git 忽略的 runs 下，不随源码推送。服务器 agent 必须重新执行上述命令，不能依赖本机归档或结果包代替验收。
