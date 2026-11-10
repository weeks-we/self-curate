# SFT：8 卡 FSDP、断点恢复与日志交接

本交接用于真实训练后端验收，与 `NESTED_CPU_MOCK_HANDOFF.md` 的逻辑联调分开。CPU mock 通过不表示真实 FSDP、27B 长上下文或显存已经验收。不要在正在跑 NICE 的 GPU 上启动本任务。

## 实现与资源安排

- 正式 EXP001 SFT 使用 `training/fsdp.py` → `torchrun` 8 个 rank → `training/fsdp_worker.py`。Accelerate 1.15.0 调用 PyTorch FSDP1 FULL_SHARD，分片参数、梯度与优化器；PEFT 官方 `fsdp_auto_wrap_policy` 隔离冻结和可训练模块。没有自行实现 all-reduce、参数分片或优化器状态转换。
- 精确数据窗口、仅 assistant payload 的 token 平均损失、阶段发布属于本项目逻辑。仍有这一层训练循环，未使用 HF Trainer；框架层由 Accelerate 执行 `prepare/backward/save_state/load_state`。不能描述成“完全没有训练循环”。
- 维持 BF16 LoRA。冻结底座，沿用固定初始 adapter、学习率、seed 和一次数据曝光，不因为换 FSDP 改科研参数。启用 CPU RAM efficient loading；默认不进行 CPU activation offload，以减少 PCIe 开销。现有梯度 checkpointing 和精确分块输出损失保留。
- 8×H200（用户说明约 141GB/卡），一个训练分支同时使用全部 8 卡；4 个研究分支依次训练。训练前关闭本项目管理的推理服务，分支训练结束释放显存后重启服务，用全部卡验证三个已发布节点。不会修改其他任务或 NICE 的进程。
- global batch=8：8 个 rank × micro batch 1 × 累积 1；recipe.accumulation_steps 在项目旧契约中表示全局优化器窗口大小，不能再乘 8。每条完整长轨迹单独前向，不截断。尾窗口只计真实样本，空 rank 用零权重占位保持 collective 一致，不计重复曝光。
- 三阶段分别在实际数据波次边界结束优化器窗口，步数仍为各波次 ceil(样本数/8) 之和。不是一边用 FSDP 一边启动四个单卡训练 worker。

## 配置位置

正式模板 `configs/experiments/exp001/server.template.json`：

```json
{
  "training_backend": "accelerate_fsdp",
  "fsdp": {
    "world_size": 8,
    "micro_batch_size": 1,
    "save_every_steps": 4,
    "mixed_precision": "bf16",
    "fsdp_version": 1,
    "keep_recovery_checkpoints": 2,
    "cpu_ram_efficient_loading": true
  }
}
```

以上片段属于 adapter_settings。必须填入 8 个授权 GPU UUID、固定来源、模型及服务配置。不要直接执行未填模板。运行镜像通过 `.[server]` 安装 Accelerate；在联网构建阶段完成依赖安装并保存 pip freeze，GPU 执行阶段不得临时升级。

## 先做服务器真实验收

1. 在空闲授权卡上检查实际容量、NVLink/NCCL、主机 RAM、共享 checkpoint 文件系统和本地模型/adapter。FSDP 不是序列并行，131072 长上下文的激活仍可能超显存，不能仅凭参数分片宣称能跑。
2. 选少量完整真实轨迹执行短训练与 adapter 重新加载，验证仅 LoRA 变化、训练监督范围不变、8 个 rank 实际启动、阶段 adapter 可以被任务服务读取。
3. 做真实的强制中断恢复对照：保存后终止训练进程，再用同一输出目录 `--resume`；确认从最近已提交优化器边界继续，数据游标、优化器、RNG 和阶段清单一致。先完成这个小任务，再跑正式预算。
4. 完成正式最大上下文容量验收，记录实际每卡显存、利用率、tokens/s 和 checkpoint 开销。若 OOM 不得静默截断、减上下文、换 batch 或改学习率；保存证据并报告。micro batch 的修改必须重新验收和冻结配置。

本机已进行微型 CPU Accelerate 的真实中断/恢复对照，权重逐张量相等；这不能代替上述 8 卡验收。不同硬件上的浮点结果不承诺逐 bit 相同。

## 新运行与恢复命令

从仓库根目录，使用构建好的服务器环境和已解析的正式配置：

```bash
python -m self_curate.experiments.exp001 \
  --config /absolute/resolved-server.json \
  --output-root /persistent/rsi/runs \
  --assets /persistent/rsi/shared-assets \
  --package-results
```

自动分配 `/persistent/rsi/runs/YYYYMMDD-HHMMSS-10位hash/`，立即打印 run_directory 和 controller attempt_log。把这个原始目录记下来。

中断后：

```bash
python -m self_curate.experiments.exp001 \
  --config /absolute/resolved-server.json \
  --output /persistent/rsi/runs/原来的时间编号 \
  --assets /persistent/rsi/shared-assets \
  --resume --package-results
```

恢复不能用 --output-root 新建目录，不能更新正在运行的源码，不改旧请求、GPU 拓扑、环境版本或 snapshot。框架恢复模型、优化器、各 rank RNG/混合精度状态；项目恢复精确数据游标和阶段记录。若保存途中中断，仅认有 COMMITTED.json 且 checksum 完整的状态，未提交目录保留。

默认每 4 个优化器步骤保存，并在每个阶段结束保存。最坏需要重放最近未提交的窗口；重放不增加最终声明的数据预算，但物理花费和未知中断区间需单独核算。SIGKILL/掉电后无法精确知道的 token/时间不能补为 0。已有 EXP001 的未知中断曝光和成本诊断继续保留。

保留最近两份完整恢复状态；删除旧大分片前保存原 checksum 清单到 pruned-recovery。阶段 adapter 不跟随恢复状态清理。FSDP 恢复状态包含冻结模型分片，开销显著：建议为整个四分支运行预留至少 1TB 可用 checkpoint 空间，按实际 bytes 校核；每次保存前还检查磁盘余额。恢复状态不是可回传的小 adapter。

## 日志与结果

```text
runs/
  时间编号/                         # 一个研究运行，恢复不改编号
    progress/                      # 控制进度与 ETA
    branches/分支名/training/checkpoints/ # 实际 backend 工作目录，以当前输出结构为准
      job.json                     # 冻结训练请求和身份，含数据，不回传
      launch-logs/时间编号.log      # torchrun 全部 stdout/stderr
      logs/时间编号/rank-00/events.jsonl
      logs/时间编号/rank-01/events.jsonl
      ...
      latest.json                  # 最近完整恢复状态指针
      recovery/时间编号/            # Accelerate 状态、游标及 checksum
      stages/1/weights/            # 可用于任务验证的阶段 adapter
      stages/2/weights/
      stages/3/weights/
      complete/result.json
  controller-logs/运行编号/时间编号/rank-00/events.jsonl
  deliverables/运行编号/results.zip
```

每 rank 日志记录 UTC 时间、attempt ID、PID、版本、设备与容量、全局 batch、数据游标、优化器 step、输入 token、监督 token、累计值、loss、学习率、step 时间、token/s、ETA、CUDA 已分配/保留/峰值显存、NVML GPU 利用率、checkpoint 时间和失败原因。NVML 不可用时明确记缺失与原因。gpu_seconds 不用 wall×卡数冒充内核实测；未知保留 null。完整阶段验证、模型调用输入/输出 token 和环境执行时间继续使用原有 ledger/任务证据。

结果 ZIP 排除 job.json、恢复权重分片、adapter 权重和原始训练数据；保留训练/任务日志、指标、决策和证据索引。原始权重和恢复状态留在持久盘。controller-logs 位于运行目录旁边，可单独回传排障；不要把正在写入的日志加进已封存的 completion checksum。

## 边界

通用 `TrainingService.train(request, output, resume=True)` 也支持完成结果复用及可恢复后端继续；使用非初始模型时 FSDP 后端要显式传入已接受 starting_checkpoint。当前 FSDP 后端只支持一遍训练，其他 epochs 明确拒绝。OuterCycle 已另行实现原子 journal 的进程恢复，并在 NESTED_CPU_MOCK_HANDOFF.md 中执行三种跨进程恢复验收。FSDP 会接收 OuterCycle 当前接受模型对应的 checkpoint；真实 GPU 训练中断仍需服务器验收。

参考：[Accelerate FSDP](https://huggingface.co/docs/accelerate/usage_guides/fsdp)、[官方 checkpoint API](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)、[PEFT 与 FSDP](https://huggingface.co/docs/peft/main/accelerate/fsdp)。

Global token counters are repeated across rank logs; aggregate rank-0 global fields once. Use rank_forward_input_tokens/rank_supervised_tokens for per-rank physical work. Recovery replays are separate attempts. New FSDP runs cannot resume old Torch-backend snapshots or reuse a changed adapter configuration.
