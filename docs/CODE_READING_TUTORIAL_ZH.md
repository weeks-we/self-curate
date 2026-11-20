# 代码阅读顺序

1. 先读 `ORDINARY_EXPERIMENTS_ZH.md` 与 `configs/ordinary/exp001.json`：明确实验一不调用 agent，训练和评测不是同一个进程职责。
2. 看 `preparation/ordinary.py` 的 `adapt/prepare`：真实一条样本怎样成为固定 ID、原生标签与 user/assistant 对；再看 `ordinary_m1.py` 如何接既有 data/MetricView。
3. 看 `experiments/ordinary/sources.py`、`workstation.py`、`benchmarks.py`：来源、下载、题单、隔离和校验如何冻结。
4. 看 `experiments/ordinary/workflow.py` 的 `plan/execute`：一个领域如何产生五分支，哪个状态先落盘，中断后怎样避免重复执行。
5. 看 `training/ordinary.py`：token 掩码 → HF Trainer → FSDP2/LoRA → checkpoint/final。与既有内外层训练契约分开看，不把协议实现误认为 GPU 已验收。
6. 看 `evaluation.py` → `grade.py` → `backend.py`：模型只看题，标准答案只由判分读取，生成代码只在 CPU 隔离容器执行。
7. 最后看 `agents/ordinary.py`、`smol_client.py`、`agents/runtime.py` 和三个 prompt：它们为实验二准备，保持已有 Lexec/Leval/Lopt 协议，不是实验一中的额外评测器。

运行 `python -m self_curate.experiments.ordinary smoke --output runs/readthrough-new`，对照输出 plan/selections、state、events、角色 request/steps/usage、acceptance；所有示例明确 synthetic。
