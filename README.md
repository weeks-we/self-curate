当前 agent harness 已统一迁移到 Claude Agent SDK，见 [接入与验收说明](docs/HARNESS_CLAUDE_SDK_ZH.md)。

# Self-Curate

数据筛选与自改进实验项目，版本 0.16.0。工作目录按正式实验运行维护；开发测试与旧模块演示已归档。

- [SFT FSDP / recovery handoff](docs/handoffs/SFT_FSDP_RECOVERY_HANDOFF.md): eight GPU ranks, timestamped outputs, optimizer-state continuation.

## 内外层 CPU 联调

小规模 3×5 嵌套循环的配置和交接见 [CPU mock Handoff](docs/handoffs/NESTED_CPU_MOCK_HANDOFF.md)。
它只调用合成 mock 后端，不加载真实模型、不使用 GPU；真实实验仍走各自正式入口。

```bash
PYTHONPATH=src python -m self_curate.smoke --config configs/smoke/nested_cpu_mock.json --output runs/nested-cpu-new
PYTHONPATH=src python -m self_curate.smoke --output runs/nested-cpu-new --verify-only
```

## 运行入口

- [当前 Rebench EXP001 Handoff](docs/handoffs/EXP001_REBENCH_WORKSTATION_V1.md)：唯一总交接入口。
- [容器部署](docs/SERVER_CONTAINER_DEPLOYMENT_ZH.md)：无 GPU 联网准备、GPU 验收与执行。
- [文档索引](docs/README.md)、[当前状态](docs/progress.md)、[运行链路](docs/CODE_READING_TUTORIAL_ZH.md)。

当前使用 `rebench.workstation.json` 和 `workstation.bindings.template.json`，按总交接生成
冻结的 `server.resolved.json`。`server_bootstrap.template.json`、`sources.json` 和旧 V3
交接中的 SWE-smith 来源不是本次入口；不要直接执行旧 bootstrap 的 stage all。
`server.template.json` 仅作为配置生成的底稿，必须由 Rebench 准备流程替换来源及真实回执。
实际数据、权重、镜像和 GPU 容量必须在服务器核验；不能把模板当成已验收配置。

```bash
python -m self_curate.server --help
python -m self_curate.experiments.exp001 --help
python -m self_curate.experiments.exp001.status --output <run-directory> --watch
```

EXP001 的 mock.json 与整体 CPU 预演后端保留，用于部署前检查训练/验证/恢复/打包链路，
不下载目标模型，不产生研究结论。五个旧模块级 *-demo 命令和测试工程已删除。

## 目录和提交

src/ 为运行模块，configs/ 为实验/角色配置，docker/ 为环境构建，docs/ 为需求、接口与交接。
没有维护中的 tests/ 或 pytest 开发依赖。数据、模型、缓存、凭据、状态和结果不提交 Git。
运行必需的版本/数据校验、任务验证、恢复检查、容量验收继续保留。

开发测试和演示的完整源码备份在本地被忽略的 runs/runtime-only-20261109/；
该备份不会随 Git 更新到师兄服务器。用户自行 commit/push，助手不代执行。
