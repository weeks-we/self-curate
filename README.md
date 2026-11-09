# Self-Curate

数据筛选与自改进实验项目，版本 0.16.0。工作目录按正式实验运行维护；开发测试与旧模块演示已归档。

## 运行入口

- [EXP001 V3 Handoff](docs/handoffs/EXP001_SERVER_V3.md)：正式实验交接与输出。
- [容器部署](docs/SERVER_CONTAINER_DEPLOYMENT_ZH.md)：无 GPU 联网准备、GPU 验收与执行。
- [文档索引](docs/README.md)、[当前状态](docs/progress.md)、[运行链路](docs/CODE_READING_TUTORIAL_ZH.md)。

`configs/experiments/exp001/server_bootstrap.template.json` 是服务器计划模板，
`server.template.json` 保存正式参数模板（含唯一 LoRA 超参），`sources.json` 固定来源。
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
