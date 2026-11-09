# EXP001 容器准备和运行

配置环境和申请显卡分开时，使用 `docker/Dockerfile`。默认 `runtime` target 包含 Python、
CUDA 开发依赖、vLLM/PEFT/Transformers、mini-SWE-agent 与锁定版本的官方 harness。
模型、原始数据、结果放持久共享盘；不会为每个任务再复制一份 27B 权重。

## 联网节点：不申请 GPU

先把三个 harness 完整 SHA 写入平台构建参数。SWE-bench 使用 v4.1.0 对应 commit，兼容
当前固定来源的旧数据 schema；SWE-smith 和 Harbor 首次解析后冻结，后续沿用，不在作业中追 main。
基座镜像也应由平台解析并固定 digest。首次构建会解析 Python 依赖，构建内 freeze 和 direct_url
写到 `/opt/build-evidence`；复现实验使用同一个镜像 digest，而非重新构建同名 tag。

```bash
docker build -f docker/Dockerfile --target runtime \
  --build-arg BASE_IMAGE="$BASE_IMAGE_DIGEST" \
  --build-arg SWESMITH_COMMIT="$SWESMITH_COMMIT" \
  --build-arg SWEBENCH_COMMIT="$SWEBENCH_COMMIT" \
  --build-arg HARBOR_COMMIT="$HARBOR_COMMIT" \
  -t self-curate:exp001-v3 .
```

先复制 bootstrap 模板到共享盘，修改 storage_root/run_id/Teacher 配置。下面用
`/storage/self-curate` 举例，实际路径由平台指定：

```bash
docker run --rm --network host \
  -v /storage/self-curate:/storage/self-curate \
  -v /var/run/docker.sock:/var/run/docker.sock \
  self-curate:exp001-v3 python -m self_curate.server \
  --plan /storage/self-curate/plan.json --stage assets --execution server
```

此命令下载固定来源、预拉 SWE 任务镜像、用 Harbor 原生生成器构建 Terminal 环境并
做仅安装工具的快照。该阶段不会进行模型推理、任务求解或任务判分；安装容器不申请 GPU，
正式运行仍按原任务的 CPU/RAM/GPU 数配置。输出 `offline-assets.json`，记录本地 image ID、
RepoDigest、来源锁和预安装内容。原生 API 不兼容、缺任务或安装失败会阻塞。

构建平台若要求“模型和数据也在镜像中”，可改用 `--target with-sources`。这会产生很大的
来源镜像，只执行下载，不操作 Docker daemon。HF 凭据使用
`--secret id=hf_token,env=HF_TOKEN`，不写 ARG、ENV 或 Dockerfile。不要把挂载的空目录直接盖在
镜像 `/storage/self-curate` 上；先用 `docker cp` 导出该目录到持久盘，再挂载使用。
运行时推理 key 和 Teacher key 也仅从 secret/环境变量传入，不进 Git 或镜像。

**任务镜像不在运行镜像的文件系统内。** 若联网节点和 GPU 节点不是同一 Docker daemon，
需按 offline-assets 清单推送私有仓库或 `docker save`/`docker load` 到执行 daemon；本地
构建的 Harbor 安装快照也要一起迁移。只复制模型目录或只上传运行镜像不够。

## GPU 节点：验收后启动

需要 NVIDIA Container Toolkit、8 张授权 H200、足够共享内存和主机 RAM、Docker daemon、
Compose >=2.24.4。运行容器使用 host network，持久盘按**相同绝对路径**挂载；宿主 Docker
启动的兄弟任务容器才能访问这些路径，桥接推理 relay 也才能绑定宿主 bridge 地址。
如果平台禁止 Docker socket/host network，当前执行后端不能直接使用，需要平台提供等价能力，
不能把有 Docker CLI 误认为有任务容器环境。

```bash
docker run --rm --gpus '"device=GPU-UUID0,GPU-UUID1,GPU-UUID2,GPU-UUID3,GPU-UUID4,GPU-UUID5,GPU-UUID6,GPU-UUID7"' \
  --network host --ipc host \
  -v /storage/self-curate:/storage/self-curate \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e DEEPSEEK_API_KEY \
  self-curate:exp001-v3 python -m self_curate.server \
  --plan /storage/self-curate/plan.json --execution server --stage all \
  --gpu-uuids GPU-UUID0 GPU-UUID1 GPU-UUID2 GPU-UUID3 GPU-UUID4 GPU-UUID5 GPU-UUID6 GPU-UUID7
```

UUID、镜像 digest 和路径必须替换。`all` 读取已准备的来源和镜像，不回退到在线下载；
它仍包含尚未缓存的 Teacher 标注 API（不等于大文件下载）。可先在联网阶段完成标注，
但当前统一 prepare 入口仍做硬件 inventory；不要把示例宣称为已执行的无 GPU 标注方案。
若只需分步操作，用 prepare → preflight → run；恢复追加 `--resume`。

进度：`executions/<run_id>/progress`；服务/利用率：同目录 services；正式报告及恢复状态：
`runs/<run_id>`；发送结果：`runs/deliverables/<run_id>/results.zip`。

HF 离线模式只限制 Hub 请求，不封锁任务网络。预安装避免常见包安装，任务自己的测试脚本
可能仍访问外部服务；不能未经 smoke 就承诺严格断网运行。服务器真实验收前不标记 ready。

依据：[Docker 构建 secret](https://docs.docker.com/build/building/secrets/)、
[HF 离线环境变量](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)。
