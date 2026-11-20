> Current ordinary-task entrypoint and experiment parameters: ORDINARY_EXPERIMENTS_ZH.md and HANDOFF.md. Below describes reusable interfaces, not an active experiment configuration.

# 共享实验资产与服务器更新（0.11.0）

2026-11-07。用户确认本轮先落实资产保存/复用与更新规范；Coding 数据准备、Teacher 标注及首次真实测评在第一次实验时执行。附件是后续数据模块的实施规范，本轮没有执行其中的数据下载、API、任务环境或 benchmark 指令。

## 1. 递进实验共享什么

共享的单位是身份完整匹配的计算，而不是“第一次产生的所有结果”。

| 资产 | 可共享条件 | 变化后 |
|---|---|---|
| 来源锁、精选匹配、静态准入、七技能标注、开发分组、测试注册 | 来源/内容、Teacher/ontology/prompt、tokenizer/template/mask、抽样/隔离规则等一致 | 新建准备资产，旧版本保留 |
| M0 开发集测评 | 模型/adapter、任务 union、环境/评分、sampling/seeds、预算、聚合及执行实现一致 | 不是 M1 的当前能力；改变条件后重新评测 |
| M0 NLL/entropy | 数据 ID/内容/来源/输入/mask、模型、scope/skill、reduction/log-base 和计算 hash 一致 | 新身份计算，不给旧值改名 |
| 第一次 Lopt 调用 | 模型、完整 system/payload、当时 history、编辑 policy、采样/seed/限制、超时与实现一致 | 改变这些条件后不是相同生成调用 |
| 冻结 ProgramSpec | 作为比较前确定的共同初始策略，明确保留来源 | 可迁移程序；不自动继承它在旧状态下的选集或得分 |
| BatchManifest | 程序、数据/指标快照、质量/去重/预算约束、seed 一致 | 重新筛选，不能只凭同一程序复用选集 |
| 训练共同前缀 | 起点、实际选集/顺序、配方、数值条件、RNG/接续规则一致 | 从首次分歧处分支；后续沿各自真实参数路径继续 |

V2/V3 的编辑权限不同，第一次 Lopt 不自动是共同计算。若需要共同初始策略，在对照开始前冻结同一个程序并在配置中引用；不要将它登记成各模式又生成了一次策略。不同 seed 的独立重复实验也不能默认共享其随机输出。

## 2. 已实现的公共接口

新包 `self_curate.assets` 使用普通文件，不引入数据库或 operator 查询索引。

| API | 职责 |
|---|---|
| AssetSpec.create / Producer | 不可变输入 key 与生产者 stamp：run、源码内容、环境/配置 hash、可选完整 Git commit |
| AssetStore.publish/get/inspect | 快照、逐文件 checksum、依赖验证、COMPLETE 标记及原子发布；重启/搬迁存储根后可读；同 key 不覆盖冲突结果 |
| AssetStore.reuse | 返回既有资产并可写独立消费回执；原 CostLedger 保留，不重新计模型/训练费用，另记读取耗时 |
| register_signals/reuse_signals | 核验既有评分产物；只将匹配的 signals 附到当前 M1 记录，保留当前质量、metadata、技能与训练历史 |
| register_capability/reuse_capability | 核验完整 valid/protocol_verified 开发测评；绑定 execution_hash，不调用 TaskRunner |
| strategy_spec/register_strategy/reuse_strategy | 保存完整生成输入、回复、程序；复用 ProgramSpec，不回放角色 history，也不返回旧选集 |
| register_lopt_call | 从现有 RoleRuntime 导出指定已完成调用的真实 before-context 和该次物理费用；不混入后来 history |
| freeze_run_inputs/load_run_inputs | 冻结 resolved config 与 key+manifest hash；核对预期 lock、源码/环境和依赖，不执行配置 |
| inspect-asset CLI | 只读检查共享资产 |

数据准备模块将来可调用通用 publish；本轮没有实现或验收该模块。当前 signal 桥接按完整有序批次复用，装入后仍用 MetricView 按 ID 取子集；没有实现跨资产的自动 token/梯度缓存合并。

```text
<storage_root>/
  shared/<key 的紧凑编码>/
    manifest.json
    COMPLETE.json
    p/                       # 原模块的完整输出快照
  runs/<run_id>/
    frozen-inputs.json
    reused-baseline.json
    reused-signals.json
    reused-initial-strategy.json
    ...本实验新输出...
```

源结果不原地修改。内部原 URI 保留历史事实，StoredAsset.resolve_evidence(ref) 可按 hash 在资产快照内定位证明文件。这里只保存登记的输出树，不自动获取外部 URI、镜像或大权重；外部依赖应另有不可变资产/版本锁。

checksum 证明字节和声明身份一致，不认证外部数据、模型或服务设置真实。通用 publish 不代替数据质量门禁；typed 桥接复用原模块的语义 loader。

## 3. 首次生产与后续消费

以下变量由真实运行的结果和锁文件提供，不是内置模型或已存在数据。

```python
from self_curate.assets import (
    AssetStore, Producer, register_signals, reuse_signals,
    register_capability, reuse_capability,
    register_lopt_call, reuse_strategy, freeze_run_inputs,
)
from self_curate.contracts import digest

store = AssetStore(storage_root / "shared")
producer = Producer(
    run_id=run_id, source_tree_hash=source_lock_hash,
    environment_hash=environment_lock_hash,
    config_hash=digest(effective_config), code_commit=resolved_git_commit,
)

# 首次计算完成后登记，不再做一次模型调用。
scores_asset = register_signals(
    store, scoring_output, model=M0, specs=signal_specs, producer=producer,
)
baseline_asset = register_capability(
    store, evaluation_output, model=M0, suite=development_suite,
    execution_hash=task_runner_and_inference_config_hash, producer=producer,
)
strategy_asset = register_lopt_call(
    store, original_runtime, first_lopt_call_id,
    policy=edit_policy, sampling=frozen_sampling_artifact,
    implementation_hash=generation_code_and_config_hash,
    snapshot_output=run_output / "initial-strategy", producer=producer,
)

# 后续实验显式固定已保存的 manifest 引用。
current_records, receipt = reuse_signals(
    store, current_records, model=M0, specs=signal_specs,
    backend=execution_domain, consumer_run=next_run_id,
    reference=scores_asset.reference,
    receipt_path=next_output / "reused-signals.json",
)
baseline, receipt = reuse_capability(
    store, model=M0, suite=development_suite,
    execution_hash=task_runner_and_inference_config_hash,
    consumer_run=next_run_id, reference=baseline_asset.reference,
    receipt_path=next_output / "reused-baseline.json",
)
program, receipt = reuse_strategy(
    store, strategy_asset.spec, consumer_run=next_run_id,
    reference=strategy_asset.reference,
    receipt_path=next_output / "reused-initial-strategy.json",
)
```

生产调用的费用只保存在原账本。消费回执记录原生产者、资产 hash、new_model_calls=0、new_training_supervised_tokens=0、读取耗时和 origin_cost_recharged=false；未知原成本保留未知，不能补造零。原账本不重新绑定成消费者的一次物理工作。

已存在 pin 时，缺失或损坏明确报错。首次生产前可用 store.get(spec) 查找，未命中且未 pin 时返回 None，由未来控制器决定是否计算。本模块不自动调度昂贵任务或保证跨进程计算 single-flight；共同前缀应安排一个生产任务，其余实验等待完成资产。

相同输出/生产者/费用的重复登记幂等；同 key 出现不同输出或生产者会拒绝覆盖。重新采样应将预设 seed/replicate 纳入输入身份。若评分产物包含其他调用的 cached 值，需 pin 来源资产；优先登记首次 fresh scoring，不能把空的新账本当历史零成本。

register_lopt_call 的 sampling/implementation_hash 须来自原运行的冻结设置；外部适配器确认实际服务采用这些配置。保存的程序仍须经过 P1/Leval，登记资产不证明它合法改进或优于其他程序。

每次运行可用 freeze_run_inputs 保存下一实验的 resolved config 与三个 AssetRef；Producer.config_hash 必须等于 digest(config)。恢复前用 load_run_inputs，提供原 lock_hash 以及实际源码/环境 hash；不同代码/环境被拒绝。它只是输入核验，不是训练/外层崩溃恢复。配置保存密钥环境变量名，不保存真实密钥。开发时 commit 可为 None，但源码内容 hash 仍必填；正式服务器运行使用明确 commit。

## 4. GitHub 更新怎样不干扰运行

用户后续确认先采用串行、单源码目录：前一实验及相关子任务结束并保存完整状态/结果后再更新。下方 worktree 示例保留为并行运行或恢复旧版本时的选项，不要求每次实验复制源码或新建分支。逐实验步骤见 [handoff 约定](handoffs/README.md)。运行中仍不得在该作业目录 pull 或升级共享环境。

现有 ignore 规则覆盖 data/models/checkpoints/gradient_cache/runs/artifacts、缓存和 .env。未跟踪的这些产物通常不会形成源码合并冲突；已被 Git 跟踪的文件不受 ignore 保护。新配置路径与服务器已有本地文件重名，也可能阻止更新。[Git ignore 官方说明](https://git-scm.com/docs/gitignore)

Git 无冲突不等于运行无影响：在正在执行的代码目录 pull，后续导入/配置读取/source hash 可能看到新文件，已加载 Python 对象却仍是旧版本。本项目部分版本检查会因此失败，也可能出现未被检查覆盖的混用。GPU 已加载权重不会因 pull 自动升级，关联的代码和配置仍需固定。

服务器分开保存：

```text
/srv/code/self-curate-main/        # 干净更新入口，接收 GitHub 更新
/srv/jobs/<run_id>/code/           # 固定 commit 的 detached worktree
/srv/jobs/<run_id>/venv/           # 对应固定环境
/data/self-curate/shared/          # 跨实验资产
/data/self-curate/runs/<run_id>/   # 独立输出与冻结配置
```

路径为示例，由配置提供；无需移动已有历史产物。同版本只读代码/环境可以共享，运行中不升级共享依赖。尤其避免 editable install 指向持续 pull 的 main。

以下流程由师兄在服务器执行，本轮未运行：

```bash
RSI_MAIN=/srv/code/self-curate-main
RSI_RUN=v0_first_run
RSI_JOB=/srv/jobs/$RSI_RUN

git -C "$RSI_MAIN" status --short
# 有本地源代码/配置修改时先明确处理，再更新。
git -C "$RSI_MAIN" pull --ff-only
RSI_COMMIT=$(git -C "$RSI_MAIN" rev-parse HEAD)
mkdir -p "$RSI_JOB"
git -C "$RSI_MAIN" worktree add --detach "$RSI_JOB/code" "$RSI_COMMIT"
python3 -m venv "$RSI_JOB/venv"
# 先按服务器已验收的依赖锁安装依赖，再安装固定代码。
"$RSI_JOB/venv/bin/python" -m pip install --no-deps -e "$RSI_JOB/code"
```

--ff-only 在历史分叉时拒绝更新，不自动生成合并，也不替代本地修改检查。[git pull 官方说明](https://git-scm.com/docs/git-pull) detached worktree 支持独立工作目录并固定指定 commit。[git worktree 官方说明](https://git-scm.com/docs/git-worktree)

后续配置进入 GitHub，只更新 main 并准备新作业；已有作业保留原 commit、环境和 resolved config。机器路径放环境变量/ignored configs/local，运行代码不反复读取 main 的配置。记录 source/env/config hash、输入 AssetRef、seed 和实际模型/数据身份；相关数值协议变化时新建运行，只有新增别的实验配置不会自动使共同 M0 资产失效。

## 5. 范围与验收

本轮实现资产 API、三个核心桥接、冻结输入与只读检查。完整实验 runner、首次数据准备/Teacher 标注、模型/任务服务、自动持久化 NICE 特征重组及外层恢复继续按 [就绪清单](progress.md) 推进。没有运行真实服务器 Git 更新、API、GPU 或任务环境。

专项验证覆盖：源文件变化/存储根迁移、完成标记与文件损坏、冲突/依赖/pin、消费费用不重记、三类资产身份失配、M1 当前标签保留、Lopt 实际 before-history/单调用费用，以及冻结配置/源码/环境。最终全量回归、兼容和交付包以本地 runs/shared-assets/verification.json 为准。
