# 师兄A：从零准备环境并执行完整实验一

## A与B的关系

**A与B是两个独立地点、两套独立机器和存储。A从零完成自己的环境、模型、数据、benchmark、镜像及验收，再执行正式实验。** B只提前测试代码、定位兼容问题，交付问题说明和修复建议；不要求A访问或复用B的模型目录、数据缓存、镜像、GPU、挂载路径或验收凭据。B通过也不能代替A本机验收，A不必等待B传来物理资产才能准备。

两边共同使用本仓库的固定源码版本、配置规范、HF revision和官方checker版本；B发现的修复先合入双方采用的代码版本，再分别验证。下文“共享目录”仅指**A自己的联网工作站和离线集群**之间的目录，不指A/B跨地点共享。

权威配置：configs/ordinary/exp001.json、exp001_nice.json、sources.json。先读本文件、AGENTS.md、[训练与环境总表](EXP001_QWEN35_9B_RECIPE_AND_ENV_ZH.md)、[长度与预算核查](TRAINING_LENGTH_AND_BENCHMARK_BUDGETS_ZH.md)、[技能映射](NATIVE_SKILL_MAPPING_ZH.md)。NICE在本项目内执行，不进入相邻旧NICE目录。

## 1. A自己的联网工作站：源码、环境和镜像

1. 在A的新目录获取指定源码commit，记录commit及必要补丁hash；使用独立环境／存储根。正式运行期间固定源码，不在运行目录pull。后续更新使用另一个固定版本目录。
2. 配置A自己的HF访问授权，尤其Llama gated权重。凭据不进入Git、数据、判分容器或结果包。
3. 构建docker/Dockerfile的prepare、gpu、grader三个target；BCB另用docker/Dockerfile.bigcodebench。base digest已在configs/ordinary/environment-lock.json固定，构建后核验pip check、pip freeze、最终项目镜像ID/digest及H200驱动兼容性。不能把base digest当作项目镜像digest。

```bash
docker build --target prepare -t self-curate-a:prepare -f docker/Dockerfile .
docker build --target gpu -t self-curate-a:gpu -f docker/Dockerfile .
docker build --target grader -t self-curate-a:grader -f docker/Dockerfile .
docker build -t self-curate-a:bcb-grader -f docker/Dockerfile.bigcodebench .
```

以上命令在项目根的联网工作站执行。prepare环境运行后续下载／导出命令，gpu环境用于离线GPU计算，grader环境只做受限CPU判分；不要在离线集群安装包。工作站如用容器执行准备命令，应显式挂载A的资产根，并保持导出绝对路径在计算节点可访问。

## 2. A自己的联网工作站：从头下载和冻结资产

示例A目录：项目/shared/self-curate-a/project，资产/shared/self-curate-a/assets，运行/shared/self-curate-a/runs。示例不是现成路径，agent按A实际存储规划建立。两路GPU／CPU判分都须访问同一套A本地资产。记录路径，不引用B的目录。

需完整准备：

| 类型 | 资产 |
|---|---|
| 模型3个 | Qwen3.5-9B、Qwen3.5-4B、Llama-3.1-8B-Instruct；本次只训练9B |
| 原始训练来源 | BAAI/TACO train、NuminaMath-1.5原集、Tulu Personas IF；TACO test仅保留测试来源 |
| coding benchmark | LCB release_v6、HumanEval+、MBPP+、BCB instruct/full |
| math benchmark | MATH500、AIME25、AIME26、Olympiad英文纯文本开放数学、GSM-Plus |
| IF benchmark | IFEval、IFBench、Multi-IF英文三轮 |
| 配套资源 | 锁定官方checker源码、EvalPlus完整增强测试release、NLTK等判分资源、通用／BCB独立判分镜像 |

固定地址、revision、split及文件白名单以sources.json为准，不能换成latest。准备CLI示例：

```bash
python -m self_curate.experiments.ordinary lock --output /shared/self-curate-a/assets/source-lock.json
python -m self_curate.experiments.ordinary download --lock /shared/self-curate-a/assets/source-lock.json --storage /shared/self-curate-a/assets --include-models
python -m self_curate.experiments.ordinary evalplus-cache --storage /shared/self-curate-a/assets --bindings /shared/self-curate-a/assets/source-bindings.json
python -m self_curate.experiments.ordinary benchmarks --lock /shared/self-curate-a/assets/source-lock.json --bindings /shared/self-curate-a/assets/source-bindings.json --output /shared/self-curate-a/assets/benchmarks
```

download生成A自己的source-bindings.json；benchmarks生成manifest.json及冻结题单。HF的EvalPlus镜像不能顶替完整增强测试。检查全部模型分片、tokenizer/template和所有题单／checker资源。

## 3. A自己的数据导出、数学50k和候选池

分别export三个来源，保留自动生成的.source.json来源边车：

```bash
python -m self_curate.experiments.ordinary export --lock /shared/self-curate-a/assets/source-lock.json --bindings /shared/self-curate-a/assets/source-bindings.json --key taco_train --output /shared/self-curate-a/assets/raw/taco-train.jsonl
python -m self_curate.experiments.ordinary export --lock /shared/self-curate-a/assets/source-lock.json --bindings /shared/self-curate-a/assets/source-bindings.json --key numina --output /shared/self-curate-a/assets/raw/numina-full.jsonl
python -m self_curate.experiments.ordinary export --lock /shared/self-curate-a/assets/source-lock.json --bindings /shared/self-curate-a/assets/source-bindings.json --key personas_if --output /shared/self-curate-a/assets/raw/personas-if.jsonl
python -m self_curate.experiments.ordinary sample-math --raw /shared/self-curate-a/assets/raw/numina-full.jsonl --output /shared/self-curate-a/assets/raw/numina-50k.jsonl --count 50000 --seed 42
```

确保父目录存在。数学从source→各source内problem_type两层比例抽50,000条后再准入，最后合格人数可少于50k。保存配额和ID／hash，不从B复制抽样结果。

三个领域分别执行prepare：domain为coding/math/if，传对应raw、A本机固定9B tokenizer路径、max-length8192及新的output目录。**--tokenizer必须从A的source-bindings中读取qwen9b路径；--blocked须包含当前冻结benchmark题单（按各jsonl逐个传入）及其他实际持出清单。** 不用preview／synthetic／B的128条短训文件构建正式池。具体CLI参数可用prepare --help核对，不能省略隔离输入。

题目／完整参考解／模板合计8192，超长整条排除；检查manifest里的长度排除前分布。证明题可SFT，禁止作为自动判分开发题。技能维持8/7/10类；少量未映射先不补标、保留训练池，不调用teacher。每技能50不足如实记录，不复制凑数。

生成A自己的pools.json，内容为coding/math/if到三个完整prepared-pool绝对目录的映射。准备完成后seal原始下载资产：

```bash
python -m self_curate.experiments.ordinary seal --storage /shared/self-curate-a/assets --lock /shared/self-curate-a/assets/source-lock.json --bindings /shared/self-curate-a/assets/source-bindings.json
```

prepared池和benchmark各自的manifest/hash也必须保留。后续A自己的实验可以复用这些已核验资产；本轮没有跨A/B资产依赖。

## 4. 将A的资产和镜像部署到A的离线集群

按A平台支持方式传输本机镜像，例如docker save/load或公司镜像仓库；仅A工作站联网准备，集群不临时下载模型／数据／包。迁移后核验checksum，保持绝对挂载路径一致；如必须改挂载路径，在启动正式计划前重新生成路径绑定，不能改写已运行快照。

通过server-bindings命令，指定A自己的lock、source-bindings、benchmarks/manifest.json、storage、project、run-root、**实际构建的通用grader镜像digest**和output。然后填写A独立构建的BCB镜像digest到grader_images.bigcodebench。先用server-bindings --help核对字段；不能沿用B的bindings／seal／image ID。

核实A自己的8张H200 UUID、CPU资源和容器执行能力。GPU镜像安装docker命令不代表有可用daemon；若A平台只允许CPU作业，执行agent需实现／验收相应部署适配，不能跳过代码沙箱。

## 5. A本机预检并生成A自己的凭据

正式run的验收门继续保留，但凭据由**A在A自己的环境**生成，不要求B签发或传文件。

1. 独立短训／判分验收：三领域小样本的9B LoRA加载、真实反向、原生checkpoint保存／中断恢复；全部12个checker先做少量正解／错误／超时验证。两路各4GPU、global32、vLLM热切／并发和进程隔离在A本机验证，另做最长8192反向容量验收。可参考B交接的测试方法，不能拿B的passed替代。
2. 验收用独立的小任务job/run_id，勿启动正式Full/Random矩阵，勿复用短训adapter作为正式分支初始权重。现有项目没有一条自动完成所有平台预检的命令，执行agent需使用原生训练／生成／判分入口编写薄控制脚本并保留它。
3. 真实检查通过后，将A实测证据汇总为A自己的server_smoke_receipt，并绑定checksum、A的asset_seal_hash及grader_image，写入A自己的server-bindings。不能先手填passed跳过测试。ServerBackend只验证凭据内容与资产身份，不识别“由A或B执行”。
4. 在A自己的完整coding池上准备并执行NICE反向probe：

```bash
python -m self_curate.experiments.ordinary.nice_experiment prepare --config configs/ordinary/exp001.json --pool /shared/self-curate-a/assets/pools/coding --output /shared/self-curate-a/runs/A-nice-preflight-plan
python -m self_curate.experiments.ordinary.nice_experiment gradient-smoke --plan /shared/self-curate-a/runs/A-nice-preflight-plan/plan.json --bindings /shared/self-curate-a/server-bindings.json --output /shared/self-curate-a/runs/A-nice-gradient-preflight
```

NICE probe为2题×1次最多64token回放、8候选反向，不训练／不额外测benchmark；命令自动写A的nice_gradient_smoke_receipt和SHA到A的bindings。它不证明正式20题有成功奖励，也不替代正式TACO参考解判分或最长生成回放容量验证。NICE probe依赖前面A自己的通用server smoke，执行顺序不能颠倒。

在正式plan前将A实际8个UUID写入运行配置。预检通过后放行approved，再生成新正式plan；不修改已有计划／源码hash。B未来提交修复时，A先核对补丁再在自己的环境重新验收，新源码不能用旧凭据放行。

## 6. 冻结计划，执行完整实验一

科研参数不变：LR5e-5/1e-4；常规LoRA r32/alpha64/global32/3epoch，epoch1/2/3均测，epoch3主报告；NICE r16/alpha32/global8/1epoch，20题×4rollout、3,000候选，高低各1,000，42/84/125步三节点均测原20题和四个coding benchmark。全部关闭thinking，预算逐benchmark读取配置，不按benchmark选权重。两路各4卡训练／生成，CPU异步判分；NICE梯度阶段独占8卡。

```bash
python -m self_curate.experiments.ordinary plan --config configs/ordinary/exp001.json --pools /shared/self-curate-a/pools.json --output /shared/self-curate-a/runs/exp001-plan
python -m self_curate.experiments.ordinary run --plan /shared/self-curate-a/runs/exp001-plan/plan.json --bindings /shared/self-curate-a/server-bindings.json
# 中断后保持原源码、配置、数据和run身份：
python -m self_curate.experiments.ordinary run --plan /shared/self-curate-a/runs/exp001-plan/plan.json --bindings /shared/self-curate-a/server-bindings.json --resume
```

run先进入内置NICE，再执行常规Initial/Full/Random5/10/20/30%。不单独重复跑完整NICE。正式NICE原20题参考解自检失败、全零方向等明确停止并报告，不能按模型表现换题。所有SFT分支独立从共同起点开始，不串联各LR／高低组。

常规33分支/30次SFT/90 checkpoint/372组benchmark；NICE4次SFT/12 checkpoint/60组结果。总34次SFT、102训练checkpoint、420 benchmark组＋12原题诊断。原20题诊断不是独立泛化验证。

## 7. 标签／未来开发集与交付

实验末尾汇报8/7/10类技能及长度分布，原生字段映射见NATIVE_SKILL_MAPPING_ZH.md。未映射先不标，保留训练池；probability/style等类别即使0成员仍保留并报告缺口。finalize-labels按每技能50发布未来开发集，缺额会保存分布并停止发布，不谎称全技能题单已齐备。未来开发题可能被EXP001 Full见过，因此后续从基础模型重启、隔离后才可用作独立开发验证。

结果位于A运行根下results/<UTC时间-ID>/，普通及nice子目录均保存表格、逐题结果、checkpoint索引／hash、token／曝光量／耗时及日志。打包：

```bash
python -m self_curate.experiments.ordinary bundle --run /shared/self-curate-a/runs/exp001-plan/results/ACTUAL_RUN_ID --output /shared/self-curate-a/exp001-results.zip
```

权重、完整原始数据和大梯度留A服务器，结果包保留路径与checksum。附A_REPORT.md，分别列环境构建／下载／预检／正式训练／各节点测评／标签发布的完成、失败和未执行；环境准备时间与正式GPU时间分开统计。A自己的已核验资产可供后续实验复用。B的问题报告可附为参考，不能列作A环境已验收的证据。用户自行Git提交／推送。
