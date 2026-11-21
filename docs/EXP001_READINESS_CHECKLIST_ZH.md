# 实验一／NICE：配置就绪与待确定项

核对日期：2026-11-21。配置权威为exp001.json、exp001_nice.json、sources.json。当前科研训练配方已选定；正式服务器验收尚未完成。没有新要求用户重填学习率、batch或数学抽样数量。

## 已确定

| 项目 | 实验一 | NICE |
|---|---|---|
| 模型 | Qwen3.5-9B，关闭thinking | 同一模型 |
| 领域／数据 | TACO train；Numina1.5 source→problem_type比例抽50,000原题后准入；Tulu Personas IF | TACO train持出20任务，隔离后随机3,000候选 |
| 随机基线 | Initial、Full、Random5/10/20/30%，两个LR同组同成员与顺序 | 高低各1,000，不用中间组 |
| LR | 5e-5／1e-4 | 同两档 |
| LoRA | r32／alpha64／dropout0 | r16／alpha32／dropout0 |
| batch | 每分支4GPU×micro4×累积2＝32 | 4GPU×micro2×累积1＝8 |
| epoch／节点 | 3epoch，三个checkpoint都测benchmark；epoch3主报告 | 1epoch，42/84/125步三节点，分别测原20题和4个代码benchmark |
| 长度 | 完整输入8192，不截断；正式池先输出排除前长度统计 | 同训练长度；随机HF rollout4096输出／16384上下文 |
| seed | 42 | 42，CountSketch8192维／seed42 |
| 预算／测评 | 12 benchmark各自预算，见长度与预算核查 | benchmark同代码领域；原20题诊断独立预算 |
| GPU | 两路各4GPU训练／生成，CPU异步判分 | 梯度阶段先独占8GPU，之后分路SFT |

完整训练量：30次普通SFT＋4次NICE SFT，102份训练checkpoint；常规372组benchmark＋NICE48组benchmark＋12组原题诊断。所有分支独立起点。B仍是短测，每benchmark一次，NICE仅反向probe。

## 已确认的标签口径

用户已确认保留Coding八类／数学七类／IF十类，不删除probability或style。少量未映射项先不标，保留训练池，不调用teacher。映射详见[NATIVE_SKILL_MAPPING_ZH.md](NATIVE_SKILL_MAPPING_ZH.md)。

未映射或映射不完整的记录暂不用于按技能抽开发题。当前probability／style可能0成员，保留类别并记录缺口；每技能50题不足时仍报告shortages，不编造、不重复凑数。主训练／benchmark不因此改配方；包含空位类别的完整未来开发集仍不能声称已齐全。

## 数学抽样与8192

数学数量与抽样方法已明确实现，不再是待填写项：原Numina1.5按source，再按每个source内problem_type比例抽50,000，seed42，最大余数精确补齐。抽样在准入前，最终合格人数可能少于50,000。8192先保留；由工作站实际50k的length_audit_before_length_filter确认超长排除量，若需改变上限必须重新准备共用池、容量验收和新计划。

## HF来源齐全

当前3个模型、3个训练数据来源、12个benchmark均有HF URL和40位revision，共18项必需入口；另外TACO test有保留来源，不进入开发集。所有具体地址、split/config和文件白名单见[sources.json](../configs/ordinary/sources.json)，此前已逐项在线核对来源。

模型：Qwen/Qwen3.5-9B、Qwen/Qwen3.5-4B、meta-llama/Llama-3.1-8B-Instruct；本次只有9B正式训练，另外两个准备资产供后续使用。Llama需工作站HF授权。

训练数据：BAAI/TACO、project-numina/NuminaMath-1.5、allenai/tulu-3-sft-personas-instruction-following。

12 benchmark：LCB release_v6、HumanEval+、MBPP+、BCB instruct/full、MATH500、AIME25、AIME26、Olympiad英文纯文本开放数学、GSM-Plus、IFEval、IFBench、Multi-IF英文三轮。EvalPlus正式增强测试另由官方release缓存；官方Git checker、NLTK资源和容器也须工作站提前准备，不仅下载HF题面。

## 不需用户决定、由A在自己的服务器填入／验收

- 8个独占GPU UUID（当前[]）、共享模型／数据／运行根绝对路径、CPU配额、Docker daemon或公司容器执行能力。
- 工作站实际构建的最终项目镜像digest（当前null），pip freeze、源码与资产seal。三套base image摘要已有。
- Llama访问授权、模型完整分片、全部增强测试／checker资源、本地离线加载。
- 最长8192反向、4rank FSDP2完整保存／中断恢复；global32固定，micro×累积可在已列候选中实测后冻结。
- vLLM真实LoRA加载／热切、每卡64序列容量、官方容器判分、双路进程隔离；NICE真实反向probe与正式20题参考解自检。
- A本机新源码对应的server_smoke_receipt和nice_gradient_smoke_receipt；不导入B凭据。approved=false是这些验收的门，不需要用户把它当作额外科研超参。

## 本轮实际修复

- 末尾标签发布按冻结epochs计算评测数量：Initial一次、训练分支三次，接受372组常规结果，不再错误要求132组。用真实已完成mock矩阵和独立八技能fixture验证，发布400开发题／80未来训练题成功。
- Tulu IF原生约束的空格、前导空格和别名映射修复；in english and capital保留language＋case两个技能，未知约束显式待复核，不无声忽略。

IF原始29,980条全部可映射到九类，无未标样本，原生多标签分布如下（未做最终去重／测试隔离）：

| 技能 | 样本成员数 |
|---|---:|
| format | 14217 |
| lexical | 8760 |
| length | 8875 |
| case | 6769 |
| language | 4547 |
| punctuation | 2390 |
| composition | 4668 |
| required_content | 4575 |
| boundary | 2486 |

代码／配置语法、HF入口字段、IF复合／未知约束、标签发布计数专项检查通过；GPU与容器仍未本地验收。本轮不调用付费teacher。

2026-11-21部署口径：A/B异地独立。A从零下载资产、构建镜像并本机验收；B仅提前测试反馈，B的目录、缓存、镜像或凭据不是A的运行依赖。A内工作站与集群之间可共享／迁移A自己的资产。
