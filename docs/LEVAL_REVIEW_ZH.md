# Leval 文档与模块两轮复查（0.9.1）

2026-11-06。本轮核对原蓝图、D102–D108 用户扩展、P0/P1 契约、M1 使用历史、两套 Leval、共享角色、内层搜索及外层训练/验证。

结论：发现了版本、数值条件、恢复与费用边界的问题，已补回归并修复。原蓝图保持原文，候选 SFT 仍按用户新增授权运行在隔离分支；解释角色与共享模型不提交参数。

## 第一轮：要求、数值与模块接口

| 问题 | 原行为 | 修复 |
|---|---|---|
| SFT 协议未绑定实现 | 换训练器/任务执行器仍可能保持同一 protocol，历史被当成同口径 | protocol 绑定数值内核、服务/后端类的源码版本、声明的 implementation_hash；真实适配器须明确声明实现与配置身份 |
| 调用方 autocast 泄入 | 相同输入/配方在 AMP 上下文里得到另一份梯度或 SFT 更新，缓存 key 未区分 | Torch NICE/SFT 在自身计算范围禁用外部 autocast；保留参数原生 dtype，CE 用 float32；记录 torch/设备/TF32/确定性等数值条件，变化则失效或拒绝 |
| 模型 factory 未绑定 | 同 state_dict 与不同模型构造实现可能拥有相同后端身份 | StateDictModelStore 把 factory 源码/声明身份加入 implementation_hash；GPU/远端 store 仍负责核验实际权重与隔离 |
| 恢复丢失 M1 使用记录 | restore_reference 恢复基线，却没带回先前候选的 candidate_sft | 保存紧凑的 usage_records，按 ID/content/tokenization 绑定恢复，单调合并；旧 pending 不覆盖现有 accepted/rejected |
| 坏返回/异常丢费用 | 服务无 ledger 时丢弃有效 usage；空集合汇总甚至使未知费用显示为 0 | 保留服务详细 ledger；缺 ledger 则记录边界调用及有效 usage，其余未知为 null；不编造 rollout 数，也不把已报告的空 ledger 当成一次物理执行 |
| 预算预检时机太晚 | 已知轨迹过长，仍先计算基线后才拒绝 TrainingRequest | 在任何基线/训练调用前检查完整轨迹长度，不裁剪、不浪费已可避免的验证费用 |
| 无效基线证据缺项 | provider.baseline 只接受有效报告，导致失败 evidence 中看不到本次无效报告 | 额外保存 attempted_baseline，保留真实失败报告而不把它缓存为有效基线 |

最初 10 个专项在修复前全部实际失败。其中 CPU 微型模型的环境 autocast 泄入使原始梯度最大差约 0.00210、SFT 权重最大差约 0.00228；修复后与默认数值上下文一致。失败证据保留在本次运行目录，不把测试奖励当研究结果。

## 第二轮：恢复、历史、边界与旧产物

1. **自动基线不能冒充已存在参考。** 原先两个新 provider 在相同配置下会拥有相同 protocol，即使各自重新测了一次基线。现在自动基线有独立不可变 reference_namespace；旧搜索只能使用同一 provider 或恢复其已保存基线。显式导入同一份基线仍拥有稳定身份。namespace 是引用身份，不是采样 seed，不改变训练/任务随机规则。
2. **真实域中的 mock 识别一致。** NICEReference 与 NICE 分数证据在检查 URI 前统一 strip/lower，禁止空白或大小写掩盖的 fixture/mock。元数据反例仅用于本机单测，不代表真实模型或任务执行。
3. **Windows 嵌套证据路径。** 深目录复查实际遇到 WinError 206。评价 run 名改用完整 128 位 UUID 的 URL-safe 紧凑编码，减少路径占用；不削减身份位数，不声称支持任意长路径。任务执行的完整 request hash 目录格式保持兼容。
4. **重叠批次只复用兼容梯度。** 测试同模型/同配方的较小重叠批次只命中既有样本；修改投影配置重新计算。原有 source/model/mask/坐标身份继续保留。
5. **恢复不逆转历史。** 检查使用记录内容绑定、终态单调性、无重复收费、原始池不修改，以及两种内核到 Leval→Lopt→P1→内层快照的连通性。原有外层“子集→验证/判断→接续轨迹训练→验证/判断”继续沿用原模块。

另三项 factory/真实域反例和两项参考身份/真实适配器身份反例也在原 0.9.0 源码下实际复现失败。专项共新增 20 项，首次失败证据共覆盖 15 项；其余案例核验重叠缓存、稳定导入和终态恢复等边界。

## 接口与兼容

- 既有 ValidationRequest/Result、RoleEvaluation、RoleScope、P0/P1、TaskSuite/TaskEvaluation、TrainingRequest/Outcome 不改格式。Leval/Lopt prompt 继续用现有结构化槽位，正文待填。
- `SFTValidator.save_reference()` 输出 reference schema=2，新增 reference_namespace 和 usage_records；restore 核验独立文件 hash、protocol、模型/suite/recipe，以及逐条训练输入身份。它不恢复在途服务或把旧费用重新收取。
- 0.9.0 的搜索/cycle/报告仍可只读核验，不改写旧文件。新实现改变了执行协议身份，因此旧计算会话不能静默接着新版数值内核运行；需要新建会话，或显式核验旧 TaskEvaluation 后作为新会话的固定参考导入。旧 schema=1 reference 文件可解析字段，但不伪造新版 namespace/实现身份来绕过 protocol 检查。
- Python 本地 mock 适配器可使用类/源码自动身份；真实候选验证的 TrainingBackend 和 TaskRunner 必须提供 SHA256 字符串 implementation_hash，覆盖远端实现和实例配置。数字身份不是远端执行真实性的证明，加载权重/环境/判分仍由真实 adapter 与服务器验收负责。
- NICE 的梯度矩阵及版本证据属于 valuation；M1 仍通过既有外部 GradientFeature 接口接入指标，本轮不把矩阵行伪装成已经回填的 M1 特征或持久化 M5 缓存。候选训练使用历史则通过 updated_records 显式返回，原始数据视图不改写。

全量验收、两种闭环、旧产物恢复、原 NICE 数值串行对照、旧证据 checksum 及源码包证据，以 `runs/leval-review/verification.json` 为准。真实 GPU、目标模型长上下文容量、任务容器/判分、提示词正文和研究阈值继续待接入；本机测试不证明模型能力提升。
