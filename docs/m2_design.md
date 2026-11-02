# M2：实现前的逻辑、接口与依赖设计

日期：2026-11-02。依据原技术蓝图第 7 节，实现完整 M2/P1，不扩展到 Agent 搜索或模型计算。

## 输入与输出

- M1 的 `PoolSnapshot / MetricView`：唯一 ID、完整轨迹、技能/证据/长度/质量/使用历史、版本化 NLL/entropy。
- `ProgramSpec`：只读 JSON 无环程序，节点参数严格校验，禁止代码执行和修改系统配置。
- `ExecutionContext`：系统提供的固定 QualityGate/BudgetPack、模型和信号/CDF 上下文、训练 seed、可选预计算语义相似度。
- 输出 `SelectionResult`：精确 BatchManifest、节点输入输出/排除理由、指标读取版本和不可行缺口。

## 与其他模块的关系

```text
M1 数据与版本指标 ─→ M2 确定性筛选/混合/完整轨迹组批
                         ↓ 精确成员、顺序、分支、token 与身份
                  M5 按需梯度 → M6 评价此批次
                         ↓ 搜索决定（M7）
                  M8 按原清单训练，不重新抽样
```

M2 不标注数据，不计算 NLL/entropy/梯度，不调用任务环境或训练模型；也不维护持久化 operator 索引。
读取指标通过 M1 的 ID 保序内存视图；上下文中的模型、scope、计算配置和 CDF 必须精确匹配。

## 程序结构

共享 `eligible_pool` 入口 → 有序的单输入分支链 → 可选的终端 MixAllocate → 系统 BudgetPack。
程序允许共享前缀和分支，节点列表顺序不改变依赖关系；拒绝环、未知依赖、不可达节点和中途多输入混合。
QualityGate/BudgetPack 不能出现在 Agent 可编辑节点中。算子注册表和语义版本由系统代码维护。

## 八个算子

| 算子 | 规则 |
|---|---|
| SkillSelect | 从实际标签读取 any/all；未知标签集合按显式 missing_policy 处理 |
| EvidenceSelect | 使用 M1 已计算指标，统一检查占比、密度、token、完整事件与行为，不重算分母 |
| NLLSelect / EntropySelect | 固定 scope/原始值或指定 CDF 百分位；先区间，再可选排序/比例截取；同分按 ID |
| LengthSelect | 对 M1 的独立长度字段作闭区间过滤 |
| DedupSelect | 精确内容、标注重复组闭包、或外部提供的版本化成对相似度；不下载/计算 embedding |
| SampleSelect | 可复现 uniform/ranked/stratified，按条数或完整轨迹 token 上限抽取，无放回 |
| MixAllocate | 按监督 token 分配分支，重叠成员唯一归属；固定优先级，短缺明确报告 |

固定过滤保持输入顺序；需要排序/截断/抽样的算子才改变顺序或成员。
范围为闭区间；比例截取使用 ceil；随机选择使用稳定 hash 优先级，绑定程序/节点 seed。
缺失可声明 error/exclude；技能/证据另可显式 include，模型信号另可 request。
request 只产生结构化的缺失计算请求，交给后续计算服务处理，不隐式调用模型。

## 固定外壳与离散预算

QualityGate 默认只接受 pass，pending 训练排除，已接受/被拒绝数据按系统固定政策处理。
允许系统配置额外排除 ID/来源家族，程序无权更改。

BudgetPack 使用版本化 ordered-greedy：只选择完整轨迹、去除 ID/可选精确内容重复，
检查固定 token 窗口、样本数及有效 batch 整除。Mix 的实际监督 token 比例最后再次检查。
计数修整时优先保留排序靠前的成员，并尽量维持分支比例。
这是确定性启发式组批，不宣称穷尽所有组合；找不到合法组批时给出缺口，不能伪造“数学上无解”或悄悄放宽约束。

## 可重放性和验收

程序语义 hash 关联算子版本与固定政策；执行身份另外绑定输入快照、模型/信号上下文。
BatchManifest 同时保存排序后的成员与明确训练顺序、内容和实际 tokenization hash、唯一分支归属和实际 token。
后续梯度追加不改变训练 token 身份；训练仍须核对实际输入/mask。

验证覆盖八算子、图校验、缺失与错误上下文、重复/重叠、比例残差、不可行、seed 重放、固定过滤换序，
以及 M1→M2 CLI、输入不变性和训练/估值使用同一 manifest 的交接约束。
