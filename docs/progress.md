# M1 交付与后续开发计划

日期：2026-11-01。对应用户技术蓝图的 **M1：数据与标注**，不是整个 Agent 系统。

## 本次完成范围

- 独立 Python 包、严格 JSON 配置、CLI、统一错误类型和 M1 公共契约。
- source/messages/quality/annotations/usage 的严格导入及工具调用关系校验。
- 分词 adapter、assistant-only mask、可供训练使用的 labels。
- 技能证据的监督交集、重复区间并集、share/density/绝对量、缺失与零分母处理。
- assistant 回合、结构化工具调用、工具观察、上下文与监督 token 分别计数。
- 独立保留前向、梯度、候选训练和已接受训练历史；显式排除/复用政策。
- 不裁剪超长轨迹；输出稳定排序的数据池、排除清单、审计、内容及文件 checksum。
- 完整产物发布后，下游可验证读取，坏记录或坏 checksum 不悄悄回退。
- 三条小型合成 fixture：重叠技能证据、缺失标注、工具长输出。

原始 `docs/self_curate_codex_blueprint.md` 保留原样。没有导入相邻项目，没有接模型/GPU、
下载数据、运行容器或训练，也没有 Git 提交/推送。

## 验收状态

新项目目录已完成全量验收：**44 passed、0 failed、0 skipped**，包括独立手算 token 对齐，以及 100 组随机区间的逐 token 集合对照。
CLI、三条 fixture 导入、重复只读校验、black、flake8 F/E9、compileall、wheel 构建/CRC/独立导入均通过。
证据位于忽略的 `runs/acceptance/`：pytest.xml、verification.json、demo/、各命令日志与源码交付 zip。
打包初次在受管暂存环境遇到临时目录 ACL 限制；在获授权的新目录以正常权限构建通过，未修改构建算法或全局环境。

| 层次 | 状态 |
|---|---|
| M1 结构、数据和数值语义 | 已实现，CPU 测试通过 |
| fixture CLI 导入与池读取 | 已通过测试，显式 synthetic |
| 外部预计算 token 接口 | 已实现并通过手工独立对齐样例；不等于真实 tokenizer 已实测 |
| 模型专用 tokenizer/chat template adapter | 尚未实现，后续接入 |
| 真实原始来源/证据真实性校验 | 未执行，audit 明确标 false |
| M2 算子、Agent、梯度、训练与实验结论 | 本次范围之外 |

## 近期按轮推进

| 轮次 | 交付目标 | 验收标准 |
|---|---|---|
| 下一轮 A | M2/P1 的 ProgramSpec、执行上下文、算子注册与固定质量门禁；先做 Skill/Evidence/Length/Dedup | 直接读 M1 池；缺失政策明确；不改轨迹；纯过滤交换顺序不凭空产生差异 |
| 下一轮 B | 完成 NLL/Entropy/Sample/Mix 与 BudgetPack | 先使用显式 fixture 信号；分支重叠只计一次；token 配额与残差可复核；不可行明确返回 |
| 随后 C | P2 模拟内外层循环与恢复 | 先升后降仍返回历史最佳的精确批次；接受切身份、拒绝留参数；恢复不重复提交 |
| 随后 D | P3 小模型真实 tokenizer、mask、前向与样本梯度 | 真实模板对齐；参数不更新；按样本缓存与身份失效有独立对照 |

先做确定性算子，再让 Agent 调用这些工具。三角色 prompt/记忆与真实训练在相应阶段接入，
不为了“先搭大框架”提前创建不能使用的空模块。

## 下一轮可以直接给 Codex 的任务

```text
阅读 AGENTS.md、docs/self_curate_codex_blueprint.md、docs/decisions.md、
docs/m1_interfaces.md、docs/progress.md。

本次执行 M2/P1 的第一小步：ProgramSpec 与纯筛选算子。
以已交付 M1 的 read_pool() 为输入，不重写数据规范化、证据分母或监督 mask。
实现严格 DAG schema/算子注册、固定 QualityGate，以及 SkillSelect、EvidenceSelect、
LengthSelect、基于精确内容/已标注组的 DedupSelect。
明确 EvidenceSelect 缺失策略，并区分缺失与已标注零证据。
拒绝未知字段、环、未知依赖和任意代码执行。
保存节点输入输出 ID、理由、程序 hash；相同输入和配置结果一致。
本轮不接真实模型、语义去重服务、NLL/entropy 计算或训练。
测试输入不变性、缺失、重叠、质量门禁、过滤换序等；跑全量项目测试。
同步公共接口、决策与进度，交付可运行命令与下一小步输入。
```

完整 M2/P1 验收后，再执行下一轮 B；不能把部分算子完成标成八算子全阶段完成。
