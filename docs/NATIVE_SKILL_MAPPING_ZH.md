# 三领域预设技能与原生字段映射

2026-11-21用户确认：保留原先所有预设类别，即Coding八类、数学七类、IF十类；少量未映射样本先不标，不自动调用teacher。配置权威：[skills.json](../configs/ordinary/skills.json)，映射实现：[ordinary.py](../src/self_curate/preparation/ordinary.py)。

## 共同规则

- 保留原生字段、raw_labels、内容hash和标签版本。允许多标签，不强行指定唯一主技能。
- 未映射项保留在原候选／训练池，label_status=missing；不猜测、不强贴相近标签，不因缺少技能标签从Full/Random训练池删除。
- 未映射／映射不完整的记录暂不进入按技能抽取的开发集。未知IF约束另记metadata.unmapped_constraints，已识别部分仍保留。
- 保留原本数学probability和IF style，即使其当前原生统计为0。零成员不表示删除类别，也不自动从组合、语气等文本推断。
- 每技能50题规则不变：类别没有50个合格开发候选时输出shortages，不复制、不编造、不把未覆盖类别当已完成；训练和benchmark结果照常保留。

## Coding：BAAI/TACO，字段skill_types

| 标签ID | 名称 | 原生skill_types值 |
|---|---|---|
| amortized_analysis | 摊还分析 | Amortized analysis |
| bit_manipulation | 位运算 | Bit manipulation |
| complete_search | 完全搜索／穷举 | Complete search |
| data_structures | 数据结构 | Data structures |
| dynamic_programming | 动态规划 | Dynamic programming |
| greedy_algorithms | 贪心算法 | Greedy algorithms |
| range_queries | 区间查询 | Range queries |
| sorting | 排序 | Sorting |

skill_types可含多个值，逐项保留；统一小写、非字母数字转下划线。tags/raw_tags只保留为原始元数据，不把它们无规则地混入这八类。原生出现未预设类别时不会自行发明新技能或映射为最近一类。

## 数学：NuminaMath-1.5，字段problem_type

| 标签ID | 名称 | 原生problem_type值 |
|---|---|---|
| algebra | 代数与不等式 | Algebra；Inequalities |
| geometry | 几何 | Geometry |
| number_theory | 数论 | Number Theory |
| combinatorics | 组合 | Combinatorics |
| probability | 概率与统计 | 兼容Probability／Statistics；当前已观察原生类别没有这两项，保留空位 |
| calculus | 微积分与极限 | Calculus |
| logic | 逻辑与谜题 | Logic and Puzzles |

Other、空值和其他未映射类别：skills=[]、label_status=missing，原始problem_type仍保留，不执行teacher补标。数学原生problem_type通常为单值，因此这一层一般生成一个技能；不从解答内容推断额外标签。question_type是证明／选择／数值题等题型，source是来源，二者不直接当技能。证明题仍可SFT，但不能进入自动判分开发集。

## IF：Tulu Personas Instruction Following，字段constraints

按每个约束字符串第一个冒号之前的类别名映射；去首尾空白、统一小写，并将空格／下划线／连字符统一。原始约束内容完整保留。

| 标签ID | 名称 | 本数据已观察到的原生类别 | 兼容类别名 |
|---|---|---|---|
| lexical | 词汇／关键词 | keywords；include keywords | keywords |
| format | 格式 | format；use quotation | detectable_format |
| length | 长度约束 | length constraints | length_constraints；length |
| required_content | 指定内容 | content | detectable_content |
| boundary | 开头／结尾边界 | specific ending | startend |
| language | 回答语言 | response language；in english and capital | language |
| case | 大小写 | case；in english and capital | change_case |
| punctuation | 标点 | punctuation | punctuation |
| composition | 组合指令 | give two responses；repeat the prompt | combination |
| style | 风格 | 当前数据未出现直接对应类别；保留空位 | style |

in english and capital必须同时满足语言和大小写，因此同时赋language和case，不能只保留一项。一个样本多个约束的技能取并集；各技能成员数之和可能超过样本数。

示例：constraints=["length constraints:number of words", "specific ending:..."] → skills=["boundary","length"]。这是技能映射，不代表已经重建官方checker所需的参数或验证参考答案满足所有约束。

## 当前覆盖事实与后续边界

IF原始29,980条均可按原生约束映射，无整条缺标，覆盖九个预设类别，style目前0成员。数学2000条抽查中，质量准入后1624条有15条Other未映射；probability没有直接原生映射。正式去重／隔离／长度筛查后的分布会变化，最终以各池distribution.json为准。

本轮仅冻结“保留标签体系＋原生映射＋未映射先不标”的口径。数学概率与IF风格的细分／补标留后续；不自动修改题量、类别或付费预算。实验一主训练／benchmark不依赖这些空位，但包含这些类别的完整每技能50开发集目前不能宣称已经齐备。
