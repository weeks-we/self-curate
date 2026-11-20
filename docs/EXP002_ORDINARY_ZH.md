> Latest TACO split policy: development tasks come from train only, 50 per skill; test is reserved exclusively for final evaluation. Historical test-as-development instructions below are superseded.

# EXP002：代码领域动态性消融

独立于 EXP001。模型仅 Qwen3.5-9B，关闭 thinking。复用固定原生标签、开发题 ID、初始模型结果和初始版本指标；不复用其他训练分支的模型冒充初始权重。

任务来源是 TACO test 的 1,000 题，先统计/标注技能，再每组取 50 个唯一任务。不足时合并/调整标签需用户确认；不得凑重复题。它不再作为最终 benchmark。

- V0：初始程序与指标固定。
- V1：程序固定；每次外层训练后，依据 Lopt 下一轮粗粒度技能方向，只刷新相关且尚未用于真实参数更新的数据 NLL/entropy，并绑定新模型版本。
- V3：允许内层依据反馈修改程序；指标刷新规则与 V1 一致。

拟采用三外层、每外层五次内层，Python workflow 对接现有模块。NICE 或候选 SFT 的评估方式、选样总预算、技能分组、停止与接受条件仍须单独定稿；当前不创建可误启动的正式配置。域内选样比例必须与后续模型的对应实验一致。
