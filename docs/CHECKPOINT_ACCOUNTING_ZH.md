# Checkpoint统计与实验一汇总

最新确认：Initial、Full、Random5/10/20/30%，三个领域；Full/Random全部3 epoch，保存每个epoch但仅测评epoch3。18分支、15次训练、45 checkpoint、72 benchmark结果。Random50不再执行；旧run不可改写为新协议，必须新plan/run_id。

## 训练计量

training/checkpoint-STEP/training-cost.json随原生checkpoint发布，内容：unique_examples、sample_exposures、input_tokens、supervised_tokens、micro_batches、global_step、epoch、train_wall_seconds、gpu_count、gpu_hours、native_input_tokens_seen、session_id、resume_checkpoint。

- unique_examples：本分支唯一训练样本数，不随epoch增加。
- planned_sample_exposures：唯一样本数×epoch，只是计划。实际sample_exposures在Trainer实际完成的microbatch计数，8 rank求和，包含分布式sampler补齐导致的重复，不假设正好等于计划。
- input_tokens：实际attention_mask有效位置，排除padding；监督token只计因果移位后labels非-100的位置。题面/system/模板等非监督内容不会误算为监督。
- micro_batches：所有rank实际完成microbatch总数；global_step为全局optimizer步，两者不能混用。
- 时间：trainer.train和原生epoch保存期间的全rank最大墙钟时间，恢复时累加已有checkpoint时间；不含模型初次加载/预编码。gpu_hours=该时间×8/3600，为分配GPU时长估计，不等于利用率或实际货币费用。
- 每rank的events.jsonl按session记录consumed_work、session start/exit、GPU显存；硬杀可能丢失最近日志，checkpoint之外重做/丢失工作的完整费用不能伪称已精确恢复。

计数不在DataLoader/collator做，以免prefetch计算尚未训练的数据；常规microbatch只更新设备计数器，发布时同步，避免每步item造成GPU同步。

## 测评计量

每题预测保存模型返回的input/output token，generation.json汇总。evaluation-cost.json记录输入/输出token、生成墙钟、并发请求耗时之和、CPU判分墙钟、整次评测墙钟。整次包含必要的vLLM启动；多任务并行时请求耗时之和不能当真实墙钟。

## 自动交付表

| 文件 | 行/用途 |
|---|---|
| summary.csv | 72行：领域、方法、checkpoint、benchmark、成绩、模型identity、独立样本、计划/实际曝光、训练token/时间、生成token/时间、判分时间及原始文件路径 |
| training_costs.csv | 45行，每checkpoint累计成本与该epoch增量；训练费用在这里汇总，不能把summary重复的训练列相加 |
| scores_wide.csv | Initial与每个最终checkpoint一行，各benchmark一列，用于主表排版 |
| REPORT.md | 可直接阅读的成绩/成本表；epoch3预设主结果，1/2不挑最好 |
| report-manifest.json | 行数和plan hash |

Initial训练量为0；未测量值保持空白/null，mock不会伪造真实吞吐/成绩。无实际单价时monetary_cost=null。成绩不能跨benchmark简单平均并当同一种量纲。

汇总输出在results/<UTC时间-随机ID>/，旧run恢复身份严格校验。结果zip包含上述表格，不能拿旧无成本checkpoint绕过当前计量要求。

本轮CPU验收：33项全流程smoke通过，含18分支/72结果/45成本行、恢复与幂等；生产计数器使用真实CPU tensor检查监督mask、模拟8rank归约、恢复offset，另核对45行epoch增量/48行宽表。结果zip含表格并通过CRC。验收凭据位于忽略的runs/checkpoint-accounting-acceptance-20261120。真实FSDP2/9B/vLLM/判分容器未在本机验证。
