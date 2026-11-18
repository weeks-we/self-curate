# SDK修复服务器交接

先读 [逐项修复记录](../SDK_REPAIR_20261118_ZH.md)。本轮28项CPU专项、3×5嵌套mock、EXP001四分支mock、恢复及两份ZIP CRC通过。真实SDK、GPU和Docker未运行。

Terminal跨进程verifier-only尚需实际冻结Harbor commit；不要用整题重跑替代。先提供构建时HARBOR_COMMIT或已安装harbor的direct_url.json。其他平台验收按修复记录执行。

在新源码目录、新冻结身份和新输出根验收，旧run只读。禁止改旧snapshot/source hash绕过恢复校验。先做单题工程fixture，包含清理失败、SWE/Pro判分恢复、取消及原生压缩，再做原预算容量与数值验收。模态、研究预算、数据与模型配置沿用本项目，不复制NICE五模型600题协议。
