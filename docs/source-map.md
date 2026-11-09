# 代码来源与整理范围

来源由“撰写简历初版”会话核对，并实际读取本地文件确认。原实验文件与大体量结果保持原位。本仓库保存代码副本和精简证据。

|原入口|整理后|处理|
|---|---|---|
|`ai_infra_md/overnight_core.py`|`inferbench/core.py`、`metrics.py`、`server.py`、`io.py`、`sse.py`|保留请求结果字段；拆出生命周期、计数器、SSE 和存储；移除个人路径与隐式旧结果缓存|
|`overnight_assets.py`|`experiments/prepare_tasks.py`|仅保留任务数据准备；显式模型目录、输出目录与 revision；移除大模型自动下载|
|`overnight_quality.py`|`inferbench/quality.py`、`perplexity.py`|保留数学、隔离代码测试和 prompt NLL；改为显式输入|
|`prepare_self_quant.py`、`quantize_own_awq.py`|`experiments/quantization/`|保留实际成功的 AutoAWQ 路径，统一环境变量路径|
|`quantize_own_checkpoint.py`|同名量化脚本|保留 GPTQ 分支；删除未作为成功主线的 GPTQModel AWQ 尝试|
|`calibrate_overnight_kv.py`、`kv_scale_audit.py`|同名量化脚本|明确 Qwen3-8B 36 层范围；保留 scale-only shard、108 个尺度核验；增加 hook 清理|
|`spec_nvtx_extension.py`、`summarize_overnight_nsight.py`|`experiments/profiling/`|保留诊断 scope 与 launch correlation；输出目录参数化|
|WSL `RMSNorm_test.py`、`a01_*`、`a02_*`、`a03_*`、`w08_*`|`experiments/kernels/`|保留选中实现与数值参考；精简过时作业说明；RMSNorm 改为 main 入口，导入时不启动 GPU 测试|
|WSL `w10_bench_concurrency.py` 的注入思路|`inferbench/mixed.py`|单独保留“短请求限流、长请求绕过 semaphore”语义；统一 token ID 客户端与时间单位；这是新的整理实现，不宣称与历史 JSONL 逐字段兼容|

CLI、配置、报告、测试、模拟服务、目录文档为本次整理新增。`docs/source-map.json` 保存迁移前文件的 SHA256，便于追溯；指纹不是整理后文件的哈希。

未纳入主入口：`launch_*`、`supervise_*`、`finish_*`、`finalize_*` 等一次性补跑/收尾脚本、旧版扫描器、报告页面生成器、简历个人信息、完整数据集、模型权重和大型 Nsight trace。历史输出一致性审计与多轮 γ 扫描尚未包装成一键流程，现阶段通过显式配置和独立运行完成。

`evidence/native-kv-summary.json` 来自 2026-11-08 原生 KV 补测已有分析，包含原分析文件 SHA256。它不证明整理后入口已经重跑得到相同结果。环境快照去除了本地 editable/URL 安装路径，仅作历史环境记录；不应将三个环境拼成一个 requirements 文件。
