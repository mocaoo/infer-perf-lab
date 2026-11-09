# 本次整理验证

验证日期：2026-11-09。

|检查|结果|
|---|---|
|Windows CPU/HTTP 测试|15 项中 13 项通过，2 项 Linux 进程测试按平台跳过|
|Ubuntu-24.04 / WSL CPU/HTTP 测试|15 项全部通过，含端口占用保护、启动超时清理|
|Python 编译|`inferbench`、`experiments`、`tests` 通过|
|Triton 计算体保护|19 个 `@triton.jit` 函数 AST 与来源一致，排除 docstring/格式变化|
|安装与模拟示例|editable 安装、CLI help/plan 和模拟服务到报告链路通过|
|格式与轻量静态检查|Black 全部文件通过；客户端、量化、诊断、准备脚本与测试的 Ruff F 检查通过|
|本地真实 tokenizer 准备输入|Qwen3 tokenizer + 本地 WikiText 文本生成 11 条不重叠长短输入，manifest 校验通过；未执行模型 forward|
|结果展示页|16 张内嵌 SVG、23 个表格全部加载，桌面和 390px 移动宽度无页面溢出，无脚本错误；Markdown 的 16 张 PNG 相对路径全部存在|
|真实 GPU 性能回归|本次未执行|
|AWQ/GPTQ 重新校准导出|本次未执行；保留历史成功入口与环境记录|
|完整 Nsight 重新采集|本次未执行|

HTTP 测试使用本地 aiohttp 模拟服务，覆盖 UTF-8/SSE 跨传输分片、一个 chunk 多 token、usage 不一致、流截断、HTTP 错误、并发上限、预热与测量计数分离、完整 CLI 到报告路径、拒绝覆盖、长请求绕过短请求 semaphore。离线检查覆盖计数器 reset/系列消失、带空格标签与时间戳、精确指标名、数据划分重叠、线性 P99 和 JSON 原子写入。

直接审查发现并修复整理过程中的 AutoAWQ 目标路径初始化顺序、校准 hook 异常清理与审计输出目录创建问题。没有改变 Triton 计算体；只保证保留原实现，不保证原实现支持任意 shape。

下一次真实 GPU 验收应选一个小而完整的配对：BF16 和一种投机方案，同 manifest、输出长度、并发与 KV 设置；各独立启动、保存日志并核对生成计数，再执行数值微基准。完成后才将整理后的性能结果替换或补入 `evidence/`。
