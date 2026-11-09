# 测量口径与结论边界

## 请求时序

`request()` 保留全部携带 token ID 或文本的 SSE 事件；TTFT 和 chunk gap 只使用非空文本事件。usage 的 prompt token 数须等于输入 ID 数；输出 ID 覆盖数须等于 completion_tokens。固定输出时还要求输出长度等于请求值。

TPOT 沿用原项目的“首末可见输出跨度 / (输出 token 数 − 1)”定义。一次 SSE 返回多个 token 时无法恢复逐 token 时间；只有一个可见 chunk 时，即使它包含多个 token，跨度仍为 0。这是传输观测的分辨率限制，不能据此宣称引擎零延迟。单 token 输出 TPOT 为 null。

E2E 使用请求开始至读完 SSE 的墙钟，包含尾部 usage 和流结束。客户端 semaphore 之前的排队不计入请求级 TTFT；批次吞吐的墙钟包含这一排队和最后请求排空。混合负载的延迟释放也计入总墙钟。

## 对账

预热使用 tune 输入，并使用与正式请求相同的输出上限；不回退到 heldout。等待预热指标入账后读计数器。正式请求完成后短暂轮询，要求生成 token 增量与客户端输出 token 总量相符。计数器重置、系列消失、缺失或额外流量都使本轮无效。

Prometheus 保留完整 label，不把不同 position 的投机指标覆盖。计数器按精确名称求和，避免把直方图或同名前缀指标混入。多个 model/worker 标签是否重复暴露取决于实际部署；本项目的精确总量校验会拒绝明显重复计数，正式运行仍需核对指标含义。

benchmark 模式周期采集队列、KV 占用等指标；mixed 模式保留请求与边界快照。两种模式的输出 schema 单独标注；mixed 不声称拥有 benchmark 的完整计数器验收。

## 统计与对照

同一方案每轮给出请求 TPOT/TTFT 中位数、请求 TPOT P99 和批次吞吐，再对轮次取中位数。跨服务重启比较时，应先计算每次配对倍率，再汇总倍率。两次启动的中位数不能替代置信区间或显著性检验。

固定 3 GiB KV 池用于减少容量差异；固定 80% 总显存、自动分配 KV 池用于容量实验。后者的速度差异同时包含容量、调度和内核效应，不能全归因于单个 kernel。

γ 增大后的收益回退只说明当前输入与配置下性能下降。历史 DFlash code γ16/32 和另一组 math γ15 NVTX trace 未严格匹配，不能拼接成同一点位的瓶颈因果证明。

混合负载用客户端长请求实际发送时间查找跨界 chunk。发送时间不是服务端 prefill 开始时间；跨界间隔仅作关联证据。若没有正在输出的短请求跨过注入时刻，应调整负载并另跑，不能把该轮写成成功复现干扰。

## 质量与诊断

greedy 模式与相同 seed 不保证不同投机执行路径逐 token 相等。原路径 logprob、共同前缀重提交、跨路径输出一致性是不同检查；本仓库没有把历史全套输出一致性审计改写成通用“无损证明”。

数学质量区分严格 `####` 格式和末尾数字宽松提取。HumanEval 使用 bubblewrap；沙箱不可用返回 `sandbox_unavailable`，不能算作模型错误，也不会回退到本机裸执行。此隔离是原实验方案的整理，不是面对恶意代码的完整安全评测平台。

PPL 使用不重叠的 2048-token 窗口、teacher-forced prompt logprob，跳过每窗第一个无条件概率的 token；不是滑动窗口官方分数，也不能单独证明 cached decode 量化质量。

NVTX CPU 范围内异步提交 GPU 工作，须关联 CUDA runtime launch 与 kernel。`target_execute` 包含准备、prefill 和 decode/verify，不等于纯验证时间。GPU span 含间隙、kernel union 去重计时；不能把各范围中位数相加当作端到端分解。GPU busy 时间覆盖率也不等于 SM/Tensor Core 利用率。
