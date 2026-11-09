# infer-perf-lab

单卡 LLM 推理性能实验仓库：把 vLLM 服务压测、投机解码、权重与 KV 量化、Chunked Prefill 混合负载，以及 Triton 算子微基准放进一条可检查的实验链路。

项目来自 RTX 5090 D / Qwen3-8B 的实际实验。重点是明确的配置、原始请求记录和有边界的结论。当前版本完成代码整理与 CPU 链路验证，整理后的 GPU 入口尚未重新跑完性能回归。

**实验结果：[图文报告](docs/experiments.md) · [独立 HTML 页面](docs/index.html)**

报告包含投机解码、量化、原生 FP8 KV、Chunked Prefill 和 Split-K 的历史实测与限制说明。Markdown 可在仓库内阅读；HTML 可下载后直接打开，或通过 GitHub Pages 展示。发布步骤见 [GitHub 发布说明](docs/publishing.md)。

## 先跑通一条链路

Python 3.10+；纯 HTTP 客户端和模拟示例支持 Windows/Linux，vLLM 服务与 Triton 实验使用 Linux/WSL。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/mock_demo.py
python -m inferbench report results/mock-demo
```

Windows 用 `.venv\Scripts\Activate.ps1` 激活。模拟示例不下载模型、不使用 GPU，演示请求 → SSE → token/usage 校验 → 服务端计数器对账 → JSON/Markdown 报告。模拟延迟不能用于展示推理性能。

## 目录

```text
inferbench/                 HTTP/SSE、并发、指标、进程管理、报告与质量检查
configs/                    BF16、三种投机、权重量化、FP8 KV、prefill 预算配置
experiments/
  prepare_tasks.py          MT-Bench 首轮 / HumanEval / GSM8K 数据准备
  prepare_mixed.py          本地语料到精确长短 token ID 工作负载
  quantization/             成功使用过的 AWQ、GPTQ、Q/K/V scale 校准入口
  kernels/                  RMSNorm、Online Softmax、Attention、Split-K、FP8 KV
  profiling/                NVTX worker 扩展与 Nsight SQLite 关联分析
examples/mock_demo.py        无 GPU 端到端示例
tests/                      客户端、边界检查和服务清理测试
evidence/                   历史环境快照与精简结果，不含模型或完整数据集
docs/                       复现、指标口径、来源映射和本次验证记录
  index.html                含 16 张内嵌图的独立结果展示页
  experiments.md            GitHub 内可阅读的图文版
```

## 跑真实服务压测

首先在独立服务环境准备 Qwen3-8B 和匹配的 vLLM；历史环境版本见 `evidence/serving_environment_freeze.txt`。客户端安装不会安装或替换 GPU 框架。

在仓库根目录执行：

```bash
export INFERBENCH_MODELS=/path/to/models

# 使用包含 transformers / pyarrow 的数据环境；只下载任务数据，不下载模型。
python -m pip install -e '.[data]'
python experiments/prepare_tasks.py \
  --model-root "$INFERBENCH_MODELS" --output-dir data/tasks

# 先检查实际命令；plan 不启动服务。
python -m inferbench plan configs/bf16.fixed.json

# 自动管理本次创建的服务进程，保存命令、日志、输入和结果。
python -m inferbench bench \
  --server-config configs/bf16.fixed.json \
  --manifest data/tasks/datasets/manifest.json \
  --task chat --concurrency 1 --requests 32 --repeats 3 \
  --run-dir results/bf16-chat-r1
```

已有服务可省略 `--server-config`，传 `--base-url` 和 `--model`。客户端不会接管已有进程。正式对账应使用独占服务；其他客户端流量会使生成 token 计数不一致并导致验收失败。

每次运行使用新目录。成功时输出 `protocol.json`、冻结的 `manifest.json`、`cases/*.json`、`summary.json`、`report.md` 和 `status.json`；失败时返回非零退出码并保存状态。缺少 usage/token ID、流截断或计数器边界不一致都会拒绝给出有效性能摘要。

## 对照实验

|问题|入口|控制项|
|---|---|---|
|EAGLE3 / DFlash / DSpark 收益与 γ 边界|`configs/{eagle3,dflash,dspark}.fixed.json`|先在 tune 输入选 γ，再冻结并测 heldout；示例 γ 不是通用最优值|
|权重量化速度|`configs/{bf16,fp8,awq,gptq}.fixed.json`|相同 tokenizer、任务、输出长度和固定 3 GiB KV 池|
|原生 FP8 KV 容量|`configs/{bf16-triton,fp8-kv}.capacity.json`|相同 TRITON_ATTN 与 80% 显存预算；KV 池自动分配|
|长输入对短请求的影响|`inferbench mixed` 与 `configs/chunked-{512,8192}.json`|相同输入与注入计划，仅修改 prefill token 预算|
|算子优化|`experiments/kernels/`|数值参考与 CUDA Event 微基准，独立于服务性能|

将相同 manifest 用于各方案，并至少另起一轮服务、反转方案顺序。不要在同一次正式测速中载入 NVTX 诊断扩展。更完整操作见 [复现说明](docs/reproduce.md)。

混合负载 manifest 的 `task` 用 `short` / `long`，短任务还需 `tune` 行。注入请求绕过短请求的客户端并发槽，并记录实际发送时间：

```bash
python experiments/prepare_mixed.py \
  --tokenizer "$INFERBENCH_MODELS/Qwen3-8B" --corpus /path/to/corpus.txt \
  --output data/mixed-manifest.json

python -m inferbench mixed \
  --server-config configs/chunked-512.json \
  --manifest data/mixed-manifest.json --concurrency 8 --burst-delay 3 --output-tokens 512 \
  --run-dir results/chunked512-r1
```

`configs/prompts.example.json` 仅示范 token ID 输入格式。真实 manifest 需要用目标模型的 tokenizer 生成，不能把几条示例 token 当成工作负载。输入重复用于增加负载时，应保留原始内容身份；重复不增加独立内容样本数。

## 指标口径

- TTFT：发出请求到首个非空文本 SSE 事件，包含等待、prefill 与传输。
- 请求级 TPOT：`(末个可见文本时间 − 首个可见文本时间) / (usage 输出 token 数 − 1)`。
- chunk gap：相邻可见文本事件的间隔。一个事件可携带多个 token，不能称作逐 token ITL。
- 吞吐：有限批次实际输出 token 总数 / 客户端批次墙钟时间。
- 接受率：服务端 accepted draft tokens / draft tokens；另记 accepted tokens / draft rounds。
- P99：每轮请求级 TPOT 线性插值，再对各轮取中位数；不合并所有轮次样本。

统计、质量及因果解释边界见 [测量说明](docs/measurement.md)。

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q inferbench experiments tests
```

Windows 验证通过；Linux/WSL 还验证占用端口保护和启动失败清理。19 个 Triton JIT 函数的 AST 与原版一致（忽略注释、docstring、格式）。这些检查不代替 GPU 数值与性能回归。详情见 [验证记录](docs/validation.md)。

代码来源与删减范围见 [整理说明](docs/source-map.md)。
