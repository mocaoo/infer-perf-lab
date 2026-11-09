# 复现与扩展

以下命令从仓库根目录执行。GPU 环境使用已有验证环境，并在每次运行中记录实际版本。三个历史 freeze 文件分别属于服务、GPTQ、AutoAWQ 环境，依赖不同，单独维护。

## 数据和输入

`prepare_tasks.py` 将数据 revision、下载文件哈希、token ID、原始任务身份和参考答案写入 manifest。`--revisions revisions.json` 可传 `{ "chat": "<commit>", "code": "<commit>", "math": "<commit>" }` 固定数据版本；未指定时解析一次远端 revision 并写入结果。每次准备使用新目录，防止复用旧下载但写入新 revision。

原始协议对每类输入独立打乱，先划 8 条 tune，再将其余作为 heldout；超过 7000 token 的条目过滤。准备后确认各类仍有足够 tune 样本。CLI 会检查 ID 唯一、token ID 非负和 tune/heldout 精确 token 序列不重叠。它不声称消除了语义近重复或预训练污染。

长短混合输入可用 `experiments/prepare_mixed.py` 从本地语料生成 manifest：`short/tune` 用于预热，`short/heldout` 用于持续负载，`long/heldout` 用于注入。使用确定性、互不重叠的语料区间，记录来源偏移与哈希；重复内容仍由 CLI 检查。长短由实际 token ID 长度定义，`task` 名称不隐式截断/填充。历史方案为 38 条短请求和 2 条长输入、短请求并发 8、延迟 3 秒注入、固定输出 512；本仓库不内置数据全文。

## 投机和量化对照

修改配置内 `speculative_config.num_speculative_tokens` 即可扫 γ。先只用调参池挑参数，另存冻结配置，再测留出池。CLI 的 `bench` 固定使用 heldout；调参时请准备独立的调参实验 manifest，以清楚保留来源，不能将正式 heldout 改作调参数据。

每种 arm 每次重启使用不同 `--run-dir`。配置中的 draft 模型目录必须与目标 Qwen3-8B 和安装的 vLLM 方法实现匹配。`.fixed` 使用相同 3 GiB KV 池；`.capacity` 使用相同总显存预算、自动分配 KV。显式选择 Attention 后端，不按模型名称偷偷改变后端或自动切换已存在的校准模型。

## 自主 AWQ / GPTQ

```bash
export INFERBENCH_MODELS=/path/to/models
export INFERBENCH_QUANT_RESULTS="$PWD/results/self-quant"
export INFERBENCH_TEST_CORPUS=/path/to/wikitext2_test.txt

# 在数据/transformers 环境运行，读取 WikiText train，冻结 128×512 token。
python experiments/quantization/prepare_self_quant.py

# 分别在隔离的 AutoAWQ / GPTQModel 环境中运行；均需安装本仓库客户端包。
python experiments/quantization/quantize_own_awq.py --samples 128
python experiments/quantization/quantize_own_checkpoint.py gptq --samples 128
```

32 段是 128 段的前缀子集；两算法共用冻结 token 输入。输出为 `Qwen3-8B-Own-awq_n128` / `Qwen3-8B-Own-gptq_n128`，并记录配置、导出权重 SHA256、校准时间与显存。检测到已有导出目录会拒绝覆盖。中断导出应由操作者检查后另设干净目录。准备脚本仍校验历史 WikiText train 文件的固定 SHA256；上游数据改变时会失败，不能静默替换数据后比较。

## 原生 FP8 KV

```bash
export INFERBENCH_MANIFEST="$PWD/data/tasks/datasets/manifest.json"
export INFERBENCH_KV_RESULTS="$PWD/results/kv-calibration"
python experiments/quantization/calibrate_overnight_kv.py
```

限 36 层 Qwen3-8B：chat/code/math 各 8 条 tune、每条最多 512 token，采样 Q/K norm+RoPE 后与 V projection 后 absmax，除以 448 得到每层 Q/K/V tensor scale。BF16 原权重通过符号链接复用，增加 scale-only shard 和索引；不覆盖原模型。仅 prompt forward 校准不能保证覆盖 decode 离群值。

smoke 核验时让 worker 能导入脚本目录：

```bash
export PYTHONPATH="$PWD/experiments/quantization${PYTHONPATH:+:$PYTHONPATH}"
# 把下面参数追加到 plan 打印出的 FP8 KV 启动命令，单独启动 smoke 服务：
# --worker-extension-cls kv_scale_audit.ScaleAudit
```

核验文件为 `$INFERBENCH_KV_RESULTS/calibration/runtime_scale_audit.json`；必须 count=108 且 all_match=true。随后关闭诊断服务，正式性能测试使用不含该扩展的配置。使用 `.capacity` 两臂比较时，同时记录启动日志中的权重加载显存、KV 池 GiB、token 槽位和抢占。

## 质量

生成任务输出使用 `--natural-eos --output-tokens 1024`，用保存的 manifest 中参考答案离线检查：

```bash
python -m inferbench.quality results/run/cases/baseline_math_c8.json \
  --manifest results/run/manifest.json --output results/run/quality.json

python -m inferbench.perplexity \
  --base-url http://127.0.0.1:8023 --model Qwen3-8B \
  --tokenizer "$INFERBENCH_MODELS/Qwen3-8B" \
  --corpus "$INFERBENCH_TEST_CORPUS" --output results/ppl-bf16.json
```

HumanEval 依赖 Linux bubblewrap，可用 `INFERBENCH_BWRAP` 指定二进制。不可用或权限不足时检查 evaluation status；不可把不可评估当作模型低分。

## Triton 微基准

在已有 CUDA/Triton 环境运行。正式重跑前先用小形状验证数值；矩阵大小、分块和 head 映射须满足脚本约束。这些实现用于研究特定形状，未封装成通用生产算子。

```bash
python experiments/kernels/RMSNorm_test.py
python experiments/kernels/a01_online_softmax.py --seq-lens 1024
python experiments/kernels/a02b_splitk.py --s-list 4096 --splits-list 1,2,4,8,16
python experiments/kernels/a03b_matrix.py --splits-list 32 --tag local
```

RMSNorm 有 fp64 参考、奇数隐藏维度测试及 warp 轮转；attention 有 FP64/SDPA 参考。先检查误差，再看完整 split+combine 时间。微基准可能取多次最小值；服务层报告按中位数和分位数，不混用。规格带宽/算力常量沿用原卡实验估计，换卡需修改或解释，不能当作硬件计数器实测。

## Nsight

`spec_nvtx_extension.NsightAnnotations` 只用于独立诊断运行，通过 `--worker-extension-cls` 注入。将 profiling 目录加入 `PYTHONPATH`，使用 Nsight Systems 抓取 `cuda,nvtx` 并导出 SQLite。该扩展依赖历史 vLLM V2 内部 API，升级后先 smoke，不保证跨版本。

分析约定：`results/profile/nsight/<tag>/trace.sqlite` 和 `phases.json`。后者是列表，每项包含 `task`、`concurrency`、`start_unix`、`end_unix`（同机 Unix 秒）。历史分析器完整性判据要求每份 trace 有两段非空测量窗口；其他协议需显式修改判据。

```bash
python experiments/profiling/summarize_overnight_nsight.py results/profile
```

分析器核对 SQLite 的 kernel 内容，再按 runtime correlation ID、进程与线程关联 NVTX 范围；未抓到 kernel 或关联不完整时不使用 CPU 时长代替。完整原生 Nsight 报告不纳入 Git。
