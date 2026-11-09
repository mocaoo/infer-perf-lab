#!/usr/bin/env python3
"""Standalone FP8 KV quantization/dequantization microbenchmarks.

These kernels do not replace the native vLLM KV implementation."""

import argparse
import statistics
import subprocess

import torch
import triton
import triton.language as tl

# ── 常量 ─────────────────────────────────────────────────────────────────────
FP8_MAX = 448.0  # float8_e4m3fn 的最大有限值（e4m3 = 1 符号 + 4 指数 + 3 尾数）
FP8_TINY = 1e-12  # scale 下限：防止「整行全 0」时 amax=0 ⇒ 除零

READ_BW_GBPS = 1646.5  # p01b 实测纯读带宽（GB/s），仅作参考，不参与判据
L2_MIB = 96
READ_BLOCK = 4096  # 流式读内核的 BLOCK（实测调优，见文件头）
READ_WARPS = 2

# ★★ 跑之前先填预测（判据在最后）★★
PREDICT = {
    "quant_share_of_copy": 0.76,  # quant 时间 / copy 时间（有效字节比 3.03/4.00 = 0.76）
    "read_fp8_over_bf16": 0.50,  # read_fp8 时间 / read_bf16 时间（字节比 1.0/2.0）
    # v1 实测：quant/dequant = copy 的 151%（768 GB/s）—— 映象没优化
    # v2 目标 76%；已验证 ROWS=4 可达 71% / 79% ⇒「占 copy %」口径下的一次真实优化
}


# ═══════════════════════════════════════════════════════════════════════════════
#  §量具：Triton 流式读内核（**已提供**，不用改）
#     一个 program 读 BLOCK 个连续元素、归约成一个数写到 Out[pid]。
#     实测（T*H*D = 1.34e8）：bf16 1623 GB/s / fp8 1615 GB/s（= 98.6% / 98.1% 峰值）
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit
def _stream_read(X, Out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)  # 一整块连续载入（dtype 决定字节数）
    tl.store(Out + pid, tl.sum(x.to(tl.float32), axis=0))


def stream_read(t):
    """跑一次流式读。t 是**一维连续**张量（bf16 或 fp8 都行）。"""
    n = t.numel()
    out = torch.empty(n // READ_BLOCK, device=t.device, dtype=torch.float32)
    _stream_read[(n // READ_BLOCK,)](t, out, BLOCK=READ_BLOCK, num_warps=READ_WARPS)


@triton.jit
def _kv_quant_kernel(X, Xq, S, D, BLOCK: tl.constexpr):
    """bf16 [T*H, D] → fp8_e4m3 [T*H, D] + fp32 scale [T*H]（每行一个 scale）。

    规格见文件 §规格 kernel ①。实现后删掉下面这一行。
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + pid * D + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    sc = tl.maximum(amax / 448.0, 1e-12)
    xq = tl.clamp(x / sc, -448.0, 448.0).to(Xq.dtype.element_ty)
    tl.store(Xq + pid * D + offs, xq, mask=mask)
    tl.store(S + pid, sc)


# ── v2：行块化版本（把 ROWS 行合成一个 2-D tile 交给一个 program）─────────
#   动机：v1 是「1 program = 1 行（128 元素）」⇒ 每 lane 只有 128/32=4 个元素 = 4 B，
#   达不到 16 B 的向量化宽度；而且 grid 有 1,048,576 个微型 CTA。
#   v2 用 [ROWS, D] 的 2-D tile，让每 lane 拿到 16 B 以上，同时把 CTA 数降到 1/ROWS。
#   规格（与 v1 同样的数学，只是搬成 2-D）：
#     ① pid = tl.program_id(0)                          # 负责第 pid 块 ROWS 行
#     ② r   = pid * ROWS + tl.arange(0, ROWS)[:, None]   # [ROWS,1] 行号
#     ③ c   = tl.arange(0, BLOCK)[None, :]               # [1,BLOCK] 列号
#     ④ m   = c < D                                      # [1,BLOCK]，与 [ROWS,BLOCK] 广播
#     ⑤ x   = tl.load(X + r * D + c, mask=m, other=0.0).to(tl.float32)   # [ROWS,BLOCK]
#     ⑥ amax = tl.max(tl.abs(x), axis=1)                 # ★ axis=1（行内归约）→ [ROWS]
#     ⑦ sc   = tl.maximum(amax / 448.0, 1e-12)            # [ROWS]
#     ⑧ xq   = tl.clamp(x / sc[:, None], -448.0, 448.0).to(Xq.dtype.element_ty)
#     ⑨ tl.store(Xq + r * D + c, xq, mask=m)
#     ⑩ tl.store(S + pid * ROWS + tl.arange(0, ROWS), sc)
#   注意：`sc[:, None]` 把 [ROWS] 变成 [ROWS,1] 才能和 [ROWS,BLOCK] 广播 ——
#   这和 v1 里标量直接参与逐元素运算是同一件事，只是这里 scale 是**一行一个**。


@triton.jit
def _kv_quant_kernel_tiled(X, Xq, S, D, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """v2：量化，一个 program 处理 ROWS 行。规格见上方注释。"""
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)[:, None]
    c = tl.arange(0, BLOCK)[None, :]
    mask = c < D
    x = tl.load(X + r * D + c, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    sc = tl.maximum(amax / 448.0, 1e-12)
    xq = tl.clamp(x / sc[:, None], -448.0, 448.0).to(Xq.dtype.element_ty)
    tl.store(Xq + r * D + c, xq, mask=mask)
    tl.store(S + pid * ROWS + tl.arange(0, ROWS), sc)


@triton.jit
def _kv_dequant_kernel_tiled(Xq, S, Y, D, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """v2：反量化，一个 program 处理 ROWS 行。规格同 v1 的量 ② + 2-D 化。"""
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)[:, None]
    c = tl.arange(0, BLOCK)[None, :]
    mask = c < D
    xq = tl.load(Xq + r * D + c, mask=mask, other=0.0).to(tl.float32)
    sc = tl.load(S + pid * ROWS + tl.arange(0, ROWS))[:, None]
    tl.store(Y + r * D + c, (xq * sc).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _kv_dequant_kernel(Xq, S, Y, D, BLOCK: tl.constexpr):
    """fp8_e4m3 [T*H, D] + fp32 scale [T*H] → bf16 [T*H, D]。

    规格见文件 §规格 kernel ②。实现后删掉下面这一行。
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    xq = tl.load(Xq + pid * D + offs, mask=mask, other=0.0).to(tl.float32)
    sc = tl.load(S + pid)
    tl.store(Y + pid * D + offs, (xq * sc).to(Y.dtype.element_ty), mask=mask)


# ── kernel 启动器（已写好，不需要动）────────────────────────────────────────


def quant_fp8(x2d, q2d, s1d, num_warps):
    """x2d:[T*H,D] bf16 → q2d:[T*H,D] fp8 + s1d:[T*H] fp32。"""
    rows, D = x2d.shape
    BLOCK = triton.next_power_of_2(D)  # D=128 时 BLOCK=128；D 非 2 的幂时向上取整 + mask
    _kv_quant_kernel[(rows,)](x2d, q2d, s1d, D, BLOCK=BLOCK, num_warps=num_warps)


def quant_fp8_tiled(x2d, q2d, s1d, num_warps, rows):
    """v2 启动器：grid = rows/ROWS，一个 program 处理 ROWS 行。"""
    n_rows, D = x2d.shape
    assert n_rows % rows == 0, f"rows={n_rows} 不能被 ROWS={rows} 整除"
    BLOCK = triton.next_power_of_2(D)
    _kv_quant_kernel_tiled[(n_rows // rows,)](
        x2d, q2d, s1d, D, ROWS=rows, BLOCK=BLOCK, num_warps=num_warps
    )


def dequant_fp8_tiled(q2d, s1d, y2d, num_warps, rows):
    """v2 启动器：反量化。"""
    n_rows, D = q2d.shape
    assert n_rows % rows == 0, f"rows={n_rows} 不能被 ROWS={rows} 整除"
    BLOCK = triton.next_power_of_2(D)
    _kv_dequant_kernel_tiled[(n_rows // rows,)](
        q2d, s1d, y2d, D, ROWS=rows, BLOCK=BLOCK, num_warps=num_warps
    )


def dequant_fp8(q2d, s1d, y2d, num_warps):
    """q2d:[T*H,D] fp8 + s1d:[T*H] fp32 → y2d:[T*H,D] bf16。"""
    rows, D = q2d.shape
    BLOCK = triton.next_power_of_2(D)
    _kv_dequant_kernel[(rows,)](q2d, s1d, y2d, D, BLOCK=BLOCK, num_warps=num_warps)


# ── torch 参考实现（判断手写 kernel 的**算法一致性**，不是计时对象）──────────


def ref_quant(x2d):
    """fp32 中间量的参考量化。返回 (q, s)，与手写 kernel 的输出**应当逐位相同**。"""
    xf = x2d.to(torch.float32)
    amax = xf.abs().amax(dim=-1)  # [rows]
    s = (amax / FP8_MAX).clamp_min(FP8_TINY)
    q = (xf / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, s


def ref_dequant(q2d, s1d):
    """参考反量化。返回 bf16 [rows, D]。"""
    return (q2d.to(torch.float32) * s1d.unsqueeze(-1)).to(torch.bfloat16)


# ── 正确性比对 ──────────────────────────────────────────────────────────────


def compare_bits(q_triton, q_ref):
    """逐位比较两个 fp8 张量。返回 (不一致个数, 总个数)。"""
    a = q_triton.view(torch.uint8)  # fp8_e4m3fn 与 uint8 同为 1 字节 ⇒ view 零拷贝
    b = q_ref.view(torch.uint8)
    return (a != b).sum().item(), a.numel()


def error_stats(x2d, y2d):
    """round-trip 误差：x（原始 bf16） vs y（dequant 回来的 bf16）。"""
    xf = x2d.to(torch.float32).reshape(-1)
    yf = y2d.to(torch.float32).reshape(-1)
    d = (xf - yf).abs()
    rms_x = xf.pow(2).mean().sqrt().item()
    rms_d = d.pow(2).mean().sqrt().item()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rms_abs": rms_d,
        "rms_x": rms_x,
        "rel_rms": rms_d / rms_x if rms_x > 0 else float("nan"),
    }


# ── 计时（与 p01 / p01b 同一套口径）────────────────────────────────────────


def time_once(fn, inner):
    """用一对 CUDA event 夹住 inner 次调用，返回**单次平均毫秒数**。

    fn    —— 零参可调用对象（一次调用 = 完成一次被测操作）
    inner —— 内层重复次数，用来摊薄 CUDA event 本身约 2 µs 的开销
    """
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(inner):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / inner


def bw_gbps(nbytes, ms):
    """有效带宽（GB/s）—— 一律用 1e9 字节 / 1e3 毫秒换算（与 p01b 一致）。"""
    return nbytes / (ms * 1e-3) / 1e9


def gpu_state():
    """采一次 GPU 状态（污染的另一条独立证据链）。"""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return out
    except Exception:
        return "(nvidia-smi 不可用)"


# ── 主流程 ──────────────────────────────────────────────────────────────────


def build(args):
    """建张量 + 臂列表。返回 (tensors, arms)。"""
    T, H, D = args.tokens, args.kv_heads, args.head_dim
    dev = "cuda"
    rows = T * H
    n_elem = rows * D

    # 数据用随机正态 * 0.4：幅度接近真实激活，且不会整行全 0（全 0 行测不出量化误差）
    x = (torch.randn(n_elem, device=dev, dtype=torch.bfloat16) * 0.4).view(rows, D)
    y = torch.empty_like(x)  # 写目标（bf16）
    q = torch.empty((rows, D), device=dev, dtype=torch.float8_e4m3fn)
    s = torch.empty((rows,), device=dev, dtype=torch.float32)
    q.view(torch.uint8).fill_(0x38)  # 0x38 ≈ 1.0(e4m3)，让 read_fp8 不依赖 kernel ①

    x_flat, q_flat = x.reshape(-1), q.reshape(-1)  # 流式读要一维连续

    scale_b = 4.0 / D  # scale 元数据：每行一个 fp32 ⇒ 摊到每元素 4/D 字节

    tensors = {"x": x, "y": y, "q": q, "s": s, "n_elem": n_elem, "rows": rows, "D": D}

    arms = [
        # 名字        有效字节/elem        被测操作
        ("copy", 2.0 + 2.0, lambda: y.copy_(x)),
        ("read_bf16", 2.0, lambda: stream_read(x_flat)),
        ("quant", 2.0 + 1.0 + scale_b, lambda: quant_fp8(x, q, s, args.num_warps)),
        ("dequant", 1.0 + scale_b + 2.0, lambda: dequant_fp8(q, s, y, args.num_warps)),
        ("read_fp8", 1.0, lambda: stream_read(q_flat)),
    ]
    # v2 行块化臂：每个 ROWS 一个臂，同一进程内交替 ⇒ 与 v1 的对比才成立
    for r in [int(t) for t in str(args.rows_sweep).split(",") if t.strip()]:
        arms.append(
            (
                f"q_tile_r{r}",
                2.0 + 1.0 + scale_b,
                (lambda rr: lambda: quant_fp8_tiled(x, q, s, args.num_warps, rr))(r),
            )
        )
        arms.append(
            (
                f"d_tile_r{r}",
                1.0 + scale_b + 2.0,
                (lambda rr: lambda: dequant_fp8_tiled(q, s, y, args.num_warps, rr))(r),
            )
        )
    return tensors, arms


def ratio_stats(results, name, base="copy", est=min):
    """同轮比值的跨轮极差（%）+ 用 `est` 估计的比值（%）。

    返回 (est 比值, 逐轮比值的跨轮极差)。
    """
    a, b = results.get(name), results.get(base)
    if not a or not b:
        return None, None
    n = min(len(a), len(b))
    r = [a[i] / b[i] * 100 for i in range(n)]
    med = statistics.median(r)
    spread = (max(r) - min(r)) / med * 100 if n > 1 and med else float("nan")
    return est(a) / est(b) * 100, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokens",
        type=int,
        default=131072,
        help="序列长度 T（默认 131072 ⇒ bf16 KV = 256 MiB，远超 L2）",
    )
    ap.add_argument("--kv-heads", type=int, default=8, help="KV head 数 H（Qwen3-8B 默认 8）")
    ap.add_argument("--head-dim", type=int, default=128, help="head 维度 D（Qwen3-8B 默认 128）")
    ap.add_argument("--inner", type=int, default=5, help="每次计时内层重复次数（短促连发）")
    ap.add_argument(
        "--repeat", type=int, default=5, help="整轮重复遍数（min 随轮数从上方收敛；离群多就加大）"
    )
    ap.add_argument("--warmup", type=int, default=3, help="每个臂的预热调用次数")
    ap.add_argument("--num-warps", type=int, default=1, help="手写 kernel 的 num_warps")
    ap.add_argument(
        "--rows-sweep",
        type=str,
        default="1,2,4,8,16,32",
        help="v2 行块化的 ROWS 扫描列表（逗号分隔；同一进程内交替测 ⇒ 可比）",
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    tensors, arms = build(args)
    x, y, q, s = tensors["x"], tensors["y"], tensors["q"], tensors["s"]
    n_elem, rows, D = tensors["n_elem"], tensors["rows"], tensors["D"]

    print("=" * 104)
    print("W2-5  手写 FP8 KV 量化 / 反量化 kernel —— 正确性 + 访存成本")
    print("=" * 104)
    print(
        f"形状      : T={args.tokens}  H={args.kv_heads}  D={args.head_dim}  "
        f"⇒ [T*H, D] = [{rows}, {D}]"
    )
    print(
        f"元素数    : {n_elem:,}    bf16 {n_elem*2/2**20:.1f} MiB   "
        f"fp8 {n_elem/2**20:.1f} MiB   scale {rows*4/2**20:.2f} MiB"
    )
    footprint = n_elem * 2 * 2  # x(bf16) + y(bf16) 两个大张量的总足迹
    print(
        f"L2        : {L2_MIB} MiB ｜ x+y 足迹 {footprint/2**20:.1f} MiB  ⇒ "
        f"{'✅ 超出 L2，走 HBM' if footprint > L2_MIB*2**20 else '⚠️ 落在 L2 内，测的是 L2 不是 HBM（加大 --tokens）'}"
    )
    print(
        f"参考带宽  : 纯读 {READ_BW_GBPS} GB/s（p01b 实测）；量具 = Triton 流式读"
        f"（BLOCK={READ_BLOCK} / warps={READ_WARPS}，实测 ≈98% 峰值）"
    )
    print(
        f"协议      : warmup={args.warmup} inner={args.inner} repeat={args.repeat}"
        f"（长跑会让 GPU 热起来：实测 copy 从 1538 → 1140 GB/s，−26%）"
    )
    print(f"GPU 起始  : {gpu_state()}")
    print()

    # ── ① 正确性 ────────────────────────────────────────────────────────────
    print("=" * 104)
    print(
        "【一、正确性】（手写 kernel vs torch fp32 参考实现）"
        "—— 两个 kernel 可**分别**测：反量化用 torch 参考造的 q/s，不必等量化写完"
    )
    print("=" * 104)
    # 参考结果**先算好**：它既是 kernel ① 的比对基准，也是 kernel ② 的输入
    # ⇒ 这样「反量化」可以独立被测，不必等「量化」写完（两个 kernel 能分别验证）
    rq, rs = ref_quant(x)

    # ── ① 量化 kernel ──────────────────────────────────────────────────────
    ok_quant = False
    try:
        quant_fp8(x, q, s, args.num_warps)
        torch.cuda.synchronize()
        n_bad, n_tot = compare_bits(q, rq)
        ds = (s - rs).abs() / rs.abs().clamp_min(1e-30)
        print(
            f"  ① 量化  : fp8 逐位不一致 {n_bad} / {n_tot}"
            f"    scale 相对误差 max {ds.max().item():.3e}  mean {ds.mean().item():.3e}"
        )
        print(
            f"            scale 范围 [{s.min().item():.3e}, {s.max().item():.3e}]"
            f"（应为正，量级 ≈ amax/448）"
        )
        # ★ 判据不能写成「必须逐位 0」：Triton 与 torch 的 fp32→fp8 转换在
        #   **舍入平局**（恰好落在两个可表示值中间）时可能选不同的方向。
        #   实测只有 11 / 134,217,728（8e-8）处如此，而 scale **逐位相同** ⇒ 算法一致。
        frac = n_bad / n_tot
        if n_bad == 0:
            verdict = "✅ 算法一致（逐位相同）"
        elif frac < 1e-6:
            verdict = f"✅ 算法一致（{n_bad} 处为舍入「平局」，占比 {frac:.1e} —— 不是 bug）"
        else:
            verdict = f"❌ 与参考实现不一致（占比 {frac:.1e}），先查 kernel"
        print(f"            ⇒ {verdict}")
        ok_quant = True
    except Exception as e:
        print(f"  ① [跳过] 量化 kernel 未就绪：{type(e).__name__}: {str(e)[:140]}")
        # ★ 把参考结果灌进 q/s，好让 kernel ② 仍能被**独立**验证
        q.view(torch.uint8).copy_(rq.view(torch.uint8))
        s.copy_(rs)

    # ── ② 反量化 kernel（输入用**当前** q/s ⇒ 与 torch 参考同输入，才可比）──
    ok_dequant = False
    try:
        y.zero_()
        dequant_fp8(q, s, y, args.num_warps)
        torch.cuda.synchronize()
        ry = ref_dequant(q, s)
        dy = (y.to(torch.float32) - ry.to(torch.float32)).abs().max().item()
        print(
            f"  ② 反量化: vs torch 参考，max|Δ| = {dy:.6e}   "
            f"⇒ {'✅ 一致' if dy == 0.0 else '⚠️ 有差异（bf16 末位舍入允许）'}"
        )
        ok_dequant = True
    except Exception as e:
        print(f"  ② [跳过] 反量化 kernel 未就绪：{type(e).__name__}: {str(e)[:140]}")

    # ── ③ round-trip 误差（两个 kernel 都实现了才有意义）──────────────────
    if ok_quant and ok_dequant:
        st = error_stats(x, y)
        print()
        print("  ③ round-trip 量化误差（**算法固有**，不是实现 bug）：")
        print(f"      max|Δ|        = {st['max_abs']:.6e}")
        print(f"      mean|Δ|       = {st['mean_abs']:.6e}")
        print(f"      RMS(Δ)        = {st['rms_abs']:.6e}")
        print(f"      RMS(x)        = {st['rms_x']:.6e}")
        print(f"      相对 RMS 误差 = {st['rel_rms']*100:.3f} %")
        print(
            "      理论量级      : e4m3 只有 3 位尾数 ⇒ ULP 相对 2^-3 = 12.5%，"
            "单次舍入 ≤6.25%；\n"
            "                      再叠加 scale 自身的舍入 ⇒ 最坏 ≈12.5%，**RMS 通常 2~4%**"
        )

    # ── ② 计时 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 104)
    print("【二、访存成本】（同一轮内交替测量；主指标是**同轮比值**）")
    print("=" * 104)

    results = {}  # 臂名 → 每轮的 ms 列表
    skipped = {}
    for rep in range(args.repeat):
        k = rep % len(arms)
        order = arms[k:] + arms[:k]  # ★ 轮转顺序 ⇒ 位置效应被均摊掉
        for name, bpe, fn in order:
            try:
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                ms = time_once(fn, args.inner)
            except Exception as e:
                skipped.setdefault(name, f"{type(e).__name__}: {str(e)[:70]}")
                continue
            results.setdefault(name, []).append(ms)

    for name, why in skipped.items():
        print(f"  [跳过] {name:<10} {why}")

    if not results:
        print("  没有任何臂可测 —— 先实现 kernel。")
        return

    hdr = (
        f"  {'臂':<11}{'有效字节/elem':>13}{'min ms':>10}{'中位 ms':>10}{'max/min':>9}"
        f"{'GB/s @min':>11}{'占 copy %':>11}{'比值跨轮极差':>14}   说明"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    notes = {
        "copy": "天花板（读2+写2）",
        "read_bf16": "现状：attention 读 bf16 KV",
        "quant": "手写 kernel ①（每 token 一次）",
        "dequant": "手写 kernel ②（每步，若不融合则如此）",
        "read_fp8": "目标：attention 直接吃 fp8",
    }
    for name, bpe, _ in arms:
        v = results.get(name)
        if not v:
            continue
        mn, md, mx = min(v), statistics.median(v), max(v)
        share, rspread = ratio_stats(results, name)
        print(
            f"  {name:<11}{bpe:>13.3f}{mn:>10.4f}{md:>10.4f}{mx/mn:>8.2f}x"
            f"{bw_gbps(n_elem * bpe, mn):>11.1f}"
            f"{(f'{share:.1f}%' if share else '—'):>11}"
            f"{(f'{rspread:.2f}%' if rspread == rspread else '—'):>14}   {notes.get(name, '')}"
        )

    # 离群告警：max/min 衡量「单边离群」程度（干扰只**加**时间）。
    # >1.3 说明该轮被外部干扰打过 —— min 仍可用，但建议加大 --repeat 再跑一遍看 min 是否下移。
    noisy = [(n, max(v) / min(v)) for n, v in results.items() if max(v) / min(v) > 1.3]
    if noisy:
        print()
        print("  ⚠️ 离群臂（max/min > 1.3 ⇒ 本轮被干扰打过；建议 `--repeat 8` 复跑）：")
        for n, r in noisy:
            print(f"       {n:<11} {r:.2f}x   原始值 " + " ".join(f"{x:.4f}" for x in results[n]))

    # ── ③ 判读 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 104)
    print("【三、判读】")
    print("=" * 104)
    rbf, rfp = results.get("read_bf16"), results.get("read_fp8")
    if rbf and rfp:
        a, b = min(rfp), min(rbf)
        print("  ① FP8 KV 的**核心收益**（每步读整个 KV cache）：")
        print(
            f"       read_fp8 {a:.4f} ms  vs  read_bf16 {b:.4f} ms  ⇒  **{a/b:.3f}×**"
            f"   （预测 {PREDICT['read_fp8_over_bf16']:.2f}×）"
        )
        print(
            f"       ⇒ 有效带宽 {bw_gbps(n_elem*1.0, a):.1f} / {bw_gbps(n_elem*2.0, b):.1f} GB/s"
            f"（p01b 实测纯读 {READ_BW_GBPS:.0f}）"
            f"⇒ {bw_gbps(n_elem*1.0, a)/READ_BW_GBPS*100:.1f}% / "
            f"{bw_gbps(n_elem*2.0, b)/READ_BW_GBPS*100:.1f}%"
        )
    else:
        print("  ① [待测] read_fp8 vs read_bf16")

    copy_ms = min(results["copy"]) if results.get("copy") else None
    if results.get("quant") and copy_ms:
        m = min(results["quant"])
        print("  ② 量化的一次性成本（每个 token 一次，可摊销）：")
        print(
            f"       quant {m:.4f} ms = copy 的 {m/copy_ms*100:.1f}%"
            f"（预测 {PREDICT['quant_share_of_copy']*100:.0f}% —— 读2+写1，本就该比 copy 的 4B 便宜 ~24%）"
        )
        print("       摊销：S=4096 时每 token 一次 ⇒ 均摊到之后 4096 步 decode，可忽略")

    if results.get("dequant") and rbf:
        m = min(results["dequant"])
        b = min(rbf)
        print("  ③ ★ 关键结论 —— 「反量化单独落盘」这条路是**亏的**：")
        print(
            f"       dequant {m:.4f} ms  vs  直接读 bf16 {b:.4f} ms  ⇒  **{m/b:.3f}×**"
            f"{'（更慢！）' if m > b else '（更快）'}"
        )
        print("       因为 dequant 要「读 1B + 写 2B = 3B」，比直接读 bf16 的 2B 还多 50%。")
        print("       ⇒ **FP8 KV 只在「dequant 融进 attention、不落盘」时才有意义** ——")
        print("         这正是 vLLM 把 FP8 KV 绑死在 FlashInfer XQA 上的原因（4.19）。")

    print()
    print(f"  GPU 结束  : {gpu_state()}")
    print("  ⚠️ 边界：本脚本只到**算子级**。要端到端，需另写一个能直接吃 fp8 KV 的 attention")
    print("     kernel（把 dequant 融进 QK^T 的 inner loop）—— 那是下一步的工作。")


if __name__ == "__main__":
    raise SystemExit(main())
