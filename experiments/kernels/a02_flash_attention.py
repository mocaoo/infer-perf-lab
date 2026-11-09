#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Noncausal, single-head attention: materialized scores versus tiled online softmax.

FP64/SDPA references and CUDA-event timing; no vLLM integration.
Roofline traffic estimates assume cache reuse and are not measured HBM traffic."""

import argparse
import statistics
import subprocess
import time

import torch
import triton
import triton.language as tl

SPEC_GBPS = 1792.0
PEAK_TFLOPS_BF16 = 209.7  # 本机 bf16 稠密张量峰值
RIDGE = PEAK_TFLOPS_BF16 * 1e12 / (SPEC_GBPS * 1e9)  # 117 FLOP/B
BW_MIXED_GBPS = 1550.0  # 混合负载可达带宽（w09/p01b 实测）


# ══════════════════════════════════════════════════════════════════════════════
# 实现：3 个 kernel（bf16 输入，fp32 累加，bf16 输出）
# ══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _qk_kernel(Q, K, Sm, S, D: tl.constexpr, scale, BM: tl.constexpr, BN: tl.constexpr):
    """① 分数矩阵：Sm[i,j] = (Q_i · K_j) * scale，落到 [S,S] fp32。**这就是 naive 的原罪。**"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])
    k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])
    s = tl.dot(q, tl.trans(k)) * scale
    tl.store(Sm + offs_m[:, None] * S + offs_n[None, :], s)


@triton.jit
def _pv_kernel(P, V, O, S, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """③ O_i = Σ_j P_ij · V_j —— 沿 j **分块累加**（不能一次把 [BM,S] 装进寄存器）。"""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(0, S, BN):
        offs_n = n0 + tl.arange(0, BN)
        p = tl.load(P + offs_m[:, None] * S + offs_n[None, :])
        v = tl.load(V + offs_n[:, None] * D + offs_d[None, :])
        acc += tl.dot(p.to(tl.bfloat16), v)
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], acc.to(tl.bfloat16))


@triton.jit
def _flash_kernel(Q, K, V, O, S, D: tl.constexpr, scale, BM: tl.constexpr, BN: tl.constexpr):
    """★ 核心：一个 kernel 算完 attention，**不产生 [S,S] 矩阵**。"""
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])
    m_i = tl.full([BM], -1.0e38, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(0, S, BN):
        offs_n = n0 + tl.arange(0, BN)
        k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])
        v = tl.load(V + offs_n[:, None] * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], o.to(tl.bfloat16))


# ══════════════════════════════════════════════════════════════════════════════
# 参考实现（fp64，从**同一份 bf16 输入**算，不把输入量化误差算到实现头上）
# ══════════════════════════════════════════════════════════════════════════════
def ref_attention_fp64(q, k, v, scale):
    qf, kf, vf = q.double(), k.double(), v.double()
    s = (qf @ kf.transpose(-1, -2)) * scale  # [S, S]
    p = torch.softmax(s, dim=-1)
    return (p @ vf).float()


def fmt_error(e, indent="      "):
    """把 Triton 的编译错误**完整**打出来。

    ★ 原来只印 `str(e).splitlines()[0]`，而 Triton 报错的第一行只是 `at 29:13:`（纯位置），
    真正的信息在**后面几行**（源码片段 + 失败原因）⇒ 等于把最需要的部分吞了。
    这里保留全文，只在超过 40 行时掐头留尾。
    """
    lines = str(e).strip().splitlines() or [type(e).__name__]
    if len(lines) > 40:
        lines = lines[:8] + ["…（省略 %d 行）…" % (len(lines) - 16)] + lines[-8:]
    head = "%s%s" % (indent, type(e).__name__)
    body = "\n".join(indent + "  " + ln for ln in lines)
    return head + "\n" + body


def rel_rms(a, b):
    """相对 RMS 误差：||a−b|| / ||b||。比 max|Δ| 更能反映「整体偏了多少」。"""
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


# ══════════════════════════════════════════════════════════════════════════════
# 计时 / 统计
# ══════════════════════════════════════════════════════════════════════════════
def _sample_power_during(fn, seconds=3.0):
    """在 fn **持续运行**时读一次 power.draw。

    ★ 为什么必须这样测：脚本里原来三处 `clocks()` 分别在「起始空转」「预热（copy 循环，
    memory-bound）」「收尾」，读到的都是 245~260 W —— 那是拷贝的功耗，不是 kernel 的功耗。
    要判断有没有撞 Power Limit（本例 75% × 575 W ≈ 431 W），必须在 GPU 忙的时候采样。
    """
    import subprocess
    import threading
    import time

    box = {}

    def worker():
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        box["p"] = r.stdout.strip()

    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    th = threading.Thread(target=worker)
    th.start()  # nvidia-smi 在后台查询（约 100 ms），期间 GPU 保持忙
    t_end = time.time() + seconds
    while time.time() < t_end:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    th.join()

    p, clk = [x.strip() for x in box.get("p", "0, 0").split(",")[:2]]
    return float(p), float(clk)


def time_once(fn, inner):
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(inner):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / inner


def clocks():
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception as e:
        return f"(采样失败: {type(e).__name__})"


def prewarm(seconds):
    """把频率拉到稳态（本卡开局 1600~1700 MHz、稳态 2685 MHz）。"""
    if seconds <= 0:
        return
    a = torch.empty(128 * 2**20 // 4, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            b.copy_(a)
        torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 单个 S
# ══════════════════════════════════════════════════════════════════════════════
def run_seq(S, args):
    D = args.head_dim
    dev = "cuda"
    scale = 1.0 / (D**0.5)
    assert S % args.bm == 0 and S % args.bn == 0, "先做整除版本：S 必须被 BM/BN 整除"

    torch.manual_seed(0)
    q = torch.randn(S, D, device=dev, dtype=torch.bfloat16) * 0.1
    k = torch.randn(S, D, device=dev, dtype=torch.bfloat16) * 0.1
    v = torch.randn(S, D, device=dev, dtype=torch.bfloat16) * 0.1
    o_naive = torch.empty(S, D, device=dev, dtype=torch.bfloat16)
    o_flash = torch.empty(S, D, device=dev, dtype=torch.bfloat16)
    o_sdpa = torch.empty(S, D, device=dev, dtype=torch.bfloat16)
    ref = ref_attention_fp64(q, k, v, scale)

    score_mib = S * S * 4 / 2**20  # naive 的中间矩阵（fp32）
    qkv_mib = 4 * S * D * 2 / 2**20

    grid2 = (S // args.bm, S // args.bn)
    grid1 = (S // args.bm,)

    # ★ 编译旋钮：不传就用 Triton 默认（num_warps=4 / num_stages=3）。
    #   这两个参数决定 SMEM/CTA 与 warps/SM —— 见本文件「片上资源」探针的输出。
    kw = {}
    if getattr(args, "num_warps", 0):
        kw["num_warps"] = args.num_warps
    if getattr(args, "num_stages", 0):
        kw["num_stages"] = args.num_stages

    def naive():
        sm = torch.empty((S, S), device=dev, dtype=torch.float32)  # ← S² 的显存
        _qk_kernel[grid2](q, k, sm, S, D, scale, BM=args.bm, BN=args.bn, **kw)
        torch.softmax(sm, dim=-1, out=sm)  # 原地，少一次分配
        _pv_kernel[grid1](sm, v, o_naive, S, D, BM=args.bm, BN=args.bn, **kw)
        del sm

    def flash():
        _flash_kernel[grid1](q, k, v, o_flash, S, D, scale, BM=args.bm, BN=args.bn, **kw)

    def sdpa():
        """第四臂：PyTorch 官方融合实现（不落 [S,S]，后端由 PyTorch 自选）。
        形状必须是 [B, H, S, D] ⇒ 补两个 1 维；scale 与其它臂保持一致。
        ★ 若它的显存峰值 ≈ S²·4 B，说明退化到了 math 后端，这一臂的结果作废。"""
        o = torch.nn.functional.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None], scale=scale
        )
        o_sdpa.copy_(o[0, 0])

    # ★ 计时专用：sm **预分配在计时区外**
    #   原写法让同一个闭包同时服务「峰值显存」和「计时」两个目的，而两者要求相反：
    #     · 峰值必须在闭包内分配，否则测不到；
    #     · 计时希望预分配，否则每轮都要付 malloc/free 与 allocator 抖动。
    #   S=16384 时 sm 是 1 GiB，抖动会把 naive 罚重（实测中位/min 达 1.42×）⇒ 拆成两个闭包。
    #   注意：它必须在这里分配（早于 peaks 里的 `base = memory_allocated()`），
    #   这样它进基线的账、不会污染任何一臂的「峰值增量」。
    sm_timing = torch.empty((S, S), device=dev, dtype=torch.float32)

    def naive_reuse():
        _qk_kernel[grid2](q, k, sm_timing, S, D, scale, BM=args.bm, BN=args.bn, **kw)
        torch.softmax(sm_timing, dim=-1, out=sm_timing)
        _pv_kernel[grid1](sm_timing, v, o_naive, S, D, BM=args.bm, BN=args.bn, **kw)

    print(f"\n{'=' * 100}")
    print(
        f"S={S:<6} D={D}   BM={args.bm} BN={args.bn}   "
        f"naive 中间矩阵 {score_mib:.0f} MiB（L2 = 96 MiB）   QKV+O {qkv_mib:.1f} MiB"
    )
    print(f"{'=' * 100}")

    # ── ① 正确性 ────────────────────────────────────────────────────────────
    print("【一、正确性】参考 = fp64（从同一份 bf16 输入算）")
    ok = {}
    failures = []
    for name, fn, out in (
        ("naive", naive, o_naive),
        ("flash", flash, o_flash),
        ("sdpa", sdpa, o_sdpa),
    ):
        try:
            out.zero_()
            fn()
            torch.cuda.synchronize()
            r = rel_rms(out.float(), ref)
            mx = (out.float() - ref).abs().max().item()
            ok[name] = r
            print(
                f"  {name:<6} 相对 RMS = {r:.3e}   max|Δ| = {mx:.3e}   "
                f"{'✅' if r <= 1e-2 else '❌'}（阈值 1e-2）"
            )
        except Exception as e:
            ok[name] = None
            failures.append(f"{name}（正确性）")
            print(f"  {name:<6} ❌ 失败：")
            print(fmt_error(e))

    if ok.get("flash") is not None and ok.get("sdpa") is not None:
        r = rel_rms(o_flash.float(), o_sdpa.float())
        print(
            f"  ★ flash vs sdpa：相对 RMS = {r:.3e}   {'✅' if r <= 1e-2 else '❌'}"
            f"（「自研实现离官方有多远」的直接读数）"
        )

    if ok.get("naive") is not None and ok.get("flash") is not None:
        r = rel_rms(o_flash.float(), o_naive.float())
        print(f"  两臂互比：相对 RMS = {r:.3e}   {'✅' if r <= 1e-2 else '❌'}（阈值 1e-2）")
        print(
            "     注：两臂都把 p round 到 bf16（各 ~0.2%），但 round 的对象不同"
            "（归一化 vs 未归一化）⇒ 差值 ~3e-3 属**设计**；地址类 bug 会给 O(1) 的错，仍能抓到"
        )

    # ── ② 显存峰值 ──────────────────────────────────────────────────────────
    print("\n【二、显存】峰值增量（中间矩阵是唯一差别）")
    peaks = {}
    for name, fn in (("naive", naive), ("flash", flash), ("sdpa", sdpa)):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fn()
            torch.cuda.synchronize()
            peaks[name] = (torch.cuda.max_memory_allocated() - base) / 2**20
            print(f"  {name:<6} 峰值 {peaks[name]:8.1f} MiB")
        except Exception:
            peaks[name] = None
    # ★ 原来写的是 `if peaks.get("naive") and peaks.get("flash")` —— flash 恒为 0.0，
    #   而 0.0 是 falsy ⇒ 这一行**从来没打印过**。必须用 `is not None`。
    if peaks.get("naive") is not None and peaks.get("flash") is not None:
        ratio = (
            f"{peaks['naive'] / peaks['flash']:.1f}×"
            if peaks["flash"] > 0.05
            else "∞（flash 没有做任何 global 分配）"
        )
        print(
            f"  ⇒ naive / flash = **{ratio}**"
            f"　（理论：中间矩阵 {score_mib:.0f} MiB；flash 的 acc 在寄存器里）"
        )

    # ── ②b 片上资源：allocator 看不见的那一半 ────────────────────────────────
    #   `max_memory_allocated` 只统计 torch caching allocator 的 global 内存。
    #   flash 的 acc/s/p 在寄存器、k/v tile 在 SMEM、溢出部分在 local memory —— 全都不进那个数，
    #   所以它报 0.0 MiB 是**口径**，不是「不占显存」。这里读编译产物把真实占用补上。
    print("  ★ 片上资源（Triton 编译产物，allocator 看不见）：")
    try:
        ck = _flash_kernel.warmup(
            q, k, v, o_flash, S, D, scale, BM=args.bm, BN=args.bn, grid=(1,), **kw
        )
        if isinstance(ck, tuple):
            ck = ck[0]
        md = getattr(ck, "metadata", {})
        shm = md.get("shared") if isinstance(md, dict) else getattr(md, "shared", None)
        n_regs = getattr(ck, "n_regs", "?")
        n_spills = getattr(ck, "n_spills", "?")
        print(
            f"    flash  n_regs = {n_regs}/thread   n_spills = {n_spills} 字节/thread"
            f"   SMEM = {shm} B"
        )
        if isinstance(n_spills, int) and n_spills > 0:
            print(
                "    ⚠ n_spills > 0 ⇒ 有寄存器溢出，溢出量走 local memory（也在 HBM 上，"
                "由 driver 分配，同样不计入 max_memory_allocated）"
            )
    except Exception as e:
        print("    ⚠ 读编译产物失败（Triton 版本的字段名可能不同，不影响本节结论）：")
        print(fmt_error(e))

    # ── ③ 性能 ──────────────────────────────────────────────────────────────
    #   naive       = 预分配 sm（公平口径，主表用它）
    #   naive_alloc = 每轮 torch.empty((S,S))+del（原口径，单独报，差值 = 分配开销的上界）
    #   flash       = 无中间矩阵
    arms_all = (("naive", naive_reuse), ("naive_alloc", naive), ("flash", flash), ("sdpa", sdpa))
    # naive_alloc 与 naive 是同一个算法，正确性门禁共用 ok["naive"]
    arms = [(n, f) for n, f in arms_all if ok.get("naive" if n == "naive_alloc" else n) is not None]
    # ★ **每轮轮转臂顺序**：实验证明同一条臂换个位置能差 5~26%（排第一的最冷），
    #   这是 w09 已经踩过的臂顺序偏差 —— 它不会被 repeat 平均掉，只能靠轮转消掉。
    res = {}
    for rep in range(args.repeat):
        # ★ 变量名不能叫 k —— 那会覆盖外层 Q/K/V 里的 K 张量（闭包晚绑定），
        #   症状是 kernel 收到 int，报 `Unsupported ptr type <['64','128'], int32> in tl.load`
        shift = 0 if not arms else rep % len(arms)
        for name, fn in arms[shift:] + arms[:shift]:
            times = res.setdefault(name, [])  # ★ 不能叫 v —— 会覆盖外层的 V 张量
            try:
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                times.append(time_once(fn, args.inner))
            except Exception as e:
                failures.append(f"{name}（计时）")
                res.pop(name, None)  # 失败就别留在表里（min() 不能吃空列表）
                print(f"  {name:<6} ❌ 计时失败：")
                print(fmt_error(e))
                break

    # ── ④ roofline 对账 ─────────────────────────────────────────────────────
    flops = 4.0 * S * S * D
    bytes_flash = 8.0 * S * D  # Q+K+V+O
    bytes_naive = 16.0 * S * S + 8.0 * S * D  # 中间矩阵 4 遍 × 4 B
    floor_compute = flops / (PEAK_TFLOPS_BF16 * 1e12) * 1e3
    floor_flash = bytes_flash / (BW_MIXED_GBPS * 1e9) * 1e3
    floor_naive = bytes_naive / (BW_MIXED_GBPS * 1e9) * 1e3
    ai = {
        "naive": flops / bytes_naive,
        "flash": flops / bytes_flash,
        "sdpa": flops / bytes_flash,
    }  # sdpa 与 flash 字节相同（都不落 [S,S]）

    if res:
        print("\n【三、性能 + roofline】")
        print(
            f"  {'臂':<8}{'min ms':>10}{'中位 ms':>10}{'AI FLOP/B':>12}{'bound':>10}"
            f"{'floor ms':>11}{'实测/floor':>12}"
        )
        print("  " + "-" * 88)
        for name in ("naive", "flash", "sdpa"):
            if name not in res:
                continue
            mn = min(res[name])
            bound = "compute" if ai[name] > RIDGE else "memory"
            fl = (
                floor_compute
                if bound == "compute"
                else (floor_naive if name == "naive" else floor_flash)
            )
            print(
                f"  {name:<8}{mn:>10.4f}{statistics.median(res[name]):>10.4f}"
                f"{ai[name]:>12.0f}{bound:>10}{fl:>11.4f}{mn / fl:>11.1f}×"
            )
        print(
            f"\n  FLOPs = {flops / 1e9:.1f} GFLOP；算力 floor {floor_compute:.4f} ms"
            f"（@ {PEAK_TFLOPS_BF16} TFLOPS）"
        )
        print(f"  flash 访存 floor {floor_flash:.4f} ms（{bytes_flash / 1e6:.2f} MB）")
        print(f"  naive 访存 floor {floor_naive:.4f} ms（{bytes_naive / 1e6:.0f} MB）")
        crossed = ai["flash"] > RIDGE > ai["naive"]
        print(
            f"\n  ★ AI：naive **{ai['naive']:.0f}** → flash **{ai['flash']:.0f}** FLOP/B，ridge = {RIDGE:.0f}"
        )
        print(
            "    ⇒ "
            + (
                "**跨过 ridge：naive 访存受限、flash 算力受限 —— 瓶颈换了，这才是「融合」的意义**"
                if crossed
                else "未跨过 ridge，检查 S 或 D"
            )
        )
        if "naive" in res and "flash" in res:
            print(
                f"  ⇒ naive/flash = **{min(res['naive']) / min(res['flash']):.2f}×**"
                f"（「融合 vs 落盘」，预测 2.5~3.5×）"
            )
        if "flash" in res and "sdpa" in res:
            fr = min(res["flash"]) / min(res["sdpa"])
            # ★ fr = t_flash / t_sdpa ⇒ fr>1 表示自研花的时间更长（更慢）
            verdict = (
                "自研**慢** %.0f%%" % ((fr - 1) * 100)
                if fr > 1
                else "自研**快** %.0f%%" % ((1 / fr - 1) * 100)
            )
            print(
                f"  ⇒ flash/sdpa = **{fr:.2f}×**　（{verdict}）"
                f"　★ 这才是能写进简历的「自研 vs 官方」数"
            )
            print(
                "     注：sdpa 走 PyTorch 自己的融合后端（flash / mem-efficient / math 之一）。"
                "若它的显存峰值 ≈ S²·4 B，说明退化成 math 后端，此臂作废。"
            )
        if "naive_alloc" in res and "naive" in res:
            a_, b_ = min(res["naive_alloc"]), min(res["naive"])
            print(
                f"  ★ 口径对比：naive 带 torch.empty((S,S))+del = {a_:.4f} ms，"
                f"预分配 = {b_:.4f} ms　⇒ **{100 * (a_ / b_ - 1):+.1f}%**"
            )
            print("    · 主表用预分配口径（真实部署 buffer 复用，不该把 malloc 算进算法成本）")
            print("    · 这个差值是「分配开销」的**上界**，不是它的估计：它同时含臂顺序偏差")
            print("      （预分配臂排在前面，预热更少）和 4 个 kernel 边界的抖动。")
            print("    · 若差值为负 ⇒ caching allocator 复用了同一块，分配**没有成本**，")
            print(
                "      那小 S 的 中位/min 抖动来自别处（不是 malloc/free）。判据：看 S≥4096 的符号。"
            )

    # ── ③b 真实功耗（--power-probe）─────────────────────────────────────────
    if getattr(args, "power_probe", False) and res:
        print("\n【三b、功耗】在 kernel **持续运行**时采样（3 秒，nvidia-smi）")
        for name, fn in (("flash", flash), ("sdpa", sdpa)):
            if name not in res:
                continue
            try:
                w, clk = _sample_power_during(fn, seconds=3.0)
                mn = min(res[name])
                tf = flops / (mn * 1e-3) / 1e12
                print(
                    f"  {name:<6} {w:6.1f} W @ {clk:.0f} MHz   ⇒ 能效 **{tf / w:.3f} TFLOPS/W**"
                    f"（t={mn:.4f} ms，{tf:.1f} TFLOPS）"
                )
            except Exception as e:
                print(f"  {name:<6} ❌ 采样失败：{type(e).__name__}: {e}")
        print("  ★ 上限 = 75% × 575 W ≈ **431 W**；若实测接近它 ⇒ 已撞功耗墙，频率会被拉低。")

    del q, k, v, o_naive, o_flash, o_sdpa, ref, sm_timing
    torch.cuda.empty_cache()
    # ★ 并行度诊断：flash 的 grid = S/BM，与 SM 数比才知道在哪个区间
    #   η_occ = grid / (ceil(grid / SM) * SM) —— 把「波量化」也算进去
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    waves = -(-grid1[0] // sm)
    eta_occ = grid1[0] / (waves * sm)
    # SM 归一 floor：每 SM 峰值 = 总峰值 / SM 数
    #   每 CTA 的活 = flops / grid；每 SM 峰值 = 总峰值 / SM 数；
    #   波数 waves 个 CTA 是串行的 ⇒ makespan ≈ waves × 每 CTA 时间
    floor_sm = waves * (flops / grid1[0]) / (PEAK_TFLOPS_BF16 * 1e12 / sm) * 1e3
    return {
        "failures": failures,
        "naive_ms": min(res["naive"]) if "naive" in res else None,
        "naive_alloc_ms": min(res["naive_alloc"]) if "naive_alloc" in res else None,
        "grid": grid1[0],
        "waves": waves,
        "eta_occ": eta_occ,
        "floor_sm": floor_sm,
        "eta_sm": (floor_sm / min(res["flash"])) if "flash" in res else None,
        "mfu": (
            (flops / (min(res["flash"]) * 1e-3) / (PEAK_TFLOPS_BF16 * 1e12))
            if "flash" in res
            else None
        ),
        "flops": flops,
        "flash_ms": min(res["flash"]) if "flash" in res else None,
        "sdpa_ms": min(res["sdpa"]) if "sdpa" in res else None,
        "score_mib": score_mib,
        "ai_flash": ai["flash"],
    }


def main():
    ap = argparse.ArgumentParser(description="naive attention vs flash attention（单头，bf16）")
    ap.add_argument("--s-list", type=str, default="1024,2048,4096,8192,16384", help="序列长度列表")
    ap.add_argument("--head-dim", type=int, default=128, help="D（Qwen3-8B 的 head_dim）")
    ap.add_argument("--bm", type=int, default=64, help="Q 方向块大小")
    ap.add_argument("--bn", type=int, default=64, help="K/V 方向块大小")
    ap.add_argument(
        "--num-warps", type=int, default=0, help="每 CTA 的 warp 数（0 = Triton 默认 4）"
    )
    ap.add_argument(
        "--num-stages",
        type=int,
        default=0,
        help="软件流水级数（0 = Triton 默认 3）；与 BN 一起决定 SMEM/CTA",
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--prewarm-sec", type=float, default=3.0)
    ap.add_argument(
        "--power-probe",
        action="store_true",
        help="在 kernel 持续运行时采样功耗与频率（判断是否撞 Power Limit）",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("没有 CUDA 设备")

    print("=" * 100)
    print("a02  naive attention vs flash attention —— 把 [S,S] 从显存里拿掉")
    print("=" * 100)
    print(
        f"卡      : {torch.cuda.get_device_name(0)}   bf16 峰值 {PEAK_TFLOPS_BF16} TFLOPS"
        f"   ridge {RIDGE:.0f} FLOP/B"
    )
    print(f"协议    : warmup={args.warmup} inner={args.inner} repeat={args.repeat}")
    print(
        f"编译    : num_warps={args.num_warps or '默认4'}  "
        f"num_stages={args.num_stages or '默认3'}  BM={args.bm} BN={args.bn}"
    )
    print(f"GPU 起始: {clocks()}")
    prewarm(args.prewarm_sec)
    print(f"预热后  : {clocks()}")

    rows = {}
    for S in [int(s) for s in args.s_list.split(",")]:
        rows[S] = run_seq(S, args)

    if len(rows) > 1:
        print(f"\n{'=' * 100}")
        print(
            "【跨 S 汇总】—— naive 的中间矩阵在 S≈5000 处跨过 L2（96 MiB）；"
            "flash 的 grid 在 S≈10880 处跨过 SM 数（170）"
        )
        print(f"{'=' * 100}")
        print(
            f"  {'S':>7}{'mid MiB':>9}{'naive ms':>10}{'flash ms':>10}{'提速':>9}"
            f"{'grid':>6}{'波数':>6}{'η_occ':>8}{'SMfloor':>9}"
            f"{'η_sm':>7}{'MFU':>7}{'T比':>7}{'vsSDPA':>8}"
        )
        print("  " + "-" * 96)
        prev = None
        for S, r in rows.items():
            nv = "{:.4f}".format(r["naive_ms"]) if r["naive_ms"] else "—"
            fl = "{:.4f}".format(r["flash_ms"]) if r["flash_ms"] else "—"
            sp = (
                "{:.2f}×".format(r["naive_ms"] / r["flash_ms"])
                if (r["naive_ms"] and r["flash_ms"])
                else "—"
            )
            # T 比 = 相邻两档 flash 时间的比；S 翻倍时 2.0× ⇒ T∝S，4.0× ⇒ T∝S²
            tr = "{:.2f}".format(r["flash_ms"] / prev) if (prev and r["flash_ms"]) else "—"
            vs = (
                "{:.2f}×".format(r["flash_ms"] / r["sdpa_ms"])
                if (r["flash_ms"] and r.get("sdpa_ms"))
                else "—"
            )
            prev = r["flash_ms"] if r["flash_ms"] else prev
            es = "{:.3f}".format(r["eta_sm"]) if r["eta_sm"] is not None else "—"
            mf = "{:.1%}".format(r["mfu"]) if r["mfu"] is not None else "—"
            print(
                f"  {S:>7}{r['score_mib']:>9.0f}{nv:>10}{fl:>10}{sp:>9}"
                f"{r['grid']:>6}{r['waves']:>6}{r['eta_occ']:>8.3f}"
                f"{r['floor_sm']:>9.4f}{es:>7}{mf:>7}{tr:>7}{vs:>8}"
            )
        print(
            "\n  ★ 读法：grid = S/BM 个 CTA 对 {0} 个 SM ⇒ 波数 = ceil(grid/SM)".format(
                torch.cuda.get_device_properties(0).multi_processor_count
            )
        )
        print("     η_occ < 1 且波数 = 1 ⇒ SMs 没开满 ⇒ **T ∝ S**（S 翻倍时间只 ×2）")
        print("     η_occ ≈ 1 或波数 ≥ 2 ⇒ 并行度用光 ⇒ **T ∝ S²**（S 翻倍时间 ×4）")
        print("     SMfloor  = 波数 × (FLOPs/grid) / 每SM峰值 —— 这一档该比的 floor")
        print("     η_occ    = grid / (波数×SM)         —— 有多少 SM 在干活（负载均衡）")
        print("     η_sm     = SMfloor / 实测           —— 单个 SM 用到了峰值的几成")
        print("     MFU      = 实测FLOPs/s ÷ 209.7 TFLOPS = **η_occ × η_sm**（恒等式）")
        print(
            "     η_occ 不是单调的：grid 略超 k×170 时掉到最低（171 个 CTA ⇒ 0.50）⇒ 有「网格共振」"
        )

    print(f"\n  GPU 结束: {clocks()}")

    bad = [f"S={S} {f}" for S, r in rows.items() for f in r.get("failures", [])]
    if bad:
        print(f"\n  ❌ 有 {len(bad)} 处失败：{'、'.join(bad)}")
        print("     ⇒ 退出码 1（不再「跳过即成功」）")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
