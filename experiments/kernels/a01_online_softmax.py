#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare three-pass, online and single-pass softmax against FP64.

Synthetic standalone benchmark; byte counts are analytical estimates."""

import argparse
import statistics
import subprocess
import time

import torch
import triton
import triton.language as tl

SPEC_GBPS = 1792.0


# ══════════════════════════════════════════════════════════════════════════════
# 实现：5 个 kernel
# ══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _row_max_kernel(X, M, S, BLOCK: tl.constexpr):
    """每行求 max，写到 M[row]。"""
    pid = tl.program_id(0)
    m = -1.0e38
    for s0 in range(0, S, BLOCK):
        cols = s0 + tl.arange(0, BLOCK)
        mask = cols < S
        x = tl.load(X + pid * S + cols, mask=mask, other=-1.0e38)
        m = tl.maximum(m, tl.max(x, axis=0))
    tl.store(M + pid, m)


@triton.jit
def _row_sumexp_kernel(X, M, L, S, BLOCK: tl.constexpr):
    """每行求 Σexp(x - m)，写到 L[row]。与 _row_max_kernel 同形，只多一步 exp。"""
    pid = tl.program_id(0)
    m = tl.load(M + pid)
    l = 0.0
    for s0 in range(0, S, BLOCK):
        cols = s0 + tl.arange(0, BLOCK)
        mask = cols < S
        x = tl.load(X + pid * S + cols, mask=mask, other=-1.0e38)
        l += tl.sum(tl.exp(x - m), axis=0)
    tl.store(L + pid, l)


@triton.jit
def _online_reduce_kernel(X, M, L, S, BLOCK: tl.constexpr):
    """★ 核心：单遍流式求 (m, l)，只读 X 一次。"""
    pid = tl.program_id(0)
    m = -1.0e38
    l = 0.0
    for s0 in range(0, S, BLOCK):
        cols = s0 + tl.arange(0, BLOCK)
        mask = cols < S
        x = tl.load(X + pid * S + cols, mask=mask, other=-1.0e38)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    tl.store(M + pid, m)
    tl.store(L + pid, l)


@triton.jit
def _normalize_kernel(X, M, L, Y, S, BLOCK: tl.constexpr):
    """读 X、M、L，把 exp(x - m) / l 写到 Y。three_pass 与 online 共用。"""
    pid = tl.program_id(0)
    m = tl.load(M + pid)
    l = tl.load(L + pid)
    for s0 in range(0, S, BLOCK):
        cols = s0 + tl.arange(0, BLOCK)
        mask = cols < S
        x = tl.load(X + pid * S + cols, mask=mask, other=0.0)
        tl.store(Y + pid * S + cols, tl.exp(x - m) / l, mask=mask)


@triton.jit
def _single_pass_kernel(X, Y, S, BLOCK: tl.constexpr):
    """整行常驻：把一行全部 load 进寄存器，一次算完 m、Σ、写出。只读 1 遍。"""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < S
    x = tl.load(X + pid * S + cols, mask=mask, other=-1.0e38)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    l = tl.sum(e, axis=0)
    tl.store(Y + pid * S + cols, e / l, mask=mask)


# ══════════════════════════════════════════════════════════════════════════════
# 参考实现：分两层，用来区分「算法错」和「kernel 错」
# ══════════════════════════════════════════════════════════════════════════════
def ref_softmax_fp64(x):
    """标准答案：fp64 计算，再转回 fp32。判「数值对不对」以它为准。"""
    xf = x.double()  # 提到 fp64，避免参考实现自己就不准
    m = xf.max(dim=-1, keepdim=True).values
    e = (xf - m).exp()
    return (e / e.sum(dim=-1, keepdim=True)).float()


def ref_online_torch(x, block):
    """用 torch 把「分块重缩放」这个**算法**跑一遍（fp64），返回 (m, l)。

    作用：如果 kernel 与它一致、但和 fp64 标准答案不一致 ⇒ **算法想错了**；
          如果 kernel 与它不一致 ⇒ **kernel 写错了**。
    这是用「一份独立的规格实现」把错误一分为二的老办法（w08 已用过）。
    """
    R, S = x.shape
    xf = x.double()
    m = torch.full((R,), -1.0e38, dtype=torch.float64, device=x.device)
    l = torch.zeros((R,), dtype=torch.float64, device=x.device)
    for s0 in range(0, S, block):
        blk = xf[:, s0 : s0 + block]
        bm = blk.max(dim=-1).values
        m_new = torch.maximum(m, bm)
        l = l * (m - m_new).exp() + (blk - m_new[:, None]).exp().sum(-1)
        m = m_new
    return m, l


# ══════════════════════════════════════════════════════════════════════════════
# 计时 / 统计（沿用 w08、w09 的协议）
# ══════════════════════════════════════════════════════════════════════════════
def time_once(fn, inner):
    """CUDA Event 计时，内层连发 inner 次取平均。"""
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(inner):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / inner


def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


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
    """把频率拉到稳态：实测本卡开局 1600~1700 MHz、稳态 2685 MHz。"""
    if seconds <= 0:
        return
    a = torch.empty(128 * 2**20 // 4, device="cuda", dtype=torch.float32)  # 128 MiB
    b = torch.empty_like(a)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            b.copy_(a)
        torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 每个 S 的完整测量
# ══════════════════════════════════════════════════════════════════════════════
def run_seq(S, args):
    R = args.elems // S
    dev = "cuda"
    # 用「标准差 1 的随机数 × 3」：softmax 对幅值敏感，太小则所有概率接近均匀、误差看不出来
    x = torch.randn((R, S), device=dev, dtype=torch.float32) * 3.0
    y = torch.empty_like(x)
    ref = ref_softmax_fp64(x)  # fp64 标准答案
    m_buf = torch.empty(R, device=dev, dtype=torch.float32)
    l_buf = torch.empty(R, device=dev, dtype=torch.float32)

    grid = (R,)
    BLK = args.block  # three_pass / online 的分块
    BP2 = triton.next_power_of_2(S)  # single_pass 需要 ≥ S 的 2 的幂
    nw_sp = max(4, min(32, BP2 // 512))  # single_pass 的 warp 数随行宽增长

    NW = args.warps  # ★ 显式传：默认 4 时 BLOCK=1024 会因归约开销变慢

    def three_pass():
        _row_max_kernel[grid](x, m_buf, S, BLOCK=BLK, num_warps=NW)
        _row_sumexp_kernel[grid](x, m_buf, l_buf, S, BLOCK=BLK, num_warps=NW)
        _normalize_kernel[grid](x, m_buf, l_buf, y, S, BLOCK=BLK, num_warps=NW)

    def online():
        _online_reduce_kernel[grid](x, m_buf, l_buf, S, BLOCK=BLK, num_warps=NW)
        _normalize_kernel[grid](x, m_buf, l_buf, y, S, BLOCK=BLK, num_warps=NW)

    def single_pass():
        # ★ 不变量：BLOCK 必须 ≥ S。否则 kernel 只会处理前 BLOCK 个元素，
        #   **不报错、结果静默错**（本启动器由 next_power_of_2 保证，这行是防以后改坏）
        assert BP2 >= S, f"single_pass 需要 BLOCK({BP2}) ≥ S({S})"
        _single_pass_kernel[grid](x, y, S, BLOCK=BP2, num_warps=nw_sp)

    # 名字, 有效字节/elem, 是否产出 y, 可调用对象
    arms = [
        ("copy", 8.0, False, lambda: y.copy_(x)),
        ("three_pass", 16.0, True, three_pass),
        ("online", 12.0, True, online),
        ("torch", 12.0, True, lambda: torch.softmax(x, dim=-1, out=y)),
        ("single_pass", 8.0, True, single_pass),
    ]

    # ── ① 正确性 ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(
        f"S={S:<6} R={R:<7} 工作集 {2 * x.numel() * 4 / 2**20:.0f} MiB   "
        f"block={BLK} warps={args.warps} ｜ single_pass BLOCK={BP2} warps={nw_sp}"
    )
    print(f"{'=' * 100}")
    print('【一、正确性】与 fp64 标准答案比；"—" 表示该臂不可用')
    ok = {}
    for name, _, produces, fn in arms:
        if not produces:
            ok[name] = None
            continue
        try:
            y.zero_()
            fn()
            torch.cuda.synchronize()
            d = (y - ref).abs().max().item()
            # 相对误差也要看：fp32 的 softmax 结果量级 ~1/S，绝对误差会被 S 稀释
            rel = ((y - ref).abs() / (ref.abs() + 1e-30)).max().item()
            ok[name] = d
            flag = "✅" if d <= 2e-6 else "❌"
            print(f"  {name:<12} max|Δ| = {d:.3e}   max 相对 = {rel:.3e}   {flag}（阈值 2e-6）")
        except Exception as e:
            ok[name] = None
            msg = str(e).splitlines()[0][:60]
            if "UnsupportedLanguageConstruct" in type(e).__name__:
                msg = "该配置不受支持，检查下面的异常"
            print(f"  {name:<12} [跳过] {msg}")

    # ── ①b (m, l) 级比对：把「算法错」与「kernel 错」分开 ───────────────────
    if ok.get("online") is not None:
        try:
            _online_reduce_kernel[grid](x, m_buf, l_buf, S, BLOCK=BLK)
            torch.cuda.synchronize()
            m_ref, l_ref = ref_online_torch(x, BLK)  # fp64 + torch 版分块重缩放
            dm = (m_buf.double() - m_ref).abs().max().item()
            dl = ((l_buf.double() - l_ref).abs() / l_ref.abs().clamp_min(1e-30)).max().item()
            print(f"\n  (m, l) vs torch 分块参考：max|Δm| = {dm:.3e}   max 相对 Δl = {dl:.3e}")
            print(
                f"  ⇒ {'✅ 算法与规格一致（错只可能在 kernel 实现）' if dm < 1e-5 and dl < 1e-6 else '❌ 算法层就偏了'}"
            )
        except Exception as e:
            print(f"  (m, l) 比对跳过：{type(e).__name__}: {str(e).splitlines()[0][:60]}")

    # ── ② 分块不变性（只测 online；三种 block 结果必须一致）─────────────────
    if ok.get("online") is not None:
        outs = []
        for b in args.blocks:
            try:
                _online_reduce_kernel[grid](x, m_buf, l_buf, S, BLOCK=b, num_warps=NW)
                _normalize_kernel[grid](x, m_buf, l_buf, y, S, BLOCK=b, num_warps=NW)
                torch.cuda.synchronize()
                outs.append((b, y.clone()))
            except Exception as e:
                print(f"  [block={b}] 失败: {type(e).__name__}")
        if len(outs) >= 2:
            base_b, base = outs[0]
            print(f"\n【二、分块不变性】以 block={base_b} 为基准（差 ≤ 1e-6 才算重缩放写对）")
            worst = 0.0
            for b, t in outs[1:]:
                d = (t - base).abs().max().item()
                worst = max(worst, d)
                print(f"  block={b:<5} vs {base_b:<5}  max|Δ| = {d:.3e}")
            print(f"  ⇒ 最坏 {worst:.3e}  {'✅' if worst <= 1e-6 else '❌ 重缩放公式有问题'}")

    # ── ③ 性能（同一轮内交替、轮转顺序、min 估计）───────────────────────────
    res = {name: [] for name, *_ in arms}
    skipped = {}
    for rep in range(args.repeat):
        rot = arms[rep % len(arms) :] + arms[: rep % len(arms)]
        for name, bpe, produces, fn in rot:
            if name in skipped:
                continue
            try:
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                res[name].append(time_once(fn, args.inner))
            except Exception as e:
                skipped[name] = (
                    "kernel 未实现"
                    if "UnsupportedLanguageConstruct" in type(e).__name__
                    else f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"
                )

    if skipped:
        print()
        for name, why in skipped.items():
            print(f"  [跳过] {name:<12} {why}")

    print("\n【三、性能】（min 估计；『占 copy %』是**时间比**，可以 >100%）")
    print(
        f"  {'臂':<12}{'字节/elem':>10}{'min ms':>10}{'中位 ms':>10}{'GB/s':>10}{'占 copy %':>11}   预测"
    )
    print("  " + "-" * 90)
    pred = {
        "copy": "100%（标尺）",
        "three_pass": "190~200%",
        "online": "140~150%",
        "torch": "140~155%",
        "single_pass": "95~102%",
    }
    copy_ms = min(res["copy"]) if res.get("copy") else None
    for name, bpe, _, _ in arms:
        v = res.get(name)
        if not v:
            print(f"  {name:<12}{bpe:>10.2f}{'—':>10}{'—':>10}{'—':>10}{'—':>11}   {pred[name]}")
            continue
        mn = min(v)
        share = f"{mn / copy_ms * 100:.1f}%" if copy_ms else "—"
        print(
            f"  {name:<12}{bpe:>10.2f}{mn:>10.4f}{statistics.median(v):>10.4f}"
            f"{gbps(x.numel() * bpe, mn):>10.1f}{share:>11}   {pred[name]}"
        )

    # ── ④ 判读：字节模型是否成立 + roofline ─────────────────────────────────
    print("\n【四、判读】")
    if copy_ms and res.get("three_pass") and res.get("online") and res.get("single_pass"):
        t3, t2, t1 = min(res["three_pass"]), min(res["online"]), min(res["single_pass"])
        print(
            f"  遍数效应（都用 copy 归一化）：three_pass {t3 / copy_ms * 100:.0f}% : "
            f"online {t2 / copy_ms * 100:.0f}% : single {t1 / copy_ms * 100:.0f}%"
            f"　（字节模型 200 : 150 : 100）"
        )
        print(
            f"  实测 online / three_pass = {t2 / t3:.3f}（字节模型 0.75）"
            f"　⇒ {'✅ 与模型吻合' if abs(t2 / t3 - 0.75) < 0.06 else '❌ 偏离模型，找非带宽瓶颈'}"
        )
        print(
            f"  实测 single_pass / copy   = {t1 / copy_ms:.3f}（模型 1.00）"
            f"　⇒ {'✅ 已达 copy 水平（exp 免费）' if t1 / copy_ms < 1.05 else '⚠️ 高于 copy，查寄存器压力/occupancy'}"
        )
    # 算术强度：每元素约 1 次 max + 1 次 exp + 1 次乘 + 1 次加 ≈ 5 FLOPs（exp 记 ~2）
    ai = 5.0 / 8.0
    print(
        f"  算术强度 ≈ {ai:.2f} FLOP/B（按 5 FLOPs / 8 B）≪ ridge point 58.5（FP32 CUDA core）"
        f"　⇒ **纯 memory-bound，exp 应当免费**"
    )
    print("  每条结论都要能回答：「这个数字的物理上限是多少？差的那部分具体是什么？」")

    del x, y, ref, m_buf, l_buf
    torch.cuda.empty_cache()
    return {name: (min(v) if v else None) for name, v in res.items()}, copy_ms


def main():
    ap = argparse.ArgumentParser(description="online softmax：数值 + 访存遍数")
    ap.add_argument("--elems", type=int, default=2**26, help="总元素数（默认 2^26 = 256 MiB fp32）")
    ap.add_argument("--seq-lens", type=str, default="1024,4096,16384", help="S 列表")
    ap.add_argument("--block", type=int, default=256, help="three_pass / online 的分块大小")
    ap.add_argument(
        "--warps",
        type=int,
        default=4,
        help="three_pass / online 的 num_warps（约束：BLOCK ≥ 8×32×warps 才能每 lane 16 B）",
    )
    ap.add_argument("--blocks", type=str, default="64,128,256,1024", help="分块不变性扫描")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--prewarm-sec", type=float, default=3.0)
    args = ap.parse_args()
    args.blocks = [int(b) for b in args.blocks.split(",")]

    if not torch.cuda.is_available():
        raise SystemExit("没有 CUDA 设备")

    print("=" * 100)
    print("a01  online softmax —— 分块重缩放的数值正确性 + 按「读几遍」排的访存成本")
    print("=" * 100)
    print(f"卡      : {torch.cuda.get_device_name(0)}   理论带宽 {SPEC_GBPS:.0f} GB/s")
    print(f"协议    : warmup={args.warmup} inner={args.inner} repeat={args.repeat}（min + 轮转）")
    print(f"GPU 起始: {clocks()}")
    prewarm(args.prewarm_sec)
    print(f"预热后  : {clocks()}")

    summary = {}
    for S in [int(s) for s in args.seq_lens.split(",")]:
        summary[S] = run_seq(S, args)

    if len(summary) > 1:
        print(f"\n{'=' * 100}")
        print("【跨 S 汇总】占 copy %（single_pass 在 S 大时会失效，这正是「必须融合」的动机）")
        print(f"{'=' * 100}")
        print(f"  {'S':>8}{'three_pass':>13}{'online':>10}{'torch':>10}{'single_pass':>14}")
        for S, (r, c) in summary.items():

            def fmt(k):
                return f"{r[k] / c * 100:.0f}%" if (c and r.get(k)) else "—"

            print(
                f"  {S:>8}{fmt('three_pass'):>13}{fmt('online'):>10}"
                f"{fmt('torch'):>10}{fmt('single_pass'):>14}"
            )

    print(f"\n  GPU 结束: {clocks()}")


if __name__ == "__main__":
    main()
