#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split-K attention and numerically stable partial-state combination.

Reuse a02 references; compare complete split+combine latency and FP64 error."""

import argparse
import os
import sys

import torch
import triton
import triton.language as tl

# 复用 a02 的参考实现 / 计时 / 报错打印 —— 不让「fp64 参考」有两份实现
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a02_flash_attention as base  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# 阶段 ①：split kernel —— 每个 CTA 只负责 j 的一段
# ══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _split_kernel(
    Q,
    K,
    V,
    Acc,
    M,
    L,
    S,
    D: tl.constexpr,
    scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
    JBLK,
):
    """每段保存未归一化 (acc, m, l)；只在 combine 最后除以 l。"""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_d)
    m_i = tl.full([BM], -1.0e38, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(pid_s * JBLK, (pid_s + 1) * JBLK, BN):
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
    tl.store(Acc + pid_s * S * D + offs_m[:, None] * D + offs_d[None, :], acc)
    tl.store(M + pid_s * S + offs_m, m_i)
    tl.store(L + pid_s * S + offs_m, l_i)


# ══════════════════════════════════════════════════════════════════════════════
# 阶段 ②：combine kernel —— 把 SPLITS 段的 partial 按 online softmax 合并
# ══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _combine_kernel(Acc, M, L, O, S, D: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr):
    """按全局最大值重缩放各段，再归一化。

    m=max(m_s), w_s=exp(m_s-m), O=sum(acc_s*w_s)/sum(l_s*w_s)。
    M/L 的段 stride 是 S；权重广播到 [BM,D] 时必须用 [:,None]。
    """
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_g = tl.full([BM], -1.0e38, tl.float32)
    for s in range(SPLITS):
        m_g = tl.maximum(m_g, tl.load(M + s * S + offs_m))
    acc = tl.zeros([BM, D], tl.float32)
    l_g = tl.zeros([BM], tl.float32)
    for s in range(SPLITS):
        ms = tl.load(M + s * S + offs_m)
        ls = tl.load(L + s * S + offs_m)
        w = tl.exp(ms - m_g)
        l_g = l_g + ls * w
        a_s = tl.load(Acc + s * S * D + offs_m[:, None] * D + offs_d[None, :])
        acc = acc + a_s * w[:, None]
    o = acc / l_g[:, None]
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], o.to(tl.bfloat16))


# ══════════════════════════════════════════════════════════════════════════════
# 参考实现与基准流程
# ══════════════════════════════════════════════════════════════════════════════
def run_seq(S, args):
    D = args.head_dim
    dev = "cuda"
    scale = 1.0 / (D**0.5)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    peak = base.PEAK_TFLOPS_BF16

    torch.manual_seed(0)
    q = torch.randn(S, D, device=dev, dtype=torch.bfloat16) * 0.1
    k = torch.randn(S, D, device=dev, dtype=torch.bfloat16) * 0.1
    v = torch.randn(S, D, device=dev, dtype=torch.bfloat16) * 0.1
    o_ref = torch.empty(S, D, device=dev, dtype=torch.bfloat16)
    o_one = torch.empty(S, D, device=dev, dtype=torch.bfloat16)
    ref = base.ref_attention_fp64(q, k, v, scale)

    grid1 = (S // args.bm,)
    kw = {}
    if args.num_warps:
        kw["num_warps"] = args.num_warps
    if args.num_stages:
        kw["num_stages"] = args.num_stages

    def flash():
        base._flash_kernel[grid1](q, k, v, o_one, S, D, scale, BM=args.bm, BN=args.bn, **kw)

    print(f"\n{'=' * 96}")
    print(
        f"S={S}  D={D}  BM={args.bm} BN={args.bn}   SM={sm_count}   "
        f"roofline floor = {4 * S * S * D / (peak * 1e12) * 1e3:.4f} ms"
    )
    print(f"{'=' * 96}")

    # 基准臂：单 kernel flash（同时给出 η_sm 的基线，用来算预测）
    o_one.zero_()
    flash()
    torch.cuda.synchronize()
    rms_one = base.rel_rms(o_one.float(), ref)
    print(
        f"【基准】flash（单 kernel）  vs fp64 相对 RMS = {rms_one:.3e}   "
        f"{'✅' if rms_one <= 1e-2 else '❌'}"
    )
    for _ in range(args.warmup):  # ★ 计时前必须 warmup：首次调用含 JIT 与冷启动
        flash()
    torch.cuda.synchronize()
    ts = [base.time_once(flash, args.inner) for _ in range(args.repeat)]
    t_one = min(ts)
    flops = 4.0 * S * S * D
    eta_occ1 = grid1[0] / (-(-grid1[0] // sm_count) * sm_count)
    sm_floor1 = -(-grid1[0] // sm_count) * (flops / grid1[0]) / (peak * 1e12 / sm_count) * 1e3
    eta_sm_base = sm_floor1 / t_one
    print(
        f"  t = {t_one:.4f} ms   grid = {grid1[0]}   η_occ = {eta_occ1:.3f}   "
        f"η_sm = {eta_sm_base:.3f}   MFU = {flops / (t_one * 1e-3) / (peak * 1e12):.1%}"
    )

    print(
        f"\n{'splits':>7}{'grid':>7}{'η_occ':>8}{'SMfloor':>10}{'预测 ms':>10}"
        f"{'实测 ms':>10}{'vs 单核':>9}{'vs fp64':>10}{'splits=1 一致':>14}"
    )
    print("  " + "-" * 84)

    rows = []
    for sp in [int(x) for x in args.splits_list.split(",")]:
        if S % sp or (S // sp) % args.bn:
            print(f"{sp:>7}   ⚠ 跳过：S//SPLITS={S // sp} 不能被 BN={args.bn} 整除")
            continue
        JBLK = S // sp
        # 用 NaN 初始化中间状态，便于发现未覆盖的写入。

        Acc = torch.full((sp, S, D), float("nan"), device=dev, dtype=torch.float32)
        Mbuf = torch.full((sp, S), float("nan"), device=dev, dtype=torch.float32)
        Lbuf = torch.full((sp, S), float("nan"), device=dev, dtype=torch.float32)
        o_sp = torch.empty(S, D, device=dev, dtype=torch.bfloat16)
        g2 = (S // args.bm, sp)

        def split_run():
            _split_kernel[g2](
                q,
                k,
                v,
                Acc,
                Mbuf,
                Lbuf,
                S,
                D,
                scale,
                BM=args.bm,
                BN=args.bn,
                SPLITS=sp,
                JBLK=JBLK,
                **kw,
            )
            _combine_kernel[grid1](Acc, Mbuf, Lbuf, o_sp, S, D, SPLITS=sp, BM=args.bm, **kw)

        try:
            o_sp.zero_()
            split_run()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"{sp:>7}   ❌ 运行失败：")
            print(base.fmt_error(e))
            continue

        rms_ref = base.rel_rms(o_sp.float(), ref)
        if rms_ref != rms_ref:
            print(f"{sp:>7}   ❌ 输出是 nan ⇒ 检查 split/combine 输出与输入范围")
            continue
        if o_sp.abs().max().item() == 0.0:
            # 全零不属于本基准的正常输出，避免把错误结果纳入测速。
            print(f"{sp:>7}   ❌ 输出全 0 ⇒ 检查 split/combine 输出与输入范围")
            continue
        rms_one_cmp = base.rel_rms(o_sp.float(), o_one.float())
        for _ in range(args.warmup):
            split_run()
        torch.cuda.synchronize()
        ts = [base.time_once(split_run, args.inner) for _ in range(args.repeat)]
        t_sp = min(ts)

        # ★ 分开测：split kernel 与 combine kernel 各自的耗时
        if args.split_timing:

            def split_only():
                _split_kernel[g2](
                    q,
                    k,
                    v,
                    Acc,
                    Mbuf,
                    Lbuf,
                    S,
                    D,
                    scale,
                    BM=args.bm,
                    BN=args.bn,
                    SPLITS=sp,
                    JBLK=JBLK,
                    **kw,
                )

            def combine_only():
                _combine_kernel[grid1](Acc, Mbuf, Lbuf, o_sp, S, D, SPLITS=sp, BM=args.bm, **kw)

            split_only()
            torch.cuda.synchronize()  # 先把 Acc/M/L 填好
            for _ in range(max(1, args.warmup // 2)):
                combine_only()
            torch.cuda.synchronize()
            t_s = min(base.time_once(split_only, args.inner) for _ in range(args.repeat))
            t_c = min(base.time_once(combine_only, args.inner) for _ in range(args.repeat))
            print(
                f"          ├─ split only {t_s * 1000:7.1f} µs ｜ "
                f"combine only {t_c * 1000:6.1f} µs ｜ 合计 {t_sp * 1000:7.1f} µs ｜ "
                f"combine 占比 {t_c / t_sp * 100:5.1f}%"
            )

        grid = g2[0] * sp
        waves = -(-grid // sm_count)
        eta = grid / (waves * sm_count)
        sm_floor = waves * (flops / grid) / (peak * 1e12 / sm_count) * 1e3
        pred = sm_floor / eta_sm_base  # 假设 η_sm 不变，只补并行度
        one_ok = (rms_one_cmp <= 1e-6) if sp == 1 else None
        one_txt = "—" if one_ok is None else ("✅" if one_ok else f"❌ {rms_one_cmp:.1e}")
        print(
            f"{sp:>7}{grid:>7}{eta:>8.3f}{sm_floor:>10.4f}{pred:>10.4f}"
            f"{t_sp:>10.4f}{t_one / t_sp:>8.2f}×{rms_ref:>10.2e}{one_txt:>14}"
        )
        rows.append((sp, grid, eta, t_sp, pred))

    print("\n【判据】")
    print(
        "  ① splits=1 必须与单 kernel 逐值一致（相对 RMS ≤ 1e-6）—— 分开「combine 数学错」与「split 地址错」"
    )
    print("  ② 所有 splits 的 vs fp64 相对 RMS ≤ 1e-2")
    print(
        "  ③ 实测 vs 预测：预测只假设「η_sm 不变、只补并行度」。明显高于预测 ⇒ combine 开销或访存变差"
    )
    print("  ④ 若某 splits 突然变慢：先看 grid 是否踩到 170 的整数倍（η_occ 那一列）")
    if rows:
        best = max(rows, key=lambda r: t_one / r[3])
        print(
            f"  ⇒ 目前最佳：splits={best[0]}（η_occ {best[2]:.3f}，提速 {t_one / best[3]:.2f}×，"
            f"实测 {best[3]:.4f} ms vs 预测 {best[4]:.4f} ms）"
        )

    del q, k, v, o_ref, o_one, ref
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="a02b：flash + split-K（flash-decoding）")
    ap.add_argument("--s-list", type=str, default="4096")
    ap.add_argument("--splits-list", type=str, default="1,2,4,8,16")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--num-warps", type=int, default=0)
    ap.add_argument("--num-stages", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--prewarm-sec", type=float, default=2.0)
    ap.add_argument(
        "--split-timing",
        action="store_true",
        help="额外分开测 split kernel 与 combine kernel 的耗时",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("没有 CUDA 设备")

    print("=" * 96)
    print("a02b  split-K（flash-decoding）—— 把并行度从 S/BM 提到 (S/BM)×SPLITS")
    print("=" * 96)
    print(
        f"卡      : {torch.cuda.get_device_name(0)}   峰值 {base.PEAK_TFLOPS_BF16} TFLOPS（@3090 MHz 规格）"
    )
    print(f"协议    : warmup={args.warmup} inner={args.inner} repeat={args.repeat}")
    base.prewarm(args.prewarm_sec)

    for S in [int(s) for s in args.s_list.split(",")]:
        run_seq(S, args)


if __name__ == "__main__":
    main()
