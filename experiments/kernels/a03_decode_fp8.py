#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decode/GQA microbenchmarks with fused FP8 KV dequantization.

Quantization error and attention arithmetic error are evaluated separately."""

import argparse
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a02_flash_attention as base  # noqa: E402


@triton.jit
def _decode_kernel(
    Q,
    K,
    V,
    Ks,
    Vs,
    Acc,
    M,
    L,
    H,
    G,
    SKV,
    D: tl.constexpr,
    scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
    JBLK,
    GM,
    GRP: tl.constexpr,
    FP8: tl.constexpr,
):
    """一个 CTA = (某个 head 的 BM 行 query) × (第 pid_s 段 KV)。"""
    pid = tl.program_id(0)
    pid_s = tl.program_id(1)
    h = pid // GM
    pid_m = pid % GM
    h_kv = h // GRP
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + (h * G + offs_m[:, None]) * D + offs_d[None, :])
    m_i = tl.full([BM], -1.0e38, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(pid_s * JBLK, (pid_s + 1) * JBLK, BN):
        offs_n = n0 + tl.arange(0, BN)
        off = (h_kv * SKV + offs_n[:, None]) * D + offs_d[None, :]
        if FP8:
            ks = tl.load(Ks + h_kv * SKV + offs_n).to(tl.bfloat16)
            vs = tl.load(Vs + h_kv * SKV + offs_n).to(tl.bfloat16)
            k = tl.load(K + off).to(tl.bfloat16) * ks[:, None]
            v = tl.load(V + off).to(tl.bfloat16) * vs[:, None]
        else:
            k = tl.load(K + off)
            v = tl.load(V + off)
        s = tl.dot(q, tl.trans(k)) * scale
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    base = (pid_s * H + h) * G
    tl.store(Acc + ((base + offs_m[:, None]) * D) + offs_d[None, :], acc)
    tl.store(M + base + offs_m, m_i)
    tl.store(L + base + offs_m, l_i)


@triton.jit
def _decode_fused_kernel(
    Q,
    K,
    V,
    Ks,
    Vs,
    Acc,
    M,
    L,
    H,
    G,
    SKV,
    D: tl.constexpr,
    scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
    JBLK,
    GM,
    GRP: tl.constexpr,
    FP8: tl.constexpr,
):
    """head-fused decode：**一个 CTA 同时处理 GRP 个 Q head × BM 行 query**，共用一份 KV。"""
    pid = tl.program_id(0)
    pid_s = tl.program_id(1)
    h_kv = pid // GM
    pid_m = pid % GM
    idx = tl.arange(0, GRP * BM)
    g = idx // BM
    m = idx % BM
    row = (h_kv * GRP + g) * G + pid_m * BM + m
    offs_d = tl.arange(0, D)
    q = tl.load(Q + row[:, None] * D + offs_d[None, :])
    m_i = tl.full([GRP * BM], -1.0e38, tl.float32)
    l_i = tl.zeros([GRP * BM], tl.float32)
    acc = tl.zeros([GRP * BM, D], tl.float32)
    for n0 in range(pid_s * JBLK, (pid_s + 1) * JBLK, BN):
        offs_n = n0 + tl.arange(0, BN)
        off = (h_kv * SKV + offs_n[:, None]) * D + offs_d[None, :]
        if FP8:
            ks = tl.load(Ks + h_kv * SKV + offs_n).to(tl.bfloat16)
            vs = tl.load(Vs + h_kv * SKV + offs_n).to(tl.bfloat16)
            k = tl.load(K + off).to(tl.bfloat16) * ks[:, None]
            v = tl.load(V + off).to(tl.bfloat16) * vs[:, None]
        else:
            k = tl.load(K + off)
            v = tl.load(V + off)

        s = tl.dot(q, tl.trans(k)) * scale
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    tl.store(Acc + (pid_s * H * G + row[:, None]) * D + offs_d[None, :], acc)
    tl.store(M + pid_s * H * G + row, m_i)
    tl.store(L + pid_s * H * G + row, l_i)


@triton.jit
def _combine_kernel(
    Acc, M, L, O, H, G, D: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr, GM: tl.constexpr
):
    """把 SPLITS 段的 partial 按 online softmax 合并。布局与 _decode_kernel 完全对称："""
    pid = tl.program_id(0)
    h = pid // GM
    pid_m = pid % GM
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_g = tl.full([BM], -1.0e38, tl.float32)
    for s in range(0, SPLITS):
        m_g = tl.maximum(m_g, tl.load(M + ((s * H + h) * G) + offs_m))
    acc = tl.zeros([BM, D], tl.float32)
    l_g = tl.zeros([BM], tl.float32)
    for s in range(SPLITS):
        base = (s * H + h) * G
        ms = tl.load(M + base + offs_m)
        ls = tl.load(L + base + offs_m)
        w = tl.exp(ms - m_g)
        l_g = l_g + ls * w
        a_s = tl.load(Acc + (base + offs_m[:, None]) * D + offs_d[None, :])
        acc = acc + a_s * w[:, None]
    o = acc / l_g[:, None]
    tl.store(O + (h * G + offs_m[:, None]) * D + offs_d[None, :], o.to(tl.bfloat16))


def quant_kv_fp8(x):
    """per-token 对称量化到 e4m3。★ 严格除法 x/s，不写 x*(1/s)（舍入会差，见 4.32）。"""
    amax = x.float().abs().amax(dim=-1).clamp_min(1e-8)  # [H, SKV]
    s = (amax / 448.0).to(torch.float32)
    qf = (x.float() / s[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return qf, s


def run_seq(H, HKV, G, SKV, args):
    D = args.head_dim
    assert H % HKV == 0, "H 必须是 HKV 的整数倍"
    GRP = H // HKV
    dev = "cuda"
    scale = 1.0 / (D**0.5)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    peak = base.PEAK_TFLOPS_BF16

    if args.qkv:
        blob = torch.load(args.qkv, map_location="cpu")
        rq, rk, rv = blob["q"], blob["k"], blob["v"]  # [H,S,D] / [HKV,S,D]
        assert (
            rq.shape[0] == H and rk.shape[0] == HKV
        ), f"真实张量形状不符：q{rq.shape} k{rk.shape}，期望 H={H} HKV={HKV}"

        def tile(t, n, axis=1):
            """沿 seq 轴平铺到长度 n（保持真实分布）"""
            r = -(-n // t.shape[axis])
            reps = [r if i == axis else 1 for i in range(t.dim())]
            return t.repeat(*reps).narrow(axis, 0, n).contiguous()

        q = tile(rq, G, 1).to(dev)
        k = tile(rk, SKV, 1).to(dev)
        v = tile(rv, SKV, 1).to(dev)
        print(
            f"  [真实张量] q{q.shape} k{k.shape} v{v.shape} "
            f"（amax: q={q.float().abs().max():.2f} k={k.float().abs().max():.2f} "
            f"v={v.float().abs().max():.1f}）"
        )
    else:
        torch.manual_seed(0)
        q = torch.randn(H, G, D, device=dev, dtype=torch.bfloat16) * 0.1
        k = torch.randn(HKV, SKV, D, device=dev, dtype=torch.bfloat16) * 0.1
        v = torch.randn(HKV, SKV, D, device=dev, dtype=torch.bfloat16) * 0.1
    kf, ks = quant_kv_fp8(k)
    vf, vs = quant_kv_fp8(v)

    # kernel 里是 bf16 的 scale 乘 bf16 的 k ⇒ 参考也必须这样算，否则 ② 会把 bf16 舍入算成 dequant 的错
    k_dq = (kf.to(torch.bfloat16).float() * ks.bfloat16().float()[..., None]).to(torch.bfloat16)
    v_dq = (vf.to(torch.bfloat16).float() * vs.bfloat16().float()[..., None]).to(torch.bfloat16)

    # ★ 参考也要按 GQA 映射：第 i 个 Q head 用第 i//GRP 个 KV head
    ref = torch.stack(
        [base.ref_attention_fp64(q[i], k[i // GRP], v[i // GRP], scale) for i in range(H)]
    )
    ref_dq = torch.stack(
        [base.ref_attention_fp64(q[i], k_dq[i // GRP], v_dq[i // GRP], scale) for i in range(H)]
    )

    o_bf16 = torch.zeros(H, G, D, device=dev, dtype=torch.bfloat16)
    o_fp8 = torch.zeros(H, G, D, device=dev, dtype=torch.bfloat16)
    o_sdpa = torch.zeros(H, G, D, device=dev, dtype=torch.bfloat16)

    assert G % args.bm == 0 and SKV % args.splits == 0 and (SKV // args.splits) % args.bn == 0
    GM = G // args.bm
    # head-fused 的 axis0 按 KV head 数算（比 per-head 少 GRP 倍）
    grid = ((HKV if args.fused else H) * GM, args.splits)
    gcomb = (H * GM,)
    JBLK = SKV // args.splits
    Acc = torch.full((args.splits, H, G, D), float("nan"), device=dev, dtype=torch.float32)
    Mb = torch.full((args.splits, H, G), float("nan"), device=dev, dtype=torch.float32)
    Lb = torch.full((args.splits, H, G), float("nan"), device=dev, dtype=torch.float32)

    def run(kk, vv, sks, svs, out, is_fp8):
        if args.fused:
            _decode_fused_kernel[grid](
                q,
                kk,
                vv,
                sks if sks is not None else q,
                svs if svs is not None else q,
                Acc,
                Mb,
                Lb,
                H,
                G,
                SKV,
                D,
                scale,
                BM=args.bm,
                BN=args.bn,
                SPLITS=args.splits,
                JBLK=JBLK,
                GM=GM,
                GRP=GRP,
                FP8=is_fp8,
            )
        else:
            _decode_kernel[grid](
                q,
                kk,
                vv,
                sks if sks is not None else q,
                svs if svs is not None else q,
                Acc,
                Mb,
                Lb,
                H,
                G,
                SKV,
                D,
                scale,
                BM=args.bm,
                BN=args.bn,
                SPLITS=args.splits,
                JBLK=JBLK,
                GM=GM,
                GRP=GRP,
                FP8=is_fp8,
            )
        _combine_kernel[gcomb](Acc, Mb, Lb, out, H, G, D=D, SPLITS=args.splits, BM=args.bm, GM=GM)

    def a_bf16():
        run(k, v, None, None, o_bf16, False)

    def a_fp8():
        run(kf, vf, ks, vs, o_fp8, True)

    def a_sdpa():
        # ★ GQA：sdpa 不会替你广播 head，必须先把 KV 按 GRP 展开成 H 份
        #   （这也是「为什么真实推理要专门写 GQA kernel」的直观原因）
        kk = k.repeat_interleave(GRP, dim=0)[None]
        vv = v.repeat_interleave(GRP, dim=0)[None]
        o = torch.nn.functional.scaled_dot_product_attention(q[None], kk, vv, scale=scale)
        o_sdpa.copy_(o[0])

    flops = 4.0 * H * G * SKV * D  # 计算量由 Q head 数决定
    by_bf16 = 2 * HKV * SKV * D * 2  # ★ KV 字节由 KV head 数决定 ⇒ GQA 下 AI ×GRP
    by_fp8 = 2 * HKV * SKV * D * 1 + 2 * HKV * SKV * 4
    waves = -(-(grid[0] * grid[1]) // sm)
    eta = grid[0] * grid[1] / (waves * sm)

    print(f"\n{'=' * 96}")
    print(
        f"H={H} HKV={HKV}（GRP={GRP}） G={G} SKV={SKV} D={D}  "
        f"BM={args.bm} BN={args.bn} SPLITS={args.splits}  "
        f"kernel={'head-fused' if args.fused else 'per-head'}"
    )
    print(
        f"   split grid={grid} = {grid[0] * grid[1]} CTA（η_occ {eta:.3f}）　"
        f"combine grid={gcomb[0]}　JBLK={JBLK}（{JBLK // args.bn} 轮/CTA）"
    )
    print(
        f"   AI = {flops / by_bf16:.1f} FLOP/B（bf16）/ {flops / by_fp8:.1f}（fp8）　ridge {base.RIDGE:.0f}"
        f" ⇒ **{'compute-bound' if flops / by_bf16 > base.RIDGE else 'memory-bound'}**"
        f"（AI = GRP×G = {GRP*G}，ridge {base.RIDGE:.0f}；GRP×G > ridge 时降字节没有意义）"
    )
    print(
        f"   KV 字节：bf16 {by_bf16 / 1e6:.0f} MB，fp8 {by_fp8 / 1e6:.0f} MB（比值 {by_fp8 / by_bf16:.2f}）"
    )
    print(f"{'=' * 96}")

    print("【一、正确性】")
    res = {}
    for name, fn, out in (
        ("bf16", a_bf16, o_bf16),
        ("fp8", a_fp8, o_fp8),
        ("sdpa", a_sdpa, o_sdpa),
    ):
        try:
            fn()
            torch.cuda.synchronize()
            res[name] = base.rel_rms(out.float(), ref)
        except Exception as e:
            res[name] = None
            print(f"  {name:<5} ❌ {type(e).__name__}")
            print(base.fmt_error(e))

    if res.get("bf16") is not None and o_bf16.abs().max().item() == 0.0:
        print(
            "  ❌ 输出全 0 ⇒ `_decode_kernel` / `_combine_kernel` 未写出有效结果，请检查输入范围与索引"
        )

    r1 = res.get("bf16")
    r3 = res.get("fp8")
    if r1 is not None:
        print(
            f"  ① bf16 vs fp64(bf16 KV)   = {r1:.3e}   {'✅' if r1 <= 1e-2 else '❌'}（阈值 1e-2）"
        )
    if r3 is not None:
        r2 = base.rel_rms(o_fp8.float(), ref_dq)
        # ① 没过时 ② 没有意义（否则 r1=r2=1.0 会「通过」）
        ok = (r1 is not None and r1 <= 1e-2) and (r2 <= max(1.5 * r1, 1e-3))
        print(
            f"  ② fp8  vs **dq 后 fp64**  = {r2:.3e}   {'✅' if ok else '❌'}"
            f"（判据：≤ max(1.5×①, 1e-3) ⇒ dequant 没引入额外误差）"
        )
        print(
            f"  ③ fp8  vs fp64(bf16 KV)  = {r3:.3e}   （≈1~3e-2 是 e4m3 三位尾数的**量化代价**，不是 bug）"
        )

    print("\n【二、性能】")
    print(f"  {'臂':<6}{'min ms':>10}{'中位 ms':>10}{'us/行q':>10}{'有效带宽 GB/s':>16}{'MFU':>9}")
    print("  " + "-" * 54)
    # ── 轮转臂顺序（counterbalancing）：每轮的起点循环左移一位 ──────────────────
    #   arms = [A,B,C] ⇒ rep0: A,B,C  rep1: B,C,A  rep2: C,A,B
    #   这样每个臂在 repeat 轮里"当第几个跑"各占一次，位置效应被平均掉。
    #   不轮转的话，第一个跑的臂永远占最有利（或最不利）的位置，min 只反映那一个位置。
    arms = [
        (n, f, b)
        for n, f, b in (
            ("bf16", a_bf16, by_bf16),
            ("fp8", a_fp8, by_fp8),
            ("sdpa", a_sdpa, by_bf16),
        )
        if res.get(n) is not None
    ]
    ts_all = {n: [] for n, _, _ in arms}
    for rep in range(args.repeat):
        shift = rep % len(arms) if arms else 0
        for name, fn, byt in arms[shift:] + arms[:shift]:
            for _ in range(args.warmup):  # 每轮都重新预热：否则这一轮第一个臂吃亏
                fn()
            torch.cuda.synchronize()
            ts_all[name].append(base.time_once(fn, args.inner))

    t = {}
    for name, fn, byt in arms:
        ts = ts_all[name]
        t[name] = min(ts)
        print(
            f"  {name:<6}{t[name]:>10.4f}{sorted(ts)[len(ts) // 2]:>10.4f}"
            f"{t[name] / G * 1000:>10.2f}"
            f"{byt / (t[name] * 1e-3) / 1e9:>16.0f}"
            f"{flops / (t[name] * 1e-3) / (peak * 1e12):>8.1%}"
        )

    if "bf16" in t and "fp8" in t:
        ideal = by_fp8 / by_bf16
        med = {n: sorted(v)[len(v) // 2] for n, v in ts_all.items()}
        print(
            f"\n  ⇒ fp8/bf16 时间比 = **{t['fp8'] / t['bf16']:.2f}×**（min 口径）"
            f"　中位口径 **{med['fp8'] / med['bf16']:.2f}×**"
            f"　（纯 memory-bound 理论值 {ideal:.2f}×）"
        )
        print(
            f"     两个口径差 {abs(med['fp8'] / med['bf16'] - t['fp8'] / t['bf16']) * 100:.1f}%，"
            f"越大说明两臂的抖动越不对称 ⇒ 报区间而不是单点。"
        )
        print(
            f"     越接近 {ideal:.2f}× ⇒ 收益来自读字节减半；明显更大 ⇒ dequant 的转换/算力开始显现。"
        )
    del q, k, v, kf, vf, k_dq, v_dq, ref, ref_dq
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="a03：decode 形状 + FP8 KV 融合")
    ap.add_argument("--h-list", type=str, default="8", help="Q head 数")
    ap.add_argument(
        "--hkv-list", type=str, default="8", help="KV head 数（=H 即 MHA；H//4 即 GQA）"
    )
    ap.add_argument("--g-list", type=str, default="16")
    ap.add_argument("--skv-list", type=str, default="131072")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument(
        "--qkv", type=str, default="", help="真实 Q/K/V 路径（dump_qkv.py 产出）；空=合成 randn*0.1"
    )
    ap.add_argument("--bm", type=int, default=16, help="query 方向块（decode 时小）")
    ap.add_argument("--bn", type=int, default=32, help="KV 方向块")
    ap.add_argument("--splits", type=int, default=32, help="KV 分片数")
    ap.add_argument(
        "--fused",
        action="store_true",
        help="用 head-fused 版 split kernel（一个 CTA 吃 GRP 个 Q head）",
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    print("=" * 96)
    print("a03  decode 形状的 attention + FP8 KV（dequant 融进内循环）")
    print("=" * 96)
    print(f"卡      : {torch.cuda.get_device_name(0)}   峰值 {base.PEAK_TFLOPS_BF16} TFLOPS")
    for SKV in [int(x) for x in args.skv_list.split(",")]:
        for H in [int(x) for x in args.h_list.split(",")]:
            for HKV in [int(x) for x in args.hkv_list.split(",")]:
                for G in [int(x) for x in args.g_list.split(",")]:
                    run_seq(H, HKV, G, SKV, args)


if __name__ == "__main__":
    main()
