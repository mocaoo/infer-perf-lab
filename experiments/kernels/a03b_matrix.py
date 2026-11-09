#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rotate four BF16/FP8, per-head/fused-head arms and record kernel resources."""

import argparse
import json
import statistics
from pathlib import Path

import torch

import a02_flash_attention as base
from a03_decode_fp8 import _combine_kernel, _decode_fused_kernel, _decode_kernel, quant_kv_fp8

DEV = "cuda"
OUT_DIR = Path("results/a03_matrix")


def ref_sdpa(q, k, v, GRP, scale):
    """fp32 SDPA 参考（GQA 需按 GRP 展开 KV）。"""
    kk = k.repeat_interleave(GRP, dim=0)
    vv = v.repeat_interleave(GRP, dim=0)
    o = torch.nn.functional.scaled_dot_product_attention(
        q.float(), kk.float(), vv.float(), scale=scale
    )
    return o


def rel_rms(a, b):
    d = (a.float() - b.float()).pow(2).mean().sqrt()
    return (d / b.float().pow(2).mean().sqrt()).item()


def kernel_meta(fn, args, kwargs, grid, name):
    """编译并读资源。注意：n_regs 需 _init_handles() 之后才可读。"""
    try:
        ck = fn.warmup(*args, grid=grid, **kwargs)
        if isinstance(ck, tuple):
            ck = ck[0]
        try:
            ck._init_handles()
        except Exception:
            pass
        md = getattr(ck, "metadata", {})
        shm = md.get("shared") if isinstance(md, dict) else getattr(md, "shared", None)
        nw = md.get("num_warps") if isinstance(md, dict) else getattr(md, "num_warps", None)
        ns = md.get("num_stages") if isinstance(md, dict) else getattr(md, "num_stages", None)
        regs = getattr(ck, "n_regs", "?")
        spill = getattr(ck, "n_spills", "?")
        print(
            f"    {name:<22} n_regs={regs}/thread  n_spills={spill}  SMEM={shm} B"
            f"  warps={nw} stages={ns}"
        )
        return {
            "name": name,
            "n_regs": regs,
            "n_spills": spill,
            "smem": shm,
            "warps": nw,
            "stages": ns,
            "grid": list(grid),
        }
    except Exception as e:
        print(f"    {name:<22} 元数据读取失败：{type(e).__name__}: {str(e)[:90]}")
        return {"name": name, "error": str(e)[:120]}


def run_one(H, HKV, G, SKV, args, jsonl):
    D = args.head_dim
    GRP = H // HKV
    GM = G // args.bm
    scale = D**-0.5
    torch.manual_seed(0)

    q = torch.randn(H, G, D, device=DEV, dtype=torch.bfloat16) * 0.1
    k = torch.randn(HKV, SKV, D, device=DEV, dtype=torch.bfloat16) * 0.1
    v = torch.randn(HKV, SKV, D, device=DEV, dtype=torch.bfloat16) * 0.1
    kf, ks = quant_kv_fp8(k)
    vf, vs = quant_kv_fp8(v)
    # dequant 参考（与 kernel 同口径：bf16 scale × bf16 k）
    k_dq = (kf.to(torch.bfloat16).float() * ks.bfloat16().float()[..., None]).to(torch.bfloat16)
    v_dq = (vf.to(torch.bfloat16).float() * vs.bfloat16().float()[..., None]).to(torch.bfloat16)

    ref = ref_sdpa(q, k, v, GRP, scale)
    ref_dq = ref_sdpa(q, k_dq, v_dq, GRP, scale)
    print(f"\n{'═' * 108}")
    print(
        f"  SKV={SKV}  H={H} HKV={HKV} GRP={GRP} G={G} D={D}  BM={args.bm} BN={args.bn}  "
        f"KV bf16 = {2 * HKV * SKV * D * 2 / 2**20:.0f} MiB"
    )

    outs = {
        name: torch.zeros(H, G, D, device=DEV, dtype=torch.bfloat16)
        for name in ("ph_bf16", "ph_fp8", "fu_bf16", "fu_fp8")
    }
    acc = torch.full((args.splits, H, G, D), float("nan"), device=DEV, dtype=torch.float32)
    mb = torch.full((args.splits, H, G), float("nan"), device=DEV, dtype=torch.float32)
    lb = torch.full((args.splits, H, G), float("nan"), device=DEV, dtype=torch.float32)

    JBLK = SKV // args.splits
    g_ph = (H * GM, args.splits)
    g_fu = (HKV * GM, args.splits)
    g_comb = (H * GM,)

    def launch(fused, fp8, out):
        kk, vv = (kf, vf) if fp8 else (k, v)
        sks, svs = (ks, vs) if fp8 else (q, q)
        fn = _decode_fused_kernel if fused else _decode_kernel
        grid = g_fu if fused else g_ph
        fn[grid](
            q,
            kk,
            vv,
            sks,
            svs,
            acc,
            mb,
            lb,
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
            FP8=fp8,
        )
        _combine_kernel[g_comb](acc, mb, lb, out, H, G, D=D, SPLITS=args.splits, BM=args.bm, GM=GM)

    fns = {
        "ph_bf16": lambda: launch(False, False, outs["ph_bf16"]),
        "ph_fp8": lambda: launch(False, True, outs["ph_fp8"]),
        "fu_bf16": lambda: launch(True, False, outs["fu_bf16"]),
        "fu_fp8": lambda: launch(True, True, outs["fu_fp8"]),
    }

    # ── 正确性（先跑一遍再判）──────────────────────────────
    for name, fn in fns.items():
        outs[name].zero_()
        fn()
    torch.cuda.synchronize()
    r = {n: rel_rms(outs[n], ref) for n in outs}
    r_dq = {n: rel_rms(outs[n], ref_dq) for n in ("ph_fp8", "fu_fp8")}
    r_cross = {
        "fu_bf16 vs ph_bf16": rel_rms(outs["fu_bf16"], outs["ph_bf16"]),
        "fu_fp8 vs ph_fp8": rel_rms(outs["fu_fp8"], outs["ph_fp8"]),
    }
    print(
        f"  正确性：ph_bf16 {r['ph_bf16']:.2e}｜fu_bf16 {r['fu_bf16']:.2e}"
        f"（vs fp32 SDPA）　ph_fp8 {r['ph_fp8']:.2e}｜fu_fp8 {r['fu_fp8']:.2e}"
        f"（≈2~4e-2 = e4m3 量化代价）"
    )
    print(
        f"  交叉：{r_cross['fu_bf16 vs ph_bf16']:.2e}（融合 vs per-head, bf16）"
        f"｜{r_cross['fu_fp8 vs ph_fp8']:.2e}（fp8）⇒ 应 ≈0（同一算法）"
    )
    print(
        f"  fp8 vs **dequant 后**参考：ph {r_dq['ph_fp8']:.2e}｜fu {r_dq['fu_fp8']:.2e}"
        f"（≈①⇒ dequant 没引入额外误差）"
    )

    print(
        f"  grid：per-head {g_ph} = {g_ph[0] * g_ph[1]} CTA｜"
        f"fused {g_fu} = {g_fu[0] * g_fu[1]} CTA｜combine {g_comb[0]} CTA"
    )

    if args.meta:
        print("  ★ 编译资源（本次实际产物）：")
        kw = dict(BM=args.bm, BN=args.bn, SPLITS=args.splits, JBLK=JBLK, GM=GM, GRP=GRP)
        m1 = kernel_meta(
            _decode_kernel,
            (q, k, v, q, q, acc, mb, lb, H, G, SKV, D, scale),
            dict(kw, FP8=False),
            g_ph,
            "split per-head bf16",
        )
        m2 = kernel_meta(
            _decode_fused_kernel,
            (q, k, v, q, q, acc, mb, lb, H, G, SKV, D, scale),
            dict(kw, FP8=False),
            g_fu,
            "split fused bf16",
        )
        m3 = kernel_meta(
            _decode_kernel,
            (q, kf, vf, ks, vs, acc, mb, lb, H, G, SKV, D, scale),
            dict(kw, FP8=True),
            g_ph,
            "split per-head fp8",
        )
        m4 = kernel_meta(
            _decode_fused_kernel,
            (q, kf, vf, ks, vs, acc, mb, lb, H, G, SKV, D, scale),
            dict(kw, FP8=True),
            g_fu,
            "split fused fp8",
        )
        m5 = kernel_meta(
            _combine_kernel,
            (acc, mb, lb, out_dummy(H, G, D), H, G),
            dict(D=D, SPLITS=args.splits, BM=args.bm, GM=GM),
            g_comb,
            "combine",
        )
    else:
        m1 = m2 = m3 = m4 = m5 = None

    # ── 计时：四臂轮转 ────────────────────────────────────
    names = ["ph_bf16", "ph_fp8", "fu_bf16", "fu_fp8"]
    raw = {n: [] for n in names}
    for rep in range(args.repeat):
        shift = rep % len(names)
        for name in names[shift:] + names[:shift]:
            for _ in range(args.warmup):
                fns[name]()
            torch.cuda.synchronize()
            raw[name].append(base.time_once(fns[name], args.inner))

    print(f"\n  {'方案':<9}{'min ms':>9}{'中位 ms':>9}{'max/min':>9}{'us/行q':>9}{'有效GB/s':>10}")
    stat = {}
    for name in names:
        ts = raw[name]
        mn, md = min(ts), statistics.median(ts)
        mx = max(ts)
        by = 2 * HKV * SKV * D * (1 if name.endswith("fp8") else 2)
        stat[name] = {
            "min": mn,
            "med": md,
            "max": mx,
            "raw": ts,
            "us_per_row": mn / G * 1000,
            "gbps": by / (mn * 1e-3) / 1e9,
        }
        print(
            f"  {name:<9}{mn:>9.4f}{md:>9.4f}{mx / mn:>8.2f}x"
            f"{mn / G * 1000:>9.2f}{by / (mn * 1e-3) / 1e9:>10.0f}"
        )

    # ── 三个比例（回答三个问题）────────────────────────────
    s = stat
    print(
        f"\n  ① 融合收益（同精度同 splits）：bf16 {s['ph_bf16']['min'] / s['fu_bf16']['min']:.2f}×"
        f"｜fp8 {s['ph_fp8']['min'] / s['fu_fp8']['min']:.2f}×"
    )
    print(
        f"  ② FP8 收益（同实现同 splits）：per-head {s['ph_bf16']['min'] / s['ph_fp8']['min']:.2f}×"
        f"｜fused {s['fu_bf16']['min'] / s['fu_fp8']['min']:.2f}×"
        f"（纯读字节减半的理想上界 2.00×）"
    )
    best = {}
    for tag in ("ph", "fu"):
        cand = [n for n in names if n.startswith(tag + "_")]
        best[tag] = min(cand, key=lambda n: s[n]["min"])
    print(f"  ③ 每种实现的最佳精度（本 splits 内）：{best}")

    rec = {
        "SKV": SKV,
        "H": H,
        "HKV": HKV,
        "GRP": GRP,
        "G": G,
        "D": D,
        "BM": args.bm,
        "BN": args.bn,
        "splits": args.splits,
        "grid_ph": list(g_ph),
        "grid_fu": list(g_fu),
        "grid_comb": list(g_comb),
        "stat": {n: {k: v for k, v in d.items() if k != "raw"} for n, d in stat.items()},
        "raw": {n: d["raw"] for n, d in stat.items()},
        "rms": r,
        "rms_dq": r_dq,
        "rms_cross": r_cross,
        "meta": [x for x in (m1, m2, m3, m4, m5) if x],
    }
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    del q, k, v, kf, vf, k_dq, v_dq, ref, ref_dq, acc, mb, lb, outs
    torch.cuda.empty_cache()


def out_dummy(H, G, D):
    return torch.zeros(H, G, D, device=DEV, dtype=torch.bfloat16)


def split_combine_timing(H, HKV, G, SKV, args, jsonl):
    """分开测 split / combine（四方案），用于定位；不加到完整流程上。"""
    D = args.head_dim
    GRP = H // HKV
    GM = G // args.bm
    scale = D**-0.5
    torch.manual_seed(0)
    q = torch.randn(H, G, D, device=DEV, dtype=torch.bfloat16) * 0.1
    k = torch.randn(HKV, SKV, D, device=DEV, dtype=torch.bfloat16) * 0.1
    v = torch.randn(HKV, SKV, D, device=DEV, dtype=torch.bfloat16) * 0.1
    kf, ks = quant_kv_fp8(k)
    vf, vs = quant_kv_fp8(v)
    acc = torch.zeros((args.splits, H, G, D), device=DEV, dtype=torch.float32)
    mb = torch.zeros((args.splits, H, G), device=DEV, dtype=torch.float32)
    lb = torch.ones((args.splits, H, G), device=DEV, dtype=torch.float32)
    o = out_dummy(H, G, D)
    JBLK = SKV // args.splits
    g_ph, g_fu = (H * GM, args.splits), (HKV * GM, args.splits)

    print(f"\n{'─' * 100}")
    print(f"  分段计时：SKV={SKV} splits={args.splits}（JBLK={JBLK}）")

    def mk(fused, fp8, part):
        kk, vv = (kf, vf) if fp8 else (k, v)
        sks, svs = (ks, vs) if fp8 else (q, q)
        fn = _decode_fused_kernel if fused else _decode_kernel
        grid = g_fu if fused else g_ph

        def split_only():
            fn[grid](
                q,
                kk,
                vv,
                sks,
                svs,
                acc,
                mb,
                lb,
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
                FP8=fp8,
            )

        def combine_only():
            _combine_kernel[(H * GM,)](
                acc, mb, lb, o, H, G, D=D, SPLITS=args.splits, BM=args.bm, GM=GM
            )

        return split_only if part == "split" else combine_only

    print(f"  {'方案':<9}{'split ms':>10}{'combine ms':>12}{'占比':>8}")
    rows = []
    for fused, tag in ((False, "ph"), (True, "fu")):
        for fp8, fmt in ((False, "bf16"), (True, "fp8")):
            sp = mk(fused, fp8, "split")
            cb = mk(fused, fp8, "combine")
            sp()
            torch.cuda.synchronize()
            for f in (sp, cb):
                for _ in range(args.warmup):
                    f()
            torch.cuda.synchronize()
            ts = min(base.time_once(sp, args.inner) for _ in range(args.repeat))
            tc = min(base.time_once(cb, args.inner) for _ in range(args.repeat))
            rows.append(
                {
                    "arm": f"{tag}_{fmt}",
                    "split": ts,
                    "combine": tc,
                    "splits": args.splits,
                    "SKV": SKV,
                }
            )
            print(f"  {tag + '_' + fmt:<9}{ts:>10.4f}{tc:>12.4f}{tc / ts * 100:>7.1f}%")
    with open(jsonl, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"split_timing": r}, ensure_ascii=False) + "\n")
    del q, k, v, kf, vf, acc, mb, lb, o
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-list", type=str, default="8")
    ap.add_argument("--hkv-list", type=str, default="2")
    ap.add_argument("--g-list", type=str, default="16")
    ap.add_argument("--skv-list", type=str, default="131072")
    ap.add_argument("--splits-list", type=str, default="32")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--bm", type=int, default=16)
    ap.add_argument("--bn", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--meta", action="store_true", default=True)
    ap.add_argument("--no-meta", dest="meta", action="store_false")
    ap.add_argument("--split-timing", action="store_true")
    ap.add_argument("--st-splits", type=str, default="32,128", help="对哪些 splits 做分段计时")
    ap.add_argument("--tag", type=str, default="matrix")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUT_DIR / f"{args.tag}.jsonl"
    print("=" * 108)
    print("a03b  四方案 × splits：per-head / head-fused × BF16 / FP8（同进程轮转）")
    print(
        f"协议：warmup={args.warmup} inner={args.inner} repeat={args.repeat}"
        f"｜BM={args.bm} BN={args.bn}｜输出 {jsonl}"
    )
    print("=" * 108)

    for SKV in [int(x) for x in args.skv_list.split(",")]:
        for splits in [int(x) for x in args.splits_list.split(",")]:
            for H in [int(x) for x in args.h_list.split(",")]:
                for HKV in [int(x) for x in args.hkv_list.split(",")]:
                    for G in [int(x) for x in args.g_list.split(",")]:
                        args.splits = splits
                        assert (
                            G % args.bm == 0
                            and SKV % splits == 0
                            and (SKV // splits) % args.bn == 0
                        ), "形状不整除"
                        run_one(H, HKV, G, SKV, args, jsonl)
                        if args.split_timing and splits in [
                            int(x) for x in args.st_splits.split(",")
                        ]:
                            split_combine_timing(H, HKV, G, SKV, args, jsonl)


if __name__ == "__main__":
    main()
