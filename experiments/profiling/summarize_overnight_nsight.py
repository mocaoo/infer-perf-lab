"""Validate real trace content and launch-correlate GPU work to CPU NVTX scopes."""

import bisect
import csv
import sqlite3
import statistics as st
from inferbench.io import read, save, stamp
from pathlib import Path
import argparse

ROOT = Path("results/profile")


def union_ns(intervals):
    total = 0
    end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def main():
    folder = ROOT / "nsight"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    scope_rows = []
    audit = []
    for dest in sorted(folder.iterdir()):
        if not dest.is_dir():
            continue
        db = dest / "trace.sqlite"
        phases = dest / "phases.json"
        if not db.exists() or not phases.exists():
            audit.append(
                {"tag": dest.name, "ok": False, "reason": "missing SQLite or phase events"}
            )
            continue
        try:
            with sqlite3.connect(db) as c:
                epoch = c.execute(
                    "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
                ).fetchone()[0]
                kernels = c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
                ranges = c.execute(
                    "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE eventType=59 AND text LIKE 'spec/%' ORDER BY start"
                ).fetchall()
                by_thread = {}
                for i, (a, b, label, tid) in enumerate(ranges):
                    by_thread.setdefault(tid, []).append((a, b, label, i))
                starts = {tid: [r[0] for r in rs] for tid, rs in by_thread.items()}
                associated = [[] for r in ranges]
                query = """SELECT k.start,k.end,r.start,r.end,r.globalTid
                         FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                         ON k.correlationId=r.correlationId
                         AND k.globalPid=(r.globalTid & 0xFFFFFFFFFF000000)"""
                for ka, kb, ra, rb, tid in c.execute(query):
                    if tid not in by_thread:
                        continue
                    j = bisect.bisect_right(starts[tid], ra) - 1
                    if j >= 0:
                        a, b, label, i = by_thread[tid][j]
                        if a <= ra and rb <= b:
                            associated[i].append((ka, kb))
                phase_counts = []
                for phase in read(phases):
                    begin = round(phase["start_unix"] * 1e9) - epoch
                    end = round(phase["end_unix"] * 1e9) - epoch
                    selected = [
                        (max(a, begin), min(b, end)) for a, b in kernels if a < end and b > begin
                    ]
                    phase_counts.append(len(selected))
                    common = {
                        "tag": dest.name,
                        "task": phase["task"],
                        "concurrency": phase["concurrency"],
                    }
                    rows.append(
                        {
                            **common,
                            "window_seconds": (end - begin) / 1e9,
                            "kernel_count": len(selected),
                            "summed_kernel_ms": sum(b - a for a, b in selected) / 1e6,
                            "kernel_union_busy_fraction": union_ns(selected) / (end - begin),
                        }
                    )
                    for label in sorted(set(r[2] for r in ranges)):
                        spans = []
                        unions = []
                        count = 0
                        for i, (a, b, t, tid) in enumerate(ranges):
                            if t != label or a < begin or b > end or not associated[i]:
                                continue
                            intervals = associated[i]
                            spans.append(
                                (max(b for a, b in intervals) - min(a for a, b in intervals)) / 1e6
                            )
                            unions.append(union_ns(intervals) / 1e6)
                            count += len(intervals)
                        scope_rows.append(
                            {
                                **common,
                                "scope": label,
                                "instances_with_kernels": len(spans),
                                "median_gpu_span_ms": st.median(spans) if spans else None,
                                "median_kernel_union_ms": st.median(unions) if unions else None,
                                "kernel_count": count,
                            }
                        )
                audit.append(
                    {
                        "tag": dest.name,
                        "ok": bool(kernels) and len(phase_counts) == 2 and all(phase_counts),
                        "kernel_count": len(kernels),
                        "phase_kernel_counts": phase_counts,
                        "nvtx_scope_count": len(ranges),
                        "launch_correlated_kernels": sum(len(a) for a in associated),
                    }
                )
        except Exception as e:
            audit.append({"tag": dest.name, "ok": False, "error": repr(e)})
    for name, data in [("phase_summary.csv", rows), ("phase_gpu_scopes.csv", scope_rows)]:
        if data:
            with (folder / name).open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    save(
        folder / "trace_audit.json",
        {
            "at": stamp(),
            "captures": audit,
            "all_nonempty": bool(audit) and all(r["ok"] for r in audit),
        },
    )
    lines = [
        "# Nsight时间线证据",
        "",
        "独立诊断运行，不用于正式TPOT加速比。GPU busy是kernel区间去重后占客户端时间窗口的比例，不是SM或Tensor Core利用率。kernel累计时间重叠时可超过墙钟。",
        "",
        "| 配置 | 并发 | kernel数 | kernel累计ms | GPU时间覆盖 |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f'| {r["tag"]} | {r["concurrency"]} | {r["kernel_count"]} | {r["summed_kernel_ms"]:.2f} | {r["kernel_union_busy_fraction"]:.1%} |'
        for r in rows
    ]
    lines += [
        "",
        "## GPU归因",
        "",
        "用CUDA runtime correlation ID、进程与线程关联CPU NVTX范围。GPU span是对应kernel首尾跨度，包含间隙；union仅合并kernel执行区间。target_execute包含prefill、decode/verify及准备，不是纯验证。各范围时间不能直接相加或解释为端到端关键路径百分比。",
        "",
        "| 配置 | 并发 | 范围 | 关联调用数 | GPU span中位ms | kernel union中位ms |",
        "|---|---:|---|---:|---:|---:|",
    ]
    lines += [
        f'| {r["tag"]} | {r["concurrency"]} | {r["scope"]} | {r["instances_with_kernels"]} | {r["median_gpu_span_ms"]:.3f} | {r["median_kernel_union_ms"]:.3f} |'
        for r in scope_rows
        if r["instances_with_kernels"]
    ]
    lines += [
        "",
        "完整性见 trace_audit.json；没有关联kernel的范围不能用CPU时长替代GPU耗时。未采集硬件计数器时，不声称某管线已饱和。",
    ]
    (folder / "Nsight实测结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("NSIGHT_CONTENT_AUDIT", len(audit), len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="Directory containing nsight/<tag>/trace.sqlite and phases.json"
    )
    args = parser.parse_args()
    ROOT = args.run_dir
    main()
