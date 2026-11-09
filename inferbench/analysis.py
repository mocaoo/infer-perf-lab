"""Offline summaries with explicit per-round percentile aggregation."""

from pathlib import Path
import statistics as st

from .io import read, save


def percentile(values, q):
    if not 0 <= q <= 1:
        raise ValueError("Quantile must be in [0, 1]")
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def summarize(case):
    if not case.get("counter_boundary_valid"):
        raise ValueError("Refusing to summarize a case with an invalid counter boundary")
    rounds = case["rounds"]

    def med(field):
        values = [r[field] for r in rounds if r[field] is not None]
        return st.median(values) if values else None

    p99 = [
        percentile([r["tpot_ms"] for r in rd["requests"] if r["tpot_ms"] is not None], 0.99)
        for rd in rounds
    ]
    return {
        "tag": case["tag"],
        "task": case["task"],
        "concurrency": case["concurrency"],
        "requests": sum(len(r["requests"]) for r in rounds),
        "tpot_p50_ms": med("tpot_ms"),
        "ttft_p50_ms": med("ttft_ms"),
        "tpot_p99_ms": (
            st.median(v for v in p99 if v is not None) if any(v is not None for v in p99) else None
        ),
        "output_tokens_per_second": med("output_tokens_per_second"),
        "acceptance_rate": case["acceptance_rate"],
        "accepted_per_round": case["accepted_per_round"],
        "aggregation": "Median of per-round request medians/P99; P99 uses linear interpolation. Not pooled P99 or token ITL.",
    }


def report(folder):
    folder = Path(folder)
    rows = [summarize(read(p)) for p in sorted((folder / "cases").glob("*.json"))]
    if not rows:
        raise ValueError("No case JSON found under cases/")
    save(folder / "summary.json", rows)
    lines = [
        "# 本次运行摘要",
        "",
        "TPOT 单位 ms/token；每轮统计后取中位数。",
        "",
        "|配置|任务|并发|请求数|TPOT P50|TPOT P99|输出 token/s|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(value):
        return f"{value:.3f}" if value is not None else "N/A"

    for r in rows:
        lines.append(
            f'|{r["tag"]}|{r["task"]}|{r["concurrency"]}|{r["requests"]}|{fmt(r["tpot_p50_ms"])}|{fmt(r["tpot_p99_ms"])}|{r["output_tokens_per_second"]:.1f}|'
        )
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
