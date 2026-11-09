"""Release long requests while a bounded pool of short requests is decoding."""

import asyncio
import time

import aiohttp

from . import core
from .io import save, stamp


async def run(short, long, concurrency, delay, output, destination):
    if not short or not long or concurrency < 1 or delay < 0 or output < 1:
        raise ValueError("Mixed load requires both pools and positive capacities")
    timeout = aiohttp.ClientTimeout(total=900, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        warm = core.pools(short[0]["task"], "tune", min(concurrency, 8))
        if not warm:
            raise ValueError("Manifest needs tuning-only warmup rows")
        await core.batch(session, warm, concurrency, output)
        before = await core.metrics(session)
        start = time.perf_counter()
        sem = asyncio.Semaphore(concurrency)

        async def fire(prompt, group):
            if group == "long":
                await asyncio.sleep(max(0, delay - (time.perf_counter() - start)))

            async def send():
                sent = time.perf_counter() - start
                result = await core.request(session, prompt, output)
                return {**result, "group": group, "sent_seconds": sent}

            # Burst traffic must bypass the short-request semaphore. Otherwise it
            # queues behind all short requests and cannot interfere with decode.
            if group == "long":
                return await send()
            async with sem:
                return await send()

        tasks = [
            asyncio.create_task(fire(p, g))
            for g, pool in [("short", short), ("long", long)]
            for p in pool
        ]
        try:
            rows = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - start
        after = await core.metrics(session)
    bursts = [r["sent_seconds"] for r in rows if r["group"] == "long"]
    crossing = []
    overlap = set()
    for i, row in enumerate(rows):
        if row["group"] != "short":
            continue
        visible = [row["sent_seconds"] + e["ms"] / 1000 for e in row["events"] if e["text"]]
        for a, b in zip(visible, visible[1:]):
            if any(a <= sent < b for sent in bursts):
                crossing.append((b - a) * 1000)
                overlap.add(i)
    result = {
        "at": stamp(),
        "schema_version": 1,
        "kind": "mixed",
        "model": core.MODEL,
        "base_url": core.BASE,
        "seed": core.SEED,
        "concurrency_short": concurrency,
        "burst_delay_seconds": delay,
        "long_requests": len(long),
        "actual_burst_sent_seconds": bursts,
        "short_requests_with_crossing_chunk": len(overlap),
        "crossing_chunk_gaps_ms": crossing,
        "wall_seconds": wall,
        "requests": rows,
        "output_tokens_per_second": sum(r["completion_tokens"] for r in rows) / wall,
        "metrics_before": before,
        "metrics_after": after,
        "note": "Crossing a client send time is an association, not a server prefill execution boundary. Inspect server traces for causality. No crossing chunks means this run did not demonstrate interference.",
    }
    save(destination, result)
    return result
