"""HTTP/SSE measurements adapted from the original task/quantization suite."""

import asyncio
import hashlib
from pathlib import Path
import statistics as st
import time

import aiohttp

from .io import read, save, stamp
from .metrics import counter, counter_delta, metrics as scrape_metrics
from .sse import events as sse_events

# One run per process. CLI configures these before starting any async work.
ROOT = Path("results/run")
MANIFEST = Path("configs/prompts.example.json")
BASE = "http://127.0.0.1:8023"
MODEL = "Qwen3-8B"
SEED = 20261103


def log(message):
    ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{stamp()} {message}"
    print(line, flush=True)
    with (ROOT / "driver.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def pools(task, split="heldout", count=None):
    rows = [r for r in read(MANIFEST)["rows"] if r["task"] == task and r["split"] == split]
    return rows[:count] if count else rows


async def metrics(session):
    return await scrape_metrics(session, BASE)


async def post(session, path, body):
    async with session.post(BASE + path, json=body) as r:
        data = await r.json()
        if r.status != 200 or "error" in data:
            raise RuntimeError(f"HTTP {r.status}: {str(data)[:1000]}")
        return data


async def request(session, prompt, output=256, natural=False, logprobs=None):
    """Measure visible chunk arrivals; token counts come from usage, not chunk count."""
    body = {
        "model": MODEL,
        "prompt": prompt["token_ids"],
        "max_tokens": output,
        "temperature": 0,
        "ignore_eos": not natural,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
        "seed": SEED,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    started = time.perf_counter()
    events = []
    ids = []
    texts = []
    usage = {}
    finish = None
    probs = []
    async with session.post(BASE + "/v1/completions", json=body) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:1000]}")
        async for obj in sse_events(r.content):
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices", [])
            if len(choices) > 1 or any(c.get("index", 0) != 0 for c in choices):
                raise RuntimeError("Only one completion per request is supported")
            for choice in choices:
                delta_ids = choice.get("token_ids") or []
                text = choice.get("text") or ""
                if delta_ids or text:
                    events.append(
                        {
                            "ms": (time.perf_counter() - started) * 1000,
                            "token_ids": delta_ids,
                            "text": text,
                        }
                    )
                    ids.extend(delta_ids)
                    texts.append(text)
                if choice.get("logprobs"):
                    probs.append(choice["logprobs"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    n = usage.get("completion_tokens", 0)
    if (
        not events
        or type(n) is not int
        or n < 1
        or n > output
        or (not natural and n != output)
        or usage.get("prompt_tokens") != len(prompt["token_ids"])
    ):
        raise RuntimeError(f'Invalid response usage: {usage}; prompt={prompt["id"]}')
    if len(ids) != n:
        raise RuntimeError(f"Token ID coverage mismatch {len(ids)} != {n}")
    if any(type(token) is not int or token < 0 for token in ids) or finish is None:
        raise RuntimeError("Invalid output token IDs or missing finish_reason")
    # Only nonempty text arrivals define user-visible TTFT / chunks.
    visible = [e for e in events if e["text"]]
    if not visible:
        raise RuntimeError("No user-visible text")
    gaps = [b["ms"] - a["ms"] for a, b in zip(visible, visible[1:])]
    tpot = (visible[-1]["ms"] - visible[0]["ms"]) / (n - 1) if n > 1 else None
    return {
        "id": prompt["id"],
        "task": prompt["task"],
        "prompt_sha256": prompt["prompt_sha256"],
        "prompt_tokens": len(prompt["token_ids"]),
        "completion_tokens": n,
        "token_ids": ids,
        "text": "".join(texts),
        "events": events,
        "ttft_ms": visible[0]["ms"],
        "tpot_ms": tpot,
        "max_chunk_gap_ms": max(gaps, default=0),
        "chunk_gaps_ms": gaps,
        "chunk_gap_exceed_200ms": sum(g > 200 for g in gaps),
        "chunk_count": len(visible),
        "e2e_ms": (time.perf_counter() - started) * 1000,
        "finish_reason": finish,
        "logprobs": probs,
    }


async def batch(session, prompts, concurrency, output, natural=False, lp=None):
    if concurrency < 1 or output < 1 or not prompts:
        raise ValueError("Positive concurrency/output and nonempty prompts required")
    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        async with sem:
            return await request(session, p, output, natural, lp)

    start = time.perf_counter()
    tasks = [asyncio.create_task(one(p)) for p in prompts]
    try:
        rows = await asyncio.gather(*tasks)
        return rows, time.perf_counter() - start
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def measure(
    tag, task, rows, concurrency=1, repeats=3, output=256, natural=False, dest_group="cases"
):
    dest = ROOT / dest_group / f"{tag}_{task}_c{concurrency}.json"
    if dest.exists():
        raise FileExistsError(f"Result already exists; use a new run directory: {dest}")
    if repeats < 1 or not rows:
        raise ValueError("Positive repeats and nonempty heldout rows required")
    timeout = aiohttp.ClientTimeout(total=900, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warmup inputs are tuning-only, excluded from this case's counter interval.
        warm = pools(task, "tune", min(concurrency, 8))
        if not warm:
            raise ValueError("Manifest needs tuning-only warmup rows")
        await batch(session, warm, concurrency, output, natural)
        await asyncio.sleep(1.2)
        before = await metrics(session)
        samples = []
        done = asyncio.Event()

        async def watch():
            while not done.is_set():
                try:
                    samples.append({"time": time.time(), "values": await metrics(session)})
                except Exception as e:
                    samples.append({"error": repr(e)})
                try:
                    await asyncio.wait_for(done.wait(), 0.5)
                except asyncio.TimeoutError:
                    pass

        monitor = asyncio.create_task(watch())
        rounds = []
        try:
            for rep in range(repeats):
                out, wall = await batch(session, rows, concurrency, output, natural)
                valid = [r["tpot_ms"] for r in out if r["tpot_ms"] is not None]
                rounds.append(
                    {
                        "repeat": rep,
                        "requests": out,
                        "wall_seconds": wall,
                        "tpot_ms": st.median(valid) if valid else None,
                        "ttft_ms": st.median(r["ttft_ms"] for r in out),
                        "output_tokens_per_second": sum(r["completion_tokens"] for r in out) / wall,
                    }
                )
                log(
                    f'CASE {tag} {task} c={concurrency} rep={rep+1}/{repeats} TPOT={rounds[-1]["tpot_ms"]}'
                )
        finally:
            done.set()
            await monitor
        expected = sum(r["completion_tokens"] for b in rounds for r in b["requests"])
        for _ in range(12):
            await asyncio.sleep(0.5)
            after = await metrics(session)
            delta = counter_delta(before, after)
            if counter(delta, "generation_tokens_total") >= expected:
                break
    accepted = counter(delta, "spec_decode_num_accepted_tokens_total")
    drafted = counter(delta, "spec_decode_num_draft_tokens_total")
    drafts = counter(delta, "spec_decode_num_drafts_total")
    result = {
        "tag": tag,
        "task": task,
        "concurrency": concurrency,
        "natural_eos": natural,
        "output_cap": output,
        "schema_version": 1,
        "base_url": BASE,
        "model": MODEL,
        "seed": SEED,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "at": stamp(),
        "prompt_ids": [r["id"] for r in rows],
        "metrics_before": before,
        "metrics_after": after,
        "metrics_delta": delta,
        "metrics_samples": samples,
        "expected_generation_tokens": expected,
        "observed_generation_tokens": counter(delta, "generation_tokens_total"),
        "counter_boundary_valid": bool(
            any(k.split("{", 1)[0] == "vllm:generation_tokens_total" for k in before)
        )
        and abs(counter(delta, "generation_tokens_total") - expected) < 0.5,
        "acceptance_rate": accepted / drafted if drafted else None,
        "accepted_per_round": accepted / drafts if drafts else None,
        "output_per_draft_round": expected / drafts if drafts else None,
        "output_per_draft_round_note": "Request-total output / draft rounds includes initial and terminal effects; not exact verification-round output.",
        "rounds": rounds,
    }
    save(dest, result)
    if not result["counter_boundary_valid"]:
        raise RuntimeError(f"Invalid counter boundary {tag} {task}")
    return result


def score(case):
    return st.median(r["tpot_ms"] for r in case["rounds"] if r["tpot_ms"] is not None)


def fail(tag, error):
    save(ROOT / "failures" / f"{tag}.json", {"tag": tag, "error": repr(error), "at": stamp()})
    log(f"FAILED {tag}: {error}")
