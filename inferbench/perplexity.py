"""Teacher-forced WikiText NLL; preserves the original disjoint-window protocol."""

import argparse
import asyncio
import hashlib
import math
import json
from pathlib import Path
import aiohttp
from .io import save


async def post(session, path, body):
    async with session.post(BASE + path, json=body) as response:
        response.raise_for_status()
        data = await response.json()
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return data


BASE = "http://127.0.0.1:8023"


async def ppl(args):
    dest = args.output
    if dest.exists():
        raise FileExistsError(dest)
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    path = args.corpus
    ids = tk.encode(path.read_text(), add_special_tokens=False)
    wins = [ids[i : i + 2048] for i in range(0, len(ids) - 2048 + 1, 2048)]
    if not wins:
        raise ValueError("Corpus must contain at least 2048 tokens")
    timeout = aiohttp.ClientTimeout(total=600, sock_read=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        sem = asyncio.Semaphore(4)

        async def one(i, w):
            async with sem:
                obj = await post(
                    session,
                    "/v1/completions",
                    {
                        "model": args.model,
                        "prompt": w,
                        "max_tokens": 1,
                        "temperature": 0,
                        "prompt_logprobs": 1,
                        "return_token_ids": True,
                    },
                )
                lp = obj["choices"][0]["prompt_logprobs"]
                nll = []
                for j in range(1, len(w)):
                    d = lp[j]
                    value = d.get(str(w[j]), d.get(w[j])) if d else None
                    if value is None:
                        raise RuntimeError(f"Missing true token logprob window {i} index {j}")
                    nll.append(-value["logprob"])
                return {
                    "window": i,
                    "token_ids_sha256": hashlib.sha256(json.dumps(w).encode()).hexdigest(),
                    "n_tokens": len(nll),
                    "nll_sum": sum(nll),
                    "token_nll": nll,
                }

        windows = await asyncio.gather(*(one(i, w) for i, w in enumerate(wins)))
    total = sum(w["n_tokens"] for w in windows)
    mean = sum(w["nll_sum"] for w in windows) / total
    result = {
        "tag": args.tag,
        "window": 2048,
        "n_windows": len(windows),
        "n_tokens": total,
        "mean_nll": mean,
        "ppl": math.exp(mean),
        "text_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "windows": windows,
        "scope": "Teacher-forced prompt-forward NLL; KV quantization must also be checked with generated tasks because this does not isolate cached decode error.",
    }
    save(dest, result)
    print(f'PPL {args.tag}: {result["ppl"]:.5f}')
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8023")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()
    BASE = args.base_url.rstrip("/")
    asyncio.run(ppl(args))
