"""Build exact-length token-ID workloads from a local text corpus."""

import argparse
import hashlib
import json
from pathlib import Path
import random

from inferbench.cli import manifest
from inferbench.io import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-tokens", type=int, default=128)
    parser.add_argument("--long-tokens", type=int, default=4096)
    parser.add_argument("--short-requests", type=int, default=38)
    parser.add_argument("--long-requests", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.short_tokens, args.long_tokens, args.short_requests, args.long_requests) < 1:
        raise ValueError("Lengths and request counts must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokens = tokenizer.encode(args.corpus.read_text(encoding="utf-8"), add_special_tokens=False)
    needed = (args.short_requests + 8) * args.short_tokens + args.long_requests * args.long_tokens
    if len(tokens) < needed:
        raise ValueError(f"Need at least {needed} corpus tokens for disjoint slices")
    # Disjoint source intervals avoid accidentally warming on a heldout prefix.
    # These are raw continuation slices, not task-quality chat prompts.
    rng = random.Random(args.seed)
    offset = rng.randrange(len(tokens) - needed + 1)
    rows = []
    for task, split, length, count in [
        ("short", "tune", args.short_tokens, 8),
        ("short", "heldout", args.short_tokens, args.short_requests),
        ("long", "heldout", args.long_tokens, args.long_requests),
    ]:
        for i in range(count):
            ids = tokens[offset : offset + length]
            rows.append(
                {
                    "id": f"{task}_{split}_{i}",
                    "task": task,
                    "split": split,
                    "token_ids": ids,
                    "source_offset": offset,
                    "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                }
            )
            offset += length
    save(
        args.output,
        {
            "seed": args.seed,
            "tokenizer": str(args.tokenizer),
            "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            "template": "Raw continuation; disjoint corpus intervals; no chat template",
            "rows": rows,
        },
    )
    manifest(args.output)
    print(f"Saved {len(rows)} prompts to {args.output}")


if __name__ == "__main__":
    main()
