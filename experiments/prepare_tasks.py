"""Freeze task source revisions and tokenized tune/heldout splits; no model downloads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from inferbench.io import save

DATA = {
    "chat": ("HuggingFaceH4/mt_bench_prompts", "raw/question.jsonl"),
    "code": ("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"),
    "math": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
}
ROOT = Path("data")
MODELS = Path("models")
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
REVISIONS = {}


def get(url, timeout=50):
    import requests

    for retry in range(4):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception:
            if retry == 3:
                raise
            time.sleep(2 * (retry + 1))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def datasets():
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq

    tk = AutoTokenizer.from_pretrained(MODELS / "Qwen3-8B", local_files_only=True)
    output = []
    sources = {}
    for task, (repo, rel) in DATA.items():
        revision = REVISIONS.get(task) or get(f"{ENDPOINT}/api/datasets/{repo}").json()["sha"]
        dest = ROOT / "datasets" / task / revision / Path(rel).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(get(f"{ENDPOINT}/datasets/{repo}/resolve/{revision}/{rel}").content)
        sources[task] = {"repo": repo, "revision": revision, "file": rel, "sha256": sha(dest)}
        if task == "chat":
            records = [json.loads(x) for x in dest.read_text().splitlines() if x]
        else:
            records = pq.read_table(dest).to_pylist()
        rng = random.Random(20261103)
        rng.shuffle(records)
        for i, r in enumerate(records):
            if task == "chat":
                user = r["prompt"][0]
                rid = str(r["question_id"])
                reference = r.get("reference", [])
            elif task == "code":
                user = (
                    "Complete the following Python function. Return only Python code, including the function definition.\n\n"
                    + r["prompt"]
                )
                rid = r["task_id"]
                reference = {
                    "prompt": r["prompt"],
                    "test": r["test"],
                    "entry_point": r["entry_point"],
                }
            else:
                user = (
                    r["question"]
                    + '\nSolve the problem step by step. End with "#### <numerical answer>".'
                )
                rid = str(i)
                reference = r["answer"]
            ids = tk.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            if len(ids) > 7000:
                continue
            split = "tune" if i < 8 else "heldout"
            output.append(
                {
                    "task": task,
                    "id": f"{task}_{rid}",
                    "split": split,
                    "source_index_shuffled": i,
                    "prompt_tokens": len(ids),
                    "token_ids": ids,
                    "user": user,
                    "reference": reference,
                    "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                }
            )
    save(
        ROOT / "datasets" / "manifest.json",
        {
            "sources": sources,
            "seed": 20261103,
            "template": "base tokenizer; enable_thinking=False; first turn only",
            "split": "8 tune items/task, heldout excludes tune; same prompt IDs across arms",
            "rows": output,
        },
    )
    print("DATASETS_READY", len(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh directory for source files and datasets/manifest.json",
    )
    parser.add_argument(
        "--revisions",
        type=Path,
        help="Optional JSON mapping chat/code/math to frozen dataset commit IDs",
    )
    args = parser.parse_args()
    ROOT, MODELS = args.output_dir, args.model_root
    ROOT.mkdir(parents=True, exist_ok=False)
    if args.revisions:
        REVISIONS = json.loads(args.revisions.read_text(encoding="utf-8"))
    datasets()
