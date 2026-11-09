"""Freeze training-only calibration token IDs and original-model identity."""

import hashlib
import json
import random
import time

from paths import ROOT, BASE, CORPUS
import os

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def get(url):
    import requests

    for i in range(6):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            return r
        except Exception:
            if i == 5:
                raise
            time.sleep(2)


def main():
    if not CORPUS.is_file():
        raise FileNotFoundError("Set INFERBENCH_TEST_CORPUS to the WikiText-2 test text")
    ROOT.mkdir(parents=True, exist_ok=True)
    dest = ROOT / "calibration/train.parquet"
    if not dest.exists():
        meta = get(f"{HF_ENDPOINT}/api/datasets/Salesforce/wikitext").json()
        revision = meta["sha"]
        rel = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(
            get(f"{HF_ENDPOINT}/datasets/Salesforce/wikitext/resolve/{revision}/{rel}").content
        )
        save(
            ROOT / "calibration/source.json",
            {
                "repo": "Salesforce/wikitext",
                "revision": revision,
                "file": rel,
                "split": "train",
                "sha256": sha(dest),
            },
        )
    assert sha(dest) == "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    text = "\n\n".join(pq.read_table(dest)["text"].to_pylist())
    ids = tk.encode(text, add_special_tokens=False)
    positions = list(range(0, len(ids) - 511, 512))
    random.Random(20261104).shuffle(positions)
    rows = []
    for i, start in enumerate(positions[:128]):
        tokens = ids[start : start + 512]
        rows.append(
            {
                "id": f"wiki_train_{i}",
                "start": start,
                "input_ids": tokens,
                "attention_mask": [1] * 512,
                "sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            }
        )
    save(
        ROOT / "calibration/token_inputs.json",
        {
            "seed": 20261104,
            "samples": 128,
            "tokens_per_sample": 512,
            "total_tokens": 65536,
            "template": "raw train continuation, no chat template",
            "ablation": "first32 is a nested subset; unchanged order within both algorithms",
            "rows": rows,
        },
    )
    # Test-only prompts are not read into calibration. Check exact full-window duplication as an additional audit.
    test = CORPUS
    testids = tk.encode(test.read_text(), add_special_tokens=False)
    train_windows = {tuple(r["input_ids"]) for r in rows}
    collisions = sum(
        tuple(testids[i : i + 512]) in train_windows for i in range(len(testids) - 511)
    )
    assert collisions == 0
    save(
        ROOT / "calibration/separation_audit.json",
        {
            "exact_512token_matches_in_test": collisions,
            "test_sha256": sha(test),
            "scope": "Does not establish absence of semantic near-duplicates or pretraining contamination.",
        },
    )
    checks = [
        {"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(BASE.glob("*.safetensors"))
    ]
    save(
        ROOT / "original_model.json",
        {
            "path": str(BASE),
            "config": json.loads((BASE / "config.json").read_text()),
            "weights": checks,
            "tokenizer_sha256": sha(BASE / "tokenizer.json"),
        },
    )
    print("CALIBRATION_FROZEN", len(rows), flush=True)


if __name__ == "__main__":
    main()
