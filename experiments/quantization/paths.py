"""Local paths for Qwen3-only experiments; GPU runtimes stay isolated."""

import os
from pathlib import Path
from inferbench.io import save as save, stamp as stamp

MODELS = Path(os.environ.get("INFERBENCH_MODELS", "models")).expanduser().resolve()
ROOT = (
    Path(os.environ.get("INFERBENCH_QUANT_RESULTS", "results/quantization")).expanduser().resolve()
)
BASE = MODELS / "Qwen3-8B"
CORPUS = Path(os.environ.get("INFERBENCH_TEST_CORPUS", "data/wikitext2_test.txt")).expanduser()
KV_ROOT = (
    Path(os.environ.get("INFERBENCH_KV_RESULTS", "results/kv-calibration")).expanduser().resolve()
)
MANIFEST = Path(os.environ.get("INFERBENCH_MANIFEST", "configs/prompts.example.json")).expanduser()


def pools(task, split="tune", count=None):
    import json

    rows = [
        r
        for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["rows"]
        if r["task"] == task and r["split"] == split
    ]
    if count and len(rows) < count:
        raise ValueError(f"Need {count} {task}/{split} calibration prompts")
    return rows[:count] if count else rows
