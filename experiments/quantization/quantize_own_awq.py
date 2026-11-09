"""AutoAWQ0.2.9/Qwen3 version-pinned independent calibration/export."""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import time
from prepare_self_quant import sha
from paths import ROOT, BASE, MODELS, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    name = f"awq_n{args.samples}"
    if not 1 <= args.samples <= 128:
        raise ValueError("samples must be in [1, 128]")
    dest = MODELS / f"Qwen3-8B-Own-{name}"
    marker = ROOT / "checkpoints" / f"{name}.json"
    if marker.exists() or dest.exists():
        raise FileExistsError("Use a fresh output/model directory; exports are not overwritten")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(20261104)
    rows = json.loads((ROOT / "calibration/token_inputs.json").read_text())["rows"][: args.samples]
    assert len(rows) == args.samples
    cfg = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    meta = {
        "name": name,
        "method": "awq",
        "samples": args.samples,
        "sequence_length": 512,
        "total_tokens": args.samples * 512,
        "base": str(BASE),
        "output": str(dest),
        "calibration_dtype": "float16 (same as GPTQ)",
        "calibration_inputs_sha256": hashlib.sha256(
            json.dumps([r["input_ids"] for r in rows]).encode()
        ).hexdigest(),
        "configuration": cfg,
        "duo_scaling": True,
        "apply_clip": True,
        "parallel_calibration_samples": 1,
        "packages": {
            p: md.version(p)
            for p in ["autoawq", "torch", "transformers", "tokenizers", "safetensors"]
        },
        "tool_choice": "AutoAWQ0.2.9 isolated legacy-compatible environment; GPTQModel AWQ failures retained in logs.",
        "quantization_algorithm_modified": False,
    }
    save(ROOT / "quantization" / f"{name}_started.json", meta)
    start = time.perf_counter()
    model = AutoAWQForCausalLM.from_pretrained(
        str(BASE),
        trust_remote_code=False,
        torch_dtype=torch.float16,
        device_map="cpu",
        safetensors=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    loaded = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model.quantize(
        tokenizer,
        quant_config=cfg,
        calib_data=[r["input_ids"] for r in rows],
        max_calib_samples=args.samples,
        max_calib_seq_len=512,
        n_parallel_calib_samples=1,
        duo_scaling=True,
        apply_clip=True,
        max_chunk_memory=512 * 1024**2,
    )
    torch.cuda.synchronize()
    quantified = time.perf_counter()
    model.save_quantized(str(dest))
    tokenizer.save_pretrained(dest)
    weights = [
        {"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(dest.glob("*.safetensors"))
    ]
    assert weights
    meta.update(
        load_seconds=loaded - start,
        quantize_seconds=quantified - loaded,
        total_seconds=time.perf_counter() - start,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        weights=weights,
        exported_config=json.loads((dest / "config.json").read_text()),
    )
    save(marker, meta)
    print("EXPORTED_AND_HASHED", name, flush=True)


if __name__ == "__main__":
    main()
