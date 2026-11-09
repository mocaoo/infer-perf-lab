"""GPTQModel calibration from frozen BF16 weights. AWQ uses quantize_own_awq.py."""

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
    parser.add_argument("method", choices=["gptq"])
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.samples <= 128:
        raise ValueError("samples must be in [1, 128]")
    name = f"{args.method}_n{args.samples}"
    dest = MODELS / f"Qwen3-8B-Own-{name}"
    if (ROOT / "checkpoints" / f"{name}.json").exists() or dest.exists():
        raise FileExistsError("Use fresh output/model directories")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(20261104)
    from gptqmodel import GPTQModel, QuantizeConfig, DEVICE_THREAD_POOL
    from gptqmodel.quantization import METHOD, FORMAT
    from gptqmodel.utils.backend import BACKEND

    rows = json.loads((ROOT / "calibration/token_inputs.json").read_text())["rows"][: args.samples]
    assert len(rows) == args.samples
    config = QuantizeConfig(
        bits=4,
        group_size=128,
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        lm_head=False,
        offload_to_disk=True,
        offload_to_disk_path=str((ROOT / "offload") / name),
        wait_for_submodule_finalizers=True,
        failsafe=None,
        device="cuda:0",
    )
    backend = BACKEND.TORCH
    metadata = {
        "name": name,
        "method": args.method,
        "samples": args.samples,
        "sequence_length": 512,
        "total_tokens": args.samples * 512,
        "calibration_inputs_sha256": hashlib.sha256(
            json.dumps([r["input_ids"] for r in rows]).encode()
        ).hexdigest(),
        "base": str(BASE),
        "output": str(dest),
        "calibration_dtype": "float16 (same for AWQ/GPTQ)",
        "quantization_backend": backend.value,
        "serving_backend": "pending vLLM verification",
        "packages": {
            p: md.version(p)
            for p in ["gptqmodel", "torch", "transformers", "triton", "safetensors"]
        },
        "configuration": config.to_dict(),
        "seed": 20261104,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    save(ROOT / "quantization" / f"{name}_started.json", metadata)
    start = time.perf_counter()
    try:
        DEVICE_THREAD_POOL.wait()
        model = GPTQModel.load(
            str(BASE),
            quantize_config=config,
            dtype=torch.float16,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        torch.cuda.reset_peak_memory_stats()
        loaded = time.perf_counter()
        calibration = [
            {"input_ids": r["input_ids"], "attention_mask": r["attention_mask"]} for r in rows
        ]
        result = model.quantize(calibration, batch_size=1, calibration_sort=None, backend=backend)
        DEVICE_THREAD_POOL.wait()
        torch.cuda.synchronize()
        quantified = time.perf_counter()
        model.save(str(dest))
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(BASE, local_files_only=True).save_pretrained(dest)
        DEVICE_THREAD_POOL.wait()
        files = [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(dest.glob("*.safetensors"))
        ]
        assert files
        metadata.update(
            load_seconds=loaded - start,
            quantize_seconds=quantified - loaded,
            total_seconds=time.perf_counter() - start,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            weights=files,
            exported_config=json.loads((dest / "config.json").read_text()),
            quantization_log=result,
        )
        save(ROOT / "checkpoints" / f"{name}.json", metadata)
        print("EXPORTED_AND_HASHED", name, flush=True)
    finally:
        DEVICE_THREAD_POOL.shutdown(wait=True)


if __name__ == "__main__":
    main()
