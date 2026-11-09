"""Calibrate per-layer FP8 Q/K/V scales using tuning prompts; keep BF16 weights."""

import hashlib
import json

from paths import KV_ROOT as ROOT, MODELS, pools, save, stamp


def main():
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    from safetensors.torch import save_file

    dest = MODELS / "Qwen3-8B-KVcalibrated"
    marker = ROOT / "calibration/kv_scales.json"
    if marker.exists() or dest.exists():
        raise FileExistsError("Use a fresh KV result/model directory")
    model = AutoModelForCausalLM.from_pretrained(
        MODELS / "Qwen3-8B",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    buf = {}
    maxima = {i: {k: 0.0 for k in ["q", "k", "v"]} for i in range(36)}
    if len(model.model.layers) != 36:
        raise ValueError("This recipe is specific to 36-layer Qwen3-8B")
    handles = []
    for i, layer in enumerate(model.model.layers):
        for name, module in [
            ("q", layer.self_attn.q_norm),
            ("k", layer.self_attn.k_norm),
            ("v", layer.self_attn.v_proj),
        ]:

            def hook(m, inputs, output, i=i, name=name):
                buf[i, name] = output.detach()

            handles.append(module.register_forward_hook(hook))
    try:
        calibration = []
        for task in ["chat", "code", "math"]:
            for prompt in pools(task, "tune", 8):
                ids = torch.tensor([prompt["token_ids"][:512]], device="cuda")
                pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
                with torch.inference_mode():
                    # No generation, no heldout examples: scales calibrated on prompt forwards.
                    model.model(ids, use_cache=False)
                    cos, sin = model.model.rotary_emb(model.model.embed_tokens(ids), pos)
                    for i in range(36):
                        q = buf[i, "q"].transpose(1, 2)
                        k = buf[i, "k"].transpose(1, 2)
                        q, k = apply_rotary_pos_emb(q, k, cos, sin)
                        for name, tensor in [("q", q), ("k", k), ("v", buf[i, "v"])]:
                            maxima[i][name] = max(
                                maxima[i][name], tensor.float().abs().max().item()
                            )
                calibration.append(
                    {
                        "id": prompt["id"],
                        "prompt_sha256": prompt["prompt_sha256"],
                        "n": ids.shape[1],
                    }
                )
                buf.clear()
    finally:
        for handle in handles:
            handle.remove()
    scales = {}
    for i, m in maxima.items():
        for name, value in m.items():
            scales[f"model.layers.{i}.self_attn.attn.{name}_scale"] = torch.tensor(
                max(value / 448.0, 1e-8), dtype=torch.float32
            )
    dest.mkdir(parents=True, exist_ok=False)
    base = MODELS / "Qwen3-8B"
    for p in base.iterdir():
        if p.is_file() and p.name not in ["config.json", "model.safetensors.index.json"]:
            target = dest / p.name
            if not target.exists():
                target.symlink_to(p)
    save_file(scales, str(dest / "kv_scales.safetensors"))
    config = json.loads((base / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "dense",
        "config_groups": {},
        "ignore": [],
        "kv_cache_scheme": {
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "symmetric": True,
            "dynamic": False,
        },
    }
    save(dest / "config.json", config)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    for key in scales:
        index["weight_map"][key] = "kv_scales.safetensors"
    index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + len(scales) * 4
    save(dest / "model.safetensors.index.json", index)
    save(
        marker,
        {
            "at": stamp(),
            "calibration_prompts": calibration,
            "maxima": maxima,
            "scale_method": "per-layer tensor absmax/448 after QK norm+RoPE; V projection; compressed-tensors Q/K/V scale placeholders only",
            "source": "Same original BF16 safetensors via symlinks; added scale-only shard; no weight quantization",
            "limitation": "Prompt-only calibration does not guarantee coverage of decode outliers.",
            "scale_sha256": hashlib.sha256(
                (dest / "kv_scales.safetensors").read_bytes()
            ).hexdigest(),
        },
    )
    print("KV_CALIBRATION_READY", dest, flush=True)


if __name__ == "__main__":
    main()
