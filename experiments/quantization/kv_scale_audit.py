"""Smoke-only worker instrumentation; verify loaded Q/K/V scales exactly."""

from functools import wraps
import json
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from paths import MODELS, KV_ROOT

original_load = GPUModelRunner.load_model


@wraps(original_load)
def audited_load(self, *args, **kwargs):
    result = original_load(self, *args, **kwargs)
    from safetensors.torch import load_file

    scales = load_file(str(MODELS / "Qwen3-8B-KVcalibrated/kv_scales.safetensors"))
    rows = []
    for name, module in self.model.named_modules():
        for kind in ["q", "k", "v"]:
            key = f"{name}.{kind}_scale"
            if key not in scales:
                continue
            value = getattr(module, f"_{kind}_scale").item()
            expected = scales[key].item()
            rows.append(
                {"name": key, "loaded": value, "expected": expected, "matches": value == expected}
            )
    payload = {
        "rows": rows,
        "count": len(rows),
        "all_match": len(rows) == 108 and all(r["matches"] for r in rows),
    }
    dest = KV_ROOT / "calibration/runtime_scale_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["all_match"]:
        raise RuntimeError("Loaded KV scales failed exact sidecar comparison")
    return result


GPUModelRunner.load_model = audited_load


class ScaleAudit:
    pass
