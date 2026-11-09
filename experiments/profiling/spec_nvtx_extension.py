"""Diagnostic NVTX scopes loaded only by --worker-extension-cls.

The original callables and inputs are preserved. This file is never loaded by
the formal benchmark commands. CPU scopes enqueue GPU work asynchronously;
use CUDA launch correlation / NVTX GPU projection, not CPU scope durations.
"""

from functools import wraps

import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def mark(function, label):
    @wraps(function)
    def annotated(*args, **kwargs):
        with torch.cuda.nvtx.range(label):
            return function(*args, **kwargs)

    return annotated


if not getattr(GPUModelRunner, "_spec_nsight_annotations", False):
    original_load = GPUModelRunner.load_model

    @wraps(original_load)
    def load_with_draft_scope(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if self.speculator is not None:
            self.speculator.propose = mark(self.speculator.propose, "spec/draft_propose")
        return result

    GPUModelRunner.load_model = load_with_draft_scope
    GPUModelRunner.execute_model = mark(GPUModelRunner.execute_model, "spec/target_execute")
    GPUModelRunner.sample = mark(GPUModelRunner.sample, "spec/target_lm_head_and_sampling")
    GPUModelRunner._spec_nsight_annotations = True


class NsightAnnotations:
    """Worker mixin marker. Importing this module installs diagnostic scopes."""

    pass
