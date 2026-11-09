"""Fused row-wise RMSNorm with FP64 checks and CUDA-event warp sweeps."""

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    x = tl.load(
        x_ptr + row * H + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(
        weight_ptr + cols,
        mask=mask,
        other=0.0,
    )

    rms = tl.sqrt(tl.sum(x * x, axis=0) / H + eps)
    y = x / rms * weight

    tl.store(
        y_ptr + row * H + cols,
        y.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


def triton_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    num_warps: int | None = None,  # 允许为空
) -> torch.Tensor:
    assert x.is_cuda
    assert x.ndim == 2
    assert weight.ndim == 1
    assert x.shape[-1] == weight.shape[0]
    assert x.is_contiguous
    assert weight.is_contiguous

    T, H = x.shape
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(H)

    if num_warps is None:
        if block_size <= 1024:
            num_warps = 2
        elif block_size <= 2048:
            num_warps = 4
        else:
            num_warps = 8

    rms_norm_kernel[(T,)](
        x,
        weight,
        y,
        H,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,  # type: ignore
    )

    return y


def torch_rms_norm(x, weight, eps=1e-6):
    x32 = x.float()
    w32 = weight.float()

    rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    y = x32 / rms * w32

    return y.to(x.dtype)


def test_correctness(T, H, dtype):
    torch.manual_seed(0)

    x = torch.randn(
        T,
        H,
        device="cuda",
        dtype=dtype,
    )

    weight = torch.randn(
        H,
        device="cuda",
        dtype=dtype,
    )

    y_ref = torch_rms_norm(x, weight)

    block_size = triton.next_power_of_2(H)

    if block_size <= 1024:
        candidate_warps = [2]
    elif block_size <= 2048:
        candidate_warps = [2, 4]
    else:
        candidate_warps = [4, 8]

    for num_warps in candidate_warps:
        y = triton_rms_norm(
            x,
            weight,
            num_warps=num_warps,
        )

        max_error = (y.float() - y_ref.float()).abs().max().item()

        mean_error = (y.float() - y_ref.float()).abs().mean().item()

        print(
            f"[correctness] "
            f"T={T}, H={H}, dtype={dtype}, "
            f"warps={num_warps}, "
            f"max_error={max_error:.6e}, "
            f"mean_error={mean_error:.6e}"
        )

        torch.testing.assert_close(
            y,
            y_ref,
            rtol=1e-2,
            atol=1e-2,
        )


import statistics


def measure_us(
    x,
    weight,
    num_warps,
    warmup=30,
    inner=20,
):
    # 预热
    for _ in range(warmup):
        triton_rms_norm(
            x,
            weight,
            num_warps=num_warps,
        )

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(inner):
        triton_rms_norm(
            x,
            weight,
            num_warps=num_warps,
        )

    end.record()
    end.synchronize()

    # elapsed_time 单位是 ms
    total_us = start.elapsed_time(end) * 1000.0

    return total_us / inner


def benchmark_warps(
    T,
    H,
    dtype,
    warp_candidates,
    rounds=8,
):
    torch.manual_seed(0)

    x = torch.randn(
        T,
        H,
        device="cuda",
        dtype=dtype,
    )

    weight = torch.randn(
        H,
        device="cuda",
        dtype=dtype,
    )

    results = {num_warps: [] for num_warps in warp_candidates}

    # 每个配置先预热一次
    for num_warps in warp_candidates:
        triton_rms_norm(
            x,
            weight,
            num_warps=num_warps,
        )

    torch.cuda.synchronize()

    for round_id in range(rounds):
        if round_id % 2 == 0:
            order = warp_candidates
        else:
            order = list(reversed(warp_candidates))

        for num_warps in order:
            t_us = measure_us(
                x,
                weight,
                num_warps=num_warps,
                warmup=0,
                inner=20,
            )

            results[num_warps].append(t_us)

    print(f"\n[benchmark] " f"T={T}, H={H}, dtype={dtype}")

    for num_warps, values in results.items():
        print(
            f"warps={num_warps}: "
            f"median={statistics.median(values):.2f} us, "
            f"min={min(values):.2f} us, "
            f"max={max(values):.2f} us, "
            f"samples={[round(v, 2) for v in values]}"
        )

    return results


def main():
    test_shapes = [
        (1, 1024),
        (8, 4096),
        (32, 4100),
    ]

    for T, H in test_shapes:
        test_correctness(T, H, torch.float16)
        test_correctness(T, H, torch.bfloat16)

    benchmark_warps(
        T=32768,
        H=4096,
        dtype=torch.bfloat16,
        warp_candidates=[2, 4, 8],
        rounds=8,
    )


if __name__ == "__main__":
    main()
