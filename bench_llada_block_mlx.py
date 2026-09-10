from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


D_MODEL = 4096
MLP_HIDDEN = 12288
N_HEADS = 32
HEAD_DIM = D_MODEL // N_HEADS
ROPE_THETA = 500000.0
RMS_EPS = 1e-5

WEIGHT_NAMES = {
    "attn_norm": "model.transformer.blocks.0.attn_norm.weight",
    "ff_norm": "model.transformer.blocks.0.ff_norm.weight",
    "q_proj": "model.transformer.blocks.0.q_proj.weight",
    "k_proj": "model.transformer.blocks.0.k_proj.weight",
    "v_proj": "model.transformer.blocks.0.v_proj.weight",
    "attn_out": "model.transformer.blocks.0.attn_out.weight",
    "ff_proj": "model.transformer.blocks.0.ff_proj.weight",
    "up_proj": "model.transformer.blocks.0.up_proj.weight",
    "ff_out": "model.transformer.blocks.0.ff_out.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / "models" / "LLaDA-8B-Instruct",
    )
    parser.add_argument("--past-len", type=int, default=1200)
    parser.add_argument("--query-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def load_block0_weights(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(index_path)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]

    loaded: dict[str, torch.Tensor] = {}

    for short_name, full_name in WEIGHT_NAMES.items():
        if full_name not in weight_map:
            raise KeyError(f"Missing weight: {full_name}")

        shard_path = model_dir / weight_map[full_name]

        with safe_open(
            str(shard_path),
            framework="pt",
            device="cpu",
        ) as file:
            loaded[short_name] = file.get_tensor(full_name).contiguous()

        print(
            f"loaded {short_name:10s} "
            f"{tuple(loaded[short_name].shape)} "
            f"{loaded[short_name].dtype}"
        )

    return loaded


def make_shared_bf16(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, mx.array]:
    """Create numerically identical BF16 arrays for MPS and MLX."""

    cpu_bf16 = tensor.to(torch.bfloat16).contiguous()

    pt_value = cpu_bf16.to("mps")

    # NumPy currently receives float32 values that are exactly representable
    # as BF16, then MLX casts them back to BF16.
    np_value = cpu_bf16.float().numpy()
    mx_value = mx.array(np_value).astype(mx.bfloat16)

    return pt_value, mx_value


def make_rope_tables(
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(
        0,
        HEAD_DIM,
        2,
        dtype=np.float32,
    )

    inv_freq = 1.0 / (
        ROPE_THETA ** (indices / HEAD_DIM)
    )

    positions = np.arange(
        sequence_length,
        dtype=np.float32,
    )

    frequencies = np.outer(positions, inv_freq)
    duplicated = np.concatenate(
        [frequencies, frequencies],
        axis=-1,
    )

    return (
        np.sin(duplicated)[None, None, :, :],
        np.cos(duplicated)[None, None, :, :],
    )


def torch_rotate_half(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = x.shape

    split = x.reshape(
        batch,
        heads,
        seq,
        2,
        dim // 2,
    )

    first = split[..., 0, :]
    second = split[..., 1, :]

    return torch.cat((-second, first), dim=-1)


def mlx_rotate_half(x: mx.array) -> mx.array:
    batch, heads, seq, dim = x.shape

    split = x.reshape(
        batch,
        heads,
        seq,
        2,
        dim // 2,
    )

    first = split[..., 0, :]
    second = split[..., 1, :]

    return mx.concatenate((-second, first), axis=-1)


def torch_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    original_dtype = x.dtype

    normalized = x.float()
    variance = normalized.square().mean(
        dim=-1,
        keepdim=True,
    )

    normalized = normalized * torch.rsqrt(
        variance + RMS_EPS
    )

    normalized = normalized.to(original_dtype)

    return normalized * weight


def mlx_rms_norm(
    x: mx.array,
    weight: mx.array,
) -> mx.array:
    return mx.fast.rms_norm(
        x,
        weight,
        RMS_EPS,
    )


def percentile(
    values: list[float],
    q: float,
) -> float:
    return float(np.percentile(np.asarray(values), q))


def format_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    ref = reference.astype(np.float64).reshape(-1)
    cand = candidate.astype(np.float64).reshape(-1)

    difference = cand - ref

    denominator = (
        np.linalg.norm(ref)
        * np.linalg.norm(cand)
        + 1e-12
    )

    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(
            np.linalg.norm(difference)
            / (np.linalg.norm(ref) + 1e-12)
        ),
        "cosine": float(np.dot(ref, cand) / denominator),
    }


def main() -> None:
    args = parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable")

    if args.query_len <= 0:
        raise ValueError("query_len must be positive")

    if args.past_len < args.query_len:
        raise ValueError(
            "past_len must be >= query_len "
            "for the replacement-cache benchmark"
        )

    update_end = args.past_len
    update_start = update_end - args.query_len

    print("=" * 88)
    print("LLaDA block-0: PyTorch MPS vs MLX")
    print("=" * 88)
    print("model_dir :", args.model_dir)
    print("past_len  :", args.past_len)
    print("query_len :", args.query_len)
    print("update    :", update_start, "->", update_end)
    print("warmup    :", args.warmup)
    print("repeats   :", args.repeats)
    print("torch     :", torch.__version__)

    try:
        import mlx

        print("mlx       :", mlx.__version__)
    except Exception:
        pass

    mx.set_default_device(mx.gpu)

    cpu_weights = load_block0_weights(args.model_dir)

    pt_weights: dict[str, torch.Tensor] = {}
    mx_weights: dict[str, mx.array] = {}

    print()
    print("Moving block-0 weights to MPS and MLX...")

    for name, tensor in cpu_weights.items():
        pt_weights[name] = tensor.to(
            device="mps",
            dtype=torch.bfloat16,
        )

        mx_weights[name] = mx.array(
            tensor.float().numpy()
        ).astype(mx.bfloat16)

    mx.eval(mx_weights)
    mx.synchronize()
    torch.mps.synchronize()

    del cpu_weights
    gc.collect()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    input_cpu = torch.randn(
        (1, args.query_len, D_MODEL),
        generator=generator,
        dtype=torch.float32,
    ) * 0.02

    past_key_cpu = torch.randn(
        (1, N_HEADS, args.past_len, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ) * 0.02

    past_value_cpu = torch.randn(
        (1, N_HEADS, args.past_len, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ) * 0.02

    pt_input, mx_input = make_shared_bf16(input_cpu)
    pt_past_key, mx_past_key = make_shared_bf16(past_key_cpu)
    pt_past_value, mx_past_value = make_shared_bf16(
        past_value_cpu
    )

    sin_np, cos_np = make_rope_tables(args.past_len)

    pt_sin = torch.from_numpy(sin_np).to("mps")
    pt_cos = torch.from_numpy(cos_np).to("mps")

    mx_sin = mx.array(sin_np)
    mx_cos = mx.array(cos_np)

    mx.eval(
        mx_input,
        mx_past_key,
        mx_past_value,
        mx_sin,
        mx_cos,
    )
    mx.synchronize()
    torch.mps.synchronize()

    attention_scale = HEAD_DIM ** -0.5

    @torch.inference_mode()
    def torch_forward(
        x: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        normalized = torch_rms_norm(
            x,
            pt_weights["attn_norm"],
        )

        q = F.linear(normalized, pt_weights["q_proj"])
        new_k = F.linear(normalized, pt_weights["k_proj"])
        new_v = F.linear(normalized, pt_weights["v_proj"])

        batch, query_length, _ = q.shape

        q = q.reshape(
            batch,
            query_length,
            N_HEADS,
            HEAD_DIM,
        ).transpose(1, 2)

        new_k = new_k.reshape(
            batch,
            query_length,
            N_HEADS,
            HEAD_DIM,
        ).transpose(1, 2)

        new_v = new_v.reshape(
            batch,
            query_length,
            N_HEADS,
            HEAD_DIM,
        ).transpose(1, 2)

        # Exact DualCache replacement behavior:
        # cache stores unrotated K/V.
        full_k = past_key
        full_v = past_value

        full_k[
            :,
            :,
            update_start:update_end,
            :,
        ] = new_k

        full_v[
            :,
            :,
            update_start:update_end,
            :,
        ] = new_v

        q_float = q.float()
        k_float = full_k.float()

        q_sin = pt_sin[
            :,
            :,
            update_start:update_end,
            :,
        ]
        q_cos = pt_cos[
            :,
            :,
            update_start:update_end,
            :,
        ]

        q_rotated = (
            q_float * q_cos
            + torch_rotate_half(q_float) * q_sin
        ).to(q.dtype)

        k_rotated = (
            k_float * pt_cos
            + torch_rotate_half(k_float) * pt_sin
        ).to(full_k.dtype)

        attention = F.scaled_dot_product_attention(
            q_rotated,
            k_rotated,
            full_v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=attention_scale,
        )

        attention = attention.transpose(
            1,
            2,
        ).contiguous().reshape(
            batch,
            query_length,
            D_MODEL,
        )

        attention_output = F.linear(
            attention,
            pt_weights["attn_out"],
        )

        hidden = x + attention_output

        mlp_input = torch_rms_norm(
            hidden,
            pt_weights["ff_norm"],
        )

        gate = F.linear(
            mlp_input,
            pt_weights["ff_proj"],
        )

        up = F.linear(
            mlp_input,
            pt_weights["up_proj"],
        )

        activated = F.silu(gate) * up

        mlp_output = F.linear(
            activated,
            pt_weights["ff_out"],
        )

        output = hidden + mlp_output

        return output, attention_output, full_k, full_v

    def mlx_forward(
        x: mx.array,
        past_key: mx.array,
        past_value: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        normalized = mlx_rms_norm(
            x,
            mx_weights["attn_norm"],
        )

        q = normalized @ mx_weights["q_proj"].T
        new_k = normalized @ mx_weights["k_proj"].T
        new_v = normalized @ mx_weights["v_proj"].T

        batch, query_length, _ = q.shape

        q = q.reshape(
            batch,
            query_length,
            N_HEADS,
            HEAD_DIM,
        ).transpose(0, 2, 1, 3)

        new_k = new_k.reshape(
            batch,
            query_length,
            N_HEADS,
            HEAD_DIM,
        ).transpose(0, 2, 1, 3)

        new_v = new_v.reshape(
            batch,
            query_length,
            N_HEADS,
            HEAD_DIM,
        ).transpose(0, 2, 1, 3)

        # MLX indexed assignment maps to a GPU update operation.
        full_k = past_key
        full_v = past_value

        full_k[
            :,
            :,
            update_start:update_end,
            :,
        ] = new_k

        full_v[
            :,
            :,
            update_start:update_end,
            :,
        ] = new_v

        q_float = q.astype(mx.float32)
        k_float = full_k.astype(mx.float32)

        q_sin = mx_sin[
            :,
            :,
            update_start:update_end,
            :,
        ]
        q_cos = mx_cos[
            :,
            :,
            update_start:update_end,
            :,
        ]

        q_rotated = (
            q_float * q_cos
            + mlx_rotate_half(q_float) * q_sin
        ).astype(mx.bfloat16)

        k_rotated = (
            k_float * mx_cos
            + mlx_rotate_half(k_float) * mx_sin
        ).astype(mx.bfloat16)

        attention = mx.fast.scaled_dot_product_attention(
            q_rotated,
            k_rotated,
            full_v,
            scale=attention_scale,
            mask=None,
        )

        attention = attention.transpose(
            0,
            2,
            1,
            3,
        ).reshape(
            batch,
            query_length,
            D_MODEL,
        )

        attention_output = (
            attention @ mx_weights["attn_out"].T
        )

        hidden = x + attention_output

        mlp_input = mlx_rms_norm(
            hidden,
            mx_weights["ff_norm"],
        )

        gate = mlp_input @ mx_weights["ff_proj"].T
        up = mlp_input @ mx_weights["up_proj"].T

        activated = (
            gate * mx.sigmoid(gate)
        ) * up

        mlp_output = (
            activated @ mx_weights["ff_out"].T
        )

        output = hidden + mlp_output

        return output, attention_output, full_k, full_v

    print()
    print("=" * 88)
    print("Numerical alignment")
    print("=" * 88)

    pt_output, pt_attention, _, _ = torch_forward(
        pt_input,
        pt_past_key,
        pt_past_value,
    )
    torch.mps.synchronize()

    mx_output, mx_attention, _, _ = mlx_forward(
        mx_input,
        mx_past_key,
        mx_past_value,
    )
    mx.eval(mx_output, mx_attention)
    mx.synchronize()

    pt_output_np = (
        pt_output.float().cpu().numpy()
    )
    pt_attention_np = (
        pt_attention.float().cpu().numpy()
    )

    mx_output_np = np.array(
        mx_output.astype(mx.float32)
    )
    mx_attention_np = np.array(
        mx_attention.astype(mx.float32)
    )

    for label, reference, candidate in [
        (
            "attention_output",
            pt_attention_np,
            mx_attention_np,
        ),
        (
            "block_output",
            pt_output_np,
            mx_output_np,
        ),
    ]:
        metrics = format_metrics(reference, candidate)

        print(f"\n{label}")
        print(
            f"  max_abs    = {metrics['max_abs']:.8f}"
        )
        print(
            f"  mean_abs   = {metrics['mean_abs']:.8f}"
        )
        print(
            f"  relative_l2= {metrics['relative_l2']:.8f}"
        )
        print(
            f"  cosine     = {metrics['cosine']:.10f}"
        )

    def benchmark_torch() -> list[float]:
        for _ in range(args.warmup):
            result, _, _, _ = torch_forward(
                pt_input,
                pt_past_key,
                pt_past_value,
            )
            torch.mps.synchronize()
            del result

        timings = []

        for _ in range(args.repeats):
            start = time.perf_counter()

            result, _, _, _ = torch_forward(
                pt_input,
                pt_past_key,
                pt_past_value,
            )

            torch.mps.synchronize()

            timings.append(
                (time.perf_counter() - start) * 1000
            )

            del result

        return timings

    def benchmark_mlx(
        function: Callable,
    ) -> list[float]:
        for _ in range(args.warmup):
            result, _, _, _ = function(
                mx_input,
                mx_past_key,
                mx_past_value,
            )
            mx.eval(result)
            mx.synchronize()

        timings = []

        for _ in range(args.repeats):
            start = time.perf_counter()

            result, _, _, _ = function(
                mx_input,
                mx_past_key,
                mx_past_value,
            )

            mx.eval(result)
            mx.synchronize()

            timings.append(
                (time.perf_counter() - start) * 1000
            )

        return timings

    print()
    print("=" * 88)
    print("Performance")
    print("=" * 88)

    torch_times = benchmark_torch()
    mlx_eager_times = benchmark_mlx(mlx_forward)

    results = {
        "PyTorch MPS": torch_times,
        "MLX eager": mlx_eager_times,
    }

    try:
        mlx_compiled_forward = mx.compile(mlx_forward)

        # First invocation includes compilation and is excluded.
        compiled_times = benchmark_mlx(
            mlx_compiled_forward
        )

        results["MLX compiled"] = compiled_times
    except Exception as exc:
        print()
        print("MLX compile unavailable:")
        print(repr(exc))

    torch_mean = statistics.mean(torch_times)

    print()
    print(
        f"{'Backend':<18}"
        f"{'Mean(ms)':>12}"
        f"{'P50(ms)':>12}"
        f"{'P90(ms)':>12}"
        f"{'Speedup':>12}"
    )
    print("-" * 66)

    for name, values in results.items():
        mean_ms = statistics.mean(values)
        p50_ms = percentile(values, 50)
        p90_ms = percentile(values, 90)
        speedup = torch_mean / mean_ms

        print(
            f"{name:<18}"
            f"{mean_ms:>12.3f}"
            f"{p50_ms:>12.3f}"
            f"{p90_ms:>12.3f}"
            f"{speedup:>11.3f}x"
        )

    print()
    print("Reference full-model cached forward: ~173 ms")
    print(
        "One-block rough share at 32 layers: "
        "~5.4 ms, but do not extrapolate until measured."
    )


if __name__ == "__main__":
    main()
