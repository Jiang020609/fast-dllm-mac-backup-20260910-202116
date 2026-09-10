from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from transformers import AutoConfig

# 导入即应用 Rotated-K v2b attention patch，
# 但不会启动 lm-eval。
import rotated_k_v2b_runtime_patch  # noqa: F401

from model.modeling_llada import LLaDAModelLM
from rotated_k_position_context import position_forward_scope


DEVICE = torch.device("mps")
DTYPE = torch.bfloat16

DEFAULT_PAST_LEN = 1331
THEORETICAL_MEMORY_GBPS = 546.0


def synchronize() -> None:
    torch.mps.synchronize()


def clear_mps() -> None:
    gc.collect()
    torch.mps.empty_cache()
    synchronize()


def tensor_bytes(parameters) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameters
    )


def median_latency_ms(
    callback: Callable[[], Any],
    warmups: int,
    repeats: int,
) -> tuple[float, list[float]]:
    # 每个新 shape 的第一次执行可能包含 MPSGraph 编译；
    # warmup 不计入结果。
    for _ in range(warmups):
        output = callback()
        synchronize()
        del output

    times_ms: list[float] = []

    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()

        output = callback()

        synchronize()
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        times_ms.append(elapsed_ms)
        del output

    return statistics.median(times_ms), times_ms


def make_past_key_values(
    *,
    batch_size: int,
    layers: int,
    kv_heads: int,
    past_len: int,
    head_dim: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    shape = (
        batch_size,
        kv_heads,
        past_len,
        head_dim,
    )

    result = []

    for _ in range(layers):
        key = torch.zeros(
            shape,
            dtype=DTYPE,
            device=DEVICE,
        )
        value = torch.zeros_like(key)
        result.append((key, value))

    synchronize()
    return result


def get_blocks(model: LLaDAModelLM):
    core = getattr(model, "model", None)

    if core is None:
        raise RuntimeError(
            "LLaDAModelLM 没有 .model 子模块"
        )

    transformer = getattr(core, "transformer", None)

    if transformer is None:
        raise RuntimeError(
            "内部模型没有 transformer"
        )

    try:
        blocks = transformer["blocks"]
    except Exception as error:
        keys = (
            list(transformer.keys())
            if hasattr(transformer, "keys")
            else []
        )
        raise RuntimeError(
            "没有找到 transformer['blocks']; "
            f"available={keys}"
        ) from error

    return core, list(blocks)


def run_block_stack_point(
    *,
    blocks,
    batch_size: int,
    query_len: int,
    layers: int,
    past_len: int,
    d_model: int,
    kv_heads: int,
    head_dim: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    clear_mps()

    hidden = torch.randn(
        batch_size,
        query_len,
        d_model,
        dtype=DTYPE,
        device=DEVICE,
    )

    past_key_values = make_past_key_values(
        batch_size=batch_size,
        layers=layers,
        kv_heads=kv_heads,
        past_len=past_len,
        head_dim=head_dim,
    )

    block_start = past_len - query_len
    block_end = past_len

    if block_start < 0:
        raise ValueError(
            "query_len must not exceed past_len"
        )

    def call_stack():
        x = hidden

        with position_forward_scope(
            block_start=block_start,
            block_end=block_end,
        ):
            for layer_index in range(layers):
                x, present = blocks[layer_index](
                    x,
                    attention_bias=None,
                    layer_past=past_key_values[layer_index],
                    use_cache=True,
                )

                if present is not None:
                    past_key_values[layer_index] = present

        return x

    median_ms, raw_ms = median_latency_ms(
        call_stack,
        warmups=warmups,
        repeats=repeats,
    )

    parameter_bytes = sum(
        tensor_bytes(blocks[index].parameters())
        for index in range(layers)
    )

    effective_weight_gbps = (
        parameter_bytes / 1e9
    ) / (median_ms / 1000.0)

    allocated_gib = (
        torch.mps.current_allocated_memory()
        / (1024 ** 3)
    )

    row = {
        "kind": "block_stack",
        "batch_size": batch_size,
        "query_len": query_len,
        "layers": layers,
        "past_len": past_len,
        "median_ms": median_ms,
        "min_ms": min(raw_ms),
        "max_ms": max(raw_ms),
        "mean_ms": statistics.mean(raw_ms),
        "stdev_ms": (
            statistics.stdev(raw_ms)
            if len(raw_ms) > 1
            else 0.0
        ),
        "tokens_per_second": (
            batch_size
            * query_len
            / (median_ms / 1000.0)
        ),
        "parameter_gib": (
            parameter_bytes / (1024 ** 3)
        ),
        "effective_weight_gbps": (
            effective_weight_gbps
        ),
        "theoretical_bandwidth_pct": (
            effective_weight_gbps
            / THEORETICAL_MEMORY_GBPS
            * 100.0
        ),
        "mps_allocated_gib": allocated_gib,
        "raw_ms": json.dumps(raw_ms),
        "error": "",
    }

    del hidden
    del past_key_values
    clear_mps()

    return row


def run_model_endpoint_point(
    *,
    model: LLaDAModelLM,
    batch_size: int,
    query_len: int,
    layers: int,
    past_len: int,
    kv_heads: int,
    head_dim: int,
    mask_token_id: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    clear_mps()

    input_ids = torch.full(
        (batch_size, query_len),
        int(mask_token_id),
        dtype=torch.long,
        device=DEVICE,
    )

    past_key_values = make_past_key_values(
        batch_size=batch_size,
        layers=layers,
        kv_heads=kv_heads,
        past_len=past_len,
        head_dim=head_dim,
    )

    block_start = past_len - query_len
    block_end = past_len

    def call_model():
        with position_forward_scope(
            block_start=block_start,
            block_end=block_end,
        ):
            return model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

    median_ms, raw_ms = median_latency_ms(
        call_model,
        warmups=warmups,
        repeats=repeats,
    )

    row = {
        "kind": "full_cached_model",
        "batch_size": batch_size,
        "query_len": query_len,
        "layers": layers,
        "past_len": past_len,
        "median_ms": median_ms,
        "min_ms": min(raw_ms),
        "max_ms": max(raw_ms),
        "mean_ms": statistics.mean(raw_ms),
        "stdev_ms": (
            statistics.stdev(raw_ms)
            if len(raw_ms) > 1
            else 0.0
        ),
        "tokens_per_second": (
            batch_size
            * query_len
            / (median_ms / 1000.0)
        ),
        "parameter_gib": "",
        "effective_weight_gbps": "",
        "theoretical_bandwidth_pct": "",
        "mps_allocated_gib": (
            torch.mps.current_allocated_memory()
            / (1024 ** 3)
        ),
        "raw_ms": json.dumps(raw_ms),
        "error": "",
    }

    del input_ids
    del past_key_values
    clear_mps()

    return row


def failed_row(
    kind: str,
    batch_size: int,
    query_len: int,
    layers: int,
    past_len: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "batch_size": batch_size,
        "query_len": query_len,
        "layers": layers,
        "past_len": past_len,
        "median_ms": "",
        "min_ms": "",
        "max_ms": "",
        "mean_ms": "",
        "stdev_ms": "",
        "tokens_per_second": "",
        "parameter_gib": "",
        "effective_weight_gbps": "",
        "theoretical_bandwidth_pct": "",
        "mps_allocated_gib": "",
        "raw_ms": "",
        "error": (
            f"{type(error).__name__}: {error}"
        ),
    }


def lookup(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    query_len: int,
    layers: int,
) -> float | None:
    for row in rows:
        if (
            row["kind"] == "block_stack"
            and row["batch_size"] == batch_size
            and row["query_len"] == query_len
            and row["layers"] == layers
            and row["median_ms"] != ""
        ):
            return float(row["median_ms"])

    return None


def build_summary(
    rows: list[dict[str, Any]],
) -> str:
    lines = []

    lines.extend([
        "=" * 116,
        "P1 Cached-Forward Scaling — Rotated-K v2b exact path",
        "=" * 116,
        (
            f"{'Kind':<20}"
            f"{'B':>5}"
            f"{'Q':>7}"
            f"{'Layers':>9}"
            f"{'Median(ms)':>14}"
            f"{'Tok/s':>14}"
            f"{'Weight GB/s':>16}"
            f"{'BW %':>10}"
            f"{'MPS GiB':>11}"
        ),
        "-" * 116,
    ])

    for row in rows:
        if row["median_ms"] == "":
            lines.append(
                f"{row['kind']:<20}"
                f"{row['batch_size']:>5}"
                f"{row['query_len']:>7}"
                f"{row['layers']:>9}"
                f"{'FAILED':>14}  {row['error']}"
            )
            continue

        weight_gbps = row["effective_weight_gbps"]
        bw_pct = row["theoretical_bandwidth_pct"]

        lines.append(
            f"{row['kind']:<20}"
            f"{row['batch_size']:>5}"
            f"{row['query_len']:>7}"
            f"{row['layers']:>9}"
            f"{float(row['median_ms']):>14.3f}"
            f"{float(row['tokens_per_second']):>14.2f}"
            f"{float(weight_gbps):>16.2f}"
            if weight_gbps != ""
            else (
                f"{row['kind']:<20}"
                f"{row['batch_size']:>5}"
                f"{row['query_len']:>7}"
                f"{row['layers']:>9}"
                f"{float(row['median_ms']):>14.3f}"
                f"{float(row['tokens_per_second']):>14.2f}"
                f"{'—':>16}"
            )
            + (
                f"{float(bw_pct):>9.2f}%"
                if bw_pct != ""
                else f"{'—':>10}"
            )
            + f"{float(row['mps_allocated_gib']):>11.2f}"
        )

    layer_rows = [
        row
        for row in rows
        if (
            row["kind"] == "block_stack"
            and row["batch_size"] == 1
            and row["query_len"] == 32
            and row["layers"] in {8, 16, 32}
            and row["median_ms"] != ""
        )
    ]

    lines.append("")
    lines.append("=" * 116)
    lines.append("Layer scaling fit")
    lines.append("=" * 116)

    if len(layer_rows) == 3:
        layer_rows.sort(
            key=lambda row: row["layers"]
        )

        x = np.array(
            [row["layers"] for row in layer_rows],
            dtype=np.float64,
        )
        y = np.array(
            [row["median_ms"] for row in layer_rows],
            dtype=np.float64,
        )

        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept

        ss_res = float(
            np.sum((y - predicted) ** 2)
        )
        ss_tot = float(
            np.sum((y - np.mean(y)) ** 2)
        )

        r_squared = (
            1.0 - ss_res / ss_tot
            if ss_tot > 0
            else 1.0
        )

        predicted_32 = (
            intercept + slope * 32
        )

        intercept_share = (
            intercept / predicted_32 * 100.0
            if predicted_32 > 0
            else float("nan")
        )

        lines.extend([
            (
                "latency_ms = "
                f"{intercept:.4f} + "
                f"{slope:.4f} × layers"
            ),
            f"R² = {r_squared:.6f}",
            f"Fixed intercept = {intercept:.4f} ms",
            f"Per-layer cost = {slope:.4f} ms",
            (
                "Intercept share at 32 layers = "
                f"{intercept_share:.2f}%"
            ),
        ])
    else:
        lines.append(
            "Layer fit unavailable: missing valid 8/16/32 points."
        )

    lines.append("")
    lines.append("=" * 116)
    lines.append("Key scaling ratios")
    lines.append("=" * 116)

    baseline = lookup(
        rows,
        batch_size=1,
        query_len=32,
        layers=32,
    )

    if baseline is not None:
        for batch_size in (2, 4, 8):
            value = lookup(
                rows,
                batch_size=batch_size,
                query_len=32,
                layers=32,
            )

            if value is not None:
                lines.append(
                    f"B={batch_size} / B=1 latency ratio: "
                    f"{value / baseline:.3f}x"
                )

        for query_len in (64, 128):
            value = lookup(
                rows,
                batch_size=1,
                query_len=query_len,
                layers=32,
            )

            if value is not None:
                lines.append(
                    f"Q={query_len} / Q=32 latency ratio: "
                    f"{value / baseline:.3f}x"
                )

    endpoint = next(
        (
            row
            for row in rows
            if (
                row["kind"] == "full_cached_model"
                and row["median_ms"] != ""
            )
        ),
        None,
    )

    if baseline is not None and endpoint is not None:
        endpoint_ms = float(endpoint["median_ms"])
        lines.extend([
            (
                "Full cached model / 32-block stack ratio: "
                f"{endpoint_ms / baseline:.3f}x"
            ),
            (
                "Non-block endpoint overhead: "
                f"{endpoint_ms - baseline:.3f} ms"
            ),
        ])

    lines.extend([
        "",
        "Interpretation guardrails:",
        (
            "- Weight GB/s assumes each block parameter is read once; "
            "it is an effective lower-bound diagnostic, not a hardware counter."
        ),
        (
            "- Layer-fit intercept includes Python/framework and stack-level "
            "fixed costs, but excludes embedding/final LM-head costs."
        ),
        (
            "- The full_cached_model point quantifies embedding/final norm/"
            "LM-head and model-wrapper overhead at B=1,Q=32."
        ),
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        default=str(
            Path.home()
            / "models"
            / "LLaDA-8B-Instruct"
        ),
    )
    parser.add_argument(
        "--past-len",
        type=int,
        default=DEFAULT_PAST_LEN,
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    torch.set_grad_enabled(False)

    model_path = Path(args.model_path)

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    # LLaDA checkpoint compatibility:
    # some Hugging Face configs expose max_sequence_length but omit
    # the training-time alias expected by ModelConfig conversion.
    if not hasattr(config, "train_max_sequence_length"):
        config.train_max_sequence_length = int(
            config.max_sequence_length
        )

    print(
        "Config compatibility:",
        {
            "max_sequence_length": config.max_sequence_length,
            "train_max_sequence_length": (
                config.train_max_sequence_length
            ),
        },
    )

    print("Loading model...")

    model = LLaDAModelLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=DTYPE,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = model.to(DEVICE)
    model.eval()
    synchronize()

    core, blocks = get_blocks(model)

    n_layers = int(config.n_layers)
    n_heads = int(config.n_heads)

    kv_heads = int(
        getattr(
            config,
            "effective_n_kv_heads",
            getattr(config, "n_kv_heads", n_heads)
            or n_heads,
        )
    )

    d_model = int(config.d_model)
    head_dim = d_model // n_heads

    mask_token_id = int(
        getattr(config, "mask_token_id", 0)
        or 0
    )

    print(
        "Model:",
        {
            "layers": n_layers,
            "d_model": d_model,
            "heads": n_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "mask_token_id": mask_token_id,
        },
    )

    if len(blocks) < 32:
        raise RuntimeError(
            f"Expected at least 32 blocks, got {len(blocks)}"
        )

    if args.output_dir is None:
        run_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        output_dir = (
            Path.home()
            / "Fast-dLLM"
            / "results"
            / f"p1_cached_scaling_{run_id}"
        )
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.smoke:
        points = [
            (1, 32, 8),
        ]
        warmups = 1
        repeats = 2
    else:
        points = [
            # Layer scaling
            (1, 32, 8),
            (1, 32, 16),
            (1, 32, 32),

            # Batch scaling
            (2, 32, 32),
            (4, 32, 32),
            (8, 32, 32),

            # Query-length scaling
            (1, 64, 32),
            (1, 128, 32),
        ]
        warmups = args.warmups
        repeats = args.repeats

    rows: list[dict[str, Any]] = []

    seen = set()

    for batch_size, query_len, layers in points:
        key = (
            batch_size,
            query_len,
            layers,
        )

        if key in seen:
            continue

        seen.add(key)

        print()
        print("=" * 80)
        print(
            f"Block stack: "
            f"B={batch_size}, "
            f"Q={query_len}, "
            f"L={layers}"
        )
        print("=" * 80)

        try:
            row = run_block_stack_point(
                blocks=blocks,
                batch_size=batch_size,
                query_len=query_len,
                layers=layers,
                past_len=args.past_len,
                d_model=d_model,
                kv_heads=kv_heads,
                head_dim=head_dim,
                warmups=warmups,
                repeats=repeats,
            )

            print(
                f"median={row['median_ms']:.3f} ms, "
                f"effective BW="
                f"{row['effective_weight_gbps']:.2f} GB/s"
            )
        except Exception as error:
            print(
                "FAILED:",
                type(error).__name__,
                error,
            )

            row = failed_row(
                "block_stack",
                batch_size,
                query_len,
                layers,
                args.past_len,
                error,
            )

            clear_mps()

        rows.append(row)

    # 完整模型 endpoint 只测真实 baseline shape。
    print()
    print("=" * 80)
    print("Full cached model endpoint: B=1,Q=32,L=32")
    print("=" * 80)

    try:
        endpoint_row = run_model_endpoint_point(
            model=model,
            batch_size=1,
            query_len=32,
            layers=n_layers,
            past_len=args.past_len,
            kv_heads=kv_heads,
            head_dim=head_dim,
            mask_token_id=mask_token_id,
            warmups=warmups,
            repeats=repeats,
        )

        print(
            f"median={endpoint_row['median_ms']:.3f} ms"
        )
    except Exception as error:
        print(
            "FAILED:",
            type(error).__name__,
            error,
        )

        endpoint_row = failed_row(
            "full_cached_model",
            1,
            32,
            n_layers,
            args.past_len,
            error,
        )

    rows.append(endpoint_row)

    csv_path = output_dir / "p1_results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows)

    summary_path = output_dir / "p1_summary.txt"
    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print("CSV:", csv_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
