from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


DEVICE = torch.device("mps")
DTYPE = torch.bfloat16

D_MODEL = 4096
MLP_HIDDEN = 12288

GEMM_SPECS = [
    ("q_proj", D_MODEL, D_MODEL),
    ("k_proj", D_MODEL, D_MODEL),
    ("v_proj", D_MODEL, D_MODEL),
    ("o_proj", D_MODEL, D_MODEL),
    ("gate_proj", D_MODEL, MLP_HIDDEN),
    ("up_proj", D_MODEL, MLP_HIDDEN),
    ("down_proj", MLP_HIDDEN, D_MODEL),
]


def synchronize() -> None:
    torch.mps.synchronize()


def clear_mps() -> None:
    gc.collect()
    torch.mps.empty_cache()
    synchronize()


def allocate_tensor(
    shape: tuple[int, ...],
) -> torch.Tensor:
    tensor = torch.randn(
        shape,
        device=DEVICE,
        dtype=torch.float32,
    )
    tensor = tensor.to(DTYPE)
    synchronize()
    return tensor


def timed(
    callback: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> tuple[float, list[float]]:
    for _ in range(warmups):
        output = callback()
        synchronize()
        del output

    values: list[float] = []

    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()

        output = callback()

        synchronize()
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        values.append(elapsed_ms)
        del output

    return statistics.median(values), values


def load_p1_layer_times(
    path: Path | None,
) -> dict[int, float]:
    if path is None or not path.exists():
        return {}

    result: dict[int, float] = {}

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as file:
        for row in csv.DictReader(file):
            if row.get("kind") != "block_stack":
                continue

            if int(row["batch_size"]) != 1:
                continue

            if int(row["layers"]) != 32:
                continue

            query_len = int(row["query_len"])
            median_ms = float(row["median_ms"])

            result[query_len] = median_ms / 32.0

    return result


def tflops(
    *,
    m: int,
    k: int,
    n: int,
    milliseconds: float,
) -> float:
    operations = 2.0 * m * k * n

    return (
        operations
        / (milliseconds / 1000.0)
        / 1e12
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--m-values",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256, 1331],
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--p1-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    torch.set_grad_enabled(False)

    p1_layer_ms = load_p1_layer_times(
        args.p1_csv
    )

    if args.output_dir is None:
        run_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        output_dir = (
            Path.home()
            / "Fast-dLLM"
            / "results"
            / f"p2_bare_gemm_{run_id}"
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Allocating real-shape BF16 weights...")

    weights: dict[str, torch.Tensor] = {}

    for name, k, n in GEMM_SPECS:
        # F.linear expects weight [out_features, in_features].
        weights[name] = allocate_tensor((n, k))

        print(
            f"{name}: "
            f"weight={tuple(weights[name].shape)}, "
            f"{weights[name].numel() * 2 / 1024**2:.1f} MiB"
        )

    total_weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in weights.values()
    )

    print(
        "Total layer weights:",
        f"{total_weight_bytes / 1024**3:.3f} GiB",
    )

    individual_rows: list[dict] = []
    summary_rows: list[dict] = []

    for m in args.m_values:
        print()
        print("=" * 88)
        print(f"M={m}")
        print("=" * 88)

        x_model = allocate_tensor(
            (m, D_MODEL)
        )
        x_mlp = allocate_tensor(
            (m, MLP_HIDDEN)
        )

        callbacks: dict[str, Callable] = {
            "q_proj": lambda: F.linear(
                x_model,
                weights["q_proj"],
            ),
            "k_proj": lambda: F.linear(
                x_model,
                weights["k_proj"],
            ),
            "v_proj": lambda: F.linear(
                x_model,
                weights["v_proj"],
            ),
            "o_proj": lambda: F.linear(
                x_model,
                weights["o_proj"],
            ),
            "gate_proj": lambda: F.linear(
                x_model,
                weights["gate_proj"],
            ),
            "up_proj": lambda: F.linear(
                x_model,
                weights["up_proj"],
            ),
            "down_proj": lambda: F.linear(
                x_mlp,
                weights["down_proj"],
            ),
        }

        medians: dict[str, float] = {}

        for name, k, n in GEMM_SPECS:
            median_ms, raw_ms = timed(
                callbacks[name],
                warmups=args.warmups,
                repeats=args.repeats,
            )

            medians[name] = median_ms

            achieved_tflops = tflops(
                m=m,
                k=k,
                n=n,
                milliseconds=median_ms,
            )

            individual_rows.append({
                "m": m,
                "name": name,
                "k": k,
                "n": n,
                "median_ms": median_ms,
                "min_ms": min(raw_ms),
                "max_ms": max(raw_ms),
                "mean_ms": statistics.mean(raw_ms),
                "stdev_ms": (
                    statistics.stdev(raw_ms)
                    if len(raw_ms) > 1
                    else 0.0
                ),
                "tflops": achieved_tflops,
                "raw_ms": json.dumps(raw_ms),
            })

            print(
                f"{name:<12} "
                f"{median_ms:>8.3f} ms  "
                f"{achieved_tflops:>7.2f} TFLOPS"
            )

        def serial_seven_gemms():
            return (
                F.linear(
                    x_model,
                    weights["q_proj"],
                ),
                F.linear(
                    x_model,
                    weights["k_proj"],
                ),
                F.linear(
                    x_model,
                    weights["v_proj"],
                ),
                F.linear(
                    x_model,
                    weights["o_proj"],
                ),
                F.linear(
                    x_model,
                    weights["gate_proj"],
                ),
                F.linear(
                    x_model,
                    weights["up_proj"],
                ),
                F.linear(
                    x_mlp,
                    weights["down_proj"],
                ),
            )

        serial_ms, serial_raw = timed(
            serial_seven_gemms,
            warmups=args.warmups,
            repeats=args.repeats,
        )

        sum_individual_ms = sum(
            medians.values()
        )

        total_flops = sum(
            2.0 * m * k * n
            for _, k, n in GEMM_SPECS
        )

        serial_tflops = (
            total_flops
            / (serial_ms / 1000.0)
            / 1e12
        )

        effective_weight_gbps = (
            total_weight_bytes / 1e9
        ) / (serial_ms / 1000.0)

        p1_ms = p1_layer_ms.get(m)

        if p1_ms is None:
            gemm_share_pct = None
            non_gemm_ms = None
        else:
            gemm_share_pct = (
                serial_ms / p1_ms * 100.0
            )
            non_gemm_ms = p1_ms - serial_ms

        summary_rows.append({
            "m": m,
            "serial_7_gemm_ms": serial_ms,
            "sum_individual_ms": sum_individual_ms,
            "serial_over_sum": (
                serial_ms / sum_individual_ms
            ),
            "serial_tflops": serial_tflops,
            "effective_weight_gbps": (
                effective_weight_gbps
            ),
            "p1_block_layer_ms": (
                p1_ms if p1_ms is not None else ""
            ),
            "gemm_share_of_p1_pct": (
                gemm_share_pct
                if gemm_share_pct is not None
                else ""
            ),
            "estimated_non_gemm_ms": (
                non_gemm_ms
                if non_gemm_ms is not None
                else ""
            ),
            "serial_raw_ms": json.dumps(
                serial_raw
            ),
        })

        print()
        print(
            f"7 GEMMs serial:       {serial_ms:.3f} ms"
        )
        print(
            f"Sum of medians:       "
            f"{sum_individual_ms:.3f} ms"
        )
        print(
            f"Serial TFLOPS:        "
            f"{serial_tflops:.2f}"
        )
        print(
            f"Effective weight BW:  "
            f"{effective_weight_gbps:.2f} GB/s"
        )

        if p1_ms is not None:
            print(
                f"P1 block/layer:       {p1_ms:.3f} ms"
            )
            print(
                f"GEMM share of P1:     "
                f"{gemm_share_pct:.2f}%"
            )
            print(
                f"Estimated non-GEMM:   "
                f"{non_gemm_ms:.3f} ms"
            )

        del x_model
        del x_mlp
        clear_mps()

        time.sleep(0.5)

    individual_csv = (
        output_dir / "p2_individual_gemms.csv"
    )

    with individual_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                individual_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(individual_rows)

    summary_csv = (
        output_dir / "p2_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "=" * 116,
        "P2 Bare GEMM Calibration — Apple M4 Max / MPS BF16",
        "=" * 116,
        (
            f"{'M':>7}"
            f"{'7 GEMMs ms':>14}"
            f"{'Sum ms':>12}"
            f"{'TFLOPS':>12}"
            f"{'Weight GB/s':>15}"
            f"{'P1 layer ms':>14}"
            f"{'GEMM share':>14}"
            f"{'Non-GEMM ms':>15}"
        ),
        "-" * 116,
    ]

    for row in summary_rows:
        p1_value = row["p1_block_layer_ms"]
        share_value = row["gemm_share_of_p1_pct"]
        non_gemm_value = row["estimated_non_gemm_ms"]

        lines.append(
            f"{row['m']:>7}"
            f"{row['serial_7_gemm_ms']:>14.3f}"
            f"{row['sum_individual_ms']:>12.3f}"
            f"{row['serial_tflops']:>12.2f}"
            f"{row['effective_weight_gbps']:>15.2f}"
            + (
                f"{float(p1_value):>14.3f}"
                if p1_value != ""
                else f"{'—':>14}"
            )
            + (
                f"{float(share_value):>13.2f}%"
                if share_value != ""
                else f"{'—':>14}"
            )
            + (
                f"{float(non_gemm_value):>15.3f}"
                if non_gemm_value != ""
                else f"{'—':>15}"
            )
        )

    lines.extend([
        "",
        "Decision thresholds:",
        "- GEMM share >= 80%: GEMM/weight-flow dominated; prioritize quantization.",
        "- GEMM share <= 60%: non-GEMM/dispatch significant; run MLX Block spike.",
        "- GEMM share 60-80%: mixed bottleneck; run a targeted Metal trace.",
        "",
        "Notes:",
        "- P1 comparison exists only for M=32,64,128.",
        "- Effective weight GB/s assumes each weight is read once.",
        "- Bare GEMM excludes RoPE, RMSNorm, activation, SDPA, cache update and residual ops.",
    ])

    summary_text = "\n".join(lines)

    summary_path = (
        output_dir / "p2_summary.txt"
    )
    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)
    print()
    print("Individual CSV:", individual_csv)
    print("Summary CSV:", summary_csv)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
