from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import rotated_k_v2b_runtime_patch  # noqa: F401

from model.modeling_llada import LLaDAModelLM


DEVICE = torch.device("mps")
DTYPE = torch.bfloat16


def sync() -> None:
    torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-len", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline-ms", type=float, default=41.0)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    torch.set_grad_enabled(False)

    model_path = (
        Path.home()
        / "models"
        / "LLaDA-8B-Instruct"
    )

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    if not hasattr(config, "train_max_sequence_length"):
        config.train_max_sequence_length = int(
            config.max_sequence_length
        )

    print("Loading model...", flush=True)

    model = LLaDAModelLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=DTYPE,
        local_files_only=True,
    ).to(DEVICE)

    model.eval()
    sync()

    modules = [
        module
        for name, module in model.named_modules()
        if (
            ".transformer.blocks." in name
            and name.endswith(".attn_out")
        )
    ]

    if len(modules) != 32:
        raise RuntimeError(
            f"Expected 32 attn_out modules, got {len(modules)}"
        )

    in_features = int(modules[0].in_features)

    x = torch.randn(
        1,
        args.query_len,
        in_features,
        device=DEVICE,
        dtype=DTYPE,
    )

    print(f"attn_out modules: {len(modules)}")
    print(f"input shape: {tuple(x.shape)}")

    with torch.inference_mode():
        # 首次 MPSGraph 编译不进入计时。
        for _ in range(args.warmups):
            for module in modules:
                y = module(x)

        sync()

        samples_ms = []

        for repeat in range(args.repeats):
            sync()
            start = time.perf_counter()

            for _ in range(args.inner):
                for module in modules:
                    y = module(x)

            sync()

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0 / args.inner

            samples_ms.append(elapsed_ms)

            print(
                f"repeat {repeat + 1}: "
                f"{elapsed_ms:.3f} ms / 32 layers"
            )

    mean_ms = statistics.mean(samples_ms)
    median_ms = statistics.median(samples_ms)
    per_layer_us = mean_ms * 1000.0 / len(modules)
    share = mean_ms / args.baseline_ms * 100.0

    print()
    print("========== ATTN OUT GROUP ==========")
    print(f"mean_ms_per_forward : {mean_ms:.3f}")
    print(f"median_ms           : {median_ms:.3f}")
    print(f"mean_us_per_layer   : {per_layer_us:.3f}")
    print(f"baseline_ms         : {args.baseline_ms:.3f}")
    print(f"estimated_share     : {share:.2f}%")
    print("====================================")

    del y
    del x
    del model
    torch.mps.empty_cache()
    sync()


if __name__ == "__main__":
    main()
